#!/usr/bin/env python3
"""
On-site agent: pulls RTSP from cameras on the local network and pushes each one
out to the cloud over RTMP, so the existing pipeline records/relays/detects it
exactly like a camera that pushes RTMP itself.

Why an agent instead of the cloud pulling RTSP directly:
  RTSP/ONVIF pull needs someone to open a connection *into* the camera. Cameras
  sit behind the home router (NAT) or, on 4G, behind carrier-grade NAT, so from
  the internet there's nothing to dial. This agent runs on the same LAN as the
  cameras (where RTSP always works) and *dials out* to the cloud, which crosses
  any NAT/CGNAT/4G with no port-forwarding and nothing exposed to the internet.

Two ways to name cameras in the config:
  - "rtsp": an RTSP URL you already know (works with any RTSP camera, no ONVIF).
  - "onvif": {host, port, user, pass} - the agent asks the camera over ONVIF for
    its RTSP URL (and stream keys stay out of the config).

Discover cameras on the LAN and print ready-to-paste config, then exit:
  python3 nvr_agent.py --discover
  python3 nvr_agent.py --discover --user admin --pass secret   # also fetch RTSP URLs

Run the relays (default):
  python3 nvr_agent.py --config agent.json

Config (agent.json):
  {
    "server": "rtmp://YOUR.SERVER:1935/live",
    "cameras": [
      {"key": "frontdoor", "rtsp": "rtsp://admin:pass@192.168.1.50:554/stream1"},
      {"key": "backyard",  "onvif": {"host": "192.168.1.51", "user": "admin", "pass": "pass"}}
    ]
  }
The stream "key" is what the cloud sees as the stream name (like an RTMP stream
key); keep it unique per camera per account.
"""
import argparse
import hashlib
import base64
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
STALL_US = 15_000_000            # kill+restart a relay if the RTSP input stalls this long
BACKOFF_MAX = 30                 # seconds between reconnect attempts (grows to this)
DISCOVER_WAIT = 3               # seconds to listen for ONVIF discovery replies


def log(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)


# ---- ONVIF: find cameras (WS-Discovery) and ask each for its RTSP URL -------

WSD_ADDR, WSD_PORT = "239.255.255.250", 3702
WSD_PROBE = """<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
 xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
 xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
 xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
 <e:Header>
  <w:MessageID>uuid:{mid}</w:MessageID>
  <w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
  <w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
 </e:Header>
 <e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body>
</e:Envelope>"""


def discover(timeout=DISCOVER_WAIT):
    """Multicast an ONVIF probe and collect the service URLs that answer."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    sock.settimeout(timeout)
    msg = WSD_PROBE.format(mid=uuid.uuid4()).encode()
    sock.sendto(msg, (WSD_ADDR, WSD_PORT))
    found, seen = [], set()
    end = time.time() + timeout
    while time.time() < end:
        try:
            data, addr = sock.recvfrom(65535)
        except socket.timeout:
            break
        text = data.decode("utf-8", "replace")
        for url in re.findall(r"https?://[^\s<>]+", text):
            host = urllib.parse.urlparse(url).hostname
            if host and host not in seen:
                seen.add(host)
                found.append({"host": host, "xaddr": url})
    sock.close()
    return found


def _wss_header(user, password):
    """ONVIF WS-Security UsernameToken with a password digest (nonce+created)."""
    nonce = os.urandom(16)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    digest = base64.b64encode(
        hashlib.sha1(nonce + created.encode() + password.encode()).digest()).decode()
    return f"""<s:Header><Security s:mustUnderstand="1"
 xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd">
 <UsernameToken><Username>{user}</Username>
 <Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">{digest}</Password>
 <Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">{base64.b64encode(nonce).decode()}</Nonce>
 <Created xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">{created}</Created>
 </UsernameToken></Security></s:Header>"""


def _soap(xaddr, body, user, password, action):
    header = _wss_header(user, password) if user else ""
    env = (f'<?xml version="1.0" encoding="UTF-8"?>'
           f'<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">'
           f'{header}<s:Body>{body}</s:Body></s:Envelope>')
    req = urllib.request.Request(xaddr, data=env.encode(),
                                 headers={"Content-Type": f'application/soap+xml; action="{action}"'})
    with urllib.request.urlopen(req, timeout=8) as r:
        return r.read().decode("utf-8", "replace")


def onvif_rtsp(host, port=80, user="", password="", xaddr=None):
    """Ask an ONVIF camera for the RTSP URL of its first media profile."""
    dev = xaddr or f"http://{host}:{port}/onvif/device_service"
    media = dev.replace("device_service", "media_service")
    ns = ('xmlns:trt="http://www.onvif.org/ver10/media/wsdl" '
          'xmlns:tt="http://www.onvif.org/ver10/schema"')
    profiles = _soap(media, f"<trt:GetProfiles {ns}/>", user, password,
                     "http://www.onvif.org/ver10/media/wsdl/GetProfiles")
    tok = re.search(r'token="([^"]+)"', profiles) or re.search(r"token='([^']+)'", profiles)
    if not tok:
        raise RuntimeError("no media profiles returned")
    body = (f'<trt:GetStreamUri {ns}><trt:StreamSetup>'
            f'<tt:Stream>RTP-Unicast</tt:Stream>'
            f'<tt:Transport><tt:Protocol>RTSP</tt:Protocol></tt:Transport>'
            f'</trt:StreamSetup><trt:ProfileToken>{tok.group(1)}</trt:ProfileToken></trt:GetStreamUri>')
    uri = _soap(media, body, user, password,
                "http://www.onvif.org/ver10/media/wsdl/GetStreamUri")
    m = re.search(r"<[^>]*Uri>(rtsp://[^<]+)</", uri)
    if not m:
        raise RuntimeError("no RTSP URI in GetStreamUri reply")
    rtsp = m.group(1)
    # cameras return the URL without credentials; splice them back in for ffmpeg
    if user and "@" not in rtsp:
        rtsp = rtsp.replace("rtsp://", f"rtsp://{urllib.parse.quote(user)}:{urllib.parse.quote(password)}@", 1)
    return rtsp


# ---- resolving a camera entry to an RTSP URL --------------------------------

def rtsp_of(cam):
    if cam.get("rtsp"):
        return cam["rtsp"]
    o = cam.get("onvif") or {}
    return onvif_rtsp(o.get("host"), o.get("port", 80), o.get("user", ""),
                      o.get("pass", ""), o.get("xaddr"))


# ---- relay: RTSP in, RTMP out, re-encode nothing ----------------------------

def relay_cmd(rtsp, out):
    # -c copy: no transcode, so the agent runs on a Pi. -rtsp_transport tcp:
    # RTSP over UDP loses packets on busy/wifi networks. FLV is what RTMP carries.
    return ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "warning",
            "-rw_timeout", str(STALL_US),
            "-rtsp_transport", "tcp", "-i", rtsp,
            "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy",
            "-f", "flv", out]


def run(config):
    server = config["server"].rstrip("/")
    cams = config["cameras"]
    procs = {}      # key -> Popen
    backoff = {}    # key -> current retry delay
    nextrtsp = {}   # key -> resolved rtsp (re-resolved on ONVIF failure)
    stop = {"v": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(v=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(v=True))

    def start(cam):
        key = cam["key"]
        try:
            rtsp = nextrtsp.get(key) or rtsp_of(cam)
        except Exception as e:                       # ONVIF/network hiccup: back off, retry
            log(f"[{key}] resolve failed: {e}")
            return None
        nextrtsp[key] = rtsp
        safe = re.sub(r"://[^@/]+@", "://***@", rtsp)
        log(f"[{key}] relaying {safe} -> {server}/{key}")
        return subprocess.Popen(relay_cmd(rtsp, f"{server}/{key}"))

    while not stop["v"]:
        for cam in cams:
            key = cam["key"]
            p = procs.get(key)
            if p and p.poll() is None:
                continue                              # healthy
            if p:                                     # it exited: schedule a retry with backoff
                nextrtsp.pop(key, None)               # re-ask ONVIF in case the URL rotated
                backoff[key] = min(BACKOFF_MAX, backoff.get(key, 1) * 2)
                log(f"[{key}] relay stopped (code {p.returncode}); retry in {backoff[key]}s")
                procs[key] = None
                cam["_retry_at"] = time.time() + backoff[key]
                continue
            if time.time() < cam.get("_retry_at", 0):
                continue
            np = start(cam)
            if np:
                procs[key] = np
                backoff[key] = 1
            else:
                cam["_retry_at"] = time.time() + min(BACKOFF_MAX, backoff.get(key, 1) * 2)
                backoff[key] = min(BACKOFF_MAX, backoff.get(key, 1) * 2)
        time.sleep(1)

    log("stopping relays")
    for p in procs.values():
        if p and p.poll() is None:
            p.terminate()
    for p in procs.values():
        if p:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


# ---- CLI --------------------------------------------------------------------

def cmd_discover(args):
    found = discover()
    if not found:
        log("no ONVIF cameras answered on this network")
        log("(some cameras don't advertise; you can still add them by RTSP URL)")
        return
    cams = []
    for i, dev in enumerate(found):
        entry = {"key": f"cam{i+1}", "onvif": {"host": dev["host"]}}
        if args.user:
            entry["onvif"].update(user=args.user, **{"pass": args.password or ""})
            try:
                rtsp = onvif_rtsp(dev["host"], user=args.user, password=args.password or "",
                                  xaddr=dev["xaddr"])
                safe = re.sub(r"://[^@/]+@", "://***@", rtsp)
                log(f"found {dev['host']} -> {safe}")
            except Exception as e:
                log(f"found {dev['host']} (RTSP lookup failed: {e})")
        else:
            log(f"found {dev['host']} at {dev['xaddr']}")
        cams.append(entry)
    print("\n--- paste into agent.json (add \"server\", fill user/pass) ---")
    print(json.dumps({"server": "rtmp://YOUR.SERVER:1935/live", "cameras": cams}, indent=2))


def main():
    ap = argparse.ArgumentParser(description="On-site RTSP/ONVIF -> cloud RTMP agent")
    ap.add_argument("--config", default=os.path.join(HERE, "agent.json"))
    ap.add_argument("--discover", action="store_true", help="list ONVIF cameras and exit")
    ap.add_argument("--user", help="ONVIF username (for --discover RTSP lookup)")
    ap.add_argument("--pass", dest="password", help="ONVIF password")
    args = ap.parse_args()
    if args.discover:
        return cmd_discover(args)
    try:
        with open(args.config) as f:
            config = json.load(f)
    except FileNotFoundError:
        sys.exit(f"no config at {args.config} (see agent.example.json, or run --discover)")
    if not config.get("cameras"):
        sys.exit("config has no cameras")
    log(f"agent starting: {len(config['cameras'])} camera(s) -> {config['server']}")
    run(config)


if __name__ == "__main__":
    main()
