#!/usr/bin/env python3
"""
Run this ON THE SAME WI-FI/LAN as the camera. It scans the local network for a
XiongMai / iCSee (NetSDK / DVRIP) camera, confirms the platform, and prints the
RTSP URL you can hand to the agent (nvr_agent.py).

  python3 identify.py                       # scan + confirm platform
  python3 identify.py --user admin --pass secret   # also verify RTSP login

Why this exists: the T18290S is a Wi-Fi camera, so it lives on your LAN. On the
LAN it speaks DVRIP (TCP 34567) and usually RTSP (TCP 554) - no vendor cloud and
no P2P needed. Reaching it from outside is then just the agent relaying outward.

It does not need any credentials to CONFIRM the platform: even a rejected DVRIP
login comes back in XiongMai's packet format, which is the fingerprint. Creds are
only used to verify the RTSP URL actually plays.
"""
import argparse
import hashlib
import json
import socket
import struct
import subprocess
import sys

DVRIP_PORT = 34567
RTSP_PORT = 554
SCAN_PORTS = [DVRIP_PORT, RTSP_PORT, 80, 8899]

# iCSee/XiongMai RTSP paths (main + sub stream). Channel 0, subtype 0=main,1=extra.
RTSP_PATHS = [
    "/user={u}&password={p}&channel=1&stream=0.sdp?real_stream",  # common XM/iCSee
    "/cam/realmonitor?channel=1&subtype=0",                        # some OEM firmwares
    "/live/ch0",
    "/11",  # a few XM builds
]


def local_subnet():
    """Best-effort /24 the machine sits on."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip.rsplit(".", 1)[0], ip


def port_open(ip, port, timeout=0.5):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        return s.connect_ex((ip, port)) == 0
    finally:
        s.close()


def sofia_hash(password):
    """XiongMai's 'sofia' password hash (MD5 folded into 8 chars over a 62-char set)."""
    md5 = hashlib.md5(password.encode()).digest()
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    return "".join(chars[(md5[i * 2] + md5[i * 2 + 1]) % 62] for i in range(8))


def dvrip_probe(ip, user="admin", password="", timeout=2.0):
    """Send a DVRIP login. A XiongMai device replies with a 20-byte header whose
    first byte is 0xFF - that's the fingerprint, regardless of login success.
    Returns (is_xiongmai, info_dict_or_error)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((ip, DVRIP_PORT))
        body = json.dumps({
            "EncryptType": "MD5", "LoginType": "DVRIP-Web",
            "PassWord": sofia_hash(password), "UserName": user,
        }).encode() + b"\x00"
        # header: 0xFF 0x00 0x00 0x00, session(4), seq(4), 0,0, msgid(2)=1000(Login), len(4)
        header = struct.pack("<BBBBIIBBHI", 0xFF, 0, 0, 0, 0, 0, 0, 0, 1000, len(body))
        s.sendall(header + body)
        head = s.recv(20)
        if len(head) < 20 or head[0] != 0xFF:
            return False, "not a DVRIP/XiongMai response"
        (length,) = struct.unpack("<I", head[16:20])
        data = b""
        while len(data) < length and length < 1_000_000:
            chunk = s.recv(min(4096, length - len(data)))
            if not chunk:
                break
            data += chunk
        try:
            reply = json.loads(data.rstrip(b"\x00").decode("latin1"))
        except ValueError:
            reply = {}
        ret = reply.get("Ret")
        note = {100: "login OK", 205: "login failed (wrong user/pass)",
                203: "login failed (wrong password)"}.get(ret, f"Ret={ret}")
        return True, note
    except Exception as e:
        return False, str(e)
    finally:
        s.close()


def rtsp_banner(ip, timeout=2.0):
    """RTSP OPTIONS; return the Server: header if any (another platform hint)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((ip, RTSP_PORT))
        s.sendall(f"OPTIONS rtsp://{ip}:{RTSP_PORT} RTSP/1.0\r\nCSeq: 1\r\n\r\n".encode())
        resp = s.recv(1024).decode("latin1", "replace")
        for line in resp.splitlines():
            if line.lower().startswith("server:"):
                return line.split(":", 1)[1].strip()
        return "(RTSP open, no Server header)"
    except Exception as e:
        return None
    finally:
        s.close()


def ffprobe_ok(url, timeout=12):
    """True if ffprobe can read a video stream from url (verifies creds+path)."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-rtsp_transport", "tcp",
             "-select_streams", "v:0", "-show_entries", "stream=codec_name",
             "-of", "csv=p=0", url],
            capture_output=True, timeout=timeout, text=True)
        return bool(r.stdout.strip()), (r.stdout or r.stderr).strip()
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return False, str(e)


def main():
    ap = argparse.ArgumentParser(description="Find and identify a XiongMai/iCSee camera on the LAN")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--pass", dest="password", default="")
    ap.add_argument("--subnet", help="e.g. 192.168.1 (default: auto-detect)")
    ap.add_argument("--host", help="probe just this IP (skip the scan)")
    args = ap.parse_args()

    if args.host:
        hosts = [args.host]
    else:
        sub, me = (args.subnet, None) if args.subnet else local_subnet()
        print(f"scanning {sub}.0/24 (this machine: {me or 'n/a'}) for camera ports...")
        hosts = []
        for i in range(1, 255):
            ip = f"{sub}.{i}"
            opened = [p for p in SCAN_PORTS if port_open(ip, p)]
            if opened:
                print(f"  {ip}: open {opened}")
                if DVRIP_PORT in opened or RTSP_PORT in opened:
                    hosts.append(ip)
        if not hosts:
            print("no camera-like hosts found. Make sure this machine is on the "
                  "SAME Wi-Fi as the camera, or pass --host <camera-ip>.")
            return

    for ip in hosts:
        print(f"\n=== {ip} ===")
        if port_open(ip, DVRIP_PORT):
            xm, note = dvrip_probe(ip, args.user, args.password)
            if xm:
                print(f"  DVRIP/34567: XiongMai/iCSee CONFIRMED - {note}")
            else:
                print(f"  DVRIP/34567: open but not XiongMai-shaped ({note})")
        if port_open(ip, RTSP_PORT):
            banner = rtsp_banner(ip)
            print(f"  RTSP/554: {banner}")
            if args.password or args.user:
                for path in RTSP_PATHS:
                    url = f"rtsp://{args.user}:{args.password}@{ip}:{RTSP_PORT}" + \
                          path.format(u=args.user, p=args.password)
                    ok, detail = ffprobe_ok(url)
                    safe = url.replace(f":{args.password}@", ":***@")
                    if ok:
                        print(f"  ✓ STREAM WORKS ({detail}): {safe}")
                        print(f"    -> add to agent.json as: "
                              f'{{"key":"cam1","rtsp":"{safe}"}}')
                        break
                    else:
                        print(f"  ✗ {safe}  ({detail.splitlines()[0] if detail else 'no video'})")
            else:
                print("  (pass --user/--pass to verify the RTSP URL and print the "
                      "agent config line)")


if __name__ == "__main__":
    main()
