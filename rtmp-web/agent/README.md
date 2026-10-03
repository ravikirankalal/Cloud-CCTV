# NVR on-site agent

Pulls RTSP/ONVIF cameras on a local network and pushes each one to the cloud
over RTMP, so cameras that can only be reached on their own LAN still land in
the cloud recording pipeline.

## Why an agent (and not the cloud pulling RTSP directly)

RTSP/ONVIF pull means opening a connection **into** the camera. Cameras sit
behind the home router (NAT), or on 4G behind carrier-grade NAT, so from the
internet there is nothing to dial — direct cloud pull only works with router
port-forwarding, which is fragile and exposes the camera to the internet.

The agent runs on the **same LAN as the cameras** (a Raspberry Pi, mini-PC, NAS,
or any always-on machine) where RTSP always works, and **dials outward** to the
cloud. Outbound connections cross any NAT/CGNAT/4G with no router configuration
and nothing exposed. On the cloud side it looks identical to a camera that
pushes RTMP itself, so **no server changes are needed**.

```
 camera LAN                         internet
 ┌──────────┐   RTSP    ┌────────┐   RTMP (outbound)   ┌───────────────┐
 │ IP camera│ ────────► │ agent  │ ──────────────────► │ cloud :1935   │
 └──────────┘  (local)  └────────┘                     │ /live/<key>   │
                                                        │ record/detect │
                                                        └───────────────┘
```

## Requirements

- Python 3.8+ and `ffmpeg` on the machine (`sudo apt install ffmpeg`).
- The machine stays on and is on the same network as the cameras.
- No external Python packages — standard library only.

## Identify a XiongMai / iCSee camera first (optional)

Trueview / TrueCloud Plus (and many Indian Wi-Fi/4G cameras) run XiongMai's
"NetSDK" stack: on the LAN they speak DVRIP on TCP 34567 and usually RTSP on 554.
Run `identify.py` **on the same Wi-Fi as the camera** to confirm the platform and
get the working RTSP URL:

```bash
python3 identify.py                          # scan + confirm platform
python3 identify.py --user admin --pass YOURPASS   # also print the RTSP config line
```

It scans the subnet, fingerprints DVRIP (a XiongMai reply starts with 0xFF even
when the login is rejected), and prints the `agent.json` line to paste. No P2P is
needed for a LAN camera - the agent below relays it outward for remote viewing.

## Setup

1. Find your cameras (optional, ONVIF only):
   ```bash
   python3 nvr_agent.py --discover
   # also fetch each camera's RTSP URL:
   python3 nvr_agent.py --discover --user admin --pass secret
   ```
   It prints a ready-to-edit config. Cameras that don't advertise over ONVIF can
   still be added by their RTSP URL.

2. Create `agent.json` (copy `agent.example.json`):
   ```json
   {
     "server": "rtmp://YOUR.SERVER.IP:1935/live",
     "cameras": [
       {"key": "frontdoor", "rtsp": "rtsp://admin:pass@192.168.1.50:554/stream1"},
       {"key": "backyard",  "onvif": {"host": "192.168.1.51", "user": "admin", "pass": "pass"}}
     ]
   }
   ```
   - `server` — the cloud RTMP ingest, i.e. `.../live`.
   - `key` — unique per camera; this is the stream name the cloud records under.
   - `rtsp` — a known RTSP URL, **or** `onvif` `{host, user, pass}` to look it up.

3. Run it:
   ```bash
   python3 nvr_agent.py --config agent.json
   ```

## Run it always-on

**systemd:** copy the folder to `/opt/nvr-agent`, edit paths/user in
`nvr-agent.service`, then:
```bash
sudo cp nvr-agent.service /etc/systemd/system/
sudo systemctl enable --now nvr-agent
journalctl -u nvr-agent -f
```

**Docker:** (`--network host` so LAN discovery and RTSP reach the cameras)
```bash
docker build -t nvr-agent .
docker run -d --name nvr-agent --restart unless-stopped \
  --network host -v "$PWD/agent.json:/app/agent.json:ro" nvr-agent
```

## Behaviour

- No transcoding (`-c copy`), so it runs on low-power hardware.
- RTSP over TCP (survives lossy Wi-Fi/busy LANs).
- Each camera relays independently; if one drops, only it reconnects, with
  backoff up to 30s. ONVIF URLs are re-fetched on reconnect in case they rotate.
- Credentials are never printed in logs (masked as `***`).

## Security notes

- The agent only makes **outbound** connections; nothing is exposed on the LAN.
- Protect the cloud RTMP ingest per account (unique stream keys; consider
  `on_publish` auth or SRT with a passphrase for encryption in transit).
- Keep `agent.json` readable only by the service user (it holds camera
  passwords): `chmod 600 agent.json`.
