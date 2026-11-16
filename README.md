# Cloud-CCTV

A small NVR for one TrueCloud camera, running on an Oracle Cloud free-tier VM
(VM.Standard.E2.1.Micro: 2 vCPU, 1 GB RAM, 200 GB disk). HD only: the camera's 2304×1296
H.265 stream is recorded without re-encoding and played back in the browser.

## How it fits together

```
camera ──TrueCloud cloud──▶ camera_stream.js ──MPEG-TS──▶ recorder.py ──ffmpeg (copy)──▶ nvr-data/<cam>/<day>/<hour>/seg-<epoch>.m4s
                                 │                          └──────────────▶ nvr-data/nvr.db (segment index)
                                 └── WebSocket relay (run/hd-live-<cam>.sock): zero-delay live view
browser ──https──▶ nginx ──auth_request──▶ signin.js (Google sign-in → session cookie)
                     ├── /            the page (nvr/web/index.html)
                     ├── /media/…     segments and init files, straight from disk
                     ├── /api/…       playback_api.py: VOD / live playlists, coverage, MP4 export
                     └── /api/live-ws/<cam>  the live relay
```

| File | What it does |
|---|---|
| `nvr/camera_stream.js` | Connects to the camera through the TrueCloud cloud (the vendor's own p2p worker, `truecloud/*.connector.js`, not in this repo) and outputs H.265 + AAC as MPEG-TS; also relays frames live over a WebSocket. |
| `nvr/recorder.py` | Runs `camera_stream.js`, has ffmpeg cut the stream into keyframe-aligned fMP4 HLS segments (~10 s), indexes every segment in SQLite, and deletes the oldest hours when free space drops below `MIN_FREE_GB` (25). |
| `nvr/playback_api.py` | Builds HLS playlists for any time range from the index (PROGRAM-DATE-TIME on every segment, DISCONTINUITY at gaps), plus coverage spans and stream-copy MP4 exports. |
| `nvr/web/index.html` | One page: live view (WebSocket + MSE, ~0.4 s behind; HLS live as fallback), a day timeline (24h / 6h / 1h) to scrub recordings, speed, clip download. Google sign-in via Firebase. |
| `nvr/signin.js` | Turns the Firebase sign-in into a session cookie nginx checks on every private request. Only emails in `allowed-emails.txt` may in (when that file has any). |
| `nvr/gapfill.py` | Fills gaps in the recording from the camera's own SD card (search + replay through the recorder's camera session, via `camera_stream.js`'s control socket), as soon as the card has them; marks what the card lacks too as `camera_offline`. |
| `nvr/models/` | ONNX models (YOLOX-Tiny person detector, OSNet re-identification) for people detection. |

H.265 plays in Safari, Android Chrome, and Chrome / Edge on hardware with H.265 decoding.

## Deploy (server layout)

```
~/nvr/            this folder, plus node_modules/ (npm install), truecloud/ and state/hd-device.json
~/nvr-data/       segments + nvr.db
~/auth/           auth-secret, allowed-emails.txt
```

```sh
sudo install -m 644 nvr/nvr-recorder.service nvr/nvr-playback.service nvr/nvr-signin.service nvr/nvr-gapfill.service /etc/systemd/system/
sudo install -m 644 nvr/deploy/nginx-site.conf /etc/nginx/conf.d/nvr.conf
sudo usermod -aG ubuntu www-data      # nginx reads ~/nvr/web and ~/nvr-data
sudo systemctl daemon-reload && sudo systemctl enable --now nvr-recorder nvr-playback nvr-signin nvr-gapfill && sudo systemctl reload nginx
```

## History

Before November 2026 this was an nginx-rtmp setup that recorded the camera's 800×448 RTMP
stream with HD recorded alongside, SD-card gap filling and people detection on the SD
footage. That code is in git history (and the `backup/oracle-server-2026-11-16` branch).
