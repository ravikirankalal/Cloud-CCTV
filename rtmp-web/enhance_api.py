#!/usr/bin/env python3
"""
On-demand footage enhancement for the recording page. The page asks for a short
clip around the moment being watched; this runs one niced, single-threaded
ffmpeg with a filter chain (no GPU/AI - the box is tiny) and returns an mp4 the
page can play and download.

  POST /api/enhance   {"file": "<stream-...>.mp4", "at": 42.0, "dur": 15,
                       "modes": ["light","detail","upscale"]}
      -> {"job": "<id>"}                     (starts one job; 409 if one runs)
  GET  /api/enhance?job=<id>
      -> {"status": "running"} | {"status":"done","url":"/recordings/.enhanced/<id>.mp4"}
         | {"status":"error","error":"..."}

Modes (any combination; applied in a fixed order):
  light   - lift shadows/gamma for dark scenes (eq)
  detail  - deblock compression + denoise + sharpen for clarity
  upscale - 2x with lanczos

Only one job runs at a time (this server has 2 vCPUs and shares them with
detection). Output goes to <REC_DIR>/.enhanced/, served by nginx at
/recordings/.enhanced/; old clips there are pruned.

Listens on a Unix socket in run/ (nginx proxies /api/enhance/ to it). Service:
  sudo cp /home/ubuntu/rtmp-web/enhance-api.service /etc/systemd/system/
  sudo systemctl daemon-reload && sudo systemctl enable --now enhance-api
"""
import json
import os
import re
import secrets
import socketserver
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
REC_DIR = os.path.realpath(os.environ.get("REC_DIR", "/home/ubuntu/rtmp-recordings"))
SOCK = os.path.join(HERE, "run", "enhance-api.sock")
OUT_DIR = os.path.join(REC_DIR, ".enhanced")
OUT_URL = "/recordings/.enhanced"

MAX_BODY = 2048
MIN_DUR, MAX_DUR = 3, 30
KEEP_FILES = 20            # prune the enhanced folder to this many newest clips
KEEP_SECONDS = 2 * 3600   # ...and drop anything older than this
JOB_TIMEOUT = 240         # kill an ffmpeg that runs longer than this

# each mode -> its ffmpeg filters, emitted in this order so they compose sensibly
FILTERS = {
    "detail_pre": "deblock=filter=strong:block=8",
    "light": "eq=brightness=0.06:contrast=1.12:gamma=1.35:saturation=1.08",
    "detail_dn": "hqdn3d=3:3:6:6",
    "detail_sharp": "unsharp=5:5:0.9:5:5:0.0",
    "upscale": "scale=iw*2:ih*2:flags=lanczos",
}

lock = threading.Lock()
jobs = {}   # id -> {"status", "url"/"error", "at"}


def build_chain(modes):
    modes = set(modes)
    chain = []
    if "detail" in modes:
        chain.append(FILTERS["detail_pre"])
    if "light" in modes:
        chain.append(FILTERS["light"])
    if "detail" in modes:
        chain += [FILTERS["detail_dn"], FILTERS["detail_sharp"]]
    if "upscale" in modes:
        chain.append(FILTERS["upscale"])
    return ",".join(chain)


def prune():
    try:
        files = [os.path.join(OUT_DIR, f) for f in os.listdir(OUT_DIR) if f.endswith(".mp4")]
    except OSError:
        return
    now = time.time()
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for i, p in enumerate(files):
        try:
            if i >= KEEP_FILES or now - os.path.getmtime(p) > KEEP_SECONDS:
                os.remove(p)
        except OSError:
            pass


def resolve(file):
    """The file must sit inside REC_DIR and be an existing .mp4 (no traversal)."""
    if not isinstance(file, str) or "\x00" in file or not file.endswith(".mp4"):
        return None
    path = os.path.realpath(os.path.join(REC_DIR, file.lstrip("/")))
    if (path == REC_DIR or not path.startswith(REC_DIR + os.sep)
            or os.path.basename(path).startswith(".") or not os.path.isfile(path)):
        return None
    return path


def run_job(job_id, src, at, dur, modes):
    out = os.path.join(OUT_DIR, job_id + ".mp4")
    chain = build_chain(modes)
    cmd = ["nice", "-n", "15", "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
           "-ss", f"{at:.2f}", "-t", f"{dur:.2f}", "-i", src, "-threads", "1"]
    if chain:
        cmd += ["-vf", chain]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", out]
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        r = subprocess.run(cmd, capture_output=True, timeout=JOB_TIMEOUT, text=True)
        if r.returncode == 0 and os.path.isfile(out) and os.path.getsize(out) > 0:
            jobs[job_id] = {"status": "done", "url": f"{OUT_URL}/{job_id}.mp4", "at": time.time()}
        else:
            for p in (out,):
                if os.path.exists(p):
                    os.remove(p)
            jobs[job_id] = {"status": "error", "error": (r.stderr or "ffmpeg failed").strip()[:200],
                            "at": time.time()}
    except subprocess.TimeoutExpired:
        jobs[job_id] = {"status": "error", "error": "took too long", "at": time.time()}
    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e)[:200], "at": time.time()}
    finally:
        lock.release()
        prune()


def start(body):
    src = resolve(body.get("file"))
    if not src:
        return 400, {"error": "unknown file"}
    try:
        at = max(0.0, float(body.get("at", 0)))
        dur = float(body.get("dur", 15))
    except (TypeError, ValueError):
        return 400, {"error": "bad at/dur"}
    dur = max(MIN_DUR, min(MAX_DUR, dur))
    modes = body.get("modes") or ["light", "detail"]
    if not isinstance(modes, list) or not set(modes) <= {"light", "detail", "upscale"}:
        return 400, {"error": "bad modes"}
    if not lock.acquire(blocking=False):
        return 409, {"error": "another clip is being enhanced; try again in a moment"}
    job_id = secrets.token_hex(8)
    jobs[job_id] = {"status": "running", "at": time.time()}
    # drop jobs we finished reporting on a while ago
    for jid, j in list(jobs.items()):
        if j["status"] != "running" and time.time() - j["at"] > 600:
            jobs.pop(jid, None)
    threading.Thread(target=run_job, args=(job_id, src, at, dur, modes), daemon=True).start()
    return 200, {"job": job_id}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if urlparse(self.path).path != "/api/enhance":
            return self.reply(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BODY:
            return self.reply(400, {"error": "bad request"})
        try:
            body = json.loads(self.rfile.read(n))
            assert isinstance(body, dict)
        except (ValueError, AssertionError):
            return self.reply(400, {"error": "bad json"})
        self.reply(*start(body))

    def do_GET(self):
        u = urlparse(self.path)
        if u.path != "/api/enhance":
            return self.reply(404, {"error": "not found"})
        jid = (parse_qs(u.query).get("job") or [""])[0]
        job = jobs.get(jid)
        if not job:
            return self.reply(404, {"error": "unknown job"})
        out = {"status": job["status"]}
        if job["status"] == "done":
            out["url"] = job["url"]
        elif job["status"] == "error":
            out["error"] = job.get("error", "failed")
        self.reply(200, out)

    def reply(self, code, data):
        out = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass

    def address_string(self):
        return "-"


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def get_request(self):
        conn, _ = self.socket.accept()
        return conn, ("", 0)


def main():
    os.makedirs(os.path.dirname(SOCK), exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    prune()
    if os.path.exists(SOCK):
        os.remove(SOCK)
    with Server(SOCK, Handler) as srv:
        os.chmod(SOCK, 0o666)
        srv.serve_forever()


if __name__ == "__main__":
    main()
