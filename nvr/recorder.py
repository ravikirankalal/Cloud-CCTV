#!/usr/bin/env python3
"""
HD recorder: one camera's HD stream, stored as fMP4 HLS segments and indexed in SQLite.

    python3 recorder.py DEVICE.json [stream=cam1]          (nvr-recorder.service)

camera_stream.js (from the TrueCloud cloud) writes H.265 + AAC as MPEG-TS to stdout, and also relays
frames live over HD_LIVE_SOCK for the zero-delay WebSocket view. ffmpeg copies the stream,
without re-encoding, into keyframe-aligned fMP4 segments:

    DATA/<stream>/<YYYY-MM-DD>/<HH>/seg-<epoch>.m4s     (day/hour in REC_TZ)
    DATA/<stream>/init/init-<run>.mp4                   (one init segment per source run)

ffmpeg's own playlist (DATA/<stream>/run.m3u8, a short sliding window) is only a feed: a thread
reads it every second and records each new segment (start time, duration, size, init file,
run) in DATA/nvr.db. Everything else (the playlist API, the timeline, people detection,
retention) reads the database.

Retention by free space: when the disk has less than MIN_FREE_GB free, the oldest hours are
deleted until it has MIN_FREE_GB + 5.
"""
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("NVR_DATA", "/home/ubuntu/nvr-data")
TZ_NAME = os.environ.get("REC_TZ", "Asia/Kolkata")
TZ = ZoneInfo(TZ_NAME)
STREAM_ID = os.environ.get("HD_STREAM_ID", "2")
MIN_FREE_GB = float(os.environ.get("MIN_FREE_GB", "25"))
SOURCE = os.environ.get("HD_SOURCE", os.path.join(HERE, "camera_stream.js"))
LIVE_SOCK = os.environ.get("HD_LIVE_SOCK", "")  # empty: no live relay
RETRY = 2

stopping = False
src = None


def log(msg):
    print(datetime.now(TZ).strftime("%H:%M:%S"), msg, flush=True)


SCHEMA = """
CREATE TABLE IF NOT EXISTS segments (
  id      INTEGER PRIMARY KEY,
  stream  TEXT NOT NULL,
  start   REAL NOT NULL,      -- epoch seconds (PROGRAM-DATE-TIME)
  dur     REAL NOT NULL,
  path    TEXT NOT NULL UNIQUE,   -- relative to DATA
  init    TEXT NOT NULL,          -- relative to DATA
  run     INTEGER NOT NULL,       -- source run (epoch it started): a new run = discontinuity
  bytes   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS seg_time ON segments(stream, start);
"""


def db():
    c = sqlite3.connect(os.path.join(DATA, "nvr.db"), timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    c.executescript(SCHEMA)
    return c


# ---------- indexer: ffmpeg's sliding playlist -> database ----------

def parse_playlist(text):
    """[(start_epoch, dur, uri, init_uri)] from an HLS media playlist."""
    out, dur, pdt, init = [], None, None, None
    for line in text.splitlines():
        if line.startswith("#EXT-X-MAP:"):
            m = re.search(r'URI="([^"]+)"', line)
            init = m and m[1]
        elif line.startswith("#EXTINF:"):
            dur = float(line[8:].split(",")[0])
        elif line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
            pdt = datetime.fromisoformat(line[25:].strip().replace("Z", "+00:00")).timestamp()
        elif line and not line.startswith("#"):
            if dur is not None and pdt is not None and init:
                out.append((pdt, dur, line, init))
            pdt = None  # ffmpeg writes PROGRAM-DATE-TIME before every segment
            dur = None
    return out


def indexer(stream, run, playlist, done):
    conn = db()
    base = os.path.dirname(playlist)
    seen = set()

    def scan():
        try:
            text = open(playlist).read()
        except FileNotFoundError:
            return
        for start, dur, uri, init in parse_playlist(text):
            if uri in seen:
                continue
            full = os.path.normpath(os.path.join(base, uri))
            try:
                size = os.path.getsize(full)
            except FileNotFoundError:
                continue
            seen.add(uri)
            conn.execute("INSERT OR IGNORE INTO segments(stream,start,dur,path,init,run,bytes) VALUES(?,?,?,?,?,?,?)",
                         (stream, start, dur, os.path.relpath(full, DATA),
                          os.path.relpath(os.path.normpath(os.path.join(base, init)), DATA), run, size))
        conn.commit()

    while not done.is_set():
        scan()
        done.wait(1)
    scan()  # the last segments, written as ffmpeg exits
    conn.close()


# ---------- retention ----------

def free_gb():
    return shutil.disk_usage(DATA).free / 1e9


def retention(stop_evt):
    while not stop_evt.wait(300):
        if free_gb() >= MIN_FREE_GB:
            continue
        conn = db()
        target = MIN_FREE_GB + 5
        while free_gb() < target:
            # the oldest hour that has segments
            row = conn.execute("SELECT stream, path FROM segments ORDER BY start LIMIT 1").fetchone()
            if not row:
                break
            hour_dir = os.path.dirname(row[1])  # <stream>/<day>/<HH>
            conn.execute("DELETE FROM segments WHERE path LIKE ?", (hour_dir + "/%",))
            conn.commit()
            shutil.rmtree(os.path.join(DATA, hour_dir), ignore_errors=True)
            day_dir = os.path.dirname(os.path.join(DATA, hour_dir))
            if os.path.isdir(day_dir) and not os.listdir(day_dir):
                os.rmdir(day_dir)
            log(f"retention: deleted {hour_dir}")
        # init files no segment refers to any more
        used = {r[0] for r in conn.execute("SELECT DISTINCT init FROM segments")}
        for s in {r[0] for r in conn.execute("SELECT DISTINCT stream FROM segments")}:
            idir = os.path.join(DATA, s, "init")
            for f in os.listdir(idir):
                rel = os.path.join(s, "init", f)
                if rel not in used and not f.startswith(f"init-{current_run}"):
                    os.remove(os.path.join(idir, f))
        conn.close()


current_run = 0


# ---------- one source run ----------

def run_once(device, stream):
    global src, current_run
    run = current_run = int(time.time())
    sdir = os.path.join(DATA, stream)
    os.makedirs(os.path.join(sdir, "init"), exist_ok=True)
    playlist = os.path.join(sdir, "run.m3u8")
    if os.path.exists(playlist):
        os.remove(playlist)

    env = {**os.environ}
    if LIVE_SOCK:
        env["HD_LIVE_SOCK"] = LIVE_SOCK
    else:
        env.pop("HD_LIVE_SOCK", None)
    src = subprocess.Popen(["node", SOURCE, device, STREAM_ID], stdout=subprocess.PIPE, env=env)

    seg_pattern = os.path.join(sdir, "%Y-%m-%d", "%H", "seg-%s.m4s")
    rec = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error",
         "-f", "mpegts", "-i", "-",
         "-map", "0:v", "-map", "0:a?", "-c", "copy", "-tag:v", "hvc1", "-bsf:a", "aac_adtstoasc",
         "-f", "hls", "-hls_time", "2", "-hls_list_size", "30",
         "-hls_segment_type", "fmp4", "-hls_fmp4_init_filename", f"init/init-{run}.mp4",
         "-hls_flags", "program_date_time+independent_segments+temp_file",
         "-strftime", "1", "-strftime_mkdir", "1",
         "-hls_segment_filename", seg_pattern,
         playlist],
        stdin=src.stdout, env={**os.environ, "TZ": TZ_NAME})
    src.stdout.close()

    done = threading.Event()
    t = threading.Thread(target=indexer, args=(stream, run, playlist, done), daemon=True)
    t.start()
    src.wait()
    rec.wait()
    done.set()
    t.join(10)
    log(f"run {run} ended (source {src.returncode}, ffmpeg {rec.returncode})")


def stop(*_):
    global stopping
    stopping = True
    if src and src.poll() is None:
        src.terminate()  # ffmpeg then finishes its segment and exits


def main():
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    device = sys.argv[1]
    stream = sys.argv[2] if len(sys.argv) > 2 else "cam1"
    os.makedirs(DATA, exist_ok=True)
    db().close()
    stop_evt = threading.Event()
    threading.Thread(target=retention, args=(stop_evt,), daemon=True).start()
    while not stopping:
        run_once(device, stream)
        if not stopping:
            time.sleep(RETRY)
    stop_evt.set()


if __name__ == "__main__":
    main()
