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
import select
import shutil
import struct
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
BACKOFF = [2, 5, 10, 15, 30]  # seconds between attempts after quick failures (the cloud refuses
                              # a new session for a while after one ends)

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
-- one row per source run that recorded something; seq only ever grows (the live playlist's
-- DISCONTINUITY-SEQUENCE), even when retention deletes old runs' segments
CREATE TABLE IF NOT EXISTS runs (
  seq     INTEGER PRIMARY KEY AUTOINCREMENT,
  run     INTEGER NOT NULL UNIQUE,
  stream  TEXT NOT NULL
);
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


# ---------- segment timing from the media itself ----------

def _boxes(b, start=0, end=None):
    end = len(b) if end is None else end
    i = start
    while i + 8 <= end:
        size, kind = struct.unpack(">I4s", b[i:i + 8])
        if size == 1:
            size = struct.unpack(">Q", b[i + 8:i + 16])[0]
        if size < 8 or i + size > end:
            return
        yield kind.decode("latin1"), i, i + size
        i += size


def _child(b, path, start=0, end=None):
    """(start, end) of the first box along path, e.g. ["moov", "trak"]; None if missing."""
    for kind, s, e in _boxes(b, start, end):
        if kind == path[0]:
            return (s, e) if len(path) == 1 else _child(b, path[1:], s + 8, e)
    return None


def video_timescale(init_path):
    """the timescale of the init segment's first track (video)."""
    b = open(init_path, "rb").read()
    mdhd = _child(b, ["moov", "trak", "mdia", "mdhd"])
    s = mdhd[0] + 8
    return struct.unpack(">I", b[s + 20:s + 24] if b[s] == 1 else b[s + 12:s + 16])[0]


def first_tfdt(seg_path, track=1):
    """the base media decode time of a segment's first fragment, for the given track."""
    b = open(seg_path, "rb").read(65536)
    for kind, s, e in _boxes(b):
        if kind != "moof":
            continue
        for k2, ts, te in _boxes(b, s + 8, e):
            if k2 != "traf":
                continue
            tfhd = _child(b, ["tfhd"], ts + 8, te)
            if not tfhd or struct.unpack(">I", b[tfhd[0] + 12:tfhd[0] + 16])[0] != track:
                continue
            tfdt = _child(b, ["tfdt"], ts + 8, te)
            if tfdt:
                v = tfdt[0] + 8
                return struct.unpack(">Q", b[v + 4:v + 12])[0] if b[v] == 1 else struct.unpack(">I", b[v + 4:v + 8])[0]
        return None
    return None


def indexer(stream, run, playlist, done, t0_file):
    conn = db()
    base = os.path.dirname(playlist)
    seen = set()
    timing = {}  # t0: wall time of the first frame; scale, tfdt0: the first segment's media time

    def start_of(full, init_full, fallback):
        """wall-clock start: the first frame's time + the segment's media time since then.
        ffmpeg's PROGRAM-DATE-TIME (the fallback) runs ~5 s late."""
        try:
            if "t0" not in timing:
                timing["t0"] = float(open(t0_file).read())
            if "scale" not in timing:
                timing["scale"] = video_timescale(init_full)
            d = first_tfdt(full)
            if d is None:
                return fallback
            timing.setdefault("tfdt0", d)
            return timing["t0"] + (d - timing["tfdt0"]) / timing["scale"]
        except (OSError, ValueError, TypeError, struct.error):
            return fallback

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
            init_full = os.path.normpath(os.path.join(base, init))
            start = start_of(full, init_full, start)
            conn.execute("INSERT OR IGNORE INTO runs(run, stream) VALUES(?,?)", (run, stream))
            conn.execute("INSERT OR IGNORE INTO segments(stream,start,dur,path,init,run,bytes) VALUES(?,?,?,?,?,?,?)",
                         (stream, start, dur, os.path.relpath(full, DATA), os.path.relpath(init_full, DATA), run, size))
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
            row = conn.execute("SELECT stream, path FROM segments ORDER BY stream, start LIMIT 1").fetchone()
            if not row:
                break
            hour_dir = os.path.dirname(row[1])  # <stream>/<day>/<HH>
            # a path range (uses the path index; LIKE read the whole table)
            conn.execute("DELETE FROM segments WHERE path >= ? AND path < ?", (hour_dir + "/", hour_dir + "0"))
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

    t0_file = os.path.join(sdir, "run.t0")
    if os.path.exists(t0_file):
        os.remove(t0_file)
    env = {**os.environ, "REC_T0_FILE": t0_file}
    if LIVE_SOCK:
        env["HD_LIVE_SOCK"] = LIVE_SOCK
    else:
        env.pop("HD_LIVE_SOCK", None)
    src = subprocess.Popen(["node", SOURCE, device, STREAM_ID], stdout=subprocess.PIPE, env=env)
    # ffmpeg only once the camera sends something (a failed connect otherwise leaves it erroring)
    while not stopping and src.poll() is None:
        if select.select([src.stdout], [], [], 1)[0]:
            break
    if stopping or src.poll() is not None:
        src.wait()
        log(f"run {run}: no stream (source {src.returncode})")
        return 0
    started = time.time()

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
    t = threading.Thread(target=indexer, args=(stream, run, playlist, done, t0_file), daemon=True)
    t.start()
    src.wait()
    rec.wait()
    done.set()
    t.join(10)
    log(f"run {run} ended after {time.time() - started:.0f} s (source {src.returncode}, ffmpeg {rec.returncode})")
    return time.time() - started


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
    fails = 0
    while not stopping:
        lasted = run_once(device, stream)
        fails = 0 if lasted > 120 else fails + 1  # a run that recorded a while resets the back-off
        if not stopping:
            time.sleep(BACKOFF[min(fails, len(BACKOFF)) - 1] if fails else BACKOFF[0])
    stop_evt.set()


if __name__ == "__main__":
    main()
