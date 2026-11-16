#!/usr/bin/env python3
"""
Gap filler: holes in the HD recording, fetched from the camera's own SD card as soon as the card
has them, and stored as ordinary segments in the index.

    python3 gapfill.py [stream=cam1]          (nvr-gapfill.service)

The live recording loses footage whenever the TrueCloud cloud link drops (a reconnect costs
~20-50 s). The camera's card records on its own, in full HD. Every POLL_S this:
  1. finds gaps between indexed segments (the newest first), older than GAP_SETTLE_S;
  2. asks the card what it has there (search, through camera_stream.js's control socket: the
     recorder's own session, since a second session to the camera knocks the live one off);
  3. replays the covered part in pieces of up to CHUNK_S (the card sends ~0.2x real time and
     shares the camera's upload with live; camera_stream stops a replay if live stalls);
  4. cuts it into fMP4 segments with ffmpeg (no re-encoding), times each from the card's frame
     clock, keeps the ones that don't overlap what we have, and files them like live ones:
     DATA/<stream>/<day>/<HH>/seg-<epoch>.m4s + DATA/<stream>/init/init-<fill id>.mp4.
Telling "not on the card yet" from "the camera wasn't recording": the card's own list. Once the
card has recordings from after a gap, whatever it lacks inside the gap it will never have (the
camera was off, rebooting or offline: it records the card and the cloud stream alike): that
part is final, `camera_offline`. Only while the card hasn't got past the gap yet, or a replay
failed, is it tried again later (growing waits, at most MAX_TRIES). State: the `fills` table.
"""
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime

import recorder  # db(), segment timing helpers, DATA, TZ

DATA = recorder.DATA
TZ = recorder.TZ
CTRL = os.environ.get("CTRL_SOCK", "/home/ubuntu/nvr/run/camera-cam1.sock")
POLL_S = 30
GAP_MIN_S = 4          # smaller holes aren't worth a replay
GAP_SETTLE_S = 20      # a gap's end must be this old (live resumed) before we try
LOOKBACK_S = 7 * 86400
CHUNK_S = 300
MAX_TRIES = 12
OVERLAP_OK_S = 1.5     # a fetched segment may overlap what we have by this much
RETRY_S = [60, 120, 300, 600, 1800, 3600]

SCHEMA = """
CREATE TABLE IF NOT EXISTS fills (
  stream   TEXT NOT NULL,
  gap_from REAL NOT NULL,
  gap_to   REAL NOT NULL,
  tries    INTEGER NOT NULL DEFAULT 0,
  next_try REAL NOT NULL DEFAULT 0,
  status   TEXT NOT NULL DEFAULT 'open',   -- open | done | camera_offline | gave_up
  note     TEXT,
  PRIMARY KEY (stream, gap_from)
);
"""


def log(msg):
    print(datetime.now(TZ).strftime("%H:%M:%S"), msg, flush=True)


def db():
    c = recorder.db()
    c.executescript(SCHEMA)
    return c


# ---------- talking to camera_stream.js ----------

def ctl(req, timeout=40):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(CTRL)
    s.sendall((json.dumps(req) + "\n").encode())
    return s


def card_search(t0, t1):
    """[(begin, end)] of the card's recordings in [t0, t1], epoch seconds."""
    s = ctl({"op": "search", "from": t0, "to": t1})
    out, buf = [], b""
    with s:
        while True:
            d = s.recv(65536)
            if not d:
                break
            buf += d
            *lines, buf = buf.split(b"\n")
            for ln in lines:
                m = json.loads(ln)
                if "error" in m:
                    raise RuntimeError(m["error"])
                if "rec" in m:
                    out.append((m["rec"][0], m["rec"][1]))
    return sorted(out)


# ---------- gaps ----------

def find_gaps(conn, stream, now):
    rows = conn.execute("SELECT start, start+dur FROM segments WHERE stream=? AND start>? ORDER BY start",
                        (stream, now - LOOKBACK_S)).fetchall()
    gaps, end = [], None
    for s, e in rows:
        if end is not None and s - end >= GAP_MIN_S:
            gaps.append((end, s))
        end = e if end is None else max(end, e)
    return [g for g in gaps if g[1] < now - GAP_SETTLE_S]


def covered(conn, stream, a, b):
    """seconds of [a, b] the index already has."""
    rows = conn.execute("SELECT start, start+dur FROM segments WHERE stream=? AND start>? AND start<?",
                        (stream, a - 60, b)).fetchall()
    return sum(max(0.0, min(b, e) - max(a, s)) for s, e in rows)


# ---------- one replay -> segments ----------

def fetch(conn, stream, a, b):
    """replay [a, b] from the card; index the new segments. -> seconds added."""
    fill_id = int(time.time() * 1000)
    tmp = tempfile.mkdtemp(prefix="fill-", dir=os.path.join(DATA, "tmp"))
    try:
        s = ctl({"op": "replay", "from": a, "to": b}, timeout=60)
        # the first line: {"start": epoch of the first frame} (or an error), then MPEG-TS
        head = b""
        while b"\n" not in head:
            d = s.recv(4096)
            if not d:
                raise RuntimeError("no reply")
            head += d
        first, rest = head.split(b"\n", 1)
        m = json.loads(first)
        if "error" in m:
            raise RuntimeError(m["error"])
        start0 = float(m["start"])
        ff = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-v", "error", "-f", "mpegts", "-i", "-",
             "-map", "0:v", "-map", "0:a?", "-c", "copy", "-tag:v", "hvc1", "-bsf:a", "aac_adtstoasc",
             "-f", "hls", "-hls_time", "2", "-hls_list_size", "0", "-hls_segment_type", "fmp4",
             "-hls_fmp4_init_filename", "init.mp4", "-hls_segment_filename", os.path.join(tmp, "s%05d.m4s"),
             os.path.join(tmp, "p.m3u8")], stdin=subprocess.PIPE)
        try:
            ff.stdin.write(rest)
            s.settimeout(60)
            while True:
                d = s.recv(262144)
                if not d:
                    break
                ff.stdin.write(d)
        finally:
            s.close()
            ff.stdin.close()
            ff.wait()
        return place(conn, stream, tmp, start0, fill_id)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def place(conn, stream, tmp, start0, fill_id):
    pl = os.path.join(tmp, "p.m3u8")
    init = os.path.join(tmp, "init.mp4")
    if not os.path.exists(pl) or not os.path.exists(init):
        return 0.0
    names = [ln.strip() for ln in open(pl) if ln.strip().endswith(".m4s")]
    durs = [float(x) for x in re.findall(r"#EXTINF:([\d.]+)", open(pl).read())]
    scale = recorder.video_timescale(init)
    tfdt0, added, kept = None, 0.0, []
    for name, dur in zip(names, durs):
        f = os.path.join(tmp, name)
        d = recorder.first_tfdt(f)
        if d is None:
            continue
        tfdt0 = d if tfdt0 is None else tfdt0
        start = start0 + (d - tfdt0) / scale
        if covered(conn, stream, start, start + dur) > OVERLAP_OK_S:
            continue
        kept.append((f, start, dur))
    if not kept:
        return 0.0
    sdir = os.path.join(DATA, stream)
    init_rel = os.path.join(stream, "init", f"init-{fill_id}.mp4")
    shutil.move(init, os.path.join(DATA, init_rel))
    for f, start, dur in kept:
        t = datetime.fromtimestamp(start, TZ)
        hour = os.path.join(sdir, t.strftime("%Y-%m-%d"), t.strftime("%H"))
        os.makedirs(hour, exist_ok=True)
        n = int(start)
        while os.path.exists(os.path.join(hour, f"seg-{n}.m4s")):
            n += 1
        dst = os.path.join(hour, f"seg-{n}.m4s")
        shutil.move(f, dst)
        conn.execute("INSERT OR IGNORE INTO segments(stream,start,dur,path,init,run,bytes) VALUES(?,?,?,?,?,?,?)",
                     (stream, start, dur, os.path.relpath(dst, DATA), init_rel, fill_id, os.path.getsize(dst)))
        added += dur
    conn.commit()
    return added


# ---------- the loop ----------

def hm(t):
    return datetime.fromtimestamp(t, TZ).strftime("%H:%M:%S")


def once(conn, stream):
    now = time.time()
    gaps = find_gaps(conn, stream, now)
    for g0, g1 in gaps:
        # a gap that's partly filled comes back smaller: keep its row (and its tries)
        row = conn.execute("SELECT gap_from FROM fills WHERE stream=? AND gap_from<? AND gap_to>? "
                           "ORDER BY tries DESC LIMIT 1", (stream, g1, g0)).fetchone()
        if row and row[0] != g0:
            conn.execute("DELETE FROM fills WHERE stream=? AND gap_from=?", (stream, g0))
            conn.execute("UPDATE fills SET gap_from=?, gap_to=? WHERE stream=? AND gap_from=?", (g0, g1, stream, row[0]))
        elif row:
            conn.execute("UPDATE fills SET gap_to=? WHERE stream=? AND gap_from=?", (g1, stream, g0))
        else:
            conn.execute("INSERT INTO fills(stream, gap_from, gap_to) VALUES(?,?,?)", (stream, g0, g1))
    conn.commit()
    todo = conn.execute("SELECT gap_from, gap_to, tries FROM fills WHERE stream=? AND status='open' AND next_try<=? "
                        "ORDER BY gap_from DESC", (stream, now)).fetchall()
    open_gaps = {round(g0, 3) for g0, _ in gaps}
    for g0, g1, tries in todo:
        if round(g0, 3) not in open_gaps:  # filled since (or aged out of the lookback)
            conn.execute("UPDATE fills SET status='done' WHERE stream=? AND gap_from=?", (stream, g0))
            conn.commit()
            continue
        note, added, card_past = "", 0.0, False
        try:
            card = card_search(g0 - 60, g1 + 600)
            card_past = any(b >= g1 for b, e in card)  # the card has moved on past the gap
            want = [(max(g0, b), min(g1, e)) for b, e in card if e > g0 and b < g1]
            if not want:
                note = "the camera wasn't recording (nothing on the card)" if card_past else "not on the card yet"
            for a, b in want:
                while a < b - 1:
                    piece = min(b, a + CHUNK_S)
                    log(f"fill {hm(a)}-{hm(piece)} ({piece - a:.0f} s) from the card")
                    got = fetch(conn, stream, a, piece)
                    added += got
                    log(f"  added {got:.0f} s")
                    a = piece
        except (OSError, RuntimeError, ValueError) as e:
            note = f"error: {e}"
        left = max(0.0, (g1 - g0) - covered(conn, stream, g0, g1))
        if left < GAP_MIN_S:
            status = "done"
        elif card_past and not note.startswith("error"):
            status = "camera_offline"   # what's left isn't on the card and never will be
        elif tries + 1 >= MAX_TRIES:
            status = "gave_up"
        else:
            status = "open"
        wait = RETRY_S[min(tries, len(RETRY_S) - 1)]
        conn.execute("UPDATE fills SET tries=tries+1, next_try=?, status=?, note=? WHERE stream=? AND gap_from=?",
                     (time.time() + wait, status, note or f"added {added:.0f} s, {left:.0f} s left", stream, g0))
        conn.commit()
        log(f"gap {hm(g0)}-{hm(g1)} ({g1 - g0:.0f} s): {status}; {note or f'added {added:.0f} s, {left:.0f} s left'}")
        if note.startswith("error"):
            break  # the session is down or busy: try again next round


def main():
    stream = sys.argv[1] if len(sys.argv) > 1 else "cam1"
    os.makedirs(os.path.join(DATA, "tmp"), exist_ok=True)
    conn = db()
    while True:
        try:
            once(conn, stream)
        except Exception as e:  # keep going; log it
            log(f"error: {e!r}")
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
