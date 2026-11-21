#!/usr/bin/env python3
"""
NVR API: playlists, coverage and exports, all from the recorder's SQLite index.

    python3 playback_api.py            (nvr-playback.service; listens on 127.0.0.1:PORT, nginx in front)

  GET /api/streams                         [{"id", "name"}]
  GET /api/coverage?stream&from&to         {"spans": [[start, end], ...],   (epoch s; gaps < GAP s merged)
                                            "card": [...],    the parts filled from the camera's SD card
                                            "offline": [...]} gaps the camera itself didn't record
  GET /api/vod.m3u8?stream&from&to         HLS VOD playlist of the segments in [from, to]
  GET /api/live.m3u8?stream                HLS live playlist: the newest LIVE_SEGS segments
  GET /api/export.mp4?stream&from&to       one MP4 of [from, to] (stream copy, no re-encode)
  GET /api/status                          recorder health: newest segment, disk

Segments are served by nginx from /media/ (DATA on disk). Playlists carry PROGRAM-DATE-TIME
on every segment, and a DISCONTINUITY (with the new init segment) wherever there's a gap
or the source restarted, so a player always knows the wall-clock time it's showing.
"""
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

DATA = os.environ.get("NVR_DATA", "/home/ubuntu/nvr-data")
PORT = int(os.environ.get("PORT", "3400"))
MEDIA = "/media/"          # where nginx serves DATA
STREAMS = {"cam1": "Guttedar Nilaya"}
GAP = 2.0                  # seconds: a smaller hole between segments isn't a gap
LIVE_SEGS = 4
LIVE_TARGET = 15          # fixed TARGETDURATION for live (segments run up to ~13 s: the camera's
                          # key frames are 10 s apart); it mustn't change between reloads
CARD_RUN = 10 ** 11        # gapfill.py's run ids are epoch ms: above this = from the SD card
MAX_SEG = 60              # no segment is longer: lets time-range queries use the index both ways
MAX_VOD = 24 * 3600        # longest playlist
MAX_EXPORT = 3600          # longest export

# ---- background jobs dashboard (/api/jobs) ----------------------------------
# Fixed allowlist of the systemd units that make up the NVR; the /api/jobs endpoints
# only ever name a unit from here, so no caller-supplied string reaches the shell
# (and we never use shell=True — every call is an argv list).
JOBS = [
    ("nvr-recorder",   "Recorder"),
    ("nvr-detect",     "Detection poller"),
    ("nvr-detect-api", "Detection API"),
    ("nvr-playback",   "Playback API"),
    ("nvr-gapfill",    "SD gap-fill"),
    ("nvr-signin",     "Sign-in"),
]
JOB_UNITS = {u for u, _ in JOBS}
_LOGLINE = re.compile(r"^(\S+)\s+\S+\s+[^:]+:\s?(.*)$")   # "<iso> <host> <comm>[pid]: <msg>" -> iso, msg


def _run(argv, timeout=6):
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def unit_logs(unit, n=120):
    if unit not in JOB_UNITS:
        return []
    out = _run(["journalctl", "-u", unit, "-n", str(n), "--no-pager", "-o", "short-iso"])
    lines = []
    for ln in out.splitlines():
        if not ln or ln.startswith("-- "):            # drop "-- No entries --" / boot markers
            continue
        m = _LOGLINE.match(ln)
        lines.append(f"{m.group(1)} {m.group(2)}" if m else ln)
    return lines


def job_status(unit, tail=6):
    p = {}
    for ln in _run(["systemctl", "show", unit, "-p",
                    "ActiveState,SubState,NRestarts,MemoryCurrent,CPUUsageNSec,"
                    "ActiveEnterTimestampMonotonic,Description,MainPID"]).splitlines():
        k, _, v = ln.partition("=")
        p[k] = v

    def num(k):
        try:
            return int(p.get(k, ""))
        except ValueError:
            return None

    mem, cpu = num("MemoryCurrent"), num("CPUUsageNSec")
    entered = num("ActiveEnterTimestampMonotonic")
    active = p.get("ActiveState") == "active"
    uptime = None
    if active and entered:
        uptime = round(time.clock_gettime(time.CLOCK_MONOTONIC) - entered / 1e6)
    return {
        "unit": unit, "active": p.get("ActiveState"), "sub": p.get("SubState"),
        "desc": p.get("Description"), "restarts": num("NRestarts"), "pid": num("MainPID") or None,
        "uptime_s": uptime if uptime and uptime > 0 else None,
        "mem_mb": round(mem / 1048576, 1) if mem is not None and mem < (1 << 62) else None,
        "cpu_s": round(cpu / 1e9) if cpu else None,
        "tail": unit_logs(unit, tail),
    }


def db():
    c = sqlite3.connect(f"file:{os.path.join(DATA, 'nvr.db')}?mode=ro", uri=True, timeout=10)
    c.row_factory = sqlite3.Row
    return c


def pdt(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def segments(stream, t0, t1):
    with db() as c:
        return c.execute("SELECT start, dur, path, init, run FROM segments WHERE stream=? AND start>? AND start<? "
                         "AND start+dur>? ORDER BY start", (stream, t0 - MAX_SEG, t1, t0)).fetchall()


def playlist(rows, live=False, seq=0, dseq=0):
    tgt = max([math.ceil(r["dur"]) for r in rows] + [LIVE_TARGET if live else 10])
    out = ["#EXTM3U", "#EXT-X-VERSION:7", f"#EXT-X-TARGETDURATION:{tgt}", "#EXT-X-INDEPENDENT-SEGMENTS"]
    if live:
        out.append(f"#EXT-X-MEDIA-SEQUENCE:{seq}")
        out.append(f"#EXT-X-DISCONTINUITY-SEQUENCE:{dseq}")
    else:
        out.append("#EXT-X-PLAYLIST-TYPE:VOD")
    prev = None
    for r in rows:
        if prev is None or r["init"] != prev["init"] or r["start"] - (prev["start"] + prev["dur"]) > GAP:
            if prev is not None:
                out.append("#EXT-X-DISCONTINUITY")
            out.append(f'#EXT-X-MAP:URI="{MEDIA}{r["init"]}"')
        out.append(f"#EXT-X-PROGRAM-DATE-TIME:{pdt(r['start'])}")
        out.append(f"#EXTINF:{r['dur']:.3f},")
        out.append(MEDIA + r["path"])
        prev = r
    if not live:
        out.append("#EXT-X-ENDLIST")
    return "\n".join(out) + "\n"


def offline(stream, t0, t1, rows):
    """the parts of gaps gapfill.py found the camera didn't record (not on its card either)."""
    try:
        with db() as c:
            gaps = c.execute("SELECT gap_from, gap_to FROM fills WHERE stream=? AND status='camera_offline' "
                             "AND gap_to>? AND gap_from<?", (stream, t0, t1)).fetchall()
    except sqlite3.OperationalError:  # no gap filler yet
        return []
    out = []
    for g0, g1 in gaps:
        # minus what's recorded inside the gap (the card's part)
        cur = g0
        for r in rows:
            s, e = r["start"], r["start"] + r["dur"]
            if e <= cur or s >= g1:
                continue
            if s > cur:
                out.append([round(cur, 3), round(s, 3)])
            cur = max(cur, e)
        if cur < g1:
            out.append([round(cur, 3), round(g1, 3)])
    return out


def spans(rows):
    out = []
    for r in rows:
        s, e = r["start"], r["start"] + r["dur"]
        if out and s - out[-1][1] <= GAP:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [[round(s, 3), round(e, 3)] for s, e in out]


class H(BaseHTTPRequestHandler):
    server_version = "nvr-playback"

    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def err(self, code, msg):
        self.send(code, json.dumps({"error": msg}))

    def args(self, q, need_range=True, limit=MAX_VOD):
        stream = q.get("stream", ["cam1"])[0]
        if stream not in STREAMS:
            raise ValueError("unknown stream")
        if not need_range:
            return stream, None, None
        t0, t1 = float(q["from"][0]), float(q["to"][0])
        if not 0 < t1 - t0 <= limit:
            raise ValueError(f"range must be 0..{limit} s")
        return stream, t0, t1

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/api/streams":
                return self.send(200, json.dumps([{"id": k, "name": v} for k, v in STREAMS.items()]))
            if u.path == "/api/coverage":
                s, t0, t1 = self.args(q, limit=40 * 86400)
                rows = segments(s, t0, t1)
                return self.send(200, json.dumps({"spans": spans(rows),
                                                  "card": spans([r for r in rows if r["run"] > CARD_RUN]),
                                                  "offline": offline(s, t0, t1, rows)}))
            if u.path == "/api/vod.m3u8":
                s, t0, t1 = self.args(q)
                rows = segments(s, t0, t1)
                if not rows:
                    return self.err(404, "no recording in that range")
                return self.send(200, playlist(rows), "application/vnd.apple.mpegurl")
            if u.path == "/api/live.m3u8":
                s, _, _ = self.args(q, need_range=False)
                with db() as c:
                    rows = c.execute("SELECT id, start, dur, path, init, run FROM segments WHERE stream=? "
                                     "ORDER BY start DESC LIMIT ?", (s, LIVE_SEGS)).fetchall()[::-1]
                    # discontinuities before the window: the runs table's seq only ever grows
                    # (counting the runs still on disk went down when retention deleted one)
                    dseq = 0
                    if rows:
                        r0 = c.execute("SELECT seq FROM runs WHERE run=?", (rows[0]["run"],)).fetchone()
                        dseq = r0["seq"] if r0 else 0
                if not rows or time.time() - (rows[-1]["start"] + rows[-1]["dur"]) > 60:
                    return self.err(503, "the HD stream is down")
                return self.send(200, playlist(rows, live=True, seq=rows[0]["id"], dseq=dseq),
                                 "application/vnd.apple.mpegurl")
            if u.path == "/api/status":
                with db() as c:
                    r = dict(c.execute("SELECT count(*) AS n, sum(bytes) AS bytes FROM segments").fetchone())
                    newest = c.execute("SELECT start+dur FROM segments ORDER BY start DESC LIMIT 1").fetchone()
                    oldest = c.execute("SELECT start FROM segments ORDER BY start LIMIT 1").fetchone()
                    r["last"], r["first"] = newest and newest[0], oldest and oldest[0]
                du = shutil.disk_usage(DATA)
                return self.send(200, json.dumps({
                    "newest": r["last"], "oldest": r["first"], "segments": r["n"], "bytes": r["bytes"],
                    "behind_s": round(time.time() - r["last"], 1) if r["last"] else None,
                    "disk_free_gb": round(du.free / 1e9, 1), "disk_total_gb": round(du.total / 1e9, 1)}))
            if u.path == "/api/export.mp4":
                return self.export(*self.args(q, limit=MAX_EXPORT))
            if u.path == "/api/jobs":
                return self.send(200, json.dumps({"now": time.time(),
                                                  "services": [job_status(un) for un, _ in JOBS]}))
            if u.path == "/api/jobs/logs":
                unit = q.get("unit", [""])[0]
                if unit not in JOB_UNITS:
                    return self.err(400, "unknown unit")
                n = max(1, min(int(q.get("n", ["300"])[0]), 1000))
                return self.send(200, json.dumps({"unit": unit, "lines": unit_logs(unit, n)}))
            return self.err(404, "not found")
        except (KeyError, ValueError) as e:
            return self.err(400, str(e))

    def export(self, stream, t0, t1):
        """The stored segments covering [t0, t1], whole (so it starts on a key frame: up to ~10 s
        early), as one MP4 without re-encoding. Each source run (its own init segment and
        timeline) is joined into a file of its own, and ffmpeg's concat demuxer puts the runs
        end to end; cutting mid-segment gave a broken start, and one playlist across runs gave
        non-increasing timestamps."""
        rows = segments(stream, t0, t1)
        if not rows:
            return self.err(404, "no recording in that range")
        tmp = tempfile.mkdtemp(prefix="export-", dir=os.path.join(DATA, "tmp") if os.path.isdir(os.path.join(DATA, "tmp")) else None)
        try:
            parts, cur = [], None
            for r in rows:
                if cur is None or r["run"] != cur[0]:
                    cur = (r["run"], os.path.join(tmp, f"part{len(parts)}.mp4"))
                    parts.append(cur)
                    with open(cur[1], "wb") as out, open(os.path.join(DATA, r["init"]), "rb") as f:
                        shutil.copyfileobj(f, out)
                with open(cur[1], "ab") as out, open(os.path.join(DATA, r["path"]), "rb") as f:
                    shutil.copyfileobj(f, out)
            # each part re-muxed to start at 0 (a card part's media time starts wherever the card's
            # did, and the concat demuxer kept that offset: a jump of minutes at the join)
            norm = []
            for i, (_, path) in enumerate(parts):
                out = os.path.join(tmp, f"norm{i}.mp4")
                if subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-map", "0", "-c", "copy",
                                   "-tag:v", "hvc1", "-avoid_negative_ts", "make_zero", "-f", "mp4", out]).returncode == 0:
                    norm.append(out)
            lst = os.path.join(tmp, "list.txt")
            with open(lst, "w") as f:
                f.writelines(f"file '{path}'\n" for path in norm)
            name = f"{stream}_{datetime.fromtimestamp(rows[0]['start']).strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            p = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
                                  "-c", "copy", "-tag:v", "hvc1",
                                  "-movflags", "frag_keyframe+empty_moov+default_base_moof", "-f", "mp4", "-"],
                                 stdout=subprocess.PIPE)
            try:
                shutil.copyfileobj(p.stdout, self.wfile, 256 * 1024)
            except (BrokenPipeError, ConnectionResetError):
                p.kill()
            finally:
                p.wait()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
