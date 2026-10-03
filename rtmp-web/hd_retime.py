#!/usr/bin/env python3
"""
Puts each live HD clip at the time its picture was taken, read off the camera's own clock in
the picture (its OSD, "2026/10/03 17:30:07" top left).

hd_record.py names a clip by when its frames ARRIVED; after a reconnect the cloud delivers
the camera's buffered picture 13-60 s late, so a clip named 17:44:48 can start with 17:44:12
on screen, and the time bar overlaps / gaps where the picture doesn't (and while the link
struggles a 37 s clip can hold 12 s of picture, stretched). This reads the clock in the first
and last READ_S seconds of a finished clip (digits by template matching, models/osd_digits.npz,
learned from SD-card copies whose times are known; see experiments/osd/), takes the consensus
at each end, and
  - renames it <stream>-hd-<true epoch>_<day>_<time>.mp4 (into the right day folder),
  - sets its mtime (the player's clip end) to the clock at its last frame.
Clips under JUNK_S are moved to <day>/junk/ (the player ignores subfolders; they go with the
day). SD-card copies (filled.log) are already named by the camera's clock and are left alone.
Done clips are listed in <stream>/hd/retimed.log (old name -> new name, offset, frames read).

  python3 hd_retime.py [stream=cam1] [--days=2]        (hd-retime.timer, every 5 min)
"""
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REC_DIR = os.environ.get("REC_DIR", "/home/ubuntu/rtmp-recordings")
TZ = ZoneInfo(os.environ.get("REC_TZ", "Asia/Kolkata"))
DIGITS = os.path.join(HERE, "models", "osd_digits.npz")
READ_S = 2.5          # seconds read at each end of a clip, every frame (after a reconnect the
                      # picture comes in a burst, so it runs ahead of the clip's own timestamps;
                      # a short stretch at each end keeps that small)
SETTLED_S = 60        # leave clips written in the last minute (still open)
JUNK_S = 2
FULL_S = 280          # clips this long are whole recording pieces: end = start + length (their picture
                      # runs at its own speed; reading the end of a long clip means decoding up to a
                      # whole ~48 s keyframe interval); shorter ones get their end read off the clock
MAX_SHIFT = (-300, 30)  # believable offsets (true start - name), seconds
# the clock's six digits (H H M M S S) in the 2304x1296 picture: cell centres, half width, rows
CENTERS, HW, Y0, Y1, SH = [358, 386, 432, 460, 508, 536], 15, 6, 54, 4
CROP = "crop=600:64:0:0"


def log(msg):
    print(msg, flush=True)


def whiteness(rgb):
    """the clock is bright and colourless; foliage and sky behind it mostly aren't"""
    a = rgb.astype(np.float32)
    w = a.min(2) - (a.max(2) - a.min(2))
    return np.clip((w - 140) / 70, 0, 1)


def cell(g, c, dx=0):
    x = g[Y0:Y1, c - HW + dx:c + HW + dx]
    x = x - x.mean()
    return x / (np.linalg.norm(x) + 1e-6)


def read_digits(g, T):
    out = []
    for c in CENTERS:
        cells = [cell(g, c, dx) for dx in range(-SH, SH + 1)]
        out.append(max(range(10), key=lambda d: max(float((x * T[d]).sum()) for x in cells)))
    return out


def frames(path, tail=False):
    """(pts seconds, 600x64 RGB) for the frames of the first (or last) READ_S seconds"""
    where = ["-sseof", f"-{READ_S}", "-copyts"] if tail else ["-t", str(READ_S)]
    p = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "info", *where, "-i", path,
         "-vf", f"{CROP},showinfo", "-fps_mode", "passthrough",
         "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, timeout=300)
    pts = [float(m) for m in re.findall(rb"pts_time:\s*([\d.]+)", p.stderr)]
    size = 600 * 64 * 3
    imgs = [np.frombuffer(p.stdout[i * size:(i + 1) * size], np.uint8).reshape(64, 600, 3)
            for i in range(len(p.stdout) // size)]
    return list(zip(pts, imgs))


def duration(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True, timeout=60)
    try:
        return float(p.stdout.strip())
    except ValueError:
        return None


def night(img):
    """the infrared night picture is grey: the digits aren't learned for it yet"""
    a = img.astype(np.int16)
    return float((a.max(2) - a.min(2)).mean()) < 4


def clock_at(path, named, T, dur=None):
    """the clip's first frame's time by the camera's clock (or with dur, its end), or (None, why)"""
    v = []
    fr = frames(path, tail=dur is not None)
    if fr and night(fr[0][1]):
        return None, "night picture (not read yet)"
    for pts, img in fr:
        d = read_digits(whiteness(img), T)
        h, m, s = d[0] * 10 + d[1], d[2] * 10 + d[3], d[4] * 10 + d[5]
        if h > 23 or m > 59 or s > 59:
            continue
        base = datetime.fromtimestamp(named, TZ)
        cands = [(base + timedelta(days=k)).replace(hour=h, minute=m, second=s, microsecond=0) for k in (-1, 0, 1)]
        t = min(cands, key=lambda c: abs(c.timestamp() - named)).timestamp()
        # the clock shows floor(true time): the start is in [t - pts, t + 1 - pts), the end likewise
        v.append(t - pts + (dur or 0))
    if len(v) < 3:
        return None, f"only {len(v)} frames read"
    v = np.array(v)
    med = np.median(v)
    ok = v[np.abs(v - med) <= 1.5]
    if len(ok) < max(3, 0.6 * len(v)):
        return None, f"reads disagree ({len(ok)}/{len(v)})"
    lo, hi = ok.max(), ok.min() + 1  # where the second ticks over pins it down
    return ((lo + hi) / 2 if lo < hi else med + 0.5), f"{len(ok)}/{len(v)} frames"


def main():
    stream = next((a for a in sys.argv[1:] if not a.startswith("--")), "cam1")
    ndays = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--days=")), "2"))
    hd = os.path.join(REC_DIR, stream, "hd")
    T = {int(k): v for k, v in np.load(DIGITS).items()}
    done = set()
    for name in ("filled.log", "retimed.log"):
        try:
            done |= set(re.findall(rf"{stream}-hd-\d+_[\d-]+_[\d-]+\.mp4", open(os.path.join(hd, name)).read()))
        except FileNotFoundError:
            pass
    today = datetime.now(TZ)
    days = [(today - timedelta(days=k)).strftime("%Y-%m-%d") for k in range(ndays)]
    n = moved = 0
    for day in days:
        folder = os.path.join(hd, day)
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            m = re.fullmatch(rf"{stream}-hd-(\d+)_[\d-]+_[\d-]+\.mp4", f)
            path = os.path.join(folder, f)
            if not m or f in done or os.path.getmtime(path) > time.time() - SETTLED_S:
                continue
            named, n = int(m.group(1)), n + 1
            dur = duration(path)
            if dur is None or dur < JUNK_S:
                os.makedirs(os.path.join(folder, "junk"), exist_ok=True)
                os.rename(path, os.path.join(folder, "junk", f))
                line = f"{datetime.now(TZ).isoformat(timespec='seconds')} {f} -> junk/ ({dur} s)"
            else:
                try:
                    start, why = clock_at(path, named, T)
                    end = clock_at(path, named, T, dur)[0] if start is not None and dur < FULL_S else None
                except Exception as e:  # e.g. ffmpeg too slow: leave this one, carry on
                    start, why, end = None, f"error {type(e).__name__}", None
                # the picture can't be much shorter than a fifth of the clip, nor run more than
                # ~a minute ahead of it (a reconnect's burst)
                if end is None or not 0.2 * dur <= end - start <= dur + 70:
                    end = None
                if start is None or not MAX_SHIFT[0] <= start - named <= MAX_SHIFT[1]:
                    line = f"{datetime.now(TZ).isoformat(timespec='seconds')} {f} kept: " + (
                        why if start is None else f"clock says {start - named:+.1f} s, not believable")
                else:
                    t = datetime.fromtimestamp(start, TZ)
                    new_day = t.strftime("%Y-%m-%d")
                    new = f"{stream}-hd-{int(start)}_{new_day}_{t.strftime('%H-%M-%S')}.mp4"
                    dest = os.path.join(hd, new_day, new)
                    if new != f and os.path.exists(dest):
                        new, dest = f, path  # that name is taken (shouldn't happen): only fix the end
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    os.rename(path, dest)
                    os.utime(dest, (end or start + dur,) * 2)
                    moved += new != f
                    done.add(new)  # it may have moved into a folder still to come
                    line = (f"{datetime.now(TZ).isoformat(timespec='seconds')} {f} -> {new} "
                            f"({start - named:+.1f} s, {dur:.0f} s long, picture {(end - start) if end else dur:.0f} s, {why})")
            open(os.path.join(hd, "retimed.log"), "a").write(line + "\n")
            log(line)
    log(f"hd_retime: {n} clips looked at, {moved} renamed")


if __name__ == "__main__":
    main()
