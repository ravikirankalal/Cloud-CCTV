#!/usr/bin/env python3
"""
nginx-rtmp recordings indexer

  * remuxes finished .flv recordings to .mp4 (stream copy, no re-encode)
  * grabs a thumbnail for every recording
  * probes duration / resolution / fps / codecs with ffprobe
  * finds motion events, then people / pets in them (detect.py, optional)
  * writes index.json, which the web page (index.html) reads

Finished recordings are filed by camera (stream: nothing is ever shared between two),
then by the day they started (in REC_TZ, India by default), one folder each, named after
that start time:

  cam1/2026-09-27/cam1-1790449500_2026-09-27_00-15-00/
      cam1-1790449500_2026-09-27_00-15-00.mp4
      thumb.jpg      info.json (probe, timestamp repair)
      motion.json    detect.json    people/<event>-<person>.jpg (person crops)

(<stream>-<start, unix time>_<start date and time>). What is about a whole camera is in
its folder too: index.json (its recordings, for the page), people.json and, in .meta, the
people grouping's state. streams.json at the top lists the cameras. Recordings in progress
stay where nginx writes them (<stream>-<unix>_<UTC date and time>.flv at the top: it can't
write per stream); the top .meta holds the locks shared by all cameras (one conversion run,
one detect.py at a time: two don't fit in memory) and failed-conversion markers.

Run it every minute from ROOT's crontab (the recordings are written by the
container, so a normal user usually can't create files in that folder):

  sudo crontab -e
  * * * * * /usr/bin/python3 /home/ubuntu/rtmp-web/recordings.py >> /var/log/rtmp-recordings.log 2>&1

Env overrides:  REC_DIR=/path/to/recordings   KEEP_FLV=1 (keep originals)   REC_TZ=Asia/Kolkata
"""
import fcntl
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

REC_DIR = os.environ.get("REC_DIR", "/home/ubuntu/rtmp-recordings")
KEEP_FLV = os.environ.get("KEEP_FLV", "0") == "1"
SETTLE = 90              # seconds without writes before an FLV counts as finished
MIN_BYTES = 64 * 1024    # ignore stub files left by aborted connections
# Some cameras open a stream with a stale frame (or a short burst of them) and then
# jump their timestamps forward several seconds (or minutes: "ramesh" sends one frame
# 714 times, then jumps 242 s). Browsers can't seek inside that jump, so it gets cut:
# a burst shorter than JUNK_SPAN followed by a jump of at least JUNK_GAP, looked for in
# the first JUNK_WINDOW packets (a count, since the jump can be any length).
JUNK_SPAN = 3.0
JUNK_GAP = 2.0
JUNK_WINDOW = 3000

# Motion detection (ffmpeg only): take the keyframes, shrink them to grey 160px, and
# measure the share of pixels whose brightness changed by more than MOTION_PIXEL since
# the previous keyframe (an erosion pass drops speckle noise).
MOTION = os.environ.get("MOTION", "1") == "1"
MOTION_MIN = float(os.environ.get("MOTION_MIN", "0.2"))  # % of picture changed = motion
# Motion is judged between consecutive keyframes (these cameras send one every 2 s):
# ffmpeg can decode those alone, so a 30-minute recording is ~900 pictures instead of
# ~27,000. The server's CPU is shared and throttled; decoding every frame took 3.5 min.
MOTION_PIXEL = 20
MOTION_TOP = 0.10     # ignore the top 10% of the picture: the camera's clock overlay
MOTION_MERGE = 5      # join motion less than this many seconds apart into one event
MOTION_PAD = 2        # seconds kept before and after each event
MOTION_STEP = 2       # seconds between keyframes: a change seen at one began after the last
MOTION_BUDGET = 240   # seconds of analysis per run, so new recordings still get indexed

# Person / pet detection on motion events (detect.py: YOLOX-Tiny in the .venv next to
# this script). Raw results are cached per recording; LABEL_MIN / LABEL_SEEN decide
# what the page shows, so they can be tuned without re-running the model.
HERE = os.path.dirname(os.path.abspath(__file__))
DETECT_PY = os.path.join(HERE, ".venv", "bin", "python")
DETECT = (os.environ.get("DETECT", "1") == "1" and os.path.exists(DETECT_PY)
          and all(os.path.exists(os.path.join(HERE, "models", m))
                  for m in ("yolox_tiny.onnx", "osnet_ain_x1_0_msmt17.onnx")))
DETECT_TRIES = 3      # attempts before a recording that detect.py fails on is left as it is
DETECT_V = 3          # bump to re-run detection on every recording (2: people fingerprints,
                      # 3: several people per event told apart)
STREAM_OK = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # stream names used in paths and arguments
DETECT_BUDGET = 240   # seconds of detection per run (on top of motion)
LABEL_MIN = 0.5       # a label counts if its best frame scores this ...
LABEL_SEEN = ((2, 0.4), (3, 0.33))  # ... or it's seen in n+ frames with a best score this high

# Retention: keep as much as fits. When free space drops below MIN_FREE_GB, the
# oldest finished recordings are deleted until it is back above. Recordings from the
# last MIN_KEEP_HOURS are never deleted, so something else filling the disk can't
# wipe recent footage (a warning is logged instead).
MIN_FREE_GB = float(os.environ.get("MIN_FREE_GB", "25"))
MIN_KEEP_HOURS = float(os.environ.get("MIN_KEEP_HOURS", "24"))

META = os.path.join(REC_DIR, ".meta")
TZ = ZoneInfo(os.environ.get("REC_TZ", "Asia/Kolkata"))  # the day folders' and names' time zone
STREAMS = os.path.join(REC_DIR, "streams.json")  # the cameras, for the page


def stream_file(stream, *part):
    """A camera's own file: stream_file("cam1", "index.json"), (.., ".meta", "x.lock")."""
    return os.path.join(REC_DIR, stream, *part)


def people_json(stream):
    return stream_file(stream, "people.json")


def feedback(stream):
    """The page's "same person" marks and names for a camera (people_api.py)."""
    return os.path.join(HERE, "state", f"people-feedback-{stream}.json")

# Matches nginx-rtmp names like:
#   mystream_2026-09-25_10-30-00             (record_suffix only)
#   mystream-1727260200_2026-09-25_10-30-00  (record_unique on + record_suffix)
NAME_RE = re.compile(
    r"^(?P<stream>.+?)(?:-(?P<unix>\d{9,11}))?"
    r"_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})$"
)


def log(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def parse_name(stem):
    """Return (stream_name, start_epoch or None)."""
    m = NAME_RE.match(stem)
    if not m:
        return stem, None
    if m["unix"]:
        return m["stream"], int(m["unix"])
    # strftime suffix is written in the container's timezone (UTC by default)
    dt = datetime.strptime(f'{m["date"]} {m["time"]}', "%Y-%m-%d %H-%M-%S")
    return m["stream"], int(dt.replace(tzinfo=timezone.utc).timestamp())


def rec_name(stream, start):
    """A finished recording's name (no extension): stream, start as unix time, and start
    as date and time in TZ."""
    return f"{stream}-{start}_{datetime.fromtimestamp(start, TZ):%Y-%m-%d_%H-%M-%S}"


def rec_dir(name):
    """The folder of a finished recording (name: with or without .mp4):
    <stream>/<day>/<name>."""
    stem = name[:-4] if name.endswith(".mp4") else name
    m = NAME_RE.match(stem)
    return os.path.join(REC_DIR, m["stream"], m["date"], stem) if m else os.path.join(REC_DIR, "undated", stem)


def rec_file(name, part):
    """A file in a finished recording's folder: "info.json", "thumb.jpg", ..."""
    return os.path.join(rec_dir(name), part)


def mp4_path(name):
    return rec_file(name, os.path.basename(rec_dir(name)) + ".mp4")


def all_mp4s():
    """The names (x.mp4) of all finished recordings."""
    out = []
    for path in glob.glob(os.path.join(REC_DIR, "*", "*", "*", "*.mp4")):
        name = os.path.basename(path)
        if not name.startswith(".") and path == mp4_path(name):
            out.append(name)
    return out


def rel(path):
    """path as the page asks for it (under /recordings/)."""
    return os.path.relpath(path, REC_DIR)


def probe(path):
    r = run(["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", path])
    if r.returncode:
        return {}
    try:
        d = json.loads(r.stdout or "{}")
    except ValueError:
        return {}
    info = {}
    fmt = d.get("format", {})
    try:
        info["duration"] = round(float(fmt.get("duration")), 2)
    except (TypeError, ValueError):
        pass
    br = fmt.get("bit_rate")
    if br and str(br).isdigit():
        info["bitrate"] = int(br)
    for s in d.get("streams", []):
        kind = s.get("codec_type")
        if kind == "video" and "vcodec" not in info:
            info.update(vcodec=s.get("codec_name"),
                        width=s.get("width"), height=s.get("height"))
            fr = s.get("avg_frame_rate") or s.get("r_frame_rate") or "0/0"
            try:
                num, den = (float(x) for x in fr.split("/"))
                if den and num:
                    info["fps"] = round(num / den, 2)
            except ValueError:
                pass
        elif kind == "audio" and "acodec" not in info:
            info["acodec"] = s.get("codec_name")
            if s.get("sample_rate"):
                info["sample_rate"] = int(s["sample_rate"])
    return info


def leading_junk(path):
    """Timestamp of the first real keyframe after a connect burst, or 0 if clean."""
    r = run(["ffprobe", "-v", "error", "-read_intervals", f"%+#{JUNK_WINDOW}",
             "-select_streams", "v", "-show_entries", "packet=pts_time,flags",
             "-of", "csv=p=0", path])
    first = prev = None
    for line in r.stdout.splitlines():
        parts = line.split(",")
        try:
            t = float(parts[0])
        except ValueError:
            continue
        if first is None:
            first = prev = t
            continue
        if t - prev >= JUNK_GAP:
            real = prev - first < JUNK_SPAN and len(parts) > 1 and "K" in parts[1]
            return round(t, 3) if real else 0.0
        if t - first >= JUNK_SPAN:
            return 0.0
        prev = t
    return 0.0


def read_info(mp4_name):
    """info.json: the probe cache (see mp4_entry) and "fix", the timestamp repair."""
    try:
        with open(rec_file(mp4_name, "info.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def write_info(mp4_name, info):
    path = rec_file(mp4_name, "info.json")
    with open(path + ".tmp", "w") as f:
        json.dump(info, f)
    os.replace(path + ".tmp", path)


def read_fix(mp4_name):
    return read_info(mp4_name).get("fix")


def write_fix(mp4_name, cut, start=None, **extra):
    info = read_info(mp4_name)
    info["fix"] = {"cut": cut, **({"start": start} if start is not None else {}), **extra}
    write_info(mp4_name, info)


# Sometimes the camera comes back from a reconnect sending every frame with timestamp 0
# (seen 2026-09-26: 30-minute files that play as 2 s). The frames are all there and in
# order, so such files are rebuilt with timestamps counted from the frames: the audio is
# AAC in 1024-sample frames, which gives the real length, and the video frame rate follows.
MAX_BYTES_PER_S = 2e6  # an apparent bitrate above 16 Mbps: timestamps are broken


def stamp_info(path):
    """Packet counts, when path's timestamps are broken (else None). Reads the whole file."""
    r = run(["ffprobe", "-v", "error", "-count_packets", "-print_format", "json",
             "-show_entries", "stream=codec_type,codec_name,nb_read_packets,sample_rate:format=duration",
             path])
    try:
        d = json.loads(r.stdout)
        dur = float(d["format"].get("duration") or 0)
    except (ValueError, KeyError):
        return None
    v = next((x for x in d.get("streams", []) if x.get("codec_type") == "video"), None)
    a = next((x for x in d.get("streams", []) if x.get("codec_type") == "audio"), None)
    if not v or v.get("codec_name") != "h264":
        return None
    nv = int(v.get("nb_read_packets") or 0)
    if nv < 100 or nv / max(dur, 0.001) < 100:  # at most 100 frames per second: fine
        return None
    info = {"frames": nv}
    if a and a.get("codec_name") == "aac" and a.get("sample_rate"):
        info["sr"] = int(a["sample_rate"])
        info["audio_s"] = int(a.get("nb_read_packets") or 0) * 1024 / info["sr"]
    return info


CLOCK_STEP = 30   # seconds of audio between the points that time the video (see audio_clock)


def audio_clock(src, sr):
    """[(video frame number, time s), ...]: how many video frames had arrived when each
    CLOCK_STEP seconds of audio had. The file keeps the packets in the order they arrived
    and AAC audio is a steady clock (1024 samples per packet), so this follows the
    camera's real frame rate even when it changes mid-recording (cam1 drops from 15 to
    ~10 fps at night)."""
    r = run(["ffprobe", "-v", "error", "-show_entries", "packet=codec_type", "-of", "csv=p=0", src])
    per = max(1, round(CLOCK_STEP * sr / 1024))  # audio packets per step
    points, na, nv = [(0, 0.0)], 0, 0
    for kind in r.stdout.split():
        if kind == "video":
            nv += 1
        elif kind == "audio":
            na += 1
            if na % per == 0 and nv > points[-1][0]:
                points.append((nv, na * 1024 / sr))
    end = (nv, na * 1024 / sr)
    if nv > points[-1][0] and end[1] > points[-1][1]:
        points.append(end)
    return points


def clock_expr(points):
    """setts expression placing frame N on the audio clock: straight lines between the
    points, the last one carried on past the end."""
    def line(i):
        (n0, t0), (n1, t1) = points[i], points[i + 1]
        return f"({t0:.4f}+(N-{n0})*{(t1 - t0) / (n1 - n0):.7f})"
    expr = line(len(points) - 2)
    for i in range(len(points) - 3, -1, -1):
        expr = f"if(lt(N\\,{points[i + 1][0]})\\,{line(i)}\\,{expr})"
    return f"setts=ts={expr}/TB"


def restamp(src, dst, info, wall_s, mtime):
    """Rebuild src as an MP4 at dst with timestamps counted from the frames: timed by the
    audio when there is some (see audio_clock), else spread evenly over the wall time."""
    length = info.get("audio_s") or wall_s
    if not length or length <= 0:
        return None
    fps = info["frames"] / length  # the average, for the log and fix.json
    if abs(fps - round(fps)) < 0.02 * fps:
        fps = float(round(fps))  # the camera's nominal rate
    clock = audio_clock(src, info["sr"]) if info.get("audio_s") else None
    if clock and len(clock) < 2:
        clock = None
    work = os.path.join(REC_DIR, "." + os.path.basename(dst) + ".restamp")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    v, a, tmp = (os.path.join(work, n) for n in ("v.h264", "a.aac", "out.mp4"))
    try:
        base = ["nice", "-n", "10", "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src]
        if run(base + ["-map", "0:v:0", "-c", "copy", "-bsf:v", "h264_mp4toannexb", "-f", "h264", v]).returncode:
            return None
        has_a = "audio_s" in info and not run(base + ["-map", "0:a:0", "-c", "copy", "-f", "adts", a]).returncode
        cmd = ["nice", "-n", "10", "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
               "-framerate", f"{fps:.4f}", "-i", v] + (["-i", a] if has_a else [])
        cmd += ["-map", "0:v"] + (["-map", "1:a"] if has_a else [])
        if clock and has_a:
            cmd += ["-bsf:v", clock_expr(clock)]
        r = run(cmd + ["-c", "copy", "-movflags", "+faststart", "-f", "mp4", tmp])
        if r.returncode or not os.path.exists(tmp):
            log(f"restamp failed: {os.path.basename(src)}: {r.stderr.strip()[:300]}")
            return None
        os.utime(tmp, (mtime, mtime))
        os.replace(tmp, dst)
        return fps
    finally:
        shutil.rmtree(work, ignore_errors=True)


def remux(src, dst, cut, mtime):
    """Stream-copy src to an MP4 at dst, dropping everything before `cut` seconds."""
    tmp = os.path.join(REC_DIR, "." + os.path.basename(dst) + ".part")
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src]
    if cut:
        # cut is a keyframe; start just before it so that keyframe is the first packet.
        # An output -ss counts from the input's start time, not from 0.
        r = run(["ffprobe", "-v", "error", "-show_entries", "format=start_time",
                 "-of", "csv=p=0", src])
        try:
            first = float(r.stdout.strip())
        except ValueError:
            first = 0.0
        cmd += ["-ss", f"{max(0.0, cut - first - 0.0005):.4f}"]
    cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart", "-f", "mp4", tmp]
    r = run(cmd)
    if r.returncode or not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
        log(f"remux failed: {os.path.basename(src)}: {r.stderr.strip()[:300]}")
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass
        return False
    os.utime(tmp, (mtime, mtime))  # keep the original end time
    os.replace(tmp, dst)
    return True


def convert(flv):
    """flv -> an MP4 in its own folder (see the top), named after its start."""
    stem = os.path.basename(flv)[:-4]
    work = os.path.join(REC_DIR, "." + stem + ".new.mp4")
    end = os.stat(flv).st_mtime
    stream, named = parse_name(stem)
    bad = stamp_info(flv)
    cut, note, extra = 0.0, "", {}
    if bad:  # no usable timestamps (so no junk cut either): count the frames instead
        fps = restamp(flv, work, bad, end - (named or end), end)
        if not fps:
            return None
        start, extra = named, {"stamps": fps}
        note = f" (camera sent no timestamps; rebuilt at {fps:g} fps)"
    else:
        cut = leading_junk(flv)
        if not remux(flv, work, cut, end):
            return None
        start = named
        if cut:
            # The jump overstates the real gap, so place the recording by when it
            # ended, kept within [name time, name time + cut].
            dur = probe(work).get("duration")
            if named is not None and dur:
                start = int(round(min(max(end - dur, named), named + cut)))
            note = f" (cut {cut:.1f}s connect junk)"
    if start is None:
        start = int(end - (probe(work).get("duration") or 0))
    name = rec_name(stream, start) + ".mp4"
    os.makedirs(rec_dir(name), exist_ok=True)
    os.replace(work, mp4_path(name))
    write_fix(name, cut, start if cut else None, **extra)
    if KEEP_FLV:
        open(os.path.join(META, os.path.basename(flv) + ".converted"), "w").close()
    else:
        os.remove(flv)
    log(f"converted {os.path.basename(flv)} -> {rel(mp4_path(name))}{note}")
    return name


def fix_existing(path):
    """One-off passes for MP4s converted before the junk cut / timestamp rebuild existed."""
    name = os.path.basename(path)
    fix = read_fix(name)
    if fix is not None:
        if "stamps" not in fix:
            fix_stamps(path, fix)
        return
    cut = leading_junk(path)
    if cut and not remux(path, path, cut, os.stat(path).st_mtime):
        return  # try again next run
    # the original end time is unknown for these, so keep the start from the name
    write_fix(name, cut)
    if cut:
        try:
            os.remove(rec_file(name, "thumb.jpg"))  # may show the junk frame
        except FileNotFoundError:
            pass
        log(f"fixed {name} (cut {cut:.1f}s connect junk)")


def fix_stamps(path, fix):
    """Rebuild an MP4 converted from a stream without timestamps (plays as ~2 s)."""
    name = os.path.basename(path)
    st = os.stat(path)
    dur = probe(path).get("duration") or 0
    bad = st.st_size / max(dur, 0.001) > MAX_BYTES_PER_S and stamp_info(path)
    if not bad:
        write_fix(name, fix.get("cut", 0.0), fix.get("start"), stamps="ok")
        return
    _, named = parse_name(name[:-4])
    fps = restamp(path, path, bad, st.st_mtime - (named or st.st_mtime), st.st_mtime)
    if not fps:
        return  # try again next run
    write_fix(name, fix.get("cut", 0.0), fix.get("start"), stamps=fps)
    keep_prev(name)  # its people are the same; only their times change
    for stale in (rec_file(name, "thumb.jpg"), motion_path(name), detect_path(name)):
        try:
            os.remove(stale)  # worked out on the broken timeline
        except FileNotFoundError:
            pass
    log(f"fixed {name} (camera sent no timestamps; rebuilt at {fps:g} fps)")


def thumbnail(mp4, duration):
    out = rec_file(os.path.basename(mp4), "thumb.jpg")
    if os.path.exists(out):
        return out
    at = max(0.0, min(10.0, (duration or 0) / 3))
    run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{at:.2f}",
         "-i", mp4, "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "4", out])
    return out if os.path.exists(out) else None


STATIC_EVENTS = {}  # file -> motion events whose only "person" is a fixed object (people.py)


def mp4_entry(path):
    name = os.path.basename(path)
    st = os.stat(path)
    meta = read_info(name)
    if meta.get("_size") != st.st_size or meta.get("_mtime") != int(st.st_mtime):
        meta = None

    if meta is None:
        info = probe(path)
        stream, start = parse_name(name[:-4])
        start = (read_fix(name) or {}).get("start", start)
        dur = info.get("duration") or 0
        if start is None:
            start = int(st.st_mtime - dur)
        thumb = thumbnail(path, dur)
        meta = {
            "file": name,
            "path": rel(path),
            "stream": stream,
            "start": start,
            "duration": dur,
            "size": st.st_size,
            "status": "ready",
            "thumb": rel(thumb) if thumb else None,
            **{k: v for k, v in info.items() if k != "duration"},
            "_size": st.st_size,
            "_mtime": int(st.st_mtime),
            **({"fix": read_fix(name)} if read_fix(name) else {}),
        }
        write_info(name, meta)

    if meta["duration"] < 1 and meta["size"] < MIN_BYTES:
        return None
    entry = {k: v for k, v in meta.items() if not k.startswith("_") and k != "fix"}
    motion = read_motion(name)
    if motion is not None:
        det = read_detect(name, motion)
        if det:  # add {label: [score, time_s]} to events where something was recognised
            fixed = set(STATIC_EVENTS.get(name, ()))
            found = [{k: v for k, v in f.items() if not (k == "person" and ev in fixed)}
                     for ev, f in enumerate(det["found"])]
            motion = [m + [labels(f)] if labels(f) else m for m, f in zip(motion, found)]
        entry["motion"] = motion  # [[start_s, end_s, peak_%, labels?], ...] relative to the file
        entry["detected"] = bool(det) and (det.get("v") == DETECT_V or det.get("tries", 0) >= DETECT_TRIES)
    return entry


def motion_path(mp4_name):
    return rec_file(mp4_name, "motion.json")


def read_motion(mp4_name):
    try:
        with open(motion_path(mp4_name)) as f:
            return json.load(f)["events"]
    except (OSError, ValueError, KeyError):
        return None


def motion_stale(mp4_name):
    """True if the motion events are marked to be found again (they are still shown
    until then). Used once: scans from 26 Sep 09:25 to the per-recording temp file could
    have read another camera's output."""
    try:
        with open(motion_path(mp4_name)) as f:
            return bool(json.load(f).get("stale"))
    except (OSError, ValueError):
        return False


def motion_vf(out=None):
    """ffmpeg filter printing, per keyframe, the share of pixels changed since the one
    before (YAVG, 0-255). Printed to `out`, or to ffmpeg's log (stderr) when out is None.
    Use with -skip_frame nokey on the input."""
    return (f"crop=iw:ih*{1 - MOTION_TOP}:0:ih*{MOTION_TOP},scale=160:-2,"
            f"format=gray,tblend=all_mode=difference,"
            f"lut=y=if(gt(val\\,{MOTION_PIXEL})\\,255\\,0),erosion,"
            f"signalstats,metadata=print:key=lavfi.signalstats.YAVG" + (f":file={out}" if out else ""))


def detect_motion(path, duration):
    """Motion events in an MP4 as [[start, end, peak %], ...] (seconds into the file)."""
    # a temp file per recording: each camera's analyser scans at the same time as the others
    out = stream_file(parse_name(os.path.basename(path)[:-4])[0], ".meta", f".motion-{os.path.basename(path)}.tmp")
    vf = motion_vf(out)
    r = run(["nice", "-n", "10", "ffmpeg", "-nostdin", "-v", "error", "-threads", "1",
             "-skip_frame", "nokey", "-i", path, "-an", "-vf", vf, "-f", "null", "-"])
    try:
        with open(out) as f:
            text = f.read()
        os.remove(out)
    except OSError:
        text = ""
    if r.returncode and not text:
        raise RuntimeError(r.stderr.strip()[:300])

    events, t = [], None
    for line in text.splitlines():
        if line.startswith("frame:"):
            m = re.search(r"pts_time:([\d.]+)", line)
            t = float(m[1]) if m else None
        elif line.startswith("lavfi.signalstats.YAVG=") and t is not None:
            pct = float(line.split("=", 1)[1]) / 255 * 100
            if pct < MOTION_MIN:
                continue
            if events and t - events[-1][1] <= MOTION_MERGE:
                events[-1][1] = t
                events[-1][2] = max(events[-1][2], pct)
            else:
                events.append([t, t, pct])
    end = duration or (events[-1][1] if events else 0)
    return [[round(max(0, s - MOTION_STEP - MOTION_PAD), 1), round(min(end, e + MOTION_PAD), 1), round(p, 2)]
            for s, e, p in events]


def by_stream_newest(entries):
    """Newest first, taking turns between streams, so a busy camera can't starve another."""
    queues = {}
    for e in sorted(entries, key=lambda e: e["start"], reverse=True):
        queues.setdefault(e["stream"], []).append(e)
    out = []
    while any(queues.values()):
        for q in queues.values():
            if q:
                out.append(q.pop(0))
    return out


def analyse_motion(entries, started):
    """Analyse recordings that have no motion data yet, newest first, within the budget."""
    done = 0
    for e in by_stream_newest(entries):
        if "motion" in e and not motion_stale(e["file"]):
            continue
        if time.time() - started > MOTION_BUDGET:
            break
        path = mp4_path(e["file"])
        try:
            events, err = detect_motion(path, e["duration"]), None
        except Exception as ex:  # noqa: BLE001
            events, err = [], str(ex)
            log(f"motion failed: {e['file']}: {err}")
        with open(motion_path(e["file"]), "w") as f:
            json.dump({"events": events, **({"error": err} if err else {})}, f)
        if not err:
            log(f"motion {e['file']}: {len(events)} event(s)")
        done += 1
    return done


def detect_path(mp4_name):
    return rec_file(mp4_name, "detect.json")


def keep_prev(mp4_name):
    """Before a recording's detect.json is replaced: keep it as detect.prev.json, so
    people.py can pair its sightings with the new ones (people keep their sightings, marks
    and names). Only the first one is kept until people.py has used it."""
    cur, prev = detect_path(mp4_name), rec_file(mp4_name, "detect.prev.json")
    if os.path.exists(cur) and not os.path.exists(prev):
        os.replace(cur, prev)


def read_detect(mp4_name, motion):
    """detect.py's results for the motion events ({"v", "spans", "found", "people"}),
    or None if not analysed (or the motion events changed since)."""
    try:
        with open(detect_path(mp4_name)) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    if d.get("spans") != [m[:2] for m in motion]:
        return None
    return d


def labels(found):
    """What the page shows for one event: {label: [score, time]} of convincing labels."""
    return {k: v[:2] for k, v in found.items()
            if v[0] >= LABEL_MIN or any(v[2] >= n and v[0] >= lo for n, lo in LABEL_SEEN)}


def analyse_objects(entries, started):
    """Look for people / pets in the motion events, newest recordings first, within the budget."""
    done = 0
    for e in by_stream_newest(entries):
        motion = e.get("motion")
        if not motion or e.get("detected"):
            continue
        if time.time() - started > DETECT_BUDGET:
            break
        spans = [m[:2] for m in motion]
        try:
            thumbs = rec_file(e["file"], "people")
            # one detect.py at a time across all cameras: two at once (models and frames,
            # ~200 MB each) don't fit in the server's 1 GB and it starts swapping
            with open(os.path.join(META, ".detect.lock"), "w") as detect_lock:
                fcntl.flock(detect_lock, fcntl.LOCK_EX)  # released when the file closes
                r = subprocess.run(["nice", "-n", "15", DETECT_PY, os.path.join(HERE, "detect.py"),
                                    "--thumbs", thumbs, mp4_path(e["file"])]
                                   + [f"{s}:{t}" for s, t in spans],
                                   capture_output=True, text=True, timeout=1800)
            res = json.loads(r.stdout or "{}")
            assert "found" in res and len(res["found"]) == len(spans), \
                f"exit {r.returncode}: {r.stderr.strip()[-300:]}" + (" (killed: out of memory?)" if r.returncode == -9 else "")
            found, people = res["found"], res["people"]
        except Exception as ex:  # noqa: BLE001
            # keep what an earlier version found (it is still shown) and try again later,
            # up to DETECT_TRIES times
            old = read_detect(e["file"], motion) or {"v": 0, "spans": spans, "found": [{} for _ in spans],
                                                     "people": [[] for _ in spans]}
            old["tries"] = old.get("tries", 0) + 1
            log(f"detect failed ({old['tries']}/{DETECT_TRIES}): {e['file']}: {ex}")
            if not old.get("v"):
                keep_prev(e["file"])  # the earlier people are replaced by none for now
            with open(detect_path(e["file"]), "w") as f:
                json.dump(old, f)
            continue
        keep_prev(e["file"])
        with open(detect_path(e["file"]), "w") as f:
            json.dump({"v": DETECT_V, "spans": spans, "found": found, "people": people}, f)
        seen = {}
        for f in found:
            for k in labels(f):
                seen[k] = seen.get(k, 0) + 1
        n = sum(len(p) for p in people)
        log(f"detect {e['file']}: " + (", ".join(f"{k} x{n}" for k, n in seen.items()) or "nothing")
            + (f"; {n} person crop(s)" if n else ""))
        done += 1
    return done


def group_people(stream):
    """people.py: group a camera's people seen into "people" for the page (its people.json).
    One at a time per camera: its analyser and people-regroup.service may both call this."""
    os.makedirs(stream_file(stream, ".meta"), exist_ok=True)
    with open(stream_file(stream, ".meta", ".people.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        r = subprocess.run(["nice", "-n", "15", DETECT_PY, os.path.join(HERE, "people.py"), stream],
                           capture_output=True, text=True, timeout=600)
    if r.returncode:
        log(f"people failed: {stream}: {r.stderr.strip()[-300:]}")


def marked(stream):
    """True if the page saved marks or names the camera's people.json doesn't include.
    people.py notes the time of the feedback file it read: a run that started just before a
    mark was saved finishes after it, so comparing the two files' times missed that mark."""
    try:
        fb = os.path.getmtime(feedback(stream))
    except OSError:
        return False
    try:
        with open(people_json(stream)) as f:
            read = json.load(f).get("feedback_mtime")
        if read is None:  # written before people.py noted it
            return fb > os.path.getmtime(people_json(stream))
    except (OSError, ValueError):
        return True
    return fb != read


def regroup_marked():
    """people-regroup.service: regroup every camera with new marks or names (again while
    marks saved during a run are newer than its result, at most 3 times)."""
    for _ in range(3):
        todo = [s for s in all_streams() if marked(s)]
        if not todo:
            return
        for s in todo:
            group_people(s)


def all_streams():
    """Cameras with a folder (finished recordings, or a recording in progress)."""
    return sorted(d for d in os.listdir(REC_DIR) if STREAM_OK.match(d) and os.path.isdir(stream_file(d)))


def lock_or_none(name, folder=META):
    """An exclusive lock file in folder (.meta), or None if another process holds it."""
    f = open(os.path.join(folder, name), "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return f
    except BlockingIOError:
        f.close()
        return None


def analyse(stream):
    """Motion, then person / pet detection, for one stream's recordings (slow). Runs as
    its own process per stream (`recordings.py --analyse STREAM`), started by main() every
    minute, so one camera's backlog never holds up another camera or the conversions.
    Results go to each recording's folder; the next main() run puts them in index.json."""
    os.makedirs(stream_file(stream, ".meta"), exist_ok=True)
    lock = lock_or_none(".analyse.lock", stream_file(stream, ".meta"))
    if not lock:
        return  # this stream's previous run is still busy

    def entries():
        out = []
        for name in all_mp4s():
            if parse_name(name[:-4])[0] != stream:
                continue
            if "stamps" not in (read_fix(name) or {}):
                continue  # main() hasn't checked / repaired it yet
            try:
                e = mp4_entry(mp4_path(name))
            except Exception:  # noqa: BLE001 - deleted meanwhile, or being repaired
                continue
            if e:
                out.append(e)
        return out

    if MOTION:
        analyse_motion(entries(), time.time())
    if MOTION and DETECT and analyse_objects(entries(), time.time()):
        group_people(stream)
    elif DETECT and not os.path.exists(people_json(stream)):
        group_people(stream)


def free_gb():
    return shutil.disk_usage(REC_DIR).free / 1e9


def enforce_space(now):
    """Delete the oldest finished recordings while free space is below MIN_FREE_GB."""
    if free_gb() >= MIN_FREE_GB:
        return
    victims = []
    for name in all_mp4s():  # finished: the whole folder goes
        try:
            st = os.stat(mp4_path(name))
        except FileNotFoundError:
            continue
        _, start = parse_name(name[:-4])
        victims.append((start or st.st_mtime, rec_dir(name), st.st_size))
    for name in os.listdir(REC_DIR):
        if name.endswith(".flv") and os.path.exists(os.path.join(META, name + ".failed")):
            path = os.path.join(REC_DIR, name)  # could not be converted; kept as FLV
            try:
                st = os.stat(path)
            except FileNotFoundError:
                continue
            _, start = parse_name(name[:-4])
            victims.append((start or st.st_mtime, path, st.st_size))
    for start, path, size in sorted(victims):
        if free_gb() >= MIN_FREE_GB:
            return
        if start > now - MIN_KEEP_HOURS * 3600:
            log(f"WARNING: only {free_gb():.1f} GB free, but everything left is from the last "
                f"{MIN_KEEP_HOURS:g} h; not deleting it. Check what else is using the disk.")
            return
        if os.path.isdir(path):
            shutil.rmtree(path)
            for up in (os.path.dirname(path), os.path.dirname(os.path.dirname(path))):
                try:
                    os.rmdir(up)  # the day's folder, then the camera's, once empty
                except OSError:
                    break
        else:
            os.remove(path)
        log(f"retention: deleted {rel(path)} ({size / 1e6:.0f} MB) "
            f"to keep {MIN_FREE_GB:g} GB free")
    if free_gb() < MIN_FREE_GB:
        log(f"WARNING: only {free_gb():.1f} GB free and no finished recordings left to delete")


def index_mp4s():
    for stream in all_streams():
        try:
            with open(people_json(stream)) as f:
                STATIC_EVENTS.update(json.load(f).get("static", {}))
        except (OSError, ValueError):
            pass
    entries = []
    for name in all_mp4s():
        try:
            fix_existing(mp4_path(name))
            e = mp4_entry(mp4_path(name))
        except Exception as ex:  # noqa: BLE001
            log(f"index failed: {name}: {ex}")
            continue
        if e:
            entries.append(e)
    return entries


def drop_orphans(now):
    """Recording folders whose MP4 is gone (deleted by hand, say), and empty day folders.
    Only ones untouched for an hour: a conversion creates the folder just before the MP4."""
    for day in glob.glob(os.path.join(REC_DIR, "*", "[0-9]*-[0-9]*-[0-9]*")):
        for d in glob.glob(os.path.join(day, "*")):
            if (os.path.isdir(d) and mp4_path(os.path.basename(d) + ".mp4").startswith(d + os.sep)
                    and not os.path.exists(mp4_path(os.path.basename(d) + ".mp4"))
                    and now - os.path.getmtime(d) > 3600):
                shutil.rmtree(d, ignore_errors=True)
                log(f"removed {rel(d)} (no recording in it)")
        try:
            os.rmdir(day)  # only if empty
        except OSError:
            pass


def write_json(path, data):
    with open(path + ".tmp", "w") as f:
        json.dump(data, f, separators=(",", ":"))
    os.replace(path + ".tmp", path)


def write_index(now, recs):
    """Each camera's index.json, and streams.json listing the cameras."""
    by_stream = {s: [] for s in all_streams()}
    for e in recs:
        if STREAM_OK.match(e.get("stream") or ""):
            by_stream.setdefault(e["stream"], []).append(e)
    top = os.stat(REC_DIR)
    for stream, rs in by_stream.items():
        os.makedirs(stream_file(stream, ".meta"), exist_ok=True)
        st = os.stat(stream_file(stream))
        if (st.st_uid, st.st_gid) != (top.st_uid, top.st_gid):
            # owned like the top folder: live_dvr.py (not root) writes .live and live-dvr.json here
            os.chown(stream_file(stream), top.st_uid, top.st_gid)
        write_json(stream_file(stream, "index.json"),
                   {"generated": int(now), "recordings": sorted(rs, key=lambda e: e["start"], reverse=True)})
    write_json(STREAMS, {"generated": int(now), "streams": sorted(by_stream)})


def main():
    os.umask(0o022)
    os.makedirs(META, exist_ok=True)

    lock = lock_or_none(".lock")
    if not lock:
        return  # previous run still busy

    now = time.time()
    live, leftovers = [], []

    # 1) convert finished FLVs, note in-progress ones
    for name in sorted(os.listdir(REC_DIR)):
        if name.startswith(".") or not name.endswith(".flv"):
            continue
        path = os.path.join(REC_DIR, name)
        st = os.stat(path)
        stream, start = parse_name(name[:-4])

        if now - st.st_mtime < SETTLE:
            start = start or int(st.st_ctime)
            live.append({"file": name, "stream": stream, "start": start,
                         "duration": round(now - start), "size": st.st_size,
                         "status": "recording"})
            continue
        if st.st_size < MIN_BYTES:
            continue
        if os.path.exists(os.path.join(META, name + ".converted")):
            continue  # already converted (KEEP_FLV=1)
        failed = os.path.join(META, name + ".failed")
        if os.path.exists(failed) or not convert(path):
            open(failed, "w").close()
            start = start or int(st.st_mtime)
            leftovers.append({"file": name, "stream": stream, "start": start,
                              "duration": max(0, round(st.st_mtime - start)),
                              "size": st.st_size, "status": "flv"})

    # 2) make room, then index MP4s
    enforce_space(now)
    entries = index_mp4s()

    # 3) drop folders left without a recording
    drop_orphans(now)

    write_index(now, live + leftovers + entries)

    # 4) motion + person / pet detection: slow, so one background process per stream
    #    (each exits at once if its previous run is still busy)
    if MOTION:
        for stream in sorted({e["stream"] for e in entries if STREAM_OK.match(e.get("stream") or "")}):
            subprocess.Popen([sys.executable, os.path.abspath(__file__), "--analyse", stream],
                             stdin=subprocess.DEVNULL, start_new_session=True)
    # "same person" marks from the page (people_api.py) that people-regroup.path missed
    if DETECT and any(marked(s) for s in all_streams()):
        subprocess.Popen([sys.executable, os.path.abspath(__file__), "--people"],
                         stdin=subprocess.DEVNULL, start_new_session=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--analyse" and STREAM_OK.match(sys.argv[2]):
        os.umask(0o022)
        analyse(sys.argv[2])
    elif sys.argv[1:] == ["--people"]:  # every camera with new marks or names
        os.umask(0o022)
        regroup_marked()
    elif len(sys.argv) == 3 and sys.argv[1] == "--people" and STREAM_OK.match(sys.argv[2]):
        os.umask(0o022)
        os.makedirs(stream_file(sys.argv[2], ".meta"), exist_ok=True)
        group_people(sys.argv[2])
    else:
        main()
