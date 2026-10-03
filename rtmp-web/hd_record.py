#!/usr/bin/env python3
"""
Records one camera's HD stream from the TrueCloud cloud, apart from the RTMP pipeline.

The camera sends only its 800x448 stream over RTMP; its 2304x1296 H.265 main stream is
reachable through the vendor's cloud relay (what their app and web portal play).
hd_source.js fetches it as raw H.265; this keeps it running and has ffmpeg cut it,
without re-encoding, into SEGMENT-second MP4 files next to the camera's recordings
(REC_TZ day and time, like theirs; the epoch is the clip's start, which the UI reads):
  <REC_DIR>/<stream>/hd/<YYYY-MM-DD>/<stream>-hd-<epoch>_<YYYY-MM-DD>_<HH-MM-SS>.mp4
and, from the same ffmpeg, a live HLS stream (fMP4 pieces, one per camera keyframe, i.e.
~10 s) whose playlist keeps the last LIVE_KEEP seconds, so it also covers the clip still
being written:
  <REC_DIR>/<stream>/hd/live/live.m3u8   (+ init.mp4, seg-<n>.m4s; PROGRAM-DATE-TIME tagged)
The player shows only these for HD cameras (index.html). nginx-rtmp can't carry H.265, so
none of this goes through it. When the link drops, hd_source.js exits and both are started again
after RETRY seconds; days older than KEEP_DAYS are deleted.

  python3 hd_record.py DEVICE.json [stream=cam1]        (hd-record.service)
Needs node with ws and xmlhttprequest-ssl (npm install in this folder).
"""
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
REC_DIR = os.environ.get("REC_DIR", "/home/ubuntu/rtmp-recordings")
TZ_NAME = os.environ.get("REC_TZ", "Asia/Kolkata")
TZ = ZoneInfo(TZ_NAME)
STREAM_ID = os.environ.get("HD_STREAM_ID", "2")   # 1 and 2 = HD main stream (2: see the 2026-10-02 lag notes)
SEGMENT = int(os.environ.get("HD_SEGMENT", "300"))
KEEP_DAYS = int(os.environ.get("HD_KEEP_DAYS", "3"))
LIVE_KEEP = 420   # seconds of live playlist: more than a clip, so recent minutes are always there
RETRY = 2
src = None      # the running hd_source.js
stopping = False


def log(msg):
    print(datetime.now(TZ).strftime("%H:%M:%S"), msg, flush=True)


def prune(folder):
    """Delete day folders older than KEEP_DAYS."""
    oldest = (datetime.now(TZ) - timedelta(days=KEEP_DAYS)).strftime("%Y-%m-%d")
    for day in sorted(os.listdir(folder)):
        if len(day) == 10 and day < oldest:
            shutil.rmtree(os.path.join(folder, day), ignore_errors=True)
            log(f"deleted {day}")


def run_once(device, stream):
    folder = os.path.join(REC_DIR, stream, "hd")
    for d in (0, 1):  # today's and tomorrow's folders (ffmpeg won't make them; a run spans midnight)
        os.makedirs(os.path.join(folder, (datetime.now(TZ) + timedelta(days=d)).strftime("%Y-%m-%d")), exist_ok=True)
    prune(folder)
    global src
    # it also relays the frames live to the browsers, over run/hd-live-<stream>.sock (nginx:
    # /api/hd-live/<stream>)
    # picture and sound come on its stdout as one MPEG-TS stream, timestamps set (two ffmpeg
    # inputs held the picture back by minutes to keep them in step, see hd_source.js)
    src = subprocess.Popen(["node", os.path.join(HERE, "hd_source.js"), device, STREAM_ID],
                           stdout=subprocess.PIPE,
                           env={**os.environ, "HD_LIVE_SOCK": os.path.join(HERE, "run", f"hd-live-{stream}.sock")})
    live = os.path.join(folder, "live")
    os.makedirs(live, exist_ok=True)  # kept across runs: the cloud ends a session every ~10 min,
                                      # and a new playlist each time broke rewinding (404s)
    clips = (f"[f=segment:segment_time={SEGMENT}:segment_atclocktime=1:reset_timestamps=1:"
             f"strftime=1:segment_format_options=movflags=+faststart]"
             + os.path.join(folder, "%Y-%m-%d", f"{stream}-hd-%s_%Y-%m-%d_%H-%M-%S.mp4"))
    # segment numbers start at the epoch: never the same name twice (no stale browser caches)
    hls = (f"[f=hls:hls_time=2:hls_list_size={LIVE_KEEP // 10}:hls_segment_type=fmp4:"
           f"hls_start_number_source=epoch:hls_fmp4_init_filename=init.mp4:"
           # a new run carries on the same playlist (a discontinuity where it reconnected)
           f"hls_flags=delete_segments+program_date_time+independent_segments+temp_file+append_list+discont_start:"
           f"hls_segment_filename=" + os.path.join(live, "seg-%d.m4s").replace(":", "\\:") + "]"
           + os.path.join(live, "live.m3u8"))
    rec = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error",
         "-f", "mpegts", "-i", "-",
         "-map", "0:v", "-map", "0:a?", "-c", "copy", "-tag:v", "hvc1", "-bsf:a", "aac_adtstoasc",
         "-f", "tee", clips + "|" + hls],
        stdin=src.stdout, env={**os.environ, "TZ": TZ_NAME})
    src.stdout.close()  # ffmpeg owns the pipe now
    src.wait()
    rec.wait()
    log(f"stopped (source {src.returncode}, ffmpeg {rec.returncode})")


def stop(*_):
    """Service stop: end the source only, so ffmpeg finishes the clip it's writing (an MP4
    cut off without its index can't be played) and exits on its own."""
    global stopping
    stopping = True
    if src and src.poll() is None:
        src.terminate()


def main():
    signal.signal(signal.SIGTERM, stop)
    device = sys.argv[1]
    stream = sys.argv[2] if len(sys.argv) > 2 else "cam1"
    while not stopping:
        run_once(device, stream)
        if not stopping:
            time.sleep(RETRY)


if __name__ == "__main__":
    main()
