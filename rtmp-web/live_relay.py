#!/usr/bin/env python3
"""
Re-times live streams for the browser's live view.

Cameras get timestamps wrong in different ways: cam1 sometimes sends every frame with
timestamp 0 (nginx's own HLS then writes one endless segment and live never plays);
"ramesh" opens each connection with one stale frame sent 714 times and then jumps its
clock 4 minutes ahead. So nginx makes no HLS from the cameras' streams ("live" app);
instead, for every stream publishing there, this runs an ffmpeg that plays it from
nginx, re-times it per packet and publishes it to the "show" app, which makes the HLS
at /hls/. Each stream is looked at when it connects:
  * timestamps stuck (all the same): new ones are counted from the packets (video at the
    stream's frame rate, AAC audio at 1024 samples each), kept close to when each packet
    arrived so audio and video can't drift apart (see counted());
  * timestamps advancing: they are kept as they are. The connect junk's jump becomes one
    discontinuity in the HLS, which players step over. (Re-timing such a stream packet by
    packet doesn't work: ramesh's audio has thousands of repeated timestamps and some
    gaps, and audio and video drift minutes apart.)
Recordings still come from "live"; recordings.py repairs their timestamps separately.

Every stream has its own worker process (`live_relay.py STREAM`), so one camera's
checks, stalls and reconnects never hold up another's. The service runs the supervisor
(`live_relay.py` with no arguments), which only starts a worker for each stream that
publishes and restarts it if it dies; a worker exits once its stream has been gone for
WORKER_IDLE seconds.

Runs as a service (see live-relay.service):

  sudo cp /home/ubuntu/rtmp-web/live-relay.service /etc/systemd/system/
  sudo systemctl daemon-reload && sudo systemctl enable --now live-relay
  journalctl -u live-relay -f
"""
import re
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

RTMP = "rtmp://127.0.0.1:1935"
STAT = "http://127.0.0.1:3200/stat"
SCAN = 5              # seconds between looks at /stat
STALL_US = 15_000_000  # give up on an input that sends nothing for this long
META_WAIT = 10        # seconds to wait for a new stream's audio details in /stat
DEFAULT_SR = 8000     # both cameras send 8 kHz AAC
WORKER_IDLE = 120     # a worker exits after its stream has been gone this long
RETRY = 3             # seconds between relay attempts for a stream
ANCHOR_BEHIND = 0.5   # counted timestamps may fall this far behind arrival time (s) ...
ANCHOR_AHEAD = 2      # ... or run this far ahead (a burst of packets after a stall)
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # stream keys come from whoever publishes


def log(msg):
    print(msg, flush=True)


def publishing():
    """{stream: (fps, audio sample rate or None)} for streams publishing to the live app."""
    try:
        with urllib.request.urlopen(STAT, timeout=5) as r:
            root = ET.fromstring(r.read())
    except Exception:  # noqa: BLE001 - nginx restarting
        return {}
    out = {}
    for app in root.iter("application"):
        if app.findtext("name") != "live":
            continue
        for st in app.iter("stream"):
            if st.find("publishing") is None:
                continue
            fps = float(st.findtext("meta/video/frame_rate") or 0) or 15.0
            sr = int(st.findtext("meta/audio/sample_rate") or 0) or None
            out[st.findtext("name")] = (fps, sr)
    return out


# ffprobe normally reads a few seconds of timestamps to learn a stream; stuck ones never
# get there, so the check is told to stop after this much instead
PROBE = ["-probesize", "300000", "-analyzeduration", "1000000"]


def stuck(name):
    """True if the stream's first packets all carry the same timestamp. Audio and video
    both count: right after connecting, nginx may send a new player only audio for a while
    (cam1: ~50 s). Decided on what arrives within the time limit: over a slow uplink
    (ramesh's connect junk takes minutes to upload) that may be only a few packets."""
    cmd = ["ffprobe", "-v", "error", "-rw_timeout", "8000000", *PROBE, "-read_intervals", "%+#60",
           "-show_entries", "packet=pts_time", "-of", "csv=p=0", f"{RTMP}/live/{name}"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ""
        out = out.decode(errors="replace") if isinstance(out, bytes) else out
    t = []
    for x in out.split():
        try:
            t.append(float(x))
        except ValueError:
            pass
    return len(t) >= 10 and max(t) - min(t) < 0.01


def counted(step):
    """setts expression: each packet one step after the previous one (smooth playback),
    but never more than ANCHOR_BEHIND behind or ANCHOR_AHEAD ahead of when it arrived (TS:
    arrival time, from -use_wallclock_as_timestamps). Counting alone drifts: over a mobile
    uplink the camera loses audio packets now and then, and after a few hours the audio
    was 5 s behind the video; nginx then split the HLS every few seconds, mid-picture,
    and players stalled. Anchored, audio and video stay within about a second."""
    prev = "PREV_OUTPTS"  # NOPTS (very negative) for the first packet, so TS decides
    return (f"max({prev}+1\\,min(max({prev}+{step}\\,TS-{ANCHOR_BEHIND}/TB)\\,"
            f"TS+{ANCHOR_AHEAD}/TB))")


def relay(name, fps, sr):
    # Both cameras send timestamps nginx-rtmp's HLS muxer can't segment: cam1's are frozen
    # at zero, and ramesh's jump backwards (non-monotonic DTS) whenever the mobile uplink
    # hiccups or the camera reconnects. On a backward DTS nginx stops writing HLS segments
    # and never recovers, so /hls/<stream>.m3u8 goes 404 and the live view dies. Passing the
    # camera timestamps through with -c copy (the old non-"stuck" path) is exactly what left
    # ramesh broken. So we ALWAYS rebuild timestamps by counting at the frame/sample rate,
    # anchored to arrival time (see counted()): the output is strictly monotonic and A/V
    # stays within ~1 s no matter what the camera sends. stuck() is kept for reference only.
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "warning",
           "-rw_timeout", str(STALL_US), "-use_wallclock_as_timestamps", "1",
           "-i", f"{RTMP}/live/{name}", "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy"]
    # steps in the stream's time base (TB): a frame at the stream's frame rate (spread so
    # they add up exactly: 67, 67, 66 ms at 15 fps), 1024 samples of AAC. The audio's sample
    # rate comes from nginx's stats: over RTMP (FLV) ffmpeg reports 8 kHz AAC as 44.1 kHz.
    cmd += ["-bsf:v", "setts=ts=" + counted(f"floor((N+1)/({fps:g}*TB))-floor(N/({fps:g}*TB))"),
            "-bsf:a", "setts=ts=" + counted(f"1024/({sr or DEFAULT_SR}*TB)")]
    cmd += ["-f", "flv", f"{RTMP}/show/{name}"]
    log(f"[{name}] relaying ({fps:g} fps, audio {sr or 'none'}; rebuilding timestamps)")
    return subprocess.Popen(cmd)


def worker(name):
    """Relay one stream for as long as it keeps publishing (with reconnects)."""
    gone_since = time.time()
    while True:
        info = publishing().get(name)
        if not info:
            if time.time() - gone_since > WORKER_IDLE:
                log(f"[{name}] stream gone for {WORKER_IDLE}s, worker exiting")
                return
            time.sleep(SCAN)
            continue
        fps, sr = info
        waited = 0
        while not sr and waited < META_WAIT:  # right after connecting, /stat may lack the audio details
            time.sleep(2)
            waited += 2
            fps, sr = publishing().get(name, (fps, sr))
        p = relay(name, fps, sr)
        p.wait()
        log(f"[{name}] relay ended (exit {p.returncode})")
        gone_since = time.time()
        time.sleep(RETRY)


def supervise():
    """Start (and restart) one worker process per publishing stream."""
    workers = {}
    while True:
        for name in publishing():
            if not NAME_RE.match(name):
                continue
            w = workers.get(name)
            if w is None or w.poll() is not None:
                workers[name] = subprocess.Popen([sys.executable, __file__, name])
                log(f"started worker for {name} (pid {workers[name].pid})")
        time.sleep(SCAN)


def main():
    if len(sys.argv) > 1:
        if NAME_RE.match(sys.argv[1]):
            worker(sys.argv[1])
    else:
        supervise()


if __name__ == "__main__":
    main()
