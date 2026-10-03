#!/usr/bin/env python3
"""
Makes the recording that is still being written rewindable before it finishes.

nginx-rtmp writes each live stream to an .flv in the recordings folder. For every
.flv that is still growing, this follows the file (like `tail -f`) and feeds it,
at a capped rate while catching up, to an ffmpeg that remuxes it (no re-encode) into
an HLS playlist under <stream>/.live/<recording>/, so the page can play and seek the whole
recording so far, 10-20 s behind live. The playlist locations go to <stream>/live-dvr.json.
A recording whose timestamps are stuck at zero (cam1 sends those) gets them counted from
the packets, as recordings.py does when it converts it to MP4.
The low-latency live view stays the /hls/ stream (live_relay.py); motion is found by
recordings.py once the recording is finished.

When the .flv stops growing the feed ends; once recordings.py has converted the
recording to MP4 (and deleted the .flv) the HLS copy is removed.

Every stream has its own worker process (`live_dvr.py STREAM`) writing its own
<stream>/live-dvr.json, so cameras never share a process or data. The service runs the
supervisor (`live_dvr.py` with no arguments), which starts a worker for each stream with
a recording in progress and restarts it if it dies. A worker exits once its stream has
had no recording in progress for WORKER_IDLE seconds.

Runs as a service (see live-dvr.service):

  sudo cp /home/ubuntu/rtmp-web/live-dvr.service /etc/systemd/system/
  sudo systemctl daemon-reload && sudo systemctl enable --now live-dvr
  journalctl -u live-dvr -f
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

import live_relay  # nginx's stream stats (frame rate, audio sample rate)
import recordings as R  # REC_DIR, file names and connect-junk detection

WORKER_IDLE = 120     # a worker exits after its stream has had no growing recording this long
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
GROWING = 20          # an .flv written to within this many seconds is still recording
IDLE_END = 25         # stop following once the file hasn't grown for this long
SEG = 6               # seconds per HLS segment (the camera sends a keyframe every 2 s)
# Catch-up speed for what was recorded before we started. The server has 1 GB of RAM
# and slow disks; reading a long recording flat out makes it unresponsive. (ffmpeg's
# -readrate doesn't help: the camera's timestamp jump at the start confuses it.)
FEED_BPS = 400 * 1024
THUMB_EVERY = 60      # seconds between thumbnails of a recording in progress (its latest segment)
# packets looked at to tell stuck timestamps: past a connect burst of identical frames
# (ramesh: ~714), so that isn't taken for them; about 2 minutes of a 15 fps stream
STUCK_PACKETS = R.JUNK_WINDOW

dirty = threading.Event()


def log(msg):
    print(msg, flush=True)


class Follower(threading.Thread):
    """Follows one growing .flv into its HLS copy."""

    def __init__(self, flv):
        super().__init__(daemon=True)
        self.flv = flv
        self.stem = os.path.basename(flv)[:-4]
        self.stream, named = R.parse_name(self.stem)
        self.named = named if named is not None else int(os.stat(flv).st_ctime)
        self.cut = None           # leading junk to skip (see recordings.leading_junk)
        self.counted = None       # (fps, audio sample rate) when timestamps are stuck
        self.proc = None
        self.ready = False        # playlist has at least one segment
        self.thumb_at = 0         # when thumb.jpg was last made

    @property
    def dir(self):
        return R.stream_file(self.stream, ".live", self.stem)

    def info(self):
        # currentTime 0 of the playlist is the first real frame, which recordings.py
        # also places at the name time for the recordings it trims
        out = {"path": f"{self.stream}/.live/{self.stem}/index.m3u8", "start": self.named}
        if self.thumb_at:  # the page adds thumbAt to the URL, so a new one is fetched
            out.update(thumb=f"{self.stream}/.live/{self.stem}/thumb.jpg", thumbAt=self.thumb_at)
        return out

    def thumbnail(self):
        """thumb.jpg from the first frame (a keyframe) of the newest segment, like the
        finished recordings' thumbnails (640 px wide). True if a new one was made."""
        if not self.ready or time.time() - self.thumb_at < THUMB_EVERY:
            return False
        try:
            with open(os.path.join(self.dir, "index.m3u8")) as f:
                segs = [x.strip() for x in f if x.strip().endswith(".ts")]
        except OSError:
            return False
        if not segs:
            return False
        out = os.path.join(self.dir, "thumb.jpg")
        tmp = os.path.join(self.dir, "thumb.tmp.jpg")
        r = R.run(["nice", "-n", "10", "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                   "-i", os.path.join(self.dir, segs[-1]), "-frames:v", "1",
                   "-vf", "scale=640:-2", "-q:v", "4", tmp])
        self.thumb_at = int(time.time())  # also after a failure: try again in a minute
        if r.returncode or not os.path.exists(tmp):
            return False
        os.replace(tmp, out)
        return True

    def stuck(self):
        """True if the file's first packets (audio and video) all carry the same timestamp,
        None if there aren't enough of them yet."""
        r = R.run(["ffprobe", "-v", "error", "-read_intervals", f"%+#{STUCK_PACKETS}",
                   "-show_entries", "packet=pts_time", "-of", "csv=p=0", self.flv])
        lines = r.stdout.split()
        if len(lines) < STUCK_PACKETS:
            return None
        t = []
        for x in lines:
            try:
                t.append(float(x.split(",")[0]))  # a packet with side data prints "0.001000,"
            except ValueError:
                pass
        return bool(t) and max(t) - min(t) < 0.01

    def find_cut(self):
        """The junk cut, once enough of the file exists to know it (None = not yet)."""
        stuck = self.stuck()
        if stuck is None:
            return None
        if stuck:  # nothing to cut by time; count timestamps instead
            fps, sr = live_relay.publishing().get(self.stream) or (15.0, None)
            self.counted = (fps, sr or live_relay.DEFAULT_SR)
            return 0.0
        cut = R.leading_junk(self.flv)
        if cut:
            return cut
        r = R.run(["ffprobe", "-v", "error", "-read_intervals", f"%+{R.JUNK_SPAN + R.JUNK_GAP + 1}",
                   "-select_streams", "v", "-show_entries", "packet=pts_time",
                   "-of", "csv=p=0", self.flv])
        times = [float(x) for x in r.stdout.split() if re.fullmatch(r"[\d.]+", x)]
        # clean only once we've seen past where a junk burst + jump could be
        return 0.0 if times and times[-1] - times[0] > R.JUNK_SPAN + R.JUNK_GAP else None

    def run(self):
        while self.cut is None:
            if not os.path.exists(self.flv):
                return
            self.cut = self.find_cut()
            if self.cut is None:
                time.sleep(3)
        shutil.rmtree(self.dir, ignore_errors=True)
        os.makedirs(self.dir, exist_ok=True)
        start = max(0.0, self.cut - 0.0005)
        cmd = ["nice", "-n", "5", "ffmpeg", "-hide_banner", "-nostats",
               "-loglevel", "info", "-threads", "1", "-f", "flv", "-i", "pipe:0",
               "-map", "0:v", "-map", "0:a?", "-c", "copy"]
        if self.counted:
            # packet number / rate, in the stream's time base (TB): video at the frame
            # rate, AAC audio at 1024 samples per packet
            fps, sr = self.counted
            cmd += ["-bsf:v", f"setts=ts=N/({fps:g}*TB)", "-bsf:a", f"setts=ts=N*1024/({sr}*TB)"]
        else:
            cmd += ["-ss", f"{start:.4f}"]
        cmd += ["-avoid_negative_ts", "make_zero", "-f", "hls",
               "-hls_time", str(SEG), "-hls_list_size", "0", "-hls_playlist_type", "event",
               "-hls_flags", "independent_segments+temp_file",
               "-hls_segment_filename", os.path.join(self.dir, "seg-%d.ts"),
               os.path.join(self.dir, "index.m3u8")]
        log(f"[{self.stream}] following {os.path.basename(self.flv)} "
            + (f"(timestamps stuck: counting at {self.counted[0]:g} fps, audio {self.counted[1]} Hz)"
               if self.counted else f"(skipping {self.cut:.1f}s of connect junk)"))
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE)
        threading.Thread(target=self.feed, daemon=True).start()
        for raw in self.proc.stderr:
            if not self.ready and b"Opening" in raw and b".m3u8" in raw:
                self.ready = True
                dirty.set()
        self.proc.wait()
        log(f"[{self.stream}] stopped following {os.path.basename(self.flv)} (ffmpeg exit {self.proc.returncode})")

    def feed(self):
        """Copy the .flv into ffmpeg as it grows (capped at FEED_BPS), until it stops growing."""
        out, idle_since = self.proc.stdin, time.time()
        try:
            with open(self.flv, "rb") as f:
                while self.proc.poll() is None:
                    chunk = f.read(64 * 1024)
                    if chunk:
                        out.write(chunk)
                        out.flush()
                        idle_since = time.time()
                        time.sleep(len(chunk) / FEED_BPS)
                    elif time.time() - idle_since > IDLE_END:
                        break
                    else:
                        time.sleep(0.5)
        except (OSError, ValueError):  # ffmpeg exited, or the file went away
            pass
        try:
            out.close()
        except OSError:
            pass

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def growing_flvs(stream=None):
    """Recordings still being written (of one stream, or all)."""
    now, out = time.time(), []
    for name in os.listdir(R.REC_DIR):
        if name.startswith(".") or not name.endswith(".flv"):
            continue
        if stream is not None and R.parse_name(name[:-4])[0] != stream:
            continue
        p = os.path.join(R.REC_DIR, name)
        try:
            st = os.stat(p)
        except FileNotFoundError:
            continue
        if now - st.st_mtime < GROWING and st.st_size >= R.MIN_BYTES:
            out.append(p)
    return out


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    os.replace(tmp, path)


def save(stream, followers):
    dvr = {os.path.basename(f.flv): f.info()  # recording file -> its HLS copy
           for f in followers.values() if f.ready and os.path.exists(f.flv)}
    write_json(R.stream_file(stream, "live-dvr.json"), {"generated": int(time.time()), "dvr": dvr})


def worker(stream):
    """Follow one stream's recordings in progress into their HLS copies."""
    os.makedirs(R.stream_file(stream, ".live"), exist_ok=True)
    for d in glob.glob(R.stream_file(stream, ".live", "*")):  # this stream's copies: rebuilt
        shutil.rmtree(d, ignore_errors=True)
    followers, last_save, last_scan, busy_at = {}, 0, 0, time.time()
    while True:
        if time.time() - last_scan >= 5:
            last_scan = time.time()
            growing = growing_flvs(stream)
            if growing or followers:
                busy_at = time.time()
            elif time.time() - busy_at > WORKER_IDLE:
                save(stream, followers)
                log(f"[{stream}] no recording in progress for {WORKER_IDLE}s, worker exiting")
                return
            for flv in growing:
                f = followers.get(flv)
                if f is None or not f.is_alive():  # new recording, or ffmpeg gave up too early
                    followers[flv] = Follower(flv)
                    followers[flv].start()
            for flv, f in list(followers.items()):
                if f.is_alive() and f.thumbnail():
                    dirty.set()
                if not os.path.exists(flv):  # converted to MP4 by recordings.py
                    f.stop()
                    shutil.rmtree(f.dir, ignore_errors=True)
                    del followers[flv]
                    dirty.set()
        if dirty.is_set() or time.time() - last_save > 30:
            dirty.clear()
            save(stream, followers)
            last_save = time.time()
        time.sleep(1)


def supervise():
    """Start (and restart) one worker per stream with a recording in progress."""
    workers = {}
    while True:
        for flv in growing_flvs():
            stream = R.parse_name(os.path.basename(flv)[:-4])[0]
            if not NAME_RE.match(stream or ""):
                continue
            w = workers.get(stream)
            if w is None or w.poll() is not None:
                workers[stream] = subprocess.Popen([sys.executable, os.path.abspath(__file__), stream])
                log(f"started worker for {stream} (pid {workers[stream].pid})")
        time.sleep(3)


def main():
    if len(sys.argv) > 1:
        if NAME_RE.match(sys.argv[1]):
            worker(sys.argv[1])
    else:
        supervise()


if __name__ == "__main__":
    main()
