#!/usr/bin/env python3
"""Is the car parked outside the gate? Once a minute, one frame of the camera's live stream:
  - night (infrared, no colour): can't tell;
  - a view model (OSNet-AIN features of the whole frame): the camera is moved now and then, or
    turned away from the gate entirely; not looking at the gate: can't tell;
  - YOLOX-Tiny (detect.py's model) looks for a car in three enlarged tiles of the frame's top;
    a clear car (SURE_MIN) is there; outside DAYLIGHT nothing else counts (can't tell);
  - else a car model on OSNet-AIN features (detect.py's re-id model) of the gate area and the
    whole frame, trained on frames from every framing (experiments/car/train2.py,
    models/car_models2.npz).

Two bits per minute, in one small file per day (REC_TZ days, like the recordings):
  <REC_DIR>/<stream>/car/<YYYY-MM-DD>.bin   360 bytes
    bytes   0..179  "car":  bit m set = the car was there in minute m of the day
    bytes 180..359  "seen": bit m set = it could be checked in minute m (a frame was read
                    and it wasn't night)
  bit m is (byte m // 8) >> (m % 8) & 1. Minutes not checked are 0 in both.

  python3 car_presence.py              run forever (car-presence.service)
  python3 car_presence.py --once FILE  check one 800x448 image, print the verdict
"""
import io, os, subprocess, sys, time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REC_DIR = os.environ.get("REC_DIR", "/home/ubuntu/rtmp-recordings")
TZ = ZoneInfo(os.environ.get("REC_TZ", "Asia/Kolkata"))
STREAM = os.environ.get("CAR_STREAM", "cam1")
LIVE = f"http://127.0.0.1:3200/hls/{STREAM}.m3u8"
YOLOX = os.path.join(HERE, "models", "yolox_tiny.onnx")
REID = os.path.join(HERE, "models", "osnet_ain_x1_0_msmt17.onnx")
MODELS = os.path.join(HERE, "models", "car_models2.npz")  # experiments/car/train2.py (3 Oct)
W, H = 800, 448
REGION = (0, 0, 600, 340)       # the gate and what's behind it, in most of the camera's views
TILES = [(x, 0, x + 320, 260) for x in (0, 240, 480)]  # where YOLOX looks, each enlarged
CAR_IDS = (2, 7)                # COCO car, truck (the hatchback is sometimes taken for a truck)
SURE_MIN = 0.35                 # a car this clear is there, whatever the gate
NIGHT_SAT = 0.08                # mean colour saturation below this: infrared, can't tell
DAYLIGHT = (6 * 60 + 30, 18 * 60)  # minutes of the day (REC_TZ) the models are trusted in: their
                                   # "no car" examples are few, and lamp light fooled them; outside
                                   # it only a clear car (SURE_MIN) counts, the rest is can't tell
SIZE = 416
BITS = 1440                     # minutes in a day
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def grids():
    xy, st = [], []
    for s in (8, 16, 32):
        n = SIZE // s
        yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        xy.append(np.stack((xv, yv), -1).reshape(-1, 2))
        st.append(np.full((n * n, 1), s))
    return np.concatenate(xy).astype(np.float32), np.concatenate(st).astype(np.float32)


def session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


class Checker:
    def __init__(self):
        self.yolox, self.reid = session(YOLOX), session(REID)
        self.m = dict(np.load(MODELS))

    def car_score(self, im):
        """Best car score over the tiles (PIL image, 800x448)."""
        best = 0.0
        for t in TILES:
            c = im.crop(t).resize((SIZE, round(SIZE * (t[3] - t[1]) / (t[2] - t[0]))))
            pad = np.full((SIZE, SIZE, 3), 114, np.uint8)
            pad[:c.height, :c.width] = np.asarray(c)[:, :, ::-1]
            out = self.yolox.run(None, {self.yolox.get_inputs()[0].name: pad.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
            cls = out[:, 5:].argmax(1)
            score = out[:, 4] * out[:, 5:].max(1)
            best = max(best, float(score[np.isin(cls, CAR_IDS)].max(initial=0)))
        return best

    def features(self, im, region=REGION):
        r = im.crop(region)
        w = r.width // 2
        halves = [r.crop((0, 0, w, r.height)), r.crop((w, 0, r.width, r.height))]
        x = np.stack([(np.asarray(h.resize((128, 256)), np.float32) / 255 - MEAN) / STD for h in halves]).transpose(0, 3, 1, 2)
        e = self.reid.run(None, {self.reid.get_inputs()[0].name: x})[0]
        return (e / np.linalg.norm(e, axis=1, keepdims=True)).reshape(-1)

    def prob(self, name, f):
        m = self.m
        z = ((f - m[name + "_mu"]) / m[name + "_sd"]) @ m[name + "_w"] + float(np.ravel(m[name + "_b"])[0])
        return float(1 / (1 + np.exp(-z)))

    def verdict(self, im, minute=None):
        """(car, seen, why) for one frame (PIL image) or None; minute: of the day, for DAYLIGHT."""
        if im is None:
            return False, False, "no frame"
        im = im.convert("RGB").resize((W, H))
        sat = float((np.asarray(im.convert("HSV"), np.float32)[..., 1] / 255).mean())
        if sat < NIGHT_SAT:
            return False, False, "night"
        # the camera is moved now and then (or turned away from the gate): not looking at the
        # gate, it can't tell
        full = self.features(im, (0, 0, W, H))
        v = self.prob("view", full)
        if v < 0.5:
            return False, False, f"gate not in view ({v:.2f})"
        s = self.car_score(im)
        if s >= SURE_MIN:
            return True, True, f"car score {s:.2f}"
        if minute is not None and not DAYLIGHT[0] <= minute < DAYLIGHT[1]:
            return False, False, f"not daylight, car score {s:.2f}"
        f = np.concatenate([self.features(im), full])  # the gate area and the whole frame
        p = self.prob("car", f)
        return p >= 0.5, True, f"car model {p:.2f}, car score {s:.2f}"


def grab():
    """One live frame as a PIL image, or None."""
    try:
        out = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", LIVE, "-frames:v", "1",
                              "-f", "image2", "-c:v", "png", "-"], capture_output=True, timeout=40).stdout
        return Image.open(io.BytesIO(out)) if out else None
    except (subprocess.TimeoutExpired, OSError):
        return None


def store(when, car, seen):
    """Set minute `when`'s two bits in its day's file."""
    t = datetime.fromtimestamp(when, TZ)
    m = t.hour * 60 + t.minute
    folder = os.path.join(REC_DIR, STREAM, "car")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{t:%Y-%m-%d}.bin")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(bytes(2 * BITS // 8))
    with open(path, "r+b") as f:
        for base, on in ((0, car), (BITS // 8, seen)):
            f.seek(base + m // 8)
            b = f.read(1)[0]
            b = b | (1 << (m % 8)) if on else b & ~(1 << (m % 8))
            f.seek(base + m // 8)
            f.write(bytes([b]))


def main():
    checker = Checker()
    if sys.argv[1:2] == ["--once"]:
        for p in sys.argv[2:]:
            print(p, checker.verdict(Image.open(p)))
        return
    while True:
        time.sleep(60 - time.time() % 60 + 5)  # 5 s into each minute
        when = time.time()
        try:
            t = datetime.fromtimestamp(when, TZ)
            car, seen, why = checker.verdict(grab(), t.hour * 60 + t.minute)
        except Exception as e:  # a bad frame mustn't stop the service
            car, seen, why = False, False, f"error {e}"
        store(when, car, seen)
        if os.environ.get("CAR_DEBUG"):
            print(datetime.fromtimestamp(when, TZ).strftime("%H:%M"), car, seen, why, flush=True)


if __name__ == "__main__":
    main()
