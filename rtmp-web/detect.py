#!/usr/bin/env python3
"""
Person / pet detection for motion events (YOLOX-Tiny, COCO classes, CPU only), plus an
appearance fingerprint of every person seen (OSNet-AIN, person re-identification), so
people.py can group sightings of the same person.

Runs in the venv next to it (numpy + onnxruntime); recordings.py calls it as a
subprocess, so it keeps working on the system python:

  .venv/bin/python detect.py [--thumbs DIR] FILE START:END [START:END ...]

Looks only at keyframes (complete pictures; these cameras send one every 2 s), which
ffmpeg can decode without the frames between them, in one pass over FILE: at most
KEY_BUDGET of them per call, shared between the events (seconds into FILE) so every
event gets at least one and long ones more. That keeps a 30-minute recording with motion
all the way through to a few minutes of CPU. Prints one JSON object with a result per
event:

  {"found":  [{"person": [0.87, 12.0, 3]}, {}, ...],   label -> [best score, time, frames seen]
   "people": [[{"t": 12.0, "box": [x0, y0, x1, y1], "score": 0.9, "emb": "<base64>",
                "thumb": "0-0.jpg"}], [], ...]}  one per person in the event

"emb" is the person's 512-d appearance vector (float16, L2-normalised). With --thumbs, a
crop of each person is saved in DIR as <event>-<person>.jpg ("thumb": its name).

Setup (once):
  python3 -m venv .venv && .venv/bin/pip install numpy onnxruntime
  curl -L -o models/yolox_tiny.onnx \
    https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx
  models/osnet_ain_x1_0_msmt17.onnx: the author's MSMT17 weights (huggingface.co/kaiyangzhou/osnet)
    exported with torch.onnx (input "images" Nx3x256x128 RGB, ImageNet-normalised)
"""
import base64
import json
import os
import subprocess
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

from people import cluster

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "yolox_tiny.onnx")
REID = os.path.join(HERE, "models", "osnet_ain_x1_0_msmt17.onnx")
SIZE = 416
LABELS = {0: "person", 14: "bird", 15: "cat", 16: "dog"}  # COCO class ids we care about
MIN_SCORE = 0.30      # a frame counts as seeing the label at this score
KEY_BUDGET = 240      # keyframes looked at per recording (of ~900 in 30 minutes)
NEAR_KEY = 2.5        # an event with no keyframe inside uses the nearest one this close (s)

PERSON_BOX = 0.45     # person boxes this confident get an appearance fingerprint
MIN_HEIGHT = 40       # px in the recording; smaller people are too blurry to tell apart
MAX_CROPS = 48        # fingerprints per event, spread over its frames ...
MIN_CROPS = 8         # ... at least this many (when there are) ...
CROP_BUDGET = 400     # ... and about this many per recording, shared like the frames
SAME_IN_EVENT = 0.70  # average cosine similarity at which crops in one event are one person
                      # (a busy shop has several people per event; unrelated people average ~0.58)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # ImageNet normalisation (RGB)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def probe_size(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height", "-of", "csv=p=0", path], capture_output=True, text=True)
    w, h = map(int, r.stdout.split(",")[:2])
    return w, h


def keyframes(path):
    """Times (s) of the video's keyframes, from the packet flags (no decoding)."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "packet=pts_time,flags", "-of", "csv=p=0", path], capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        t, _, flags = line.partition(",")
        if "K" in flags:
            try:
                out.append(float(t))
            except ValueError:
                pass
    return out


def plan(keys, spans):
    """Which keyframes to look at: {keyframe index: [events]}, and frames per event."""
    per_event = []
    for s, e in spans:
        inside = [i for i, t in enumerate(keys) if s <= t <= e]
        if not inside and keys:  # a short event between two keyframes
            i = min(range(len(keys)), key=lambda i: abs(keys[i] - (s + e) / 2))
            inside = [i] if abs(keys[i] - (s + e) / 2) <= NEAR_KEY else []
        per_event.append(inside)
    total = sum(len(x) for x in per_event)
    picks = {}
    for ev, inside in enumerate(per_event):
        n = len(inside)
        if total > KEY_BUDGET and n > 1:  # spread over the event
            q = max(1, round(n * KEY_BUDGET / total))
            inside = [inside[i] for i in sorted(set(np.linspace(0, n - 1, q).round().astype(int)))]
        per_event[ev] = inside
        for i in inside:
            picks.setdefault(i, []).append(ev)
    return picks, [len(x) for x in per_event]


def frames(path, keys, picks, w, h):
    """(keyframe index, detector input 1x3x416x416, full frame HxWx3 BGR) for the picked
    keyframes, in order. One ffmpeg decodes only keyframes and keeps the picked ones."""
    W = max(w, SIZE)
    wanted = sorted(picks)
    pick = "+".join(f"eq(n,{i})" for i in wanted)  # n counts keyframes (the only frames decoded)
    # one raw frame = the detector's letterboxed 416x416 (keep aspect, top-left, grey 114,
    # like YOLOX's own preprocessing) stacked above the full-resolution frame for the crops
    # (formats pinned: left to itself ffmpeg may pick grey for the stack and drop the colour)
    vf = (f"select='{pick}',format=yuv420p,split[a][b];"
          f"[a]scale={SIZE}:{SIZE}:force_original_aspect_ratio=decrease,"
          f"pad={W}:{SIZE}:0:0:color=0x727272,format=yuv420p[s];"
          f"[b]pad={W}:{h}:0:0,format=yuv420p[f];[s][f]vstack,format=bgr24")
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-skip_frame", "nokey",
           "-i", path, "-an", "-vf", vf, "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = W * (SIZE + h) * 3
    for i in wanted:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        img = np.frombuffer(buf, np.uint8).reshape(SIZE + h, W, 3)
        yield i, img[:SIZE, :SIZE].transpose(2, 0, 1)[None].astype(np.float32), img[SIZE:, :w]
    proc.stdout.close()
    proc.wait()


def grids():
    """Cell offsets and strides for decoding YOLOX's raw 416x416 output."""
    xy, st = [], []
    for s in (8, 16, 32):
        n = SIZE // s
        yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        xy.append(np.stack((xv, yv), -1).reshape(-1, 2))
        st.append(np.full((n * n, 1), s))
    return np.concatenate(xy).astype(np.float32), np.concatenate(st).astype(np.float32)


GRID, STRIDE = grids()


def person_boxes(out, cls, score, scale):
    """Person boxes [x0, y0, x1, y1, score] in full-frame pixels, after NMS."""
    keep = (cls == 0) & (score >= PERSON_BOX)
    if not keep.any():
        return []
    p, g, st = out[keep], GRID[keep], STRIDE[keep]
    c = (p[:, :2] + g) * st
    wh = np.exp(p[:, 2:4]) * st
    boxes = np.concatenate([c - wh / 2, c + wh / 2], 1) / scale
    s = score[keep]
    order, picked = s.argsort()[::-1], []
    while order.size:
        i = order[0]
        picked.append(i)
        xx0 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy0 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx1 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy1 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
        iou = inter / (area(boxes[i]) + area(boxes[order[1:]]) - inter + 1e-6)
        order = order[1:][iou < 0.45]
    return [[*boxes[i], float(s[i])] for i in picked]


def resize(img, h, w):
    """Bilinear resize of an HxWxC uint8 image (no OpenCV on the server)."""
    H, W = img.shape[:2]
    ys = np.clip((np.arange(h) + .5) * H / h - .5, 0, H - 1)
    xs = np.clip((np.arange(w) + .5) * W / w - .5, 0, W - 1)
    y0, x0 = ys.astype(int), xs.astype(int)
    y1, x1 = np.minimum(y0 + 1, H - 1), np.minimum(x0 + 1, W - 1)
    wy, wx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
    f = img.astype(np.float32)
    top = f[y0][:, x0] * (1 - wx) + f[y0][:, x1] * wx
    bot = f[y1][:, x0] * (1 - wx) + f[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def crop(frame, box, pad=0.05):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box[:4]
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    x0, y0 = int(max(0, x0 - px)), int(max(0, y0 - py))
    x1, y1 = int(min(w, x1 + px)), int(min(h, y1 + py))
    return frame[y0:y1, x0:x1], [x0, y0, x1, y1]


REID_BATCH = 8        # crops per OSNet run: a big batch takes hundreds of MB (the server has 1 GB)


def fingerprints(reid, crops):
    """L2-normalised appearance vectors for a list of BGR person crops."""
    out = []
    for i in range(0, len(crops), REID_BATCH):
        x = np.stack([(resize(c, 256, 128)[..., ::-1] / 255 - MEAN) / STD for c in crops[i:i + REID_BATCH]])
        out.append(reid.run(None, {"images": x.transpose(0, 3, 1, 2).astype(np.float32)})[0])
    v = np.concatenate(out)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def save_jpg(img, path):
    """A person crop (BGR) as a 160 px high JPEG, in-process (an ffmpeg per crop was
    a second of CPU each on this server)."""
    h, w = img.shape[:2]
    size = (max(2, round(w * 160 / h / 2) * 2), 160)
    Image.fromarray(np.ascontiguousarray(img[..., ::-1])).resize(size, Image.BICUBIC).save(path, quality=85)


def people_in(reid, cands, thumbs, ev, cap=MAX_CROPS):
    """Group one event's person crops (at most cap of them, spread over the event) into
    people (average linkage, by appearance)."""
    if len(cands) > cap:
        cands = [cands[i] for i in np.linspace(0, len(cands) - 1, cap).round().astype(int)]
    embs = fingerprints(reid, [c["img"] for c in cands])
    groups = sorted(cluster(embs, SAME_IN_EVENT), key=lambda g: -max(cands[i]["rank"] for i in g))
    out = []
    for k, g in enumerate(groups):
        v = embs[g].sum(0)
        v /= np.linalg.norm(v)
        # shown crop: the most typical of the clearer half (the biggest may be a stray)
        top = max(cands[i]["rank"] for i in g)
        best = cands[max((i for i in g if cands[i]["rank"] >= top / 2), key=lambda i: float(embs[i] @ v))]
        # when the person was in view: first and last keyframe they were seen in (keyframes
        # are ~2 s apart, so each end is good to about that)
        ts = [cands[i]["t"] for i in g]
        p = {"t": round(best["t"], 1), "t0": round(min(ts), 1), "t1": round(max(ts), 1),
             "box": best["box"], "score": round(best["score"], 2),
             "emb": base64.b64encode(v.astype(np.float16).tobytes()).decode()}
        if thumbs:
            p["thumb"] = f"{ev}-{k}.jpg"
            save_jpg(best["img"], os.path.join(thumbs, p["thumb"]))
        out.append(p)
    return out


def main():
    args = sys.argv[1:]
    thumbs = None
    if args[0] == "--thumbs":
        thumbs, args = args[1], args[2:]
        os.makedirs(thumbs, exist_ok=True)
    path, spans = args[0], [tuple(map(float, a.split(":"))) for a in args[1:]]
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    # no preallocated memory pools: the server has 1 GB, and they hold on to far more
    # than one frame at a time needs
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    sess = ort.InferenceSession(MODEL, opts, providers=["CPUExecutionProvider"])
    reid = ort.InferenceSession(REID, opts, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    w, h = probe_size(path)
    scale = min(SIZE / w, SIZE / h)
    keys = keyframes(path)
    if not keys:
        sys.exit(f"no keyframes found in {path}")
    picks, nframes = plan(keys, spans)
    total = max(1, sum(nframes))
    caps = [min(MAX_CROPS, max(MIN_CROPS, round(CROP_BUDGET * n / total))) for n in nframes]
    results = [{} for _ in spans]
    people = [[] for _ in spans]
    cands = [[] for _ in spans]
    left = list(nframes)  # frames still to come per event: at 0 its people are grouped
    seen = 0
    for i, x, frame in frames(path, keys, picks, w, h):
        seen += 1
        t = keys[i]
        out = sess.run(None, {name: x})[0][0]         # 3549 x 85: box, objectness, 80 classes
        cls = out[:, 5:].argmax(1)                    # each box counts only as its top class,
        score = out[:, 4] * out[:, 5:].max(1)         # so a person isn't also a "dog"
        labels = {}
        for cid, label in LABELS.items():
            s = score[cls == cid].max(initial=0)
            if s >= MIN_SCORE:
                labels[label] = round(float(s), 2)
        boxes = []
        for b in person_boxes(out, cls, score, scale):
            img, box = crop(frame, b)
            if box[3] - box[1] >= MIN_HEIGHT and box[2] - box[0] >= MIN_HEIGHT / 4:
                boxes.append({"t": t, "box": box, "score": b[4], "img": img,
                              "rank": b[4] * (box[3] - box[1])})
        for ev in picks[i]:
            for label, s in labels.items():
                cur = results[ev].setdefault(label, [0.0, round(t, 1), 0])
                cur[2] += 1
                if s > cur[0]:
                    cur[0], cur[1] = s, round(t, 1)
            cands[ev] += boxes
            left[ev] -= 1
            if left[ev] == 0 and cands[ev]:  # event done: group its people, free the crops
                people[ev] = people_in(reid, cands[ev], thumbs, ev, caps[ev])
                cands[ev] = []
    if seen < len(picks):
        sys.exit(f"ffmpeg gave {seen} of {len(picks)} keyframes for {path}")
    print(json.dumps({"found": results, "people": people}, separators=(",", ":")))


if __name__ == "__main__":
    main()
