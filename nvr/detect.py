#!/usr/bin/env python3
# Person detection for the NVR.
#
# Runs YOLOX-Tiny on sampled frames from the RECORDED segments (post-hoc, never touches the live
# recording path) to find people, and OSNet for re-ID embeddings so the same person can be searched
# across time. CPU-only onnxruntime. Meant to run server-side under tight systemd limits (MemoryMax +
# Nice) so it can never starve nvr-recorder on this small box.
#
# DECOUPLING (deliberate): standalone consumer. Never imports recorder.py/camera_stream.js, never opens
# their sockets, never touches the live path. Reads already-written segment files only. Results go to a
# separate database (the Oracle vector store), never nvr.db.
#
# This file is the detection CORE + a CLI for testing one frame; the poller is built on top of
# detect_embed_frame() (see detect_service.py).
#
#   python detect.py frame  <init.mp4> <seg.m4s>      # run on the first frame of one fMP4 segment
#   python detect.py segid   <segment id>             # look the segment up (read-only) and run it
import os, sys, io, json, subprocess
import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("NVR_DATA", "/home/ubuntu/nvr-data")
YOLOX_PATH = os.path.join(HERE, "models", "yolox_tiny.onnx")
OSNET_PATH = os.path.join(HERE, "models", "osnet_ain_x1_0_msmt17.onnx")

YOLOX_SIZE = 416
STRIDES = (8, 16, 32)
PERSON_CLASS = 0          # COCO: 0 = person
CONF_THR = float(os.environ.get("NVR_DET_CONF", 0.30))  # person-score cut (low: distant people on a fixed cam)
NMS_THR = float(os.environ.get("NVR_DET_NMS", 0.45))
PAD = 114                 # YOLOX letterbox pad value (0x72), image anchored top-left
OSNET_IN = (256, 128)     # H, W the re-ID model wants
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)     # ImageNet (torchreid) normalization
_STD = np.array([0.229, 0.224, 0.225], np.float32)

_yolox = _osnet = None


# Intra-op threads for CPU inference. Defaults to all cores: on the dedicated detect box (2 cores) that
# lets one YOLOX/OSNet call use both, roughly halving per-segment inference wall-time so a higher live
# sample rate keeps up. Set DET_THREADS=1 when detection shares a box with the recorder (the old setup).
DET_THREADS = int(os.environ.get("DET_THREADS", str(os.cpu_count() or 1)))


def session(path, threads=None):
    """A CPU inference session using DET_THREADS intra-op threads (all cores by default)."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads or DET_THREADS
    so.inter_op_num_threads = 1
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


# ---- frame decode: ONE shell-free ffmpeg call per frame, shared by detection and embedding ----
_DIMS = {}                       # init path -> (w, h); the camera resolution is constant per run


def _frame_dims(init):
    """Frame size, probed once per init file and cached (every segment of a run shares one resolution)."""
    if init not in _DIMS:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                              "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", init],
                             capture_output=True).stdout.decode().strip()
        w, h = (int(x) for x in out.split("x"))
        _DIMS[init] = (w, h)
    return _DIMS[init]


def decode_frame(init, seg, ss=None):
    """One RGB frame (HxWx3) from the segment. The fMP4 media segment needs its init prepended; we feed
    init+seg to ffmpeg on stdin (no shell, so paths with spaces/quotes are safe) and read RAW rgb24
    back — NOT PNG: encoding a ~3 MP lossless PNG per frame cost ~4x the decode itself (≈3.9s vs ≈1.0s).
    Dimensions come from a cheap, cached ffprobe of the init file. `ss` (seconds into the segment) goes
    AFTER -i: stdin is a pipe and input-seek (-ss before -i) needs a seekable source, so it would
    silently decode nothing; output-seek works on a pipe."""
    w, h = _frame_dims(init)
    with open(init, "rb") as f1, open(seg, "rb") as f2:
        data = f1.read() + f2.read()
    cmd = ["ffmpeg", "-v", "error", "-i", "pipe:0"] + (["-ss", str(ss)] if ss else []) + \
          ["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    out = subprocess.run(cmd, input=data, capture_output=True).stdout
    need = w * h * 3
    if len(out) < need:
        raise RuntimeError(f"frame decode failed ({len(out)}/{need} bytes from ffmpeg)")
    return np.frombuffer(out[:need], np.uint8).reshape(h, w, 3)


def decode_frames(init, seg, n=1, dur=None):
    """Up to `n` RGB frames evenly spaced across the segment, decoded in ONE ffmpeg pass. These segments
    are single-GOP (one keyframe at the very start), so a frame at time t costs decoding every P-frame up
    to t; seeking per frame would re-decode overlapping ranges. Instead we decode once and let the `fps`
    filter emit frames at 0, dur/n, 2·dur/n, …, so the pass only runs to ~(n-1)/n of the segment. Across
    consecutive segments that yields uniform ~dur/n sampling. n<=1 (or unknown dur) uses the cheap
    first-frame-only path. Returns a list of HxWx3 arrays (fewer than n only if the segment is short)."""
    w, h = _frame_dims(init)
    need = w * h * 3
    with open(init, "rb") as f1, open(seg, "rb") as f2:
        data = f1.read() + f2.read()
    if n <= 1 or not dur or dur <= 0:
        return [decode_frame(init, seg)]
    interval = max(float(dur) / n, 0.1)
    cmd = ["ffmpeg", "-v", "error", "-i", "pipe:0", "-vf", f"fps=1/{interval:.3f}",
           "-frames:v", str(int(n)), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    out = subprocess.run(cmd, input=data, capture_output=True).stdout
    k = len(out) // need
    if k == 0:
        raise RuntimeError(f"frame decode failed ({len(out)}/{need} bytes from ffmpeg)")
    return [np.frombuffer(out[i * need:(i + 1) * need], np.uint8).reshape(h, w, 3) for i in range(k)]


def _letterbox(frame_rgb, size=YOLOX_SIZE):
    """Resize-and-pad to size*size (image top-left, pad bottom-right). Returns (RGB canvas, ratio),
    where ratio maps letterbox px -> original px (orig = px / ratio)."""
    h, w = frame_rgb.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    resized = np.asarray(Image.fromarray(frame_rgb).resize((nw, nh)))
    canvas = np.full((size, size, 3), PAD, np.uint8)
    canvas[:nh, :nw] = resized
    return canvas, r


# ---- YOLOX ----
def _grids(size=YOLOX_SIZE, strides=STRIDES):
    grids, strs = [], []
    for s in strides:
        g = size // s
        xv, yv = np.meshgrid(np.arange(g), np.arange(g))
        grids.append(np.stack((xv, yv), 2).reshape(1, -1, 2))
        strs.append(np.full((1, g * g, 1), s))
    return np.concatenate(grids, 1), np.concatenate(strs, 1)


_GRID, _GSTR = _grids()


def _nms(boxes, scores, thr):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1); h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thr]
    return keep


def persons(output, ratio, conf_thr=None, nms_thr=None):
    """YOLOX raw head [1,N,85] -> list of {box:[x1,y1,x2,y2] (original px), score} for PERSON only."""
    conf_thr = CONF_THR if conf_thr is None else conf_thr
    nms_thr = NMS_THR if nms_thr is None else nms_thr
    o = output[0].astype(np.float32).copy()                 # [N, 85]
    o[:, :2] = (o[:, :2] + _GRID[0]) * _GSTR[0]              # grid-decode centers
    o[:, 2:4] = np.exp(o[:, 2:4]) * _GSTR[0]                 # and sizes
    cls = o[:, 5:]
    cls_id = cls.argmax(1)
    score = o[:, 4] * cls[np.arange(len(cls)), cls_id]
    keep = (cls_id == PERSON_CLASS) & (score > conf_thr)
    if not keep.any():
        return []
    b, s = o[keep, :4], score[keep]
    xy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                   b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1) / ratio
    return [{"box": [round(float(v), 1) for v in xy[i]], "score": round(float(s[i]), 3)}
            for i in _nms(xy, s, nms_thr)]


def detect_in(frame_rgb):
    """Person detections for an already-decoded RGB frame."""
    global _yolox
    if _yolox is None:
        _yolox = session(YOLOX_PATH)
    canvas, ratio = _letterbox(frame_rgb)
    blob = canvas[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None]  # RGB->BGR, NCHW, 0-255
    return persons(_yolox.run(None, {"images": blob})[0], ratio)


# ---- OSNet re-ID embeddings (the known/unknown signal) ----
# A person crop -> 512-d L2-normalized vector. Same person across time has high cosine similarity;
# different people low. These vectors are what the Oracle 23ai VECTOR column stores and matches.
def embed(frame_rgb, boxes):
    """[n,512] L2-normalized OSNet embeddings for the given boxes (original-px xyxy) in frame_rgb."""
    global _osnet
    if _osnet is None:
        _osnet = session(OSNET_PATH)
    H, W = frame_rgb.shape[:2]
    batch = []
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(W, int(x2)), min(H, int(y2))
        if x2 <= x1 or y2 <= y1:
            batch.append(np.zeros((3, *OSNET_IN), np.float32)); continue
        crop = Image.fromarray(frame_rgb[y1:y2, x1:x2]).resize((OSNET_IN[1], OSNET_IN[0]))  # (W,H)
        a = (np.asarray(crop, np.float32) / 255.0 - _MEAN) / _STD
        batch.append(a.transpose(2, 0, 1))
    feat = _osnet.run(None, {"images": np.stack(batch).astype(np.float32)})[0]
    return feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-9)


def detect_frame(init, seg, ss=None):
    """Person boxes for one frame (no embeddings)."""
    return detect_in(decode_frame(init, seg, ss))


def detect_embed(frame):
    """Detections for an already-decoded RGB frame, each with a 512-d 'embedding'."""
    dets = detect_in(frame)
    if dets:
        for d, f in zip(dets, embed(frame, [d["box"] for d in dets])):
            d["embedding"] = f.astype(float).tolist()
    return dets


def detect_embed_frame(init, seg, ss=None):
    """Detections for one frame, each with a 512-d 'embedding'. Empty list if no people.
    Decodes the frame ONCE and shares it between detection and embedding."""
    return detect_embed(decode_frame(init, seg, ss))


def crop_to_jpeg(frame_rgb, box, pad=0.12, max_side=320, quality=82):
    """A JPEG of one person box cut from the already-decoded frame (no ffmpeg). Done at detection
    time so the UI can serve it as a static file forever — even after the segment is deleted.
    Returns bytes, or None if the box is empty."""
    H, W = frame_rgb.shape[:2]
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
    x2, y2 = min(W, int(x2 + pw)), min(H, int(y2 + ph))
    if x2 <= x1 or y2 <= y1:
        return None
    im = Image.fromarray(frame_rgb[y1:y2, x1:x2])
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def focus_score(rgb, box=None):
    """A blur/sharpness score for a crop (or a box within a frame): the variance of the Laplacian
    normalized by the crop's own variance, so it reads high-frequency DETAIL relative to overall
    contrast and stays roughly size/brightness-invariant. Sharp subjects score high; motion-blurred,
    tiny or out-of-focus ones score low. Used to keep blurred crops out of the re-ID training gallery.
    Pure numpy. Returns 0.0 for a too-small region."""
    H, W = rgb.shape[:2]
    if box is not None:
        x1, y1, x2, y2 = (max(0, int(box[0])), max(0, int(box[1])), min(W, int(box[2])), min(H, int(box[3])))
        rgb = rgb[y1:y2, x1:x2]
    if rgb.shape[0] < 10 or rgb.shape[1] < 10:
        return 0.0
    g = rgb.astype(np.float32).mean(axis=2)                       # grayscale
    lap = (-4.0 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    v = g.var()
    return float(lap.var() / (v + 1e-3))


# ---- CLI (testing) ----
def _seg_by_id(segid):
    import sqlite3
    # read-only, immutable: never locks or blocks the recorder's own writes to nvr.db
    uri = f"file:{os.path.join(DATA, 'nvr.db')}?mode=ro&immutable=1"
    c = sqlite3.connect(uri, uri=True)
    row = c.execute("SELECT path, init FROM segments WHERE id=?", (segid,)).fetchone()
    c.close()
    if not row:
        sys.exit(f"no segment id {segid}")
    return os.path.join(DATA, row[0]), os.path.join(DATA, row[1])


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "frame":
        init, seg = sys.argv[2], sys.argv[3]
    elif len(sys.argv) >= 3 and sys.argv[1] == "segid":
        seg, init = _seg_by_id(int(sys.argv[2]))
    else:
        sys.exit(__doc__)
    import time
    t = time.time()
    dets = detect_embed_frame(init, seg)
    print(json.dumps({"ms": round((time.time() - t) * 1000), "persons": len(dets),
                      "detections": [{k: v for k, v in d.items() if k != "embedding"} for d in dets]},
                     indent=2))
