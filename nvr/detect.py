#!/usr/bin/env python3
# Person detection for the NVR.
#
# Runs YOLOX-Tiny on sampled frames from the RECORDED segments (post-hoc, never touches the live
# recording path) to find people, and later OSNet for re-ID embeddings so the same person can be
# searched across time. CPU-only onnxruntime. Meant to run server-side under tight systemd limits
# (MemoryMax + Nice) so it can never starve nvr-recorder on this small box.
#
# DECOUPLING (deliberate): this is a standalone consumer. It must NOT be coupled to the camera or
# the recorder. It never imports recorder.py/camera_stream.js, never opens their sockets, and never
# touches the live path. It only READS already-written segment files, discovering them from nvr.db
# opened READ-ONLY (immutable=1, no locks), and writes its results to its OWN database (detect.db) so
# it adds zero write contention to the recorder's index. It runs as its own process/service and can
# be stopped, restarted, or fall far behind with no effect on recording.
#
# This file is the detection CORE + a CLI for testing one frame; the polling service is built on
# top of detect_frame() once the core is proven on real footage.
#
#   python detect.py frame  <init.mp4> <seg.m4s>      # run on the first frame of one fMP4 segment
#   python detect.py segid   <segment id>             # look the segment up (read-only) and run it
import os, sys, json, subprocess
import numpy as np
import onnxruntime as ort

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


def session(path, threads=1):
    """A CPU session pinned to a small thread count — this box has 2 cores and little RAM."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


# ---- frame extraction (ffmpeg; the fMP4 media segment needs its init segment prepended) ----
def _probe_dims(init, seg):
    out = subprocess.run(
        ["bash", "-c", f'cat "{init}" "{seg}" | ffprobe -v error -select_streams v:0 '
         f'-show_entries stream=width,height -of csv=p=0 -i pipe:0'],
        capture_output=True, text=True).stdout.strip()
    w, h = (int(x) for x in out.split(",")[:2])
    return w, h


def frame_letterboxed(init, seg, ss=None, size=YOLOX_SIZE):
    """One BGR frame from the segment, letterboxed to size*size (image top-left, pad bottom-right).
    Returns (frame uint8 HxWx3, ratio) where ratio maps letterbox px -> original px (orig = px/ratio)."""
    w, h = _probe_dims(init, seg)
    ratio = min(size / h, size / w)
    seek = f"-ss {ss} " if ss else ""
    vf = (f"scale={size}:{size}:force_original_aspect_ratio=decrease,"
          f"pad={size}:{size}:0:0:color=0x{PAD:02x}{PAD:02x}{PAD:02x}")
    buf = subprocess.run(
        ["bash", "-c", f'cat "{init}" "{seg}" | ffmpeg -v error {seek}-i pipe:0 '
         f'-frames:v 1 -vf "{vf}" -f rawvideo -pix_fmt bgr24 pipe:1'],
        capture_output=True).stdout
    need = size * size * 3
    if len(buf) < need:
        raise RuntimeError(f"ffmpeg returned {len(buf)} bytes, need {need} (decode failed?)")
    return np.frombuffer(buf[:need], np.uint8).reshape(size, size, 3), ratio


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


def persons(output, ratio, conf_thr=CONF_THR, nms_thr=NMS_THR):
    """YOLOX raw head [1,N,85] -> list of {box:[x1,y1,x2,y2] (original px), score} for PERSON only."""
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
    return [{"box": [round(v, 1) for v in xy[i]], "score": round(float(s[i]), 3)}
            for i in _nms(xy, s, nms_thr)]


_yolox = None
def detect_frame(init, seg, ss=None):
    global _yolox
    if _yolox is None:
        _yolox = session(YOLOX_PATH)
    frame, ratio = frame_letterboxed(init, seg, ss)
    blob = frame.astype(np.float32).transpose(2, 0, 1)[None]   # BGR 0-255, NCHW
    out = _yolox.run(None, {"images": blob})[0]
    return persons(out, ratio)


# ---- OSNet re-ID embeddings (the known/unknown signal) ----
# A person crop -> 512-d L2-normalized vector. Same person across time has high cosine similarity;
# different people low. These vectors are what the Oracle 23ai VECTOR column stores and matches.
OSNET_IN = (256, 128)                                   # H, W the model wants
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)     # ImageNet (torchreid) normalization
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_osnet = None


def full_frame(init, seg, ss=None):
    """Full-resolution BGR frame (for cropping people) — only decoded when there's a person to embed."""
    w, h = _probe_dims(init, seg)
    seek = f"-ss {ss} " if ss else ""
    buf = subprocess.run(
        ["bash", "-c", f'cat "{init}" "{seg}" | ffmpeg -v error {seek}-i pipe:0 '
         f'-frames:v 1 -f rawvideo -pix_fmt bgr24 pipe:1'], capture_output=True).stdout
    need = w * h * 3
    if len(buf) < need:
        raise RuntimeError(f"full-frame decode failed ({len(buf)}/{need} bytes)")
    return np.frombuffer(buf[:need], np.uint8).reshape(h, w, 3)


def embed(frame_bgr, boxes):
    """[n,512] L2-normalized OSNet embeddings for the given boxes (original-px xyxy) in frame_bgr."""
    from PIL import Image
    global _osnet
    if _osnet is None:
        _osnet = session(OSNET_PATH)
    H, W = frame_bgr.shape[:2]
    batch = []
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(W, int(x2)), min(H, int(y2))
        if x2 <= x1 or y2 <= y1:
            batch.append(np.zeros((3, *OSNET_IN), np.float32)); continue
        crop = frame_bgr[y1:y2, x1:x2][:, :, ::-1]                     # BGR -> RGB
        img = Image.fromarray(crop).resize((OSNET_IN[1], OSNET_IN[0]))  # (W,H)
        a = (np.asarray(img, np.float32) / 255.0 - _MEAN) / _STD
        batch.append(a.transpose(2, 0, 1))
    feat = _osnet.run(None, {"images": np.stack(batch).astype(np.float32)})[0]
    return feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-9)


def detect_embed_frame(init, seg, ss=None):
    """Detections for one frame, each with a 512-d 'embedding'. Empty list if no people."""
    dets = detect_frame(init, seg, ss)
    if not dets:
        return []
    feats = embed(full_frame(init, seg, ss), [d["box"] for d in dets])
    for d, f in zip(dets, feats):
        d["embedding"] = f.astype(float).tolist()
    return dets


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
    dets = detect_frame(init, seg)
    print(json.dumps({"ms": round((time.time() - t) * 1000), "persons": len(dets),
                      "detections": dets}, indent=2))
