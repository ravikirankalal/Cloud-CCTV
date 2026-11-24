#!/usr/bin/env python3
# Person-detection poller. Samples recorded segments, detects + embeds people, assigns each to a
# known identity by vector distance (or leaves it unknown), and writes to the Oracle 23ai vector DB.
#
# DECOUPLED by design: reads nvr.db READ-ONLY (mode=ro, WAL-visible), writes ONLY to Oracle, imports no recorder
# code, opens no recorder sockets, never touches the live path. It can lag, stop, or restart with zero
# effect on recording. Meant to run under systemd with MemoryMax + Nice on this small box.
#
#   python detect_service.py run                          # live poller: process new segments forever
#   python detect_service.py backfill <from_id> <to_id>   # process an explicit id range (test/catch-up)
#   python detect_service.py backfill-crops               # make the static crop jpeg for stored rows
#   python detect_service.py cluster                      # group unknown detections by re-ID (one-shot)
import os, sys, time, sqlite3, array, importlib.util
import numpy as np
import oracledb

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("NVR_DATA", "/home/ubuntu/nvr-data")
STREAM = os.environ.get("NVR_STREAM", "cam1")            # presence source: sampled for YOLOX (cheap SD when set)
REID_STREAM = os.environ.get("DET_REID_STREAM", "")      # re-ID source: "" = same frame; "cam1" = decode HD only
                                                         # for the OSNet crop when SD presence finds a person
REID_SHOTS = int(os.environ.get("DET_REID_SHOTS", "6"))  # pool the top-N shots of a track into one averaged
                                                         # embedding. Single-shot SD re-ID can't separate people
                                                         # (intra-person spread 0.45-0.65 > inter-person 0.13-0.21);
                                                         # averaging N>=3 shots cancels per-frame SD noise and the
                                                         # same-person spread drops below the nearest other person
                                                         # (measured: intra 0.40->0.09 at N=6). 1 = old best-shot.
STRIDE = int(os.environ.get("DET_STRIDE", "2"))          # sample every Nth segment
FRAMES = int(os.environ.get("DET_FRAMES", "1"))          # frames/segment at the LIVE edge (evenly spaced)
FPS = float(os.environ.get("DET_FPS", "0"))              # live target frames/SEC; >0 overrides FRAMES (n = dur*FPS)
FRAMES_BACKFILL = int(os.environ.get("DET_FRAMES_BACKFILL", "1"))  # coarser rate while catching up on old footage
BACKFILL_AGE = float(os.environ.get("DET_BACKFILL_AGE", "120"))    # a segment older than this (s) is backfill, not live
BATCH = int(os.environ.get("DET_BATCH", "25"))           # segments per loop before committing/sleeping
IDLE = int(os.environ.get("DET_IDLE", "15"))             # sleep seconds when caught up
RETRY = int(os.environ.get("DET_RETRY", "30"))           # sleep before reconnecting to Oracle after a DB error
MATCH_DIST = float(os.environ.get("DET_MATCH", "0.28"))  # cosine distance < this => same known person
CLUSTER_DIST = float(os.environ.get("DET_CLUSTER", "0.26"))  # avg-linkage: group unknowns whose MEAN cosine
#   similarity >= 1 - CLUSTER_DIST (0.74). On the weak SD embeddings a looser cut (<0.66) collapses the dense
#   core into one mixed blob; 0.74 keeps groups PURE (biggest stays ~coherent, intra-sim ~0.77) and taggable.
# CROP_DIR / BG_PATH default under DATA, but are overridable so a remote detector (DATA = a read-only
# mount of the recorder's segments) can still WRITE its crops + gate model to a local, writable path.
CROP_DIR = os.environ.get("NVR_CROPS") or os.path.join(DATA, "crops")   # static person-crop jpegs
BG_PATH = os.environ.get("NVR_BG") or os.path.join(DATA, "detect_bg.npz")  # persisted motion-gate model
# Segment DB: local nvr.db (WAL, mode=ro) by default. A remote detector points NVR_DB_PATH at a periodic
# consistent SNAPSHOT (VACUUM INTO, no WAL) shared read-only, and sets NVR_DB_IMMUTABLE=1 so SQLite reads
# it without locks/-shm (safe over NFS); the snapshot is replaced per poll, so run() reopens it each batch.
DB_PATH = os.environ.get("NVR_DB_PATH") or os.path.join(DATA, "nvr.db")
DB_IMMUTABLE = os.environ.get("NVR_DB_IMMUTABLE") == "1"
GATE_ENABLED = os.environ.get("DET_GATE", "1") != "0"    # skip YOLOX on empty (no-foreground) frames
GATE_MIN_PX = int(os.environ.get("DET_GATE_MIN_PX", "140"))  # foreground blob px (160-wide grid) to trigger
GATE_K = float(os.environ.get("DET_GATE_K", "2.5"))      # per-pixel std multiple for the background model
TRACK = os.environ.get("DET_TRACK", "0") != "0"          # track-based re-ID: group a person's SD sightings,
                                                         # then ONE HD decode+embed per track (not per frame)
TRACK_IOU = float(os.environ.get("DET_TRACK_IOU", "0.3"))   # min IoU to link a box to an open track
TRACK_GAP = float(os.environ.get("DET_TRACK_GAP", "8"))     # close a track after this many sec (processed-ts) of no obs
TRACK_MAX = float(os.environ.get("DET_TRACK_MAX", "25"))    # emit a chunk at least this often during a long track
HD_CONFIRM = os.environ.get("DET_HD_CONFIRM", "0") != "0"   # optional: after SD presence finds a person, decode
#   the HD stream at the same ts, re-run YOLOX there, and KEEP the SD box only if a HD person-box overlaps it
#   (IoU) — so an SD false positive (a box YOLOX fires on in the blurry SD frame but not in sharp HD) is dropped
#   BEFORE it is embedded, matched or grouped. DECOUPLED from re-ID: HD is used ONLY to validate; the crop and
#   embedding still come from the aligned SD frame (or DET_REID_STREAM when set), so this does NOT reintroduce
#   HD/SD crop drift. Costs a 2nd YOLOX pass on HD per hit (per track best-shot in track mode), off by default.
HD_CONFIRM_STREAM = os.environ.get("DET_HD_CONFIRM_STREAM") or (REID_STREAM or "cam1")  # HD stream to confirm on
HD_CONFIRM_IOU = float(os.environ.get("DET_HD_CONFIRM_IOU", "0.3"))  # min IoU of a scaled-SD box vs some HD box
# HD content-alignment (option "b"): the HD stream lags/leads SD by a per-RUN offset (the difference in the
# two streams' first-frame latency); it is constant within a run but re-anchors at every reconnect, and the
# recorder's own timestamps don't expose it. So instead of a fixed offset we CALIBRATE from content: on the
# first detection of a run, search a small time window for the HD frame whose person-box best matches the
# scaled SD box; that lag is the offset. Cache it and reuse it (one HD decode/hit) until matches stop landing
# — which is itself the "a reconnect happened" signal — then recalibrate. Supersedes the naive same-ts
# HD_CONFIRM when on, and does re-ID + confirm from the SAME aligned HD frame.
HD_ALIGN = os.environ.get("DET_HD_ALIGN", "0") != "0"
HD_ALIGN_LO = float(os.environ.get("DET_HD_ALIGN_LO", "-2"))    # calibration search window low  (s, ts+lo)
HD_ALIGN_HI = float(os.environ.get("DET_HD_ALIGN_HI", "6"))     # calibration search window high (s, ts+hi)
HD_ALIGN_STEP = float(os.environ.get("DET_HD_ALIGN_STEP", "1")) # calibration step (s); offset resolution
HD_ALIGN_IOU = float(os.environ.get("DET_HD_ALIGN_IOU", "0.3")) # IoU for a scaled-SD box vs a HD person-box
HD_MISS_MAX = int(os.environ.get("DET_HD_MISS_MAX", "4"))       # consecutive misses before recalibrating
HD_RECAL_MIN = float(os.environ.get("DET_HD_RECAL_MIN", "20"))  # min seconds between calibration sweeps
_HD = {"off": 0.0, "ok": False, "miss": 0, "last_cal": 0.0}     # per-run HD->SD offset cache (module state)

os.makedirs(CROP_DIR, exist_ok=True)

sys.path.insert(0, HERE)
import adb                                                # shared Oracle connection
import match                                              # prototype gallery + nearest-proto live match
from motion_gate import MotionGate                        # background-subtraction gate (skips YOLOX on empty frames)
_spec = importlib.util.spec_from_file_location("detect", os.path.join(HERE, "detect.py"))
detect = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(detect)


def ro_nvr():
    # Local (default): mode=ro, NOT immutable — the recorder writes nvr.db in WAL mode, so new segments
    # live in -wal until a checkpoint; immutable=1 would ignore the WAL and freeze at the last checkpoint.
    # A plain read-only handle reads the WAL and sees the live edge, and WAL still lets the recorder write
    # concurrently without blocking.
    # Remote (NVR_DB_IMMUTABLE=1): DB_PATH is a static snapshot (no WAL), so immutable=1 reads it with no
    # locks and no -shm — the only mode that is safe over an NFS mount. run() reopens it each poll batch.
    q = "immutable=1" if DB_IMMUTABLE else "mode=ro"
    return sqlite3.connect(f"file:{DB_PATH}?{q}", uri=True)


def match_known(cur, emb):
    """Nearest known PROTOTYPE by cosine distance; return its identity_id if within MATCH_DIST.
    Matches against the small per-person prototype gallery (match.rebuild_protos) rather than every
    confirmed crop, so the live per-detection match stays cheap as the store grows."""
    return match.nearest_proto(cur, emb, MATCH_DIST)


def get_state(cur, k, default):
    cur.execute("SELECT v FROM detect_state WHERE k = :1", [k])
    row = cur.fetchone()
    return int(row[0]) if row else default


def set_state(cur, k, v):
    cur.execute("MERGE INTO detect_state d USING (SELECT :1 k, :2 v FROM dual) s "
                "ON (d.k = s.k) WHEN MATCHED THEN UPDATE SET d.v = s.v "
                "WHEN NOT MATCHED THEN INSERT (k, v) VALUES (s.k, s.v)", [k, v])


def _save_crop(frame, box, det_id):
    """Write the static person crop for a detection (atomic). Best-effort; never fatal."""
    try:
        jpg = detect.crop_to_jpeg(frame, box)
        if jpg:
            tmp = os.path.join(CROP_DIR, f"{det_id}.jpg.tmp")
            with open(tmp, "wb") as f:
                f.write(jpg)
            os.replace(tmp, os.path.join(CROP_DIR, f"{det_id}.jpg"))
    except Exception as e:
        print(f"  crop {det_id} failed: {e}", flush=True)


def _scale_box(box, src_wh, dst_wh):
    """Map a box from presence-frame px to re-ID-frame px. The sub-stream is the SAME field of view at a
    lower resolution, so an independent x/y scale is exact (bar the tiny aspect rounding of 800x448)."""
    (sw, sh), (dw, dh) = src_wh, dst_wh
    fx, fy = dw / sw, dh / sh
    return [box[0] * fx, box[1] * fy, box[2] * fx, box[3] * fy]


def _reid_frame(nvr, ts):
    """The re-ID (HD) segment covering wall-clock `ts` and its decoded frame, or (None, None) when the
    re-ID stream has a gap there — the caller then re-IDs on the presence frame itself (the fallback case
    the SD recording exists for). Single-GOP segments, so the decode seeks to ts within the segment."""
    row = nvr.execute("SELECT id, start, dur, path, init FROM segments WHERE stream = ? AND start <= ? "
                      "ORDER BY start DESC LIMIT 1", (REID_STREAM, ts)).fetchone()
    if not row:
        return None, None
    sid, start, dur, path, init = row
    if dur and ts > float(start) + float(dur):
        return None, None                                 # ts lands in a re-ID-stream gap
    frame = detect.decode_frame(os.path.join(DATA, init), os.path.join(DATA, path), max(0.0, ts - float(start)))
    return int(sid), frame


def _presence(nvr, seg_id, gate=None):
    """Decode this presence segment's sampled frame(s), feed the motion gate, run YOLOX, and return the
    per-frame person boxes (in presence-frame px) WITHOUT touching the re-ID stream or the DB. Returns
    (seg_start, pres_wh, [(ts, frame, boxes), ...]); only frames with >=1 box are listed, but the gate
    still sees every frame so its background model stays current. seg_start is returned even with no
    boxes, so a caller's tracker can advance its clock over empty footage.

    Sample rate adapts to how fresh the footage is: a LIVE segment (recorded within BACKFILL_AGE seconds)
    is sampled at DET_FPS frames/sec (n = round(dur*FPS)) when DET_FPS>0, else at a fixed DET_FRAMES/segment;
    an OLD segment (we're catching up on a backlog) gets the coarser DET_FRAMES_BACKFILL so the poller drains
    fast and returns to live. With n>1 we sample that many frames evenly across the segment (single ffmpeg
    pass; see detect.decode_frames). Each frame's ts is its own wall-clock time (start + i*dur/n)."""
    row = nvr.execute("SELECT start, path, init, dur FROM segments WHERE id = ?", (seg_id,)).fetchone()
    if not row:
        return None, None, []
    start, path, init, dur = row
    if time.time() - float(start) > BACKFILL_AGE:
        n = FRAMES_BACKFILL                               # old footage: catch up cheap
    elif FPS > 0 and dur:
        n = max(1, round(float(dur) * FPS))               # live: sample at a target frames/sec
    else:
        n = FRAMES                                        # live: fixed frames/segment
    init_abs = os.path.join(DATA, init)
    frames = detect.decode_frames(init_abs, os.path.join(DATA, path), n, dur)
    pres_wh = detect._frame_dims(init_abs)
    out = []
    for i, frame in enumerate(frames):
        if gate is not None and not gate.check(frame)[0]:
            continue                                      # empty scene: don't pay for YOLOX
        ts = float(start) + (i * float(dur) / n if dur and n > 1 else 0.0)
        boxes = detect.detect_in(frame)                   # YOLOX presence (on the SD frame when so configured)
        if boxes:
            out.append((ts, frame, boxes))
    return float(start), pres_wh, out


def _reid_boxes(nvr, ts, boxes, pres_wh):
    """Resolve the re-ID target for presence `boxes` (presence px) at `ts`. Returns (hd_frame, hd_seg,
    hd_boxes) when DET_REID_STREAM has footage covering ts (boxes scaled SD->HD), else (None, None, None)
    so the caller re-IDs on the presence frame itself — the SD-only fallback the sub-stream exists for."""
    if REID_STREAM:
        hd_seg, hd_frame = _reid_frame(nvr, ts)
        if hd_frame is not None:
            rboxes = [_scale_box(b, pres_wh, (hd_frame.shape[1], hd_frame.shape[0])) for b in boxes]
            return hd_frame, hd_seg, rboxes
    return None, None, None


def _hd_confirm(nvr, ts, sd_boxes, pres_wh):
    """Validate SD presence boxes against the HD stream, WITHOUT changing the crop/embedding source.
    Decodes the HD confirm-stream frame covering `ts`, runs YOLOX there, and returns a keep-mask over
    `sd_boxes` (presence px): True where the SD box (scaled to HD px) overlaps some HD person-box by
    >= HD_CONFIRM_IOU. This is the one place a 2nd YOLOX pass runs, and only on frames SD already flagged.
    FAIL-OPEN — an HD gap at ts, or a decode/inference error, keeps every box, so a real sighting is never
    lost to a confirm miss; the HD frame is used ONLY to reject, never to crop (crops stay SD-aligned)."""
    if not sd_boxes:
        return []
    try:
        row = nvr.execute("SELECT start, dur, path, init FROM segments WHERE stream = ? AND start <= ? "
                          "ORDER BY start DESC LIMIT 1", (HD_CONFIRM_STREAM, ts)).fetchone()
        if not row:
            return [True] * len(sd_boxes)                 # no HD stream here at all: can't confirm, keep
        start, dur, path, init = row
        if dur and ts > float(start) + float(dur):
            return [True] * len(sd_boxes)                 # ts lands in an HD gap: keep
        hd = detect.decode_frame(os.path.join(DATA, init), os.path.join(DATA, path), max(0.0, ts - float(start)))
        hd_dets = [d["box"] for d in detect.detect_in(hd)]
        hd_wh = (hd.shape[1], hd.shape[0])
    except Exception as e:
        print(f"  HD-confirm failed ({e}); keeping all boxes", flush=True)
        return [True] * len(sd_boxes)
    scaled = [_scale_box(b, pres_wh, hd_wh) for b in sd_boxes]
    return [any(_iou(s, h) >= HD_CONFIRM_IOU for h in hd_dets) for s in scaled]


# ---- HD content-alignment (option "b"): calibrate a per-run HD->SD offset from content, cache it ----
def _hd_reset():
    """Forget the cached offset so the next detection recalibrates. Called when the detector (re)connects."""
    _HD.update(off=0.0, ok=False, miss=0, last_cal=0.0)


def _hd_decode(nvr, t):
    """Decode the HD align/confirm stream (HD_CONFIRM_STREAM) frame covering wall-clock `t`:
    (seg_id, frame), or (None, None) when that stream has no footage at `t`."""
    row = nvr.execute("SELECT id, start, dur, path, init FROM segments WHERE stream = ? AND start <= ? "
                      "ORDER BY start DESC LIMIT 1", (HD_CONFIRM_STREAM, t)).fetchone()
    if not row:
        return None, None
    sid, start, dur, path, init = row
    if dur and t > float(start) + float(dur):
        return None, None
    return int(sid), detect.decode_frame(os.path.join(DATA, init), os.path.join(DATA, path), max(0.0, t - float(start)))


def _hd_best(frame, scaled):
    """The HD person-box in `frame` that best overlaps `scaled` (an SD box in HD px), and that IoU."""
    best, biou = None, 0.0
    for d in detect.detect_in(frame):
        v = _iou(scaled, d["box"])
        if v > biou:
            biou, best = v, d["box"]
    return best, biou


def _hd_calibrate(nvr, ts, sd_box, pres_wh):
    """Sweep the offset window for the HD frame whose person-box best matches the scaled SD box; on a match
    (IoU >= HD_ALIGN_IOU) set the cache and return (frame, seg_id, hd_box), else return None (e.g. an SD
    animal/noise FP that no HD frame confirms). One HD decode + YOLOX per grid step — runs only to (re)learn
    the offset, then the cache serves every hit for the rest of the run."""
    _HD["last_cal"] = time.time()
    n = max(1, int(round((HD_ALIGN_HI - HD_ALIGN_LO) / HD_ALIGN_STEP)))
    best = None                                              # (iou, k, frame, seg, hd_box)
    for i in range(n + 1):
        k = round(HD_ALIGN_LO + i * HD_ALIGN_STEP, 3)
        seg, f = _hd_decode(nvr, ts + k)
        if f is None:
            continue
        scaled = _scale_box(sd_box, pres_wh, (f.shape[1], f.shape[0]))
        hb, iou = _hd_best(f, scaled)
        if hb is not None and (best is None or iou > best[0]):
            best = (iou, k, f, seg, hb)
    if best and best[0] >= HD_ALIGN_IOU:
        _HD.update(off=best[1], ok=True, miss=0)
        print(f"  HD align: calibrated offset={best[1]:+.2f}s (IoU {best[0]:.2f})", flush=True)
        return best[2], best[3], best[4]
    return None


def _aligned_hd(nvr, ts, sd_box, pres_wh):
    """Resolve one SD box against the time-aligned HD frame. Returns (hd_frame, hd_seg, hd_box, matched):
    matched=True => a HD person-box overlaps the aligned SD box, so the detection is confirmed and hd_box is
    the sharp HD box to crop/embed; matched=False => HD did not confirm it (drift-tolerant reject — drops SD
    false positives incl. animals). Uses the cached offset, calibrating when unset (respecting a cooldown)
    and invalidating the cache after HD_MISS_MAX consecutive misses (a reconnect shifted the offset)."""
    if pres_wh is None:
        return None, None, None, False
    if not _HD["ok"] and (time.time() - _HD["last_cal"] >= HD_RECAL_MIN):
        got = _hd_calibrate(nvr, ts, sd_box, pres_wh)
        if got:
            return got[0], got[1], got[2], True
        return None, None, None, False                      # couldn't calibrate on this box -> unconfirmed
    seg, f = _hd_decode(nvr, ts + _HD["off"])               # cached offset (or default while in cooldown)
    if f is None:
        return None, None, None, False
    scaled = _scale_box(sd_box, pres_wh, (f.shape[1], f.shape[0]))
    hb, iou = _hd_best(f, scaled)
    if iou >= HD_ALIGN_IOU:
        _HD["miss"] = 0
        _HD["ok"] = True                                    # a good match confirms the offset is current
        return f, seg, hb, True
    _HD["miss"] += 1
    if _HD["ok"] and _HD["miss"] >= HD_MISS_MAX:
        _HD["ok"] = False                                   # stale: recalibrate on a later hit (cooldown-gated)
        print(f"  HD align: {_HD['miss']} consecutive misses; will recalibrate", flush=True)
    return f, seg, scaled, False


def _boxfrac(box, frame_wh):
    """How much of the frame the box fills, as the larger of its width/height fractions (0..1+). ~1 means
    an EXTREME close-up (the person fills the frame → a partial, distorted crop) — poor training data."""
    w, h = frame_wh
    return max((box[2] - box[0]) / float(w or 1), (box[3] - box[1]) / float(h or 1))


def _insert_detection(cur, ts, seg_id, box, score, ev, ts_start, ts_end, focus=None, boxfrac=None):
    """Insert one detection row (the caller saves its crop) and return its id. Shared by the per-frame and
    the track paths so the row shape — including the [ts_start, ts_end] presence span — stays identical.
    ts is the representative/best-shot time to navigate to; the span is when the person was actually seen.
    `focus` (detect.focus_score) and `boxfrac` (_boxfrac) are crop-quality signals the training gallery
    gates on, so blurry or extreme close-up crops never seed a person's prototypes."""
    emb = array.array("f", [float(x) for x in ev])
    ident = match_known(cur, emb)
    idv = cur.var(oracledb.NUMBER)
    cur.execute("""INSERT INTO detections (ts, seg_id, box, score, embedding, identity_id, ts_start, ts_end,
                                           focus, boxfrac)
                   VALUES (:1, :2, :3, :4, :5, :6, :7, :8, :9, :10) RETURNING id INTO :11""",
                (ts, seg_id, ",".join(str(int(v)) for v in box),
                 float(score), emb, ident, float(ts_start), float(ts_end),
                 (float(focus) if focus is not None else None),
                 (float(boxfrac) if boxfrac is not None else None), idv))
    return int(idv.getvalue()[0])


def process_segment(nvr, cur, seg_id, gate=None):
    """Per-frame path (DET_TRACK off): detect people in one presence segment and insert one row each,
    re-ID'd on the HD frame at the same ts when DET_REID_STREAM is set (box scaled SD->HD, stored against
    the HD seg_id), else on the presence frame itself. Returns the number stored. DB errors propagate (to
    trigger a reconnect); decode errors are the caller's to swallow."""
    start, pres_wh, frames = _presence(nvr, seg_id, gate)
    if start is None:
        return 0
    stored = 0
    for ts, frame, boxes in frames:
        if HD_ALIGN:                                      # option b: each box re-ID'd + confirmed on aligned HD
            for b in boxes:
                rframe, rseg, rbox, ok = _aligned_hd(nvr, ts, b["box"], pres_wh)
                if not ok:
                    continue
                ev = detect.embed(rframe, [rbox])[0]
                did = _insert_detection(cur, ts, rseg, rbox, b["score"], ev, ts, ts,
                                        focus=detect.focus_score(rframe, rbox),
                                        boxfrac=_boxfrac(rbox, (rframe.shape[1], rframe.shape[0])))
                _save_crop(rframe, rbox, did)
                stored += 1
            continue
        sd_boxes = [b["box"] for b in boxes]
        if HD_CONFIRM:                                    # validate on HD first, then re-ID the survivors
            keep = _hd_confirm(nvr, ts, sd_boxes, pres_wh)
            boxes = [b for b, k in zip(boxes, keep) if k]
            sd_boxes = [b for b, k in zip(sd_boxes, keep) if k]
            if not sd_boxes:
                continue
        rframe, rseg, rboxes = _reid_boxes(nvr, ts, sd_boxes, pres_wh)
        if rframe is None:                                # no HD there: re-ID on the presence frame
            rframe, rseg, rboxes = frame, seg_id, sd_boxes
        rwh = (rframe.shape[1], rframe.shape[0])
        for b, hbox, ev in zip(boxes, rboxes, detect.embed(rframe, rboxes)):
            did = _insert_detection(cur, ts, rseg, hbox, b["score"], ev, ts, ts,
                                    focus=detect.focus_score(rframe, hbox), boxfrac=_boxfrac(hbox, rwh))
            _save_crop(rframe, hbox, did)
            stored += 1
    return stored


# ---- track-based re-ID: group a person's SD sightings, decode HD once on the best shot ----
def _decode_at(nvr, seg_id, ts):
    """Decode the frame at wall-clock `ts` from a given segment (by id), or None if the segment is gone."""
    r = nvr.execute("SELECT start, path, init FROM segments WHERE id = ?", (seg_id,)).fetchone()
    if not r:
        return None
    start, path, init = r
    return detect.decode_frame(os.path.join(DATA, init), os.path.join(DATA, path), max(0.0, ts - float(start)))


def _iou(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _quality(box, score):
    """Best-shot score: confident AND large — a big, sure box makes the best re-ID crop."""
    return float(score) * ((box[2] - box[0]) * (box[3] - box[1])) ** 0.5


class Tracker:
    """Online single-camera tracker over sampled presence frames. Each frame's YOLOX boxes are associated
    to open tracks by IoU, so one person's run of sightings becomes one track. A track closes when it gets
    no obs for TRACK_GAP seconds of PROCESSED footage (clock = segment ts via tick(), not wall time, so an
    idle sleep never splits a track and a departed person closes on schedule); a long track also emits a
    chunk every TRACK_MAX seconds so identity latency stays bounded. On emit it calls finalize(obs, s0, s1)
    — the ONE place HD is decoded + embedded + inserted, on the track's best shot, instead of per frame."""

    def __init__(self, finalize):
        self.finalize = finalize
        self.tracks = []
        self._next = 1

    def update(self, ts, seg_id, boxes):
        """Associate this frame's boxes (presence px, all at wall-clock ts) to open tracks; greedy by IoU,
        at most one box per track per frame; unmatched boxes open new tracks."""
        used = set()
        for b in boxes:
            box, score = b["box"], float(b["score"])
            best, bj = TRACK_IOU, None
            for ti, tr in enumerate(self.tracks):
                if ti in used or ts - tr["last_ts"] > TRACK_GAP:
                    continue
                v = _iou(box, tr["last_box"])
                if v >= best:
                    best, bj = v, ti
            if bj is None:
                self.tracks.append({"id": self._next, "last_ts": ts, "last_box": box,
                                    "s0": ts, "obs": [(ts, seg_id, box, score)]})
                self._next += 1
                used.add(len(self.tracks) - 1)
            else:
                tr = self.tracks[bj]
                tr["last_ts"], tr["last_box"] = ts, box
                tr["obs"].append((ts, seg_id, box, score))
                used.add(bj)

    def tick(self, now_ts, flush=False):
        """Advance the clock to now_ts (a processed segment's start). Finalize stale tracks; cap long ones
        (emit a chunk, keep the track open for continuity). flush=True emits everything (shutdown/backfill end)."""
        keep = []
        for tr in self.tracks:
            if flush or now_ts - tr["last_ts"] > TRACK_GAP:
                self._emit(tr)
            elif tr["obs"] and tr["obs"][-1][0] - tr["s0"] >= TRACK_MAX:
                self._emit(tr)
                tr["obs"], tr["s0"] = [], tr["last_ts"]    # fresh chunk; track stays open to keep associating
                keep.append(tr)
            else:
                keep.append(tr)
        self.tracks = keep

    def _emit(self, tr):
        if tr["obs"]:
            self.finalize(tr["obs"], tr["obs"][0][0], tr["obs"][-1][0])


def _reid_vec(nvr, ts, seg_id, box, pres_wh):
    """Resolve the best shot's (frame, seg_id, box, embedding) for one track: HD when the re-ID stream
    covers ts, else the SD frame. A decode that comes back EMPTY (under memory pressure ffmpeg can yield 0
    bytes) falls back to the SD best shot rather than losing the sighting. Returns (None, ...) only when
    nothing decodes. NEVER raises except on an Oracle error from embed's caller — decode/embed failures are
    swallowed here so one bad frame can't abort the tracker's tick and strand every other open track."""
    try:
        rframe, rseg, rboxes = _reid_boxes(nvr, ts, [box], pres_wh) if pres_wh else (None, None, None)
        if rframe is not None:
            return rframe, rseg, rboxes[0], detect.embed(rframe, [rboxes[0]])[0]
    except Exception as e:
        print(f"  track re-ID HD decode failed ({e}); SD fallback", flush=True)
    try:                                                  # HD gap OR HD decode failed: SD best shot
        rframe = _decode_at(nvr, seg_id, ts)
        if rframe is not None:
            return rframe, seg_id, box, detect.embed(rframe, [box])[0]
    except Exception as e:
        print(f"  track re-ID SD decode failed ({e}); dropping track", flush=True)
    return None, None, None, None


def _pool_embedding(nvr, obs, k, best_ev):
    """Average the top-k shots' L2-normalized OSNet embeddings into one vector, to denoise single-frame SD
    re-ID (the per-frame blur/pose noise cancels; the identity signal survives). `best_ev` is the already-
    resolved embedding of the top-quality shot (reused so it isn't decoded/embedded twice). Each extra shot
    is embedded from its own SD frame; a shot that fails to decode/embed is skipped. Returns a normalized
    numpy vector; with k<=1 or no extra shots it just returns the best-shot vector normalized."""
    def norm(v):
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    vecs = [norm(best_ev)]
    if k > 1 and len(obs) > 1:
        ranked = sorted(obs, key=lambda o: _quality(o[2], o[3]), reverse=True)
        for ts_i, seg_i, box_i, _score in ranked[1:k]:        # [0] is the best shot, already in vecs
            try:
                fr = _decode_at(nvr, seg_i, ts_i)
                if fr is not None:
                    vecs.append(norm(detect.embed(fr, [box_i])[0]))
            except Exception:
                continue                                      # a bad shot just doesn't contribute
    m = np.mean(vecs, axis=0)
    n = np.linalg.norm(m)
    return m / n if n > 0 else m


def make_finalize(nvr_h, cur, pres_wh_box, stats):
    """Build the Tracker finalize callback: pick the track's best shot, decode HD once there (SD fallback
    on a gap OR a failed HD decode), embed, insert one detection spanning [s0, s1], and save its crop.
    nvr_h[0] is the current segment-DB handle (a holder, since remote mode reopens the snapshot each batch);
    pres_wh_box[0] is the current presence-frame size (constant for SD); stats['n'] counts emitted
    detections for the batch log. finalize never raises on a decode failure (see _reid_vec), so a single
    bad frame can't wedge the tracker into re-decoding the same shot on every following segment."""
    def finalize(obs, s0, s1):
        nvr = nvr_h[0]
        ts, seg_id, box, score = max(obs, key=lambda o: _quality(o[2], o[3]))
        if HD_ALIGN:                                       # option b: time-aligned HD re-ID + confirm in one
            rframe, rseg, rbox, ok = _aligned_hd(nvr, ts, box, pres_wh_box[0])
            if not ok:
                return                                     # HD didn't confirm the best shot (incl. animals): drop
            ev = detect.embed(rframe, [rbox])[0]
        else:
            if HD_CONFIRM and pres_wh_box[0] and not _hd_confirm(nvr, ts, [box], pres_wh_box[0])[0]:
                return                                     # HD didn't re-confirm the track's best shot: drop
            rframe, rseg, rbox, ev = _reid_vec(nvr, ts, seg_id, box, pres_wh_box[0])
            if rframe is None:
                return
            if REID_SHOTS > 1:                             # pool top-N shots -> one denoised embedding (crop/
                ev = _pool_embedding(nvr, obs, REID_SHOTS, ev)   # focus/nav stay the best shot; only ev pools
        did = _insert_detection(cur, ts, rseg, rbox, score, ev, s0, s1,    # DB error propagates -> reconnect
                                focus=detect.focus_score(rframe, rbox),
                                boxfrac=_boxfrac(rbox, (rframe.shape[1], rframe.shape[0])))
        try:
            _save_crop(rframe, rbox, did)
        except Exception as e:
            print(f"  crop save failed for det {did} ({e})", flush=True)
        stats["n"] += 1
    return finalize


def _cluster(cur):
    """Group the still-UNKNOWN detections (identity_id IS NULL) by re-ID similarity so the UI shows one
    tile per person instead of a tile per frame. AVERAGE-LINKAGE agglomerative over L2-normalized OSNet
    embeddings: repeatedly merge the two groups with the highest MEAN pairwise cosine similarity, while
    that mean is >= (1 - CLUSTER_DIST). Group id = smallest member id. Returns the number of groups.

    Why average-linkage and not connected-components (single-linkage): single-linkage merges two groups
    as soon as ANY one pair is within threshold, so a single borderline/ambiguous crop bridges two
    different people and the link is transitive — on low-res SD embeddings that CHAINS dozens of distinct
    people into one giant mixed group (measured: 89 detections, median pairwise sim 0.53, all chained at
    0.72). Average-linkage needs the groups to be similar ON AVERAGE, so a lone bridge can't fuse them;
    the result is many smaller but PURE groups, which the UI can actually tag (a mixed group can't be).

    Only confirmed=0 unknowns are (re)grouped. A detection the user manually split out (match.unassign:
    confirmed=1, cluster_id=id) is PINNED — excluded here and keeps its own cluster_id, so the split
    sticks across passes instead of being re-merged by embedding.

    O(n^3) worst case but numpy-vectorized per merge; the unknown pool is self-limiting (rows get tagged
    or dismissed), so n is hundreds in practice."""
    cur.execute("SELECT id, embedding FROM detections WHERE identity_id IS NULL "
                "AND (confirmed = 0 OR confirmed IS NULL) AND embedding IS NOT NULL ORDER BY id")
    rows = cur.fetchall()
    if not rows:
        return 0
    ids = [int(r[0]) for r in rows]
    E = np.asarray([np.asarray(r[1], dtype=np.float32) for r in rows])  # n x 512
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9                # defensive renorm -> cosine via dot
    n = len(ids)
    thr = 1.0 - CLUSTER_DIST
    sim = (E @ E.T).astype(np.float64)
    S = sim.copy()                              # S[a,b] = SUM of member-member sims between groups a and b
    size = np.ones(n)
    alive = np.ones(n, dtype=bool)
    members = [[i] for i in range(n)]
    while True:
        avg = S / np.outer(size, size)          # mean pairwise sim between every pair of live groups
        np.fill_diagonal(avg, -1.0)
        avg[~alive] = -1.0
        avg[:, ~alive] = -1.0
        a, b = np.unravel_index(int(np.argmax(avg)), avg.shape)
        if avg[a, b] < thr:
            break                               # no pair of groups is similar enough on average
        if b < a:
            a, b = b, a
        S[a] += S[b]                            # fold group b into a (sums add; sizes add)
        S[:, a] += S[:, b]
        size[a] += size[b]
        alive[b] = False
        members[a] += members[b]
        members[b] = []
    updates = []
    ngroups = 0
    for m in members:
        if not m:
            continue
        ngroups += 1
        cid = min(ids[k] for k in m)            # stable group id: smallest detection id in the group
        updates.extend((cid, ids[k]) for k in m)
    cur.executemany("UPDATE detections SET cluster_id = :1 WHERE id = :2", updates)
    return ngroups


def cluster_unknowns():
    """CLI/entry wrapper for _cluster with its own connection."""
    orc = adb.connect(); cur = orc.cursor()
    n = _cluster(cur); orc.commit(); orc.close()
    print(f"clustered unknown detections into {n} group(s) (thr {CLUSTER_DIST})", flush=True)
    return n


def _due(nvr, last):
    rows = nvr.execute("SELECT id FROM segments WHERE id > ? AND stream = ? ORDER BY id LIMIT ?",
                       (last, STREAM, BATCH * STRIDE)).fetchall()
    return [r[0] for r in rows if r[0] % STRIDE == 0]


def run():
    nvr_h = [ro_nvr()]                                    # holder: remote mode reopens the snapshot per batch
    gate = MotionGate(BG_PATH, min_px=GATE_MIN_PX, k=GATE_K) if GATE_ENABLED else None
    if gate is not None:
        print(f"motion gate ON (min_px={GATE_MIN_PX}, k={GATE_K}): YOLOX runs only on foreground frames", flush=True)
    if REID_STREAM:
        print(f"presence on '{STREAM}', re-ID crops from '{REID_STREAM}' (HD decoded only on a hit)", flush=True)
    if HD_ALIGN:
        print(f"HD align+confirm ON: calibrate HD<->SD offset on '{HD_CONFIRM_STREAM}' over "
              f"[{HD_ALIGN_LO:+g},{HD_ALIGN_HI:+g}]s @ {HD_ALIGN_STEP}s, match IoU>={HD_ALIGN_IOU}, "
              f"recal after {HD_MISS_MAX} misses (>={HD_RECAL_MIN}s apart)", flush=True)
    elif HD_CONFIRM:
        print(f"HD-confirm ON: validate on '{HD_CONFIRM_STREAM}', keep an SD box only if a HD person-box "
              f"overlaps (IoU>={HD_CONFIRM_IOU}); crops stay SD-aligned", flush=True)
    if TRACK:
        shots = f"{REID_SHOTS}-shot pooled" if REID_SHOTS > 1 else "best-shot"
        print(f"track mode ON: IoU>={TRACK_IOU}, close after {TRACK_GAP}s idle, chunk cap {TRACK_MAX}s "
              f"(re-ID on {REID_STREAM or STREAM}, {shots} per track)", flush=True)
    if DB_IMMUTABLE:
        print(f"remote mode: segments from snapshot '{DB_PATH}' (reopened each batch)", flush=True)
    while True:                                           # reconnect loop: a DB error drops to here
        orc = None
        try:
            orc = adb.connect(); cur = orc.cursor()
            # per-connection so a reconnect restarts cleanly (in-flight tracks are dropped, re-acquired live)
            stats = {"n": 0}; pres_wh_box = [None]
            if HD_ALIGN:
                _hd_reset()                                   # recalibrate the HD<->SD offset after a reconnect
            tracker = Tracker(make_finalize(nvr_h, cur, pres_wh_box, stats)) if TRACK else None
            newest = nvr_h[0].execute("SELECT max(id) FROM segments WHERE stream = ?", (STREAM,)).fetchone()[0] or 0
            last = get_state(cur, "last_seg_id", newest)  # first run: start at the live edge
            print(f"detect poller: starting at seg {last} (newest {newest}), stride {STRIDE}", flush=True)
            while True:
                if DB_IMMUTABLE:                          # pick up the newest snapshot (the file was replaced)
                    try: nvr_h[0].close()
                    except Exception: pass
                    nvr_h[0] = ro_nvr()
                nvr = nvr_h[0]
                ids = _due(nvr, last)
                if not ids:
                    time.sleep(IDLE); continue
                stats["n"] = 0
                for sid in ids:
                    try:
                        if tracker is not None:
                            start, pres_wh, frames = _presence(nvr, sid, gate)
                            if start is not None:
                                pres_wh_box[0] = pres_wh
                                for ts, frame, boxes in frames:
                                    tracker.update(ts, sid, boxes)
                                tracker.tick(start)        # advance clock; close/cap tracks, emitting detections
                        else:
                            stats["n"] += process_segment(nvr, cur, sid, gate)
                    except oracledb.DatabaseError:
                        raise                              # connection trouble -> reconnect, keep `last`
                    except Exception as e:
                        print(f"  seg {sid} decode/skip: {e}", flush=True)  # bad frame: skip it
                    last = sid
                set_state(cur, "last_seg_id", last); orc.commit()  # inserts+cursor advance commit together
                if gate is not None:
                    gate.save()                            # persist the background model for restart recovery
                if stats["n"]:
                    print(f"  up to seg {last}: {stats['n']} person-detection(s) stored", flush=True)
                    try:
                        g = _cluster(cur); orc.commit()
                        print(f"    grouped unknowns into {g} cluster(s)", flush=True)
                    except oracledb.DatabaseError:
                        raise                                       # connection trouble -> reconnect
                    except Exception as e:
                        print(f"    cluster skip: {e}", flush=True)
        except oracledb.DatabaseError as e:
            print(f"oracle error; reconnecting in {RETRY}s: {e}", flush=True)
            time.sleep(RETRY)
        finally:
            if orc is not None:
                try: orc.close()
                except Exception: pass


def backfill(a, b):
    orc = adb.connect(); cur = orc.cursor(); nvr = ro_nvr()
    # Ephemeral gate (path=None): a backfill over an arbitrary range must NOT clobber the live poller's
    # persisted background model. It warms up over the range's own first frames.
    gate = MotionGate(None, min_px=GATE_MIN_PX, k=GATE_K) if GATE_ENABLED else None
    ids = [r[0] for r in nvr.execute(
        "SELECT id FROM segments WHERE id BETWEEN ? AND ? AND stream = ? ORDER BY id",
        (a, b, STREAM)).fetchall() if r[0] % STRIDE == 0]
    print(f"backfill {len(ids)} segments ({a}..{b}, stride {STRIDE})"
          + (", track mode" if TRACK else "")
          + (f", motion gate min_px={GATE_MIN_PX}" if gate is not None else ""), flush=True)
    stats = {"n": 0}; pres_wh_box = [None]
    tracker = Tracker(make_finalize([nvr], cur, pres_wh_box, stats)) if TRACK else None
    t0 = time.time()
    for i, sid in enumerate(ids):
        try:
            if tracker is not None:
                start, pres_wh, frames = _presence(nvr, sid, gate)
                if start is not None:
                    pres_wh_box[0] = pres_wh
                    for ts, frame, boxes in frames:
                        tracker.update(ts, sid, boxes)
                    tracker.tick(start)
            else:
                stats["n"] += process_segment(nvr, cur, sid, gate)
        except Exception as e:
            print(f"  seg {sid} error: {e}", flush=True)
        if i % 10 == 9:
            orc.commit(); print(f"  {i+1}/{len(ids)} done, {stats['n']} detections, "
                                f"{(time.time()-t0)/(i+1):.1f}s/seg", flush=True)
    if tracker is not None:
        tracker.tick(0, flush=True)                       # emit any still-open tracks at the end of the range
    orc.commit()
    print(f"backfill done: {stats['n']} detections from {len(ids)} segments in {time.time()-t0:.0f}s", flush=True)


def backfill_crops():
    """One-shot: make the static crop for any stored detection that doesn't have one yet and whose
    segment is still on disk. Pure read of segments + Oracle; safe to re-run."""
    orc = adb.connect(); cur = orc.cursor(); nvr = ro_nvr()
    cur.execute("SELECT id, seg_id, box FROM detections ORDER BY id")
    rows = cur.fetchall()
    made = skip = have = 0
    for det_id, seg_id, box in rows:
        det_id = int(det_id)
        if os.path.exists(os.path.join(CROP_DIR, f"{det_id}.jpg")):
            have += 1; continue
        r = nvr.execute("SELECT path, init FROM segments WHERE id = ?", (int(seg_id),)).fetchone()
        if not r or not box:
            skip += 1; continue
        try:
            frame = detect.decode_frame(os.path.join(DATA, r[1]), os.path.join(DATA, r[0]))
            _save_crop(frame, [int(v) for v in box.split(",")], det_id); made += 1
        except Exception as e:
            print(f"  seg {seg_id} skip: {e}", flush=True); skip += 1
    print(f"crop backfill: {made} made, {have} already had one, {skip} skipped (no segment)", flush=True)
    orc.close()


def _sd_box(box_str, sd_wh):
    """The stored box mapped to SD-frame px. A detection re-ID'd on HD stored an HD box (coords up to
    2304x1296); one re-ID'd on SD already stored an SD box. Detect which by whether it fits the SD frame."""
    b = [float(v) for v in box_str.split(",")]
    sw, sh = sd_wh
    if b[2] > sw or b[3] > sh:                             # stored in HD px -> scale down to SD
        b = [b[0] * sw / 2304.0, b[1] * sh / 1296.0, b[2] * sw / 2304.0, b[3] * sh / 1296.0]
    return b


def reembed_sd():
    """Re-crop + re-embed the detections whose re-ID landed on the drifting HD stream, using the ALIGNED
    SD frame at their own ts instead. Only rows with a surviving cam1-sd segment covering ts are touched
    (the rest predate the sub-stream and were HD-presence+HD-reID, already aligned). Identity labels,
    ts and the presence span are preserved — only embedding, box, seg_id and the crop jpeg change — so a
    person's tag survives; only the (previously wrong) crop and vector are corrected. Re-clusters the
    unknowns at the end. Idempotent: re-running reprocesses from SD again to the same result."""
    orc = adb.connect(); cur = orc.cursor(); nvr = ro_nvr()
    span = nvr.execute("SELECT MIN(start), MAX(start) FROM segments WHERE stream = ?", (STREAM,)).fetchone()
    if not span or span[0] is None:
        print("no SD segments in the snapshot; nothing to re-embed", flush=True); return
    sd_min, sd_max = float(span[0]), float(span[1])
    cur.execute("SELECT id, ts, box, identity_id FROM detections WHERE ts BETWEEN :1 AND :2 ORDER BY id",
                (sd_min, sd_max))
    rows = cur.fetchall()
    print(f"re-embed candidates with a surviving SD frame: {len(rows)}", flush=True)
    done = skip = 0
    for det_id, ts, box, ident in rows:
        det_id, ts = int(det_id), float(ts)
        if not box:
            skip += 1; continue
        seg = nvr.execute("SELECT id, start, dur, path, init FROM segments WHERE stream = ? AND start <= ? "
                          "ORDER BY start DESC LIMIT 1", (STREAM, ts)).fetchone()
        if not seg or (seg[2] and ts > float(seg[1]) + float(seg[2])):
            skip += 1; continue                           # ts falls in an SD gap
        sid, start, dur, path, init = seg
        try:
            frame = detect.decode_frame(os.path.join(DATA, init), os.path.join(DATA, path),
                                        max(0.0, ts - float(start)))
            if frame is None or frame.size == 0:
                skip += 1; continue
            sbox = _sd_box(box, (frame.shape[1], frame.shape[0]))
            ev = detect.embed(frame, [sbox])[0]
            emb = array.array("f", [float(x) for x in ev])
            cur.execute("UPDATE detections SET embedding = :1, box = :2, seg_id = :3 WHERE id = :4",
                        (emb, ",".join(str(int(v)) for v in sbox), int(sid), det_id))
            _save_crop(frame, sbox, det_id)
            done += 1
        except Exception as e:
            print(f"  det {det_id} skip: {e}", flush=True); skip += 1
        if (done + skip) % 50 == 0:
            orc.commit(); print(f"  {done+skip}/{len(rows)} ({done} re-embedded)", flush=True)
    orc.commit()
    g = _cluster(cur); orc.commit()
    print(f"re-embed done: {done} corrected, {skip} skipped; re-clustered unknowns into {g} group(s)", flush=True)
    orc.close()


def backfill_quality():
    """Fill the crop-quality signals (`focus` sharpness, `boxfrac` zoom) for stored detections that lack
    them: focus from the saved crop jpeg, boxfrac from the stored box vs its frame size (HD 2304x1296 when
    the box exceeds the SD frame, else SD 800x448). Lets the training gate apply to history. Safe to re-run."""
    from PIL import Image
    orc = adb.connect(); cur = orc.cursor()
    cur.execute("SELECT id, box FROM detections WHERE focus IS NULL OR boxfrac IS NULL ORDER BY id")
    rows = cur.fetchall()
    print(f"quality backfill: {len(rows)} rows missing focus/boxfrac", flush=True)
    done = skip = 0
    for did, box in rows:
        did = int(did)
        p = os.path.join(CROP_DIR, f"{did}.jpg")
        if not os.path.exists(p) or not box:
            skip += 1; continue
        try:
            a = np.asarray(Image.open(p).convert("RGB"))
            foc = float(detect.focus_score(a))
            b = [float(v) for v in box.split(",")]
            wh = (2304, 1296) if (b[2] > 800 or b[3] > 448) else (800, 448)
            cur.execute("UPDATE detections SET focus = :1, boxfrac = :2 WHERE id = :3",
                        (foc, float(_boxfrac(b, wh)), did))
            done += 1
        except Exception as e:
            print(f"  det {did} skip: {e}", flush=True); skip += 1
        if (done + skip) % 200 == 0:
            orc.commit(); print(f"  {done+skip}/{len(rows)}", flush=True)
    orc.commit()
    print(f"quality backfill done: {done} set, {skip} skipped (no crop)", flush=True)
    orc.close()


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "run":
        run()
    elif len(sys.argv) >= 2 and sys.argv[1] == "backfill-quality":
        backfill_quality()
    elif len(sys.argv) >= 4 and sys.argv[1] == "backfill":
        backfill(int(sys.argv[2]), int(sys.argv[3]))
    elif len(sys.argv) >= 2 and sys.argv[1] == "backfill-crops":
        backfill_crops()
    elif len(sys.argv) >= 2 and sys.argv[1] == "reembed-sd":
        reembed_sd()
    elif len(sys.argv) >= 2 and sys.argv[1] == "cluster":
        cluster_unknowns()
    else:
        sys.exit(__doc__)
