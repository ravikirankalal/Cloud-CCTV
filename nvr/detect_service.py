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
STRIDE = int(os.environ.get("DET_STRIDE", "2"))          # sample every Nth segment
FRAMES = int(os.environ.get("DET_FRAMES", "1"))          # frames/segment at the LIVE edge (evenly spaced)
FRAMES_BACKFILL = int(os.environ.get("DET_FRAMES_BACKFILL", "1"))  # coarser rate while catching up on old footage
BACKFILL_AGE = float(os.environ.get("DET_BACKFILL_AGE", "120"))    # a segment older than this (s) is backfill, not live
BATCH = int(os.environ.get("DET_BATCH", "25"))           # segments per loop before committing/sleeping
IDLE = int(os.environ.get("DET_IDLE", "15"))             # sleep seconds when caught up
RETRY = int(os.environ.get("DET_RETRY", "30"))           # sleep before reconnecting to Oracle after a DB error
MATCH_DIST = float(os.environ.get("DET_MATCH", "0.28"))  # cosine distance < this => same known person
CLUSTER_DIST = float(os.environ.get("DET_CLUSTER", "0.28"))  # cosine distance to group unknowns together
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
    gets DET_FRAMES frames for fine temporal resolution; an OLD segment (we're catching up on a backlog)
    gets the coarser DET_FRAMES_BACKFILL so the poller drains fast and returns to live. With n>1 we sample
    that many frames evenly across the segment (single ffmpeg pass; see detect.decode_frames). Each frame's
    ts is its own wall-clock time (start + i*dur/n)."""
    row = nvr.execute("SELECT start, path, init, dur FROM segments WHERE id = ?", (seg_id,)).fetchone()
    if not row:
        return None, None, []
    start, path, init, dur = row
    n = FRAMES if (time.time() - float(start)) <= BACKFILL_AGE else FRAMES_BACKFILL
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


def _insert_detection(cur, ts, seg_id, box, score, ev, ts_start, ts_end):
    """Insert one detection row (the caller saves its crop) and return its id. Shared by the per-frame and
    the track paths so the row shape — including the [ts_start, ts_end] presence span — stays identical.
    ts is the representative/best-shot time to navigate to; the span is when the person was actually seen."""
    emb = array.array("f", [float(x) for x in ev])
    ident = match_known(cur, emb)
    idv = cur.var(oracledb.NUMBER)
    cur.execute("""INSERT INTO detections (ts, seg_id, box, score, embedding, identity_id, ts_start, ts_end)
                   VALUES (:1, :2, :3, :4, :5, :6, :7, :8) RETURNING id INTO :9""",
                (ts, seg_id, ",".join(str(int(v)) for v in box),
                 float(score), emb, ident, float(ts_start), float(ts_end), idv))
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
        sd_boxes = [b["box"] for b in boxes]
        rframe, rseg, rboxes = _reid_boxes(nvr, ts, sd_boxes, pres_wh)
        if rframe is None:                                # no HD there: re-ID on the presence frame
            rframe, rseg, rboxes = frame, seg_id, sd_boxes
        for b, hbox, ev in zip(boxes, rboxes, detect.embed(rframe, rboxes)):
            did = _insert_detection(cur, ts, rseg, hbox, b["score"], ev, ts, ts)
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
        rframe, rseg, rbox, ev = _reid_vec(nvr, ts, seg_id, box, pres_wh_box[0])
        if rframe is None:
            return
        did = _insert_detection(cur, ts, rseg, rbox, score, ev, s0, s1)   # DB error propagates -> reconnect
        try:
            _save_crop(rframe, rbox, did)
        except Exception as e:
            print(f"  crop save failed for det {did} ({e})", flush=True)
        stats["n"] += 1
    return finalize


def _cluster(cur):
    """Group the still-UNKNOWN detections (identity_id IS NULL) by re-ID similarity so the UI can show
    one tile per person instead of a tile per frame. CONNECTED COMPONENTS over L2-normalized OSNet
    embeddings (pure vector math): two detections within CLUSTER_DIST are linked, and the link is
    transitive, so a person drifting across poses/time collapses into ONE group even when the first
    and last frame aren't themselves within threshold. Group id = smallest member id. Returns the
    number of groups. Light; safe to re-run."""
    cur.execute("SELECT id, embedding FROM detections WHERE identity_id IS NULL ORDER BY id")
    rows = cur.fetchall()
    if not rows:
        return 0
    ids = [int(r[0]) for r in rows]
    E = np.asarray([np.asarray(r[1], dtype=np.float32) for r in rows])  # n x 512, already unit-norm
    n = len(ids)
    sim = E @ E.T                               # cosine similarity (unit vectors); distance = 1 - sim
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    thr = 1.0 - CLUSTER_DIST
    for i in range(n):
        for j in np.where(sim[i] >= thr)[0]:    # neighbors within CLUSTER_DIST
            j = int(j)
            if j > i:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)   # union, keep the lower index as root

    comp = {}
    for i in range(n):
        comp.setdefault(find(i), []).append(i)
    updates = []
    for members in comp.values():
        cid = min(ids[m] for m in members)      # stable group id: smallest detection id in the group
        updates.extend((cid, ids[m]) for m in members)
    cur.executemany("UPDATE detections SET cluster_id = :1 WHERE id = :2", updates)
    return len(comp)


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
    if TRACK:
        print(f"track mode ON: IoU>={TRACK_IOU}, close after {TRACK_GAP}s idle, chunk cap {TRACK_MAX}s "
              f"(one HD re-ID per track)", flush=True)
    if DB_IMMUTABLE:
        print(f"remote mode: segments from snapshot '{DB_PATH}' (reopened each batch)", flush=True)
    while True:                                           # reconnect loop: a DB error drops to here
        orc = None
        try:
            orc = adb.connect(); cur = orc.cursor()
            # per-connection so a reconnect restarts cleanly (in-flight tracks are dropped, re-acquired live)
            stats = {"n": 0}; pres_wh_box = [None]
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


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "run":
        run()
    elif len(sys.argv) >= 4 and sys.argv[1] == "backfill":
        backfill(int(sys.argv[2]), int(sys.argv[3]))
    elif len(sys.argv) >= 2 and sys.argv[1] == "backfill-crops":
        backfill_crops()
    elif len(sys.argv) >= 2 and sys.argv[1] == "cluster":
        cluster_unknowns()
    else:
        sys.exit(__doc__)
