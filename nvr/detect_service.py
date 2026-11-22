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
FRAMES = int(os.environ.get("DET_FRAMES", "1"))          # frames sampled per segment (evenly spaced)
BATCH = int(os.environ.get("DET_BATCH", "25"))           # segments per loop before committing/sleeping
IDLE = int(os.environ.get("DET_IDLE", "15"))             # sleep seconds when caught up
RETRY = int(os.environ.get("DET_RETRY", "30"))           # sleep before reconnecting to Oracle after a DB error
MATCH_DIST = float(os.environ.get("DET_MATCH", "0.28"))  # cosine distance < this => same known person
CLUSTER_DIST = float(os.environ.get("DET_CLUSTER", "0.28"))  # cosine distance to group unknowns together
CROP_DIR = os.path.join(DATA, "crops")                   # one static person-crop jpeg per detection
BG_PATH = os.path.join(DATA, "detect_bg.npz")            # persisted motion-gate background model
GATE_ENABLED = os.environ.get("DET_GATE", "1") != "0"    # skip YOLOX on empty (no-foreground) frames
GATE_MIN_PX = int(os.environ.get("DET_GATE_MIN_PX", "140"))  # foreground blob px (160-wide grid) to trigger
GATE_K = float(os.environ.get("DET_GATE_K", "2.5"))      # per-pixel std multiple for the background model

os.makedirs(CROP_DIR, exist_ok=True)

sys.path.insert(0, HERE)
import adb                                                # shared Oracle connection
import match                                              # prototype gallery + nearest-proto live match
from motion_gate import MotionGate                        # background-subtraction gate (skips YOLOX on empty frames)
_spec = importlib.util.spec_from_file_location("detect", os.path.join(HERE, "detect.py"))
detect = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(detect)


def ro_nvr():
    # mode=ro (NOT immutable): the recorder writes nvr.db in WAL mode, so new segments live in the -wal
    # file until a checkpoint. immutable=1 reads only the main db file and ignores the WAL, freezing the
    # poller at the last checkpoint (it never sees live segments). A plain read-only handle reads the WAL
    # and sees the live edge, and WAL still lets the recorder write concurrently without blocking.
    return sqlite3.connect(f"file:{os.path.join(DATA, 'nvr.db')}?mode=ro", uri=True)


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


def process_segment(nvr, cur, seg_id, gate=None):
    """Detect people in one presence segment's frame(s) and insert them, saving each one's crop while the
    frame is in hand. Returns the number of people stored. DB errors propagate (to trigger a reconnect);
    decode errors are the caller's to swallow.

    PRESENCE vs RE-ID (DET_REID_STREAM): YOLOX presence runs on the sampled frame of this (presence)
    segment — cheap when that is the low-res SD sub-stream. When DET_REID_STREAM is set, the OSNet re-ID
    embedding + crop come from the full-res HD frame at the same wall-clock ts (box scaled SD->HD), and the
    detection is stored against the HD seg_id so it still navigates to the HD footage. If HD has no frame
    there (a gap), we re-ID on the presence frame and store its seg_id, so an SD-only sighting is still
    kept. With DET_REID_STREAM empty the presence frame is also the re-ID frame (the original HD-only path).

    The optional `gate` is a MotionGate: it is fed EVERY presence frame in order (to keep its background
    model current) and, on a frame with no foreground blob, we skip the YOLOX + re-ID pass entirely.

    With DET_FRAMES>1 we sample that many frames evenly across the segment (single ffmpeg pass; see
    detect.decode_frames). Each detection's ts is the frame's own wall-clock time (start + i·dur/FRAMES)."""
    row = nvr.execute("SELECT start, path, init, dur FROM segments WHERE id = ?", (seg_id,)).fetchone()
    if not row:
        return 0
    start, path, init, dur = row
    init_abs = os.path.join(DATA, init)
    frames = detect.decode_frames(init_abs, os.path.join(DATA, path), FRAMES, dur)
    pres_wh = detect._frame_dims(init_abs) if REID_STREAM else None   # presence-frame size, for SD->HD scaling
    stored = 0
    for i, frame in enumerate(frames):
        if gate is not None and not gate.check(frame)[0]:
            continue                                      # empty scene: don't pay for YOLOX
        ts = float(start) + (i * float(dur) / FRAMES if dur and FRAMES > 1 else 0.0)
        boxes = detect.detect_in(frame)                   # YOLOX presence (on the SD frame when so configured)
        if not boxes:
            continue
        rframe, rseg = frame, seg_id                      # re-ID frame/seg: the presence frame by default
        rboxes = [b["box"] for b in boxes]
        if REID_STREAM:
            hd_seg, hd_frame = _reid_frame(nvr, ts)
            if hd_frame is not None:                       # re-ID + crop from HD; else fall back to SD above
                rframe, rseg = hd_frame, hd_seg
                rboxes = [_scale_box(b["box"], pres_wh, (hd_frame.shape[1], hd_frame.shape[0])) for b in boxes]
        for b, hbox, ev in zip(boxes, rboxes, detect.embed(rframe, rboxes)):
            emb = array.array("f", [float(x) for x in ev])
            ident = match_known(cur, emb)
            idv = cur.var(oracledb.NUMBER)
            cur.execute("""INSERT INTO detections (ts, seg_id, box, score, embedding, identity_id)
                           VALUES (:1, :2, :3, :4, :5, :6) RETURNING id INTO :7""",
                        (ts, rseg, ",".join(str(int(v)) for v in hbox),
                         float(b["score"]), emb, ident, idv))
            _save_crop(rframe, hbox, int(idv.getvalue()[0]))
            stored += 1
    return stored


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
    nvr = ro_nvr()
    gate = MotionGate(BG_PATH, min_px=GATE_MIN_PX, k=GATE_K) if GATE_ENABLED else None
    if gate is not None:
        print(f"motion gate ON (min_px={GATE_MIN_PX}, k={GATE_K}): YOLOX runs only on foreground frames", flush=True)
    if REID_STREAM:
        print(f"presence on '{STREAM}', re-ID crops from '{REID_STREAM}' (HD decoded only on a hit)", flush=True)
    while True:                                           # reconnect loop: a DB error drops to here
        orc = None
        try:
            orc = adb.connect(); cur = orc.cursor()
            newest = nvr.execute("SELECT max(id) FROM segments WHERE stream = ?", (STREAM,)).fetchone()[0] or 0
            last = get_state(cur, "last_seg_id", newest)  # first run: start at the live edge
            print(f"detect poller: starting at seg {last} (newest {newest}), stride {STRIDE}", flush=True)
            while True:
                ids = _due(nvr, last)
                if not ids:
                    time.sleep(IDLE); continue
                n = 0
                for sid in ids:
                    try:
                        n += process_segment(nvr, cur, sid, gate)
                    except oracledb.DatabaseError:
                        raise                              # connection trouble -> reconnect, keep `last`
                    except Exception as e:
                        print(f"  seg {sid} decode/skip: {e}", flush=True)  # bad frame: skip it
                    last = sid
                set_state(cur, "last_seg_id", last); orc.commit()  # inserts+cursor advance commit together
                if gate is not None:
                    gate.save()                            # persist the background model for restart recovery
                if n:
                    print(f"  up to seg {last}: {n} person-detection(s) stored", flush=True)
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
          + (f", motion gate min_px={GATE_MIN_PX}" if gate is not None else ""), flush=True)
    total = 0; t0 = time.time()
    for i, sid in enumerate(ids):
        try:
            total += process_segment(nvr, cur, sid, gate)
        except Exception as e:
            print(f"  seg {sid} error: {e}", flush=True)
        if i % 10 == 9:
            orc.commit(); print(f"  {i+1}/{len(ids)} done, {total} detections, "
                                f"{(time.time()-t0)/(i+1):.1f}s/seg", flush=True)
    orc.commit()
    print(f"backfill done: {total} detections from {len(ids)} frames in {time.time()-t0:.0f}s", flush=True)


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
