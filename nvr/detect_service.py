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
STREAM = os.environ.get("NVR_STREAM", "cam1")
STRIDE = int(os.environ.get("DET_STRIDE", "2"))          # sample every Nth segment (1 frame each)
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
    """Nearest CONFIRMED known detection by cosine distance; return its identity_id if close enough."""
    cur.execute("""SELECT identity_id, VECTOR_DISTANCE(embedding, :1, COSINE) d
                   FROM detections WHERE confirmed = 1 AND identity_id IS NOT NULL
                   ORDER BY d FETCH FIRST 1 ROWS ONLY""", [emb])
    row = cur.fetchone()
    return row[0] if row and row[1] is not None and row[1] < MATCH_DIST else None


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


def process_segment(nvr, cur, seg_id, gate=None):
    """Detect+embed one segment's frame and insert its people, saving each one's crop jpeg while the
    frame is in hand. Returns the number of people stored. DB errors propagate (to trigger a
    reconnect); decode errors are the caller's to swallow.

    The optional `gate` is a MotionGate: it is fed EVERY decoded frame in order (to keep its background
    model current) and, on a frame with no foreground blob, we skip the ~0.9 s YOLOX+re-ID pass entirely.
    Decode is codec-bound and happens either way; the gate's saving is the skipped YOLOX on empty frames."""
    row = nvr.execute("SELECT start, path, init FROM segments WHERE id = ?", (seg_id,)).fetchone()
    if not row:
        return 0
    start, path, init = row
    frame = detect.decode_frame(os.path.join(DATA, init), os.path.join(DATA, path))
    if gate is not None and not gate.check(frame)[0]:
        return 0                                          # empty scene: don't pay for YOLOX
    dets = detect.detect_embed(frame)
    for d in dets:
        emb = array.array("f", d["embedding"])
        ident = match_known(cur, emb)
        idv = cur.var(oracledb.NUMBER)
        cur.execute("""INSERT INTO detections (ts, seg_id, box, score, embedding, identity_id)
                       VALUES (:1, :2, :3, :4, :5, :6) RETURNING id INTO :7""",
                    (float(start), seg_id, ",".join(str(int(v)) for v in d["box"]),
                     float(d["score"]), emb, ident, idv))
        _save_crop(frame, d["box"], int(idv.getvalue()[0]))
    return len(dets)


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
