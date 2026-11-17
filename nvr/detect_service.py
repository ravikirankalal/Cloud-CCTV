#!/usr/bin/env python3
# Person-detection poller. Samples recorded segments, detects + embeds people, assigns each to a
# known identity by vector distance (or leaves it unknown), and writes to the Oracle 23ai vector DB.
#
# DECOUPLED by design: reads nvr.db READ-ONLY (immutable), writes ONLY to Oracle, imports no recorder
# code, opens no recorder sockets, never touches the live path. It can lag, stop, or restart with zero
# effect on recording. Meant to run under systemd with MemoryMax + Nice on this small box.
#
#   python detect_service.py run                          # live poller: process new segments forever
#   python detect_service.py backfill <from_id> <to_id>   # process an explicit id range (test/catch-up)
import os, sys, time, sqlite3, array, importlib.util
import oracledb

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("NVR_DATA", "/home/ubuntu/nvr-data")
STREAM = os.environ.get("NVR_STREAM", "cam1")
STRIDE = int(os.environ.get("DET_STRIDE", "2"))          # sample every Nth segment (1 frame each)
BATCH = int(os.environ.get("DET_BATCH", "25"))           # segments per loop before committing/sleeping
IDLE = int(os.environ.get("DET_IDLE", "15"))             # sleep seconds when caught up
RETRY = int(os.environ.get("DET_RETRY", "30"))           # sleep before reconnecting to Oracle after a DB error
MATCH_DIST = float(os.environ.get("DET_MATCH", "0.28"))  # cosine distance < this => same known person
CROP_DIR = os.path.join(DATA, "crops")                   # one static person-crop jpeg per detection

os.makedirs(CROP_DIR, exist_ok=True)

sys.path.insert(0, HERE)
import adb                                                # shared Oracle connection
_spec = importlib.util.spec_from_file_location("detect", os.path.join(HERE, "detect.py"))
detect = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(detect)


def ro_nvr():
    return sqlite3.connect(f"file:{os.path.join(DATA, 'nvr.db')}?mode=ro&immutable=1", uri=True)


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


def process_segment(nvr, cur, seg_id):
    """Detect+embed one segment's frame and insert its people, saving each one's crop jpeg while the
    frame is in hand. Returns the number of people stored. DB errors propagate (to trigger a
    reconnect); decode errors are the caller's to swallow."""
    row = nvr.execute("SELECT start, path, init FROM segments WHERE id = ?", (seg_id,)).fetchone()
    if not row:
        return 0
    start, path, init = row
    frame = detect.decode_frame(os.path.join(DATA, init), os.path.join(DATA, path))
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


def _due(nvr, last):
    rows = nvr.execute("SELECT id FROM segments WHERE id > ? AND stream = ? ORDER BY id LIMIT ?",
                       (last, STREAM, BATCH * STRIDE)).fetchall()
    return [r[0] for r in rows if r[0] % STRIDE == 0]


def run():
    nvr = ro_nvr()
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
                        n += process_segment(nvr, cur, sid)
                    except oracledb.DatabaseError:
                        raise                              # connection trouble -> reconnect, keep `last`
                    except Exception as e:
                        print(f"  seg {sid} decode/skip: {e}", flush=True)  # bad frame: skip it
                    last = sid
                set_state(cur, "last_seg_id", last); orc.commit()  # inserts+cursor advance commit together
                if n:
                    print(f"  up to seg {last}: {n} person-detection(s) stored", flush=True)
        except oracledb.DatabaseError as e:
            print(f"oracle error; reconnecting in {RETRY}s: {e}", flush=True)
            time.sleep(RETRY)
        finally:
            if orc is not None:
                try: orc.close()
                except Exception: pass


def backfill(a, b):
    orc = adb.connect(); cur = orc.cursor(); nvr = ro_nvr()
    ids = [r[0] for r in nvr.execute(
        "SELECT id FROM segments WHERE id BETWEEN ? AND ? AND stream = ? ORDER BY id",
        (a, b, STREAM)).fetchall() if r[0] % STRIDE == 0]
    print(f"backfill {len(ids)} segments ({a}..{b}, stride {STRIDE})", flush=True)
    total = 0; t0 = time.time()
    for i, sid in enumerate(ids):
        try:
            total += process_segment(nvr, cur, sid)
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
    else:
        sys.exit(__doc__)
