#!/usr/bin/env python3
# Person-detection poller. Samples recorded segments, detects + embeds people, assigns each to a
# known identity by vector distance (or leaves it unknown), and writes to the Oracle 23ai vector DB.
#
# DECOUPLED by design: reads nvr.db READ-ONLY (immutable), writes ONLY to Oracle, imports no recorder
# code, opens no recorder sockets, never touches the live path. It can lag, stop, or restart with zero
# effect on recording. Meant to run under systemd with MemoryMax + Nice on this small box.
#
#   python detect_service.py run                      # live poller: process new segments forever
#   python detect_service.py backfill <from_id> <to_id>   # process an explicit id range (test/catch-up)
import os, sys, time, sqlite3, array, importlib.util
import oracledb

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("NVR_DATA", "/home/ubuntu/nvr-data")
WDIR = os.environ.get("ADB_WALLET", "/home/ubuntu/nvr/adb-wallet")
DSN = os.environ.get("ADB_DSN", "nvrdetect_tp")
STREAM = os.environ.get("NVR_STREAM", "cam1")
STRIDE = int(os.environ.get("DET_STRIDE", "2"))        # sample every Nth segment (1 frame each)
BATCH = int(os.environ.get("DET_BATCH", "25"))         # segments per loop before committing/sleeping
IDLE = int(os.environ.get("DET_IDLE", "15"))           # sleep seconds when caught up
MATCH_DIST = float(os.environ.get("DET_MATCH", "0.28"))  # cosine distance < this => same known person

spec = importlib.util.spec_from_file_location("detect", os.path.join(HERE, "detect.py"))
detect = importlib.util.module_from_spec(spec); spec.loader.exec_module(detect)


def connect():
    pw = open(os.path.join(WDIR, "admin.pw")).read().strip()
    wpw = open(os.path.join(WDIR, "wallet.pw")).read().strip()
    return oracledb.connect(user="ADMIN", password=pw, dsn=DSN,
                            config_dir=WDIR, wallet_location=WDIR, wallet_password=wpw)


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


def process_segment(nvr, cur, seg_id):
    row = nvr.execute("SELECT start, path, init FROM segments WHERE id = ?", (seg_id,)).fetchone()
    if not row:
        return 0
    start, path, init = row
    dets = detect.detect_embed_frame(os.path.join(DATA, init), os.path.join(DATA, path))
    for d in dets:
        emb = array.array("f", d["embedding"])
        ident = match_known(cur, emb)
        cur.execute("""INSERT INTO detections (ts, seg_id, box, score, embedding, identity_id)
                       VALUES (:1, :2, :3, :4, :5, :6)""",
                    (float(start), seg_id, ",".join(str(int(v)) for v in d["box"]),
                     float(d["score"]), emb, ident))
    return len(dets)


def run():
    orc = connect(); cur = orc.cursor()
    nvr = ro_nvr()
    newest = nvr.execute("SELECT max(id) FROM segments WHERE stream = ?", (STREAM,)).fetchone()[0] or 0
    last = get_state(cur, "last_seg_id", newest)       # first run: start at the live edge
    print(f"detect poller: starting at seg {last} (newest {newest}), stride {STRIDE}", flush=True)
    while True:
        rows = nvr.execute("SELECT id FROM segments WHERE id > ? AND stream = ? ORDER BY id LIMIT ?",
                           (last, STREAM, BATCH * STRIDE)).fetchall()
        ids = [r[0] for r in rows if r[0] % STRIDE == 0]
        if not ids:
            time.sleep(IDLE)
            continue
        n = 0
        for sid in ids:
            try:
                n += process_segment(nvr, cur, sid)
            except Exception as e:
                print(f"  seg {sid} error: {e}", flush=True)
            last = sid
        set_state(cur, "last_seg_id", last); orc.commit()
        if n:
            print(f"  up to seg {last}: {n} person-detection(s) stored", flush=True)


def backfill(a, b):
    orc = connect(); cur = orc.cursor(); nvr = ro_nvr()
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


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "run":
        run()
    elif len(sys.argv) >= 4 and sys.argv[1] == "backfill":
        backfill(int(sys.argv[2]), int(sys.argv[3]))
    else:
        sys.exit(__doc__)
