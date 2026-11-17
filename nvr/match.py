#!/usr/bin/env python3
# The LIGHT stage of person detection: pure vector operations over ALREADY-extracted embeddings.
# No decode, no model inference — just Oracle 23ai VECTOR_DISTANCE over what Stage 1 already stored.
# This is what the feedback loop and the (future) tagging UI call.
#
# Stage 1 (heavy, once/frame): detect.py/detect_service.py -> detections{ts,seg_id,box,score,embedding}.
# Stage 2 (light, on demand): tag() records a user's label; rematch() re-assigns unconfirmed detections
# to the nearest CONFIRMED known person. Re-running it after each new tag is cheap and never touches
# the camera, the recorder, or the models.
import os, sys
import oracledb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adb

DEFAULT_THR = float(os.environ.get("DET_MATCH", "0.28"))   # cosine distance < this => same person


def get_or_create_identity(cur, name):
    cur.execute("SELECT id FROM identities WHERE name = :1", [name])
    row = cur.fetchone()
    if row:
        return int(row[0])
    idv = cur.var(oracledb.NUMBER)
    cur.execute("INSERT INTO identities (name, kind) VALUES (:1, 'known') RETURNING id INTO :2", [name, idv])
    return int(idv.getvalue()[0])


def tag(cur, detection_id, name):
    """User feedback: label a detection as a named (known) person. 'confirmed' marks it ground truth
    that rematch() will never overwrite and will match future/past detections against."""
    ident = get_or_create_identity(cur, name)
    cur.execute("UPDATE detections SET identity_id = :1, confirmed = 1 WHERE id = :2", [ident, detection_id])
    cur.execute("INSERT INTO labels (detection_id, identity_id, action) VALUES (:1, :2, 'name')",
                [detection_id, ident])
    return ident


def tag_cluster(cur, detection_id, name):
    """Tag a whole duplicate-group at once: assign `name`'s identity to every still-unconfirmed
    detection sharing this one's cluster_id, and confirm the tapped detection as ground truth.
    Falls back to tagging just this detection when it isn't clustered."""
    ident = get_or_create_identity(cur, name)
    cur.execute("SELECT cluster_id FROM detections WHERE id = :1", [detection_id])
    row = cur.fetchone()
    cid = row[0] if row else None
    if cid is not None:
        cur.execute("UPDATE detections SET identity_id = :1 WHERE cluster_id = :2 AND confirmed = 0",
                    [ident, cid])
    cur.execute("UPDATE detections SET identity_id = :1, confirmed = 1 WHERE id = :2", [ident, detection_id])
    cur.execute("INSERT INTO labels (detection_id, identity_id, action) VALUES (:1, :2, 'name')",
                [detection_id, ident])
    return ident


def reject(cur, detection_id):
    """User feedback: this auto-label is wrong. Clear it and remember the correction."""
    cur.execute("INSERT INTO labels (detection_id, action) VALUES (:1, 'reject')", [detection_id])
    cur.execute("UPDATE detections SET identity_id = NULL WHERE id = :1 AND confirmed = 0", [detection_id])


HIDDEN = "·not-a-person·"     # reserved identity for dismissed (non-person / junk) groups


def dismiss_cluster(cur, detection_id):
    """'Not a person' feedback: move this detection's whole duplicate-group out of the unknown pool
    by assigning it to a reserved hidden identity (kind='hidden', never shown as a known person)."""
    cur.execute("SELECT id FROM identities WHERE name = :1 AND kind = 'hidden'", [HIDDEN])
    row = cur.fetchone()
    if row:
        hid = int(row[0])
    else:
        idv = cur.var(oracledb.NUMBER)
        cur.execute("INSERT INTO identities (name, kind) VALUES (:1, 'hidden') RETURNING id INTO :2",
                    [HIDDEN, idv])
        hid = int(idv.getvalue()[0])
    cur.execute("SELECT cluster_id FROM detections WHERE id = :1", [detection_id])
    r = cur.fetchone()
    cid = r[0] if r else None
    if cid is not None:
        cur.execute("UPDATE detections SET identity_id = :1 WHERE cluster_id = :2 AND confirmed = 0", [hid, cid])
    else:
        cur.execute("UPDATE detections SET identity_id = :1 WHERE id = :2 AND confirmed = 0", [hid, detection_id])
    cur.execute("INSERT INTO labels (detection_id, action) VALUES (:1, 'dismiss')", [detection_id])


def rematch(cur, thr=DEFAULT_THR):
    """(Re)assign every UNCONFIRMED detection to the nearest CONFIRMED known person within `thr`.
    Pure vector search over stored embeddings — no inference. Returns the number of rows changed."""
    cur.execute("""UPDATE detections d SET identity_id = (
                     SELECT k.identity_id FROM detections k
                     WHERE k.confirmed = 1 AND k.identity_id IS NOT NULL
                       AND VECTOR_DISTANCE(k.embedding, d.embedding, COSINE) < :thr
                     ORDER BY VECTOR_DISTANCE(k.embedding, d.embedding, COSINE)
                     FETCH FIRST 1 ROWS ONLY)
                   WHERE d.confirmed = 0""", [thr])
    return cur.rowcount


def summary(cur):
    cur.execute("""SELECT NVL(i.name, '(unknown)') who, COUNT(*) n
                   FROM detections d LEFT JOIN identities i ON i.id = d.identity_id
                   GROUP BY NVL(i.name, '(unknown)') ORDER BY n DESC""")
    return cur.fetchall()


if __name__ == "__main__":
    orc = adb.connect(); cur = orc.cursor()
    if len(sys.argv) >= 4 and sys.argv[1] == "tag":
        print("tagged -> identity", tag(cur, int(sys.argv[2]), sys.argv[3])); orc.commit()
    elif len(sys.argv) >= 2 and sys.argv[1] == "rematch":
        thr = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_THR
        print("rematched", rematch(cur, thr), "detection(s)"); orc.commit()
    elif len(sys.argv) >= 2 and sys.argv[1] == "summary":
        for who, n in summary(cur):
            print(f"  {who}: {n}")
    else:
        sys.exit("usage: match.py tag <detection_id> <name> | rematch [thr] | summary")
