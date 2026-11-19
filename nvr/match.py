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
    """Tag a whole duplicate-group at once: assign `name`'s identity to EVERY detection sharing this
    one's cluster_id and CONFIRM them all. Falls back to the single detection when it isn't clustered.

    Why confirm the whole group (not just the tapped crop): the user has asserted the group is this
    person, and tag() runs rematch() straight after. rematch re-evaluates every confirmed=0 row and
    nulls the ones with no near KNOWN match — so if the group were left unconfirmed, any member whose
    embedding sits past the re-ID threshold from the tapped crop would be stripped back to unknown,
    leaving the tag on only one. confirmed=1 makes the group label stick and seeds future re-ID.

    But claim only UNCONFIRMED members (plus any already this identity) — never crops the user confirmed
    as a DIFFERENT person. A crop removed from another group (unassign) keeps its old cluster_id until
    re-clustered, and without this guard re-tagging it would drag its old cluster-mates along."""
    ident = get_or_create_identity(cur, name)
    cur.execute("SELECT cluster_id FROM detections WHERE id = :1", [detection_id])
    row = cur.fetchone()
    cid = row[0] if row else None
    if cid is not None:
        cur.execute("""UPDATE detections SET identity_id = :ident, confirmed = 1
                       WHERE cluster_id = :cid AND (confirmed = 0 OR identity_id = :ident)""",
                    {"ident": ident, "cid": cid})
    cur.execute("UPDATE detections SET identity_id = :1, confirmed = 1 WHERE id = :2", [ident, detection_id])
    cur.execute("INSERT INTO labels (detection_id, identity_id, action) VALUES (:1, :2, 'name')",
                [detection_id, ident])
    return ident


def reject(cur, detection_id):
    """User feedback: this auto-label is wrong. Clear it and remember the correction."""
    cur.execute("INSERT INTO labels (detection_id, action) VALUES (:1, 'reject')", [detection_id])
    cur.execute("UPDATE detections SET identity_id = NULL WHERE id = :1 AND confirmed = 0", [detection_id])


def unassign(cur, detection_id):
    """Remove ONE detection from whatever identity it's on, back to the unknown pool (identity_id NULL,
    confirmed 0) — used when the user prunes a wrong crop from a tagged person. Clears confirmed too, so
    it also works on a detection that a group tag had confirmed. A later rematch() only re-adds it if its
    embedding is actually within the re-ID threshold of the person's remaining crops (i.e. it really is
    them); a genuinely wrong crop stays out.

    Also DETACH it into its own cluster (cluster_id = id): it no longer shares the old group's cluster_id,
    so re-tagging this one crop to another person can't drag its former cluster-mates with it. The next
    _cluster() pass re-groups it among the unknowns by embedding."""
    cur.execute("INSERT INTO labels (detection_id, action) VALUES (:1, 'remove')", [detection_id])
    cur.execute("UPDATE detections SET identity_id = NULL, confirmed = 0, cluster_id = id WHERE id = :1",
                [detection_id])


def rename(cur, ident_id, name):
    """Rename an identity. If `name` already belongs to another known person, MERGE into them: move this
    identity's detections and labels over and drop the now-empty identity. Returns the surviving id."""
    cur.execute("SELECT id FROM identities WHERE name = :1 AND kind = 'known' AND id <> :2", [name, ident_id])
    row = cur.fetchone()
    if row:
        target = int(row[0])
        cur.execute("UPDATE detections SET identity_id = :1 WHERE identity_id = :2", [target, ident_id])
        cur.execute("UPDATE labels SET identity_id = :1 WHERE identity_id = :2", [target, ident_id])
        cur.execute("DELETE FROM identities WHERE id = :1", [ident_id])
        return target
    cur.execute("UPDATE identities SET name = :1 WHERE id = :2", [name, ident_id])
    return ident_id


HIDDEN = "·not-a-person·"     # reserved identity for dismissed (non-person / junk) groups


def dismiss_cluster(cur, detection_id):
    """'Not a person' feedback: move this detection's whole duplicate-group out of the unknown pool
    by assigning it to a reserved hidden identity (kind='hidden', never shown as a known person).

    confirmed=1 so the dismissal STICKS: a later tag() runs rematch(), which re-evaluates every
    confirmed=0 row — a dismissed-but-unconfirmed group would have no near KNOWN match and get nulled
    straight back to unknown. rematch only matches against kind='known', so a confirmed hidden row is
    never used to auto-hide a real person near the junk cluster."""
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
        # Claim only unconfirmed members (plus any already hidden) — don't hide a crop the user confirmed
        # as a real known person that happens to share this cluster_id.
        cur.execute("""UPDATE detections SET identity_id = :hid, confirmed = 1
                       WHERE cluster_id = :cid AND (confirmed = 0 OR identity_id = :hid)""",
                    {"hid": hid, "cid": cid})
    cur.execute("UPDATE detections SET identity_id = :1, confirmed = 1 WHERE id = :2", [hid, detection_id])
    cur.execute("INSERT INTO labels (detection_id, action) VALUES (:1, 'dismiss')", [detection_id])


def rematch(cur, thr=DEFAULT_THR):
    """(Re)assign every UNCONFIRMED detection to the nearest CONFIRMED KNOWN person within `thr`.
    Pure vector search over stored embeddings — no inference. Returns the number of rows changed.

    Matches only kind='known' (not the reserved hidden/'not-a-person' identity): a dismissed junk
    cluster must never be propagated onto — or used to auto-hide — a real person that happens to sit
    near it in embedding space. Confirmed rows (tagged or dismissed groups) are left untouched."""
    cur.execute("""UPDATE detections d SET identity_id = (
                     SELECT k.identity_id FROM detections k
                     JOIN identities ki ON ki.id = k.identity_id
                     WHERE k.confirmed = 1 AND ki.kind = 'known'
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
