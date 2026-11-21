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
PROTO_K = int(os.environ.get("DET_PROTO_K", "16"))         # prototype centroids per known person (gallery size)
APPROX_K = int(os.environ.get("DET_APPROX_K", "1000"))     # per-proto candidate fan-out for the vector index
SUPPRESS_THR = float(os.environ.get("DET_SUPPRESS", "0.22"))  # TIGHTER cosine gate for auto-hiding a match to
#   the reserved 'not a person' gallery. Deliberately well below DET_MATCH: a dismissed static false positive
#   (a bucket, a bag) re-detects near-identically, so its members sit ~0.1 from their hidden prototype, while a
#   real person sits >0.3 from that junk — so this suppresses the recurring false positive but never a person.


# ---- prototype gallery: a few spherical-k-means centroids per known person, kept in person_protos.
# Matching against these (not every confirmed crop) bounds the gallery and denoises it, while centroids
# (unlike farthest-point medoids) stay central so recall holds: measured ~97% self-coverage at K=8,
# 100% at K=16 on real data. rebuild after any change to a person's confirmed set. numpy is imported
# lazily so the light detect-api process only pays for it when a tag actually happens.
def _skmeans(E, K, iters=15):
    import numpy as np
    n = len(E)
    if n <= K:
        return E.copy()
    rng = np.random.default_rng(0)
    C = E[rng.choice(n, K, replace=False)].copy()
    for _ in range(iters):
        a = (E @ C.T).argmax(1)
        NC = C.copy()
        for k in range(K):
            m = E[a == k]
            if len(m):
                v = m.sum(0)
                NC[k] = v / (np.linalg.norm(v) + 1e-9)     # re-normalize: cosine (spherical) k-means
        C = NC
    return C


def rebuild_protos(cur, identity_id=None):
    """Recompute prototype centroids for one identity (or all known + hidden). Replaces that identity's
    rows in person_protos. Cheap: an identity has at most a few hundred confirmed crops. The hidden
    'not a person' identity is included so its prototypes drive auto-suppression (see nearest_proto)."""
    import numpy as np, array
    if identity_id is not None:
        idents = [int(identity_id)]
    else:
        cur.execute("SELECT id FROM identities WHERE kind IN ('known', 'hidden')")
        idents = [int(r[0]) for r in cur.fetchall()]
    for iid in idents:
        cur.execute("SELECT embedding FROM detections WHERE identity_id = :1 AND confirmed = 1", [iid])
        rows = cur.fetchall()
        cur.execute("DELETE FROM person_protos WHERE identity_id = :1", [iid])
        if not rows:
            continue
        E = np.asarray([np.asarray(r[0], np.float32) for r in rows])
        P = _skmeans(E, PROTO_K)
        cur.executemany("INSERT INTO person_protos (identity_id, embedding) VALUES (:1, :2)",
                        [(iid, array.array("f", p.tolist())) for p in P])


def nearest_proto(cur, emb, thr=DEFAULT_THR, suppress_thr=SUPPRESS_THR):
    """Nearest prototype to an embedding -> its identity_id, or None. The live match for each new
    detection. Known people match within `thr`; failing that, the reserved hidden ('not a person')
    gallery matches within a TIGHTER `suppress_thr` — so a learned false positive is auto-suppressed,
    but a real person (always well outside suppress_thr of junk) is never hidden. Known wins ties."""
    cur.execute("""SELECT p.identity_id, VECTOR_DISTANCE(p.embedding, :e, COSINE) d
                   FROM person_protos p JOIN identities i ON i.id = p.identity_id
                   WHERE i.kind = 'known' ORDER BY d FETCH FIRST 1 ROWS ONLY""", {"e": emb})
    r = cur.fetchone()
    if r and r[1] is not None and r[1] < thr:
        return int(r[0])
    cur.execute("""SELECT p.identity_id, VECTOR_DISTANCE(p.embedding, :e, COSINE) d
                   FROM person_protos p JOIN identities i ON i.id = p.identity_id
                   WHERE i.kind = 'hidden' ORDER BY d FETCH FIRST 1 ROWS ONLY""", {"e": emb})
    r = cur.fetchone()
    return int(r[0]) if r and r[1] is not None and r[1] < suppress_thr else None


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
    """Remove ONE detection from whatever identity it's on — used when the user prunes a wrong crop from a
    tagged person. The removal must STICK even when the crop genuinely resembles that person: a wrong match
    happens precisely because the crop is within the re-ID threshold of the person's centroid, so leaving it
    confirmed=0 would let the immediately-following rematch() re-claim it to the very same person (the "I
    removed it but it came back, count's 2 again" bug). So we mark it confirmed=1 with identity_id=NULL: a
    user-asserted "this is an unknown, do not auto-assign it". rematch() only re-evaluates confirmed=0 rows,
    so it never touches this one; rebuild_protos() only reads confirmed=1 rows that HAVE an identity, so a
    confirmed-but-unidentified row seeds no gallery. The crop drops back into the Unknown pool where it can
    be dismissed or tagged correctly.

    Also DETACH it into its own cluster (cluster_id = id): it no longer shares the old group's cluster_id,
    so re-tagging this one crop to another person can't drag its former cluster-mates with it. The next
    _cluster() pass re-groups it among the unknowns by embedding."""
    cur.execute("INSERT INTO labels (detection_id, action) VALUES (:1, 'remove')", [detection_id])
    cur.execute("UPDATE detections SET identity_id = NULL, confirmed = 1, cluster_id = id WHERE id = :1",
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


def rematch(cur, thr=DEFAULT_THR, suppress_thr=SUPPRESS_THR):
    """(Re)assign every UNCONFIRMED detection to the nearest matching PROTOTYPE, nulling those that match
    none. Known people match within `thr`; the hidden 'not a person' gallery auto-suppresses within the
    tighter `suppress_thr`, and only where no known person matched (known wins). Returns rows (re)assigned.
    Confirmed rows are untouched.

    Prototype-driven over the IVF vector index: for each of the (few) prototypes we ask the index for its
    nearest UNCONFIRMED detections, then resolve each detection in python — O(prototypes · log N) instead
    of the old O(unconfirmed · gallery) full scan."""
    import array, numpy as np
    cur.execute("UPDATE detections SET identity_id = NULL WHERE confirmed = 0")   # clear stale auto-labels
    cur.execute("""SELECT p.identity_id, i.kind, p.embedding FROM person_protos p
                   JOIN identities i ON i.id = p.identity_id""")
    protos = cur.fetchall()
    known, hidden = {}, {}                                # det_id -> (distance, identity), nearest-proto-wins
    for iid, kind, emb in protos:
        e = array.array("f", np.asarray(emb, np.float32).tolist())
        lim = thr if kind == "known" else suppress_thr
        tgt = known if kind == "known" else hidden
        cur.execute(f"""SELECT id, VECTOR_DISTANCE(embedding, :e, COSINE) d
                        FROM detections WHERE confirmed = 0
                        ORDER BY VECTOR_DISTANCE(embedding, :e, COSINE)
                        FETCH APPROX FIRST {APPROX_K} ROWS ONLY""", {"e": e})
        for did, d in cur.fetchall():
            if d is None or d >= lim:
                continue                                  # exact distance re-check on the candidates
            did = int(did)
            if did not in tgt or d < tgt[did][0]:
                tgt[did] = (d, int(iid))
    final = {did: iid for did, (d, iid) in hidden.items()}      # hidden suppression where no person matches
    final.update({did: iid for did, (d, iid) in known.items()}) # a known person always overrides
    if final:
        cur.executemany("UPDATE detections SET identity_id = :1 WHERE id = :2 AND confirmed = 0",
                        [(iid, did) for did, iid in final.items()])
    return len(final)


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
