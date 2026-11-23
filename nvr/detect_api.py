#!/usr/bin/env python3
# People / feedback API for person detection — the read + tagging side of the detection store.
#
# DECOUPLED by design (same rule as the poller): reads the Oracle 23ai vector DB, writes ONLY to
# Oracle (labels/identities via match.py), imports no recorder code and opens no recorder sockets.
# It is deliberately LIGHT: no onnxruntime/numpy, no ffmpeg, no nvr.db. Person crops are STATIC
# jpegs the detector wrote at detection time; this service just serves them from disk, so the image
# path can never stall the service and the gallery stays fast on the tiny box.
#
#   python3 detect_api.py            (nvr-detect-api.service; 127.0.0.1:PORT, nginx /detapi/ in front)
#
#   GET  /detapi/people                 known people: [{id,name,count,sample}]
#   GET  /detapi/unknowns?limit=N       recent un-identified detections: [{id,ts,seg_id,score}]
#   GET  /detapi/person?id=N            one identity's crops: {name, detections:[{id,ts,score}]}
#   GET  /detapi/training               known people + their training-set sizes: [{id,name,confirmed,trained,sample}]
#   GET  /detapi/training_person?id=N   one person's confirmed training crops (all dates) + quality flags
#   GET  /detapi/group?id=N             one unknown group's crops (N = a member): {grp, detections:[...]}
#   GET  /detapi/crop/<id>.jpg          the person crop for one detection (decoded on demand, cached)
#   GET  /detapi/stats                  detect VM: {host, detect:{cursor,lag_s,backlog}, services:[unit...]}
#   GET  /detapi/jobs/logs?unit&n        full log tail for a detect-VM unit (nvr-detect / nvr-detect-api)
#   POST /detapi/tag      {id|ids,name} name one/many groups -> known identity, then re-match history
#   POST /detapi/reject   {id|ids}      mark one/many groups 'not a person' (hidden)
#   POST /detapi/unassign {id|ids}      remove detection(s) from their identity, back to the unknown pool
#   POST /detapi/train_pick {id|ids,pick} force crops into(1)/out of(0)/auto(null) the training gallery
#   POST /detapi/set_cover {id,det}     set a person's cover crop (det=null clears to auto); display-only
#   POST /detapi/rename   {id,name}     rename an identity (merges into an existing same-name person)
import json
import os
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import oracledb

import adb
import hoststat
import jobstat
import match

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("NVR_DATA", "/home/ubuntu/nvr-data")
PORT = int(os.environ.get("DETAPI_PORT", "3401"))
HOST = os.environ.get("DETAPI_HOST", "127.0.0.1")  # 0.0.0.0 when nginx is on another box (remote detector)
CROP_DIR = os.environ.get("NVR_CROPS") or os.path.join(DATA, "crops")  # static crops from detect_service
                                          # (NVR_CROPS must match the detector's when they share a box)
SNAP_DB = os.environ.get("NVR_DB_PATH")               # static immutable snapshot, for the lag readout (optional)
DET_STREAM = os.environ.get("NVR_STREAM", "cam1-sd")  # the presence stream the poller consumes
UNKNOWN_LIMIT = 120

# This box's (the detect VM's) own units, surfaced on the dashboard alongside the recorder VM's.
DET_JOBS = [("nvr-detect", "Detection poller"), ("nvr-detect-api", "Detection API")]
DET_UNITS = {u for u, _ in DET_JOBS}

os.makedirs(CROP_DIR, exist_ok=True)

# ---- Oracle: one connection, serialized (low-traffic internal UI), reconnect on error ----
_conn = None
_lock = threading.Lock()


def _cursor():
    global _conn
    if _conn is None:
        _conn = adb.connect()
    return _conn.cursor()


def db(fn, write=False):
    """Run fn(cur) under the lock; reconnect once on a dropped connection."""
    global _conn
    with _lock:
        for attempt in (1, 2):
            try:
                cur = _cursor()
                r = fn(cur)
                if write:
                    _conn.commit()
                return r
            except oracledb.DatabaseError:
                try:
                    _conn.close()
                except Exception:
                    pass
                _conn = None
                if attempt == 2:
                    raise


# ---- person crops are STATIC files written by the detector (detect_service._save_crop) ----
# Serving them straight from disk means no ffmpeg and no Oracle on the image path: the gallery
# stays fast and can never stall the service, and a crop survives even after its segment is gone.
def crop_file(det_id):
    p = os.path.join(CROP_DIR, f"{det_id}.jpg")
    return p if os.path.exists(p) else None


def has_crop(det_id):
    return os.path.exists(os.path.join(CROP_DIR, f"{det_id}.jpg"))


def _cover(cur, ident_id):
    """The user-chosen cover crop for an identity (identities.cover_id), or None to fall back to an
    auto-picked sample. None if cover_id is unset, its file is gone, or the crop no longer belongs to
    this person (e.g. it was later unassigned or re-tagged to someone else) — the JOIN enforces that so
    a stale cover_id can never surface a crop that is now in the unknown pool or on another identity."""
    cur.execute("""SELECT i.cover_id FROM identities i
                   JOIN detections d ON d.id = i.cover_id AND d.identity_id = i.id AND d.confirmed = 1
                   WHERE i.id = :1""", [ident_id])
    r = cur.fetchone()
    cid = int(r[0]) if r and r[0] is not None else None
    return cid if (cid is not None and has_crop(cid)) else None


# ---- queries ----
def people(frm=None, to=None):
    """Known people. With a [frm,to) epoch range, only those seen in it, counts/sample scoped to it."""
    ranged = frm is not None and to is not None
    def q(cur):
        if ranged:
            cur.execute("""SELECT i.id, i.name, COUNT(d.id), MIN(d.ts), MAX(d.ts)
                           FROM identities i JOIN detections d ON d.identity_id = i.id
                           WHERE i.kind = 'known' AND d.ts >= :1 AND d.ts < :2
                           GROUP BY i.id, i.name ORDER BY LOWER(i.name)""", [frm, to])
        else:
            cur.execute("""SELECT i.id, i.name, COUNT(d.id), MIN(d.ts), MAX(d.ts)
                           FROM identities i LEFT JOIN detections d ON d.identity_id = i.id
                           WHERE i.kind = 'known'
                           GROUP BY i.id, i.name ORDER BY LOWER(i.name)""")
        return cur.fetchall()

    def pick_sample(cur, ident_id):
        cv = _cover(cur, ident_id)
        if cv:
            return cv
        if ranged:
            cur.execute("""SELECT id FROM detections WHERE identity_id = :1 AND ts >= :2 AND ts < :3
                           ORDER BY confirmed DESC, id DESC FETCH FIRST 12 ROWS ONLY""", [ident_id, frm, to])
        else:
            cur.execute("""SELECT id FROM detections WHERE identity_id = :1
                           ORDER BY confirmed DESC, id DESC FETCH FIRST 12 ROWS ONLY""", [ident_id])
        for (did,) in cur.fetchall():
            if has_crop(int(did)):
                return int(did)
        return None

    out = []
    for r in db(q):
        ident_id = int(r[0])
        out.append({"id": ident_id, "name": r[1], "count": int(r[2]),
                    "first": float(r[3]) if r[3] is not None else None,
                    "last": float(r[4]) if r[4] is not None else None,
                    "sample": db(lambda cur: pick_sample(cur, ident_id))})
    return out


def person(ident_id, frm=None, to=None):
    """One identity's crops, newest first, confirmed ground truth ahead of re-matched. Backs the person
    detail view where the user prunes wrong crops. With a [frm,to) epoch range, only that day's crops
    (the detail matches the day-scoped grid it was opened from)."""
    ranged = frm is not None and to is not None
    def q(cur):
        cur.execute("SELECT name FROM identities WHERE id = :1", [ident_id])
        nm = cur.fetchone()
        if ranged:
            cur.execute("""SELECT id, ts, score FROM detections WHERE identity_id = :1
                           AND ts >= :2 AND ts < :3
                           ORDER BY confirmed DESC, ts DESC FETCH FIRST 500 ROWS ONLY""", [ident_id, frm, to])
        else:
            cur.execute("""SELECT id, ts, score FROM detections WHERE identity_id = :1
                           ORDER BY confirmed DESC, ts DESC FETCH FIRST 500 ROWS ONLY""", [ident_id])
        return (nm[0] if nm else None), cur.fetchall()
    name, rows = db(q)
    dets = [{"id": int(d), "ts": float(t), "score": round(float(s), 2) if s is not None else None}
            for d, t, s in rows if has_crop(int(d))]
    return {"name": name, "detections": dets}


def _gate_pass(score, focus, boxfrac):
    """The auto quality gate alone (no manual pick, no fallback): confident, sharp, not an extreme close-up.
    NULL focus/boxfrac (not yet backfilled) pass; NULL score fails. Mirrors the gate in match.rebuild_protos."""
    if score is None or float(score) < match.TRAIN_MIN_SCORE:
        return False
    if focus is not None and float(focus) < match.TRAIN_MIN_FOCUS:
        return False
    if boxfrac is not None and float(boxfrac) > match.TRAIN_MAX_BOXFRAC:
        return False
    return True


def _trained_ids(rows):
    """Given an identity's confirmed crops as (id, score, focus, boxfrac, pick), return the set of ids that
    ACTUALLY seed the gallery — mirroring match.rebuild_protos INCLUDING the PROTO_FLOOR fallback: when fewer
    than PROTO_FLOOR crops pass the gate, the gallery falls back to all confirmed crops except force-excluded
    (train_pick=0). Without this the training page under-reports 'trained' for people whose crops are all
    sub-threshold (the gate starves and the fallback trains on all of them)."""
    gated, kept = set(), set()
    for did, s, f, b, p in rows:
        did = int(did); pk = None if p is None else int(p)
        if pk == 0:
            continue                      # force-excluded: never trains, in gate or fallback
        kept.add(did)                     # eligible for the fallback (train_pick=1 or NULL)
        if pk == 1 or _gate_pass(s, f, b):
            gated.add(did)
    return kept if len(gated) < match.PROTO_FLOOR else gated


def training():
    """Known people with their TRAINING set (manually-confirmed crops, any date). Each carries how many
    confirmed crops exist and how many actually train (pass the quality gate), plus a sample crop."""
    def q(cur):
        cur.execute("""SELECT i.id, i.name,
                              SUM(CASE WHEN d.confirmed = 1 THEN 1 ELSE 0 END)
                       FROM identities i LEFT JOIN detections d ON d.identity_id = i.id
                       WHERE i.kind = 'known'
                       GROUP BY i.id, i.name ORDER BY LOWER(i.name)""")
        return cur.fetchall()

    def detail(cur, iid):
        cur.execute("""SELECT id, score, focus, boxfrac, train_pick FROM detections
                       WHERE identity_id = :1 AND confirmed = 1 ORDER BY id DESC""", [iid])
        rows = cur.fetchall()
        used = len(_trained_ids(rows))
        sample = _cover(cur, iid) or next((int(r[0]) for r in rows if has_crop(int(r[0]))), None)
        return used, sample

    out = []
    for r in db(q):
        iid = int(r[0]); confirmed = int(r[2] or 0)
        used, sample = db(lambda cur: detail(cur, iid))
        out.append({"id": iid, "name": r[1], "confirmed": confirmed, "trained": used, "sample": sample})
    return out


def training_person(ident_id):
    """One person's full TRAINING set (every manually-confirmed crop, all dates) with the quality signals
    and whether each actually trains, so the user can prune blurry / zoomed / wrong crops from the gallery."""
    def q(cur):
        cur.execute("SELECT name FROM identities WHERE id = :1", [ident_id])
        nm = cur.fetchone()
        cover = _cover(cur, ident_id)          # validated: never a dangling / foreign crop id
        cur.execute("""SELECT id, ts, score, focus, boxfrac, train_pick FROM detections
                       WHERE identity_id = :1 AND confirmed = 1 ORDER BY id DESC
                       FETCH FIRST 1500 ROWS ONLY""", [ident_id])
        return (nm[0] if nm else None), cover, cur.fetchall()
    name, cover, rows = db(q)
    tids = _trained_ids((d, s, f, b, p) for d, t, s, f, b, p in rows)  # mirrors the gallery incl. fallback
    dets = []
    for d, t, s, f, b, p in rows:
        if not has_crop(int(d)):
            continue
        dets.append({"id": int(d), "ts": float(t), "score": round(float(s), 2) if s is not None else None,
                     "focus": round(float(f), 3) if f is not None else None,
                     "boxfrac": round(float(b), 2) if b is not None else None,
                     "pick": (int(p) if p is not None else None), "used": int(d) in tids})
    return {"name": name, "cover": cover, "detections": dets}


def group(rep_id, frm=None, to=None):
    """Every unknown crop in one duplicate-group (the cluster the given representative belongs to).
    Mirrors person() so the UI can open an unknown group the same way it opens a tagged person. With a
    [frm,to) epoch range, only that day's crops (matches the day-scoped grid)."""
    ranged = frm is not None and to is not None
    def q(cur):
        cur.execute("SELECT NVL(cluster_id, id) FROM detections WHERE id = :1", [rep_id])
        r = cur.fetchone()
        grp = int(r[0]) if r else int(rep_id)
        if ranged:
            cur.execute("""SELECT id, ts, score FROM detections
                           WHERE identity_id IS NULL AND NVL(cluster_id, id) = :1 AND ts >= :2 AND ts < :3
                           ORDER BY ts DESC FETCH FIRST 500 ROWS ONLY""", [grp, frm, to])
        else:
            cur.execute("""SELECT id, ts, score FROM detections
                           WHERE identity_id IS NULL AND NVL(cluster_id, id) = :1
                           ORDER BY ts DESC FETCH FIRST 500 ROWS ONLY""", [grp])
        return grp, cur.fetchall()
    grp, rows = db(q)
    dets = [{"id": int(d), "ts": float(t), "score": round(float(s), 2) if s is not None else None}
            for d, t, s in rows if has_crop(int(d))]
    return {"grp": grp, "detections": dets}


def det_times(frm, to):
    """Presence spans for the live-page timeline: [ts_start, ts_end] for each real detection whose ts is
    in [frm,to). In track mode one row spans a whole visit; older per-frame rows collapse to a point
    (NVL to ts). Hidden ('not a person') crops are excluded; unknown (identity NULL) and known are kept.
    Capped so a dense legacy day can't bloat the payload."""
    def q(cur):
        cur.execute("""SELECT NVL(ts_start, ts), NVL(ts_end, ts) FROM detections
                       WHERE ts >= :frm AND ts < :tto
                         AND (identity_id IS NULL OR identity_id NOT IN
                              (SELECT id FROM identities WHERE kind = 'hidden'))
                       ORDER BY NVL(ts_start, ts) FETCH FIRST 5000 ROWS ONLY""", {"frm": frm, "tto": to})
        return [[float(a), float(b)] for a, b in cur.fetchall()]
    return {"spans": db(q)}


def unassign(det_ids):
    def w(cur):
        for did in det_ids:
            match.unassign(cur, did)
        match.rebuild_protos(cur)                 # affected people's galleries changed; re-propagate
        match.rematch(cur)
        return {"ok": True, "count": len(det_ids)}
    return db(w, write=True)


def rename(ident_id, name):
    def w(cur):
        surviving = match.rename(cur, ident_id, name)
        match.rebuild_protos(cur, surviving)
        changed = match.rematch(cur)
        return {"ok": True, "identity": surviving, "rematched": changed}
    return db(w, write=True)


def set_cover(ident_id, det_id):
    """Set (or clear, det_id=None) the cover crop shown for a person on the people / training grids.
    The crop must be a confirmed crop of this identity. Display-only — no gallery rebuild."""
    def w(cur):
        if det_id is not None:
            cur.execute("SELECT 1 FROM detections WHERE id = :1 AND identity_id = :2 AND confirmed = 1",
                        [int(det_id), int(ident_id)])
            if not cur.fetchone():
                return {"ok": False, "error": "crop not a confirmed crop of this person"}
        cur.execute("UPDATE identities SET cover_id = :1 WHERE id = :2",
                    [int(det_id) if det_id is not None else None, int(ident_id)])
        return {"ok": True, "id": int(ident_id), "cover": (int(det_id) if det_id is not None else None)}
    return db(w, write=True)


def train_pick(det_ids, pick):
    """Manually set whether crops train: pick=1 force INTO the gallery, 0 force OUT, None back to the auto
    gate. Only affects CONFIRMED crops of a known person (a pick on anything else is a no-op). Rebuilds the
    affected galleries so the change takes effect immediately. `rematch` re-propagates to unconfirmed rows."""
    val = None if pick is None else (1 if int(pick) == 1 else 0)
    def w(cur):
        idents = set()
        for did in det_ids:
            cur.execute("""SELECT identity_id FROM detections WHERE id = :1 AND confirmed = 1
                           AND identity_id IN (SELECT id FROM identities WHERE kind = 'known')""", [int(did)])
            r = cur.fetchone()
            if not r:
                continue
            cur.execute("UPDATE detections SET train_pick = :1 WHERE id = :2", [val, int(did)])
            idents.add(int(r[0]))
        for iid in idents:
            match.rebuild_protos(cur, iid)
        if idents:
            match.rematch(cur)
        return {"ok": True, "count": len(det_ids), "people": len(idents)}
    return db(w, write=True)


def unknowns(limit, frm=None, to=None):
    """Unknown detections GROUPED by duplicate-cluster: one tile per person. Each entry's `id` is the
    representative (best-scoring crop) and `count` is how many detections the group holds. Tagging
    that id names the whole group (detect_service assigned the cluster_id). With a [frm,to) epoch range,
    only crops seen in it (so a group's shown count is its sightings that day)."""
    def q(cur):
        # Cap the raw scan so a large legacy untagged pool can't blow up memory / stat every crop file.
        # Rows are newest-first, so the cap keeps the most recent sightings (which is what the grid shows).
        if frm is not None and to is not None:
            cur.execute("""SELECT id, ts, score, NVL(cluster_id, id) grp FROM detections
                           WHERE identity_id IS NULL AND ts >= :1 AND ts < :2
                           ORDER BY ts DESC FETCH FIRST 20000 ROWS ONLY""", [frm, to])
        else:
            cur.execute("""SELECT id, ts, score, NVL(cluster_id, id) grp FROM detections
                           WHERE identity_id IS NULL ORDER BY ts DESC FETCH FIRST 20000 ROWS ONLY""")
        return cur.fetchall()
    groups = {}
    for did, ts, score, grp in db(q):
        did, grp, ts = int(did), int(grp), float(ts)
        score = round(float(score), 2) if score is not None else 0.0
        if not has_crop(did):
            continue
        g = groups.get(grp)
        if g is None:
            g = groups[grp] = {"id": did, "count": 0, "score": score, "ts": ts, "first": ts}
        g["count"] += 1
        if score > g["score"]:                  # representative = best-scoring crop in the group
            g["id"], g["score"] = did, score
        g["ts"] = max(g["ts"], ts)              # ts = most recent sighting; first = earliest
        g["first"] = min(g["first"], ts)
    return sorted(groups.values(), key=lambda x: x["ts"], reverse=True)[:limit]


def _range(q):
    """Optional [from, to) epoch window from the query string, else (None, None) for all-time."""
    try:
        if "from" in q and "to" in q:
            return float(q["from"][0]), float(q["to"][0])
    except (ValueError, KeyError):
        pass
    return None, None


def _ids(b):
    """Accept either {id} (single group) or {ids:[...]} (batch), as a de-duped list of ints."""
    raw = b["ids"] if isinstance(b.get("ids"), list) else ([b["id"]] if "id" in b else [])
    return list(dict.fromkeys(int(x) for x in raw))


def tag(det_ids, name):
    def w(cur):
        ident = None
        for did in det_ids:
            ident = match.tag_cluster(cur, did, name)  # names each whole duplicate-group
        match.rebuild_protos(cur, ident)                # refresh this person's prototype gallery
        changed = match.rematch(cur)                    # one re-match covers the whole batch
        return {"ok": True, "identity": ident, "rematched": changed, "count": len(det_ids)}
    return db(w, write=True)


def reject(det_ids):
    def w(cur):
        for did in det_ids:
            match.dismiss_cluster(cur, did)
        match.rebuild_protos(cur)                 # refresh the 'not a person' gallery
        match.rematch(cur)                        # auto-suppress other detections matching the dismissed junk
        return {"ok": True, "count": len(det_ids)}
    return db(w, write=True)


def detect_lag():
    """How far the poller is behind the recorder. The cursor (last processed seg) is read from Oracle;
    the newest presence segment and the cursor's wall-clock come from the STATIC snapshot copy
    (immutable, short timeout — never the live recorder DB, and never blocks the service). Every field is
    guarded so NFS being away just yields nulls. lag_s is now minus the cursor segment's start."""
    out = {"cursor": None, "newest_seg": None, "newest_start": None,
           "lag_s": None, "backlog": None}
    try:
        def q(cur):
            cur.execute("SELECT v FROM detect_state WHERE k = :1", ["last_seg_id"])
            r = cur.fetchone()
            return int(r[0]) if r else None
        out["cursor"] = db(q)
    except Exception:
        pass
    if SNAP_DB:
        try:
            c = sqlite3.connect(f"file:{SNAP_DB}?immutable=1", uri=True, timeout=3)
            try:
                r = c.execute("SELECT id, start FROM segments WHERE stream = ? ORDER BY id DESC LIMIT 1",
                              (DET_STREAM,)).fetchone()
                if r:
                    out["newest_seg"], out["newest_start"] = int(r[0]), float(r[1])
                if out["cursor"]:
                    cr = c.execute("SELECT start FROM segments WHERE id = ?", (out["cursor"],)).fetchone()
                    if cr:
                        out["lag_s"] = round(time.time() - float(cr[0]))
                    bl = c.execute("SELECT COUNT(*) FROM segments WHERE stream = ? AND id > ?",
                                   (DET_STREAM, out["cursor"])).fetchone()
                    out["backlog"] = int(bl[0]) if bl else None
            finally:
                c.close()
        except Exception:
            pass
    return out


class H(BaseHTTPRequestHandler):
    server_version = "nvr-detect-api"

    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def err(self, code, msg):
        self.send(code, json.dumps({"error": msg}))

    def body_json(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/detapi/people":
                frm, to = _range(q)
                return self.send(200, json.dumps(people(frm, to)))
            if u.path == "/detapi/unknowns":
                lim = max(1, min(int(q.get("limit", [UNKNOWN_LIMIT])[0]), UNKNOWN_LIMIT))
                frm, to = _range(q)
                return self.send(200, json.dumps(unknowns(lim, frm, to)))
            if u.path == "/detapi/person":
                frm, to = _range(q)
                return self.send(200, json.dumps(person(int(q["id"][0]), frm, to)))
            if u.path == "/detapi/training":
                return self.send(200, json.dumps(training()))
            if u.path == "/detapi/training_person":
                return self.send(200, json.dumps(training_person(int(q["id"][0]))))
            if u.path == "/detapi/group":
                frm, to = _range(q)
                return self.send(200, json.dumps(group(int(q["id"][0]), frm, to)))
            if u.path == "/detapi/times":
                frm, to = _range(q)
                if frm is None:
                    return self.err(400, "from/to required")
                return self.send(200, json.dumps(det_times(frm, to)))
            if u.path == "/detapi/stats":              # detect VM host stats + poller lag + its units
                return self.send(200, json.dumps({"role": "detect",
                                                  "host": hoststat.host_stats(), "detect": detect_lag(),
                                                  "services": [jobstat.job_status(un, DET_UNITS) for un, _ in DET_JOBS]}))
            if u.path == "/detapi/jobs/logs":          # full log tail for a detect-VM unit (dashboard expand)
                unit = q.get("unit", [""])[0]
                if unit not in DET_UNITS:
                    return self.err(400, "unknown unit")
                n = max(1, min(int(q.get("n", ["300"])[0]), 1000))
                return self.send(200, json.dumps({"unit": unit, "lines": jobstat.unit_logs(unit, DET_UNITS, n)}))
            if u.path.startswith("/detapi/crop/") and u.path.endswith(".jpg"):
                det_id = int(u.path[len("/detapi/crop/"):-4])
                p = crop_file(det_id)
                if p is None:
                    return self.err(404, "no crop")
                with open(p, "rb") as f:
                    return self.send(200, f.read(), "image/jpeg",
                                     {"Cache-Control": "private, max-age=86400"})
            return self.err(404, "not found")
        except (KeyError, ValueError, TypeError) as e:
            return self.err(400, str(e))
        except Exception as e:
            return self.err(500, str(e))

    def do_POST(self):
        u = urlparse(self.path)
        try:
            b = self.body_json()
            if not isinstance(b, dict):            # a JSON array / string / number / null is not a request body
                return self.err(400, "body must be a JSON object")
            if u.path == "/detapi/tag":
                name = str(b.get("name", "")).strip()
                ids = _ids(b)
                if not name or not ids:
                    return self.err(400, "need id(s) and name")
                return self.send(200, json.dumps(tag(ids, name)))
            if u.path == "/detapi/reject":
                ids = _ids(b)
                if not ids:
                    return self.err(400, "need id(s)")
                return self.send(200, json.dumps(reject(ids)))
            if u.path == "/detapi/unassign":
                ids = _ids(b)
                if not ids:
                    return self.err(400, "need id(s)")
                return self.send(200, json.dumps(unassign(ids)))
            if u.path == "/detapi/rename":
                name = str(b.get("name", "")).strip()
                if not name or "id" not in b:
                    return self.err(400, "need id and name")
                return self.send(200, json.dumps(rename(int(b["id"]), name)))
            if u.path == "/detapi/train_pick":
                ids = _ids(b)
                if not ids:
                    return self.err(400, "need id(s)")
                return self.send(200, json.dumps(train_pick(ids, b.get("pick"))))
            if u.path == "/detapi/set_cover":
                if "id" not in b:
                    return self.err(400, "need id")
                det = b.get("det")
                return self.send(200, json.dumps(set_cover(int(b["id"]), None if det is None else int(det))))
            return self.err(404, "not found")
        except (KeyError, ValueError, TypeError) as e:
            return self.err(400, str(e))
        except Exception as e:
            return self.err(500, str(e))


if __name__ == "__main__":
    print(f"detect-api on {HOST}:{PORT}", flush=True)
    ThreadingHTTPServer((HOST, PORT), H).serve_forever()
