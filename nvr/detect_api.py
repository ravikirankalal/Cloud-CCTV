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
#   GET  /detapi/group?id=N             one unknown group's crops (N = a member): {grp, detections:[...]}
#   GET  /detapi/crop/<id>.jpg          the person crop for one detection (decoded on demand, cached)
#   POST /detapi/tag      {id|ids,name} name one/many groups -> known identity, then re-match history
#   POST /detapi/reject   {id|ids}      mark one/many groups 'not a person' (hidden)
#   POST /detapi/unassign {id|ids}      remove detection(s) from their identity, back to the unknown pool
#   POST /detapi/rename   {id,name}     rename an identity (merges into an existing same-name person)
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import oracledb

import adb
import match

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("NVR_DATA", "/home/ubuntu/nvr-data")
PORT = int(os.environ.get("DETAPI_PORT", "3401"))
CROP_DIR = os.environ.get("NVR_CROPS") or os.path.join(DATA, "crops")  # static crops from detect_service
                                          # (NVR_CROPS must match the detector's when they share a box)
UNKNOWN_LIMIT = 120

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
    dets = [{"id": int(d), "ts": float(t), "score": round(float(s), 2)}
            for d, t, s in rows if has_crop(int(d))]
    return {"name": name, "detections": dets}


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
    dets = [{"id": int(d), "ts": float(t), "score": round(float(s), 2)}
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


def unknowns(limit, frm=None, to=None):
    """Unknown detections GROUPED by duplicate-cluster: one tile per person. Each entry's `id` is the
    representative (best-scoring crop) and `count` is how many detections the group holds. Tagging
    that id names the whole group (detect_service assigned the cluster_id). With a [frm,to) epoch range,
    only crops seen in it (so a group's shown count is its sightings that day)."""
    def q(cur):
        if frm is not None and to is not None:
            cur.execute("""SELECT id, ts, score, NVL(cluster_id, id) grp FROM detections
                           WHERE identity_id IS NULL AND ts >= :1 AND ts < :2 ORDER BY ts DESC""", [frm, to])
        else:
            cur.execute("""SELECT id, ts, score, NVL(cluster_id, id) grp FROM detections
                           WHERE identity_id IS NULL ORDER BY ts DESC""")
        return cur.fetchall()
    groups = {}
    for did, ts, score, grp in db(q):
        did, grp, ts, score = int(did), int(grp), float(ts), round(float(score), 2)
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
                lim = min(int(q.get("limit", [UNKNOWN_LIMIT])[0]), UNKNOWN_LIMIT)
                frm, to = _range(q)
                return self.send(200, json.dumps(unknowns(lim, frm, to)))
            if u.path == "/detapi/person":
                frm, to = _range(q)
                return self.send(200, json.dumps(person(int(q["id"][0]), frm, to)))
            if u.path == "/detapi/group":
                frm, to = _range(q)
                return self.send(200, json.dumps(group(int(q["id"][0]), frm, to)))
            if u.path == "/detapi/times":
                frm, to = _range(q)
                if frm is None:
                    return self.err(400, "from/to required")
                return self.send(200, json.dumps(det_times(frm, to)))
            if u.path.startswith("/detapi/crop/") and u.path.endswith(".jpg"):
                det_id = int(u.path[len("/detapi/crop/"):-4])
                p = crop_file(det_id)
                if p is None:
                    return self.err(404, "no crop")
                with open(p, "rb") as f:
                    return self.send(200, f.read(), "image/jpeg",
                                     {"Cache-Control": "private, max-age=86400"})
            return self.err(404, "not found")
        except (KeyError, ValueError) as e:
            return self.err(400, str(e))
        except Exception as e:
            return self.err(500, str(e))

    def do_POST(self):
        u = urlparse(self.path)
        try:
            b = self.body_json()
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
            return self.err(404, "not found")
        except (KeyError, ValueError) as e:
            return self.err(400, str(e))
        except Exception as e:
            return self.err(500, str(e))


if __name__ == "__main__":
    print(f"detect-api on 127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
