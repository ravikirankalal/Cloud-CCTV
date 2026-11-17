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
#   GET  /detapi/crop/<id>.jpg          the person crop for one detection (decoded on demand, cached)
#   POST /detapi/tag     {id,name}      name a detection -> known identity, then re-match history
#   POST /detapi/reject  {id}           clear a wrong auto-label
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
CROP_DIR = os.path.join(DATA, "crops")    # static crops written by the detector (detect_service)
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
def people():
    def q(cur):
        cur.execute("""SELECT i.id, i.name, COUNT(d.id), MIN(d.ts), MAX(d.ts)
                       FROM identities i LEFT JOIN detections d ON d.identity_id = i.id
                       WHERE i.kind = 'known'
                       GROUP BY i.id, i.name ORDER BY LOWER(i.name)""")
        return cur.fetchall()

    def pick_sample(cur, ident_id):
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


def unknowns(limit):
    """Unknown detections GROUPED by duplicate-cluster: one tile per person. Each entry's `id` is the
    representative (best-scoring crop) and `count` is how many detections the group holds. Tagging
    that id names the whole group (detect_service assigned the cluster_id)."""
    def q(cur):
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


def tag(det_id, name):
    def w(cur):
        ident = match.tag_cluster(cur, det_id, name)   # names the whole duplicate-group at once
        changed = match.rematch(cur)
        return {"ok": True, "identity": ident, "rematched": changed}
    return db(w, write=True)


def reject(det_id):
    return db(lambda cur: (match.dismiss_cluster(cur, det_id), {"ok": True})[1], write=True)


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
                return self.send(200, json.dumps(people()))
            if u.path == "/detapi/unknowns":
                lim = min(int(q.get("limit", [UNKNOWN_LIMIT])[0]), UNKNOWN_LIMIT)
                return self.send(200, json.dumps(unknowns(lim)))
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
                if not name or "id" not in b:
                    return self.err(400, "need id and name")
                return self.send(200, json.dumps(tag(int(b["id"]), name)))
            if u.path == "/detapi/reject":
                if "id" not in b:
                    return self.err(400, "need id")
                return self.send(200, json.dumps(reject(int(b["id"]))))
            return self.err(404, "not found")
        except (KeyError, ValueError) as e:
            return self.err(400, str(e))
        except Exception as e:
            return self.err(500, str(e))


if __name__ == "__main__":
    print(f"detect-api on 127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
