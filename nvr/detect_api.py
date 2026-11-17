#!/usr/bin/env python3
# People / feedback API for person detection — the read + tagging side of the detection store.
#
# DECOUPLED by design (same rule as the poller): reads the Oracle 23ai vector DB and nvr.db
# READ-ONLY, writes ONLY to Oracle (labels/identities via match.py), imports no recorder code and
# opens no recorder sockets. It is deliberately LIGHT: it never loads onnxruntime/numpy — person
# crops are cut straight from the recorded segment with ffmpeg + PIL, so this service stays small
# on the tiny box while the heavy detector runs separately.
#
#   python3 detect_api.py            (nvr-detect-api.service; 127.0.0.1:PORT, nginx /detapi/ in front)
#
#   GET  /detapi/people                 known people: [{id,name,count,sample}]
#   GET  /detapi/unknowns?limit=N       recent un-identified detections: [{id,ts,seg_id,score}]
#   GET  /detapi/crop/<id>.jpg          the person crop for one detection (decoded on demand, cached)
#   POST /detapi/tag     {id,name}      name a detection -> known identity, then re-match history
#   POST /detapi/reject  {id}           clear a wrong auto-label
import io
import json
import os
import subprocess
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import oracledb
from PIL import Image

import adb
import match

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("NVR_DATA", "/home/ubuntu/nvr-data")
PORT = int(os.environ.get("DETAPI_PORT", "3401"))
CROP_DIR = os.path.join(DATA, "crops")
CROP_MAX = 320            # longest side of a cached crop jpeg
CROP_PAD = 0.12           # fraction of box size added around the person for context
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


# ---- read-only lookup of the recorded segment a detection came from ----
def ro_nvr():
    return sqlite3.connect(f"file:{os.path.join(DATA, 'nvr.db')}?mode=ro&immutable=1", uri=True)


def _seg_files(seg_id):
    with ro_nvr() as c:
        row = c.execute("SELECT path, init FROM segments WHERE id=?", (seg_id,)).fetchone()
    if not row:
        return None
    return os.path.join(DATA, row[1]), os.path.join(DATA, row[0])   # init, seg


# ---- crop a person out of the segment's first frame (ffmpeg + PIL, no models) ----
def _decode_crop(init, seg, box):
    with open(init, "rb") as f1, open(seg, "rb") as f2:
        data = f1.read() + f2.read()
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-frames:v", "1",
                          "-f", "image2", "-c:v", "png", "pipe:1"],
                         input=data, capture_output=True).stdout
    if not out:
        raise RuntimeError("decode failed")
    im = Image.open(io.BytesIO(out)).convert("RGB")
    W, H = im.size
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1) * CROP_PAD, (y2 - y1) * CROP_PAD
    x1, y1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
    x2, y2 = min(W, int(x2 + pw)), min(H, int(y2 + ph))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError("empty box")
    crop = im.crop((x1, y1, x2, y2))
    crop.thumbnail((CROP_MAX, CROP_MAX))
    buf = io.BytesIO()
    crop.save(buf, "JPEG", quality=82)
    return buf.getvalue()


def crop_jpeg(det_id):
    """Cached JPEG bytes of detection det_id's person crop, or None if it can't be made."""
    cache = os.path.join(CROP_DIR, f"{det_id}.jpg")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return f.read()

    def _lookup(cur):
        cur.execute("SELECT seg_id, box FROM detections WHERE id=:1", [det_id])
        return cur.fetchone()
    row = db(_lookup)
    if not row or row[0] is None or not row[1]:
        return None
    seg_id, box = int(row[0]), [int(v) for v in row[1].split(",")]
    files = _seg_files(seg_id)
    if not files:
        return None
    jpg = _decode_crop(files[0], files[1], box)
    tmp = cache + ".tmp"
    with open(tmp, "wb") as f:
        f.write(jpg)
    os.replace(tmp, cache)
    return jpg


# ---- queries ----
def people():
    def q(cur):
        cur.execute("""SELECT i.id, i.name, COUNT(d.id),
                              MAX(CASE WHEN d.confirmed=1 THEN d.id END), MAX(d.id)
                       FROM identities i LEFT JOIN detections d ON d.identity_id = i.id
                       WHERE i.kind = 'known'
                       GROUP BY i.id, i.name ORDER BY LOWER(i.name)""")
        return [{"id": int(r[0]), "name": r[1], "count": int(r[2]),
                 "sample": int(r[3] or r[4]) if (r[3] or r[4]) else None} for r in cur.fetchall()]
    return db(q)


def unknowns(limit):
    def q(cur):
        cur.execute("""SELECT id, ts, seg_id, score FROM detections
                       WHERE identity_id IS NULL
                       ORDER BY ts DESC FETCH FIRST :1 ROWS ONLY""", [limit])
        return [{"id": int(r[0]), "ts": float(r[1]), "seg_id": int(r[2]) if r[2] else None,
                 "score": round(float(r[3]), 2)} for r in cur.fetchall()]
    return db(q)


def tag(det_id, name):
    def w(cur):
        ident = match.tag(cur, det_id, name)
        changed = match.rematch(cur)
        return {"ok": True, "identity": ident, "rematched": changed}
    return db(w, write=True)


def reject(det_id):
    return db(lambda cur: (match.reject(cur, det_id), {"ok": True})[1], write=True)


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
                jpg = crop_jpeg(det_id)
                if jpg is None:
                    return self.err(404, "no crop")
                return self.send(200, jpg, "image/jpeg",
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
