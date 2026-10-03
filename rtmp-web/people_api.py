#!/usr/bin/env python3
"""
Saves corrections from the page's People row: "these two are the same person" (and undoing
that). people.py applies them, and learns from them, on its next run (recordings.py starts
it within a minute of a change).

  POST /api/people/same      {"a": "cam1-p3", "b": "cam1-p7"}
  POST /api/people/separate  {"id": "cam1-p3"}    undo the marks that joined others into cam1-p3
  POST /api/people/name      {"id": "cam1-p3", "name": "Asha"}   ("" removes the name)
  POST /api/people/object    {"id": "cam1-p3"}    not a person (a poster...); "undo": true takes it back

The marks and names go to state/people-feedback-<stream>.json, one file per camera (the
stream is the start of the id). Each keeps the sightings of the people as they were
grouped when it was made (from <stream>/.meta/people-keys.json), so it still means the
same people after the grouping changes.

Listens on a Unix socket in run/ (this folder is mounted into the nginx container, which
proxies /api/people/ to it), so nothing new is opened on the network. Runs as a service
(see people-api.service):

  sudo cp /home/ubuntu/rtmp-web/people-api.service /etc/systemd/system/
  sudo systemctl daemon-reload && sudo systemctl enable --now people-api
"""
import json
import os
import re
import secrets
import socketserver
import time
from http.server import BaseHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
REC_DIR = os.environ.get("REC_DIR", "/home/ubuntu/rtmp-recordings")
SOCK = os.path.join(HERE, "run", "people-api.sock")
STATE_DIR = os.path.join(HERE, "state")
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}-p\d{1,6}$")
MAX_BODY = 1024
MAX_MARKS = 2000
MAX_NAME = 40


def load(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def stream_of(pid):
    return pid.rsplit("-p", 1)[0]


# one camera's files (pid: one of its people's ids, already checked against ID_RE)
def feedback(pid):
    return os.path.join(STATE_DIR, f"people-feedback-{stream_of(pid)}.json")


def keys_of(pid):
    return load(os.path.join(REC_DIR, stream_of(pid), ".meta", "people-keys.json"), {})


def people_of(pid):
    return load(os.path.join(REC_DIR, stream_of(pid), "people.json"), {})


def save(pid, data):
    path = feedback(pid)
    with open(path + ".tmp", "w") as f:
        json.dump(data, f, separators=(",", ":"))
    os.replace(path + ".tmp", path)


def same(body):
    a, b = body.get("a"), body.get("b")
    if not (isinstance(a, str) and isinstance(b, str) and ID_RE.match(a) and ID_RE.match(b)):
        return 400, {"error": "bad ids"}
    if a == b or stream_of(a) != stream_of(b):
        return 400, {"error": "pick two different people from the same camera"}
    keys = keys_of(a)
    if a not in keys or b not in keys:
        return 409, {"error": "those people have changed; reload the page"}
    fb = load(feedback(a), {"same": []})
    dup = next((m for m in fb["same"] if {m["a"], m["b"]} == {a, b}), None)
    if dup:  # marked already (the page may not show it yet)
        return 200, {"ok": True, "mark": dup["id"]}
    if len(fb["same"]) >= MAX_MARKS:
        return 409, {"error": "too many marks"}
    mark = {"id": secrets.token_hex(6), "a": a, "b": b, "stream": stream_of(a),
            "keys_a": keys[a], "keys_b": keys[b], "at": int(time.time())}
    fb["same"].append(mark)
    save(a, fb)
    return 200, {"ok": True, "mark": mark["id"]}


def name(body):
    pid, text = body.get("id"), body.get("name")
    if not (isinstance(pid, str) and ID_RE.match(pid) and isinstance(text, str)):
        return 400, {"error": "bad request"}
    text = " ".join("".join(c for c in text if c.isprintable()).split())[:MAX_NAME]
    keys = keys_of(pid)
    if pid not in keys:
        return 409, {"error": "that person has changed; reload the page"}
    fb = load(feedback(pid), {"same": []})
    names = fb.setdefault("names", {})
    # a name given under another id to (most of) this person, e.g. before a merge, is replaced
    mine = set(keys[pid])
    for other, e in list(names.items()):
        k = e.get("keys", [])
        if other == pid or (k and 2 * len(mine.intersection(k)) > len(k)):
            names.pop(other)
    if text:
        names[pid] = {"name": text, "keys": keys[pid], "at": int(time.time())}
    save(pid, fb)
    return 200, {"ok": True, "name": text}


def not_person(body):
    """{"id": ...}: that "person" is a poster, a sack, a statue: people.py hides it for good
    (and whatever new sightings join it). {"id": ..., "undo": true} takes the mark back."""
    pid = body.get("id")
    if not (isinstance(pid, str) and ID_RE.match(pid)):
        return 400, {"error": "bad id"}
    fb = load(feedback(pid), {"same": []})
    objs = fb.setdefault("objects", {})
    if body.get("undo"):
        if objs.pop(pid, None) is None:
            return 404, {"error": "not marked"}
    else:
        keys = keys_of(pid)
        if pid not in keys:
            return 409, {"error": "that person has changed; reload the page"}
        if len(objs) >= MAX_MARKS:
            return 409, {"error": "too many marks"}
        objs[pid] = {"keys": keys[pid], "at": int(time.time())}
    save(pid, fb)
    return 200, {"ok": True}


def separate(body):
    pid = body.get("id")
    if not (isinstance(pid, str) and ID_RE.match(pid)):
        return 400, {"error": "bad id"}
    person = next((p for p in people_of(pid).get("people", []) if p["id"] == pid), None)
    marks = set((person or {}).get("joined", []))
    fb = load(feedback(pid), {"same": []})
    left = [m for m in fb["same"] if m["id"] not in marks and pid not in (m["a"], m["b"])]
    if len(left) == len(fb["same"]):
        return 404, {"error": "nothing to undo for this person"}
    fb["same"] = left
    save(pid, fb)
    return 200, {"ok": True}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BODY:
            return self.reply(400, {"error": "bad request"})
        try:
            body = json.loads(self.rfile.read(n))
            assert isinstance(body, dict)
        except (ValueError, AssertionError):
            return self.reply(400, {"error": "bad json"})
        route = {"/api/people/same": same, "/api/people/separate": separate,
                 "/api/people/name": name, "/api/people/object": not_person}.get(self.path)
        if not route:
            return self.reply(404, {"error": "not found"})
        self.reply(*route(body))

    def reply(self, code, data):
        out = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def address_string(self):  # Unix socket: no client address
        return self.headers.get("X-Real-IP", "-") if hasattr(self, "headers") else "-"


class Server(socketserver.UnixStreamServer):
    def get_request(self):
        conn, _ = self.socket.accept()
        return conn, ("", 0)


def main():
    os.makedirs(os.path.dirname(SOCK), exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)
    if os.path.exists(SOCK):
        os.remove(SOCK)
    with Server(SOCK, Handler) as srv:
        os.chmod(SOCK, 0o666)  # nginx in the container runs as another user
        srv.serve_forever()


if __name__ == "__main__":
    main()
