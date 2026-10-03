#!/usr/bin/env python3
"""
One-off: splits what all cameras shared into per-camera files (see recordings.py):
  .meta/people-{state,keys,feedback-snap,names}.json -> <stream>/.meta/
  state/people-feedback.json                         -> state/people-feedback-<stream>.json
The old shared files (and index.json, people.json, live-dvr.json, .live, .live-dvr) go to
BACKUP; recordings.py, people.py and live_dvr.py write the per-camera ones again.
Run as root with cron paused:  sudo python3 migrate_perstream.py [--dry]
"""
import json
import os
import pwd
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recordings as R  # noqa: E402

DRY = "--dry" in sys.argv
REC, META, STATE = R.REC_DIR, R.META, os.path.join(R.HERE, "state")
BACKUP = os.path.expanduser("~ubuntu/rtmp-recordings-backup-20260925/before-perstream/shared")
UBUNTU = pwd.getpwnam("ubuntu")


def load(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save(path, data, owner=None, mtime=None):
    print(f"  {os.path.relpath(path, os.path.dirname(REC))}")
    if DRY:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    if owner:
        os.chown(path, owner.pw_uid, owner.pw_gid)
    if mtime:
        os.utime(path, (mtime, mtime))


def of_pid(pid):
    return pid.rsplit("-p", 1)[0]


def of_key(key):
    return R.parse_name(key.split("#", 1)[0][:-4])[0]


streams = sorted({s for s in R.all_streams()} | {R.parse_name(n[:-4])[0] for n in R.all_mp4s()})
print("streams:", streams)

state = load(os.path.join(META, "people-state.json"), {})
keys = load(os.path.join(META, "people-keys.json"), {})
snap = load(os.path.join(META, "people-feedback-snap.json"), {})
names = load(os.path.join(META, "people-names.json"), None)
fb_path = os.path.join(STATE, "people-feedback.json")
fb = load(fb_path, {})
mark_stream = {m["id"]: m.get("stream") or of_pid(m["a"]) for m in fb.get("same", [])}

for s in streams:
    print(s)
    m = os.path.join(REC, s, ".meta")
    if state:
        save(os.path.join(m, "people-state.json"), {
            "next": {k: v for k, v in state.get("next", {}).items() if k == s},
            "assign": {k: v for k, v in state.get("assign", {}).items() if of_pid(v) == s},
            "files": {f: t for f, t in state.get("files", {}).items() if R.parse_name(f[:-4])[0] == s}})
    if keys:
        save(os.path.join(m, "people-keys.json"), {p: ks for p, ks in keys.items() if of_pid(p) == s})
    if snap:
        save(os.path.join(m, "people-feedback-snap.json"),
             {i: v for i, v in snap.items() if mark_stream.get(i) == s})
    if names:
        save(os.path.join(m, "people-names.json"),
             {"names": {p: n for p, n in names.get("names", {}).items() if of_pid(p) == s},
              "merge": {a: b for a, b in names.get("merge", {}).items() if of_pid(a) == s}})
    mine = {"same": [e for e in fb.get("same", []) if mark_stream[e["id"]] == s],
            "names": {p: e for p, e in fb.get("names", {}).items() if of_pid(p) == s}}
    if mine["same"] or mine["names"]:
        save(os.path.join(STATE, f"people-feedback-{s}.json"), mine, UBUNTU, os.path.getmtime(fb_path))
    print(f"    {len(mine['same'])} mark(s), {len(mine['names'])} name(s)")

old = [os.path.join(META, f) for f in ("people-state.json", "people-keys.json", "people-feedback-snap.json",
                                       "people-names.json", ".people.lock")]
old += [os.path.join(META, f) for f in os.listdir(META) if f.startswith((".analyse-", ".motion-"))]
old += [os.path.join(REC, f) for f in ("index.json", "people.json", "live-dvr.json", ".live", ".live-dvr")]
old += [fb_path]
old = [p for p in old if os.path.exists(p)]
print(f"to {BACKUP}:", ", ".join(os.path.basename(p) for p in old))
if not DRY:
    os.makedirs(BACKUP, exist_ok=True)
    for p in old:
        shutil.move(p, os.path.join(BACKUP, os.path.basename(p)))
