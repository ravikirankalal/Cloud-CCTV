#!/usr/bin/env python3
"""
One-off: moves finished recordings from the flat layout (x.mp4 at the top, with
.thumbs/x.jpg, .meta/x.mp4.{json,fix.json,motion.json,detect.json}, .people/x-E-P.jpg)
into day / recording folders named after their start time (see recordings.py), and
renames them in the people grouping's state so ids, names and merges stay put.

Run as root with cron paused and no analyser running:
  sudo python3 migrate_layout.py --dry     # what it would do
  sudo python3 migrate_layout.py
Leftovers (caches of recordings already deleted) go to BACKUP, not away.
"""
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recordings as R  # noqa: E402

DRY = "--dry" in sys.argv
REC, META = R.REC_DIR, R.META
BACKUP = os.path.expanduser("~ubuntu/rtmp-recordings-backup-20260925/layout-leftovers")
FEEDBACK = os.path.join(R.HERE, "state", "people-feedback.json")


def load(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save(path, data, keep_mtime=None):
    """Written in place, so the owner stays (people-feedback.json is ubuntu's)."""
    if DRY:
        return
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    if keep_mtime is not None:
        os.utime(path, (keep_mtime, keep_mtime))


def move(src, dst):
    if not DRY:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.rename(src, dst)  # same disk: keeps the file's times


def main():
    rename, n_crops = {}, 0
    olds = sorted(f for f in os.listdir(REC) if f.endswith(".mp4") and not f.startswith("."))
    for old in olds:
        stem = old[:-4]
        info = load(os.path.join(META, old + ".json"), {})
        fix = load(os.path.join(META, old + ".fix.json"))
        stream, named = R.parse_name(stem)
        start = info.get("start") or (fix or {}).get("start") or named
        if start is None:
            st = os.stat(os.path.join(REC, old))
            start = int(st.st_mtime - (info.get("duration") or 0))
        new = R.rec_name(stream, start) + ".mp4"
        assert new not in rename.values() and not os.path.exists(R.mp4_path(new)), f"clash: {new}"
        rename[old] = new
        d = R.rec_dir(new)
        move(os.path.join(REC, old), R.mp4_path(new))
        thumb = os.path.join(REC, ".thumbs", stem + ".jpg")
        has_thumb = os.path.exists(thumb)
        if has_thumb:
            move(thumb, os.path.join(d, "thumb.jpg"))
        if info:
            info.update(file=new, path=R.rel(R.mp4_path(new)),
                        thumb=R.rel(os.path.join(d, "thumb.jpg")) if has_thumb else None)
        if fix:
            info["fix"] = fix
        if not DRY:
            os.makedirs(d, exist_ok=True)
        save(os.path.join(d, "info.json"), info)
        for f in (old + ".json", old + ".fix.json"):
            if os.path.exists(os.path.join(META, f)) and not DRY:
                os.remove(os.path.join(META, f))
        m = os.path.join(META, old + ".motion.json")
        if os.path.exists(m):
            move(m, os.path.join(d, "motion.json"))
        det_path = os.path.join(META, old + ".detect.json")
        det = load(det_path)
        if det is not None:
            mtime = os.path.getmtime(det_path)  # people.py places again what changed
            for ev, people in enumerate(det.get("people", [])):
                for k, p in enumerate(people):
                    crop = os.path.join(REC, ".people", p.get("thumb") or "")
                    if p.get("thumb") and os.path.isfile(crop):
                        move(crop, os.path.join(d, "people", f"{ev}-{k}.jpg"))
                        p["thumb"] = f"{ev}-{k}.jpg"
                        n_crops += 1
                    else:
                        p.pop("thumb", None)
            save(os.path.join(d, "detect.json"), det, keep_mtime=mtime)
            if not DRY:
                os.remove(det_path)
    print(f"{len(rename)} recordings, {n_crops} person crops" + (" (dry run)" if DRY else ""))
    for old, new in list(rename.items())[:3] + list(rename.items())[-2:]:
        print(f"  {old} -> {R.rel(R.mp4_path(new))}")

    # the people grouping's sighting keys are "<file>#<event>#<person>"
    def key(k):
        f, _, rest = k.partition("#")
        return rename.get(f, f) + "#" + rest

    state_path = os.path.join(META, "people-state.json")
    state = load(state_path, {})
    if state:
        state["assign"] = {key(k): v for k, v in state.get("assign", {}).items()}
        state["files"] = {rename.get(f, f): t for f, t in state.get("files", {}).items()}
        save(state_path, state)
    keys_path = os.path.join(META, "people-keys.json")
    keys = load(keys_path, {})
    save(keys_path, {pid: [key(k) for k in ks] for pid, ks in keys.items()})
    fb = load(FEEDBACK, {})
    for e in fb.get("same", []):
        for side in ("keys_a", "keys_b"):
            if side in e:
                e[side] = [key(k) for k in e[side]]
    for e in fb.get("names", {}).values():
        if "keys" in e:
            e["keys"] = [key(k) for k in e["keys"]]
    if fb:
        save(FEEDBACK, fb, keep_mtime=os.path.getmtime(FEEDBACK))  # not a new mark
    snap_path = os.path.join(META, "people-feedback-snap.json")
    snap = load(snap_path, {})
    for sides in snap.values():
        for side in sides.values():
            for s in side:
                s[0] = rename.get(s[0], s[0])
    save(snap_path, snap)
    save(os.path.join(META, "renamed.json"), rename)  # old name -> new, for reference

    # caches of recordings deleted before (their MP4 is gone), and the emptied folders
    left = [os.path.join(META, f) for f in os.listdir(META)
            if f.endswith((".mp4.json", ".mp4.fix.json", ".mp4.motion.json", ".mp4.detect.json"))]
    left += [d for d in (os.path.join(REC, ".thumbs"), os.path.join(REC, ".people")) if os.path.isdir(d)]
    print(f"to {BACKUP}: {len(left)} leftover(s)"
          + "".join(f"\n  {os.path.relpath(p, REC)}" + (f" ({len(os.listdir(p))} files)" if os.path.isdir(p) else "")
                    for p in left[:8]))
    if not DRY:
        os.makedirs(BACKUP, exist_ok=True)
        for p in left:
            shutil.move(p, os.path.join(BACKUP, os.path.basename(p)))


if __name__ == "__main__":
    main()
