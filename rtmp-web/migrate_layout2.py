#!/usr/bin/env python3
"""
One-off: <day>/<recording>/ folders -> <stream>/<day>/<recording>/ (see recordings.py).
Names don't change, so the people grouping's state stays as it is. Updates the paths
cached in each info.json. Run as root with cron paused:  sudo python3 migrate_layout2.py [--dry]
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recordings as R  # noqa: E402

DRY = "--dry" in sys.argv
n = 0
for d in sorted(glob.glob(os.path.join(R.REC_DIR, "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]", "*"))):
    name = os.path.basename(d) + ".mp4"
    dest = R.rec_dir(name)
    assert not os.path.exists(dest), dest
    if n < 3:
        print(f"{R.rel(d)} -> {R.rel(dest)}")
    n += 1
    if DRY:
        continue
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    os.rename(d, dest)
    info = R.read_info(name)
    if info.get("path"):
        info["path"] = R.rel(R.mp4_path(name))
    if info.get("thumb"):
        info["thumb"] = R.rel(os.path.join(dest, "thumb.jpg"))
    R.write_info(name, info)
if not DRY:
    for day in glob.glob(os.path.join(R.REC_DIR, "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")):
        os.rmdir(day)  # empty now
print(f"{n} recordings" + (" (dry run)" if DRY else " moved"))
