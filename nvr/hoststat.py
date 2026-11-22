# hoststat.py — host-level stats (load, memory, disk, uptime) for the ops dashboard.
# Pure stdlib: reads /proc and statvfs only. No camera/recorder/Oracle coupling, so it is
# safe to import on either box (the recorder VM via playback_api, the detect VM via detect_api).
import os
import time


def host_stats(disk="/"):
    """A snapshot of this machine: load average, cpu count, memory and disk use, uptime. Every field is
    independently guarded so a missing /proc entry degrades to None instead of failing the whole call."""
    s = {"now": time.time(), "ncpu": os.cpu_count()}
    try:
        with open("/proc/loadavg") as f:
            p = f.read().split()
        s["load"] = [float(p[0]), float(p[1]), float(p[2])]
    except Exception:
        s["load"] = None
    try:
        mi = {}
        with open("/proc/meminfo") as f:
            for ln in f:
                k, _, v = ln.partition(":")
                mi[k] = int(v.split()[0])                      # kB
        total, avail = mi.get("MemTotal", 0), mi.get("MemAvailable", 0)
        s["mem_total_mb"], s["mem_used_mb"] = round(total / 1024), round((total - avail) / 1024)
    except Exception:
        s["mem_total_mb"] = s["mem_used_mb"] = None
    try:
        v = os.statvfs(disk)
        tot, free = v.f_blocks * v.f_frsize, v.f_bavail * v.f_frsize
        s["disk_total_gb"], s["disk_used_gb"] = round(tot / 1e9, 1), round((tot - free) / 1e9, 1)
    except Exception:
        s["disk_total_gb"] = s["disk_used_gb"] = None
    try:
        with open("/proc/uptime") as f:
            s["uptime_s"] = int(float(f.read().split()[0]))
    except Exception:
        s["uptime_s"] = None
    return s
