# jobstat.py — systemd unit status + log tail for the ops dashboard, shared by both boxes.
# Each API passes its OWN allowlist of units (the recorder VM via playback_api, the detect VM via
# detect_api); only a unit from that fixed set is ever named to systemctl/journalctl, so no
# caller-supplied string reaches a subprocess (and we never use shell=True — every call is an argv list).
import re
import subprocess
import time

_LOGLINE = re.compile(r"^(\S+)\s+\S+\s+[^:]+:\s?(.*)$")   # "<iso> <host> <comm>[pid]: <msg>" -> iso, msg


def _run(argv, timeout=6):
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def unit_logs(unit, allowed, n=120):
    if unit not in allowed:
        return []
    out = _run(["journalctl", "-u", unit, "-n", str(n), "--no-pager", "-o", "short-iso"])
    lines = []
    for ln in out.splitlines():
        if not ln or ln.startswith("-- "):            # drop "-- No entries --" / boot markers
            continue
        m = _LOGLINE.match(ln)
        lines.append(f"{m.group(1)} {m.group(2)}" if m else ln)
    return lines


def job_status(unit, allowed, tail=6):
    p = {}
    for ln in _run(["systemctl", "show", unit, "-p",
                    "ActiveState,SubState,NRestarts,MemoryCurrent,CPUUsageNSec,"
                    "ActiveEnterTimestampMonotonic,Description,MainPID"]).splitlines():
        k, _, v = ln.partition("=")
        p[k] = v

    def num(k):
        try:
            return int(p.get(k, ""))
        except ValueError:
            return None

    mem, cpu = num("MemoryCurrent"), num("CPUUsageNSec")
    entered = num("ActiveEnterTimestampMonotonic")
    active = p.get("ActiveState") == "active"
    uptime = None
    if active and entered:
        uptime = round(time.clock_gettime(time.CLOCK_MONOTONIC) - entered / 1e6)
    return {
        "unit": unit, "active": p.get("ActiveState"), "sub": p.get("SubState"),
        "desc": p.get("Description"), "restarts": num("NRestarts"), "pid": num("MainPID") or None,
        "uptime_s": uptime if uptime and uptime > 0 else None,
        "mem_mb": round(mem / 1048576, 1) if mem is not None and mem < (1 << 62) else None,
        "cpu_s": round(cpu / 1e9) if cpu else None,
        "tail": unit_logs(unit, allowed, tail),
    }
