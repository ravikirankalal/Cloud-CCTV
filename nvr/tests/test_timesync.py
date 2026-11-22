#!/usr/bin/env python3
# Time-sync tests for the detector's presence/track/re-ID pipeline. The risk this guards: a person is
# SEEN on the SD sub-stream (~2 s segments) but RE-ID'd on the HD stream (~8.5 s segments), so every ts
# must map to the right HD segment + offset, boxes must scale SD->HD exactly, and the tracker's clock
# (driven by segment ts, not wall time) must open/close/cap tracks on the footage timeline. All of this
# is pure time/geometry math, so these run with NO models and NO DB: detect.* and the DB cursor are
# faked. Run on the server venv (has numpy/oracledb/onnxruntime libs) — nothing loads a model here.
#
#   /home/ubuntu/nvr/venv-detect/bin/python tests/test_timesync.py
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import detect_service as ds
detect = ds.detect

FAILS = []
def check(name, cond, extra=""):
    print(("  ok  " if cond else "FAIL  ") + name + ("" if cond else f"   <- {extra}"))
    if not cond:
        FAILS.append(name)
def close(a, b, eps=1e-6):
    return abs(float(a) - float(b)) <= eps


# ---- fakes -----------------------------------------------------------------------------------------
class Row:
    def __init__(self, row): self._row = row
    def fetchone(self): return self._row
    def fetchall(self): return self._row if self._row else []

class FakeNvr:
    """Answers only the three segment queries the detector issues, from an in-memory segment list.
    Each seg is a dict(id, stream, start, dur, path, init)."""
    def __init__(self, segs): self.segs = segs
    def execute(self, sql, params):
        if "start <= ?" in sql:                                    # _reid_frame: covering seg for (stream, ts)
            stream, ts = params
            cand = [s for s in self.segs if s["stream"] == stream and s["start"] <= ts]
            if not cand:
                return Row(None)
            s = max(cand, key=lambda s: s["start"])
            return Row((s["id"], s["start"], s["dur"], s["path"], s["init"]))
        if "WHERE id = ?" in sql:                                  # by-id lookups
            sid = params[0]
            s = next((s for s in self.segs if s["id"] == sid), None)
            if not s:
                return Row(None)
            if "start, path, init, dur" in sql:                    # _presence
                return Row((s["start"], s["path"], s["init"], s["dur"]))
            return Row((s["start"], s["path"], s["init"]))         # _decode_at
        raise AssertionError("unexpected query: " + sql)

class FakeVar:
    def getvalue(self): return [4242]
class FakeCur:
    def __init__(self): self.inserts = []
    def var(self, t): return FakeVar()
    def execute(self, sql, params): self.inserts.append(params)


def patch(**kw):
    """Monkeypatch detect.* for the duration of a test; returns the originals to restore."""
    old = {k: getattr(detect, k) for k in kw}
    for k, v in kw.items():
        setattr(detect, k, v)
    return old
def unpatch(old):
    for k, v in old.items():
        setattr(detect, k, v)


# ---- 1. frame ts formula (_presence) ---------------------------------------------------------------
def test_frame_ts():
    seg = {"id": 1, "stream": "cam1-sd", "start": 1000.0, "dur": 2.1, "path": "p", "init": "i"}
    nvr = FakeNvr([seg])
    old = patch(decode_frames=lambda init, s, n, dur: [np.zeros((448, 800, 3), np.uint8)] * n,
                _frame_dims=lambda init: (800, 448),
                detect_in=lambda f: [{"box": [10, 10, 50, 100], "score": 0.9}])
    try:
        ds.FRAMES = 3
        start, wh, out = ds._presence(nvr, 1, None)
        check("frame-ts: 3 frames returned", len(out) == 3, len(out))
        exp = [1000.0 + i * 2.1 / 3 for i in range(3)]
        check("frame-ts: evenly spaced in [start,start+dur)", all(close(out[i][0], exp[i]) for i in range(3)), [o[0] for o in out])
        check("frame-ts: last frame before segment end", out[-1][0] < start + 2.1, out[-1][0])
        check("frame-ts: presence dims are SD", wh == (800, 448), wh)
        ds.FRAMES = 1
        _, _, out1 = ds._presence(nvr, 1, None)
        check("frame-ts: FRAMES=1 => ts == seg start", close(out1[0][0], 1000.0), out1[0][0])
    finally:
        unpatch(old); ds.FRAMES = 1


# ---- 2. ts -> HD segment mapping (_reid_frame) across different SD/HD boundaries --------------------
def test_reid_frame_mapping():
    # HD segs are long (8 s) and NOT aligned to the 2 s SD segment grid.
    hd = [{"id": 10, "stream": "cam1", "start": 1000.0, "dur": 8.0, "path": "h0", "init": "ih"},
          {"id": 11, "stream": "cam1", "start": 1008.0, "dur": 8.0, "path": "h1", "init": "ih"},
          {"id": 12, "stream": "cam1", "start": 1016.0, "dur": 8.0, "path": "h2", "init": "ih"}]
    nvr = FakeNvr(hd)
    calls = []
    old = patch(decode_frame=lambda init, seg, ss=None: calls.append(ss) or np.zeros((1296, 2304, 3), np.uint8))
    ds.REID_STREAM = "cam1"
    try:
        for ts, exp_seg, exp_off in [(1005.0, 10, 5.0), (1008.0, 11, 0.0), (1012.3, 11, 4.3), (1016.0, 12, 0.0)]:
            calls.clear()
            sid, frame = ds._reid_frame(nvr, ts)
            check(f"reid-map: ts={ts} -> HD seg {exp_seg}", sid == exp_seg, sid)
            check(f"reid-map: ts={ts} -> offset {exp_off}", calls and close(calls[0], exp_off), calls)
        # a ts past the last HD seg's end is a gap -> no re-ID frame (caller falls back to SD)
        sid, frame = ds._reid_frame(nvr, 1030.0)
        check("reid-map: ts in gap after last seg -> None", sid is None and frame is None, (sid, frame))
        # a ts before any HD seg starts -> None
        sid, frame = ds._reid_frame(nvr, 999.0)
        check("reid-map: ts before first seg -> None", sid is None, sid)
        # a ts inside a gap BETWEEN segs (make one with a short dur so there's a hole)
        nvr2 = FakeNvr([{"id": 20, "stream": "cam1", "start": 1000.0, "dur": 2.0, "path": "h", "init": "ih"}])
        sid, frame = ds._reid_frame(nvr2, 1005.0)   # 1005 > 1000+2 -> inside the hole
        check("reid-map: ts inside inter-seg gap -> None", sid is None, sid)
    finally:
        unpatch(old)


# ---- 3. SD->HD box scaling -------------------------------------------------------------------------
def test_scale_box():
    b = ds._scale_box([100, 50, 200, 250], (800, 448), (2304, 1296))
    fx, fy = 2304 / 800, 1296 / 448
    check("scale: x scaled by w ratio", close(b[0], 100 * fx) and close(b[2], 200 * fx), b)
    check("scale: y scaled by h ratio", close(b[1], 50 * fy) and close(b[3], 250 * fy), b)
    ident = ds._scale_box([10, 20, 30, 40], (800, 448), (800, 448))
    check("scale: same size => identity", ident == [10, 20, 30, 40], ident)


# ---- 4. _reid_boxes: HD present vs gap --------------------------------------------------------------
def test_reid_boxes():
    hd = [{"id": 10, "stream": "cam1", "start": 1000.0, "dur": 8.0, "path": "h", "init": "ih"}]
    nvr = FakeNvr(hd)
    old = patch(decode_frame=lambda init, seg, ss=None: np.zeros((1296, 2304, 3), np.uint8))
    ds.REID_STREAM = "cam1"
    try:
        rf, rseg, rb = ds._reid_boxes(nvr, 1004.0, [[100, 50, 200, 250]], (800, 448))
        check("reid-boxes: HD present -> HD seg", rseg == 10, rseg)
        check("reid-boxes: HD present -> scaled box", rb and close(rb[0][0], 100 * 2304 / 800), rb)
        rf2, rseg2, rb2 = ds._reid_boxes(nvr, 1030.0, [[1, 2, 3, 4]], (800, 448))
        check("reid-boxes: gap -> (None,None,None)", rf2 is None and rseg2 is None and rb2 is None, (rf2, rseg2, rb2))
    finally:
        unpatch(old)


# ---- 5-7. tracker association / close / cap --------------------------------------------------------
def _tracker():
    emitted = []
    tr = ds.Tracker(lambda obs, s0, s1: emitted.append((list(obs), s0, s1)))
    return tr, emitted

def test_tracker_assoc():
    ds.TRACK_IOU, ds.TRACK_GAP, ds.TRACK_MAX = 0.3, 8.0, 25.0
    tr, emitted = _tracker()
    tr.update(100.0, 1, [{"box": [0, 0, 100, 100], "score": 0.9}])
    tr.update(102.0, 1, [{"box": [10, 0, 110, 100], "score": 0.9}])   # overlaps -> same track
    check("assoc: overlapping boxes -> one track", len(tr.tracks) == 1, len(tr.tracks))
    check("assoc: track has both obs", len(tr.tracks[0]["obs"]) == 2, tr.tracks[0]["obs"])
    tr.update(104.0, 1, [{"box": [900, 900, 1000, 1000], "score": 0.9}])  # far apart -> new track
    check("assoc: disjoint box -> second track", len(tr.tracks) == 2, len(tr.tracks))

def test_tracker_close():
    ds.TRACK_IOU, ds.TRACK_GAP, ds.TRACK_MAX = 0.3, 8.0, 25.0
    tr, emitted = _tracker()
    tr.update(100.0, 1, [{"box": [0, 0, 100, 100], "score": 0.9}])
    tr.tick(105.0)                     # 5 s gap < 8 -> still open
    check("close: within gap stays open", len(tr.tracks) == 1 and not emitted, (len(tr.tracks), emitted))
    tr.tick(114.0)                     # 14 s since last obs > 8 -> close + emit
    check("close: stale track emitted", len(emitted) == 1, emitted)
    check("close: emitted span = obs span", emitted and close(emitted[0][1], 100.0) and close(emitted[0][2], 100.0), emitted)
    check("close: track removed", len(tr.tracks) == 0, tr.tracks)

def test_tracker_cap():
    ds.TRACK_IOU, ds.TRACK_GAP, ds.TRACK_MAX = 0.3, 8.0, 25.0
    tr, emitted = _tracker()
    box = {"box": [0, 0, 100, 100], "score": 0.9}
    for t in (100.0, 106.0, 112.0, 118.0, 124.0, 130.0):   # 6 s steps (< gap), total span 30 s (> cap 25)
        tr.update(t, 1, [box]); tr.tick(t)
    check("cap: a chunk was emitted mid-track", len(emitted) == 1, emitted)
    check("cap: chunk spans from first obs", emitted and close(emitted[0][1], 100.0), emitted)
    check("cap: track stays open after cap", len(tr.tracks) == 1, len(tr.tracks))
    check("cap: open track's chunk was reset", tr.tracks[0]["obs"] == [], tr.tracks[0]["obs"])
    # flush the continuation
    tr.tick(0.0, flush=True)
    check("cap: flush emits remaining open tracks only if they have obs", len(emitted) == 1, emitted)

def test_quality_bestshot():
    big_sure = ds._quality([0, 0, 100, 100], 0.8)       # area 10000
    small_surer = ds._quality([0, 0, 50, 50], 0.95)     # area 2500
    tiny = ds._quality([0, 0, 10, 10], 0.9)             # area 100
    check("bestshot: big+confident beats small+surer", big_sure > small_surer > tiny, (big_sure, small_surer, tiny))


# ---- 8. make_finalize: the stored row's ts / seg / span / box are all time-consistent ---------------
def test_finalize_row():
    ds.TRACK_IOU, ds.TRACK_GAP, ds.TRACK_MAX = 0.3, 8.0, 25.0
    ds.REID_STREAM = "cam1"
    hd = [{"id": 10, "stream": "cam1", "start": 1000.0, "dur": 8.0, "path": "h0", "init": "ih"},
          {"id": 11, "stream": "cam1", "start": 1008.0, "dur": 8.0, "path": "h1", "init": "ih"}]
    nvr = FakeNvr(hd)
    cur = FakeCur()
    old = patch(decode_frame=lambda init, seg, ss=None: np.zeros((1296, 2304, 3), np.uint8),
                embed=lambda f, boxes: np.ones((len(boxes), 512), np.float32),
                crop_to_jpeg=lambda f, b, **k: None)
    saved_match = ds.match_known
    ds.match_known = lambda c, e: None
    try:
        pres_wh_box = [(800, 448)]; stats = {"n": 0}
        finalize = ds.make_finalize([nvr], cur, pres_wh_box, stats)
        # best shot is the big box at ts=1002 (quality beats the others); span is [1000, 1007]
        obs = [(1000.0, 5, [0, 0, 10, 10], 0.95),
               (1002.0, 6, [0, 0, 100, 200], 0.80),    # best (big)
               (1007.0, 7, [0, 0, 40, 40], 0.90)]
        finalize(obs, 1000.0, 1007.0)
        check("finalize: exactly one row inserted", len(cur.inserts) == 1, len(cur.inserts))
        p = cur.inserts[0]   # (ts, seg_id, box, score, emb, ident, ts_start, ts_end, idv-bind omitted->0..8)
        ts, seg_id, boxstr, score, _emb, _ident, ts_start, ts_end = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]
        check("finalize: stored ts = best-shot ts", close(ts, 1002.0), ts)
        check("finalize: stored seg = HD seg covering best ts", seg_id == 10, seg_id)  # 1002 in seg 10 (1000-1008)
        check("finalize: ts_start/ts_end = track span", close(ts_start, 1000.0) and close(ts_end, 1007.0), (ts_start, ts_end))
        check("finalize: best ts within span", ts_start <= ts <= ts_end, (ts_start, ts, ts_end))
        # best box [0,0,100,200] scaled SD(800,448)->HD(2304,1296)
        xs = [int(v) for v in boxstr.split(",")]
        check("finalize: box scaled to HD", close(xs[2], int(100 * 2304 / 800)) and close(xs[3], int(200 * 1296 / 448)), xs)
        check("finalize: stats counted the emit", stats["n"] == 1, stats["n"])
    finally:
        unpatch(old); ds.match_known = saved_match


# ---- 9. SD-only fallback: no HD footage => re-ID + store against the SD seg at the same ts ----------
def test_finalize_sd_fallback():
    ds.REID_STREAM = "cam1"
    nvr = FakeNvr([{"id": 7, "stream": "cam1-sd", "start": 2000.0, "dur": 2.1, "path": "s", "init": "is"}])  # no HD at all
    cur = FakeCur()
    old = patch(decode_frame=lambda init, seg, ss=None: np.zeros((448, 800, 3), np.uint8),
                embed=lambda f, boxes: np.ones((len(boxes), 512), np.float32),
                crop_to_jpeg=lambda f, b, **k: None)
    saved_match = ds.match_known
    ds.match_known = lambda c, e: None
    try:
        finalize = ds.make_finalize([nvr], cur, [(800, 448)], {"n": 0})
        finalize([(2001.0, 7, [0, 0, 50, 100], 0.9)], 2001.0, 2001.0)
        check("sd-fallback: one row inserted", len(cur.inserts) == 1, len(cur.inserts))
        p = cur.inserts[0]
        check("sd-fallback: stored against the SD seg", p[1] == 7, p[1])
        check("sd-fallback: ts preserved", close(p[0], 2001.0), p[0])
        # box NOT scaled (SD frame used directly)
        xs = [int(v) for v in p[2].split(",")]
        check("sd-fallback: box kept in SD px (no scale)", xs == [0, 0, 50, 100], xs)
    finally:
        unpatch(old); ds.match_known = saved_match


if __name__ == "__main__":
    for t in (test_frame_ts, test_reid_frame_mapping, test_scale_box, test_reid_boxes,
              test_tracker_assoc, test_tracker_close, test_tracker_cap, test_quality_bestshot,
              test_finalize_row, test_finalize_sd_fallback):
        print(f"\n== {t.__name__} ==")
        t()
    print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILURE(S): {FAILS}"))
    sys.exit(1 if FAILS else 0)
