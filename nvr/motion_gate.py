#!/usr/bin/env python3
# Background-subtraction motion gate for the person detector.
#
# WHY: on this box a segment's frame costs ~1 s to decode and ~0.9 s to run YOLOX. Decode is codec-bound
# (can't shrink), but the YOLOX pass is pure waste on the common empty frame. This gate decides, from the
# already-decoded frame and a learned model of the EMPTY scene, whether anything is there — and only then
# do we pay for YOLOX + re-ID. Decode happens either way; the gate saves the YOLOX pass on empty frames.
#
# WHY A BACKGROUND MODEL (not frame differencing or compressed-domain size): a fixed camera's EMPTY scene
# is not static — daytime auto-exposure and wind-blown foliage change ~a third of the pixels between the
# sparse (~20 s) samples, which swamps a plain diff (and killed byte-size / P-frame gating, measured
# 2026-11-19). The model here is a per-pixel single Gaussian (running mean + variance): a pixel is
# foreground only when it deviates from ITS OWN learned range by > k·std, so a flickering leaf (high
# variance) is tolerated while a person (a new, consistent value) stands out. A standing person is still
# foreground vs the learned empty scene, so stationary people are not missed. Selective update (foreground
# pixels learn slowly) keeps a lingering person from being absorbed; the background tracks slow light drift.
#
# WHY MORPHOLOGICAL OPENING + A BLOB-SIZE GATE: after the Gaussian test, empty frames still leave SCATTERED
# foreground specks (sensor noise, leaf edges) while a person is a COMPACT blob. A 3x3 opening (erode then
# dilate) deletes the specks and keeps the blob; the gate then fires on the surviving pixel count. Validated
# 2026-11-19 on a daytime range (3 distinct visits): every visit caught (100% visit-recall) while ~70-90%
# of empty segments are skipped, tunable by `min_px`.
#
# DECOUPLED like the rest of detection: a pure function of decoded frames plus a small on-disk state file;
# no recorder code, no live path. Cheap: one downsample + numpy on a ~160x90 grid (a few ms).
import os
import numpy as np
from PIL import Image


def _erode(m):
    """3x3 binary erosion (4-connected): a pixel survives only if it and its N/S/E/W neighbours are set."""
    e = m.copy()
    e[1:, :] &= m[:-1, :]; e[:-1, :] &= m[1:, :]
    e[:, 1:] &= m[:, :-1]; e[:, :-1] &= m[:, 1:]
    return e


def _dilate(m):
    d = m.copy()
    d[1:, :] |= m[:-1, :]; d[:-1, :] |= m[1:, :]
    d[:, 1:] |= m[:, :-1]; d[:, :-1] |= m[:, 1:]
    return d


class MotionGate:
    """Per-pixel single-Gaussian background gate for a fixed camera.

    check(frame_rgb) -> (motion: bool, score: int) where score is the count of foreground pixels on the
    downsampled grid that survive a 3x3 opening (i.e. belong to a compact blob). `motion` is score >=
    `min_px`. The model (per-pixel mean + variance) updates every call: background pixels at `alpha`,
    foreground pixels at `light_alpha` so a person present for a while is not learned into the background."""

    def __init__(self, path=None, grid_w=160, k=2.5, floor=6.0, alpha=0.05, light_alpha=0.004,
                 min_px=140, var_init=60.0):
        self.path = path
        self.grid_w = int(grid_w)
        self.k = float(k)
        self.floor = float(floor)
        self.alpha = float(alpha)
        self.light_alpha = float(light_alpha)
        self.min_px = int(min_px)
        self.var_init = float(var_init)
        self.mean = None      # float32 [gh, grid_w]
        self.var = None
        if path and os.path.exists(path):
            try:
                z = np.load(path)
                self.mean, self.var = z["mean"].astype(np.float32), z["var"].astype(np.float32)
            except Exception:
                self.mean = self.var = None

    def _small(self, frame_rgb):
        h, w = frame_rgb.shape[:2]
        gh = max(1, round(self.grid_w * h / w))
        return np.asarray(Image.fromarray(frame_rgb).convert("L").resize((self.grid_w, gh)), np.float32)

    def check(self, frame_rgb):
        """Gate one frame and fold it into the model. Returns (motion: bool, score: int)."""
        g = self._small(frame_rgb)
        if self.mean is None or self.mean.shape != g.shape:
            self.mean = g.copy()
            self.var = np.full_like(g, self.var_init)
            return True, self.min_px                   # first frame of a fresh model: always process
        std = np.sqrt(self.var)
        mask = np.abs(g - self.mean) > (self.k * std + self.floor)
        score = int(_dilate(_erode(mask)).sum())
        # selective update: background pixels adapt fast, foreground pixels only slowly (illumination)
        a = np.where(mask, self.light_alpha, self.alpha).astype(np.float32)
        d = g - self.mean
        self.mean = self.mean + a * d
        self.var = (1.0 - a) * self.var + a * d * d
        return score >= self.min_px, score

    def save(self):
        """Persist the model so a poller restart keeps its learned scene. Atomic, best-effort."""
        if not self.path or self.mean is None:
            return
        tmp = self.path + ".tmp.npz"
        try:
            with open(tmp, "wb") as f:
                np.savez(f, mean=self.mean, var=self.var)
            os.replace(tmp, self.path)
        except Exception:
            pass
