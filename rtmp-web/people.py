#!/usr/bin/env python3
"""
Groups the people detect.py saw into "people", like the People view in Google Photos:
each sighting (a person in a motion event, with an appearance fingerprint) joins the
person it looks most like, or starts a new one. It is categorisation, not identification:
the fingerprint is mostly clothes and build, so the same person in different clothes (or
in colour by day vs. infrared at night) can end up as two people.

Each camera (stream) is grouped on its own, with its own ids ("cam1-p3") and files: people
at one camera are never merged with people at another. Grouping is incremental: sightings keep
the person they were given (.meta/people-state.json), and each run only places the new
ones, a recording at a time, oldest first. A recording's sightings are clustered together
with the people they could belong to (average linkage on the fingerprints: the two most
alike groups merge while their average similarity is at least JOIN, a person counting as
the group of all its sightings); a group with none of those people becomes a new person.
People already there are never merged or split by this, so ids, names and merges stay
put, and a run costs about the same however many recordings there are. A recording that
is analysed again (new motion events, a new detector) has new sightings, numbered afresh:
recordings.py keeps the old results as detect.prev.json, and each old sighting is paired
with the new one it matches (see pair()); those keep their person, and marks and names
follow them (.meta/people-keymap.json translates the sighting keys the page saved). Only
the new ones without a match are placed. `people.py --rebuild` places all
sightings afresh (people keep their ids where most of their sightings agree), which
applies what has been learned (below) to the older recordings too.

Correcting it: on the page, "Same person as..." marks two people as one (people_api.py
writes that to state/people-feedback-<stream>.json). Each mark
  * joins the two for good (found again by their sightings, so it survives regrouping), and
  * teaches the grouping. The people you have named or merged at a camera are examples:
    two sightings of one of them are the same person, of two of them different people.
    From those it learns how much to trust three things when judging a pair (see learn()):
    how alike they look (the fingerprint), how alike their size in the frame is, and how
    close in time they were seen. It groups with that combination from then on, but
    joins groups only if they also look somewhat alike (LEARN_FLOOR), and never two
    people seen in the same motion event. Until there are enough examples, nothing
    changes. What's learned is used for new recordings (and by --rebuild).
Names given on the page (also in people-feedback-<stream>.json) stay with the person's
sightings. "Not a person" on the page (posters, sacks, statues the detector takes for a
person, which the box test in fixed() misses when they're in several places or the camera's
view shifts) hides whoever most of the marked sightings are now, like a fixed object.

People placing kept apart that are really one are joined after each run (consolidate()):
average linkage over whole people at JOIN, never two seen together in one event or two
named / marked differently. Names and hand-made merges still work in <stream>/.meta/people-names.json:

  {"names": {"cam1-p3": "Asha"}, "merge": {"cam1-p7": "cam1-p3"}}

Runs in the venv (numpy); recordings.py starts it after new detections:
  .venv/bin/python people.py STREAM [--rebuild] [--dry]
Writes <stream>/people.json next to the camera's index.json for the page; its state is in
<stream>/.meta.
"""
import base64
import glob
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

REC_DIR = os.environ.get("REC_DIR", "/home/ubuntu/rtmp-recordings")
HERE = os.path.dirname(os.path.abspath(__file__))
STREAM_OK = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def use(stream):
    """Point the file names below at one camera's files."""
    global SDIR, META, STATE, NAMES, KEYS, FEEDBACK, SNAP, KEYMAP, OUT
    SDIR = os.path.join(REC_DIR, stream)
    META = os.path.join(SDIR, ".meta")
    STATE = os.path.join(META, "people-state.json")
    NAMES = os.path.join(META, "people-names.json")
    KEYS = os.path.join(META, "people-keys.json")    # {person id: [sighting keys]}, for people_api.py
    FEEDBACK = os.path.join(os.environ.get("PEOPLE_FEEDBACK_DIR", os.path.join(HERE, "state")),
                            f"people-feedback-{stream}.json")
    SNAP = os.path.join(META, "people-feedback-snap.json")  # the sightings each mark was about
    KEYMAP = os.path.join(META, "people-keymap.json")  # sighting keys before / after re-detection
    OUT = os.path.join(SDIR, "people.json")
    os.makedirs(META, exist_ok=True)

JOIN = 0.75           # average cosine similarity for two groups to be one person
                      # (unrelated sightings average ~0.58 on this camera, the same person ~0.8+)
# Things the detector takes for a person (a statue by the plants, here) form a group whose
# boxes are always in the same place; real people move. Such groups are left out, and
# their events lose the "person" label (see recordings.py).
STATIC_MIN = 4        # sightings before a group can be judged
STATIC_IOU = 0.5      # median overlap of its boxes at which it is a fixed object
STATIC_SAMPLE = 40    # boxes compared at most (spread over the person's sightings)
STATIC_RECS = 3       # ...and only if seen in this many recordings over this many different
STATIC_HOURS = 3      # hours: a person sitting still is one sitting, a statue is there all day
LEARN_SAMPLE = 200    # sightings of a person the learned score is averaged over at most
# learning from "same person" marks
LEARN_PEOPLE = 2      # named / merged people (with LEARN_SEEN+ sightings) needed to learn
LEARN_SEEN = 3
LEARN_PAIRS = 20      # pairs of each kind (same / different person) needed
LEARN_CAP = 6000      # pairs of each kind used at most
LEARN_GAIN = 0.01     # the combination is used only if it beats looks alone by this much
                      # (score on examples it wasn't fitted to)
SNAP_MAX = 60         # sightings remembered per side of a mark
LEARN_FLOOR = 0.65    # what's learned may join groups only down to this raw similarity (the same
                      # person in other light ~0.68; different people seen alike ~0.60 or less)
SAME_SIGHTING = 0.9   # a remembered sighting is found again (after re-detection) at this similarity,
FIND_LOOSE = 0.75     # ... or at this one if it is clearly the best in its recording (by FIND_MARGIN):
FIND_MARGIN = 0.05    # crops of the same person from other frames are ~0.75-0.9 alike
# pairing a recording's sightings before and after it is analysed again (one to one)
PAIR_SURE = 0.8       # this alike: the same sighting
PAIR_NEAR = 0.65      # ... or this alike and seen within PAIR_DT seconds of each other
PAIR_DT = 20
DETECT_V = (2, 3)     # detect.json versions that have people in them (2: rough in-event grouping,
                      # used until recordings.py has redone the recording)
# joining people placing left apart (see Stream.consolidate): two people seen together in an
# event are two people, except that detect.py now and then splits one person in an event in
# two, so this share of the smaller one's events may be shared
MERGE_SHARED = 0.02
# the day pass (Stream.day_pass): one day's sightings grouped on their own, where the same
# clothes and minutes-apart times make one person far easier to see than across days
DAY_JOIN = 0.70       # average similarity (with the time bonus) to be one person within a day
DAY_NEAR_MIN = 5      # minutes apart: +DAY_BONUS; within an hour: half of it
DAY_BONUS = 0.10
DAY_ATTACH = 0.60     # a sighting on its own joins the most alike person seen within
DAY_ATTACH_MIN = 30   # DAY_ATTACH_MIN minutes if at least this alike (an odd view isn't a new person)
DAY_SMALL = 3         # people this small (in all) may be moved into the day's main person
# Per-camera tuning, from the "same person" marks made at that camera (TUNING[stream]):
#   near_s, near_bonus  sightings this close in time (s) count as this much more alike: one
#                       person's sightings minutes apart are less alike than JOIN at a camera
#                       looking steeply down at people passing close by
#   split_events        sightings of one event may be one person unless they were taken at the
#                       same moment (detect.py splits one person in an event in two when their
#                       crops differ a lot); without it, two in one event are always two people
#   learn               use what learn() finds (default True)
#   quality             a person's fingerprint is their sightings' mean weighted by how good a
#                       look each is (quality(): box height x detector score, halved when cut
#                       off at the frame's edge; over the camera's median, kept within 1/3..3x),
#                       so a small, blurred or cut-off sighting that joined them pulls it less
# cam1 (2026-10-01, from its 105 marks): 39 marks joined two people seen in one event. Marks
# within 10 minutes: median similarity 0.70 (0.51 of the pairs >= 0.70); two people in the
# picture at the same moment: >= 0.70 only 2%. Marks further apart (other clothes, other
# days) stay with JOIN. What learn() found barely beat looks alone there (AUC 0.677 v 0.664)
# and learned nothing about timing, so it's off. quality (2026-10-02): regrouped from scratch
# without marks, it joined 52 pairs of pieces that plain means didn't and those mostly looked
# like one person by eye; plain means joined 41 the other way; mistakes about even either way.
TUNING = {"cam1": {"near_s": 600, "near_bonus": 0.05, "split_events": True, "learn": False, "quality": True,
                  "day_pass": True, "static_spread": True}}
Q_CAP = 3.0           # quality weights stay within 1/Q_CAP..Q_CAP times the camera's median


def load(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def dump(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    os.replace(tmp, path)


def vec(b64, dtype=np.float16):
    return np.frombuffer(base64.b64decode(b64), dtype).astype(np.float32)


STREAM_RE = re.compile(r"^(.+?)-\d+_")  # nginx-rtmp names: <stream>-<epoch>_<date>.mp4


def sightings(results="detect.json"):
    """Every person seen in one of the camera's recordings that still exists, oldest first.
    Recordings are in <stream>/<day>/<name>/ folders (see recordings.py): <name>.mp4,
    detect.json, info.json, people/. results="detect.prev.json": as seen before the
    recording was analysed again."""
    out = []
    for path in glob.glob(os.path.join(SDIR, "*", "*", results)):
        folder = os.path.dirname(path)
        mp4 = os.path.basename(folder) + ".mp4"
        if not os.path.exists(os.path.join(folder, mp4)):
            continue
        d = load(path, {})
        if d.get("v") not in DETECT_V:
            continue
        idx = load(os.path.join(folder, "info.json"), {})  # recordings.py's cached probe
        start = idx.get("start", 0)
        # what the people in it are: tells a recording analysed again from one merely copied
        h = hashlib.md5(json.dumps(d.get("people", []), sort_keys=True).encode()).hexdigest()[:16]
        for ev, (span, people) in enumerate(zip(d["spans"], d.get("people", []))):
            for k, p in enumerate(people):
                m = STREAM_RE.match(mp4)
                thumb = os.path.relpath(os.path.join(folder, "people", p["thumb"]), REC_DIR) if p.get("thumb") else None
                out.append({"key": f"{mp4}#{ev}#{k}", "file": mp4, "stream": m[1] if m else "",
                            "ev": ev, "t": p["t"], "t0": p.get("t0", p["t"]), "t1": p.get("t1", p["t"]),
                            "box": p["box"], "detect": path, "h": h,
                            "at": start + p["t"], "emb": vec(p["emb"]), "thumb": thumb,
                            "size": (p["box"][3] - p["box"][1]) * p["score"],
                            "q": quality(p["box"], p["score"], idx.get("width"), idx.get("height"))})
    return sorted(out, key=lambda s: s["at"])


def quality(box, score, w, h):
    """How good a look a sighting is: its box's height times the detector's score, halved when
    the box touches the frame's edge (the person is cut off). w, h: the frame (unknown: no edge)."""
    x0, y0, x1, y1 = box
    edge = bool(w and h) and (x0 <= 2 or y0 <= 2 or x1 >= w - 2 or y1 >= h - 2)
    return (y1 - y0) * score * (0.5 if edge else 1.0)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-6)


def is_static(boxes):
    if len(boxes) < STATIC_MIN:
        return False
    ov = [iou(boxes[a], boxes[b]) for a in range(len(boxes)) for b in range(a + 1, len(boxes))]
    return float(np.median(ov)) >= STATIC_IOU


def cluster(E, join=JOIN, sim=None, raw=None, floor=-np.inf, size=None):
    """Average-linkage clustering of unit vectors E (or of a similarity matrix sim):
    lists of row indices. With raw, a second similarity matrix, only groups at least floor
    alike on it are joined. size: how many sightings each row stands for (default 1)."""
    sim = E @ E.T if sim is None else sim.copy()
    mats = [sim] if raw is None else [sim, raw.copy()]
    n = len(sim)
    for m in mats:
        np.fill_diagonal(m, -np.inf)
    size = np.ones(n) if size is None else np.array(size, float)
    members = {i: [i] for i in range(n)}
    while True:
        pick = sim if raw is None else np.where(mats[1] >= floor, sim, -np.inf)
        i, j = divmod(int(np.argmax(pick)), n)
        if pick[i, j] < join:
            break
        # merge j into i; average similarity to the merged group (Lance-Williams)
        for m in mats:
            row = (m[i] * size[i] + m[j] * size[j]) / (size[i] + size[j])
            m[i], m[:, i] = row, row
            m[i, i] = -np.inf
            m[j], m[:, j] = -np.inf, -np.inf
        size[i] += size[j]
        members[i] += members.pop(j)
    return list(members.values())


class Stream:
    """One camera's people: which sightings each has, kept up to date as new ones are placed."""

    def __init__(self, name, ss, model=None):
        self.name, self.ss = name, ss
        self.A = arrays(ss)
        self.E = self.A[0]
        self.score, self.join = (model[0], model[1]) if model and model[0] is not None else (None, JOIN)
        tune = TUNING.get(name, {})
        self.near_s, self.bonus = tune.get("near_s", 0), tune.get("near_bonus", 0.0)
        self.split = tune.get("split_events", False)
        self.static_spread = tune.get("static_spread", False)  # fixed objects must be seen over several recordings / hours
        # each sighting's weight in its person's fingerprint (tune "quality"; else all 1)
        self.Q = np.ones(len(ss), np.float32)
        if tune.get("quality") and ss:
            q = np.array([s.get("q", 1.0) for s in ss], np.float32)
            self.Q = np.clip(q / max(float(np.median(q)), 1e-6), 1 / Q_CAP, Q_CAP).astype(np.float32)
        # members: a person's sightings (their number is the person's size); sums / weights:
        # the weighted sum of their fingerprints and the total weight, whose ratio is their fingerprint
        self.members, self.sums, self.weights, self.events = {}, {}, {}, {}

    def apart(self, i):
        """What two of one person's sightings can't share: an event, or (split_events) the
        moment in an event (two people in the picture at once)."""
        s = self.ss[i]
        return (s["file"], s["ev"], round(s["t"], 1)) if self.split else (s["file"], s["ev"])

    def add(self, pid, idx):
        self.members.setdefault(pid, []).extend(idx)
        self.sums[pid] = self.sums.get(pid, 0) + (self.E[idx] * self.Q[idx, None]).sum(0)
        self.weights[pid] = self.weights.get(pid, 0.0) + float(self.Q[idx].sum())
        self.events.setdefault(pid, set()).update(self.apart(i) for i in idx)

    def mean(self, pid):
        """A person's fingerprint: the (weighted) mean of their sightings' (not unit length)."""
        return self.sums[pid] / self.weights[pid]

    def place(self, idx, old, nxt):
        """Place new sightings (indices, one recording's) among the people; returns {index: id}.
        They are clustered together with every person they could join: the raw similarity
        of a group to a person is at most its best sighting's, so people none of them is
        close enough to are left out."""
        E, m = self.E[idx], len(idx)
        pids = list(self.members)
        cand = []
        if pids:
            C = np.stack([self.mean(p) for p in pids])  # mean fingerprints
            best = (E @ C.T).max(0)
            bound = (LEARN_FLOOR if self.score else self.join) - self.bonus
            cand = [p for p, b in zip(pids, best) if b >= bound]
        k = len(cand)
        ev = [self.apart(i) for i in idx]
        ban = np.zeros((k + m, k + m), bool)
        ban[:k, :k] = True  # people already there are never merged
        ban[k:, k:] = [[a == b for b in ev] for a in ev]  # nor two people in one event / moment
        for a, p in enumerate(cand):
            ban[a, k:] = ban[k:, a] = [e in self.events[p] for e in ev]
        R = np.zeros((k + m, k + m), np.float32)
        R[k:, k:] = E @ E.T
        if k:
            R[:k, k:] = np.stack([self.mean(p) for p in cand]) @ E.T
        if self.bonus:  # near in time: more alike (to a person: by the share of theirs near)
            at = self.A[2][idx]
            R[k:, k:] += self.bonus * (np.abs(at[:, None] - at[None, :]) <= self.near_s)
            for c, p in enumerate(cand):
                Tm = self.A[2][self.members[p]]
                R[c, k:] += self.bonus * (np.abs(Tm[None, :] - at[:, None]) <= self.near_s).mean(1)
        if k:
            R[k:, :k] = R[:k, k:].T
        S = None
        if self.score:
            S = np.zeros_like(R)
            a, b = np.meshgrid(idx, idx, indexing="ij")
            S[k:, k:] = self.score(pair_features(self.A, a.ravel(), b.ravel())).reshape(m, m)
            for c, p in enumerate(cand):
                M = sorted(self.members[p])  # in time order, so the sample is the same every run
                if len(M) > LEARN_SAMPLE:
                    M = [M[x] for x in np.linspace(0, len(M) - 1, LEARN_SAMPLE).round().astype(int)]
                s = self.score(pair_features(self.A, np.repeat(idx, len(M)), np.tile(M, m)))
                S[c, k:] = S[k:, c] = s.reshape(m, len(M)).mean(1)
            S[ban] = -np.inf
        R[ban] = -np.inf
        size = [len(self.members[p]) for p in cand] + [1] * m
        if S is None:
            groups = cluster(None, self.join, R, size=size)
        else:
            groups = cluster(None, self.join, S, raw=R, floor=LEARN_FLOOR, size=size)
        out = {}
        for g in groups:
            new = [idx[x - k] for x in g if x >= k]
            if not new:
                continue
            known = [cand[x] for x in g if x < k]
            if known:
                pid = known[0]
            else:  # a new person; or, placed again, the one most of these sightings were
                votes = {}
                for i in new:
                    v = old.get(self.ss[i]["key"])
                    if v and v not in self.members:
                        votes[v] = votes.get(v, 0) + 1
                if votes:
                    pid = max(votes, key=votes.get)
                else:
                    pid = f"{self.name}-p{nxt.get(self.name, 1)}"
                    nxt[self.name] = nxt.get(self.name, 1) + 1
            self.add(pid, new)
            out.update((i, pid) for i in new)
        return out

    def consolidate(self, label):
        """Join people that turn out to be one person; returns {gone id: kept id}.

        Placing never merges people already there, so a person started twice (say in
        colour by day and in infrared at night, before there was anything in between) grows
        as two people for good; on ramesh dozens of pairs of big people were more alike
        (average linkage) than JOIN. This joins them by average linkage over whole people at
        the same JOIN, but never two seen together in one event (beyond MERGE_SHARED, see
        there), nor two with different labels (label: {id: name / mark / "object"}: named,
        merged by hand, or marked "not a person"); an object joins only objects. The person
        with a label, else the bigger one, keeps its id. Without a learned score only: what
        is learned already decides placing, and isn't calibrated for whole people."""
        if self.score or len(self.members) < 2:
            return {}
        ids = list(self.members)
        n = len(ids)
        sums = np.stack([self.sums[p] for p in ids]).astype(np.float32)  # weighted
        wt = np.array([self.weights[p] for p in ids], np.float32)  # their total weight
        cnt = np.array([len(self.members[p]) for p in ids], np.float32)  # sightings: who keeps the id
        evs = [set(self.events[p]) for p in ids]
        lab = [label.get(p) for p in ids]
        sim = (sums @ sums.T) / np.outer(wt, wt)  # (weighted) average similarity of the two people's sightings
        np.fill_diagonal(sim, -np.inf)
        for a in range(n):  # differently labelled people stay apart
            if lab[a] is not None:
                for b in range(n):
                    if lab[b] != lab[a] and (lab[b] is not None or lab[a] == "object"):
                        sim[a, b] = sim[b, a] = -np.inf
        gone = {}
        while True:
            i, j = divmod(int(np.argmax(sim)), n)
            if sim[i, j] < JOIN:
                break
            if len(evs[i] & evs[j]) > MERGE_SHARED * min(len(evs[i]), len(evs[j])):
                sim[i, j] = sim[j, i] = -np.inf  # seen together: two people
                continue
            if (lab[j] is not None, cnt[j], ids[i]) > (lab[i] is not None, cnt[i], ids[j]):
                i, j = j, i  # i keeps its id
            sums[i] += sums[j]
            wt[i] += wt[j]
            cnt[i] += cnt[j]
            evs[i] |= evs[j]
            lab[i] = lab[i] if lab[i] is not None else lab[j]
            veto = np.isneginf(sim[i]) | np.isneginf(sim[j])  # whoever either may not join
            row = (sums @ sums[i]) / (wt * wt[i])
            row[veto] = -np.inf
            sim[i], sim[:, i] = row, row
            sim[i, i] = -np.inf
            sim[j], sim[:, j] = -np.inf, -np.inf
            gone[ids[j]] = ids[i]
        for g in gone:
            k = g
            while k in gone:
                k = gone[k]
            gone[g] = k
            self.members[k].extend(self.members.pop(g))
            self.sums[k] = self.sums[k] + self.sums.pop(g)
            self.weights[k] += self.weights.pop(g)
            self.events[k] |= self.events.pop(g)
        return gone

    def fixed(self):
        """Ids of the people that are really a fixed object (boxes always in one place)."""
        out = set()
        for pid, M in self.members.items():
            M = sorted(M)  # in time order, so the sample is the same every run
            if len(M) > STATIC_SAMPLE:
                M = [M[x] for x in np.linspace(0, len(M) - 1, STATIC_SAMPLE).round().astype(int)]
            if is_static([self.ss[i]["box"] for i in M]):
                M = self.members[pid]
                if not self.static_spread or (len({self.ss[i]["file"] for i in M}) >= STATIC_RECS
                                              and len({int(self.ss[i]["at"] // 3600) for i in M}) >= STATIC_HOURS):
                    out.add(pid)
        return out

    def day_pass(self, days, keep, protect):
        """Group each of `days` (recording dates) on its own and move the sightings of small
        people (<= DAY_SMALL sightings in all, or seen only that day) into the person holding
        most of their day group (a labelled one first). People in `keep` (named, merged by
        hand, objects) are never moved, nor any with a sighting in `protect` (the marks'
        keys): big and labelled people are never merged by this. Returns the moves."""
        moved = 0
        where = {i: p for p, M in self.members.items() for i in M}
        for d in sorted(days):
            I = [i for i, s in enumerate(self.ss) if day_of(s) == d and where.get(i) not in keep]
            if len(I) < 2:
                continue
            for G in day_groups(self.E[I], self.A[2][I] / 60, [self.apart(i) for i in I]):
                G = [I[x] for x in G]
                if len(G) < 2:
                    continue
                cnt = {}
                for i in G:
                    cnt[where[i]] = cnt.get(where[i], 0) + 1
                main = max(cnt, key=lambda p: (p in keep, cnt[p], len(self.members[p]), p))
                for p in cnt:
                    M = self.members.get(p, [])
                    if p == main or p in keep or not M or any(self.ss[i]["key"] in protect for i in M):
                        continue
                    if len(M) > DAY_SMALL and any(day_of(self.ss[i]) != d for i in M):
                        continue
                    move = [i for i in G if where[i] == p]
                    self.take(p, main, move)
                    for i in move:
                        where[i] = main
                    moved += len(move)
        return moved

    def take(self, src, dst, idx):
        """Move sightings idx from person src to person dst."""
        left = [i for i in self.members[src] if i not in set(idx)]
        if left:
            self.members[src] = left
            self.sums[src] = self.sums[src] - (self.E[idx] * self.Q[idx, None]).sum(0)
            self.weights[src] -= float(self.Q[idx].sum())
            self.events[src] = {self.apart(i) for i in left}
        else:
            for x in (self.members, self.sums, self.weights, self.events):
                x.pop(src, None)
        self.add(dst, idx)


def day_of(s):
    """A sighting's day, as the page has it: its recording's date (local time)."""
    return s["file"].split("_")[1]


def day_groups(E, at, apart):
    """One day's sightings (fingerprints E, times in minutes, what two of one person can't
    share): average linkage down to DAY_JOIN over similarity plus a time bonus, then a
    sighting left on its own joins the most alike person seen within DAY_ATTACH_MIN
    minutes. Returns lists of indices."""
    n = len(E)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    dt = np.abs(at[:, None] - at[None, :])
    S = E @ E.T + DAY_BONUS * (dt <= DAY_NEAR_MIN) + DAY_BONUS / 2 * ((dt > DAY_NEAR_MIN) & (dt <= 60))
    ban = np.array([[a == b for b in apart] for a in apart])
    S[ban] = -np.inf  # (the diagonal too)
    members, size = {x: [x] for x in range(n)}, np.ones(n)
    while True:
        i, j = divmod(int(np.argmax(S)), n)
        if S[i, j] < DAY_JOIN:
            break
        row = (S[i] * size[i] + S[j] * size[j]) / (size[i] + size[j])
        row[np.isneginf(S[i]) | np.isneginf(S[j])] = -np.inf  # a ban on either holds
        S[i], S[:, i] = row, row
        S[i, i] = -np.inf
        S[j], S[:, j] = -np.inf, -np.inf
        size[i] += size[j]
        members[i] += members.pop(j)
    groups = [g for g in members.values() if len(g) > 1]
    for g in members.values():
        if len(g) > 1:
            continue
        x, best, bi = g[0], DAY_ATTACH, None
        for k, h in enumerate(groups):
            near = [y for y in h if dt[x, y] <= DAY_ATTACH_MIN and apart[y] != apart[x]]
            if near and not any(apart[y] == apart[x] for y in h):
                s = float((E[near] @ E[x]).max())
                if s >= best:
                    best, bi = s, k
        if bi is None:
            groups.append([x])
        else:
            groups[bi].append(x)
    return groups


def b64(v):
    return base64.b64encode(np.asarray(v, np.float16).tobytes()).decode()


def snapshot(fb, seen, save=True):
    """Remember the sightings behind each mark (file + fingerprint), taken while the keys the
    page sent are still valid: keys change when a recording is analysed again."""
    snap = load(SNAP, {})
    by_key = {s["key"]: s for s in seen}
    rng = np.random.default_rng(0)
    for e in fb:
        if e["id"] in snap:
            continue
        side = {}
        for k in ("a", "b"):
            ss = [by_key[x] for x in e.get("keys_" + k, []) if x in by_key]
            if len(ss) > SNAP_MAX:
                ss = [ss[i] for i in sorted(rng.choice(len(ss), SNAP_MAX, replace=False))]
            side[k] = [[s["file"], b64(s["emb"])] for s in ss]
        snap[e["id"]] = side
    ids = {e["id"] for e in fb}
    snap = {i: v for i, v in snap.items() if i in ids}  # undone marks are forgotten
    if save:
        dump(SNAP, snap)
    return snap


def find_again(side, ss):
    """Indices in ss of a mark's remembered sightings: in the same recording, the sighting
    with (nearly) the same fingerprint, or, after the recording was analysed again from
    other frames, one clearly more alike than the rest; each sighting found once."""
    by_file = {}
    for i, s in enumerate(ss):
        by_file.setdefault(s["file"], []).append(i)
    want = {}
    for f, emb in side:
        if f in by_file:
            want.setdefault(f, []).append(vec(emb))
    out = []
    for f, vs in want.items():
        cand = by_file[f]
        S = np.stack(vs) @ np.stack([ss[i]["emb"] for i in cand]).T
        taken = set()
        for r in np.argsort(-S.max(1)):  # the surest first
            sims = [(S[r, c], c) for c in range(len(cand)) if c not in taken]
            if not sims:
                break
            sims.sort(reverse=True)
            best, c = sims[0]
            second = sims[1][0] if len(sims) > 1 else -1.0
            if best >= SAME_SIGHTING or (best >= FIND_LOOSE and best - second >= FIND_MARGIN):
                taken.add(c)
                out.append(cand[c])
    return out


def pair(olds, news):
    """{old key: new key} for a recording analysed again: each old sighting paired with the
    new one it matches (the surest pairs first, one to one); unmatched ones are left out."""
    if not olds or not news:
        return {}
    S = np.stack([o["emb"] for o in olds]) @ np.stack([n["emb"] for n in news]).T
    dt = np.abs(np.array([o["t"] for o in olds])[:, None] - np.array([n["t"] for n in news])[None, :])
    ok = (S >= PAIR_SURE) | ((S >= PAIR_NEAR) & (dt <= PAIR_DT))
    out, used = {}, set()
    for i, j in sorted(zip(*np.nonzero(ok)), key=lambda x: -S[x]):
        if olds[i]["key"] in out or j in used:
            continue
        out[olds[i]["key"]] = news[j]["key"]
        used.add(j)
    return out


def auc(pos, neg):
    """How well similarity separates same (pos) from different (neg) pairs: 0.5 chance, 1 perfect."""
    if not len(pos) or not len(neg):
        return 0.5
    r = np.concatenate([pos, neg]).argsort().argsort()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def arrays(ss):
    """What pair_features needs of each sighting."""
    return (np.stack([s["emb"] for s in ss]), np.array([max(1.0, s["box"][3] - s["box"][1]) for s in ss]),
            np.array([s["at"] for s in ss], float))


def pair_features(A, i, j):
    """Per pair of sightings (A: arrays() of them): how alike they look, how alike their
    size in the frame is, how close in time they were seen (all: bigger = more likely the
    same person)."""
    E, h, t = A
    return np.stack([(E[i] * E[j]).sum(1), -np.abs(np.log(h[i] / h[j])),
                     -np.log1p(np.abs(t[i] - t[j]) / 60)], 1).astype(np.float32)


def fit(F, y):
    """Logistic regression (standardised features, both classes weighted equally)."""
    mu, sd = F.mean(0), F.std(0) + 1e-6
    X = (F - mu) / sd
    wt = np.where(y == 1, 0.5 / y.sum(), 0.5 / (1 - y).sum())
    w, c = np.zeros(X.shape[1]), 0.0
    for _ in range(1500):
        g = (1 / (1 + np.exp(-(X @ w + c))) - y) * wt
        w -= X.T @ g + 1e-3 * w
        c -= g.sum()
    return lambda F2: ((F2 - mu) / sd) @ w + c, w


def learn(ss, label):
    """Learn from the named / merged people (label: sighting index -> person) how to weigh
    looks, size and timing, for this camera. Checked on half the examples after fitting on
    the other half (both ways round): used only if it beats looks alone by LEARN_GAIN.
    Returns (pair score function, joining threshold, summary) or None.

    The threshold is set so that about as many random pairs join as with looks alone at
    JOIN: what's learned changes which pairs join, not how readily."""
    rng = np.random.default_rng(0)
    A = arrays(ss)
    idx = np.array(sorted(label))
    names = np.array([label[i] for i in idx])
    sizes = {n: int((names == n).sum()) for n in set(names)}
    if sum(1 for v in sizes.values() if v >= LEARN_SEEN) < LEARN_PEOPLE:
        return None

    def pairs(sub):
        a, b = np.triu_indices(len(sub), 1)
        a, b = sub[a], sub[b]
        same = names[a] == names[b]
        keep = []
        for m in (same, ~same):
            k = np.flatnonzero(m)
            keep.append(rng.choice(k, LEARN_CAP, replace=False) if len(k) > LEARN_CAP else k)
        k = np.concatenate(keep)
        return idx[a[k]], idx[b[k]], same[k].astype(float)

    i, j, y = pairs(np.arange(len(idx)))
    if y.sum() < LEARN_PAIRS or (1 - y).sum() < LEARN_PAIRS:
        return None
    # check on unseen examples: split the people's sightings in two halves
    looks, both = [], []
    perm = rng.permutation(len(idx))
    halves = (np.sort(perm[: len(idx) // 2]), np.sort(perm[len(idx) // 2:]))
    for tr, te in (halves, halves[::-1]):
        a, b, ya = pairs(tr)
        c, d, yc = pairs(te)
        if not ya.sum() or not (1 - ya).sum() or not yc.sum() or not (1 - yc).sum():
            return None
        score, _ = fit(pair_features(A, a, b), ya)
        Fc = pair_features(A, c, d)
        looks.append(auc(Fc[yc == 1, 0], Fc[yc == 0, 0]))
        s = score(Fc)
        both.append(auc(s[yc == 1], s[yc == 0]))
    info = {"people": len(sizes), "pairs": [int(y.sum()), int((1 - y).sum())],
            "auc": [round(float(np.mean(looks)), 3), round(float(np.mean(both)), 3)]}
    if np.mean(both) < np.mean(looks) + LEARN_GAIN:
        return None, None, {**info, "used": False}
    score, w = fit(pair_features(A, i, j), y)
    info["weights"] = [round(float(x), 2) for x in w]  # looks, size, timing

    # random pairs of this camera's sightings (not two in one event: never one person)
    n = len(ss)
    rnd = rng.integers(0, n, (6000, 2))
    rnd = rnd[[a != b and (ss[a]["file"], ss[a]["ev"]) != (ss[b]["file"], ss[b]["ev"]) for a, b in rnd]]
    E = A[0]
    frac = float(((E[rnd[:, 0]] * E[rnd[:, 1]]).sum(1) >= JOIN).mean())
    vals = score(pair_features(A, rnd[:, 0], rnd[:, 1]))
    join = float(np.quantile(vals, 1 - frac)) if frac > 0 and len(vals) else float("inf")
    info.update(used=True, join=round(join, 3))
    return score, join, info


def typical(ss):
    """The sighting to show for a person: the one most like the rest (not just the biggest,
    which may be a stray that joined the group), among the clearer half."""
    ss = [s for s in ss if s["thumb"]]
    if not ss:
        return None
    big = max(s["size"] for s in ss)
    ss = [s for s in ss if s["size"] >= big / 2]
    mean = np.sum([s["emb"] for s in ss], 0)
    return max(ss, key=lambda s: float(s["emb"] @ mean))


def main(dry=False, rebuild=False):
    state = load(STATE, {})
    if not isinstance(state.get("next"), dict):  # before per-stream ids: start afresh
        state = {"next": {}, "assign": {}}
    seen = sightings()
    # a recording analysed again has new sightings under the old keys: its people changed.
    # files: {recording: {"h": its people's hash, "at": when they last changed}}
    was, now, files = state.get("files", {}), int(time.time()), {}
    prevs = {}  # recording -> its sightings before it was analysed again (detect.prev.json)
    for s in sightings("detect.prev.json"):
        prevs.setdefault(s["file"], []).append(s)
    redone = set()
    for s in seen:
        f, h = s["file"], s["h"]
        if f in files:
            continue
        v = was.get(f)
        if isinstance(v, int):  # before hashes: the file's time then
            same = v == int(os.path.getmtime(s["detect"]))
            files[f] = {"h": h, "at": v if same else now}
        elif v is None:
            files[f] = {"h": h, "at": now}
            same = f not in prevs
        else:
            same = v.get("h") == h
            files[f] = v if same else {"h": h, "at": now}
        if not same:
            redone.add(f)
    stamps = files
    old = state["assign"]
    # pair each redone recording's old sightings with the new ones: paired ones keep their
    # person; keymap remembers the pairs (by when), to translate keys the page saved
    keymap = {f: g for f, g in load(KEYMAP, {}).items() if f in stamps}
    now_map = {}
    for f in sorted(redone):
        news = [s for s in seen if s["file"] == f]
        m = pair(prevs.get(f, []), news)
        now_map[f] = m
        keymap.setdefault(f, []).append({"at": files[f]["at"], "map": {o["key"]: m.get(o["key"]) for o in prevs.get(f, [])}})
        if f in prevs:
            print(f"{f}: analysed again; {len(m)} of {len(prevs[f])} sightings paired with {len(news)} new")
    kept = {}
    if not rebuild:
        for k, v in old.items():
            f = k.split("#", 1)[0]
            if f in stamps and f not in redone:
                kept[k] = v
            elif f in redone and now_map[f].get(k):
                kept[now_map[f][k]] = v
    votes = {k: v for k, v in old.items() if k.split("#", 1)[0] not in redone}  # ids to reuse

    def translate(key, at):
        """A sighting key the page saved at time `at`, as it is now (None: no longer known)."""
        f = key.split("#", 1)[0]
        gens = keymap.get(f, [])
        for g in gens:
            if g["at"] > at:
                key = g["map"].get(key)
                if key is None:
                    return None
        changed = files.get(f, {}).get("at", 0)
        if changed > at and not any(g["at"] >= changed for g in gens):
            return None  # analysed again since, before pairs were kept: can't tell
        return key
    try:  # noted before reading: a mark saved while this runs makes recordings.py run it again
        fb_mtime = os.path.getmtime(FEEDBACK)
    except OSError:
        fb_mtime = None
    feedback = load(FEEDBACK, {})
    fb = [e for e in feedback.get("same", []) if isinstance(e, dict) and e.get("id")]
    snap = snapshot(fb, seen, save=not dry)
    by_stream = {}
    for s in seen:
        by_stream.setdefault(s["stream"], []).append(s)
    assign, fixed_keys, learned, unions = {}, set(), {}, []
    prev, prev_keys = load(OUT, {}), load(KEYS, {})
    curated = {}  # sighting key -> the named / merged person it was (same name = same person)
    for p in prev.get("people", []):
        if p.get("name") or p.get("joined"):
            for k in prev_keys.get(p["id"], []):
                f = k.split("#", 1)[0]
                k = now_map[f].get(k) if f in now_map else k  # keys of the last run
                if k:
                    curated[k] = p.get("name") or p["id"]
    for stream, ss in by_stream.items():
        # this camera's marks, as index lists of the two sides' sightings found again
        marks = []
        for e in fb:
            side = snap.get(e["id"]) or {}
            a, b = find_again(side.get("a", []), ss), find_again(side.get("b", []), ss)
            if a and b:
                marks.append((a, b))
                unions.append((e["id"], [ss[i]["key"] for i in a], [ss[i]["key"] for i in b]))
        # examples: the named / merged people, as the last run grouped them
        label = {i: curated[s["key"]] for i, s in enumerate(ss) if s["key"] in curated}
        model = learn(ss, label) if TUNING.get(stream, {}).get("learn", True) else None
        if model:
            learned[stream] = model[2]
            print(f"{stream}: {model[2]}")
        st = Stream(stream, ss, model)
        new = {}  # recording -> its sightings not placed yet
        for i, s in enumerate(ss):
            if s["key"] in kept:
                st.add(kept[s["key"]], [i])
            else:
                new.setdefault(s["file"], []).append(i)
        for idx in new.values():  # oldest recording first (ss is in time order)
            st.place(idx, votes, state["next"])
        placed = sum(len(x) for x in new.values())
        if placed:
            print(f"{stream}: placed {placed} new sighting(s) from {len(new)} recording(s)")
        # labels keep people apart in consolidate(): the named / merged people of the last
        # run, names in people-names.json, and things marked "not a person" on the page
        # (people_api.py; found again by their sightings, like names)
        hand = load(NAMES, {}).get("names", {})
        label = {}
        for pid, M in st.members.items():
            v = {}
            for i in M:
                c = curated.get(ss[i]["key"])
                if c is not None:
                    v[c] = v.get(c, 0) + 1
            if v:
                label[pid] = max(v, key=v.get)
            elif hand.get(pid):
                label[pid] = hand[pid]
        where = {ss[i]["key"]: pid for pid, M in st.members.items() for i in M}
        objects = set()
        for e in feedback.get("objects", {}).values():
            votes = {}
            for k in e.get("keys", []):
                k = translate(k, e.get("at", 0))
                if k in where:
                    votes[where[k]] = votes.get(where[k], 0) + 1
            if votes:
                objects.add(max(votes, key=votes.get))
        # fixed objects: found by their boxes, or marked "not a person" (new sightings that
        # join them are hidden too). Found before joining people: a fixed object joined to a
        # person would fail the box test and show up again
        objects |= st.fixed()
        label.update((pid, "object") for pid in objects)
        gone = st.consolidate(label)
        if gone:
            print(f"{stream}: joined {len(gone)} people into others (one person grouped as several)")
        fixed = {gone.get(p, p) for p in objects} | st.fixed()
        # the day pass, for the days of the recordings placed now (PEOPLE_DAY_PASS=all: every day)
        if not TUNING.get(stream, {}).get("day_pass"):
            days = set()
        elif os.environ.get("PEOPLE_DAY_PASS") == "all":
            days = {day_of(s) for s in ss}
        else:
            days = {day_of(ss[idx[0]]) for idx in new.values()}
        keep = fixed | {gone.get(p, p) for p in label}
        protect = {k for _, ka, kb in unions for k in ka + kb}
        moved = st.day_pass(days, keep, protect) if days else 0
        if moved:
            print(f"{stream}: day pass moved {moved} sighting(s) of {len(days)} day(s) into the day's people")
        for pid, M in st.members.items():
            for i in M:
                assign[ss[i]["key"]] = pid
                if pid in fixed:
                    fixed_keys.add(ss[i]["key"])
    state = {"next": state["next"], "assign": assign, "files": files}
    # the marks' remembered fingerprints follow their sightings into the new results
    for sides in snap.values():
        for side in sides.values():
            for n, (f, emb) in enumerate(side):
                if f not in now_map or f not in prevs:
                    continue
                v = vec(emb)
                sims = [float(o["emb"] @ v) for o in prevs[f]]
                new_key = now_map[f].get(prevs[f][int(np.argmax(sims))]["key"]) if max(sims) >= SAME_SIGHTING else None
                s = next((x for x in seen if x["key"] == new_key), None) if new_key else None
                if s:
                    side[n] = [f, b64(s["emb"])]

    names = load(NAMES, {})
    merge, named = dict(names.get("merge", {})), dict(names.get("names", {}))
    joined = {}  # person id -> ids of the marks that joined others into them

    def final(pid):
        for _ in range(50):  # follow merge chains
            if pid not in merge:
                break
            pid = merge[pid]
        return pid

    size = {}
    for pid in assign.values():
        size[pid] = size.get(pid, 0) + 1

    def majority(keys):
        votes = {}
        for k in keys:
            if k in assign:
                pid = final(assign[k])
                votes[pid] = votes.get(pid, 0) + 1
        return max(votes, key=votes.get) if votes else None

    def join(keep, gone):
        keep, gone = final(keep), final(gone)
        if keep == gone:
            return keep
        if (size.get(gone, 0), keep) > (size.get(keep, 0), gone):  # the bigger one keeps its id
            keep, gone = gone, keep
        merge[gone] = keep
        size[keep] = size.get(keep, 0) + size.get(gone, 0)
        joined.setdefault(keep, []).extend(joined.pop(gone, []))
        return keep

    for mid, ka, kb in unions:  # the marks join people for good, whatever the grouping did
        pa, pb = majority(ka), majority(kb)
        if not (pa and pb):
            continue
        keep = join(pa, pb)
        # a marked sighting the grouping put on its own (or in a group mostly of marked
        # sightings) comes along too
        marked = {}
        for k in ka + kb:
            if k in assign:
                marked[assign[k]] = marked.get(assign[k], 0) + 1
        for pid, n in marked.items():
            if 2 * n >= sum(1 for v in assign.values() if v == pid):
                keep = join(keep, pid)
        joined.setdefault(final(keep), []).append(mid)

    # names from the page: for whoever most of the named person's sightings are now
    for pid, e in sorted(feedback.get("names", {}).items(), key=lambda x: x[1].get("at", 0)):
        target = majority([k for k in (translate(k, e.get("at", 0)) for k in e.get("keys", [])) if k]) or final(pid)
        if e.get("name"):
            named[target] = e["name"]

    people, by_file, objects, keys = {}, {}, {}, {}
    for s in seen:
        if s["key"] in fixed_keys:
            objects.setdefault(s["file"], set()).add(s["ev"])
            continue
        pid = final(state["assign"][s["key"]])
        p = people.setdefault(pid, {"id": pid, "stream": s["stream"], "name": named.get(pid), "n": 0,
                                    "first": s["at"], "last": s["at"], "_seen": [], "_ev": set()})
        # n counts motion events, like the page: two sightings in one event (say, of two
        # people later marked as the same) are seen once
        if (s["file"], s["ev"]) not in p["_ev"]:
            p["_ev"].add((s["file"], s["ev"]))
            p["n"] += 1
        p["last"] = s["at"]
        p["_seen"].append(s)
        keys.setdefault(pid, []).append(s["key"])
        # [event, snapshot offset, person, thumb, in view from, to] (offsets in the recording)
        by_file.setdefault(s["file"], []).append([s["ev"], s["t"], pid, s["thumb"], s["t0"], s["t1"]])
    for p in people.values():
        del p["_ev"]
        b = typical(p.pop("_seen"))
        p["thumb"] = b["thumb"] if b else None
        p["first"], p["last"] = int(p["first"]), int(p["last"])
        if joined.get(p["id"]):
            p["joined"] = sorted(set(joined[p["id"]]))
        if not p["name"]:  # a name given to one of the people joined into this one
            p["name"] = next((named[x] for x, y in merge.items() if final(y) == p["id"] and x in named), None)
    # events where the only "person" was a fixed object
    static = {f: sorted(evs - {x[0] for x in by_file.get(f, [])}) for f, evs in objects.items()}
    print(f"{len(seen)} sightings, {len(people)} people, {len(fixed_keys)} of a fixed object, "
          f"{len(fb)} mark(s)")
    if dry:
        return people
    dump(STATE, state)
    dump(KEYS, keys)
    dump(KEYMAP, keymap)
    dump(SNAP, snap)
    for f in redone & set(prevs):  # paired: the old results aren't needed any more
        try:
            os.remove(os.path.join(os.path.dirname(prevs[f][0]["detect"]), "detect.prev.json"))
        except OSError:
            pass
    dump(OUT, {"generated": int(time.time()),
               "people": sorted(people.values(), key=lambda p: (-p["n"], p["first"])),
               "seen": by_file,
               "static": {f: evs for f, evs in static.items() if evs},
               "learned": learned,
               "marks": [e["id"] for e in fb],  # the marks this grouping includes
               "objects": sorted(feedback.get("objects", {})),  # ... and "not a person" marks
               "feedback_mtime": fb_mtime})  # the time of the feedback file read (see recordings.marked)


if __name__ == "__main__":
    streams = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(streams) != 1 or not STREAM_OK.match(streams[0]):
        sys.exit("usage: people.py STREAM [--rebuild] [--dry]")
    use(streams[0])
    main(dry="--dry" in sys.argv, rebuild="--rebuild" in sys.argv)
