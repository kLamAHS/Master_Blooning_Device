"""Realistic-pool synergy guard -- the check the selftest CAN'T provide.

meta.py's selftest and tools/simulate_solve.py both drive the brain with
DEGENERATE pools (near == mid == roomy == all), where a buffer's buddy
filter is never empty -- so they cannot see the real bug: on an actual scan
mask near-track spots are never "roomy", so a LARGE buffer (village) whose
candidate pool is `roomy` can never sit near the carry and lands far away.

This script rebuilds near/mid/roomy/all from a real mask EXACTLY as
mk.load_mask does, drives brain.next_genome() many times, and reports how
often the alchemist / village actually land within buff range of the carry
(optionally after the executor's large-tower roomy-snap). Run it before and
after a placement change; carry-linked support should climb toward ~90%.

    python tools/measure_synergy.py [--n 400] [--seed 0] [--snap]
"""

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _frange(lo, hi, step):
    out, x = [], lo
    while x < hi:
        out.append(round(float(x), 3))
        x += step
    return out
sys.path.insert(0, str(REPO))

import learner                                             # noqa: E402
import meta                                                # noqa: E402
from meta import MetaBrain, TrackModel                     # noqa: E402

LARGE = {"super", "village", "farm", "ace"}                # mk.LARGE_TOWERS
SNAP_LEASH = 0.20                                          # mk.py large-snap


def load_pools(mask_path):
    """Mirror mk.load_mask's near/mid/roomy/all derivation."""
    data = json.loads(Path(mask_path).read_text())
    points = data.get("valid_strict") or data.get("valid") or []
    step = data.get("step", 0.025)
    have = {(p[0], p[1]) for p in points}
    roomy = [p for p in points
             if all((round(p[0] + dx, 3), round(p[1] + dy, 3)) in have
                    for dx, dy in ((step, 0), (-step, 0),
                                   (0, step), (0, -step)))] or points
    every = {(round(p[0], 3), round(p[1], 3))
             for p in (data.get("valid") or points)}
    strict = {(round(p[0], 3), round(p[1], 3)) for p in points}

    def nb(c):
        return [(round(c[0] + dx, 3), round(c[1] + dy, 3))
                for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step))]

    near, mid = [], []
    if every:
        xs = sorted({c[0] for c in every})
        ys = sorted({c[1] for c in every})
        lattice = {(x, y)
                   for x in _frange(xs[0], xs[-1] + step / 2, step)
                   for y in _frange(ys[0], ys[-1] + step / 2, step)}
        invalid = lattice - every
        ring = {c for c in lattice
                if c[0] < xs[0] + 1.5 * step or c[0] > xs[-1] - 1.5 * step
                or c[1] < ys[0] + 1.5 * step or c[1] > ys[-1] - 1.5 * step}
        interior = invalid - ring
        seen, track = set(), set()
        for c in interior:
            if c in seen:
                continue
            comp, stack = set(), [c]
            while stack:
                q = stack.pop()
                if q in comp:
                    continue
                comp.add(q)
                stack += [n for n in nb(q) if n in interior and n not in comp]
            seen |= comp
            if len(comp) > len(track):
                track = comp
        dist = {c: 0 for c in track}
        frontier, d = list(track), 0
        while frontier and d < 5:
            d += 1
            nxt = []
            for c in frontier:
                for n in nb(c):
                    if n in lattice and n not in dist:
                        dist[n] = d
                        nxt.append(n)
            frontier = nxt
        near = [list(c) for c in strict if dist.get(c, 99) <= 2]
        mid = [list(c) for c in strict if 2 < dist.get(c, 99) <= 4]
    pools = {"near": near or points, "mid": mid,
             "all": points, "roomy": roomy}
    return pools, data


def snap_large(at, roomy):
    """Mirror the executor: a large tower is placed at the nearest roomy
    spot within the leash, not its planned spot."""
    best = min(roomy, key=lambda p: meta._dist(at, p), default=at)
    return best if meta._dist(at, best) <= SNAP_LEASH else at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask",
                    default=str(REPO / "masks" / "monkey_meadow_dart.json"))
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snap", action="store_true",
                    help="apply the executor's large-tower roomy-snap "
                         "before measuring (what the game actually places)")
    ap.add_argument("--min-buffed", type=float, default=None,
                    help="exit non-zero if carry-buffed fraction falls below "
                         "this (a regression guard the selftest can't give)")
    args = ap.parse_args()

    pools, data = load_pools(args.mask)
    near_s = {tuple(p) for p in pools["near"]}
    roomy_s = {tuple(p) for p in pools["roomy"]}
    print(f"pools: near={len(pools['near'])} mid={len(pools['mid'])} "
          f"roomy={len(pools['roomy'])} all={len(pools['all'])}  "
          f"near&roomy={len(near_s & roomy_s)}"
          f"{'  (0 => large buffers cannot sit near the track)' if not (near_s & roomy_s) else ''}")

    k = json.loads((REPO / "meta_knowledge.json").read_text())
    track = TrackModel(data)
    entry = min(track.cells, key=lambda c: track.progress[c])
    track.orient([entry[0], entry[1]])
    carries = set(k["roles"]["carry"])
    pool = ["dart", "tack", "bomb", "sniper", "ninja", "wizard", "druid",
            "alchemist", "glue", "ice"] + meta.META_EXTRA_TOWERS
    roomy = pools["roomy"]

    rng = random.Random(args.seed)
    brain = MetaBrain("m", "hard", target_round=100, explore=0.20,
                      knowledge=k, runs_path="/nonexistent", mode="chimps",
                      start_round=6)
    got = {"village": [0, 0], "alchemist": [0, 0]}
    n = buffed = both = 0
    for _ in range(args.n):
        g = brain.next_genome(rng, 5, pools, tower_pool=pool, track=track,
                              hero=True, large_towers=LARGE)
        tws = learner.towers_from_genome(g)
        carry_at = next((t["at"] for t in tws
                         if t["tower"] in carries and t.get("at")), None)
        if carry_at is None:
            continue
        n += 1
        linked = 0
        for t in tws:
            tt = t["tower"]
            if tt not in ("village", "alchemist") or not t.get("at"):
                continue
            at = t["at"]
            if args.snap and tt in LARGE:
                at = snap_large(at, roomy)
            r = (k["towers"].get(tt, {}).get("placement") or {}).get("range") \
                or 0.06
            got[tt][1] += 1
            if meta._dist(at, carry_at) <= r * 0.9:
                got[tt][0] += 1
                linked += 1
        buffed += linked >= 1
        both += linked >= 2

    tag = " (post-snap)" if args.snap else " (planned)"
    print(f"over {n} genomes{tag}:")
    for tt in ("village", "alchemist"):
        a, b = got[tt]
        print(f"  {tt:<10} linked-to-carry {a}/{b} = {100 * a / max(b, 1):.0f}%")
    buffed_frac = buffed / max(n, 1)
    print(f"  carry buffed (>=1 amp): {buffed}/{n} = {100 * buffed_frac:.0f}%")
    print(f"  both amps linked:       {both}/{n} = {100 * both / max(n, 1):.0f}%")
    if args.min_buffed is not None and buffed_frac < args.min_buffed:
        sys.exit(f"REGRESSION: carry-buffed {buffed_frac:.0%} < "
                 f"{args.min_buffed:.0%} -- support is scattering again")


if __name__ == "__main__":
    main()
