"""MetaBrain: meta-informed, self-learning layout generation for `farm`.

The idea in one paragraph: the community meta (distilled from the research
spreadsheet into meta_knowledge.json) is treated as a PRIOR, never as a
rulebook. Every tower/path/spot choice is a Thompson-sampling draw from a
Beta posterior whose pseudo-counts start at the meta score and are updated
by the bot's own episodes in runs_log.jsonl. With little data the bot
plays roughly what the spreadsheet says is good; as evidence accumulates,
its own results dominate — a tower the meta loves but that keeps dying on
THIS map gets sampled less, and an off-meta pick that keeps surviving gets
sampled more. On top of that sit two explicit emergence mechanisms: an
exploration rate (every decision has an `explore` chance of going uniform
random, so no option ever starves) and a genetic layer that mutates and
crosses over the best layouts the bot has personally discovered.

This module is deliberately free of screen/game dependencies (pure
stdlib), so it can be unit-tested and inspected without BTD6 running:

    python meta.py report monkey_meadow     # what has the bot learned?
    python meta.py selftest                 # offline sanity checks
"""

import json
import math
import random
import sys
from pathlib import Path

KNOWLEDGE_PATH = Path(__file__).parent / "meta_knowledge.json"
RUNS_PATH = Path(__file__).parent / "runs_log.jsonl"

# How many episodes' worth of pseudo-evidence the meta prior is worth.
# After ~PRIOR_STRENGTH real episodes featuring a tower, the bot's own
# experience outweighs the spreadsheet.
PRIOR_STRENGTH = 6.0

# Version handshake with mk.py. A stale meta.py sitting next to a newer
# mk.py (mixed zip extractions, leftover __pycache__) once CRASHED every
# episode with a TypeError; mk.py now checks this and degrades politely.
META_API = 3

# Rough base-cost rank (medium prices) used ONLY to order purchases when
# the learned price book hasn't seen a tower yet. Being off by $100 just
# reorders two buys; the greedy executor still waits for real cash.
ROUGH_COST = {
    "dart": 200, "glue": 275, "tack": 280, "boomerang": 325, "sniper": 350,
    "wizard": 400, "druid": 425, "engineer": 450, "ice": 500, "ninja": 500,
    "bomb": 525, "alchemist": 550, "hero": 600, "mortar": 750,
    "spike": 1000, "village": 1200, "super": 2500,
}

# The hero isn't in the knowledge base's tower table (which hero is
# equipped is chosen in the menu, invisible to the bot), so placement
# uses this profile: short-range coverage suits Sauda -- the sheet's
# low-micro early anchor -- and is a sane default for most heroes.
HERO_PLACEMENT = {"range": 0.045, "style": "coverage"}

# Rough cost of an upgrade TIER, for scheduling and cash reserves when
# the price book hasn't learned the real number yet. Coarse on purpose:
# being $500 off shifts a buy by a round or two, nothing more.
TIER_EST = {1: 300, 2: 700, 3: 1800, 4: 4500, 5: 14000}


def earned_by(r):
    """Very rough cumulative cash available by round r (start cash plus
    pop income and end-of-round bonuses, no farms). Only used to PACE
    the buy plan -- the executor still waits for real cash."""
    return 650 + 25 * r + 11 * r * r


def choose_buy(queue, cur_round, cash, cost_of, now, emergency=False,
               gate_ok=None):
    """The economy policy: which pending buy may spend money right now?

    - Buys unlock at their scheduled "round" (a good player doesn't dump
      four towers at round 1 and then upgrade whatever).
    - The most important DUE purchase is the head; while it saves up, it
      RESERVES its price. Equal-or-higher-priority items may still buy
      (two openers in parallel), lower-priority ones only if their cost
      fits in the SURPLUS above the reservation -- efficiency now so the
      big thing is affordable later.
    - A leak emergency drops all gates: buy any affordable defense NOW.

    - When cash comfortably exceeds every due obligation, the NEXT
      scheduled buy unlocks early: the income model paces the plan, but
      a rich run accelerates and a poor run waits -- reality decides.

    Items missing round/prio (the old uniform-random genomes) default to
    round 0 / prio 1, which reproduces the old first-awake behavior.
    Returns an index into queue, or None to hold every wallet."""
    head_prio = head_cost = None
    for j, it in enumerate(queue):
        prio = it.get("prio", 1)
        awake = it.get("_wake", 0) <= now
        if not emergency and gate_ok is not None and not gate_ok(it):
            continue        # conditional buy whose condition isn't met
        if emergency:
            if not awake:
                continue
            c = cost_of(it)
            if cash is None or c is None or c <= cash:
                return j
            continue
        if it.get("round", 0) > cur_round:
            # Ahead of schedule: allowed only when today's obligations
            # are covered -- cash must hold the head's reserve AND this
            # buy in full.
            if not awake:
                continue
            c = cost_of(it)
            if c is not None and cash is not None \
                    and cash - c >= (head_cost or 0):
                return j
            continue
        if head_prio is None:
            head_prio, head_cost = prio, cost_of(it)
            if awake:
                return j
            continue
        if not awake:
            continue
        if prio <= head_prio:
            return j
        c = cost_of(it)
        if head_cost is not None and c is not None and cash is not None \
                and cash - c >= head_cost:
            return j
    return None


# Land towers the meta layer may add beyond mk.py's classic FARM_TOWERS
# pool. All are placeable from a dart-scanned land mask and need no
# per-shot micro. (Heli/dartling chase the cursor, water towers need a
# sub-scanned mask, banana farms need collection clicks — all excluded.)
META_EXTRA_TOWERS = ["boomerang", "mortar", "spike", "village",
                     "super", "engineer"]


def _load_json(path):
    return json.loads(Path(path).read_text())


def _beta(rng, a, b):
    return rng.betavariate(max(a, 1e-3), max(b, 1e-3))


def _bucket(pt):
    """Coarse position bucket for spot-value learning (~5% of the map)."""
    return (round(pt[0] * 20) / 20, round(pt[1] * 20) / 20)


# Coordinates are fractions of width/height, but the screen is 16:9 --
# 0.1 of height is a much shorter walk than 0.1 of width. All physical
# distances (tower ranges, buff radii) are in width-fractions, with y
# corrected by this factor.
ASPECT = 9.0 / 16.0


def _dist(a, b):
    return math.hypot(a[0] - b[0], (a[1] - b[1]) * ASPECT)


# Minimum spacing between planned towers, in width-fractions. A tower
# footprint is ~0.013 wide-radius, so two centers need ~0.028 -- the old
# 0.015 planned towers ON TOP of each other and the executor burned a
# minute of retries discovering the game wouldn't allow it.
SEP = 0.028
SEP_LARGE = 0.045


def _spread(cands, taken, sep):
    """Candidates at least sep away from every taken spot; if none
    qualify, the single candidate FARTHEST from the layout (never an
    overlapping one at random)."""
    free = [c for c in cands
            if all(_dist(c, t) >= sep for t in taken)]
    if free:
        return free
    if not taken or not cands:
        return cands
    return [max(cands, key=lambda c: min(_dist(c, t) for t in taken))]


class TrackModel:
    """Geometry of the bloon path, derived from a scan mask. The mask's
    lattice already encodes the track: it is the largest connected blob
    of interior cells that REFUSE placement (trees hug the border and are
    excluded). On top of that blob this computes a 0..1 `progress`
    coordinate along the path (graph distance from one end) and per-spot
    coverage -- which stretch of track a tower on a spot can actually
    hit. Which end is the ENTRY cannot be known from pixels alone;
    `farm` senses it once per map by watching where the first bloons
    appear, then `orient()` pins progress 0 to the entry."""

    def __init__(self, mask):
        step = mask.get("step", 0.025)
        self.step = step
        strict = mask.get("valid_strict") or mask.get("valid") or []
        every = {(round(p[0], 3), round(p[1], 3))
                 for p in (mask.get("valid") or strict)}
        self.cells, self.progress = [], {}
        self.oriented = False
        self.ok = False
        if not every:
            return

        def nb(c):
            return [(round(c[0] + dx, 3), round(c[1] + dy, 3))
                    for dx, dy in ((step, 0), (-step, 0),
                                   (0, step), (0, -step))]

        xs = sorted({c[0] for c in every})
        ys = sorted({c[1] for c in every})
        lattice = set()
        x = xs[0]
        while x <= xs[-1] + step / 2:
            y = ys[0]
            while y <= ys[-1] + step / 2:
                lattice.add((round(x, 3), round(y, 3)))
                y += step
            x += step
        invalid = lattice - every
        ring = {c for c in lattice
                if c[0] < xs[0] + 1.5 * step or c[0] > xs[-1] - 1.5 * step
                or c[1] < ys[0] + 1.5 * step or c[1] > ys[-1] - 1.5 * step}
        interior = invalid - ring
        seen, blob = set(), set()
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
            if len(comp) > len(blob):
                blob = comp
        if len(blob) < 8:            # no believable track in this mask
            return

        def bfs(start):
            dist = {start: 0}
            frontier = [start]
            while frontier:
                nxt = []
                for c in frontier:
                    for n in nb(c):
                        if n in blob and n not in dist:
                            dist[n] = dist[c] + 1
                            nxt.append(n)
                frontier = nxt
            return dist

        # Double BFS: the two graph-farthest cells are the track's ends.
        d0 = bfs(next(iter(blob)))
        end_a = max(d0, key=d0.get)
        d1 = bfs(end_a)
        span = max(d1.values()) or 1
        self.cells = sorted(d1)
        self.progress = {c: d1[c] / span for c in d1}
        self.ok = True

    def orient(self, entry_pt):
        """Pin progress 0 to the sensed bloon entry point."""
        if not self.ok:
            return
        near = min(self.cells, key=lambda c: _dist(c, entry_pt))
        if self.progress[near] > 0.5:
            self.progress = {c: 1.0 - p for c, p in self.progress.items()}
        self.oriented = True

    def covered(self, pt, r):
        return [c for c in self.cells if _dist(c, pt) <= r]

    def exposure(self, pt, r):
        """Fraction of the track a tower at pt can hit: the 'damage more
        bloons for longer' number. Bends and long straights score high,
        corner decorations score ~0."""
        if not self.ok:
            return 0.0
        return len(self.covered(pt, r)) / len(self.cells)

    def span(self, pt, r):
        """(first, mean, last) progress of the covered stretch, or None."""
        cov = self.covered(pt, r)
        if not cov:
            return None
        ps = sorted(self.progress[c] for c in cov)
        return ps[0], sum(ps) / len(ps), ps[-1]


class MetaBrain:
    def __init__(self, map_name, difficulty, target_round=40,
                 explore=0.30, evolve=True, knowledge=None, runs_path=None):
        self.map_name = map_name
        self.difficulty = difficulty
        self.target = max(int(target_round), 1)
        self.explore = min(max(float(explore), 0.0), 1.0)
        self.evolve = evolve
        self.k = knowledge or _load_json(KNOWLEDGE_PATH)
        self.towers = self.k["towers"]
        self.roles = self.k["roles"]
        self.solutions = self.k["solutions"]
        self.threats = self.k["threats"]
        # Posteriors: Beta(a, b) per tower, per (tower, main path), and
        # per coarse map position bucket.
        self.t_post = {}
        self.p_post = {}
        self.pos_post = {}
        self.history = []
        self.last_strategy = {"kind": "uniform"}
        runs = Path(runs_path) if runs_path else RUNS_PATH
        if runs.exists():
            for line in runs.read_text().splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                self.observe(row, quiet=True)

    # ---------------------------------------------------------- learning

    def _prior(self, ttype):
        info = self.towers.get(ttype)
        q = (info["score"] / 100.0) if info else 0.55
        return [PRIOR_STRENGTH * q, PRIOR_STRENGTH * (1.0 - q)]

    def _reward(self, row):
        """0..1 per episode: how far did the layout get toward the target
        round? Survival = 1. Unknown endings don't teach anything."""
        if row.get("outcome") == "survived":
            return 1.0
        if row.get("outcome") != "defeat":
            return None
        reached = row.get("final_round")
        if reached is None:
            return None
        return min(max(reached / self.target, 0.0), 1.0) * 0.95

    def usable(self, row):
        return (row.get("mode") == "farm"
                and row.get("map") == self.map_name
                and row.get("towers")
                and self._reward(row) is not None)

    def observe(self, row, quiet=False):
        """Fold one episode row (runs_log.jsonl format) into the
        posteriors. Called for every historical line at startup and after
        each live episode."""
        if not self.usable(row):
            return
        r = self._reward(row)
        for t in row["towers"]:
            ttype = t.get("tower", "").lower()
            if not ttype:
                continue
            a, b = self.t_post.setdefault(ttype, self._prior(ttype))
            self.t_post[ttype] = [a + r, b + (1.0 - r)]
            path = t.get("path") or [0, 0, 0]
            if max(path) > 0:
                main = path.index(max(path))
                pa, pb = self.p_post.setdefault((ttype, main), [1.0, 1.0])
                self.p_post[(ttype, main)] = [pa + r, pb + (1.0 - r)]
            if t.get("at"):
                bk = _bucket(t["at"])
                ba, bb = self.pos_post.setdefault(bk, [1.0, 1.0])
                self.pos_post[bk] = [ba + r, bb + (1.0 - r)]
        self.history.append(row)
        if not quiet:
            n = len(self.history)
            print(f"   [brain] learned from episode "
                  f"(reward {r:.2f}, {n} total on this map)")

    def elites(self, top=8):
        rows = [(self._reward(r), i, r) for i, r in enumerate(self.history)
                if self.usable(r)]
        rows.sort(key=lambda x: (-x[0], -x[1]))   # best first, recent first
        return [(rw, r) for rw, _, r in rows[:top]]

    # ---------------------------------------------------------- sampling

    def _theta(self, rng, ttype):
        a, b = self.t_post.get(ttype) or self._prior(ttype)
        return _beta(rng, a, b)

    def _pick_tower(self, rng, candidates, chosen):
        """Thompson draw with a synergy nudge from already-chosen towers.
        With probability `explore` the draw is uniform instead — the
        never-starve guarantee."""
        if not candidates:
            return None
        if rng.random() < self.explore:
            return rng.choice(candidates)
        best, best_w = None, -1.0
        for ttype in candidates:
            w = self._theta(rng, ttype)
            partners = set(self.towers.get(ttype, {}).get("partners", []))
            mates = sum(1 for c in chosen
                        if c in partners
                        or ttype in self.towers.get(c, {}).get("partners", []))
            w *= 1.0 + 0.25 * mates
            # Sharply diminishing returns on duplicates: a second glue
            # is occasionally right, a third 000 glue never is.
            w *= 0.3 ** sum(1 for c in chosen if c == ttype)
            if w > best_w:
                best, best_w = ttype, w
        return best

    def _pick_spot(self, rng, pools, taken, large=False):
        """Sample a placement point: keep farm's near/mid track preference
        as the base distribution, weight candidates by the learned value
        of their map region, keep a little breathing room between towers."""
        if large:
            pool = pools.get("roomy") or pools.get("all") or []
        else:
            roll = rng.random()
            if roll < 0.70 and pools.get("near"):
                pool = pools["near"]
            elif roll < 0.95 and (pools.get("mid") or pools.get("near")):
                pool = pools.get("mid") or pools.get("near")
            else:
                pool = pools.get("all") or []
        if not pool:
            return None
        cands = [rng.choice(pool) for _ in range(min(30, len(pool)))]
        cands = _spread(cands, taken, SEP_LARGE if large else SEP)
        if rng.random() < self.explore or not self.pos_post:
            return list(rng.choice(cands))
        best, best_w = None, -1.0
        for c in cands:
            ba, bb = self.pos_post.get(_bucket(c), [1.0, 1.0])
            w = _beta(rng, ba, bb)
            if w > best_w:
                best, best_w = c, w
        return list(best)

    def _pos_theta(self, rng, pt):
        ba, bb = self.pos_post.get(_bucket(pt), [1.0, 1.0])
        return _beta(rng, ba, bb)

    def _spot_for(self, rng, ttype, pools, taken, placed, track,
                  large=False):
        """Style-aware placement. `placed` is [(ttype, spot), ...] already
        assigned in this layout. Scores candidates by what the tower
        actually wants -- track coverage for DPS, just-upstream coverage
        for debuffers, buff adjacency for alch/village, late track for
        spikes, remoteness for global towers -- multiplied by the learned
        per-region posterior, so experience still bends the geometry."""
        if track is None or not track.ok or rng.random() < self.explore:
            return self._pick_spot(rng, pools, taken, large=large)
        prof = self.towers.get(ttype, {}).get("placement") \
            or (HERO_PLACEMENT if ttype == "hero" else {})
        style = prof.get("style", "coverage")
        r = prof.get("range") or 0.06
        if large:
            base = pools.get("roomy") or pools.get("all") or []
        else:
            base = ((pools.get("near") or []) + (pools.get("mid") or [])) \
                or pools.get("all") or []
        if not base:
            return None
        cands = [rng.choice(base) for _ in range(min(70, len(base)))]
        if style == "buddy" and placed:
            # A random sample can easily miss the small disc around the
            # teammates a buffer exists to stand in -- guarantee the
            # scorer actually gets to see in-range candidates.
            anchors = [s for _, s in placed if s]
            nearby = [p for p in base
                      if any(_dist(p, a) <= r * 0.85 for a in anchors)]
            if nearby:
                cands = [rng.choice(nearby)
                         for _ in range(min(30, len(nearby)))] + cands[:20]
        cands = _spread(cands, taken, SEP_LARGE if large else SEP)

        # The carry anchors the geometry for debuffers and buffers.
        carries = set(self.roles.get("carry", []))
        carry_spot = next((s for t, s in placed if t in carries and s),
                          None)
        carry_span = None
        if carry_spot is not None:
            carry_r = (self.towers.get(
                next(t for t, s in placed if t in carries and s), {})
                .get("placement") or {}).get("range") or 0.06
            carry_span = track.span(carry_spot, carry_r)

        best, best_s = None, 0.0
        for c in cands:
            exp = track.exposure(c, r)
            if style == "upstream":
                if carry_span and track.oriented:
                    sp = track.span(c, r)
                    if sp is None:
                        s = 0.0
                    else:
                        gap = carry_span[1] - sp[1]   # + = upstream of it
                        fit = (1.0 if 0.0 <= gap <= 0.18 else
                               0.4 if -0.06 <= gap < 0.0 else 0.05)
                        s = exp * fit
                elif carry_spot is not None:
                    # Direction unknown: co-locate with the carry so the
                    # debuff at least overlaps its kill zone.
                    s = exp * max(0.1, 1.0 - 4.0 * _dist(c, carry_spot))
                else:
                    s = exp
            elif style == "buddy":
                mates = sum((2.0 if t in carries else 1.0)
                            for t, spot in placed
                            if spot and _dist(c, spot) <= r * 0.9)
                s = mates + 0.15 * exp
            elif style == "downstream":
                sp = track.span(c, r)
                late = sp[1] if (sp and track.oriented) else 0.5
                s = exp * (0.3 + late)
            elif style == "offside":
                s = 1.0 / (1.0 + 40.0 * track.exposure(c, 0.06))
            else:                                    # coverage
                s = exp
                if track.oriented:
                    sp = track.span(c, r)
                    if sp:
                        # All else equal, kill bloons EARLY: damage near
                        # the entry leaves room for error; a defense
                        # camped at the exit pops with zero margin.
                        s *= 1.25 - 0.50 * sp[1]
            # Learned-region posterior nudges the score +/-20%. (It once
            # swung 0.5x-1.5x, which drowned out the small absolute
            # exposure differences of short-range towers -- heroes were
            # landing on 1%-coverage spots on pure noise.)
            s *= 0.8 + 0.4 * self._pos_theta(rng, c)
            if s > best_s:
                best, best_s = c, s
        if best is None:
            return self._pick_spot(rng, pools, taken, large=large)
        return list(best)

    def _placement_order(self, picks):
        """Spot-assignment order: the carry anchors first, coverage DPS
        next, then debuffers/cleanup that position relative to it, and
        buffers last (they need to see where everyone sits)."""
        rank = {"coverage": 1, "upstream": 2, "downstream": 3,
                "offside": 4, "buddy": 5}
        carries = set(self.roles.get("carry", []))

        def key(i):
            ttype, role = picks[i]
            style = (self.towers.get(ttype, {}).get("placement")
                     or {}).get("style", "coverage")
            return (0 if (role == "carry" or ttype in carries)
                    else rank.get(style, 1), i)
        return sorted(range(len(picks)), key=key)

    def _pick_build(self, rng, ttype):
        """(main, cross) for a tower: meta build templates re-weighted by
        the learned per-path posterior; explore = any legal combo."""
        if rng.random() < self.explore:
            main = rng.randrange(3)
            cross = rng.choice([p for p in range(3) if p != main])
            return main, cross, "explore"
        builds = self.towers.get(ttype, {}).get("builds") or []
        if not builds:
            main = rng.randrange(3)
            cross = rng.choice([p for p in range(3) if p != main])
            return main, cross, "no-template"
        best, best_w = None, -1.0
        for b in builds:
            pa, pb = self.p_post.get((ttype, b["main"]), [1.0, 1.0])
            w = b.get("weight", 0.5) * (0.5 + _beta(rng, pa, pb))
            if w > best_w:
                best, best_w = b, w
        return best["main"], best["cross"], best.get("label", "meta")

    # ------------------------------------------------- threat coverage

    def _coverage_fixes(self, rng, picks):
        """Make sure the layout answers the threats it will actually meet
        before target_round (Meta thesis #2: solve every property). Returns
        {tower_index: {path_i: min_tier}} requirements and possibly swaps
        a tower type in `picks` to cover a hole."""
        needs = {}
        for kind in ("camo", "lead"):
            first = min((r for t in self.threats if t["kind"] == kind
                         for r in t["rounds"]), default=None)
            if first is None or first > self.target:
                continue
            solvers = self.solutions.get(kind, {})
            covered = any(t in solvers and solvers[t] is None
                          for t, _ in picks)
            if covered:
                continue
            upgradable = [(i, solvers[t]) for i, (t, _) in enumerate(picks)
                          if t in solvers and solvers[t] is not None]
            if upgradable:
                i, (path_i, tier) = rng.choice(upgradable)
                needs.setdefault(i, {})[path_i] = max(
                    needs.get(i, {}).get(path_i, 0), tier)
                continue
            # Nobody can solve it: swap the last non-carry pick for the
            # strongest solver (skip in explore-heavy runs half the time
            # so pure exploration still exists).
            if rng.random() < self.explore:
                continue
            solver_types = [t for t, req in solvers.items()
                            if t in self.towers]
            if not solver_types:
                continue
            best = max(solver_types, key=lambda t: self._theta(rng, t))
            carries = set(self.roles.get("carry", []))
            for i in range(len(picks) - 1, -1, -1):
                if picks[i][0] not in carries and picks[i][0] != "hero":
                    picks[i] = (best, picks[i][1])
                    req = solvers[best]
                    if req is not None:
                        needs.setdefault(i, {})[req[0]] = req[1]
                    break
        return needs

    def _deadline(self, ttype, main, path_i, tier, needs_for_tower):
        """Approximate round by which an upgrade step should be owned;
        used only to ORDER the buy queue (the executor stays greedy)."""
        for p_i, min_tier in (needs_for_tower or {}).items():
            if path_i == p_i and tier <= min_tier:
                kinds = [k for k, sol in self.solutions.items()
                         if sol.get(ttype) == [p_i, min_tier]]
                first = min((r for t in self.threats
                             if t["kind"] in kinds for r in t["rounds"]),
                            default=24)
                return first - 2
        if path_i == main:
            return 8 + 7 * tier          # carry path: t4 by ~r36
        return 14 + 8 * tier             # crosspath: trails the main

    # ------------------------------------------------------ genome build

    def next_genome(self, rng, n_towers, pools, is_locked=None,
                    large_towers=frozenset(), tower_pool=None,
                    price_of=None, track=None, hero=False):
        """Produce a buy list in run_episode's format. Rolls between a
        fresh meta-templated layout and an evolution of an elite one.
        `track` (a TrackModel) turns on geometry-aware placement; `hero`
        adds the equipped hero as an early anchor placement."""
        is_locked = is_locked or (lambda *a: False)
        elites = self.elites() if self.evolve else []
        p_evolve = min(0.5, len(elites) / 10.0) if len(elites) >= 3 else 0.0
        if rng.random() < p_evolve:
            genome = self._evolved_genome(rng, n_towers, pools, is_locked,
                                          large_towers, tower_pool, elites,
                                          track, hero)
            if genome:
                return genome
        return self._fresh_genome(rng, n_towers, pools, is_locked,
                                  large_towers, tower_pool, price_of,
                                  track, hero)

    def _role_slots(self, n, hero=False):
        """Roles for n tower slots. With a hero anchoring, the hero IS
        the opener -- a separate cheap opener would just split cash away
        from the carry's first tiers (the second-boomerang trap)."""
        if n <= 0:
            return []
        if hero:
            slots = ["carry", "amplifier", "control", "free"]
            return slots[:n] + ["free"] * max(0, n - 4)
        slots = ["opener", "carry", "amplifier", "control"]
        if n == 1:
            return ["carry"]
        return slots[:n] + ["free"] * max(0, n - 4)

    def _fresh_genome(self, rng, n_towers, pools, is_locked,
                      large_towers, tower_pool, price_of, track=None,
                      hero=False):
        pool = list(tower_pool or
                    [t for t in self.towers if t in ROUGH_COST])
        picks = []       # [(ttype, role), ...]
        if hero:
            picks.append(("hero", "hero"))   # free scaling: always early
        for role in self._role_slots(n_towers, hero=hero):
            cands = pool if role == "free" else \
                [t for t in self.roles.get(role, []) if t in pool] or pool
            ttype = self._pick_tower(rng, cands, [t for t, _ in picks])
            if ttype:
                picks.append((ttype, role))
        needs = self._coverage_fixes(rng, picks)
        genome, meta = self._assemble(rng, picks, needs, pools, is_locked,
                                      large_towers, price_of, track)
        self.last_strategy = {
            "kind": "meta", "explore": self.explore,
            "placement": ("track" + ("+flow" if track.oriented else "")
                          if track and track.ok else "pools"),
            "roles": [f"{r}:{t}" for t, r in picks], **meta}
        return genome

    def _schedule(self, entries):
        """Assign every buy the round it should HAPPEN. Entries are
        walked most-important-first along a rough income curve, so the
        plan never wants more money than the game can have produced --
        that is what stops the old buy-everything-at-once behavior.
        Threat answers are pinned to land before their threat round even
        if the curve says later (the executor's reserve makes the cash
        appear in time). An upgrade never schedules before its tower."""
        order = sorted(range(len(entries)),
                       key=lambda i: (entries[i]["prio"],
                                      entries[i]["deadline"]))
        cum = 0.0
        place_round = {}
        for i in order:
            e = entries[i]
            cum += e.get("est") or 500
            r = 1
            while r < 100 and earned_by(r) * 0.85 < cum:
                r += 1
            if e.get("by"):                 # threat answers keep their
                r = min(r, max(1, e["by"]))  # date whatever income says
            # Never schedule past the episode: the income model paces,
            # but real cash decides -- anything the curve puts after the
            # target unlocks just before the end and buys only if the
            # money is actually there.
            e["round"] = min(r, max(1, self.target - 2))
            if e["do"] == "place":
                place_round[e["ref"]] = e["round"]
        # A threat-capped upgrade drags its tower's BASE forward too --
        # a camo upgrade due by r22 is useless on a tower placed at r25.
        min_by = {}
        for e in entries:
            if e.get("by"):
                min_by[e["ref"]] = min(min_by.get(e["ref"], 99),
                                       max(1, e["by"]))
        for e in entries:
            if e["do"] == "place" and e["ref"] in min_by:
                e["round"] = min(e["round"], min_by[e["ref"]])
                place_round[e["ref"]] = e["round"]
        for e in entries:
            if e["do"] == "upgrade":
                e["round"] = max(e["round"], place_round.get(e["ref"], 1))
        entries.sort(key=lambda e: (e["round"], e["prio"],
                                    e["do"] != "place", e["deadline"]))
        for e in entries:
            del e["deadline"]
        return entries

    def _assemble(self, rng, picks, needs, pools, is_locked,
                  large_towers, price_of=None, track=None):
        """picks + coverage requirements -> ordered place/upgrade actions.
        Spots are assigned in placement order (carry anchors, buffers
        last); places go cheapest-first (early rounds need towers NOW);
        upgrades are sorted by threat deadline with noise for variety."""
        spots = {}
        taken, placed_ctx = [], []
        for i in self._placement_order(picks):
            ttype = picks[i][0]
            spot = self._spot_for(rng, ttype, pools, taken, placed_ctx,
                                  track, large=ttype in large_towers)
            if spot is None:
                continue
            spots[i] = spot
            taken.append(spot)
            placed_ctx.append((ttype, spot))
        placed = []      # (ref, ttype, spot, main, cross, label)
        for ref, (ttype, _role) in enumerate(picks):
            if ref not in spots:
                continue
            main, cross, label = self._pick_build(rng, ttype)
            placed.append((ref, ttype, spots[ref], main, cross, label))

        def base_cost(ttype):
            if price_of:
                known = price_of(ttype)
                if known:
                    return known
            return ROUGH_COST.get(ttype, 600)

        def tier_cost(ttype, p_i, tier):
            if price_of:
                try:
                    known = price_of(ttype, p_i, tier)
                except TypeError:      # old single-arg lookup
                    known = None
                if known:
                    return known
            return TIER_EST.get(tier, 800)

        placed.sort(key=lambda p: (base_cost(p[1]), p[0]))
        # Upgrade-first economics: a good player does NOT drop four bases
        # and then start upgrading. The hero and opener anchor, the carry
        # base follows, then the carry's first tiers -- support bases and
        # everything else joins the plan AFTER the carry has teeth.
        place_plan = {"hero": (0, 1.0), "opener": (0, 2.0),
                      "carry": (0, 3.0), "amplifier": (1, 9.0),
                      "control": (1, 13.0), "free": (2, 17.0)}
        # A tower whose BASE answers a threat (ninja = camo, bomb = lead)
        # must be down before that threat, whatever its role's pacing
        # says -- the same hard "by" cap upgrades get.
        base_by = {}
        for kind in ("camo", "lead"):
            first = min((r for t in self.threats if t["kind"] == kind
                         for r in t["rounds"]), default=None)
            if first is None or first > self.target:
                continue
            sol = self.solutions.get(kind, {})
            for i, (tt, _role) in enumerate(picks):
                if tt in sol and sol[tt] is None:
                    base_by[i] = min(base_by.get(i, 99), first - 2)
                    break
        # Tower identity: every place gets an unambiguous name so logs
        # read "boomerang#1(carry) bottom t2", never "upgrade ref2".
        carry_order = next((o for o, (r, *_x) in enumerate(placed)
                            if picks[r][1] == "carry"), None)
        entries, spot_notes = [], []
        for order, (ref, ttype, spot, main, cross, label) in enumerate(placed):
            role = picks[ref][1]
            p_prio, p_dl = place_plan.get(role, (1, 12.0))
            name = f"{ttype}#{order}({role})"
            if track and track.ok:
                prof = self.towers.get(ttype, {}).get("placement") \
                    or (HERO_PLACEMENT if ttype == "hero" else {})
                r_t = prof.get("range") or 0.06
                sp = track.span(spot, r_t)
                cov = (f"covers path {sp[0]:.2f}-{sp[2]:.2f} "
                       f"({track.exposure(spot, r_t):.0%} of track)"
                       if sp else "NO track in range")
                spot_notes.append(
                    f"{name} @ [{spot[0]:.2f},{spot[1]:.2f}]  {cov}")
            entry = {"do": "place", "tower": ttype,
                     "at": [spot[0], spot[1]], "ref": order,
                     "name": name,
                     "prio": p_prio, "deadline": p_dl,
                     "est": base_cost(ttype)}
            if ref in base_by:
                entry["by"] = base_by[ref]
            if role in ("amplifier", "control", "free") \
                    and carry_order is not None:
                # Conditional support: these bases wait until the carry
                # is stable (main path t3). A threat date ("by") or a
                # leak emergency still overrides -- support arrives when
                # NEEDED, not on a timer.
                entry["gate"] = {"ref": carry_order, "tier": 3}
            entries.append(entry)
            if ttype == "hero":
                continue          # heroes level up on their own: no buys
            need = needs.get(ref, {})
            main_target = 5 if rng.random() < 0.10 else rng.randint(3, 4)
            cross_target = rng.randint(1, 2)
            want = {main: main_target, cross: cross_target}
            for p_i, min_tier in need.items():
                want[p_i] = max(want.get(p_i, 0), min_tier)
            # Two-path rule: only one path past tier 2. A threat answer
            # that itself needs tier 3+ (e.g. Signal Flare) outranks the
            # carry main; tier-2 needs survive being capped anyway.
            deep = [p for p, t in want.items() if t > 2]
            if len(deep) > 1:
                deep_need = [p for p in deep if need.get(p, 0) > 2]
                keep = deep_need[0] if deep_need else \
                    (main if main in deep else deep[0])
                for p in deep:
                    if p != keep:
                        want[p] = 2
            if len(want) > 2:                     # at most two open paths
                # Never trim away a threat-coverage path: a dropped camo
                # answer is exactly how layouts die blind at round 24.
                keep = sorted(want, key=lambda p: (0 if p in need else 1,
                                                   -want[p]))[:2]
                want = {p: want[p] for p in keep}
            for p_i in sorted(want):
                for tier in range(1, want[p_i] + 1):
                    if is_locked(ttype, p_i, tier):
                        break
                    vec = [0, 0, 0]
                    vec[p_i] = 1
                    is_need = tier <= need.get(p_i, 0)
                    early_carry = (role == "carry" and p_i == main
                                   and tier <= 3)
                    dl = self._deadline(ttype, main, p_i, tier, need)
                    if early_carry:
                        # The carry's first tiers outrank every support
                        # BASE: upgrade the tower you have before buying
                        # three more. Tight noise -- these must not
                        # leapfrog the opener/carry bases themselves.
                        dl = 2.0 + 3.0 * tier
                    noise = 1.0 if early_carry else 4.0
                    entry = {
                        "do": "upgrade", "ref": order, "path": vec,
                        "prio": 0 if (is_need or early_carry)
                        else (1 if p_i == main else 2),
                        "deadline": dl + rng.uniform(-noise, noise),
                        "est": tier_cost(ttype, p_i, tier)}
                    if is_need:
                        entry["by"] = int(dl)  # HARD cap: pre-threat
                    entries.append(entry)
        genome = self._schedule(entries)
        meta = {"builds": [f"{t} {l}" for _r, t, _s, _m, _c, l in placed],
                "spots": spot_notes}
        return genome, meta

    # --------------------------------------------------------- evolution

    def _evolved_genome(self, rng, n_towers, pools, is_locked,
                        large_towers, tower_pool, elites, track=None,
                        hero=False):
        """Mutate (and sometimes cross over) the best layouts found so
        far. This is where genuinely emergent tactics come from: the
        parents are the bot's own discoveries, and mutations are free to
        wander off-meta."""
        pool = list(tower_pool or
                    [t for t in self.towers if t in ROUGH_COST])
        weights = [max(rw, 0.05) for rw, _ in elites]
        parent = rng.choices(elites, weights=weights, k=1)[0][1]
        towers = [dict(t) for t in parent.get("towers", []) if t.get("at")]
        label = f"evolve(r{parent.get('final_round')})"
        if len(elites) >= 2 and rng.random() < 0.4:
            other = rng.choices(elites, weights=weights, k=1)[0][1]
            merged = towers + [dict(t) for t in other.get("towers", [])
                               if t.get("at")]
            rng.shuffle(merged)
            towers = merged[:max(n_towers, 2)]
            label = (f"crossover(r{parent.get('final_round')}"
                     f"+r{other.get('final_round')})")
        if not towers:
            return None
        # Crossover (and old stacked-parent rows) can put two towers on
        # the same spot -- the game refuses that, so prune here.
        pruned = []
        for t in towers:
            if all(_dist(t["at"], u["at"]) >= SEP for u in pruned):
                pruned.append(t)
        towers = pruned
        if hero and not any(t["tower"] == "hero" for t in towers):
            others = [(x["tower"], x["at"]) for x in towers]
            spot = self._spot_for(rng, "hero", pools,
                                  [s for _, s in others], others, track)
            if spot:
                towers.insert(0, {"tower": "hero", "at": spot,
                                  "path": [0, 0, 0]})
        mutations = []
        for t in towers:
            roll = rng.random()
            if t["tower"] == "hero" and roll >= 0.20:
                continue    # the hero may relocate, never morph
            if roll < 0.20:                       # relocate (style-aware)
                others = [(x["tower"], x["at"]) for x in towers
                          if x is not t]
                spot = self._spot_for(rng, t["tower"], pools,
                                      [s for _, s in others], others,
                                      track,
                                      large=t["tower"] in large_towers)
                if spot:
                    t["at"] = spot
                    mutations.append(f"move:{t['tower']}")
            elif roll < 0.32:                     # swap species
                old = t["tower"]
                alt = self._pick_tower(rng, pool, [x["tower"] for x in towers])
                if alt and alt != old:
                    t["tower"] = alt
                    t["path"] = [0, 0, 0]
                    mutations.append(f"swap:{old}->{alt}")
            elif roll < 0.47:                     # push the build deeper
                # list() matters: dict(t) was a shallow copy, so without
                # it this would mutate the history row's own path list
                # and corrupt the elite pool for every later episode.
                path = list(t.get("path") or [0, 0, 0])
                main = path.index(max(path)) if max(path) else \
                    self._pick_build(rng, t["tower"])[0]
                path[main] = min(5, max(path[main], 0) + rng.randint(1, 2))
                t["path"] = path
                mutations.append(f"deeper:{t['tower']}")
        if len(towers) < n_towers and rng.random() < 0.5:
            ttype = self._pick_tower(rng, pool, [x["tower"] for x in towers])
            others = [(x["tower"], x["at"]) for x in towers]
            spot = self._spot_for(rng, ttype, pools,
                                  [s for _, s in others], others, track,
                                  large=ttype in large_towers) \
                if ttype else None
            if ttype and spot:
                main, cross, _ = self._pick_build(rng, ttype)
                path = [0, 0, 0]
                path[main] = rng.randint(2, 4)
                path[cross] = rng.randint(0, 2)
                towers.append({"tower": ttype, "at": spot, "path": path})
                mutations.append(f"add:{ttype}")
        elif len(towers) > 2 and rng.random() < 0.15:
            weakest = min(towers,
                          key=lambda x: self._theta(rng, x["tower"]))
            towers.remove(weakest)
            mutations.append(f"drop:{weakest['tower']}")

        entries = []
        towers.sort(key=lambda t: ROUGH_COST.get(t["tower"], 600))
        for order, t in enumerate(towers):
            entries.append({"do": "place", "tower": t["tower"],
                            "at": list(t["at"]), "ref": order,
                            "name": f"{t['tower']}#{order}(evolved)",
                            "prio": 0, "deadline": 1.0,
                            "est": ROUGH_COST.get(t["tower"], 600)})
            if t["tower"] == "hero":
                continue          # heroes level up on their own: no buys
            path = t.get("path") or [0, 0, 0]
            if max(path) == 0:
                main, cross, _ = self._pick_build(rng, t["tower"])
                path = [0, 0, 0]
                path[main] = rng.randint(2, 4)
                path[cross] = rng.randint(0, 2)
            main = path.index(max(path))
            for p_i, target in enumerate(path):
                for tier in range(1, min(target, 5) + 1):
                    if p_i != main and tier > 2:
                        break
                    if is_locked(t["tower"], p_i, tier):
                        break
                    vec = [0, 0, 0]
                    vec[p_i] = 1
                    entries.append({
                        "do": "upgrade", "ref": order, "path": vec,
                        "prio": 1 if p_i == main else 2,
                        "deadline": self._deadline(t["tower"], main, p_i,
                                                   tier, None)
                        + rng.uniform(-4, 4),
                        "est": TIER_EST.get(tier, 800)})
        genome = self._schedule(entries)
        self.last_strategy = {"kind": label, "explore": self.explore,
                              "placement": ("track"
                                            + ("+flow" if track.oriented
                                               else "")
                                            if track and track.ok
                                            else "pools"),
                              "mutations": mutations}
        return genome

    # ---------------------------------------------------------- describe

    def describe_genome(self, genome):
        places = [g for g in genome if g["do"] == "place"]
        ups = sum(1 for g in genome if g["do"] == "upgrade")
        kinds = ", ".join(g.get("name") or g["tower"] for g in places)
        s = self.last_strategy
        extra = ""
        if "mutations" in s:
            extra = f" [{', '.join(s['mutations'])}]" if s["mutations"] \
                else " [clone, no mutations]"
        elif s.get("roles"):
            extra = f" [{', '.join(s['roles'])}]"
        spots = f" spots={s['placement']}" if s.get("placement") else ""
        out = (f"   strategy={s.get('kind', '?')}{spots}{extra}\n"
               f"   layout: {kinds} (+{ups} upgrades)")
        for note in s.get("spots", []):
            out += f"\n      {note}"
        return out

    def report(self, track=None, mask_points=None):
        lines = [f"MetaBrain report -- map '{self.map_name}', "
                 f"difficulty {self.difficulty}, target r{self.target}",
                 f"episodes learned from: {len(self.history)}", ""]
        if track and track.ok and mask_points:
            ranked = sorted(mask_points,
                            key=lambda p: -track.exposure(p, 0.06))[:5]
            flow = ("bloons enter near progress 0"
                    if track.oriented else
                    "flow direction not sensed yet (farm learns it on "
                    "its first episode)")
            lines.append(f"map model: {len(track.cells)} track cells; "
                         f"{flow}")
            lines.append("prime real estate (most track in range): "
                         + "  ".join(
                             f"[{p[0]:.2f},{p[1]:.2f}]"
                             f"={track.exposure(p, 0.06):.0%}"
                             for p in ranked))
            lines.append("")
        lines.append(f"{'tower':<11}{'meta':>5}{'seen':>6}{'learned':>9}"
                     f"  verdict")
        prior_only, moved = [], []
        for ttype in sorted(self.towers, key=lambda t:
                            -(self.towers[t].get("score") or 0)):
            score = self.towers[ttype].get("score", 0)
            post = self.t_post.get(ttype)
            if post is None:
                prior_only.append(ttype)
                continue
            a, b = post
            p0 = self._prior(ttype)
            seen = (a + b) - (p0[0] + p0[1])
            mean = a / (a + b)
            delta = mean - score / 100.0
            verdict = ("meta confirmed" if abs(delta) < 0.08 else
                       "outperforming meta" if delta > 0 else
                       "underperforming meta here")
            moved.append(f"{ttype:<11}{score:>5.0f}{seen:>6.1f}"
                         f"{mean * 100:>8.0f}%  {verdict}")
        lines += moved or ["  (no episodes yet -- pure meta priors)"]
        if prior_only:
            lines.append(f"never tried yet: {', '.join(prior_only)}")
        el = self.elites(top=3)
        if el:
            lines.append("")
            lines.append("best layouts so far:")
            for rw, row in el:
                kinds = ", ".join(
                    f"{t['tower']}{t.get('path', [0, 0, 0])}"
                    for t in row.get("towers", []))
                lines.append(f"  r{row.get('final_round')} "
                             f"({row.get('outcome')}, reward {rw:.2f}): "
                             f"{kinds}")
        deaths = [r.get("final_round") for r in self.history
                  if r.get("outcome") == "defeat"
                  and r.get("final_round")]
        if deaths:
            lines.append("")
            hist = {}
            for d in deaths:
                hist[5 * (d // 5)] = hist.get(5 * (d // 5), 0) + 1
            worst = max(hist, key=hist.get)
            near = [t for t in self.threats
                    if any(abs(r - worst) <= 4 for r in t["rounds"])]
            lines.append("defeats by round bucket: " + "  ".join(
                f"r{k}-{k + 4}:{'#' * v}" for k, v in sorted(hist.items())))
            if near:
                lines.append(f"deaths cluster near r{worst} -- likely "
                             f"threat: {near[0]['threat']} "
                             f"(answers: {near[0]['answers']})")
        return "\n".join(lines)


# ------------------------------------------------------------ CLI / test

def _fake_pools(rng, n=200):
    pts = [[round(rng.uniform(0.05, 0.95), 3),
            round(rng.uniform(0.05, 0.95), 3)] for _ in range(n)]
    return {"near": pts[: n // 2], "mid": pts[n // 2: 3 * n // 4],
            "all": pts, "roomy": pts[:: 2]}


def _selftest():
    rng = random.Random(7)
    k = _load_json(KNOWLEDGE_PATH)
    brain = MetaBrain("selftest_map", "easy", target_round=40,
                      explore=0.3, knowledge=k, runs_path="/nonexistent")
    pools = _fake_pools(rng)
    pool = ["dart", "tack", "bomb", "sniper", "ninja", "wizard", "druid",
            "alchemist", "glue", "ice"] + META_EXTRA_TOWERS

    seen_types = set()
    for i in range(200):
        g = brain.next_genome(rng, 4, pools, tower_pool=pool,
                              large_towers={"super", "village"})
        places = [x for x in g if x["do"] == "place"]
        ups = [x for x in g if x["do"] == "upgrade"]
        assert places, "genome must place towers"
        refs = {p["ref"] for p in places}
        assert refs == set(range(len(places))), f"bad refs {refs}"
        for u in ups:
            assert u["ref"] in refs, "upgrade points at a missing tower"
            assert sum(u["path"]) == 1 and len(u["path"]) == 3
        for p in places:
            assert p["tower"] in pool
            assert 0 <= p["at"][0] <= 1 and 0 <= p["at"][1] <= 1
        # two-path rule: count tiers per (ref, path)
        tiers = {}
        for u in ups:
            key = (u["ref"], u["path"].index(1))
            tiers[key] = tiers.get(key, 0) + 1
        by_ref = {}
        for (ref, p_i), t in tiers.items():
            by_ref.setdefault(ref, []).append((p_i, t))
        for ref, paths in by_ref.items():
            assert len(paths) <= 2, f"3 paths opened on ref {ref}"
            assert sum(1 for _p, t in paths if t > 2) <= 1, \
                f"two paths past tier 2 on ref {ref}"
        seen_types |= {p["tower"] for p in places}
    assert len(seen_types) >= len(pool) - 2, \
        f"exploration starved: only saw {sorted(seen_types)}"

    # Feed synthetic history: tack layouts thrive, dart layouts die early.
    for i in range(30):
        brain.observe({"mode": "farm", "map": "selftest_map",
                       "difficulty": "easy", "outcome": "survived",
                       "final_round": 40,
                       "towers": [{"tower": "tack", "at": [0.3, 0.3],
                                   "path": [0, 0, 4]},
                                  {"tower": "alchemist", "at": [0.32, 0.3],
                                   "path": [3, 0, 0]}]}, quiet=True)
        brain.observe({"mode": "farm", "map": "selftest_map",
                       "difficulty": "easy", "outcome": "defeat",
                       "final_round": 6,
                       "towers": [{"tower": "dart", "at": [0.8, 0.8],
                                   "path": [1, 0, 0]}]}, quiet=True)
    ta, tb = brain.t_post["tack"]
    da, db = brain.t_post["dart"]
    assert ta / (ta + tb) > 0.85, "tack posterior should be high"
    assert da / (da + db) < 0.35, "dart posterior should have dropped"

    # Evolution engages once elites exist, produces valid genomes, and
    # never mutates the history rows it breeds from (the elite pool must
    # stay exactly what was actually played and logged).
    frozen = json.dumps([r["towers"] for _rw, r in brain.elites()],
                        sort_keys=True)
    kinds = set()
    for i in range(60):
        g = brain.next_genome(rng, 4, pools, tower_pool=pool)
        kinds.add(brain.last_strategy["kind"].split("(")[0])
        assert any(x["do"] == "place" for x in g)
    assert "evolve" in kinds or "crossover" in kinds, \
        f"evolution never engaged: {kinds}"
    assert json.dumps([r["towers"] for _rw, r in brain.elites()],
                      sort_keys=True) == frozen, \
        "evolution mutated the history rows it bred from"

    # Position learning: successful bucket beats the dead one.
    good = brain.pos_post.get(_bucket([0.3, 0.3]))
    bad = brain.pos_post.get(_bucket([0.8, 0.8]))
    assert good and bad and good[0] / sum(good) > bad[0] / sum(bad)

    # Coverage: fresh non-explore genomes answer camo before r24.
    brain2 = MetaBrain("selftest_map", "easy", target_round=40,
                       explore=0.0, knowledge=k, runs_path="/nonexistent")
    sol = k["solutions"]["camo"]
    misses = 0
    for i in range(50):
        g = brain2.next_genome(rng, 4, pools, tower_pool=pool)
        types = {}
        for x in g:
            if x["do"] == "place":
                types[x["ref"]] = x["tower"]
        ok = False
        tiers = {}
        for x in g:
            if x["do"] == "upgrade":
                key = (x["ref"], x["path"].index(1))
                tiers[key] = tiers.get(key, 0) + 1
        for ref, ttype in types.items():
            req = sol.get(ttype, "absent")
            if req is None:
                ok = True
            elif req != "absent" and tiers.get((ref, req[0]), 0) >= req[1]:
                ok = True
        misses += 0 if ok else 1
    assert misses == 0, f"camo uncovered in {misses}/50 exploit genomes"

    # ----- track-aware placement, against the real mask when present
    mask_path = Path(__file__).parent / "masks" / "monkey_meadow_dart.json"
    if mask_path.exists():
        mask = _load_json(mask_path)
        track = TrackModel(mask)
        assert track.ok and len(track.cells) >= 20, "no track blob found"
        ps = sorted(track.progress.values())
        assert ps[0] == 0.0 and ps[-1] == 1.0
        pts = mask.get("valid_strict") or mask["valid"]
        exps = sorted(track.exposure(p, 0.06) for p in pts)
        assert exps[-1] > 0 and exps[-1] >= 3 * max(exps[0], 0.005), \
            "exposure should separate prime spots from corners"
        entry = min(track.cells, key=lambda c: track.progress[c])
        track.orient([entry[0], entry[1]])
        assert track.oriented and track.progress[entry] == 0.0
        tpools = {"near": pts, "mid": [], "all": pts, "roomy": pts}
        b3 = MetaBrain("selftest_map", "easy", target_round=40,
                       explore=0.0, knowledge=k, runs_path="/nonexistent")
        med_exp = exps[len(exps) // 2]
        stats = {"carry": 0, "carry_prime": 0,
                 "buddy": 0, "buddy_near": 0,
                 "glue": 0, "glue_upstream": 0}
        carries = set(k["roles"]["carry"])
        min_gap, triples = 9.9, 0
        for i in range(60):
            g = b3.next_genome(rng, 4, tpools, tower_pool=pool,
                               track=track)
            here = {x["ref"]: (x["tower"], x["at"]) for x in g
                    if x["do"] == "place"}
            pts_here = [at for _t, at in here.values()]
            for ai in range(len(pts_here)):
                for bi in range(ai + 1, len(pts_here)):
                    min_gap = min(min_gap,
                                  _dist(pts_here[ai], pts_here[bi]))
            counts = {}
            for t, _at in here.values():
                counts[t] = counts.get(t, 0) + 1
            triples += any(v >= 3 for v in counts.values())
            spots = {t: at for t, at in here.values()}
            carry = next((t for t, _ in here.values() if t in carries),
                         None)
            if carry:
                stats["carry"] += 1
                if track.exposure(spots[carry], 0.06) >= med_exp:
                    stats["carry_prime"] += 1
            for t, at in here.values():
                style = (k["towers"][t].get("placement")
                         or {}).get("style")
                if style == "buddy" and len(here) > 1:
                    stats["buddy"] += 1
                    rr = k["towers"][t]["placement"]["range"]
                    if any(_dist(at, a2) <= rr * 0.9
                           for t2, a2 in here.values() if a2 is not at):
                        stats["buddy_near"] += 1
                if t == "glue" and carry and carry != "glue":
                    gs = track.span(at, 0.082)
                    cs = track.span(
                        spots[carry],
                        (k["towers"][carry]["placement"] or {}).get(
                            "range") or 0.06)
                    if gs and cs:
                        stats["glue"] += 1
                        if gs[1] <= cs[1] + 0.06:
                            stats["glue_upstream"] += 1
        assert min_gap >= SEP - 1e-9, \
            f"towers planned {min_gap:.3f} apart -- stacked on each other"
        assert triples == 0, \
            f"{triples} exploit layouts stacked 3+ copies of one tower"
        assert stats["carry_prime"] >= 0.8 * stats["carry"], \
            f"carries not on prime real estate: {stats}"
        assert stats["buddy_near"] >= 0.8 * max(stats["buddy"], 1), \
            f"buffers placed away from teammates: {stats}"
        if stats["glue"] >= 5:
            assert stats["glue_upstream"] >= 0.8 * stats["glue"], \
                f"glue not upstream of the carry: {stats}"
        print(f"track placement OK on {mask_path.name}: "
              f"{len(track.cells)} track cells, "
              f"carry prime {stats['carry_prime']}/{stats['carry']}, "
              f"buddy near {stats['buddy_near']}/{stats['buddy']}, "
              f"glue upstream {stats['glue_upstream']}/{stats['glue']}")

    # ----- economy policy: choose_buy reservation rules
    now = 1000.0

    def q_item(**kw):
        return {"_wake": 0, **kw}
    est = lambda it: it["est"]
    q = [q_item(do="place", tower="dart", round=1, prio=0, est=200),
         q_item(do="upgrade", ref=0, path=[1, 0, 0], round=5, prio=1,
                est=300),
         q_item(do="upgrade", ref=0, path=[0, 0, 1], round=5, prio=2,
                est=250)]
    assert choose_buy(q, 1, 700, est, now) == 0, "awake head must go first"
    q[0]["_wake"] = now + 30
    assert choose_buy(q, 1, 700, est, now) == 1, \
        "rich runs unlock the next scheduled buy early"
    assert choose_buy(q, 1, 400, est, now) is None, \
        "future buys wait when the reserve isn't covered"
    assert choose_buy(q, 5, 700, est, now) == 1, \
        "surplus above the reserve may be spent"
    assert choose_buy(q, 5, 350, est, now) is None, \
        "reserved cash must not leak to lower priorities"
    assert choose_buy(q, 5, 350, est, now, emergency=True) == 1, \
        "emergency buys any affordable defense"
    assert choose_buy(q, 1, 350, est, now, emergency=True) == 1, \
        "emergency ignores the round gate"
    q2 = [q_item(do="place", tower="a", round=1, prio=0, est=2000,
                 _wake=now + 30),
          q_item(do="place", tower="b", round=1, prio=0, est=200)]
    assert choose_buy(q2, 1, 250, est, now) == 1, \
        "equal-priority siblings buy while the head saves"
    q3 = [q_item(do="place", tower="glue", round=1, prio=1, est=275,
                 gate={"ref": 0, "tier": 3}),
          q_item(do="upgrade", ref=0, path=[0, 0, 1], round=1, prio=2,
                 est=300)]
    gated = lambda it: "gate" not in it
    assert choose_buy(q3, 1, 900, est, now, gate_ok=gated) == 1, \
        "a closed gate must skip the item entirely"
    assert choose_buy(q3, 1, 900, est, now, emergency=True,
                      gate_ok=gated) == 0, \
        "emergencies bypass gates"

    # ----- buy scheduling: paced by income, threats before deadlines
    if mask_path.exists():
        b4 = MetaBrain("selftest_map", "easy", target_round=40,
                       explore=0.0, knowledge=k, runs_path="/nonexistent")
        sol = k["solutions"]["camo"]
        late_place = camo_checked = 0
        for i in range(30):
            g = b4.next_genome(rng, 4, tpools, tower_pool=pool,
                               track=track)
            assert all("round" in x and "prio" in x and "est" in x
                       for x in g)
            rounds = [x["round"] for x in g]
            assert rounds == sorted(rounds), "genome not round-ordered"
            assert rounds[-1] <= 38, "buy scheduled past the episode"
            place_rounds = {x["ref"]: x["round"] for x in g
                            if x["do"] == "place"}
            assert min(place_rounds.values()) <= 2, "no opener at start"
            if max(place_rounds.values()) >= 6:
                late_place += 1
            types = {x["ref"]: x["tower"] for x in g if x["do"] == "place"}
            best_round = None
            reached = {}
            for x in g:
                if x["do"] == "upgrade":
                    assert x["round"] >= place_rounds[x["ref"]], \
                        "upgrade scheduled before its tower"
                    key2 = (x["ref"], x["path"].index(1))
                    reached[key2] = reached.get(key2, 0) + 1
                    req = sol.get(types[x["ref"]], "none")
                    if req not in (None, "none") \
                            and key2 == (x["ref"], req[0]) \
                            and reached[key2] == req[1]:
                        r_c = x["round"]
                        best_round = min(best_round or 99, r_c)
                elif sol.get(x["tower"], "none") is None:
                    best_round = min(best_round or 99, x["round"])
            if best_round is not None:
                camo_checked += 1
                assert best_round <= 22, \
                    f"camo answer scheduled at r{best_round}"
        assert late_place >= 10, \
            "big buys should be paced, not all placed at round 1"
        assert camo_checked >= 25
        # Upgrade-first + hero: with a hero anchoring, at most three
        # bases (hero, opener, carry) may precede the carry's early
        # tiers -- "upgrade a bit before adding 3 more".
        early_ok = hero_ok = 0
        for i in range(30):
            g = b4.next_genome(rng, 4, tpools, tower_pool=pool,
                               track=track, hero=True)
            heroes = [x for x in g if x["do"] == "place"
                      and x["tower"] == "hero"]
            hero_refs = {x["ref"] for x in heroes}
            if len(heroes) == 1 and heroes[0]["round"] <= 2 \
                    and not any(x["do"] == "upgrade"
                                and x["ref"] in hero_refs for x in g):
                hero_ok += 1
            roles = b4.last_strategy.get("roles", [])
            if roles:                          # fresh (not evolved) only
                assert not any(r.startswith("opener:") for r in roles), \
                    "hero layouts must not also buy a separate opener"
                names = [x["name"] for x in g if x["do"] == "place"]
                assert len(names) == len(set(names)) and all(names), \
                    f"tower names must be unique labels: {names}"
                gates = [x for x in g if x["do"] == "place"
                         and x.get("gate")]
                assert gates, "support bases should be carry-gated"
                for x in gates:
                    assert g[[y["ref"] for y in g
                              if y["do"] == "place"].index(
                        x["gate"]["ref"])]["tower"] != "hero"
            second_up = [j for j, x in enumerate(g)
                         if x["do"] == "upgrade"][1:2]
            if second_up:
                before = sum(1 for x in g[:second_up[0]]
                             if x["do"] == "place")
                if before <= 3:
                    early_ok += 1
        assert hero_ok == 30, f"hero placement broken ({hero_ok}/30)"
        assert early_ok >= 27, \
            f"support bases jump ahead of carry tiers ({early_ok}/30)"
        print(f"economy OK: reservation policy verified, schedules paced "
              f"({late_place}/30 layouts hold a big buy past r6), camo "
              f"answered by r22 in {camo_checked}/30, hero anchored "
              f"{hero_ok}/30, upgrade-first {early_ok}/30")

    print("selftest OK: genome format, two-path rule, exploration floor,")
    print("posterior learning, evolution engagement, spot learning, and")
    print("camo coverage all verified.")
    print("\nSample report on synthetic data:\n")
    print(brain.report())


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "selftest":
        _selftest()
        return
    if len(sys.argv) >= 3 and sys.argv[1] == "report":
        brain = MetaBrain(sys.argv[2],
                          sys.argv[3] if len(sys.argv) > 3 else "easy")
        print(brain.report())
        return
    print("usage: python meta.py selftest | report <map> [difficulty]")


if __name__ == "__main__":
    main()
