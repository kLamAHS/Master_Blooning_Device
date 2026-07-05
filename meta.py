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

# Rough base-cost rank (medium prices) used ONLY to order purchases when
# the learned price book hasn't seen a tower yet. Being off by $100 just
# reorders two buys; the greedy executor still waits for real cash.
ROUGH_COST = {
    "dart": 200, "glue": 275, "tack": 280, "boomerang": 325, "sniper": 350,
    "wizard": 400, "druid": 425, "engineer": 450, "ice": 500, "ninja": 500,
    "bomb": 525, "alchemist": 550, "mortar": 750, "spike": 1000,
    "village": 1200, "super": 2500,
}

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
        free = [c for c in cands
                if all((c[0] - t[0]) ** 2 + (c[1] - t[1]) ** 2 > 0.02 ** 2
                       for t in taken)]
        cands = free or cands
        if rng.random() < self.explore or not self.pos_post:
            return list(rng.choice(cands))
        best, best_w = None, -1.0
        for c in cands:
            ba, bb = self.pos_post.get(_bucket(c), [1.0, 1.0])
            w = _beta(rng, ba, bb)
            if w > best_w:
                best, best_w = c, w
        return list(best)

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
                if picks[i][0] not in carries:
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
                    price_of=None):
        """Produce a buy list in run_episode's format. Rolls between a
        fresh meta-templated layout and an evolution of an elite one."""
        is_locked = is_locked or (lambda *a: False)
        elites = self.elites() if self.evolve else []
        p_evolve = min(0.5, len(elites) / 10.0) if len(elites) >= 3 else 0.0
        if rng.random() < p_evolve:
            genome = self._evolved_genome(rng, n_towers, pools, is_locked,
                                          large_towers, tower_pool, elites)
            if genome:
                return genome
        return self._fresh_genome(rng, n_towers, pools, is_locked,
                                  large_towers, tower_pool, price_of)

    def _role_slots(self, n):
        base = ["opener", "carry", "amplifier", "control"]
        if n <= 0:
            return []
        if n == 1:
            return ["carry"]
        if n == 2:
            return ["opener", "carry"]
        if n == 3:
            return ["opener", "carry", "amplifier"]
        return base + ["free"] * (n - 4)

    def _fresh_genome(self, rng, n_towers, pools, is_locked,
                      large_towers, tower_pool, price_of):
        pool = list(tower_pool or
                    [t for t in self.towers if t in ROUGH_COST])
        picks = []       # [(ttype, role), ...]
        for role in self._role_slots(n_towers):
            cands = pool if role == "free" else \
                [t for t in self.roles.get(role, []) if t in pool] or pool
            ttype = self._pick_tower(rng, cands, [t for t, _ in picks])
            if ttype:
                picks.append((ttype, role))
        needs = self._coverage_fixes(rng, picks)
        genome, meta = self._assemble(rng, picks, needs, pools, is_locked,
                                      large_towers, price_of)
        self.last_strategy = {
            "kind": "meta", "explore": self.explore,
            "roles": [f"{r}:{t}" for t, r in picks], **meta}
        return genome

    def _assemble(self, rng, picks, needs, pools, is_locked,
                  large_towers, price_of=None):
        """picks + coverage requirements -> ordered place/upgrade actions.
        Places go cheapest-first (early rounds need towers NOW); upgrades
        are sorted by threat deadline with noise for variety."""
        placed = []      # (ref, ttype, spot, main, cross, label)
        taken = []
        for ref, (ttype, _role) in enumerate(picks):
            spot = self._pick_spot(rng, pools, taken,
                                   large=ttype in large_towers)
            if spot is None:
                continue
            taken.append(spot)
            main, cross, label = self._pick_build(rng, ttype)
            placed.append((ref, ttype, spot, main, cross, label))

        def cost_of(ttype):
            if price_of:
                known = price_of(ttype)
                if known:
                    return known
            return ROUGH_COST.get(ttype, 600)

        placed.sort(key=lambda p: (cost_of(p[1]), p[0]))
        genome, buys = [], []
        for order, (ref, ttype, spot, main, cross, label) in enumerate(placed):
            genome.append({"do": "place", "tower": ttype,
                           "at": [spot[0], spot[1]], "ref": order})
            need = needs.get(ref, {})
            main_target = 5 if rng.random() < 0.10 else rng.randint(3, 4)
            cross_target = rng.randint(1, 2)
            want = {main: main_target, cross: cross_target}
            for p_i, min_tier in need.items():
                want[p_i] = max(want.get(p_i, 0), min_tier)
            # Two-path rule: only one path past tier 2.
            deep = [p for p, t in want.items() if t > 2]
            if len(deep) > 1:
                keep = main if main in deep else deep[0]
                for p in deep:
                    if p != keep:
                        want[p] = 2
            if len(want) > 2:                     # at most two open paths
                keep = sorted(want, key=lambda p: -want[p])[:2]
                want = {p: want[p] for p in keep}
            tiers = {p: 0 for p in want}
            for p_i in sorted(want):
                for tier in range(1, want[p_i] + 1):
                    if is_locked(ttype, p_i, tier):
                        break
                    tiers[p_i] = tier
                    buys.append((self._deadline(ttype, main, p_i, tier,
                                                need) + rng.uniform(-4, 4),
                                 order, p_i))
        buys.sort(key=lambda b: b[0])
        for _dl, ref, p_i in buys:
            vec = [0, 0, 0]
            vec[p_i] = 1
            genome.append({"do": "upgrade", "ref": ref, "path": vec})
        meta = {"builds": [f"{t} {l}" for _r, t, _s, _m, _c, l in placed]}
        return genome, meta

    # --------------------------------------------------------- evolution

    def _evolved_genome(self, rng, n_towers, pools, is_locked,
                        large_towers, tower_pool, elites):
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
        mutations = []
        for t in towers:
            roll = rng.random()
            if roll < 0.20:                       # relocate
                spot = self._pick_spot(rng, pools,
                                       [x["at"] for x in towers if x is not t],
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
                path = t.get("path") or [0, 0, 0]
                main = path.index(max(path)) if max(path) else \
                    self._pick_build(rng, t["tower"])[0]
                path[main] = min(5, max(path[main], 0) + rng.randint(1, 2))
                t["path"] = path
                mutations.append(f"deeper:{t['tower']}")
        if len(towers) < n_towers and rng.random() < 0.5:
            ttype = self._pick_tower(rng, pool, [x["tower"] for x in towers])
            spot = self._pick_spot(rng, pools, [x["at"] for x in towers],
                                   large=ttype in large_towers)
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

        genome = []
        buys = []
        towers.sort(key=lambda t: ROUGH_COST.get(t["tower"], 600))
        for order, t in enumerate(towers):
            genome.append({"do": "place", "tower": t["tower"],
                           "at": list(t["at"]), "ref": order})
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
                    buys.append((self._deadline(t["tower"], main, p_i,
                                                tier, None)
                                 + rng.uniform(-4, 4), order, p_i))
        buys.sort(key=lambda b: b[0])
        for _dl, ref, p_i in buys:
            vec = [0, 0, 0]
            vec[p_i] = 1
            genome.append({"do": "upgrade", "ref": ref, "path": vec})
        self.last_strategy = {"kind": label, "explore": self.explore,
                              "mutations": mutations}
        return genome

    # ---------------------------------------------------------- describe

    def describe_genome(self, genome):
        places = [g for g in genome if g["do"] == "place"]
        ups = sum(1 for g in genome if g["do"] == "upgrade")
        kinds = ", ".join(g["tower"] for g in places)
        s = self.last_strategy
        extra = ""
        if s.get("mutations"):
            extra = f" [{', '.join(s['mutations'])}]" if s["mutations"] \
                else " [no mutations]"
        elif s.get("roles"):
            extra = f" [{', '.join(s['roles'])}]"
        return (f"   strategy={s.get('kind', '?')}{extra}\n"
                f"   layout: {kinds} (+{ups} upgrades)")

    def report(self):
        lines = [f"MetaBrain report -- map '{self.map_name}', "
                 f"difficulty {self.difficulty}, target r{self.target}",
                 f"episodes learned from: {len(self.history)}", ""]
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

    # Evolution engages once elites exist, and produces valid genomes.
    kinds = set()
    for i in range(60):
        g = brain.next_genome(rng, 4, pools, tower_pool=pool)
        kinds.add(brain.last_strategy["kind"].split("(")[0])
        assert any(x["do"] == "place" for x in g)
    assert "evolve" in kinds or "crossover" in kinds, \
        f"evolution never engaged: {kinds}"

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
    assert misses <= 2, f"camo uncovered in {misses}/50 exploit genomes"

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
