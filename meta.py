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

try:
    import learner as learner_mod       # the gated ML layer (optional)
except ImportError:
    learner_mod = None

KNOWLEDGE_PATH = Path(__file__).parent / "meta_knowledge.json"
RUNS_PATH = Path(__file__).parent / "runs_log.jsonl"
# Real tower ranges measured off the in-game range circle (mk.py
# measure-ranges). When present they override the rough hardcoded ranges, so
# coverage is reasoned about with true numbers, not guesses.
MEASURED_RANGES_PATH = Path(__file__).parent / "measured_ranges.json"

# How many episodes' worth of pseudo-evidence the meta prior is worth.
# After ~PRIOR_STRENGTH real episodes featuring a tower, the bot's own
# experience outweighs the spreadsheet.
PRIOR_STRENGTH = 6.0

# Experience TRANSFERS: episodes from other maps/difficulties/modes feed
# a global posterior worth at most this many pseudo-episodes, so a new
# rung starts from everything already learned but local evidence takes
# over quickly. (Local evidence is uncapped.)
GLOBAL_CAP = 4.0
GLOBAL_PATH_CAP = 3.0

# Failure attribution (credit assignment). A defeat no longer folds one
# scalar equally onto every tower: the reward is docked for the tower
# actually responsible for the round the run died on, so the bandit
# learns WHICH piece failed, not just that the layout did. A win still
# credits everyone fully -- these only ever apply to defeats. Kept small
# and asymmetric (an in-time answer keeps full credit) so a good species
# is never demoted for a death it did not cause.
EARLY_WINDOW = 14        # a defeat within this many rounds of the start
#                          counts as an OPENER leak, not a role failure.
DPS_ZONE = 45            # a mid/late defeat at/after this round, with its
#                          threats already answered, is a raw-DPS wall.
BLAME_ABSENT = 0.25      # owed the killing threat's answer, never built it
BLAME_CARRY = 0.20       # a DPS wall (threats answered) -> the carry
BLAME_UNLINKED = 0.12    # an amplifier standing in range of nobody
LEAK_FLOOR = 0.05        # an opener that leaked early: near-zero credit,
LEAK_WEIGHT = 2.5        # ...folded with strong confidence so ONE leak is
#                          enough to steer the spot off the prime pocket.
AMPLIFIERS = frozenset({"alchemist", "village"})

# Version handshake with mk.py. A stale meta.py sitting next to a newer
# mk.py (mixed zip extractions, leftover __pycache__) once CRASHED every
# episode with a TypeError; mk.py now checks this and degrades politely.
META_API = 4

# Late-game threat solutions, injected when meta_knowledge.json predates
# them (regenerate with tools/extract_meta.py to keep the sheet as the
# source of truth). Without these a full-length run schedules nothing
# for ceramics (r63+), DDTs (r90-99) or the BAD (r100).
FALLBACK_SOLUTIONS = {
    "ceramic": {"tack": [0, 4], "bomb": [2, 4], "wizard": [2, 4],
                "druid": [2, 4], "glue": [1, 3], "ninja": [0, 4],
                "boomerang": [1, 4], "super": [0, 3], "mortar": [1, 4],
                "spike": [1, 4]},
    "ddt": {"village": [1, 3], "ninja": [1, 4], "ice": [2, 5],
            "sniper": [0, 4], "wizard": [2, 5], "spike": [1, 4]},
    "bad": {"tack": [2, 5], "super": [0, 4], "wizard": [2, 5],
            "druid": [2, 5], "boomerang": [1, 5], "spike": [1, 4],
            "dart": [1, 5], "sniper": [0, 5]},
}

# Rough base-cost rank (medium prices) used ONLY to order purchases when
# the learned price book hasn't seen a tower yet. Being off by $100 just
# reorders two buys; the greedy executor still waits for real cash.
ROUGH_COST = {
    "dart": 200, "glue": 275, "tack": 280, "boomerang": 325, "sniper": 350,
    "wizard": 400, "druid": 425, "engineer": 450, "ice": 500, "ninja": 500,
    "bomb": 525, "alchemist": 550, "hero": 600, "mortar": 750,
    "spike": 1000, "village": 1200, "super": 2500,
}

# Pure crowd-control towers: they SLOW/freeze bloons but barely pop them.
# Fine for the "control" role, but a one-life opener built on them lets the
# round walk past a single life un-killed (observed: glue-only CHIMPS openers
# dying at round 6), so they're filtered out of the one-life opener slot.
_SLOW_ONLY = {"glue", "ice"}
# The only towers that reliably HOLD round 6 on a one-life rung's starting
# wallet: a cheap base that already pops a GROUP, plus first tiers cheap
# enough to fit $650. dart/tack/boomerang qualify (and all scale -- dart ->
# Crossbow, tack -> Tack Zone, boomerang -> MOAB Press). A base-price cap is
# NOT enough on its own: wizard/druid have a cheap ~$270 base but $325 tiers,
# so they strand at 0-0-1 and leak the single life (the field failure), and a
# base sniper single-targets. So the one-life opener draws from THIS set, not
# a price threshold that the real (cheaper-than-listed) bases slip under.
_OPENER_KILLERS = ("dart", "tack", "boomerang")

# The hero isn't in the knowledge base's tower table (which hero is
# equipped is chosen in the menu, invisible to the bot), so placement
# uses this profile: short-range coverage suits Sauda -- the sheet's
# low-micro early anchor -- and is a sane default for most heroes.
HERO_PLACEMENT = {"range": 0.045, "style": "coverage"}

# Rough cost of an upgrade TIER, for scheduling and cash reserves when
# the price book hasn't learned the real number yet. Coarse on purpose:
# being $500 off shifts a buy by a round or two, nothing more.
TIER_EST = {1: 300, 2: 700, 3: 1800, 4: 4500, 5: 14000}


# CHIMPS income is pops-only: no end-of-round bonus, no farms possible, and
# every bloon MUST be popped (one leak ends the run) -- so if you are still
# alive at round r you have earned EXACTLY the cumulative pop cash for that
# round. That makes the cash-by-round curve a near-exact predictor, not a
# rough prior: cumulative(r-1) - (everything spent) is a hard lower bound on
# current cash, which the cash guard uses to reject "severely off" low OCR
# reads (see cashguard.CashFloor). Values are the standard CHIMPS round set's
# cumulative cash (start cash 650 included), per-round for 6..100, from
# topper64.co.uk/nk/btd6/income/chimps. The (5, 650) base is the wallet
# entering round 6. The learned IncomeCurve still overrides this once real
# telemetry exists (a map/hero can pop a little faster or slower).
CHIMPS_CUM_CASH = [
    (5, 650), (6, 813), (7, 995), (8, 1195), (9, 1394), (10, 1708),
    (11, 1897), (12, 2089), (13, 2371), (14, 2630), (15, 2896),
    (16, 3164), (17, 3329), (18, 3687), (19, 3947), (20, 4133),
    (21, 4484), (22, 4782), (23, 5059), (24, 5226), (25, 5561),
    (26, 5894), (27, 6556), (28, 6822), (29, 7211), (30, 7548),
    (31, 8085), (32, 8712), (33, 8917), (34, 9829), (35, 10979),
    (36, 11875), (37, 13214), (38, 14491), (39, 16250), (40, 16771),
    (41, 18952), (42, 19611), (43, 20889), (44, 22183), (45, 24605),
    (46, 25321), (47, 26958), (48, 29801), (49, 34559), (50, 37575),
    (51, 38673.5), (52, 40269), (53, 41193.5), (54, 43391), (55, 45874),
    (56, 47160.5), (57, 49019.5), (58, 51317.5), (59, 53476.5),
    (60, 54399), (61, 55631), (62, 57017.4), (63, 59843.4),
    (64, 60693.2), (65, 63764.8), (66, 64769), (67, 65792.6),
    (68, 66570.4), (69, 67961.4), (70, 70580.2), (71, 72083.2),
    (72, 73587.2), (73, 74979.8), (74, 78023.8), (75, 80691.2),
    (76, 82007.2), (77, 84547.4), (78, 89409.4), (79, 96118.4),
    (80, 97518.6), (81, 102884.6), (82, 107641.6), (83, 112390.6),
    (84, 119434.6), (85, 122060), (86, 123008.5), (87, 125635.9),
    (88, 128949.9), (89, 131120.9), (90, 131460.2), (91, 135651.2),
    (92, 140188.6), (93, 142135.2), (94, 149802.3), (95, 153520.3),
    (96, 163475.9), (97, 164893.1), (98, 174546.9), (99, 177374.8),
    (100, 178909.4),
]


def earned_by(r, mode="standard"):
    """Cumulative cash available by round r (start cash plus pop income and,
    outside CHIMPS, end-of-round bonuses; no farms). In CHIMPS this is the
    exact standard-round-set curve -- accurate enough to catch OCR misreads,
    not just pace the plan. Interpolates within the table and extrapolates
    past round 100 with the final slope (still strictly increasing)."""
    if mode == "chimps":
        pts = CHIMPS_CUM_CASH
        if r <= pts[0][0]:
            return pts[0][1]
        for (r0, c0), (r1, c1) in zip(pts, pts[1:]):
            if r <= r1:
                return c0 + (c1 - c0) * (r - r0) / (r1 - r0)
        r0, c0 = pts[-2]
        r1, c1 = pts[-1]
        return c1 + (c1 - c0) / (r1 - r0) * (r - r1)
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


def _role_of_name(name):
    """The role tag baked into a logged tower name like 'tack#1(carry)',
    or None. Lets credit assignment recover a tower's role after the fact
    without threading strategy state through the runs log."""
    if not name:
        return None
    i, j = name.rfind("("), name.rfind(")")
    return name[i + 1:j] if 0 <= i < j else None


def _covers(ttype, path, kind, solutions):
    """Does one tower (type + path tiers) answer `kind`? The single-tower
    coverage test shared by attempt_genome's repair and observe's credit
    assignment: a solver whose requirement is None answers by existing; a
    tiered requirement answers once that path reaches the required tier."""
    req = (solutions.get(kind) or {}).get(ttype, "absent")
    if req is None:
        return True
    return req != "absent" and bool(path) and path[req[0]] >= req[1]


def _buddy_linked(at, ttype, layout, k):
    """Is an amplifier at `at` within buff range of any teammate? Mirrors
    learner.features' buddy_linked_frac geometry (an alch brew that reaches
    nobody buffs nobody), so the reward can dock a dead support buy."""
    prof = (k.get("towers", {}).get(ttype, {}).get("placement")) or {}
    r = prof.get("range") or 0.06
    return any(o_at is not None and o_at is not at
               and _dist(at, o_at) <= r * 0.9
               for _o_tt, o_at, _o_path in layout)


# Minimum spacing between planned towers, in width-fractions. A tower
# footprint is ~0.013 wide-radius, so two centers need ~0.028 -- the old
# 0.015 planned towers ON TOP of each other and the executor burned a
# minute of retries discovering the game wouldn't allow it.
SEP = 0.028
SEP_LARGE = 0.045
# How close a free DPS tower should sit to the carry to share support buffs.
# ~an alchemist's range (measured 0.115): towers inside it get the village
# and alch buffs, so the free damage clusters here instead of scattering to
# its own lane spot and going unbuffed.
BUFF_CLUSTER_R = 0.115


def _spread(cands, taken, sep, pull=None):
    """Candidates at least sep away from every taken spot. If none qualify,
    fall back to a single candidate: the one NEAREST `pull` when given (a
    buffer must stay next to its carry even in a tight cluster), otherwise
    the one FARTHEST from the layout (never an overlapping one at random)."""
    free = [c for c in cands
            if all(_dist(c, t) >= sep for t in taken)]
    if free:
        return free
    if not taken or not cands:
        return cands
    if pull is not None:
        return [min(cands, key=lambda c: _dist(c, pull))]
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
                 explore=0.20, evolve=True, knowledge=None, runs_path=None,
                 mode="standard", start_round=1):
        self.map_name = map_name
        self.difficulty = difficulty
        self.mode = mode                  # "standard" | "chimps"
        self.start_round = max(int(start_round), 1)
        self.target = max(int(target_round), 1)
        self.explore = min(max(float(explore), 0.0), 1.0)
        self.evolve = evolve
        self.k = knowledge or _load_json(KNOWLEDGE_PATH)
        self.towers = self.k["towers"]
        # Measured ranges (mk.py measure-ranges) override the rough defaults so
        # coverage scoring uses each tower's TRUE range circle, not a guess.
        if MEASURED_RANGES_PATH.exists():
            try:
                measured = json.loads(MEASURED_RANGES_PATH.read_text())
            except (OSError, ValueError):
                measured = {}
            for t, r in (measured or {}).items():
                if not (isinstance(r, (int, float)) and 0.005 < r < 0.30):
                    continue
                if t in self.towers:
                    self.towers[t].setdefault("placement", {})["range"] = r
                elif t == "hero":     # the equipped hero's real reach, used by
                    HERO_PLACEMENT["range"] = r    # every hero placement
        self.roles = self.k["roles"]
        # A knowledge file from before the late-game kinds still gets
        # ceramic/ddt/bad answers -- regenerating the file overrides.
        self.solutions = dict(self.k["solutions"])
        for kind, table in FALLBACK_SOLUTIONS.items():
            self.solutions.setdefault(kind, table)
        self.threats = self.k["threats"]
        # Posteriors, kept as PURE EVIDENCE counts [sum(r), sum(1-r)]:
        # local = this exact rung (map+difficulty+mode, uncapped);
        # global = every other logged episode (capped at GLOBAL_CAP
        # pseudo-episodes) -- how a new rung inherits what the ladder
        # below it already learned. The meta prior is added at sampling
        # time.
        self.t_post = {}
        self.p_post = {}
        self.pos_post = {}
        self.g_post = {}
        self.gp_post = {}
        self.d_post = {}     # evidence from episodes where the tower
        #                      went DEEP (tier 4+): novelty's core metric
        self.history = []
        self.seed_rows = []      # near-winning rows from OTHER rungs of
        self.last_strategy = {"kind": "uniform"}   # this same map
        # The learned income curve paces buy schedules; until it has
        # telemetry it repeats the mode prior exactly.
        self._income_curve = (learner_mod.IncomeCurve(
            lambda r: earned_by(r, self.mode)) if learner_mod else None)
        self._ol = None          # OutcomeLearner, built lazily per track
        self._ol_track = None
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

    def _row_target(self, row):
        """The round a row's episode was aiming for -- its own logged
        target when present, else its rung's final round."""
        t = row.get("target_round")
        if isinstance(t, (int, float)) and t > 0:
            return float(t)
        if row.get("game_mode") == "chimps":
            return 100.0
        return float({"easy": 40, "medium": 60, "hard": 80,
                      "impoppable": 100}.get(row.get("difficulty"), 40))

    def _reward(self, row, target=None):
        """0..1 per episode: how far did the layout get toward the target
        round? Survival = 1. Unknown endings don't teach anything."""
        if row.get("outcome") in ("survived", "victory"):
            return 1.0
        if row.get("outcome") != "defeat":
            return None
        reached = row.get("final_round")
        if reached is None:
            return None
        return min(max(reached / (target or self.target), 0.0), 1.0) * 0.95

    def _is_episode(self, row):
        return (row.get("mode") in ("farm", "solve")
                and row.get("towers")
                and self._reward(row, self._row_target(row)) is not None)

    def usable(self, row):
        """Rows of THIS rung: same map, difficulty and game mode. Only
        these feed the local posteriors and the elite pool."""
        return (self._is_episode(row)
                and row.get("map") == self.map_name
                and row.get("difficulty") == self.difficulty
                and row.get("game_mode", "standard") == self.mode)

    @staticmethod
    def _fold(post, key, r, init=None, w=1.0):
        a, b = post.setdefault(key, list(init) if init else [0.0, 0.0])
        post[key] = [a + w * r, b + w * (1.0 - r)]

    def observe(self, row, quiet=False):
        """Fold one episode row (runs_log.jsonl format) into the
        posteriors. Called for every historical line at startup and after
        each live episode. Rung rows teach the local posteriors; every
        other episode teaches the capped global ones (transfer)."""
        if not self._is_episode(row):
            return
        if not self.usable(row):
            r = self._reward(row, self._row_target(row))
            for t in row["towers"]:
                ttype = t.get("tower", "").lower()
                if not ttype:
                    continue
                self._fold(self.g_post, ttype, r)
                path = t.get("path") or [0, 0, 0]
                if max(path) > 0:
                    self._fold(self.gp_post,
                               (ttype, path.index(max(path))), r)
            if row.get("map") == self.map_name and r >= 0.85:
                # A layout that (nearly) beat another rung of this very
                # map is a seed the evolution can adapt to this one.
                self.seed_rows.append((r, row))
                self.seed_rows.sort(key=lambda x: -x[0])
                del self.seed_rows[5:]
            return
        r = self._reward(row)
        credit = self._credit(row, r)
        for idx, t in enumerate(row["towers"]):
            ttype = t.get("tower", "").lower()
            if not ttype:
                continue
            r_i, w_i = credit[idx]
            self._fold(self.t_post, ttype, r_i, w=w_i)
            path = t.get("path") or [0, 0, 0]
            if max(path) > 0:
                self._fold(self.p_post, (ttype, path.index(max(path))), r_i,
                           init=[1.0, 1.0], w=w_i)
            if max(path) >= 4:
                self._fold(self.d_post, ttype, r_i, w=w_i)
            if t.get("at"):
                self._fold(self.pos_post, _bucket(t["at"]), r_i,
                           init=[1.0, 1.0], w=w_i)
        self.history.append(row)
        if self._income_curve is not None:
            self._income_curve.fit(self.history)
        self._ol = None                   # outcome model: refit next use
        if not quiet:
            n = len(self.history)
            print(f"   [brain] learned from episode "
                  f"(reward {r:.2f}, {n} total on this rung)")

    # --------------------------------------------------- credit assignment

    def _threat_kind_near(self, death, margin=4):
        """The kind of the known threat nearest a death round, or None when
        nothing is within `margin` (an early leak or a raw-DPS wall, not a
        specific role failure). Uses learner.threat_near when present so the
        two modules agree; falls back to an inline scan otherwise."""
        if death is None:
            return None
        if learner_mod is not None:
            t = learner_mod.threat_near(death, self.threats, margin=margin)
            return t.get("kind") if t else None
        near, best_d = None, margin + 1
        for t in self.threats:
            for r_ in t.get("rounds", []):
                if abs(r_ - death) < best_d:
                    near, best_d = t, abs(r_ - death)
        return near.get("kind") if near else None

    def _anchor_index(self, towers, carry_i):
        """The tower expected to hold the opening rounds -- the hero if one
        anchors, else the tower tagged 'opener', else the carry. Its early
        leak (and its spot) take the harshest blame."""
        for i, t in enumerate(towers):
            if (t.get("tower") or "").lower() == "hero":
                return i
        for i, t in enumerate(towers):
            if _role_of_name(t.get("name")) == "opener":
                return i
        return carry_i

    def _credit(self, row, r_base):
        """Per-tower (reward, weight) for the LOCAL posteriors -- failure
        attribution. A win (or an unknown ending) credits every tower with
        the base reward, exactly as before. A defeat docks the tower
        responsible for the round it died on: the missing/shallow answer to
        the nearest threat, the carry on a raw-DPS wall, an amplifier that
        buffs nobody, and -- hardest -- the opener on an early leak (its
        species AND its map spot, so the bot learns the pocket is bad). All
        others keep the base reward. Returns a list parallel to
        row['towers']."""
        towers = row.get("towers", [])
        base = [(r_base, 1.0) for _ in towers]
        if row.get("outcome") not in ("defeat",):
            return base
        death = row.get("final_round")
        if death is None:
            return base
        start = int(row.get("start_round") or self.start_round)
        kind = self._threat_kind_near(death)
        early_leak = death <= start + EARLY_WINDOW
        layout = [((t.get("tower") or "").lower(), t.get("at"),
                   t.get("path") or [0, 0, 0]) for t in towers]
        # Did the layout already answer the threat nearest the death? If so
        # that threat is NOT what killed it -- the wall was raw DPS, and the
        # blame belongs to the carry, not to a checkbox that was already
        # ticked. (This is what keeps a glue carry -- also a ceramic
        # answerer -- from reading a mid-game DPS death as "glue failed
        # ceramic" and burying the map's one good core.)
        covered_kill = bool(kind) and any(
            _covers(tt, p, kind, self.solutions) for tt, _at, p in layout)
        dps_wall = (not early_leak) and death >= DPS_ZONE \
            and (kind is None or covered_kill)
        non_hero = [(i, t) for i, t in enumerate(towers)
                    if (t.get("tower") or "").lower() != "hero"]
        carry_i = (max(non_hero,
                       key=lambda it: (max(it[1].get("path") or [0]), -it[0]))[0]
                   if non_hero else None)
        anchor_i = self._anchor_index(towers, carry_i)
        out = []
        for i, t in enumerate(towers):
            tt = (t.get("tower") or "").lower()
            is_carry = (i == carry_i)
            r_i, w_i = r_base, 1.0
            # An UNANSWERED killing threat -> the support that owed the
            # answer (a solver species that never got upgraded deep enough
            # to count). Never the carry: coverage is a support role.
            if kind and not covered_kill and not is_carry \
                    and tt in (self.solutions.get(kind) or {}):
                r_i -= BLAME_ABSENT
            # Died in the DPS zone with the threats already handled -> the
            # carry's family/depth is the problem, not a missing answer.
            if dps_wall and is_carry:
                r_i -= BLAME_CARRY
            if tt in AMPLIFIERS and t.get("at") \
                    and not _buddy_linked(t["at"], tt, layout, self.k):
                r_i -= BLAME_UNLINKED
            if early_leak and i == anchor_i:
                r_i, w_i = min(r_i, LEAK_FLOOR), LEAK_WEIGHT
            out.append((min(max(r_i, 0.0), 1.0), w_i))
        return out

    def coverage_gaps(self, towers):
        """Per-kind coverage report for a layout: which threats it answers
        and the round each must be answered by. The inspectable form of the
        role reasoning the plan is built on -- 'have MOAB damage, NO camo
        answer by r24' rather than a pile of upgrades -- surfaced by the
        `learn` report and mirrored in the outcome model's features.
        `towers` is a list of {tower, path} (a plan or a logged layout)."""
        layout = [((t.get("tower") or "").lower(),
                   t.get("path") or [0, 0, 0]) for t in towers]
        gaps = {}
        for kind in ("camo", "lead", "moab", "ceramic", "ddt", "bad"):
            first = min((r for t in self.threats if t["kind"] == kind
                         for r in t["rounds"]), default=None)
            if first is None or first > self.target:
                continue
            gaps[kind] = {
                "first_round": first,
                "deadline": max(1, first - 2),
                "covered": any(_covers(tt, p, kind, self.solutions)
                               for tt, p in layout)}
        return gaps

    def elites(self, top=8):
        """Best layouts of this rung, best-then-recent first. While the
        rung has fewer than three of its own, near-winning layouts from
        the map's other rungs seed the pool (discounted -- they beat an
        easier game)."""
        def _cov(r):
            # How many of this rung's threats the layout already answers.
            # A tiebreaker AFTER reward: among layouts that reached the same
            # round, the one that already covers camo/lead/MOAB/... is the
            # better champion -- it needs less coverage-repair when replayed,
            # and it's what the report should surface as "best".
            g = self.coverage_gaps(r.get("towers", []))
            return sum(1 for x in g.values() if x["covered"])
        rows = [(self._reward(r), _cov(r), i, r)
                for i, r in enumerate(self.history) if self.usable(r)]
        # best reward first, then most-covered, then most recent
        rows.sort(key=lambda x: (-x[0], -x[1], -x[2]))
        out = [(rw, r) for rw, _cov_n, _, r in rows[:top]]
        if len(out) < 3 and self.seed_rows:
            out += [(rw * 0.8, r) for rw, r in
                    self.seed_rows[:top - len(out)]]
        return out

    # ---------------------------------------------------------- sampling

    def _theta(self, rng, ttype):
        pa, pb = self._prior(ttype)
        ga, gb = self.g_post.get(ttype, (0.0, 0.0))
        tot = ga + gb
        if tot > GLOBAL_CAP:
            ga, gb = ga * GLOBAL_CAP / tot, gb * GLOBAL_CAP / tot
        la, lb = self.t_post.get(ttype, (0.0, 0.0))
        return _beta(rng, pa + ga + la, pb + gb + lb)

    def trials(self, ttype, deep=False):
        """How much LOCAL evidence this rung has about a tower --
        overall, or only from episodes where it went deep (tier 4+).
        The deep count is what novelty rotates on: a glue that shows up
        shallow in every layout has still never been AUDITIONED as the
        core."""
        la, lb = (self.d_post if deep else self.t_post).get(
            ttype, (0.0, 0.0))
        return la + lb

    def _pick_tower(self, rng, candidates, chosen, novelty=False,
                    deep_slot=False):
        """Thompson draw with a synergy nudge from already-chosen towers.
        With probability `explore` the draw is uniform instead — the
        never-starve guarantee. `novelty` flips the objective: when the
        campaign is PLATEAUED, the strategy family itself is what's
        wrong, so under-tried towers get a strong bonus — a coherent
        layout around something new, not more noise around the same
        core. For the deep slot (the carry) the bonus counts only DEEP
        trials, so families the meta keeps shallow still get their
        audition as the core."""
        if not candidates:
            return None
        if not novelty and rng.random() < self.explore:
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
            if novelty:
                w *= 3.0 / (1.0 + self.trials(ttype, deep=deep_slot))
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

    def _pos_mean(self, pt):
        """The learned value of a spot's region as a DETERMINISTIC mean, not
        a Thompson draw. Used to weight anchor placement: an unseen spot
        (mean 0.5) must not randomly swing the choice off the highest-coverage
        candidate, while a spot the posterior has learned is bad still scores
        low. Exploration of anchor spots is the explore branch's job, not
        this scorer's."""
        ba, bb = self.pos_post.get(_bucket(pt), [1.0, 1.0])
        return ba / (ba + bb)

    def _pos_theta(self, rng, pt):
        ba, bb = self.pos_post.get(_bucket(pt), [1.0, 1.0])
        return _beta(rng, ba, bb)

    def _spot_for(self, rng, ttype, pools, taken, placed, track,
                  large=False, anchor=False, cluster=False):
        """Style-aware placement. `placed` is [(ttype, spot), ...] already
        assigned in this layout. Scores candidates by what the tower
        actually wants -- track coverage for DPS, just-upstream coverage
        for debuffers, buff adjacency for alch/village, late track for
        spikes, remoteness for global towers -- multiplied by the learned
        per-region posterior, so experience still bends the geometry.

        `anchor` (hero/opener/carry) gates the candidate pool to the top
        exposure band before scoring: the opener has to actually hold the
        opening rounds, so it may only stand where it sees a lot of track
        -- an enforced floor, not a soft nudge. The learned posterior then
        chooses AMONG those high-coverage spots (and demotes any that
        proved leaky)."""
        if track is None or not track.ok:
            self._last_spot_src = "no-track"
            return self._pick_spot(rng, pools, taken, large=large)
        prof = self.towers.get(ttype, {}).get("placement") \
            or (HERO_PLACEMENT if ttype == "hero" else {})
        style = prof.get("style", "coverage")
        r = prof.get("range") or 0.06
        if style == "buddy":
            # A buffer exists to stand next to the carry, which sits on a
            # near/mid TRACK spot -- but a LARGE buffer (village) drawing
            # only from the `roomy` interior can never reach it (on a real
            # mask near-track spots are never roomy). Draw buddies from the
            # track rings: the large ones from (near|mid) that are ALSO
            # roomy (so the executor's roomy-snap is a no-op and they stay
            # near the carry), the small ones from near|mid|roomy (so an
            # alchemist can also reach a carry that itself sits in roomy).
            near = pools.get("near") or []
            mid = pools.get("mid") or []
            roomy = pools.get("roomy") or []
            seen, base = set(), []
            for p in near + mid + roomy:
                if tuple(p) not in seen:
                    seen.add(tuple(p))
                    base.append(p)
            base = base or pools.get("all") or []
        elif large:
            base = pools.get("roomy") or pools.get("all") or []
        else:
            base = ((pools.get("near") or []) + (pools.get("mid") or [])) \
                or pools.get("all") or []
        if not base:
            return None
        carries = set(self.roles.get("carry", []))
        carry_spot = next((s for t, s in placed if t in carries and s), None)
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
        elif style == "coverage" and not anchor and carry_spot is not None:
            # A free DPS tower clusters with the carry (shared support buffs),
            # so guarantee the sample includes spots inside the buff disc -- a
            # raw draw of 70 easily misses that small region and the tower
            # then scatters to its own lane, unbuffed (the field complaint).
            in_buff = [p for p in base
                       if _dist(p, carry_spot) <= BUFF_CLUSTER_R]
            if in_buff:
                cands = [rng.choice(in_buff)
                         for _ in range(min(40, len(in_buff)))] + cands[:30]
        elif anchor and track is not None and track.ok:
            # Anchors (the carry first, then the opener/hero) want the map's
            # highest-COVERAGE spot: a multi-lane chokepoint where the snaking
            # track passes one point several times, so one tower hits bloons on
            # several passes. A random 70-sample usually misses it, so seed the
            # genuine top-exposure spots directly. A later anchor -- the opener,
            # placed after the carry -- prefers the top spots that ALSO sit near
            # the carry, so the two damage towers land in ONE support's buff
            # disc with multi-lane coverage instead of splitting to opposite
            # ends of the map (the boomerang-in-the-corner complaint).
            ranked = sorted(base, key=lambda p: -track.exposure(p, r))
            seed = ranked[:30]
            if carry_spot is not None:
                near = [p for p in ranked[:150]
                        if _dist(p, carry_spot) <= BUFF_CLUSTER_R]
                seed = near[:20] + seed
            cands = seed + cands[:25]
        cands = _spread(cands, taken, SEP_LARGE if large else SEP,
                        pull=carry_spot if style == "buddy" else None)
        # Keep towers that shoot bloons ON the track: drop candidates that
        # see essentially NONE of it (the same near-zero-exposure test the
        # learner uses), so neither the scorer NOR exploration ever parks a
        # DPS tower or debuffer in a dead corner with no track in range.
        # Buffers (buddy) and global towers (offside) belong OFF the line,
        # so they skip this. It's a floor, not a ranking -- a well-placed
        # upstream debuffer on a modest-exposure spot still qualifies; only
        # the genuinely off-track spots are removed.
        if style in ("coverage", "upstream", "downstream") \
                and len(cands) > 3:
            on_track = [c for c in cands if track.exposure(c, r) >= 0.005]
            if len(on_track) >= 2:
                cands = on_track
        if style == "coverage" and carry_spot is not None and len(cands) > 3 \
                and (not anchor or (not cluster and self._one_life())):
            # Cluster FLOOR (not just the nudge below, which exposure
            # differences drown out): when enough on-track spots sit inside
            # the carry's buff disc, RESTRICT the damage tower to them, then
            # the scorer picks the best-coverage EARLY spot AMONG the cluster.
            # This is what actually keeps damage tight enough for one village +
            # alch to buff it all. Applies to free DPS AND the one-life opener/
            # hero (which the carry, placed first, sits at an early multi-lane
            # spot, so near-carry spots are early too). Falls back to open
            # scoring if the disc has too few track spots -- never off-track.
            clustered = [c for c in cands
                         if _dist(c, carry_spot) <= BUFF_CLUSTER_R
                         and track.exposure(c, r) >= 0.005]
            if len(clustered) >= 3:
                cands = clustered
        roomy_near = None
        one_life_anchor = anchor and not cluster and self._one_life()
        if anchor and len(cands) > 2 and not one_life_anchor:
            # Exposure floor: keep only the top ~40% by track coverage, so an
            # anchor (hero/opener/carry) always sees a lot of track.
            ranked = sorted(cands, key=lambda c: -track.exposure(c, r))
            cands = ranked[:max(2, int(len(ranked) * 0.4))]
        elif one_life_anchor and len(cands) > 4:
            # A one-life survival anchor (opener/hero) must not be HARD-gated
            # to the max-coverage spot, because on this map those sit in the
            # back half of the track -- placed there the opener has almost no
            # lane left to catch a leak and one bloon ends the run (measured:
            # 54% of openers landed past 0.6, a quarter near the exit). Keep a
            # generous on-track pool (top ~70% by coverage) and let the strong
            # kill-early bias below choose an EARLY spot among them, so the
            # whole track works for the tower that has to hold the single life.
            ranked = sorted(cands, key=lambda c: -track.exposure(c, r))
            cands = ranked[:max(4, int(len(ranked) * 0.7))]
        if cluster:
            # ONLY the carry gets nudged toward a spot a support tower can
            # cluster on -- a roomy, village-placeable spot within buff range.
            # The hero and opener are PURE coverage: never pull them off the
            # best lane spot toward open ground (that once parked the hero in
            # a low-coverage corner).
            roomy = pools.get("roomy") or []
            if roomy:
                def roomy_near(c, _r=roomy):
                    return any(_dist(c, rp) <= 0.066 for rp in _r)
        if cands and rng.random() < self.explore \
                and not (anchor and self._one_life() and not cluster):
            # Explore, but only AMONG the on-track candidates above. The old
            # branch sampled raw buildable spots here and frequently placed
            # towers with no track in range; exploration should vary WHICH
            # good spot, not whether the tower can see the track at all. A
            # buffer explores only among spots that still cover the carry, so
            # exploration never breaks the link it exists to make. On a
            # ONE-LIFE rung the survival anchors (opener + hero, i.e. anchors
            # that don't cluster) never explore their spot -- they always take
            # the highest-coverage lane spot, because a scattered opener is
            # exactly the bad lane coverage that leaks the single life. The
            # carry still explores/clusters so its support can reach it.
            self._last_spot_src = "explore"
            if style == "buddy" and carry_spot is not None:
                in_range = [c for c in cands
                            if _dist(c, carry_spot) <= r * 0.9]
                return list(rng.choice(in_range or cands))
            return list(rng.choice(cands))

        # The carry (found above) anchors the geometry for debuffers and
        # buffers -- its covered track span orients upstream/downstream and
        # the buddy overlap term.
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
                if carry_spot is not None:
                    # A buffer's whole job is to reach the carry. Every spot
                    # whose buff disc covers the carry outranks every spot
                    # that doesn't (2.0*covers dominates the 0.05*exp tie-
                    # break), then proximity + how much of the carry's kill
                    # zone the disc overlaps decide -- so it never degenerates
                    # to "wherever it sees the most track", the old bug.
                    d = _dist(c, carry_spot)
                    covers = 1.0 if d <= r * 0.9 else 0.0
                    prox = max(0.0, 1.0 - d / (r * 0.9))
                    ov = 0.0
                    if carry_span and track.oriented:
                        sp = track.span(c, r)
                        if sp:
                            ov = max(0.0, min(sp[2], carry_span[2])
                                     - max(sp[0], carry_span[0]))
                    s = 2.0 * covers + prox + 0.6 * ov + 0.3 * mates \
                        + 0.05 * exp
                else:
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
                        # the entry leaves room for error; a defense camped
                        # at the exit pops with zero margin. Bias on sp[0],
                        # the EARLIEST bloon the spot can hit -- NOT the mean
                        # (sp[1]), which would punish a prime multi-lane spot
                        # (the track snaking back past one point) just because
                        # its range also touches a later lane. A spot that
                        # engages bloons early AND covers several lanes -- the
                        # best real estate on a wrappy map -- must score high.
                        s *= 1.25 - 0.50 * sp[0]
                        # A one-life survival anchor (opener/hero) is far
                        # more exit-sensitive: placed near the end it has no
                        # lane left to catch a leak, so ONE bloon ends the
                        # run. Bias it HARD toward engaging early.
                        if one_life_anchor:
                            s *= max(0.35, 1.0 - 1.0 * sp[0])
                # A damage tower should CLUSTER with the carry, not go solo to
                # its own best lane spot: one village + alchemist can only buff
                # towers inside its range, so a tight buffed group out-damages
                # the same towers scattered and unbuffed. Reward proximity to
                # the carry within a support tower's reach. This applies to free
                # DPS AND to the one-life OPENER -- the two damage towers should
                # share the carry's buffs and multi-lane spot, not sit at
                # opposite ends of the map (the boomerang-in-the-corner
                # complaint). Exposure stays the base term, so a clustered spot
                # still has to see the track; this decides WHICH good spot.
                if (not anchor or one_life_anchor) and carry_spot is not None:
                    d = _dist(c, carry_spot)
                    if d <= BUFF_CLUSTER_R:
                        s *= 1.0 + 0.7 * (1.0 - d / BUFF_CLUSTER_R)
            # Learned-region posterior nudges the score. For most towers it
            # only swings +/-20% via a Thompson draw (a wider swing once
            # drowned out short-range towers' small exposure differences). The
            # ANCHOR uses the deterministic posterior MEAN, not a draw: among
            # the exposure-gated candidates the HIGHEST-coverage spot must win
            # on a fresh run (random draws once parked the hero on a mediocre
            # corner), while a spot the posterior has LEARNED is bad (a leaked
            # opener) still scores near zero and gets vacated.
            if anchor:
                s *= 0.15 + 0.85 * self._pos_mean(c)
                if roomy_near is not None and roomy_near(c):
                    s *= 1.25          # carry nudge toward cluster-able (it's
                    #                    already exposure-gated, so coverage
                    #                    barely moves; the hero never gets this)
            else:
                s *= 0.8 + 0.4 * self._pos_theta(rng, c)
            if s > best_s:
                best, best_s = c, s
        if best is None:
            self._last_spot_src = "fallback"
            return self._pick_spot(rng, pools, taken, large=large)
        self._last_spot_src = "style"
        return list(best)

    def _placement_order(self, picks):
        """Spot-assignment order: the carry anchors first, coverage DPS
        next, then debuffers/cleanup that position relative to it, and
        buffers last (they need to see where everyone sits) -- but FILLER
        dead last of all. Prime, high-coverage real estate is scarce; a
        free-slot dart must never claim a spot the carry or a support tower
        still needs, which the old 'coverage style ranks 1' let it do."""
        rank = {"coverage": 1, "upstream": 2, "downstream": 3,
                "offside": 4, "buddy": 5}
        carries = set(self.roles.get("carry", []))

        def key(i):
            ttype, role = picks[i]
            if role == "carry" or ttype in carries:
                return (0, i)
            if role == "free":
                return (9, i)          # filler after every claimed role
            style = (self.towers.get(ttype, {}).get("placement")
                     or {}).get("style", "coverage")
            return (rank.get(style, 1), i)
        return sorted(range(len(picks)), key=key)

    def _pick_build(self, rng, ttype, follow_template=False):
        """(main, cross) for a tower: meta build templates re-weighted by
        the learned per-path posterior; explore = any legal combo.
        follow_template skips the random-explore branch so a scaling anchor
        (the one-life opener) always builds toward a real tier-5 carry line
        -- tack -> Tack Zone, dart -> Crossbow -- instead of a throwaway
        random path."""
        if not follow_template and rng.random() < self.explore:
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
            ga, gb = self.gp_post.get((ttype, b["main"]), (0.0, 0.0))
            tot = ga + gb
            if tot > GLOBAL_PATH_CAP:
                ga = ga * GLOBAL_PATH_CAP / tot
                gb = gb * GLOBAL_PATH_CAP / tot
            w = b.get("weight", 0.5) * (0.5 + _beta(rng, pa + ga, pb + gb))
            if w > best_w:
                best, best_w = b, w
        return best["main"], best["cross"], best.get("label", "meta")

    # ------------------------------------------------- threat coverage

    # Per-kind coverage policy. cap: highest solver tier worth forcing
    # (a tier-5 by round 38 would stall the whole economy -- but by
    # round 98 it's exactly the plan). swap: whether an unanswerable
    # threat may replace a support tower with a solver. prefer_carry:
    # BAD answers are tier-5 carry lines, so the nudge lands on the
    # carry instead of avoiding it.
    KIND_POLICY = {
        "camo": {"cap": None, "swap": True, "prefer_carry": False},
        "lead": {"cap": None, "swap": True, "prefer_carry": False},
        "moab": {"cap": 4, "swap": False, "prefer_carry": False},
        "ceramic": {"cap": 4, "swap": False, "prefer_carry": False},
        "ddt": {"cap": 5, "swap": True, "prefer_carry": False},
        "bad": {"cap": 5, "swap": False, "prefer_carry": True},
    }

    @staticmethod
    def _need_tier(need, p_i):
        """Highest tier any threat requires on this path."""
        reqs = (need or {}).get(p_i)
        return max((t for t, _by in reqs), default=0) if reqs else 0

    def _coverage_fixes(self, rng, picks, tower_pool=None):
        """Make sure the layout answers the threats it will actually meet
        before target_round (Meta thesis #2: solve every property) --
        camo and lead early, MOAB prep past round 40, and for full-length
        games the late walls: ceramics (63+), DDTs (90+), the BAD (100).
        Returns {tower_index: {path_i: [(tier, threat_round), ...]}}
        requirements -- each threat keeps its OWN deadline, so a village
        that owes camo tier 2 by round 24 and MIB tier 3 by round 90
        schedules the first tiers early and only the third late -- and
        possibly swaps a tower type in `picks` to cover a hole."""
        needs = {}
        carries = set(self.roles.get("carry", []))

        def add_need(i, path_i, tier, first):
            needs.setdefault(i, {}).setdefault(path_i, []) \
                .append((tier, first))

        for kind in ("camo", "lead", "moab", "ceramic", "ddt", "bad"):
            policy = self.KIND_POLICY[kind]
            first = min((r for t in self.threats if t["kind"] == kind
                         for r in t["rounds"]), default=None)
            if first is None or first > self.target:
                continue
            solvers = self.solutions.get(kind, {})
            covered = any(t in solvers and solvers[t] is None
                          for t, _ in picks)
            if covered:
                continue

            def conflicts(i, p_i, tier):
                """A tower can push only ONE path past tier 2 -- a deep
                need stacked on a tower that already owes another path
                a deep need would silently cannibalize one of them at
                assembly time (that exact bug once trimmed MOAB Glue to
                tier 2 whenever ceramics also picked the glue)."""
                return tier > 2 and any(q != p_i
                                        and self._need_tier(
                                            needs.get(i), q) > 2
                                        for q in needs.get(i, {}))
            upgradable = [(i, solvers[t]) for i, (t, _) in enumerate(picks)
                          if t in solvers and solvers[t] is not None
                          and not conflicts(i, *solvers[t])]
            if policy["cap"] is not None:
                upgradable = [(i, s) for i, s in upgradable
                              if s[1] <= policy["cap"]]
            if policy["cap"] is not None and upgradable:
                # A capped kind is a NUDGE: prefer a tower whose answer
                # lies on its own main build path, then respect the
                # carry preference -- and if nobody fits, the carry's
                # raw DPS is the answer.
                def fit(item):
                    i, (p_i, _tier) = item
                    t = picks[i][0]
                    builds = self.towers.get(t, {}).get("builds") or []
                    main0 = builds[0]["main"] if builds else None
                    on_carry_pref = ((t in carries)
                                     != policy["prefer_carry"])
                    return (0 if main0 == p_i else 1,
                            1 if on_carry_pref else 0)
                i, (path_i, tier) = min(upgradable, key=fit)
                add_need(i, path_i, tier, first)
                continue
            if upgradable:
                i, (path_i, tier) = rng.choice(upgradable)
                add_need(i, path_i, tier, first)
                continue
            # Nobody can solve it: swap the last non-carry pick for the
            # strongest solver (skip in explore-heavy runs half the time
            # so pure exploration still exists).
            if not policy["swap"] or rng.random() < self.explore:
                continue
            solver_types = [t for t, req in solvers.items()
                            if t in self.towers
                            and (tower_pool is None or t in tower_pool)]
            if not solver_types:
                continue
            best = max(solver_types, key=lambda t: self._theta(rng, t))
            for i in range(len(picks) - 1, -1, -1):
                if picks[i][0] not in carries and picks[i][0] != "hero" \
                        and i not in needs:
                    # Only a tower NO earlier threat depends on may be
                    # replaced -- swapping out an answerer would orphan
                    # its needs onto the new species (dead buys) and
                    # could reopen the very hole an earlier kind just
                    # closed. If every support already answers
                    # something, this threat stays on the carry's raw
                    # DPS rather than trading one hole for another.
                    picks[i] = (best, picks[i][1])
                    req = solvers[best]
                    if req is not None:
                        add_need(i, req[0], req[1], first)
                    break
        return needs

    def _deadline(self, ttype, main, path_i, tier, needs_for_tower):
        """Approximate round by which an upgrade step should be owned;
        used only to ORDER the buy queue (the executor stays greedy).
        A tier that serves several stacked threats takes the EARLIEST
        deadline among the ones it satisfies -- a village owing camo t2
        (r24) and MIB t3 (r90) buys its first tiers early and only the
        third late."""
        reqs = (needs_for_tower or {}).get(path_i) or []
        cands = [first - 2 for req_tier, first in reqs if tier <= req_tier]
        if cands:
            return min(cands)
        if path_i == main:
            return 8 + 7 * tier          # carry path: t4 by ~r36
        return 14 + 8 * tier             # crosspath: trails the main

    # ------------------------------------------------------ genome build

    # How many candidate layouts the outcome model screens per episode
    # when its gate is open. Playing an episode costs ~20 minutes;
    # generating and scoring a candidate costs milliseconds.
    SCREEN_CANDIDATES = 6

    def _outcome_learner(self, track):
        """The gated outcome model for this rung, rebuilt lazily after
        new episodes. None when learner.py is absent."""
        if learner_mod is None:
            return None
        if self._ol is None or self._ol_track is not track:
            ol = learner_mod.OutcomeLearner(
                self.k, track=track, rough_cost=ROUGH_COST,
                tier_est=TIER_EST, income_of=self.income)
            ol.prepare(self.history, self._reward,
                       target_round=self.target)
            self._ol = ol
            self._ol_track = track
        return self._ol

    def _one_genome(self, rng, n_towers, pools, is_locked, large_towers,
                    tower_pool, price_of, track, hero, novelty=False):
        elites = self.elites() if self.evolve and not novelty else []
        p_evolve = min(0.5, len(elites) / 10.0) if len(elites) >= 3 else 0.0
        if p_evolve and elites[0][0] < 0.8:
            # The best layout known still dies young: breeding from it
            # is churn that crowds out the fresh sampling which finds
            # new cores. Evolution earns its share of episodes only
            # once an elite has real substance.
            p_evolve *= 0.25
        if rng.random() < p_evolve:
            genome = self._evolved_genome(rng, n_towers, pools, is_locked,
                                          large_towers, tower_pool, elites,
                                          track, hero)
            if genome:
                return genome
        return self._fresh_genome(rng, n_towers, pools, is_locked,
                                  large_towers, tower_pool, price_of,
                                  track, hero, novelty=novelty)

    def next_genome(self, rng, n_towers, pools, is_locked=None,
                    large_towers=frozenset(), tower_pool=None,
                    price_of=None, track=None, hero=False,
                    novelty=False):
        """Produce a buy list in run_episode's format. Rolls between a
        fresh meta-templated layout and an evolution of an elite one.
        `track` (a TrackModel) turns on geometry-aware placement; `hero`
        adds the equipped hero as an early anchor placement. `novelty`
        (the campaign's plateau signal) forces a FRESH layout built
        around the least-tried tower families -- when everything known
        keeps dying the same way, the answer is a different core, not
        more noise around the old one.

        When the outcome model's cross-validation gate is OPEN (it has
        proven it can rank layouts on this rung's own data), several
        candidates are generated and the best-scoring one plays --
        model-guided search. An `explore` fraction of episodes bypasses
        the screen entirely so the model can never starve the very
        exploration that trains it (novelty episodes bypass it too --
        the model can only endorse what looks like the past), and every
        genome records whether the model touched it, so `learn` can
        show whether screening is actually winning more."""
        is_locked = is_locked or (lambda *a: False)
        args = (rng, n_towers, pools, is_locked, large_towers,
                tower_pool, price_of, track, hero)
        ol = self._outcome_learner(track)
        gate = ol.gate() if ol is not None else {"open": False}
        if novelty or not gate.get("open") \
                or rng.random() < self.explore:
            genome = self._one_genome(*args, novelty=novelty)
            if gate.get("open"):
                self.last_strategy["model"] = {
                    "used": False, "auc": round(gate["auc"], 3),
                    "why": "novelty" if novelty else "exploration bypass"}
            return genome
        cands = []
        for _ in range(self.SCREEN_CANDIDATES):
            g = self._one_genome(*args)
            cands.append((ol.score_genome(g), g,
                          dict(self.last_strategy)))
        cands.sort(key=lambda c: -c[0])
        score, genome, strat = cands[0]
        strat["model"] = {"used": True, "auc": round(gate["auc"], 3),
                          "picked": round(score, 3),
                          "n_cands": len(cands),
                          "spread": round(score - cands[-1][0], 3)}
        self.last_strategy = strat
        return genome

    def _one_life(self):
        """CHIMPS and Impoppable give a single life, so ONE leaked bloon ends
        the run: surviving the opening rounds outranks economy there."""
        return self.mode == "chimps" or self.difficulty == "impoppable"

    def _role_slots(self, n, hero=False):
        """Roles for n tower slots. On forgiving rungs, with a hero anchoring
        the hero IS the opener -- a separate cheap opener would just split
        cash away from the carry's first tiers (the second-boomerang trap).
        But on a ONE-LIFE rung a hero covers only ~3% of track by its own
        small range and, if it also drains the whole starting budget, the
        run leaks rounds 6-9 it can never take back -- so a cheap popping
        defender still leads even with a hero (the carry saves up behind it)."""
        if n <= 0:
            return []
        if hero:
            if self._one_life():
                slots = ["opener", "carry", "amplifier", "control", "free"]
            else:
                slots = ["carry", "amplifier", "control", "free"]
            return slots[:n] + ["free"] * max(0, n - 4)
        slots = ["opener", "carry", "amplifier", "control"]
        if n == 1:
            return ["carry"]
        return slots[:n] + ["free"] * max(0, n - 4)

    def _fresh_genome(self, rng, n_towers, pools, is_locked,
                      large_towers, tower_pool, price_of, track=None,
                      hero=False, novelty=False):
        pool = list(tower_pool or
                    [t for t in self.towers if t in ROUGH_COST])
        picks = []       # [(ttype, role), ...]
        if hero:
            picks.append(("hero", "hero"))   # free scaling: always early
        for role in self._role_slots(n_towers, hero=hero):
            # Novelty opens the CARRY slot to every family: the map may
            # want a core the role template would never audition (the
            # meta keeps glue shallow -- this map might not).
            if role == "free" or (novelty and role == "carry"):
                cands = pool
            else:
                cands = [t for t in self.roles.get(role, [])
                         if t in pool] or pool
                if role == "opener" and self._one_life():
                    # A one-life opener should be a CHEAP tower that also
                    # SCALES into a tier-5 carry (tack -> Tack Zone, dart ->
                    # Crossbow, boomerang -> MOAB Press): it holds round 6 AND
                    # becomes the mid-game core, so its early upgrades are an
                    # investment, not throwaway. Draw from cheap, build-
                    # templated KILLERS -- never pure crowd-control (glue/ice
                    # only slow), never the pricey ones ($500 ninja) that eat
                    # the whole $650 and leave no room to upgrade or add a
                    # second tower.
                    #
                    # Draw the opener from the proven round-6 holders
                    # (_OPENER_KILLERS): a cheap group-clearing base whose first
                    # tiers fit the $650 wallet. A base-price cap is unreliable
                    # here -- the real wizard/druid base ($270) slips under it
                    # but its $325 tiers strand it at 0-0-1 -- so restrict to the
                    # set outright, intersected with what's actually available.
                    scalers = [t for t in _OPENER_KILLERS
                               if t in pool
                               and self.towers.get(t, {}).get("builds")]
                    # Dart is the most reliable one-life opener: cheap enough to
                    # reach a real 0-0-2 pre-wave AND longer-ranged than tack
                    # (0.081 vs 0.059), so it actually covers the entry instead
                    # of clipping 1% of the track. It was the only opener that
                    # cleared round 6 in the field. Prefer it outright; tack/
                    # boomerang (which strand at 0-0-1) only stand in if dart is
                    # unavailable -- and they still shine as the CARRY (Tack
                    # Zone / MOAB Press), which is a separate slot.
                    if "dart" in scalers:
                        scalers = ["dart"]
                    cands = scalers or [t for t in cands
                                        if t not in _SLOW_ONLY] or cands
            ttype = self._pick_tower(rng, cands, [t for t, _ in picks],
                                     novelty=novelty,
                                     deep_slot=role == "carry")
            if ttype:
                picks.append((ttype, role))
        needs = self._coverage_fixes(rng, picks, tower_pool=pool)
        genome, meta = self._assemble(rng, picks, needs, pools, is_locked,
                                      large_towers, price_of, track)
        self.last_strategy = {
            "kind": "novelty" if novelty else "meta",
            "explore": self.explore,
            "placement": ("track" + ("+flow" if track.oriented else "")
                          if track and track.ok else "pools"),
            "roles": [f"{r}:{t}" for t, r in picks], **meta}
        return genome

    def income(self, r):
        """Cumulative cash expected by round r: the learned curve once
        telemetry exists, the mode prior until then."""
        if self._income_curve is not None:
            return self._income_curve.cumulative(r)
        return earned_by(r, self.mode)

    def _schedule(self, entries):
        """Assign every buy the round it should HAPPEN. Entries are
        walked most-important-first along the income curve (learned from
        telemetry when possible -- crucial in CHIMPS, where income is
        pops-only), so the plan never wants more money than the game can
        have produced -- that is what stops the old
        buy-everything-at-once behavior. Threat answers are pinned to
        land before their threat round even if the curve says later (the
        executor's reserve makes the cash appear in time). An upgrade
        never schedules before its tower."""
        order = sorted(range(len(entries)),
                       key=lambda i: (entries[i]["prio"],
                                      entries[i]["deadline"]))
        cum = 0.0
        place_round = {}
        for i in order:
            e = entries[i]
            cum += e.get("est") or 500
            r = self.start_round
            while r < 100 and self.income(r) * 0.85 < cum:
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
        # No-leak opener. The most important anchor(s) must be DOWN by the
        # start round -- an anchor the income curve would park at round 8
        # leaks rounds 6-7 it can never recover.
        places = [e for e in entries if e["do"] == "place"]
        if places and self._one_life():
            # One life: a lone anchor -- especially a hero that drains the
            # whole budget and then covers ~3% of track -- leaks the opening
            # (observed: 205/205 CHIMPS runs died r6-9 with one tower down).
            # So fit as many CHEAP popping defenders as the starting budget
            # holds, preferring real DPS over the hero, all pinned to round 6.
            # Cash IN HAND before the start round plays -- the cumulative by
            # the END of the previous round (income(r) counts money earned
            # DURING round r from pops, which you do not have pre-wave: at r6
            # income(6)=813 but the wallet holds 650). Use the FIXED prior
            # (earned_by), NOT self.income(): the starting wallet is a game
            # constant, but the learned income curve extrapolated below its
            # data range (r < 6) is garbage and would mis-size the opener.
            budget = earned_by(max(1, self.start_round - 1), self.mode)  # ~$650
            prefer = {"opener": 0, "carry": 1, "control": 2,
                      "amplifier": 3, "hero": 4, "free": 5}

            def _anchor_key(e):
                return (prefer.get(_role_of_name(e.get("name")), 5),
                        e.get("est") or 500)

            def _pull(e):
                e["round"] = min(e["round"], self.start_round)
                e["_anchor"] = True
                place_round[e["ref"]] = e["round"]

            ordered = sorted(places, key=_anchor_key)
            primary = ordered[0] if ordered else None
            # Reserve the PRIMARY anchor's first two main-path tiers. A single
            # upgraded popper (a 0-0-2 dart/tack) holds round 6 far better than
            # two BARE towers that split the wallet and each leak -- the field
            # failure: 37/40 one-life runs died r6-9 with bare openers. The
            # executor buys these teeth pre-start (the pre-round phase has no
            # clock), so a second tower is pulled only if it fits WITHOUT
            # eating the opener's teeth.
            teeth = 0
            if primary is not None:
                tcosts = sorted(
                    (u.get("est") or 300 for u in entries
                     if u["do"] == "upgrade"
                     and u["ref"] == primary["ref"]))
                teeth = sum(tcosts[:2])
            spent = pulled = 0
            for e in ordered:
                cost = e.get("est") or 500
                room = budget - (teeth if e is not primary else 0)
                if spent + cost <= room:        # fits, teeth budget intact
                    _pull(e)
                    spent += cost
                    pulled += 1
            if pulled == 0:                     # nothing affordable fit --
                _pull(min(places,               # still never leak round 6
                          key=lambda e: e.get("est") or 500))
        elif places:
            # Forgiving rungs: the single most important anchor (hero, else
            # the opener, else the carry) leads; a few early leaks are fine
            # and a second cheap tower would only starve the carry.
            anchor = None
            for want in ("hero", "opener", "carry"):
                anchor = next((e for e in places
                               if _role_of_name(e.get("name")) == want), None)
                if anchor:
                    break
            if anchor is None:
                anchor = min(places, key=lambda e: (e["round"], e["prio"]))
            anchor["round"] = min(anchor["round"], self.start_round)
            anchor["_anchor"] = True
            place_round[anchor["ref"]] = anchor["round"]
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
        spots, spot_src = {}, {}
        taken, placed_ctx = [], []
        for i in self._placement_order(picks):
            ttype, role = picks[i]
            spot = self._spot_for(rng, ttype, pools, taken, placed_ctx,
                                  track, large=ttype in large_towers,
                                  anchor=role in ("hero", "opener", "carry"),
                                  cluster=role == "carry")
            if spot is None:
                continue
            spots[i] = spot
            spot_src[i] = getattr(self, "_last_spot_src", "?")
            taken.append(spot)
            placed_ctx.append((ttype, spot))
        placed = []      # (ref, ttype, spot, main, cross, label)
        for ref, (ttype, role) in enumerate(picks):
            if ref not in spots:
                continue
            main, cross, label = self._pick_build(
                rng, ttype,
                follow_template=(role == "opener" and self._one_life()))
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
        # The cheap opener defender is scheduled BEFORE the hero so it lands
        # first with the starting cash -- the hero (which drains ~$600 and
        # covers little track) must never pre-empt the tower that actually
        # holds round 6.
        place_plan = {"opener": (0, 1.0), "hero": (0, 2.0),
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
                src = spot_src.get(ref)
                if src == "explore":
                    cov += "  [exploration pick]"
                elif src == "fallback":
                    cov += "  [no scored spot -- random]"
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
            for p_i in need:
                want[p_i] = max(want.get(p_i, 0),
                                self._need_tier(need, p_i))
            # Two-path rule: only one path past tier 2. A threat answer
            # that itself needs tier 3+ (e.g. Signal Flare) outranks the
            # carry main; tier-2 needs survive being capped anyway.
            deep = [p for p, t in want.items() if t > 2]
            if len(deep) > 1:
                deep_need = [p for p in deep
                             if self._need_tier(need, p) > 2]
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
                    is_need = tier <= self._need_tier(need, p_i)
                    early_carry = (role == "carry" and p_i == main
                                   and tier <= 3)
                    # One-life opener teeth: a base tower can't pop round 6,
                    # so the opener's first two main-path tiers come even
                    # BEFORE the carry's -- an upgraded popper is what holds
                    # the single life while the carry saves up.
                    early_opener = (role == "opener" and self._one_life()
                                    and p_i == main and tier <= 2)
                    early_anchor = early_carry or early_opener
                    dl = self._deadline(ttype, main, p_i, tier, need)
                    if early_opener:
                        # BEFORE the hero (place deadline 2) and the carry
                        # BASE (3): the opener's teeth must accumulate in the
                        # income-paced walk before the plan sinks $600 into the
                        # hero and reserves $2500 for the carry, or its first
                        # tier lands ~round 18 and round 6 is held by a base.
                        dl = 1.0 + 0.3 * tier
                    elif early_carry:
                        # The carry's first tiers outrank every support
                        # BASE: upgrade the tower you have before buying
                        # three more. Tight noise -- these must not
                        # leapfrog the opener/carry bases themselves.
                        dl = 2.0 + 3.0 * tier
                    noise = 1.0 if early_anchor else 4.0
                    entry = {
                        "do": "upgrade", "ref": order, "path": vec,
                        "prio": 0 if (is_need or early_anchor)
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

    # ----------------------------------------------------- attempt mode

    @staticmethod
    def _merge_need(path, p_i, tier):
        """Fold a threat requirement into a path vector without breaking
        BTD6's build rules: only one path past tier 2, at most two paths
        open. The need always survives; whatever conflicts gets capped
        or closed instead."""
        path = list(path)
        path[p_i] = max(path[p_i], tier)
        if path[p_i] > 2:
            for q in range(3):
                if q != p_i and path[q] > 2:
                    path[q] = 2
        open_paths = [q for q in range(3) if path[q] > 0]
        if len(open_paths) > 2:
            drop = min((q for q in open_paths if q != p_i),
                       key=lambda q: path[q])
            path[drop] = 0
        return path

    def attempt_genome(self, rng, pools, is_locked=None,
                       large_towers=frozenset(), tower_pool=None,
                       price_of=None, track=None, hero=False):
        """Play to WIN: the best known layout replayed faithfully, plus
        two surgical changes -- full threat coverage for this rung's
        target, and a repair for whatever killed it last time (deepen
        the answer to the threat nearest its death round; reinforce the
        carry when the death matches no known threat). No exploration,
        no mutation roulette: this is the champion's serious run.
        Returns None when the rung has no elite to attempt yet."""
        is_locked = is_locked or (lambda *a: False)
        elites = self.elites(top=3)
        if not elites:
            return None
        reward, row = elites[0]
        towers = [dict(t) for t in row.get("towers", []) if t.get("at")]
        pruned = []
        for t in towers:
            if all(_dist(t["at"], u["at"]) >= SEP for u in pruned):
                pruned.append(t)
        towers = pruned
        if not towers:
            return None
        if hero and not any(t["tower"] == "hero" for t in towers):
            others = [(x["tower"], x["at"]) for x in towers]
            spot = self._spot_for(rng, "hero", pools,
                                  [s for _, s in others], others, track)
            if spot:
                towers.insert(0, {"tower": "hero", "at": spot,
                                  "path": [0, 0, 0]})
        for t in towers:
            t["path"] = list(t.get("path") or [0, 0, 0])
        originals = [(t["tower"], list(t["path"])) for t in towers]
        fixes = []

        # 1. Full threat coverage for THIS rung's target (a champion
        # seeded from an easier rung has never met a DDT).
        carries = set(self.roles.get("carry", []))
        picks = [(t["tower"],
                  "carry" if t["tower"] in carries else "free")
                 for t in towers]
        pool = list(tower_pool or
                    [t for t in self.towers if t in ROUGH_COST])
        needs = self._coverage_fixes(rng, picks, tower_pool=pool)
        for i, (ttype, _role) in enumerate(picks):
            if towers[i]["tower"] != ttype:   # coverage swapped a species
                fixes.append(f"swap:{towers[i]['tower']}->{ttype}")
                towers[i]["tower"] = ttype
                towers[i]["path"] = [0, 0, 0]
        for i, want in needs.items():
            for p_i in want:
                tier = self._need_tier(want, p_i)
                if towers[i]["path"][p_i] < tier:
                    fixes.append(f"cover:{towers[i]['tower']}"
                                 f" path{p_i + 1}->t{tier}")
                towers[i]["path"] = self._merge_need(
                    towers[i]["path"], p_i, tier)

        # 2. Hazard repair: what killed this layout last time?
        death = row.get("final_round") \
            if row.get("outcome") == "defeat" else None
        kind = self._threat_kind_near(death)
        def covers(kind_, layout):
            return any(_covers(tt, path, kind_, self.solutions)
                       for tt, path in layout)
        boosted = False
        if death is not None and kind and kind in self.solutions:
            solvers = self.solutions[kind]
            now = [(t["tower"], t["path"]) for t in towers]
            if covers(kind, originals):
                # It HAD the answer and still died: deepen it -- but
                # never at the cost of the champion's own deep path.
                # Pushing a tier-2 crosspath answer to tier 3 makes it
                # the tower's ONLY deep path and _merge_need would cap
                # the tier-4/5 main down to 2: the "repair" would gut
                # the very build being attempted. Such towers are
                # skipped; the carry reinforcement below covers them.
                for t in towers:
                    req = solvers.get(t["tower"], "absent")
                    if req is None or req == "absent":
                        continue
                    cur = t["path"][req[0]]
                    if not (cur >= req[1] and cur < 5) \
                            or is_locked(t["tower"], req[0], cur + 1):
                        continue
                    if cur + 1 > 2 and any(q != req[0] and t["path"][q] > 2
                                           for q in range(3)):
                        continue      # would cannibalize the main path
                    t["path"] = self._merge_need(t["path"], req[0],
                                                 cur + 1)
                    fixes.append(f"deepen:{t['tower']} vs {kind}")
                    boosted = True
                    break
            elif covers(kind, now):
                # It died BECAUSE the answer was missing, and the
                # coverage pass above just added it -- that IS the
                # repair; piling a deeper tier on top would only starve
                # the rest of the build.
                boosted = True
        if death is not None and not boosted:
            # No named threat (or a base-solved one): more raw DPS.
            carry = next((t for t in towers if t["tower"] in carries),
                         None)
            carry = carry or max(
                (t for t in towers if t["tower"] != "hero"),
                key=lambda t: max(t["path"]), default=None)
            if carry is not None and max(carry["path"]) < 5:
                main = carry["path"].index(max(carry["path"])) \
                    if max(carry["path"]) else \
                    self._pick_build(rng, carry["tower"])[0]
                if not is_locked(carry["tower"], main,
                                 carry["path"][main] + 1):
                    carry["path"] = self._merge_need(
                        carry["path"], main, carry["path"][main] + 1)
                    fixes.append(f"reinforce:{carry['tower']}")

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
                except TypeError:
                    known = None
                if known:
                    return known
            return TIER_EST.get(tier, 800)

        entries = []
        for i, t in enumerate(towers):    # pin needs BEFORE re-sorting
            t["_need"] = needs.get(i, {})
        towers.sort(key=lambda t: (0 if t["tower"] == "hero" else 1,
                                   base_cost(t["tower"])))
        for order, t in enumerate(towers):
            entries.append({"do": "place", "tower": t["tower"],
                            "at": list(t["at"]), "ref": order,
                            "name": f"{t['tower']}#{order}(attempt)",
                            "prio": 0, "deadline": 1.0 + order,
                            "est": base_cost(t["tower"])})
            if t["tower"] == "hero":
                continue          # heroes level up on their own: no buys
            path = t["path"]
            main = path.index(max(path)) if max(path) else 0
            need_here = t["_need"]
            for p_i, target_tier in enumerate(path):
                for tier in range(1, min(target_tier, 5) + 1):
                    if is_locked(t["tower"], p_i, tier):
                        break
                    vec = [0, 0, 0]
                    vec[p_i] = 1
                    dl = self._deadline(t["tower"], main, p_i, tier,
                                        need_here)
                    is_need = tier <= self._need_tier(need_here, p_i)
                    entry = {"do": "upgrade", "ref": order, "path": vec,
                             "prio": 0 if is_need
                             else (1 if p_i == main else 2),
                             "deadline": dl,
                             "est": tier_cost(t["tower"], p_i, tier)}
                    if is_need:
                        entry["by"] = int(dl)
                    entries.append(entry)
        genome = self._schedule(entries)
        self.last_strategy = {
            "kind": f"attempt(r{row.get('final_round')},"
                    f"{row.get('outcome')})",
            "explore": 0.0,
            "placement": ("track" + ("+flow" if track.oriented else "")
                          if track and track.ok else "pools"),
            "mutations": fixes}
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
        rung = self.difficulty if self.mode == "standard" else self.mode
        n_global = sum(a + b for a, b in self.g_post.values())
        lines = [f"MetaBrain report -- map '{self.map_name}', "
                 f"rung {rung}, target r{self.target}",
                 f"episodes learned from: {len(self.history)} on this "
                 f"rung"
                 + (f" (+{n_global:.0f} transferred from other "
                    f"maps/rungs, capped at {GLOBAL_CAP:.0f})"
                    if n_global else ""), ""]
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
            la, lb = post
            pa, pb = self._prior(ttype)
            ga, gb = self.g_post.get(ttype, (0.0, 0.0))
            tot = ga + gb
            if tot > GLOBAL_CAP:
                ga, gb = ga * GLOBAL_CAP / tot, gb * GLOBAL_CAP / tot
            seen = la + lb
            mean = (pa + ga + la) / (pa + pb + ga + gb + la + lb)
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
            # The champion's role reasoning: which threats it answers and
            # which it still leaves open before each deadline -- the "have
            # MOAB damage, no camo answer by r24" view of the plan.
            gaps = self.coverage_gaps(el[0][1].get("towers", []))
            if gaps:
                have = [k for k, g in gaps.items() if g["covered"]]
                miss = [f"{k}(by r{g['deadline']})"
                        for k, g in gaps.items() if not g["covered"]]
                lines.append(
                    "  champion role coverage: "
                    + ("have " + ", ".join(have) if have else "nothing yet")
                    + ("; MISSING " + ", ".join(miss) if miss
                       else "; all threats answered"))
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
        lines.append("")
        if learner_mod is not None:
            lines += self._outcome_learner(track).report_lines()
            if self._income_curve is not None:
                lines.append(self._income_curve.describe())
            benefit = learner_mod.model_benefit(self.history)
            if benefit:
                lines.append(
                    f"model benefit: screened episodes average "
                    f"r{benefit['screened_mean_round']:.1f} "
                    f"(n={benefit['screened_n']}) vs "
                    f"r{benefit['unscreened_mean_round']:.1f} "
                    f"unscreened (n={benefit['unscreened_n']})")
        else:
            lines.append("learner.py missing -- ML screening disabled, "
                         "playing on priors + evolution only")
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
        # Full-length games: buys must be scheduled INTO the endgame and
        # MOAB prep must land before round 40.
        b5 = MetaBrain("selftest_map", "hard", target_round=80,
                       explore=0.0, knowledge=k, runs_path="/nonexistent")
        moab_sol = b5.solutions["moab"]

        def moab_prepped(g, by_round=38):
            """A tier<=4 MOAB answer actually scheduled by `by_round`
            (whether its 'by' cap says 38 or the carry's own early
            deadline got there first)."""
            types = {x["ref"]: x["tower"] for x in g if x["do"] == "place"}
            tiers = {}
            for x in g:
                if x["do"] != "upgrade" or x["round"] > by_round:
                    continue
                key2 = (x["ref"], x["path"].index(1))
                tiers[key2] = tiers.get(key2, 0) + 1
            for (ref, p_i), t in tiers.items():
                req = moab_sol.get(types[ref], "absent")
                if req not in (None, "absent") and req[1] <= 4 \
                        and p_i == req[0] and t >= req[1]:
                    return True
            return False
        endgame = moab_prep = 0
        for i in range(20):
            g = b5.next_genome(rng, 4, tpools, tower_pool=pool,
                               track=track, hero=True)
            rounds = [x["round"] for x in g]
            assert max(rounds) <= 78, "buy scheduled past a hard game"
            endgame += any(r > 40 for r in rounds)
            moab_prep += moab_prepped(g)
        assert endgame >= 16, \
            f"target 80 but no endgame buys scheduled ({endgame}/20)"
        assert moab_prep >= 16, \
            f"MOAB answer missing before r40 ({moab_prep}/20)"
        print(f"economy OK: reservation policy verified, schedules paced "
              f"({late_place}/30 layouts hold a big buy past r6), camo "
              f"answered by r22 in {camo_checked}/30, hero anchored "
              f"{hero_ok}/30, upgrade-first {early_ok}/30, endgame "
              f"scheduling at target 80 in {endgame}/20 with MOAB prep "
              f"by r38 in {moab_prep}/20")

    # ----- build-rule merging for threat requirements
    assert MetaBrain._merge_need([3, 0, 2], 1, 3) == [0, 3, 2]
    assert MetaBrain._merge_need([0, 0, 0], 2, 4) == [0, 0, 4]
    assert MetaBrain._merge_need([4, 2, 0], 0, 5) == [5, 2, 0]
    assert MetaBrain._merge_need([2, 0, 2], 1, 2) in ([2, 2, 0],
                                                      [0, 2, 2])

    # ----- income priors: CHIMPS pays less through the early/mid game
    # (no end-of-round bonuses) and grows monotonically. Late-game
    # absolute levels legitimately exceed the standard quadratic, which
    # is a pacing heuristic, not a pop-income model.
    assert all(earned_by(r, "chimps") <= earned_by(r)
               for r in range(6, 45)), \
        "chimps income prior must undercut the standard curve early"
    # r45+ the real CHIMPS pop income (BFB/ceramic waves) overtakes the
    # standard quadratic, which is a pacing heuristic, not a pop model.
    assert earned_by(45, "chimps") > earned_by(45), \
        "chimps overtakes the standard curve once the big waves start"
    assert all(earned_by(r, "chimps") < earned_by(r + 1, "chimps")
               for r in range(6, 110)), \
        "chimps income prior must be strictly increasing"

    # ----- transfer: foreign episodes shift priors, capped, not history
    bt = MetaBrain("fresh_map", "easy", target_round=40, explore=0.0,
                   knowledge=k, runs_path="/nonexistent")
    for i in range(25):
        bt.observe({"mode": "farm", "map": "elsewhere",
                    "difficulty": "easy", "outcome": "survived",
                    "final_round": 40,
                    "towers": [{"tower": "mortar", "at": [0.5, 0.5],
                                "path": [0, 4, 0]}]}, quiet=True)
        bt.observe({"mode": "farm", "map": "elsewhere",
                    "difficulty": "easy", "outcome": "defeat",
                    "final_round": 5,
                    "towers": [{"tower": "engineer", "at": [0.5, 0.5],
                                "path": [0, 4, 0]}]}, quiet=True)
    assert not bt.history, "foreign rows must not enter local history"
    rng2 = random.Random(5)

    def mean_theta(tt, n=400):
        return sum(bt._theta(rng2, tt) for _ in range(n)) / n
    pm, pe = bt._prior("mortar"), bt._prior("engineer")
    assert mean_theta("mortar") > pm[0] / sum(pm) + 0.03, \
        "25 foreign wins should lift the mortar prior"
    assert mean_theta("engineer") < pe[0] / sum(pe) - 0.03, \
        "25 foreign deaths should sink the engineer prior"
    lift = mean_theta("mortar") - pm[0] / sum(pm)
    assert lift < 0.35, f"transfer must stay capped (lift {lift:.2f})"

    # Same-map near-wins from other rungs seed the elite pool, discounted.
    bt2 = MetaBrain("fresh_map", "hard", target_round=100, explore=0.0,
                    knowledge=k, runs_path="/nonexistent", mode="chimps",
                    start_round=6)
    bt2.observe({"mode": "farm", "map": "fresh_map",
                 "difficulty": "easy", "outcome": "survived",
                 "final_round": 40,
                 "towers": [{"tower": "tack", "at": [0.3, 0.3],
                             "path": [0, 0, 4]}]}, quiet=True)
    el = bt2.elites()
    assert el and el[0][1]["towers"][0]["tower"] == "tack" \
        and abs(el[0][0] - 0.8) < 1e-6, \
        f"seed elites must appear discounted: {el}"

    # ----- model screening: engages only through the gate, and the
    # explore fraction still bypasses it (brain has 60 separable rows)
    used = bypass = 0
    for i in range(30):
        brain.next_genome(rng, 4, pools, tower_pool=pool)
        m = brain.last_strategy.get("model")
        if m and m.get("used"):
            used += 1
            assert m["n_cands"] == MetaBrain.SCREEN_CANDIDATES
            assert m["auc"] >= 0.62
        elif m:
            bypass += 1
    assert used >= 10, f"model screening never engaged (used {used}/30)"
    assert bypass >= 1, "exploration must still bypass the screen"
    closed = MetaBrain("selftest_map", "easy", target_round=40,
                       explore=0.0, knowledge=k,
                       runs_path="/nonexistent")
    closed.next_genome(rng, 4, pools, tower_pool=pool)
    assert not (closed.last_strategy.get("model") or {}).get("used"), \
        "no data means the gate is closed and the model has no vote"

    if mask_path.exists():
        # ----- CHIMPS: full-length threat coverage under pops-only pace
        bc = MetaBrain("selftest_map", "hard", target_round=100,
                       explore=0.0, knowledge=k, runs_path="/nonexistent",
                       mode="chimps", start_round=6)
        assert bc.income(50) == earned_by(50, "chimps")

        def covered_by(g, kind, by_round, cap=5):
            sol = bc.solutions[kind]
            types = {x["ref"]: x["tower"] for x in g
                     if x["do"] == "place"}
            for x in g:
                if x["do"] == "place" and sol.get(x["tower"], "x") is None \
                        and x["round"] <= by_round:
                    return True
            tiers = {}
            for x in g:
                if x["do"] != "upgrade" or x["round"] > by_round:
                    continue
                key2 = (x["ref"], x["path"].index(1))
                tiers[key2] = tiers.get(key2, 0) + 1
            for (ref, p), t in tiers.items():
                req = sol.get(types[ref], "absent")
                if req not in (None, "absent") and req[1] <= cap \
                        and p == req[0] and t >= req[1]:
                    return True
            return False
        cstats = {"camo": 0, "lead": 0, "ceramic": 0, "ddt": 0,
                  "bad": 0, "endgame": 0}
        for i in range(20):
            g = bc.next_genome(rng, 5, tpools, tower_pool=pool,
                               track=track, hero=True)
            assert max(x["round"] for x in g) <= 98, \
                "buy scheduled past a chimps game"
            cstats["endgame"] += any(x["round"] > 60 for x in g)
            cstats["camo"] += covered_by(g, "camo", 22)
            cstats["lead"] += covered_by(g, "lead", 27)
            cstats["ceramic"] += covered_by(g, "ceramic", 61, cap=4)
            cstats["ddt"] += covered_by(g, "ddt", 88)
            cstats["bad"] += covered_by(g, "bad", 98)
        for kind in ("camo", "lead", "ceramic", "ddt", "bad"):
            assert cstats[kind] >= 19, \
                f"chimps genomes missing {kind} coverage: {cstats}"
        assert cstats["endgame"] >= 12, \
            f"chimps plans must schedule into the endgame: {cstats}"

        # ----- attempt mode: champion replayed, repaired, no roulette
        bA = MetaBrain("selftest_map", "hard", target_round=100,
                       explore=0.0, knowledge=k, runs_path="/nonexistent",
                       mode="chimps", start_round=6)
        assert bA.attempt_genome(rng, tpools, track=track, hero=True) \
            is None, "no elite yet -> no attempt"
        row = {"mode": "solve", "map": "selftest_map",
               "difficulty": "hard", "game_mode": "chimps",
               "outcome": "defeat", "final_round": 91,
               "towers": [
                   {"tower": "boomerang", "at": list(pts[10]),
                    "path": [0, 2, 4]},
                   {"tower": "alchemist", "at": list(pts[40]),
                    "path": [4, 2, 0]},
                   {"tower": "ninja", "at": list(pts[80]),
                    "path": [4, 0, 2]},
                   {"tower": "hero", "at": list(pts[120]),
                    "path": [0, 0, 0]}]}
        bA.observe(row, quiet=True)
        g = bA.attempt_genome(rng, tpools, track=track, hero=True)
        assert g is not None
        assert bA.last_strategy["kind"].startswith("attempt(r91")
        fixes = bA.last_strategy["mutations"]
        assert fixes, "a dead champion must get repaired"
        assert not any(f.startswith("deepen:") for f in fixes), \
            "missing answer newly covered -- deepening on top overspends"
        orig_spots = {tuple(t["at"]) for t in row["towers"]}
        for x in g:
            if x["do"] == "place":
                assert tuple(x["at"]) in orig_spots, \
                    "attempt must keep the champion's spots"
        assert covered_by(g, "ddt", 88), \
            "died at r91 (DDTs) -- the attempt must answer them"
        hero_refs = {x["ref"] for x in g if x["do"] == "place"
                     and x["tower"] == "hero"}
        assert hero_refs and not any(
            x["do"] == "upgrade" and x["ref"] in hero_refs for x in g)
        # A champion that HAD the answer and still died gets it deepened.
        row2 = dict(row)
        row2["towers"] = [
            {"tower": "ninja", "at": list(pts[10]), "path": [0, 4, 0]},
            {"tower": "super", "at": list(pts[40]), "path": [3, 0, 2]},
            {"tower": "hero", "at": list(pts[120]), "path": [0, 0, 0]}]
        row2["final_round"] = 95
        bA2 = MetaBrain("selftest_map", "hard", target_round=100,
                        explore=0.0, knowledge=k,
                        runs_path="/nonexistent", mode="chimps",
                        start_round=6)
        bA2.observe(row2, quiet=True)
        g2 = bA2.attempt_genome(rng, tpools, track=track, hero=True)
        assert any(f.startswith("deepen:ninja") for f in
                   bA2.last_strategy["mutations"]), \
            f"owned answer that failed must deepen: " \
            f"{bA2.last_strategy['mutations']}"

        # Deepen must NEVER gut the champion's own deep path: a tier-2
        # CROSSPATH threat answer (boomerang's lead = bottom [2,2]) on a
        # tower whose main is tier 4 must not be pushed to tier 3, which
        # would cap the tier-4 main down to tier 2. The repair falls
        # through to reinforcing the carry instead.
        bA3 = MetaBrain("selftest_map", "easy", target_round=40,
                        explore=0.0, knowledge=k,
                        runs_path="/nonexistent")
        bA3.observe({"mode": "solve", "map": "selftest_map",
                     "difficulty": "easy", "game_mode": "standard",
                     "outcome": "defeat", "final_round": 29,
                     "towers": [
                         {"tower": "boomerang", "at": list(pts[10]),
                          "path": [0, 4, 2]},
                         {"tower": "ninja", "at": list(pts[60]),
                          "path": [0, 2, 0]}]}, quiet=True)
        g3 = bA3.attempt_genome(rng, tpools, track=track)
        boom = {}
        for x in g3:
            if x["do"] == "place" and x["tower"] == "boomerang":
                boom["ref"] = x["ref"]; boom["path"] = [0, 0, 0]
        for x in g3:
            if x["do"] == "upgrade" and x["ref"] == boom.get("ref"):
                boom["path"][x["path"].index(1)] += 1
        assert boom["path"][1] >= 4, \
            f"deepen gutted the champion main: boomerang {boom['path']}"
        assert not any(f.startswith("deepen:boomerang") for f in
                       bA3.last_strategy["mutations"]), \
            f"crosspath deepen should have been skipped: " \
            f"{bA3.last_strategy['mutations']}"
        print(f"chimps/attempt OK: coverage {cstats}, attempt fixes "
              f"{fixes}")

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
