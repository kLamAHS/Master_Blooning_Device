"""The measurable machine-learning layer: models that must EARN their vote.

Three learners, all pure stdlib, all deliberately boring and inspectable:

1. OutcomeLearner -- a logistic model over layout features (tower mix,
   threat coverage, geometry, cost pacing) trained on the bot's own
   episodes. It is used to SCREEN candidate layouts before they are
   played -- but only while its cross-validated ranking skill beats
   chance by a clear margin (the GATE). No data, weak data, or shuffled
   noise all leave the gate closed, and the bot plays exactly as it
   would without ML. That gate is what makes the ML *genuinely*
   beneficial instead of decoratively present.

2. IncomeCurve -- an empirical cumulative-cash-by-round model per
   (difficulty, mode), learned from the cash/spend telemetry each
   episode records. It replaces the hardcoded income guess used to pace
   buy schedules once real data exists -- which matters most in CHIMPS,
   where income is pops-only and the hardcoded curve would overspend.

3. Hazard analysis -- which round kills layouts and which known threat
   that round belongs to. This is what lets the campaign layer repair a
   champion ("died at 63: ceramics -- add cleanup") instead of rolling
   dice again.

Like meta.py, this module has no screen/game dependencies:

    python learner.py selftest
"""

import json
import math
import random
from pathlib import Path

# Mirrors of meta.py's rough cost tables (learner must stay importable
# without meta.py; meta.py passes its own tables in, so these defaults
# only matter for standalone tests).
ROUGH_COST = {
    "dart": 200, "glue": 275, "tack": 280, "boomerang": 325, "sniper": 350,
    "wizard": 400, "druid": 425, "engineer": 450, "ice": 500, "ninja": 500,
    "bomb": 525, "alchemist": 550, "hero": 600, "mortar": 750,
    "spike": 1000, "village": 1200, "super": 2500,
}
TIER_EST = {1: 300, 2: 700, 3: 1800, 4: 4500, 5: 14000}

# Gate thresholds. AUC 0.5 = coin flip; 0.62 out-of-fold on real
# episodes is honest evidence of ranking skill on datasets this small.
GATE_MIN_ROWS = 12
GATE_MIN_AUC = 0.62


def towers_from_genome(genome):
    """Collapse a buy list (place/upgrade entries) into the layout the
    episode is PLANNING to reach: one row per tower with its target
    path tiers. Same shape as the 'towers' field logged after a run,
    so one feature extractor serves both training and screening."""
    by_ref = {}
    for e in genome:
        if e.get("do") == "place":
            by_ref[e["ref"]] = {"tower": e["tower"].lower(),
                                "at": list(e["at"]), "path": [0, 0, 0],
                                **({"name": e["name"]}
                                   if e.get("name") else {})}
    for e in genome:
        if e.get("do") == "upgrade" and e.get("ref") in by_ref:
            if 1 in e.get("path", []):
                by_ref[e["ref"]]["path"][e["path"].index(1)] += 1
    return list(by_ref.values())


def _sigmoid(z):
    if z >= 0:
        ez = math.exp(-min(z, 60.0))
        return 1.0 / (1.0 + ez)
    ez = math.exp(max(z, -60.0))
    return ez / (1.0 + ez)


def auc_score(scores, labels):
    """Rank AUC with tie-splitting. Returns None if one class is empty."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


class LogisticModel:
    """L2-regularized logistic regression over feature DICTS, trained by
    full-batch gradient descent with per-feature standardization. Tiny
    datasets, tiny feature counts -- nothing fancier is warranted, and
    every weight stays a number a human can read in the report."""

    def __init__(self, l2=0.03, epochs=400, lr=0.5):
        self.l2 = l2
        self.epochs = epochs
        self.lr = lr
        self.w = {}
        self.b = 0.0
        self.mean = {}
        self.std = {}

    def _vector(self, feats):
        # Iterate the MODEL's features, not the input's: a key missing
        # from `feats` means raw 0 (e.g. zero ninjas), exactly as fit()
        # treated it. Iterating the input instead silently imputed the
        # feature MEAN for absent keys -- a layout with none of a
        # beneficial tower could outrank one with some of it.
        return {k: (feats.get(k, 0.0) - self.mean[k]) / self.std[k]
                for k in self.std}

    def fit(self, feat_rows, targets):
        if not feat_rows:
            self.mean, self.std, self.w, self.b = {}, {}, {}, 0.0
            return self          # no data: predict() returns 0.5 flat
        keys = sorted({k for f in feat_rows for k in f})
        n = len(feat_rows)
        self.mean, self.std = {}, {}
        for k in keys:
            vals = [f.get(k, 0.0) for f in feat_rows]
            m = sum(vals) / n
            var = sum((v - m) ** 2 for v in vals) / n
            sd = math.sqrt(var)
            if sd < 1e-9:
                continue                    # constant feature: drop it
            self.mean[k], self.std[k] = m, sd
        self.w = {k: 0.0 for k in self.std}
        self.b = 0.0
        xs = [self._vector(f) for f in feat_rows]
        lr = self.lr
        for epoch in range(self.epochs):
            grad_w = {k: 0.0 for k in self.w}
            grad_b = 0.0
            for x, y in zip(xs, targets):
                p = _sigmoid(self.b + sum(self.w[k] * v
                                          for k, v in x.items()))
                err = p - y
                grad_b += err
                for k, v in x.items():
                    grad_w[k] += err * v
            step = lr / n
            self.b -= step * grad_b
            for k in self.w:
                self.w[k] -= step * (grad_w[k] + self.l2 * self.w[k])
        return self

    def predict(self, feats):
        x = self._vector(feats)
        return _sigmoid(self.b + sum(self.w[k] * v for k, v in x.items()))

    def top_weights(self, n=6):
        ranked = sorted(self.w.items(), key=lambda kv: -abs(kv[1]))
        return [(k, round(v, 3)) for k, v in ranked[:n] if abs(v) > 0.01]


class OutcomeLearner:
    """Features + model + gate for one rung (map, difficulty, mode).

    Usage:
        ol = OutcomeLearner(knowledge, track=track)
        ol.prepare(rows, reward_fn)      # rows: runs_log entries
        if ol.gate()["open"]:
            p = ol.score_genome(genome)  # 0..1: predicted success
    """

    def __init__(self, knowledge, track=None, rough_cost=None,
                 tier_est=None, income_of=None):
        self.k = knowledge
        self.track = track
        self.rough_cost = rough_cost or ROUGH_COST
        self.tier_est = tier_est or TIER_EST
        # income_of(target_round) -> rough total cash for cost pacing
        self.income_of = income_of or (lambda r: 650 + 25 * r + 11 * r * r)
        self.rows = []
        self.feats = []
        self.rewards = []
        self.model = None
        self._gate = None

    # ------------------------------------------------------- features

    def features(self, towers, target_round=40):
        """One flat dict per layout. Everything here is computable both
        from a plan (before playing) and from a logged row (after), so
        the model can never train on information it won't have at
        screening time."""
        f = {}
        solutions = self.k.get("solutions", {})
        carries = set(self.k.get("roles", {}).get("carry", []))
        n = 0
        cost = 0.0
        max_tier = 0
        carry_tier = 0
        for t in towers:
            ttype = (t.get("tower") or "").lower()
            if not ttype:
                continue
            n += 1
            f["n_" + ttype] = f.get("n_" + ttype, 0.0) + 1.0
            path = t.get("path") or [0, 0, 0]
            top = max(path)
            max_tier = max(max_tier, top)
            if ttype in carries:
                carry_tier = max(carry_tier, top)
            cost += self.rough_cost.get(ttype, 600)
            for p_i, tier in enumerate(path):
                cost += sum(self.tier_est.get(x, 800)
                            for x in range(1, tier + 1))
        f["n_towers"] = float(n)
        f["max_tier"] = float(max_tier)
        f["carry_tier"] = float(carry_tier)
        f["has_hero"] = 1.0 if any((t.get("tower") or "").lower() == "hero"
                                   for t in towers) else 0.0
        f["cost_ratio"] = cost / max(self.income_of(target_round), 1.0)

        for kind, table in solutions.items():
            covered = 0.0
            for t in towers:
                req = table.get((t.get("tower") or "").lower(), "absent")
                if req is None:
                    covered = 1.0
                elif req != "absent":
                    path = t.get("path") or [0, 0, 0]
                    if path[req[0]] >= req[1]:
                        covered = 1.0
            f["covers_" + kind] = covered

        # Role reasoning as scalars the model can weight: how many distinct
        # threat roles the layout answers, and how many of the EARLY ones
        # (camo/lead/moab -- the walls a CHIMPS run meets first) it still
        # leaves open. Aggregates the covers_<kind> flags above so the model
        # learns "MOAB damage but no camo answer" as a shape, not 21 tiers.
        f["role_coverage_count"] = sum(f.get("covers_" + k, 0.0)
                                       for k in solutions)
        f["uncovered_early"] = sum(
            1.0 for k in ("camo", "lead", "moab")
            if k in solutions and f.get("covers_" + k, 0.0) < 1.0)

        if self.track is not None and getattr(self.track, "ok", False):
            exposures = []
            carry_exp = 0.0
            spots = []
            for t in towers:
                ttype = (t.get("tower") or "").lower()
                at = t.get("at")
                if not at:
                    continue
                prof = (self.k.get("towers", {}).get(ttype, {})
                        .get("placement")) or {}
                r = prof.get("range") or 0.06
                exp = self.track.exposure(at, r)
                exposures.append(exp)
                spots.append((ttype, at, r))
                if ttype in carries:
                    carry_exp = max(carry_exp, exp)
            if exposures:
                f["mean_exposure"] = sum(exposures) / len(exposures)
                f["best_exposure"] = max(exposures)
                f["carry_exposure"] = carry_exp
                f["zero_exposure_frac"] = (
                    sum(1 for e in exposures if e < 0.005) / len(exposures))
            # Buffers actually reaching a teammate (alch brew that
            # reaches nobody buffs nobody).
            buddies = [(t, at, r) for t, at, r in spots
                       if t in ("alchemist", "village")]
            if buddies:
                near = 0
                for _t, at, r in buddies:
                    if any(o is not at
                           and math.hypot(at[0] - o[0],
                                          (at[1] - o[1]) * 9.0 / 16.0)
                           <= r * 0.9
                           for _t2, o, _r2 in spots):
                        near += 1
                f["buddy_linked_frac"] = near / len(buddies)
            # Is the CARRY specifically buffed? Support multiplies the DPS
            # core -- an alch on the carry is worth far more than one on a
            # filler dart -- so the model gets that link as its own signal.
            carry_spots = [at for t, at, _r in spots if t in carries]
            if carry_spots:
                f["carry_buffed"] = 1.0 if any(
                    math.hypot(bat[0] - cs[0],
                               (bat[1] - cs[1]) * 9.0 / 16.0) <= br * 0.9
                    for _bt, bat, br in buddies for cs in carry_spots) \
                    else 0.0
        return f

    # ------------------------------------------------------- training

    def prepare(self, rows, reward_fn, target_round=40):
        """Digest usable rows into (features, reward) pairs.
        reward_fn(row) -> 0..1 or None (meta.py's own reward shaping,
        injected so the two modules can never disagree about labels)."""
        self.rows, self.feats, self.rewards = [], [], []
        self.target = target_round
        for row in rows:
            rw = reward_fn(row)
            if rw is None or not row.get("towers"):
                continue
            self.rows.append(row)
            self.feats.append(self.features(row["towers"], target_round))
            self.rewards.append(rw)
        self.model = None
        self._gate = None
        return len(self.rows)

    def _labels(self):
        """Binary labels for gate evaluation. Prefer the real question
        (survived or not); when one class is missing -- early on,
        everything dies -- fall back to above/below-median progress so
        ranking skill can still be measured and used."""
        survived = [1 if r >= 0.999 else 0 for r in self.rewards]
        n_pos = sum(survived)
        if 3 <= n_pos <= len(survived) - 3:
            return survived, "survival"
        med = sorted(self.rewards)[len(self.rewards) // 2]
        labels = [1 if r > med else 0 for r in self.rewards]
        if sum(labels) < 3 or sum(labels) > len(labels) - 3:
            return None, "degenerate"
        return labels, "progress"

    def gate(self, folds=4, repeats=5, rng=None):
        """Mean out-of-fold AUC over several independent stratified
        fold shuffles. The model may screen candidates only while this
        reports open=True. A SINGLE CV split on a few dozen episodes
        has enough variance that shuffled-noise labels cleared the
        threshold roughly one time in three -- averaging over repeated
        splits is what makes the gate's promise (noise stays closed)
        actually hold."""
        if self._gate is not None:
            return self._gate
        n = len(self.rewards)
        report = {"open": False, "n": n, "auc": None, "kind": None,
                  "why": ""}
        if n < GATE_MIN_ROWS:
            report["why"] = (f"only {n} usable episodes "
                             f"(need {GATE_MIN_ROWS})")
            self._gate = report
            return report
        labels, kind = self._labels()
        report["kind"] = kind
        if labels is None:
            report["why"] = "labels degenerate (no outcome spread yet)"
            self._gate = report
            return report
        rng = rng or random.Random(1234)
        aucs = []
        for _rep in range(repeats):
            pos = [i for i, y in enumerate(labels) if y]
            neg = [i for i, y in enumerate(labels) if not y]
            rng.shuffle(pos)
            rng.shuffle(neg)
            k = max(2, min(folds, len(pos), len(neg)))
            assign = {}
            for group in (pos, neg):
                for j, i in enumerate(group):
                    assign[i] = j % k
            oof = [None] * n
            for fold in range(k):
                train_idx = [i for i in range(n) if assign[i] != fold]
                test_idx = [i for i in range(n) if assign[i] == fold]
                m = LogisticModel().fit(
                    [self.feats[i] for i in train_idx],
                    [self.rewards[i] for i in train_idx])
                for i in test_idx:
                    oof[i] = m.predict(self.feats[i])
            a = auc_score(oof, labels)
            if a is not None:
                aucs.append(a)
        auc = sum(aucs) / len(aucs) if aucs else None
        report["auc"] = auc
        if auc is not None and auc >= GATE_MIN_AUC:
            report["open"] = True
            report["why"] = (f"mean out-of-fold AUC {auc:.2f} over "
                             f"{len(aucs)} CV repeats on {n} episodes "
                             f"({kind} labels)")
        else:
            report["why"] = (f"mean out-of-fold AUC {auc:.2f} < "
                             f"{GATE_MIN_AUC} -- no proven skill yet"
                             if auc is not None else "AUC undefined")
        self._gate = report
        return report

    def train_full(self):
        if self.model is None:
            self.model = LogisticModel().fit(self.feats, self.rewards)
        return self.model

    def score_genome(self, genome):
        return self.score_towers(towers_from_genome(genome))

    def score_towers(self, towers):
        if self.model is None:
            self.train_full()
        return self.model.predict(self.features(towers, self.target))

    def report_lines(self):
        g = self.gate()
        lines = [f"outcome model: {'OPEN' if g['open'] else 'closed'} -- "
                 f"{g['why']}"]
        if g["open"]:
            self.train_full()
            tops = self.model.top_weights()
            if tops:
                lines.append("   strongest signals: "
                             + ", ".join(f"{k} {v:+.2f}" for k, v in tops))
        return lines


# ---------------------------------------------------------------- income

class IncomeCurve:
    """Cumulative cash available by round, learned from telemetry.

    Every episode logs cash_by_round (cash at the start of each round)
    and spent_by_round (sum of purchases made during each round). Cash
    plus everything spent so far IS cumulative income -- no game
    formulas needed, and farms/difficulty/mode quirks are captured for
    free. Falls back to the injected prior curve wherever data is
    missing; needs a few episodes before it speaks at all."""

    MIN_ROWS = 3
    MIN_ROUNDS = 8

    def __init__(self, prior_fn):
        self.prior = prior_fn
        self.points = {}          # round -> cumulative cash (learned)
        self.n_rows = 0

    def fit(self, rows):
        samples = {}              # round -> [cumulative observations]
        used = 0
        for row in rows:
            cash = row.get("cash_by_round") or {}
            spent = row.get("spent_by_round") or {}
            if len(cash) < 3:
                continue
            used += 1
            items = sorted((int(r), c) for r, c in cash.items()
                           if isinstance(c, (int, float)))
            for r, c in items:
                spent_before = sum(v for rr, v in spent.items()
                                   if int(rr) < r)
                samples.setdefault(r, []).append(c + spent_before)
        self.n_rows = used
        pts = {}
        for r, vals in samples.items():
            vals.sort()
            pts[r] = vals[len(vals) // 2]
        # Monotonize: cumulative income can only grow.
        best = 0.0
        for r in sorted(pts):
            best = max(best, pts[r])
            pts[r] = best
        self.points = pts
        return self

    @property
    def fitted(self):
        return (self.n_rows >= self.MIN_ROWS
                and len(self.points) >= self.MIN_ROUNDS)

    def cumulative(self, r):
        """Cumulative cash available by round r."""
        if not self.fitted:
            return self.prior(r)
        ks = sorted(self.points)
        if r <= ks[0]:
            # Scale the prior to agree with the first learned point.
            scale = self.points[ks[0]] / max(self.prior(ks[0]), 1.0)
            return self.prior(r) * scale
        if r >= ks[-1]:
            scale = self.points[ks[-1]] / max(self.prior(ks[-1]), 1.0)
            return self.prior(r) * scale
        lo = max(k for k in ks if k <= r)
        hi = min(k for k in ks if k >= r)
        if lo == hi:
            return self.points[lo]
        frac = (r - lo) / (hi - lo)
        return self.points[lo] + frac * (self.points[hi] - self.points[lo])

    def describe(self):
        if not self.fitted:
            return (f"income curve: prior only ({self.n_rows} episodes "
                    f"with telemetry, need {self.MIN_ROWS})")
        ks = sorted(self.points)
        mid = ks[len(ks) // 2]
        return (f"income curve: learned from {self.n_rows} episodes, "
                f"rounds {ks[0]}-{ks[-1]} (e.g. r{mid}: "
                f"${self.points[mid]:,.0f} vs prior "
                f"${self.prior(mid):,.0f})")


# ---------------------------------------------------------------- hazard

def death_rounds(rows):
    return [r.get("final_round") for r in rows
            if r.get("outcome") == "defeat"
            and isinstance(r.get("final_round"), int)]


def threat_near(round_no, threats, margin=4):
    """The known threat closest to a death round, or None."""
    best, best_d = None, margin + 1
    for t in threats or []:
        for r in t.get("rounds", []):
            d = abs(r - round_no)
            if d < best_d:
                best, best_d = t, d
    return best


def hazard_report(rows, threats):
    deaths = death_rounds(rows)
    if not deaths:
        return ["hazard: no defeats recorded yet"]
    hist = {}
    for d in deaths:
        hist[5 * (d // 5)] = hist.get(5 * (d // 5), 0) + 1
    lines = ["deaths by round bucket: "
             + "  ".join(f"r{k}-{k + 4}:{'#' * v}"
                         for k, v in sorted(hist.items()))]
    worst = max(hist, key=hist.get)
    t = threat_near(worst + 2, threats)
    if t:
        lines.append(f"deaths cluster near r{worst} -- likely threat: "
                     f"{t['threat']} (answers: {t['answers']})")
    return lines


def model_benefit(rows):
    """Compare episodes the model screened against those it didn't --
    the honest scoreboard for 'is the ML actually helping HERE'. Only
    counts finished episodes; returns None until both arms exist."""
    def arm(row):
        s = row.get("strategy") or {}
        m = s.get("model") or {}
        return bool(m.get("used"))
    finished = [r for r in rows
                if r.get("outcome") in ("survived", "victory", "defeat")
                and isinstance(r.get("final_round"), int)]
    with_m = [r["final_round"] for r in finished if arm(r)]
    without = [r["final_round"] for r in finished if not arm(r)]
    if len(with_m) < 3 or len(without) < 3:
        return None
    return {"screened_n": len(with_m),
            "screened_mean_round": sum(with_m) / len(with_m),
            "unscreened_n": len(without),
            "unscreened_mean_round": sum(without) / len(without)}


# ---------------------------------------------------------------- selftest

def _selftest():
    rng = random.Random(11)
    knowledge = {
        "roles": {"carry": ["tack", "super"]},
        "solutions": {
            "camo": {"ninja": None, "wizard": [2, 2]},
            "lead": {"bomb": None},
        },
        "towers": {},
    }

    def make_row(has_ninja, seed):
        r = random.Random(seed)
        towers = [{"tower": "tack", "at": [0.4, 0.4],
                   "path": [0, 0, r.randint(2, 4)]},
                  {"tower": "bomb", "at": [0.5, 0.5],
                   "path": [0, r.randint(1, 2), 0]}]
        if has_ninja:
            towers.append({"tower": "ninja", "at": [0.45, 0.45],
                           "path": [r.randint(0, 2), 0, 0]})
        else:
            towers.append({"tower": "dart", "at": [0.45, 0.45],
                           "path": [r.randint(0, 2), 0, 0]})
        survived = has_ninja and r.random() < 0.9
        final = 40 if survived else (24 if not has_ninja
                                     else r.randint(30, 39))
        return {"mode": "farm", "map": "m", "outcome":
                "survived" if survived else "defeat",
                "final_round": final, "towers": towers}

    rows = [make_row(i % 2 == 0, i) for i in range(40)]

    def reward_fn(row):
        if row.get("outcome") == "survived":
            return 1.0
        if row.get("outcome") != "defeat":
            return None
        return min(max((row.get("final_round") or 0) / 40.0, 0.0),
                   1.0) * 0.95

    ol = OutcomeLearner(knowledge)
    n = ol.prepare(rows, reward_fn, target_round=40)
    assert n == 40, f"prepared {n} rows"
    g = ol.gate()
    assert g["open"], f"gate should open on separable data: {g}"
    assert g["auc"] > 0.7, f"AUC too weak on separable data: {g}"

    # The model must rank the camo-covered layout above the naked one.
    good = [{"tower": "tack", "at": [0.4, 0.4], "path": [0, 0, 4]},
            {"tower": "ninja", "at": [0.45, 0.45], "path": [1, 0, 0]},
            {"tower": "bomb", "at": [0.5, 0.5], "path": [0, 2, 0]}]
    bad = [{"tower": "tack", "at": [0.4, 0.4], "path": [0, 0, 4]},
           {"tower": "dart", "at": [0.45, 0.45], "path": [1, 0, 0]},
           {"tower": "bomb", "at": [0.5, 0.5], "path": [0, 2, 0]}]
    assert ol.score_towers(good) > ol.score_towers(bad), \
        "model failed to prefer camo coverage"

    # Shuffled labels: the gate must CLOSE, and not by luck -- three
    # independent shuffles, every one strictly below the threshold.
    # (A single 4-fold split used to clear 0.62 on pure noise about
    # one time in three; the repeated-CV mean is what fixed it.)
    for shuffle_seed in (11, 12, 13):
        srng = random.Random(shuffle_seed)
        shuffled = [dict(r) for r in rows]
        outcomes = [(r["outcome"], r["final_round"]) for r in shuffled]
        srng.shuffle(outcomes)
        for r, (o, fr) in zip(shuffled, outcomes):
            r["outcome"], r["final_round"] = o, fr
        ol2 = OutcomeLearner(knowledge)
        ol2.prepare(shuffled, reward_fn, target_round=40)
        g2 = ol2.gate()
        assert not g2["open"] and g2["auc"] < GATE_MIN_AUC, \
            f"gate opened on shuffled noise (seed {shuffle_seed}): {g2}"

    # Missing feature keys mean raw ZERO, not the feature mean: a
    # layout with none of a beneficial tower must score below one with
    # enough of it (this was a real non-monotone-encoding bug).
    mono_rows = []
    for i in range(24):
        cnt = i % 4
        mono_rows.append({
            "outcome": "survived" if cnt >= 2 else "defeat",
            "final_round": 40 if cnt >= 2 else 20,
            "towers": [{"tower": "dart", "at": [0.4, 0.4],
                        "path": [1, 0, 0]}] * max(cnt, 1)
            if cnt else [{"tower": "bomb", "at": [0.4, 0.4],
                          "path": [1, 0, 0]}]})
    ol_m = OutcomeLearner(knowledge)
    ol_m.prepare(mono_rows, reward_fn, target_round=40)
    ol_m.train_full()
    none_of = ol_m.score_towers([{"tower": "bomb", "at": [0.4, 0.4],
                                  "path": [1, 0, 0]}])
    lots_of = ol_m.score_towers([{"tower": "dart", "at": [0.4, 0.4],
                                  "path": [1, 0, 0]}] * 3)
    assert lots_of > none_of, \
        f"missing-key encoding broken: none={none_of} >= lots={lots_of}"

    # Zero usable rows: scoring degrades to 0.5, never crashes.
    ol_e = OutcomeLearner(knowledge)
    ol_e.prepare([], reward_fn)
    assert not ol_e.gate()["open"]
    assert abs(ol_e.score_towers([{"tower": "dart", "at": [0.1, 0.1],
                                   "path": [0, 0, 0]}]) - 0.5) < 1e-9

    # Too little data: gate closed.
    ol3 = OutcomeLearner(knowledge)
    ol3.prepare(rows[:6], reward_fn)
    assert not ol3.gate()["open"], "gate must stay closed on 6 rows"

    # towers_from_genome: tiers accumulate, upgrades to missing refs drop.
    genome = [{"do": "place", "tower": "Tack", "at": [0.1, 0.2], "ref": 0},
              {"do": "upgrade", "ref": 0, "path": [0, 0, 1]},
              {"do": "upgrade", "ref": 0, "path": [0, 0, 1]},
              {"do": "upgrade", "ref": 9, "path": [1, 0, 0]}]
    tw = towers_from_genome(genome)
    assert tw == [{"tower": "tack", "at": [0.1, 0.2], "path": [0, 0, 2]}]

    # AUC sanity.
    assert auc_score([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert auc_score([0.1, 0.9], [1, 0]) == 0.0
    assert abs(auc_score([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) - 0.5) < 1e-9
    assert auc_score([0.5], [1]) is None

    # Income curve: learn a synthetic economy, ignore the prior inside
    # the data range, monotone, sane extrapolation at both ends.
    prior = lambda r: 650 + 25 * r + 11 * r * r
    curve = IncomeCurve(prior)
    rows_i = []
    for seed in range(4):
        r = random.Random(seed)
        cash, spent = {}, {}
        bank = 650
        for rd in range(6, 41):
            cash[str(rd)] = bank + r.randint(-20, 20)
            income = 20 * rd            # much slower than the prior
            buy = 15 * rd if rd % 3 == 0 else 0
            spent[str(rd)] = buy
            bank += income - buy
        rows_i.append({"cash_by_round": cash, "spent_by_round": spent})
    curve.fit(rows_i)
    assert curve.fitted, "income curve should fit 4 telemetry episodes"
    mid = curve.cumulative(25)
    true_mid = 650 + sum(20 * rd for rd in range(6, 25))
    assert abs(mid - true_mid) < 0.15 * true_mid, \
        f"learned income {mid} far from truth {true_mid}"
    assert mid < prior(25), "chimps-like slow economy must undercut prior"
    assert all(curve.cumulative(r) <= curve.cumulative(r + 1) + 1e-6
               for r in range(6, 45)), "income curve must be monotone"
    empty = IncomeCurve(prior)
    empty.fit([])
    assert not empty.fitted and empty.cumulative(20) == prior(20)

    # Hazard: cluster at 24 maps to the camo threat.
    threats = [{"rounds": [24, 33], "threat": "Camo", "kind": "camo",
                "answers": "ninja"}]
    dead = [{"outcome": "defeat", "final_round": 24} for _ in range(4)]
    lines = hazard_report(dead, threats)
    assert any("Camo" in ln for ln in lines), lines
    assert threat_near(25, threats)["kind"] == "camo"
    assert threat_near(50, threats) is None

    # Benefit report needs both arms, and solve-mode wins ('victory')
    # must count -- dropping them hid exactly the episodes where
    # screening worked.
    assert model_benefit(dead) is None
    mixed = ([{"outcome": "victory", "final_round": 100,
               "strategy": {"model": {"used": True}}}] * 4
             + [{"outcome": "defeat", "final_round": 10,
                 "strategy": {}}] * 4)
    b = model_benefit(mixed)
    assert b and b["screened_mean_round"] == 100.0 \
        and b["unscreened_mean_round"] == 10.0

    print("learner selftest OK: gate opens on signal, stays closed on")
    print("noise (3 shuffles, strict) and tiny data; missing-key")
    print("encoding is monotone; model prefers threat coverage; income")
    print("curve learns a slow economy and stays monotone; hazard maps")
    print("death clusters to known threats.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "selftest":
        _selftest()
    else:
        print("usage: python learner.py selftest")
