"""The campaign layer: the meta strategy that turns farming into SOLVING.

`farm` collects data; `solve` (mk.py) plays to WIN. This module is the
pure-logic brain behind solve: which rung of the game is next, whether
the current episode should explore (gather information) or attempt (play
the best known strategy straight), and the persistent scoreboard of what
has been beaten.

The ladder for every map runs easy -> medium -> hard -> CHIMPS: each
rung's episodes feed the same runs_log.jsonl, so everything learned on
the way up (tower posteriors, income curves, elite layouts) transfers
into the CHIMPS attempt via meta.py's hierarchical posteriors and
elite seeding.

No screen/game dependencies:

    python campaign.py selftest
"""

import json
from datetime import datetime
from pathlib import Path

PROGRESS_PATH = Path(__file__).parent / "progress.json"

# One ladder per map. CHIMPS is hard-difficulty rules with the mode
# flags on top (1 life, pops-only income, no selling/powers/continues),
# so it shares the hard price book.
LADDER = [("easy", "standard"), ("medium", "standard"),
          ("hard", "standard"), ("hard", "chimps")]

# What each rung looks like from the HUD, and what winning means.
#   lives: starting lives (how the rung is auto-detected)
#   start: the round the game begins on
#   target: the final round -- survive it and the rung is beaten
RUNG_INFO = {
    ("easy", "standard"): {"lives": 200, "start": 1, "target": 40},
    ("medium", "standard"): {"lives": 150, "start": 1, "target": 60},
    ("hard", "standard"): {"lives": 100, "start": 3, "target": 80},
    ("impoppable", "standard"): {"lives": 1, "start": 3, "target": 100},
    ("hard", "chimps"): {"lives": 1, "start": 6, "target": 100},
}


def rung_key(map_name, difficulty, mode):
    return f"{map_name}|{difficulty}|{mode}"


def detect_rung(lives, start_round):
    """(difficulty, mode) from the HUD at a fresh, un-started map.
    Lives pins the difficulty; for 1-life games the starting round
    separates CHIMPS (starts at 6) from impoppable (starts at 3)."""
    if lives is None:
        return None
    for base, rung in ((200, ("easy", "standard")),
                       (150, ("medium", "standard")),
                       (100, ("hard", "standard"))):
        if abs(lives - base) <= 5:
            return rung
    if lives <= 3:
        if start_round is not None and start_round >= 5:
            return ("hard", "chimps")
        return ("impoppable", "standard")
    return None


def rung_target(difficulty, mode):
    info = RUNG_INFO.get((difficulty, mode))
    if info:
        return info["target"]
    return {"easy": 40, "medium": 60, "hard": 80,
            "impoppable": 100}.get(difficulty, 40)


def rung_start(difficulty, mode):
    info = RUNG_INFO.get((difficulty, mode))
    return info["start"] if info else 1


class Progress:
    """The persistent scoreboard: per (map, difficulty, mode), how many
    episodes were played, the deepest round reached, and whether the
    rung has been beaten. Lives in progress.json next to the bot."""

    def __init__(self, path=None):
        self.path = Path(path) if path else PROGRESS_PATH
        self.data = {"rungs": {}}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text())
                if isinstance(loaded.get("rungs"), dict):
                    self.data = loaded
            except ValueError:
                pass                      # corrupt file: start fresh

    def _save(self):
        self.path.write_text(json.dumps(self.data, indent=1) + "\n")

    def rung(self, map_name, difficulty, mode):
        return self.data["rungs"].setdefault(
            rung_key(map_name, difficulty, mode),
            {"episodes": 0, "attempts": 0, "best_round": 0,
             "beaten": False, "beaten_at": None})

    def record_episode(self, map_name, difficulty, mode, outcome,
                       final_round, was_attempt=False):
        r = self.rung(map_name, difficulty, mode)
        r["episodes"] += 1
        if was_attempt:
            r["attempts"] += 1
        if isinstance(final_round, int):
            r["best_round"] = max(r["best_round"], final_round)
        if outcome == "victory":
            r["beaten"] = True
            r["beaten_at"] = datetime.now().isoformat(timespec="seconds")
        self._save()
        return r

    def next_rung(self, map_name, ladder=None):
        """First unbeaten rung on the map's ladder, or None if done."""
        for difficulty, mode in (ladder or LADDER):
            r = self.data["rungs"].get(rung_key(map_name, difficulty, mode))
            if not r or not r.get("beaten"):
                return (difficulty, mode)
        return None

    def board(self, maps, ladder=None):
        """Human scoreboard, one line per rung per map."""
        lines = [f"{'map':<20}{'rung':<18}{'status':<10}"
                 f"{'best':>6}{'episodes':>10}"]
        for m in maps:
            for difficulty, mode in (ladder or LADDER):
                r = self.data["rungs"].get(rung_key(m, difficulty, mode))
                rung_name = (f"{difficulty}" if mode == "standard"
                             else f"{mode}")
                target = rung_target(difficulty, mode)
                if r is None:
                    status, best, eps = "--", "", ""
                else:
                    status = "BEATEN" if r["beaten"] else "open"
                    best = f"r{r['best_round']}"
                    eps = str(r["episodes"])
                lines.append(f"{m:<20}{rung_name:<18}{status:<10}"
                             f"{best:>6}{eps:>10}"
                             + (f"   (target r{target})"
                                if r and not r["beaten"] else ""))
            nxt = self.next_rung(m, ladder)
            lines.append(f"{'':<20}next: "
                         + (f"{nxt[1] if nxt[1] != 'standard' else nxt[0]}"
                            if nxt else "map complete -- pick a new map"))
        return lines


class EpisodePolicy:
    """Explore or attempt? The bandit-flavored schedule for one rung.

    - Early on there is nothing worth attempting: explore with extra
      randomness so the dataset gets both classes.
    - As the best-known layout closes in on the target, attempts (play
      the champion straight, tiny explore) take over -- but never
      completely: every streak of attempts is broken up by an explore
      episode, so the well never poisons itself.
    - A plateau (no new best for a while) cranks exploration back up:
      the current strategy family is exhausted, go find another.
    """

    BOOTSTRAP = 4          # episodes before attempts are considered
    PLATEAU = 8            # episodes without improvement = stuck

    def __init__(self):
        self.n = 0
        self.best = 0.0
        self.since_improve = 0
        self.attempt_streak = 0

    def update(self, reward, was_attempt=False):
        """Fold one finished episode's reward (0..1, meta.py shaping)."""
        self.n += 1
        if reward is not None and reward > self.best + 1e-6:
            self.best = reward
            self.since_improve = 0
        else:
            self.since_improve += 1
        self.attempt_streak = (self.attempt_streak + 1) if was_attempt \
            else 0

    def decide(self, rng):
        """-> {"kind": "explore"|"attempt", "explore": rate}.

        Attempts only pay once the champion is genuinely CLOSE (a
        best-reward of 0.75 is "died in the mid-game" -- replaying that
        is a wasted episode). And a plateau raises exploration only
        moderately: diversity must come from trying other towers, not
        from the higher explore rates that break threat coverage and
        make every layout die young."""
        if self.n < self.BOOTSTRAP:
            return {"kind": "explore", "explore": 0.35,
                    "novelty": False}
        plateau = self.since_improve >= self.PLATEAU
        p_attempt = min(max((self.best - 0.70) * 3.0, 0.0), 0.85)
        if plateau:
            p_attempt *= 0.4
        if self.attempt_streak >= 3:
            p_attempt = 0.0            # forced breather: explore
        if rng.random() < p_attempt:
            return {"kind": "attempt", "explore": 0.03,
                    "novelty": False}
        # On a plateau, most exploration goes to NOVELTY: coherent
        # layouts built around the least-tried tower families (the
        # brain's novelty mode), because the exhausted thing is the
        # strategy family itself.
        novelty = plateau and rng.random() < 0.7
        return {"kind": "explore",
                "explore": 0.35 if plateau else 0.20,
                "novelty": novelty}


# ---------------------------------------------------------------- selftest

def _selftest():
    import random
    import tempfile

    # detect_rung: every ladder rung round-trips from its own HUD.
    for (diff, mode), info in RUNG_INFO.items():
        got = detect_rung(info["lives"], info["start"])
        assert got == (diff, mode), f"{info} detected as {got}"
    assert detect_rung(None, 1) is None
    assert detect_rung(1, None) == ("impoppable", "standard")
    assert rung_target("hard", "chimps") == 100
    assert rung_start("hard", "chimps") == 6

    # Progress: record, persist, reload, ladder advance.
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "progress.json"
        p = Progress(path)
        assert p.next_rung("monkey_meadow") == ("easy", "standard")
        p.record_episode("monkey_meadow", "easy", "standard",
                         "defeat", 25)
        p.record_episode("monkey_meadow", "easy", "standard",
                         "victory", 40, was_attempt=True)
        p2 = Progress(path)                      # reload from disk
        r = p2.rung("monkey_meadow", "easy", "standard")
        assert r["beaten"] and r["best_round"] == 40 \
            and r["episodes"] == 2 and r["attempts"] == 1
        assert p2.next_rung("monkey_meadow") == ("medium", "standard")
        for d, m in (("medium", "standard"), ("hard", "standard"),
                     ("hard", "chimps")):
            p2.record_episode("monkey_meadow", d, m, "victory", 100)
        assert p2.next_rung("monkey_meadow") is None
        board = "\n".join(p2.board(["monkey_meadow", "logs"]))
        assert "BEATEN" in board and "map complete" in board
        assert "logs" in board and "--" in board
        # Corrupt file: start fresh, no crash.
        path.write_text("{broken")
        assert Progress(path).next_rung("monkey_meadow") \
            == ("easy", "standard")

    # Policy: explores first, attempts once the champion is close,
    # never runs attempts unbroken, re-explores on plateau.
    rng = random.Random(3)
    pol = EpisodePolicy()
    assert all(pol.decide(rng)["kind"] == "explore" for _ in range(5))
    for _ in range(4):
        pol.update(0.3)
    kinds = [pol.decide(rng)["kind"] for _ in range(50)]
    assert kinds.count("attempt") == 0, \
        "attempts before anything works are wasted episodes"
    pol.update(0.97)                    # nearly beat the target
    kinds = [pol.decide(rng)["kind"] for _ in range(200)]
    assert kinds.count("attempt") >= 100, \
        f"champion at 0.97 should mostly attempt: {kinds.count('attempt')}"
    # Attempt streaks get broken up.
    pol.attempt_streak = 3
    assert pol.decide(rng)["kind"] == "explore"
    # Plateau: exploration rate rises -- but moderately, never so far
    # that threat coverage (which needs exploit-mode decisions) breaks.
    pol2 = EpisodePolicy()
    for _ in range(4):
        pol2.update(0.4)
    for _ in range(9):
        pol2.update(0.35)               # no improvement for a while
    explores = [pol2.decide(rng) for _ in range(60)]
    ex = [e for e in explores if e["kind"] == "explore"]
    assert ex and all(0.30 <= e["explore"] <= 0.40 for e in ex), \
        "plateau must raise exploration moderately"
    novel = sum(1 for e in ex if e.get("novelty"))
    assert 0.5 <= novel / len(ex) <= 0.9, \
        f"plateau exploration should mostly be novelty: {novel}/{len(ex)}"
    pre = EpisodePolicy()
    for _ in range(4):
        pre.update(0.4)
    assert not any(pre.decide(rng).get("novelty") for _ in range(40)), \
        "novelty is a plateau response, not a default"
    # A mid-game champion (best ~0.72) is NOT worth attempting yet.
    pol3 = EpisodePolicy()
    for _ in range(4):
        pol3.update(0.72)
    kinds3 = [pol3.decide(rng)["kind"] for _ in range(100)]
    assert kinds3.count("attempt") <= 15, \
        f"mid-game champions should rarely be attempted: " \
        f"{kinds3.count('attempt')}"

    print("campaign selftest OK: rung detection, progress persistence,")
    print("ladder advancement, and the explore/attempt policy verified.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "selftest":
        _selftest()
    else:
        print("usage: python campaign.py selftest")
