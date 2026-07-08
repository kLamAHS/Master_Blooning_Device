"""Offline test for the one-life (CHIMPS / Impoppable) no-leak opener.

The bug this guards: on CHIMPS the brain used to make the HERO the sole
round-6 anchor. A hero covers only ~3% of track (its range is tiny) AND
drains the whole $650 starting budget, so nothing else could be afforded and
the run leaked rounds 6-9 -- 205/205 defeats in a real training log, one
tower placed per game. The fix: on one-life rungs, fit several CHEAP popping
defenders into the starting budget at round 6, preferring real DPS over the
hero (which now schedules a couple rounds later, behind the defense).

This drives meta.py's pure planners (_one_life, _role_slots, _schedule)
directly -- no game, no cv2.

    python tools/test_opener.py        # exits non-zero on any failure
"""

import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from meta import MetaBrain, ROUGH_COST            # noqa: E402

K = json.loads((REPO / "meta_knowledge.json").read_text())
_fails = []


def check(name, cond):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}")
    if not cond:
        _fails.append(name)


def brain(mode="chimps", difficulty="hard", start=6, target=100):
    return MetaBrain("m", difficulty, target_round=target, explore=0.0,
                     knowledge=K, runs_path="/nonexistent", mode=mode,
                     start_round=start)


def place(tower, role, est, ref, prio=0, dl=1.0):
    return {"do": "place", "tower": tower, "name": f"{tower}#{ref}({role})",
            "ref": ref, "prio": prio, "deadline": dl, "est": est,
            "at": [0.5, 0.5]}


def rounds_by_role(entries):
    return {_role(e): e["round"] for e in entries if e["do"] == "place"}


def _role(e):
    n = e["name"]
    return n[n.rfind("(") + 1:n.rfind(")")]


def main():
    print("one-life no-leak opener:")

    # --- _one_life --------------------------------------------------------
    check("chimps is one-life", brain(mode="chimps")._one_life())
    check("impoppable is one-life",
          brain(mode="standard", difficulty="impoppable")._one_life())
    check("easy standard is NOT one-life",
          not brain(mode="standard", difficulty="easy")._one_life())

    # --- _role_slots: a cheap opener leads even with a hero, on CHIMPS ----
    ch = brain(mode="chimps")
    slots = ch._role_slots(5, hero=True)
    check("CHIMPS hero mode plans an 'opener' defender",
          "opener" in slots and slots[0] == "opener")
    ez = brain(mode="standard", difficulty="easy")
    check("easy hero mode still has NO separate opener (hero is the opener)",
          "opener" not in ez._role_slots(5, hero=True))

    # --- _schedule budget-fit: cheap defenders at round 6, hero deferred --
    b = brain(mode="chimps", start=6)                 # income(6) == $650
    budget = b.income(6)
    entries = [
        place("dart", "opener", 200, 0),
        place("super", "carry", 2500, 1),
        place("glue", "control", 275, 2, prio=1, dl=13.0),
        place("alchemist", "amplifier", 550, 3, prio=1, dl=9.0),
        place("hero", "hero", 600, 4, prio=0, dl=2.0),
    ]
    sched = b._schedule([dict(e) for e in entries])
    rr = rounds_by_role(sched)
    anchors = [e for e in sched if e.get("_anchor")]
    anchor_cost = sum(e.get("est") or 0 for e in anchors)
    check("cheap opener is down by round 6", rr["opener"] <= 6)
    check("a second cheap defender (control) is also down by round 6",
          rr["control"] <= 6)
    check("the $2500 super is NOT forced to round 6", rr["carry"] > 6)
    check("the hero is NOT forced to round 6 (it schedules behind defense)",
          rr["hero"] > 6)
    check("round-6 anchors fit the starting budget",
          anchor_cost <= budget and len(anchors) >= 2)

    # --- fallback: when nothing affordable fits, still never leak round 6 -
    b2 = brain(mode="chimps", start=6)
    heavy = [place("super", "carry", 2500, 0),
             place("village", "amplifier", 1200, 1, prio=1, dl=9.0)]
    s2 = b2._schedule([dict(e) for e in heavy])
    anchored = [e for e in s2 if e.get("_anchor")]
    check("no affordable tower -> still pull the single cheapest to round 6",
          len(anchored) == 1 and anchored[0]["est"] == 1200
          and anchored[0]["round"] <= 6)

    # --- forgiving rung: hero-first anchor is UNCHANGED ------------------
    e = brain(mode="standard", difficulty="easy", start=1, target=40)
    fe = [place("hero", "hero", 600, 0, prio=0, dl=2.0),
          place("dart", "carry", 200, 1, prio=0, dl=3.0),
          place("glue", "control", 275, 2, prio=1, dl=13.0)]
    fs = e._schedule([dict(x) for x in fe])
    fr = rounds_by_role(fs)
    fanchors = [x for x in fs if x.get("_anchor")]
    check("forgiving rung pins exactly one anchor (unchanged behavior)",
          len(fanchors) == 1)
    check("forgiving-rung anchor is the hero",
          _role(fanchors[0]) == "hero" and fr["hero"] <= 1)

    # --- end-to-end: real genomes on CHIMPS actually place a non-hero
    #     popping defender at the start round, most of the time -----------
    rng = random.Random(0)
    pool = ["dart", "tack", "bomb", "sniper", "ninja", "wizard", "druid",
            "alchemist", "glue", "ice", "boomerang", "super", "village",
            "engineer", "spike"]
    pools = {"near": [[0.5, 0.5]], "mid": [[0.5, 0.5]],
             "all": [[0.5, 0.5]], "roomy": [[0.5, 0.5]]}
    g = brain(mode="chimps", start=6)
    early_defender = killer_opener = early_teeth = opener_seen = 0
    cheap_opener = 0
    budget = g.income(6)
    trials = 30
    for _ in range(trials):
        genome = g.next_genome(rng, 5, pools, tower_pool=pool, hero=True,
                               large_towers={"super", "village"})
        r6 = [it for it in genome if it.get("do") == "place"
              and it.get("round", 99) <= 6]
        # at least one round-6 placement that is NOT the hero
        if any((it.get("tower") or "").lower() != "hero" for it in r6):
            early_defender += 1
        opener = next((it for it in genome if it.get("do") == "place"
                       and _role(it) == "opener"), None)
        if opener:
            opener_seen += 1
            # the one-life opener must be a killer, never pure slow (glue/ice)
            if (opener.get("tower") or "").lower() not in ("glue", "ice"):
                killer_opener += 1
            # ...and cheap enough to leave room for its own upgrades + a
            # second tower (a $500 ninja ate the whole $650, so only 1 tower
            # ever landed).
            if ROUGH_COST.get((opener.get("tower") or "").lower(), 600) \
                    <= 0.6 * budget:
                cheap_opener += 1
            # and it must get early teeth: a prio-0 upgrade by ~round 8
            ups = [it for it in genome if it.get("do") == "upgrade"
                   and it.get("ref") == opener["ref"]]
            if any(u.get("round", 99) <= 9 and u.get("prio") == 0
                   for u in ups):
                early_teeth += 1
    check(f"CHIMPS genomes place a non-hero defender by round 6 "
          f"({early_defender}/{trials})", early_defender >= trials - 2)
    check(f"the one-life opener is a KILLER, not glue/ice "
          f"({killer_opener}/{opener_seen})",
          opener_seen >= trials - 2 and killer_opener == opener_seen)
    check(f"the one-life opener is affordable, not a budget-eating $500 tower "
          f"({cheap_opener}/{opener_seen})", cheap_opener == opener_seen)
    check(f"the one-life opener gets early teeth (prio-0 upgrade by r9) "
          f"({early_teeth}/{opener_seen})", early_teeth >= 0.8 * opener_seen)

    # --- opener build ROTATION: a line that keeps leaking the opening must
    #     not be retried forever -- the bot should try a different one -------
    def leak_row(main_path, final_round=8):
        path = [0, 0, 0]
        path[main_path] = 2
        return {"map": "m", "difficulty": "hard", "mode": "solve",
                "game_mode": "chimps", "target_round": 100, "start_round": 6,
                "final_round": final_round, "outcome": "defeat",
                "towers": [{"tower": "dart", "name": "dart#0(opener)",
                            "at": [0.4, 0.5], "path": path}]}

    def opener_main_dist(br, n=30):
        rng2 = random.Random(1)
        from collections import Counter
        return Counter(br._pick_build(rng2, "dart", follow_template=True)[0]
                       for _ in range(n))

    base = brain(mode="chimps", start=6)
    top_main = opener_main_dist(base).most_common(1)[0][0]      # dart's default
    rot = brain(mode="chimps", start=6)
    for _ in range(4):
        rot.observe(leak_row(top_main), quiet=True)             # it keeps leaking
    dist = opener_main_dist(rot)
    check(f"opener rotates OFF a build line that keeps leaking "
          f"(default main {top_main}; after 4 leaks -> {dict(dist)})",
          dist.get(top_main, 0) <= 0.5 * sum(dist.values()))
    # leak the next line too -> it moves on again, never back to the first
    second = dist.most_common(1)[0][0]
    for _ in range(4):
        rot.observe(leak_row(second), quiet=True)
    dist2 = opener_main_dist(rot)
    check(f"a second leaking line is also abandoned "
          f"(-> {dict(dist2)})",
          dist2.most_common(1)[0][0] not in (top_main, second))
    # a fresh brain with NO leak history is unchanged (still the meta default)
    check("no leak history -> opener build is unchanged (the meta default)",
          opener_main_dist(brain(mode="chimps", start=6)).most_common(1)[0][0]
          == top_main)

    print()
    if _fails:
        print(f"FAILED {len(_fails)} case(s): {', '.join(_fails)}")
        sys.exit(1)
    print("all one-life opener cases passed")


if __name__ == "__main__":
    main()
