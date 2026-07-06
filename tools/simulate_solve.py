"""Offline proof that the solving stack converges -- no game needed.

This drives the REAL decision stack (meta.MetaBrain, learner's gated
outcome model, campaign.EpisodePolicy, attempt repair) against a toy
BTD6: a rule-based simulator that kills a run at the first threat its
scheduled buys don't answer (camo r24, lead r28, MOAB r40, ceramics
r63, raw-firepower wall, DDTs r90, BAD r100), with noise. If the brain,
the campaign policy, and the repair loop work, the bot must climb from
early deaths to an actual round-100 CHIMPS win within a modest episode
budget -- the same loop `mk.py solve` runs against the real game.

    python tools/simulate_solve.py            # one seeded campaign
    python tools/simulate_solve.py --seeds 5  # robustness sweep
"""

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import campaign                                          # noqa: E402
import meta                                              # noqa: E402
from meta import MetaBrain, TrackModel                   # noqa: E402


def owned_by(genome, by_round):
    """(types, tiers) actually scheduled by `by_round`."""
    types = {x["ref"]: x["tower"] for x in genome if x["do"] == "place"
             and x["round"] <= by_round}
    tiers = {}
    for x in genome:
        if x["do"] != "upgrade" or x["round"] > by_round \
                or x["ref"] not in types:
            continue
        key = (x["ref"], x["path"].index(1))
        tiers[key] = tiers.get(key, 0) + 1
    return types, tiers


def answers(genome, solutions, kind, by_round, cap=5):
    types, tiers = owned_by(genome, by_round)
    sol = solutions[kind]
    if any(sol.get(t, "x") is None for t in types.values()):
        return True
    for (ref, p), t in tiers.items():
        req = sol.get(types[ref], "absent")
        if req not in (None, "absent") and req[1] <= cap \
                and p == req[0] and t >= req[1]:
            return True
    return False


def simulate(genome, solutions, rng, quirk_carry=None):
    """Death round for a scheduled buy list, or None (= survived 100).
    Each wall is checked in game order; noise keeps labels honest.

    quirk_carry is the HIDDEN map quirk: one strong carry family that
    secretly underperforms here (think: a map whose line-of-sight
    breaks that tower). No prior can know it -- only the bot's own
    episodes can. This is what makes the sim a test of LEARNING rather
    than of the scripted baseline."""
    if rng.random() < 0.04:
        return rng.randint(10, 30)          # game chaos: bad RNG round
    if not answers(genome, solutions, "camo", 22):
        return 24 + rng.randint(0, 2)
    if not answers(genome, solutions, "lead", 27):
        return 28 + rng.randint(0, 2)
    if not answers(genome, solutions, "moab", 39, cap=4):
        return 40
    # Raw firepower wall: the mid-game needs a real carry, not just
    # threat checkboxes -- some tower at tier >= 4 by round 55.
    types, tiers = owned_by(genome, 55)
    if max(tiers.values(), default=0) < 4:
        return 47 + rng.randint(0, 12)
    # The hidden quirk: exactly ONE carry family actually performs on
    # this map (think line-of-sight, track shape, camo pockets). A
    # defense whose deep towers don't include it folds in the
    # mid-game, whatever the meta says. No prior can know which family
    # it is -- only played episodes can.
    if quirk_carry:
        types60, tiers60 = owned_by(genome, 60)
        blessed_deep = any(types60[ref] == quirk_carry and t >= 4
                           for (ref, _p), t in tiers60.items())
        if not blessed_deep:
            # Deterministic: a defense built on the wrong core dies
            # here every time. (A random escape hatch would let luck
            # promote wrong layouts into the elite pool -- a simulator
            # artifact real maps don't have.)
            return 52 + rng.randint(0, 14)
    if not answers(genome, solutions, "ceramic", 61, cap=4):
        return 63 + rng.randint(0, 15)
    # Sustain wall: total tiers by r85 measure the whole defense.
    _types, tiers85 = owned_by(genome, 85)
    if sum(tiers85.values()) < 15:
        return 76 + rng.randint(0, 10)
    if not answers(genome, solutions, "ddt", 88):
        return 90 + rng.randint(0, 5)
    if not answers(genome, solutions, "bad", 98):
        return 100
    if rng.random() < 0.08:                 # residual execution noise
        return 95 + rng.randint(0, 5)
    return None


def run_campaign(seed, episodes=60, verbose=True, learning=True):
    """One simulated solve session. learning=False is the ablation: the
    brain never observes an episode (no posteriors, no evolution, no
    attempts, no model) -- pure prior-guided random search, the
    scoreboard the learning stack has to beat."""
    rng = random.Random(seed)
    k = json.loads((REPO / "meta_knowledge.json").read_text())
    mask = json.loads(
        (REPO / "masks" / "monkey_meadow_dart.json").read_text())
    track = TrackModel(mask)
    entry = min(track.cells, key=lambda c: track.progress[c])
    track.orient([entry[0], entry[1]])
    pts = mask.get("valid_strict") or mask["valid"]
    pools = {"near": pts, "mid": [], "all": pts, "roomy": pts}
    pool = ["dart", "tack", "bomb", "sniper", "ninja", "wizard", "druid",
            "alchemist", "glue", "ice"] + meta.META_EXTRA_TOWERS

    brain = MetaBrain("sim_map", "hard", target_round=100,
                      explore=0.20, knowledge=k,
                      runs_path="/nonexistent", mode="chimps",
                      start_round=6)
    policy = campaign.EpisodePolicy()
    # The hidden map quirk rotates with the seed: the ONE carry family
    # that works "on this map".
    quirk = ["tack", "super", "wizard", "bomb", "glue"][seed % 5]
    if verbose:
        print(f"  (hidden quirk this seed: only a deep {quirk} "
              f"defense holds the mid-game)")
    screened = 0
    for ep in range(1, episodes + 1):
        decision = policy.decide(rng) if learning \
            else {"kind": "explore", "explore": 0.20}
        brain.explore = decision["explore"]
        genome = None
        if decision["kind"] == "attempt":
            genome = brain.attempt_genome(rng, pools, track=track,
                                          tower_pool=pool, hero=True)
        if genome is None:
            decision = {**decision, "kind": "explore"}
            brain.explore = decision["explore"]
            genome = brain.next_genome(rng, 5, pools, tower_pool=pool,
                                       track=track, hero=True,
                                       novelty=decision.get("novelty",
                                                            False))
        if (brain.last_strategy.get("model") or {}).get("used"):
            screened += 1
        # The game's noise gets its OWN stream, decoupled from the
        # decision stream -- otherwise one lucky draw repeats across
        # arms/quirks and fakes episode-1 wins.
        sim_rng = random.Random(seed * 100003 + ep)
        death = simulate(genome, brain.solutions, sim_rng,
                         quirk_carry=quirk)
        if learning:
            import learner
            row = {"mode": "solve", "map": "sim_map",
                   "difficulty": "hard", "game_mode": "chimps",
                   "target_round": 100, "start_round": 6,
                   "outcome": "defeat" if death else "victory",
                   "final_round": death if death else 100,
                   "strategy": brain.last_strategy,
                   "towers": learner.towers_from_genome(genome)}
            brain.observe(row, quiet=True)
            policy.update(brain._reward(row),
                          was_attempt=decision["kind"] == "attempt")
        if verbose:
            fate = "WON" if death is None else f"died r{death}"
            print(f"  ep {ep:>2} [{decision['kind']:<7}] {fate}")
        if death is None:
            return ep, screened
    return None, screened


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--ablate", action="store_true",
                    help="also run each seed with learning DISABLED "
                         "and compare -- the 'is the ML genuinely "
                         "beneficial' scoreboard")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()
    wins, ablated = [], []
    for seed in range(args.seeds):
        won_at, screened = run_campaign(seed, episodes=args.episodes,
                                        verbose=not args.quiet)
        if won_at:
            print(f"seed {seed}: CHIMPS (sim) beaten at episode "
                  f"{won_at} (model screened {screened} episodes)")
            wins.append(won_at)
        else:
            print(f"seed {seed}: NOT beaten in {args.episodes} episodes")
        if args.ablate:
            base_at, _ = run_campaign(seed, episodes=args.episodes,
                                      verbose=False, learning=False)
            ablated.append(base_at)
            print(f"   ablation (no learning): "
                  + (f"episode {base_at}" if base_at
                     else f"not beaten in {args.episodes}"))
    if len(wins) == args.seeds:
        avg = sum(wins) / len(wins)
        print(f"\nall {args.seeds} seed(s) beat the simulated CHIMPS; "
              f"mean episodes to win {avg:.1f}")
        if args.ablate:
            solved = [a for a in ablated if a]
            base = (f"mean {sum(solved) / len(solved):.1f} over "
                    f"{len(solved)}/{len(ablated)} seeds solved"
                    if solved else "never solved")
            print(f"ablation without learning: {base}")
    else:
        sys.exit(f"\n{args.seeds - len(wins)} seed(s) failed to "
                 f"converge -- the solving stack has a regression")


if __name__ == "__main__":
    main()
