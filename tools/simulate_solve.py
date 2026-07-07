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


def anchor_spot(genome):
    """The opening anchor's map spot -- hero, else opener, else carry, else
    the earliest place. Matched to meta.MetaBrain._anchor_index so the
    sim's opener wall and the brain's opener credit judge the SAME piece."""
    places = [e for e in genome
              if e.get("do") == "place" and e.get("at")]
    if not places:
        return None
    for e in places:
        if e["tower"] == "hero":
            return e["at"]
    for want in ("opener", "carry"):
        for e in places:
            if meta._role_of_name(e.get("name")) == want:
                return e["at"]
    return min(places, key=lambda e: e.get("round", 0)).get("at")


def carry_and_amps(genome):
    """The carry's spot and every alch/village spot -- for the synergy
    wall's 'is support actually reaching the carry' check."""
    places = [e for e in genome
              if e.get("do") == "place" and e.get("at")]
    carry = next((e["at"] for e in places
                  if meta._role_of_name(e.get("name")) == "carry"), None)
    amps = [e["at"] for e in places
            if e["tower"] in ("alchemist", "village")]
    return carry, amps


def simulate(genome, solutions, rng, track=None, quirk_carry=None,
             quirk_leak_bucket=None, walls=None):
    """Death round for a scheduled buy list, or None (= survived 100).
    Each wall is checked in game order; noise keeps labels honest.

    TWO hidden per-seed quirks make this a test of LEARNING, not of the
    scripted baseline -- on orthogonal axes so the bot must learn both:
      * quirk_carry: one carry FAMILY that secretly performs here (a
        line-of-sight / track-shape quirk the meta can't know).
      * quirk_leak_bucket: one high-exposure OPENER SPOT that secretly
        leaks. The static exposure heuristic PREFERS it, so a prior-only
        bot parks its anchor there and dies every game; only a bot whose
        failure attribution demoted that spot (pos_post) steers away.
    `walls` toggles individual walls off for isolation ablations."""
    walls = walls or {}
    if rng.random() < 0.04:
        return rng.randint(10, 30)          # game chaos: bad RNG round
    # Opener wall: the hidden leaky pocket. Fires on the anchor's spot, in
    # the opener window (r8-18) so the brain reads it as an opener leak.
    if walls.get("opener", True) and quirk_leak_bucket is not None:
        a = anchor_spot(genome)
        if a is not None and meta._bucket(a) == quirk_leak_bucket:
            return 8 + rng.randint(0, 10)
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
    # The hidden carry quirk: exactly ONE carry family actually performs
    # on this map. A defense whose deep towers don't include it folds in
    # the mid-game, whatever the meta says.
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
    # r78 milestone: the denser ceramic wave demands the defense be MOSTLY
    # online by r76 -- a WHEN check (timing), distinct from the r85 total.
    if walls.get("milestone", True):
        _t76, tiers76 = owned_by(genome, 76)
        if sum(tiers76.values()) < 12:
            return 78 + rng.randint(0, 8)
    # Sustain wall, synergy-multiplied: a linked alch/village MULTIPLIES
    # the carry's firepower, so a supported defense holds with fewer raw
    # tiers than one grinding alone -- CHIMPS can't brute-force with money.
    _types, tiers85 = owned_by(genome, 85)
    effective = sum(tiers85.values())
    if walls.get("synergy", True):
        carry_at, amp_ats = carry_and_amps(genome)
        n_linked = sum(1 for a in amp_ats
                       if carry_at
                       and meta._dist(a, carry_at) <= 0.06 * 0.9)
        effective *= (1.0 + 0.25 * n_linked)
    if effective < 15:
        return 76 + rng.randint(0, 10)
    if not answers(genome, solutions, "ddt", 88):
        return 90 + rng.randint(0, 5)
    if not answers(genome, solutions, "bad", 98):
        return 100
    if rng.random() < 0.08:                 # residual execution noise
        return 95 + rng.randint(0, 5)
    return None


def anchor_score(track, pt):
    """The score meta._spot_for gives a coverage anchor at `pt`, ignoring
    the learned posterior (exposure biased toward killing bloons EARLY).
    Used to find the spot a PRIOR-ONLY bot parks its opener on."""
    exp = track.exposure(pt, 0.045)
    sp = track.span(pt, 0.045)
    return exp * (1.25 - 0.5 * sp[1]) if (sp and track.oriented) else exp


def leak_bucket_for(track, pts, seed):
    """The hidden leaky OPENER pocket: the very spot a prior-only bot's
    anchor-placement heuristic likes BEST. So an unlearning bot parks its
    opener there and leaks every game; only a bot whose failure
    attribution demoted the spot (pos_post) moves to a near-equal
    neighbour. The `seed` term only breaks ties among near-identical
    top spots -- the leak stays the anchor's default pick so it actually
    fires."""
    best = {}
    for pt in pts:
        b = meta._bucket(pt)
        best[b] = max(best.get(b, 0.0), anchor_score(track, pt))
    top = sorted(best, key=lambda b: -best[b])
    if not top:
        return None
    # Keep it the anchor's default (top[0]); rotate only within a hair of
    # the best so every seed's wall still fires on the default placement.
    near = [b for b in top if best[b] >= best[top[0]] * 0.98]
    return near[seed % len(near)]


def run_campaign(seed, episodes=60, verbose=True, learning=True,
                 walls=None):
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
    leak_bucket = leak_bucket_for(track, pts, seed)
    if verbose:
        print(f"  (hidden quirks this seed: only a deep {quirk} holds the "
              f"mid-game; opener spot {leak_bucket} secretly leaks)")
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
        death = simulate(genome, brain.solutions, sim_rng, track=track,
                         quirk_carry=quirk, quirk_leak_bucket=leak_bucket,
                         walls=walls)
        if learning:
            import learner
            lbr = ({6: 1} if death is None else {6: 1, death: 0})
            row = {"mode": "solve", "map": "sim_map",
                   "difficulty": "hard", "game_mode": "chimps",
                   "target_round": 100, "start_round": 6,
                   "outcome": "defeat" if death else "victory",
                   "final_round": death if death else 100,
                   "lives_by_round": lbr,
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


def run_deploy(seed, train_episodes=60, games=25, verbose=True):
    """Offline proof of the `deploy` path (mk.py deploy). Train a
    champion the normal way, then play it STRAIGHT -- pure attempt,
    explore=0, nothing observed -- and confirm the finished model wins
    the sim on its own. Returns (wins, games), or None if no champion was
    trained within budget (nothing to deploy, exactly the case deploy
    refuses to guess through).

    This is what running the trained bot looks like once solving is done:
    no exploration, no learning, just the champion replayed."""
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
    quirk = ["tack", "super", "wizard", "bomb", "glue"][seed % 5]
    leak_bucket = leak_bucket_for(track, pts, seed)

    # An untrained model has no champion -- the gate `deploy` enforces
    # before ever starting a game.
    fresh = MetaBrain("sim_map", "hard", target_round=100, explore=0.0,
                      knowledge=k, runs_path="/nonexistent",
                      mode="chimps", start_round=6)
    assert fresh.attempt_genome(rng, pools, tower_pool=pool, track=track,
                                hero=True) is None, \
        "an untrained model must have no champion to deploy"

    # Train (the solve loop) just until a champion first survives.
    import learner
    brain = MetaBrain("sim_map", "hard", target_round=100, explore=0.20,
                      knowledge=k, runs_path="/nonexistent",
                      mode="chimps", start_round=6)
    policy = campaign.EpisodePolicy()
    trained = False
    for ep in range(1, train_episodes + 1):
        decision = policy.decide(rng)
        brain.explore = decision["explore"]
        genome = None
        if decision["kind"] == "attempt":
            genome = brain.attempt_genome(rng, pools, track=track,
                                          tower_pool=pool, hero=True)
        if genome is None:
            brain.explore = decision["explore"]
            genome = brain.next_genome(rng, 5, pools, tower_pool=pool,
                                       track=track, hero=True,
                                       novelty=decision.get("novelty",
                                                            False))
        sim_rng = random.Random(seed * 100003 + ep)
        death = simulate(genome, brain.solutions, sim_rng, track=track,
                         quirk_carry=quirk, quirk_leak_bucket=leak_bucket)
        row = {"mode": "solve", "map": "sim_map", "difficulty": "hard",
               "game_mode": "chimps", "target_round": 100,
               "start_round": 6,
               "outcome": "defeat" if death else "victory",
               "final_round": death if death else 100,
               "lives_by_round": ({6: 1} if death is None
                                  else {6: 1, death: 0}),
               "strategy": brain.last_strategy,
               "towers": learner.towers_from_genome(genome)}
        brain.observe(row, quiet=True)
        policy.update(brain._reward(row),
                      was_attempt=decision["kind"] == "attempt")
        if death is None:
            trained = True
            break
    if not trained:
        return None                 # never found a champion in budget

    # DEPLOY: champion only, explore=0, nothing observed back. Each game
    # gets its own decoupled noise stream so we measure the champion, not
    # one lucky roll.
    brain.explore = 0.0
    wins = 0
    for g in range(1, games + 1):
        genome = brain.attempt_genome(rng, pools, track=track,
                                      tower_pool=pool, hero=True)
        assert genome is not None, "trained model lost its champion"
        sim_rng = random.Random(seed * 7919 + g * 101 + 1)
        death = simulate(genome, brain.solutions, sim_rng, track=track,
                         quirk_carry=quirk, quirk_leak_bucket=leak_bucket)
        wins += 1 if death is None else 0
        if verbose:
            print(f"  deploy game {g:>2}: "
                  + ("WON" if death is None else f"died r{death}"))
    return wins, games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--ablate", action="store_true",
                    help="also run each seed with learning DISABLED "
                         "and compare -- the 'is the ML genuinely "
                         "beneficial' scoreboard")
    ap.add_argument("--deploy", action="store_true",
                    help="instead of the convergence sweep, prove the "
                         "`deploy` path: train a champion, then play it "
                         "straight (no exploration, no learning) and "
                         "confirm the finished model wins on its own")
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--no-opener-wall", action="store_true",
                    help="disable the hidden leaky-opener wall (isolate "
                         "the opener/real-estate lever)")
    ap.add_argument("--no-synergy-wall", action="store_true",
                    help="disable the synergy-multiplied sustain wall "
                         "(isolate the support-synergy lever)")
    ap.add_argument("--no-milestone-wall", action="store_true",
                    help="disable the r78 timing milestone wall")
    args = ap.parse_args()
    walls = {"opener": not args.no_opener_wall,
             "synergy": not args.no_synergy_wall,
             "milestone": not args.no_milestone_wall}
    if args.deploy:
        total_w = total_g = solved = 0
        for seed in range(args.seeds):
            res = run_deploy(seed, train_episodes=args.episodes,
                             verbose=not args.quiet)
            if res is None:
                print(f"deploy seed {seed}: no champion trained in "
                      f"{args.episodes} episodes -- nothing to deploy")
                continue
            w, gm = res
            solved += 1
            total_w += w
            total_g += gm
            print(f"deploy seed {seed}: champion won {w}/{gm} straight "
                  f"games")
        if not total_g:
            sys.exit("no champion trained on any seed -- raise "
                     "--episodes")
        rate = total_w / total_g
        print(f"\ndeploy: champion win rate {rate:.0%} over {total_g} "
              f"straight games ({solved}/{args.seeds} seed(s) had a "
              f"champion to deploy)")
        if rate < 0.5:
            sys.exit("deploy regression: a trained champion, replayed "
                     "straight, should win the sim on its own")
        print("deploy OK: the finished model wins without exploring.")
        return
    wins, ablated = [], []
    for seed in range(args.seeds):
        won_at, screened = run_campaign(seed, episodes=args.episodes,
                                        verbose=not args.quiet, walls=walls)
        if won_at:
            print(f"seed {seed}: CHIMPS (sim) beaten at episode "
                  f"{won_at} (model screened {screened} episodes)")
            wins.append(won_at)
        else:
            print(f"seed {seed}: NOT beaten in {args.episodes} episodes")
        if args.ablate:
            base_at, _ = run_campaign(seed, episodes=args.episodes,
                                      verbose=False, learning=False,
                                      walls=walls)
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
