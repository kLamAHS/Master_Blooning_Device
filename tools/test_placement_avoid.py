"""Offline unit test for the placement-avoid geometry (placement.free_
placement_spots) -- the rule that stops the bot clicking a spot it already
knows is taken and stacking towers on top of each other.

    python tools/test_placement_avoid.py     # exits non-zero on any failure
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from placement import free_placement_spots      # noqa: E402

_fails = []


def check(name, cond):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}")
    if not cond:
        _fails.append(name)


def main():
    print("placement-avoid geometry:")
    min_d = 0.03

    # No avoid list -> candidates come back exactly as given.
    cands = [[0.5, 0.5], [0.6, 0.5]]
    check("no avoid: candidates unchanged",
          free_placement_spots(cands, [], min_d, [], [0.5, 0.5]) == cands)

    # A candidate sitting on an avoided tower is dropped; a far one is kept.
    cands = [[0.5, 0.5], [0.8, 0.8]]
    out = free_placement_spots(cands, [[0.5, 0.5]], min_d, [], [0.5, 0.5])
    check("candidate on a tower is dropped", [0.5, 0.5] not in out)
    check("far candidate is kept", [0.8, 0.8] in out)

    # Exactly min_d away is free (>= min_d**2); just inside is occupied.
    on_edge = [0.5 + min_d, 0.5]
    just_in = [0.5 + min_d * 0.5, 0.5]
    out = free_placement_spots([on_edge, just_in], [[0.5, 0.5]], min_d, [],
                               [0.5, 0.5])
    check("candidate exactly min_d away is free", on_edge in out)
    check("candidate within min_d is dropped", just_in not in out)

    # Whole planned neighborhood is taken, but the mask has free points:
    # relocate to the NEAREST free pool point rather than clicking a tower.
    spot = [0.5, 0.5]
    cands = [[0.5, 0.5], [0.51, 0.5]]          # both on the tower cluster
    avoid = [[0.5, 0.5], [0.51, 0.5]]
    pool = [[0.9, 0.9], [0.6, 0.5], [0.2, 0.2]]   # 0.6,0.5 is nearest & free
    out = free_placement_spots(cands, avoid, min_d, pool, spot)
    check("all candidates taken -> relocate to a single free spot",
          len(out) == 1)
    check("relocation picks the NEAREST free pool point",
          out == [[0.6, 0.5]])

    # Relocation still refuses a pool point that is itself occupied.
    pool = [[0.5, 0.5], [0.7, 0.5]]            # 0.5,0.5 taken; 0.7,0.5 free
    out = free_placement_spots(cands, avoid, min_d, pool, spot)
    check("relocation skips an occupied pool point", out == [[0.7, 0.5]])

    # Everything -- candidates AND the whole pool -- is taken: fall back to
    # the raw candidates (a genuinely full map) rather than returning nothing.
    pool = [[0.5, 0.5], [0.51, 0.5]]           # all within min_d of avoid
    out = free_placement_spots(cands, avoid, min_d, pool, spot)
    check("full map -> falls back to raw candidates (never empty)",
          out == cands)

    # A large tower uses a bigger min_d: a spot 0.04 from a tower is fine for
    # a small tower but too close for a large one. With a free alternative
    # present, the difference shows up as the near spot being kept vs dropped.
    cands = [[0.5 + 0.04, 0.5], [0.85, 0.85]]  # 0.04 away, and a far-free one
    small_out = free_placement_spots(cands, [[0.5, 0.5]], 0.03, [], [0.5, 0.5])
    big_out = free_placement_spots(cands, [[0.5, 0.5]], 0.05, [], [0.5, 0.5])
    check("small tower keeps a spot 0.04 from a tower",
          [0.54, 0.5] in small_out)
    check("large tower drops that same spot",
          [0.54, 0.5] not in big_out)

    print()
    if _fails:
        print(f"FAILED {len(_fails)} case(s): {', '.join(_fails)}")
        sys.exit(1)
    print("all placement-avoid cases passed")


if __name__ == "__main__":
    main()
