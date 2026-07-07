"""Pure placement geometry for the executor -- deciding where act_place is
allowed to click. Kept out of mk.py (which needs cv2/pyautogui and can't be
imported in CI) so the "never click a spot we know is taken" logic can be
unit-tested offline (tools/test_placement_avoid.py).
"""


def free_placement_spots(candidates, avoid, min_d, pool, spot):
    """Which candidate spots act_place should actually try, given points to
    AVOID (towers already placed this run, plus any a probe caught a monkey
    sitting on).

    - Drop every candidate within `min_d` of an avoid point.
    - If that leaves nothing -- the whole planned neighborhood is taken --
      RELOCATE to the single nearest genuinely-free point from `pool` (the
      map's mask points), so the bot never clicks a spot it knows is occupied
      and never stacks a tower on another.
    - Only when the pool is fully taken too (a truly full map) fall back to
      the raw candidates.

    `candidates`, `pool` are lists of [x, y]; `avoid` a list of [x, y];
    `min_d` the minimum spacing (screen fraction); `spot` the planned target
    used to break ties toward the closest relocation.
    """
    if not avoid:
        return candidates

    def free(c):
        return all((c[0] - a[0]) ** 2 + (c[1] - a[1]) ** 2 >= min_d ** 2
                   for a in avoid)

    far = [c for c in candidates if free(c)]
    if far:
        return far
    free_pool = sorted(
        (p for p in (pool or []) if free(p)),
        key=lambda p: (p[0] - spot[0]) ** 2 + (p[1] - spot[1]) ** 2)
    if free_pool:
        return [list(free_pool[0])]
    return candidates
