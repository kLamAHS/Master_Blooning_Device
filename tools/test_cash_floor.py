"""Offline unit test for the provable cash-floor guard (cashguard.CashFloor).

The rest of mk.py's cash pipeline needs a real screen + cv2 and can't run
here, but the guard that actually decides "is this low read a misread?" is
pure arithmetic -- so it IS testable, and this is where we prove the fix:
a clipped/low read can never make the bot think it's broke when it provably
isn't.

    python tools/test_cash_floor.py        # exits non-zero on any failure
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from cashguard import CASH_FLOOR_MARGIN, CashFloor    # noqa: E402

# A stand-in income model roughly like meta.earned_by (round -> cash avail).
INCOME = {3: 850, 6: 650, 20: 5300}


def income_model(r):
    return INCOME.get(r, 650)


_fails = []


def check(name, cond):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}")
    if not cond:
        _fails.append(name)


def main():
    print("cash-floor guard:")

    # --- The headline case: a clipped misread must NOT read as broke. -------
    f = CashFloor(income_model=income_model)
    f.confirm(850)                       # seeded from a good read: $850
    check("seed sets floor", f.value == 850)
    f.spend(200)                         # bought a $200 tower
    check("spend lowers floor by price", f.value == 650)
    # Now the OCR clips '$2,340'-style and returns a low junk value. With no
    # corroboration it must fall back to the floor, never the misread.
    check("clipped low read -> floor, not the misread",
          f.sane(4) == 650)
    check("clipped low read is NOT trusted", f.sane(4) != 4)
    # An unreadable frame stays None (callers treat that as 'buy anyway').
    check("None read passes through", f.sane(None) is None)
    # A correct read at/above the floor passes straight through.
    check("correct read passes through unchanged", f.sane(1500) == 1500)
    check("read within margin below floor passes through",
          f.sane(650 - CASH_FLOOR_MARGIN + 1) == 650 - CASH_FLOOR_MARGIN + 1)

    # --- Corroboration: a low first read backed by a plausible confirm wins.
    f = CashFloor(income_model=income_model)
    f.confirm(850)
    f.spend(200)                         # floor 650
    # First read low, but read_cash_confirmed agrees it's a legit-ish 700
    # (income ticked): trust the corroborated value, not the stale floor.
    check("low read corroborated by plausible confirm is accepted",
          f.sane(700, confirm_fn=lambda: 700) == 700)
    # First read low AND the confirm is also junk-low: hold the floor.
    check("low read with junk-low confirm falls back to floor",
          f.sane(3, confirm_fn=lambda: 2) == 650)

    # --- confirm() only ever RAISES; spend() only ever LOWERS. --------------
    f = CashFloor(income_model=income_model)
    f.confirm(850)
    f.confirm(700)                       # a lower (mis)read must not lower it
    check("confirm never lowers the floor", f.value == 850)
    f.confirm(None)                      # unreadable frame: no-op
    check("confirm(None) is a no-op", f.value == 850)
    f.confirm(2000)                      # income grew: floor rises
    check("confirm raises the floor on income", f.value == 2000)
    f.spend(500)
    check("spend after raise", f.value == 1500)
    f.spend(1500)
    check("floor never goes negative", f.value == 0)

    # --- A real income read after spending is accepted (no false positive). -
    f = CashFloor(income_model=income_model)
    f.confirm(850)
    f.spend(210)                         # floor 640
    check("real read just below floor (margin) accepted",
          f.sane(645) == 645)
    f.confirm(645)                       # re-anchor to the true post-spend level
    check("re-anchor after spend", f.value == 645)

    # --- Before any confirmed read: the income model rejects absurd lows. ---
    f = CashFloor(income_model=income_model)   # floor is None (unseeded)
    check("unseeded: absurd low for the round is rejected",
          f.sane(4, round_hint=20) != 4)       # $4 at r20 -> discounted floor
    check("unseeded: plausible read for the round passes",
          f.sane(3000, round_hint=20) == 3000)
    check("unseeded with no round hint passes any read through",
          f.sane(4) == 4)

    # --- Stale-LOW provable floor + income: the income bound still catches a
    # misread the spent-down floor would miss (the '$1' case the user hit). ---
    f = CashFloor(income_model=income_model)   # income_model(20) = 5300
    f.confirm(5000)                            # floor high...
    f.spend(4980)                              # ...then spent down to 20
    check("floor is stale-low after heavy spending", f.value == 20)
    # A '$1' misread: the spent-down floor alone (20) wouldn't reject it, but
    # the income bound 0.5*(5300-4980)=160 does.
    check("income bound catches a $1 misread past a stale-low floor",
          f.sane(1, round_hint=20) == 160)
    # A genuine low read consistent with the income bound still passes.
    check("read at the income bound passes through",
          f.sane(200, round_hint=20) == 200)
    # Without the round hint only the (stale-low) provable floor is in play, so
    # it CAN'T catch the $1 -- which is exactly why the real sane_cash always
    # passes round_hint=last_round.
    check("no round hint: stale-low floor alone lets the $1 through",
          f.sane(1) == 1)

    print()
    if _fails:
        print(f"FAILED {len(_fails)} case(s): {', '.join(_fails)}")
        sys.exit(1)
    print("all cash-floor cases passed")


if __name__ == "__main__":
    main()
