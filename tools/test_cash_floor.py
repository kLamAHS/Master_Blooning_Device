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

    # --- The discount keeps a MODEST real read from being inflated: a good
    # $443 read, when the curve says we've only earned ~$650 by last round, is
    # above the discounted income floor (0.5*650=325), so it passes untouched
    # (the old $443->$542 churn is gone). ------------------------------------
    f = CashFloor(income_model=income_model)   # income_model(19) = 650 (default)
    f.confirm(435)                             # the field scenario: floor 435
    check("a modest real read is NOT inflated by the income model",
          f.sane(443, round_hint=20) == 443)
    # The provable floor still catches a genuine low misread, as before.
    f.confirm(3000)
    check("the seeded floor still catches a real $1 misread",
          f.sane(1, round_hint=20) == 3000)

    # --- NEW: the accurate income curve rescues a STALE-LOW provable floor. --
    # The field failure: OCR seldom corroborates, so the floor sits near its
    # seed while the wallet (by the pops-only CHIMPS curve) has climbed into
    # the thousands. A clipped '$4xx' read must NOT read as broke when we have
    # provably earned far more. income(round-1)=income(24)=5226 earned, $300
    # spent -> ~4926 available, discounted 0.5 -> ~2463.
    curve = CashFloor(income_model=lambda r: {24: 5226, 25: 5561}.get(r, 0))
    curve.confirm(650)          # only ever got the one good read (at the start)
    curve.spend(300)            # provable floor now 350 -- but we're at round 25
    out = curve.sane(432, round_hint=25)
    check("accurate income rescues a stale-low floor (not read as broke)",
          out > 2000)
    check("...but a plausible read for the round still passes untouched",
          curve.sane(4800, round_hint=25) == 4800)
    # And the income floor NEVER exceeds what we've provably earned minus
    # spend: a real broke moment (we really did spend down to ~$120) is still
    # believed once corroborated, not overridden into a phantom balance.
    check("a corroborated low read at/above the income floor is trusted",
          curve.sane(2600, confirm_fn=lambda: 2600, round_hint=25) == 2600)

    # --- NEW: a CORRECT read must beat an INFLATED income floor. The curve
    # (or a round misread high) can put the model estimate far above real cash
    # -- the field failure re-synced the counter to round 50 while really near
    # round 28, so the round-50 curve claimed ~$14k "provable" and overrode a
    # correct $4411 read every frame, coasting the run to death. Two agreeing
    # live reads must outvote the model. ------------------------------------
    f = CashFloor(income_model=lambda r: {49: 34559, 50: 37575}.get(r, 0))
    f.confirm(4400)                    # the box reads the real ~4400
    # income floor at 'round 50' = 0.5*earned_by(49) = ~17000, far above real
    check("a corroborated correct read beats an inflated income floor",
          f.sane(4411, confirm_fn=lambda: 4408, round_hint=50) == 4411)
    # ...but an UNcorroborated low read (a real clip while the box is stale)
    # is still rescued by the (income) floor -- the original defense.
    f2 = CashFloor(income_model=lambda r: {49: 34559}.get(r, 0))
    f2.confirm(650)                    # stale seed; the box then clips low
    check("an uncorroborated low read still falls back to the income floor",
          f2.sane(432, round_hint=50) > 5000)
    # A corroborated read BELOW the provable floor is NOT trusted: the provable
    # floor (confirmed minus spend) is a real bound, unlike the curve.
    f3 = CashFloor()
    f3.confirm(8000)
    f3.spend(1000)                     # provable floor 7000
    check("a corroborated read below the PROVABLE floor holds the floor",
          f3.sane(800, confirm_fn=lambda: 795) == 7000)

    # --- round/cash cross-check: a counter that jumps to a round the wallet
    # can't support is a misread, not real progress. -------------------------
    from meta import round_supported_by_cash          # noqa: E402
    check("phantom 27->50 jump on ~$1k is rejected (cash can't support it)",
          not round_supported_by_cash(50, 1011, 6185, "chimps"))
    check("a real round-28-ish wallet supports round 28",
          round_supported_by_cash(28, 900, 6185, "chimps"))
    check("non-CHIMPS is never vetoed (curve too rough)",
          round_supported_by_cash(50, 100, 0, "standard"))
    check("unknown cash is never vetoed",
          round_supported_by_cash(50, None, 6185, "chimps"))

    # The ep-18 fix: at the re-sync instant the live confirmed read was None,
    # so the check saw cash=None, DIDN'T veto, and the phantom 27->50 latched.
    # The caller now falls back to the PROVABLE floor (confirmed-minus-spend)
    # when the fresh read is None -- floor + spent is a real lower bound on
    # earnings that does NOT depend on the suspect new round, so the SAME jump
    # that slips through on None is correctly vetoed on the floor. This mirrors
    # `xcheck_cash = read_cash_confirmed(...) or cash_floor.value` in run_episode.
    g = CashFloor(income_model=lambda r: {49: 34559}.get(r, 0))
    g.confirm(2400)                 # last good read was ~$2.4k (really round 27)
    g.spend(6185)                   # provable floor now 0; spent tracked at 6185
    check("None confirm alone leaves the phantom jump un-vetoed (the old bug)",
          round_supported_by_cash(50, None, g.spent, "chimps"))
    check("...but the provable floor as fallback vetoes it (the fix)",
          not round_supported_by_cash(50, g.value, g.spent, "chimps"))
    # A real, SMALL catch-up jump still passes on the same stale floor, because
    # the loose 0.4 tolerance scales with the (much smaller) jump target.
    check("a real small catch-up jump still passes on a stale floor",
          round_supported_by_cash(30, g.value, g.spent, "chimps"))

    # --- stuck(): a BROKEN box (constant low read) is distinguished from the
    # odd intermittent misread, so the caller can recalibrate not freeze. -----
    f = CashFloor(income_model=income_model)
    f.confirm(3000)                            # floor $3000
    check("not stuck initially", not f.stuck())
    for _ in range(11):
        f.sane(1)                              # box reads a constant junk $1
    check("11 low reads: not yet flagged stuck (default n=12)", not f.stuck())
    f.sane(1)
    check("12 low reads in a row -> stuck (box is broken)", f.stuck())
    f.sane(2950)                               # one good read...
    check("a single good read clears the stuck streak", not f.stuck())
    # intermittent misreads (a good read between them) never trip stuck
    f2 = CashFloor()
    f2.confirm(3000)
    for _ in range(30):
        f2.sane(1)
        f2.sane(2950)
    check("intermittent misreads never trip stuck", not f2.stuck())
    f.reset_stuck()
    check("reset_stuck clears it", not f.stuck())

    print()
    if _fails:
        print(f"FAILED {len(_fails)} case(s): {', '.join(_fails)}")
        sys.exit(1)
    print("all cash-floor cases passed")


if __name__ == "__main__":
    main()
