"""Provable lower bound on a run's current cash, to neutralize low OCR
misreads that would otherwise freeze the buy plan and leak the run.

This is the one piece of the cash pipeline that is PURE ARITHMETIC -- no
screen, no cv2, no numpy -- so it can be unit-tested offline
(tools/test_cash_floor.py), unlike the rest of mk.py's real-game code.

The idea: a wallet only ever RISES except through purchases, and every
purchase is reported via spend(). So

    floor = last_confirmed_read - (everything spent since)

is a *provable* lower bound on the current cash. A read far below the floor
is therefore a misread -- the usual culprit being a clipped leading digit
('$2,340' read as '340'), which reads low but perfectly "valid", so no
value-range check catches it. When that happens the guard substitutes the
floor, so the bot keeps buying up to what it provably has instead of
hoarding behind a phantom-broke wallet.

A correct read always passes through untouched: it sits at or above the
floor, so the guard never fires on it.
"""

# Slack (in $) allowed below the floor before a read is judged a misread.
# Covers the fast-forward cash animation and rounding between a spend and
# the next read; a real drop never exceeds it because every spend already
# lowered the floor by the price paid.
CASH_FLOOR_MARGIN = 50

# How much of the income-curve estimate (cumulative earned by the previous
# round, minus everything spent) counts as a PROVABLE lower bound on current
# cash. In CHIMPS this estimate is near-exact -- every bloon is popped or the
# run is over -- so the only reasons it could overstate real cash are (a) a
# spend the bot didn't record or (b) mid-round rounding. Halving it swallows
# both with margin, so a read below it is "severely off" (a clipped/garbled
# number), never a real balance. It is deliberately below 1.0 so it can only
# ever RESCUE an implausibly-low read, never inflate a merely-modest one.
INCOME_DISCOUNT = 0.5


class CashFloor:
    """Tracks a provable lower bound on current cash across a run.

    Usage (see mk.run_episode):
        floor = CashFloor(income_model=lambda r: earned_by(r, mode))
        floor.confirm(read_cash_confirmed(...))   # seed / re-anchor
        floor.spend(price)                         # after every purchase
        cash = floor.sane(read_cash(...),          # at every buy gate
                          confirm_fn=lambda: read_cash_confirmed(...),
                          round_hint=cur_round)
    """

    def __init__(self, margin=CASH_FLOOR_MARGIN, income_model=None):
        self._floor = None          # provable lower bound, or None until seeded
        self._spent = 0             # cumulative spend this run
        self.margin = margin
        # income_model(round) -> cumulative cash available by that round. In
        # CHIMPS this is the exact pops-only curve, so income(r-1) - spent is
        # a hard (discounted) lower bound on current cash -- used BOTH before
        # the first confirmed read (reject absurd lows like '$4' at round 20)
        # AND after, to lift a provable floor that has gone stale-low because
        # the box rarely corroborates (the field failure: OCR keeps reading
        # low while the wallet has clearly climbed).
        self._income = income_model
        self._sub_streak = 0        # consecutive substituted (low-misread) reads

    @property
    def value(self):
        """The current provable floor, or None if not seeded yet."""
        return self._floor

    @property
    def spent(self):
        return self._spent

    def spend(self, amount):
        """Report a purchase -- the only way cash falls. Lower the floor by
        exactly what was paid so it stays a valid lower bound."""
        if amount:
            self._spent += amount
            if self._floor is not None:
                self._floor = max(0, self._floor - amount)

    def confirm(self, level):
        """Raise the floor from a corroborated read. Only ever RAISES: a low
        misread can't lower it (max ignores it) and real spends already
        lowered it via spend(). Pass None (an unreadable frame) to no-op."""
        if level is not None:
            self._floor = max(self._floor or 0, level)

    def _income_floor(self, round_hint):
        """An OCR-INDEPENDENT lower bound on current cash from the income
        curve: entering round `round_hint` we have provably earned at least
        the cumulative cash through the PREVIOUS round, minus everything
        spent, discounted (INCOME_DISCOUNT) so unrecorded spend / mid-round
        rounding keep it a true under-estimate. None when no curve or round."""
        if round_hint is None or self._income is None:
            return None
        try:
            est = self._income(round_hint - 1) - self._spent
        except Exception:
            return None
        return INCOME_DISCOUNT * est if est > 0 else None

    def lower_bound(self, round_hint=None):
        """Best available lower bound on current cash, or None if nothing is
        known yet. Two independent bounds, both provable, so the higher wins:
        the spend-tracked floor (last confirmed read minus spend since) and
        the income-curve estimate (cumulative earned by last round minus
        spend, discounted). The income bound MATTERS most when the floor has
        gone stale-low -- the box rarely corroborates, so the floor sits near
        its seed while the wallet has clearly climbed. The discount is what
        keeps it from ever inflating a merely-modest real read (the old
        $443->$542 churn): it only ever rescues a read that is *severely*
        below what we have provably earned."""
        inc = self._income_floor(round_hint)
        if self._floor is not None:
            return max(self._floor, inc) if inc is not None else self._floor
        return inc

    def sane(self, read, confirm_fn=None, round_hint=None):
        """Return a cash value safe to act on. If `read` is implausibly far
        below the floor it is a misread: corroborate via confirm_fn (called
        lazily, only when the read looks bad, so the happy path pays nothing),
        and if it still reads low, substitute the floor. A plausible read --
        anything at or near/above the floor -- passes through unchanged, as
        does None (an unreadable frame: callers treat that as 'buy anyway').

        The floor has TWO parts, and a low read is resolved against them
        differently, because they are not equally trustworthy:

          * the PROVABLE floor (a confirmed read minus tracked spend) is a
            real lower bound -- a read below it is a genuine misread;
          * the INCOME floor (the earned-curve estimate) is only a MODEL. It
            inflates whenever the curve overshoots this run's real income OR
            the ROUND is misread high -- and then it overrides a *correct*
            read and freezes the buy plan. The field failure: a blind stretch
            re-synced the counter 27 -> 50, the round-50 curve claimed ~$14k
            "provable", and a correct $4411 read was replaced by it every
            frame, so the bot "coasted rich" and died.

        So a read below the income estimate but corroborated by a second,
        independent read is TRUSTED: two live sensors outvote a model. Only a
        read below the PROVABLE floor is substituted. (A read below the income
        estimate that CANNOT be corroborated still falls back to it -- the
        original clipped-misread defense for a box that has gone stale-low.)

        Tracks a run of consecutive substitutions: an INTERMITTENT misread
        resets it (a good read lands in between), but a box that has broken
        outright (reads a constant '$1' while cash really climbs) racks the run
        up -- see stuck(), which the caller uses to trigger a recalibration
        rather than freezing the whole plan behind a phantom-low wallet."""
        if read is None:
            return read
        prov = self._floor                          # provable: confirmed-spend
        inc = self._income_floor(round_hint)        # model estimate (soft)
        lb = prov
        if inc is not None:
            lb = inc if lb is None else max(lb, inc)
        if lb is None or read >= lb - self.margin:
            self._sub_streak = 0                    # trusted read
            return read
        c = confirm_fn() if confirm_fn is not None else None
        if c is not None and c >= lb - self.margin:
            self._sub_streak = 0        # confirm lands high: use it
            return c
        if c is not None \
                and abs(c - read) <= max(2 * self.margin, 0.12 * max(read, 1)):
            # Two independent reads AGREE on a value below the floor. A model
            # estimate can't outvote two live sensors, so believe the reads --
            # unless they fall below the PROVABLE floor, which (unlike the
            # income curve) is a real bound and still wins.
            if prov is None or read >= prov - self.margin:
                self._sub_streak = 0
                return read
            self._sub_streak += 1
            return int(prov)
        self._sub_streak += 1           # uncorroborated low read: hold floor
        return int(lb)

    def stuck(self, n=12):
        """True once `n` reads in a row have been substituted with no good read
        between them -- the signature of a broken box (not the odd misread),
        so the caller should recalibrate the counter rather than trust it."""
        return self._sub_streak >= n

    def reset_stuck(self):
        self._sub_streak = 0
