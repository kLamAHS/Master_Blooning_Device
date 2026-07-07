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
        # income_model(round) -> rough cumulative cash available by that round.
        # Used only as a hard-discounted soft floor BEFORE the first confirmed
        # read, to reject absurd lows (e.g. '$4' at round 20).
        self._income = income_model

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

    def lower_bound(self, round_hint=None):
        """Best available lower bound on current cash, or None if nothing is
        known yet. The provable floor when seeded; otherwise a hard-discounted
        income estimate good only for rejecting absurd lows."""
        if self._floor is not None:
            return self._floor
        if round_hint is not None and self._income is not None:
            try:
                est = self._income(round_hint) - self._spent
            except Exception:
                return None
            # The income model assumes full pops; a struggling run has far
            # less, so discount hard and use it only as an absurd-low reject.
            return 0.5 * est if est > 0 else None
        return None

    def sane(self, read, confirm_fn=None, round_hint=None):
        """Return a cash value safe to act on. If `read` is implausibly far
        below the floor it is a misread: corroborate via confirm_fn (called
        lazily, only when the read looks bad, so the happy path pays nothing),
        and if it still reads low, substitute the floor. A plausible read --
        anything at or near/above the floor -- passes through unchanged, as
        does None (an unreadable frame: callers treat that as 'buy anyway')."""
        lb = self.lower_bound(round_hint)
        if read is not None and lb is not None and read < lb - self.margin:
            c = confirm_fn() if confirm_fn is not None else None
            if c is not None and c >= lb - self.margin:
                return c
            return int(lb)
        return read
