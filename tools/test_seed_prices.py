"""Offline test for the shipped seed price table (tower_prices.json) and the
seed-aware price logic in mk.py.

The bot used to be at the mercy of OCR for every upgrade/base cost: a misread
poisoned the price book, and a FAILED read left the cost unknown, so buys
stalled or mis-planned. tower_prices.json is an accurate BTD6 cost guide keyed
exactly like the learned book ("{difficulty}:{tower}[:{path}:{tier}]"); mk.py
now (1) falls back to it when a price is unlearned/unread (learned > seed >
None) and (2) rejects a live read that deviates from it too far to be anything
but a misread. This test proves the TABLE is well-formed and CHIMPS-correct,
and lifts mk.py's pure price logic out via `ast` (mk.py needs cv2, so it can't
be imported) to prove the fallback + guard behave.

    python tools/test_seed_prices.py        # exits non-zero on any failure
"""

import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SEED = json.loads((REPO / "tower_prices.json").read_text())

_fails = []


def check(name, cond):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}")
    if not cond:
        _fails.append(name)


class _NoWrite:
    """Stand-in for PRICES_PATH so record_price's persist is a no-op in test."""
    def write_text(self, *a, **k):
        pass


def _lift(src, names, extra=None):
    """Exec the named top-level defs/classes from a source string, injecting
    the module globals they close over (SEED_PRICES, price_key, etc.)."""
    tree = ast.parse(src)
    ns = {"dbg": lambda *a, **k: None, "json": json,
          "PRICES_PATH": _NoWrite(), "PRICE_DIFFICULTY": "hard",
          "PRICES_SRC": {}, "SEED_PRICES": SEED, **(extra or {})}
    # pull the module-level scalar constants the funcs reference
    for line in src.splitlines():
        for c in ("SEED_BUY_LO", "SEED_BUY_HI"):
            if line.startswith(c):
                ns[c] = float(line.split("=")[1].split("#")[0])
    wanted = [n for n in tree.body
              if isinstance(n, (ast.FunctionDef, ast.ClassDef))
              and n.name in names]
    mod = ast.Module(body=wanted, type_ignores=[])
    exec(compile(mod, "<lift>", "exec"), ns)
    return ns


def main():
    print("seed price table + logic:")

    # ---- table shape ------------------------------------------------------
    diffs = {k.split(":")[0] for k in SEED}
    check("all four difficulty columns present",
          diffs == {"easy", "medium", "hard", "impoppable"})
    check("every value is a positive multiple of 5 (matches record_price)",
          all(isinstance(v, int) and v > 0 and v % 5 == 0
              for v in SEED.values()))
    # core towers each have a base + full 3x5 upgrade grid on the hard column
    for tw in ("dart", "tack", "ice", "boomerang", "bomb", "super", "village"):
        base = SEED.get(f"hard:{tw}")
        grid = [f"hard:{tw}:{p}:{t}" for p in range(3) for t in range(1, 6)]
        check(f"{tw}: hard base + 15 upgrade tiers all seeded",
              base is not None and all(k in SEED for k in grid))

    # ---- CHIMPS uses the 'hard' column; spot-check known costs ------------
    known_hard = {
        "hard:dart": 215, "hard:dart:2:3": 620,          # Crossbow
        "hard:tack": 280, "hard:tack:2:5": 21600,         # The Tack Zone
        "hard:ice": 430, "hard:ice:0:5": 30240,           # Super Brittle
        "hard:boomerang:2:4": 2915,                       # MOAB Press
        "hard:dart:1:1": 110,                             # Quick Shots
    }
    for k, want in known_hard.items():
        check(f"CHIMPS(hard) {k} == {want}", SEED.get(k) == want)

    # hard should be ~1.08x medium (rounded to $5) -- the CHIMPS multiplier
    ratios = []
    for k, v in SEED.items():
        if k.startswith("hard:"):
            mv = SEED.get("medium:" + k[len("hard:"):])
            if mv:
                ratios.append(v / mv)
    check("hard column averages ~1.08x medium (CHIMPS multiplier)",
          ratios and abs(sum(ratios) / len(ratios) - 1.08) < 0.02)

    # ---- mk.py fallback + guard logic (lifted, no cv2) --------------------
    src = (REPO / "mk.py").read_text()
    ns = _lift(src, {"_PriceBook", "price_of", "learned_price", "price_key",
                     "record_price"})
    PriceBook = ns["_PriceBook"]

    # PRICES (a _PriceBook) is the shared state price_of/learned_price/
    # record_price all close over; rebind it per sub-case.
    def fresh(learned=None):
        pb = PriceBook(dict(learned or {}))
        ns["PRICES"] = pb
        ns["PRICES_SRC"] = {}
        return pb

    # fallback: PRICES.get and price_of resolve learned > seed > None
    fresh()
    check("PRICES.get falls back to the seed when unlearned",
          ns["PRICES"].get("hard:dart:2:3") == 620)
    check("price_of falls back to the seed", ns["price_of"]("tack", 2, 5) == 21600)
    fresh({"hard:dart:2:3": 527})       # a learned (discounted) value
    check("a learned value wins over the seed",
          ns["PRICES"].get("hard:dart:2:3") == 527)
    check("unknown key with no seed returns default",
          ns["PRICES"].get("hard:nonesuch:0:1", "X") == "X")
    check("seeds never leak into the persisted dict",
          json.loads(json.dumps(ns["PRICES"])) == {"hard:dart:2:3": 527})

    # learned_price: the GATES see the VERIFIED value only, never the seed
    # (an over-stated seed must not refuse an affordable buy).
    fresh()
    check("learned_price returns None for an unlearned (seed-only) tier",
          ns["learned_price"]("dart", 2, 3) is None)
    fresh({"hard:dart:2:3": 600})
    check("learned_price returns the verified value once learned",
          ns["learned_price"]("dart", 2, 3) == 600)

    # record_price: an UNVERIFIED read never overrides the guide; a verified
    # BUY may teach a real (discounted/patched) price within a wide band.
    rp = ns["record_price"]
    K = "hard:dart:2:3"           # seed 620
    pb = fresh()
    rp(K, 650, src="short")       # unverified red misread
    check("an unverified read is IGNORED when a seed exists (no shadow)",
          dict.get(pb, K) is None and pb.get(K) == 620)
    pb = fresh()
    rp(K, 650, src="seen")        # unverified green misread
    check("an unverified 'seen' read is likewise ignored over a seed",
          dict.get(pb, K) is None)
    pb = fresh()
    rp(K, 525, src="buy")         # a real ~15% discount, cash-verified (mult 5)
    check("a cash-verified discount buy IS learned (beats the seed)",
          dict.get(pb, K) == 525)
    pb = fresh()
    rp(K, 690, src="buy")         # a real +11% balance patch, cash-verified
    check("a cash-verified rebalance-UP buy is learned (self-correction)",
          dict.get(pb, K) == 690)
    pb = fresh()
    rp(K, 6200, src="buy")        # a corrupted cash delta (10x)
    check("a wildly-off buy delta (corrupted cash read) is rejected",
          dict.get(pb, K) is None)
    pb = fresh()
    rp("hard:nonesuch:0:1", 355, src="short")   # no seed -> original behavior
    check("a read with NO seed is recorded as before (any src)",
          dict.get(pb, "hard:nonesuch:0:1") == 355)

    print()
    if _fails:
        print(f"FAILED {len(_fails)} case(s): {', '.join(_fails)}")
        sys.exit(1)
    print("all seed-price cases passed")


if __name__ == "__main__":
    main()
