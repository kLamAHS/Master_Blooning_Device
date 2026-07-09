"""Offline unit test for parse_round (mk.py) -- the round-counter text parser.

mk.py imports cv2/mss/pyautogui and can't be imported here, but parse_round
is pure string logic, so we lift just that function out of the source with the
`ast` module and exercise it. This is where we prove the "counter went dark on
a perfectly visible number" fix: when the '/' between the current round and the
total fades, tesseract fuses '34/100' into '34100'/'341100', which used to
parse to a >200 number -> None -> the bot thought the HUD was lost and froze
its plan. With the HUD total known, the fused digits are recovered.

    python tools/test_round_parse.py        # exits non-zero on any failure
"""

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load(func_name):
    src = (REPO / "mk.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            ns = {}
            exec(ast.get_source_segment(src, node), ns)
            return ns[func_name]
    raise SystemExit(f"{func_name} not found in mk.py")


parse_round = _load("parse_round")

_fails = []


def check(name, cond):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}")
    if not cond:
        _fails.append(name)


def main():
    print("parse_round:")
    # --- the clean cases still work exactly as before ----------------------
    check("clean 'current/total' takes the current", parse_round("34/100") == 34)
    check("clean with a different total", parse_round("13/40") == 13)
    check("a bare number passes through", parse_round("50") == 50)
    check("99/100 -> 99", parse_round("99/100") == 99)

    # --- the headline fix: a faded/misread '/' fuses the two numbers, and a
    # known total recovers the current round instead of reading dark. --------
    check("dropped '/' ('34100') recovered with total 100",
          parse_round("34100", total=100) == 34)
    check("'/' misread as a digit ('341100') recovered",
          parse_round("341100", total=100) == 34)
    check("'/' misread as 7 ('347100') recovered",
          parse_round("347100", total=100) == 34)
    check("single-digit round fused with total ('6100') -> 6",
          parse_round("6100", total=100) == 6)
    check("round 100 of 100 ('100100') -> 100",
          parse_round("100100", total=100) == 100)

    # --- safety: without a total hint, an implausible fusion stays None (no
    # guessing), and junk is still rejected. --------------------------------
    check("no total hint -> a fused number is NOT guessed",
          parse_round("34100") is None)
    check("empty text -> None", parse_round("") is None)
    check("non-numeric -> None", parse_round("abc") is None)
    check("a number that doesn't end in the total is not mangled",
          parse_round("1234", total=100) is None)
    # freeplay passes total=None and its counter is a bare number; a legit
    # sub-200 freeplay round still parses.
    check("freeplay bare round (total=None) still parses", parse_round("150") == 150)

    print()
    if _fails:
        print(f"FAILED {len(_fails)} case(s): {', '.join(_fails)}")
        sys.exit(1)
    print("all parse_round cases passed")


if __name__ == "__main__":
    main()
