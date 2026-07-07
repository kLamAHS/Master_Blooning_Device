"""Offline unit test for the progress-dashboard aggregation (plot_progress).

The charts are drawn by browser JS we can't run in CI, but every NUMBER they
draw is computed by plain Python here -- so that is what we test: the episode
series, running best, death histogram, outcomes, win detection, and that the
emitted HTML embeds the data safely (no </script> break-out) and carries a
no-JS table twin.

    python tools/test_plot_progress.py       # exits non-zero on any failure
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import plot_progress as pp                       # noqa: E402

_fails = []


def check(name, cond):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}")
    if not cond:
        _fails.append(name)


def run(i, mode, outcome, final, target=100, extra=None):
    row = {"time": f"2026-07-07T00:{i:02d}:00", "map": "monkey_meadow",
           "game_mode": mode, "difficulty": "hard", "target_round": target,
           "final_round": final, "outcome": outcome,
           "strategy": {"kind": "meta", "explore": 0.2}}
    if extra:
        row.update(extra)
    return row


def main():
    print("progress dashboard:")

    # is_win: victory (solve/deploy) AND survived (farm) both count.
    check("is_win accepts victory and survived",
          pp.is_win("victory") and pp.is_win("survived"))
    check("is_win rejects defeat/hud_lost",
          not pp.is_win("defeat") and not pp.is_win("hud_lost"))

    rows = [
        run(1, "chimps", "defeat", 8),
        run(2, "chimps", "defeat", 12),
        run(3, "chimps", "hud_lost", 12),
        run(4, "chimps", "crashed", None),      # no final_round
        run(5, "chimps", "defeat", 41),
        run(6, "chimps", "victory", 100,
            extra={"lives_by_round": {"1": 1, "50": 1, "100": 1},
                   "cash_by_round": {"1": 650, "10": 1700, "20": 5300}}),
    ]
    g = pp.summarize_group(rows)

    check("episode count", g["episodes"] == 6)
    check("win count (victory)", g["wins"] == 1)
    check("best round is the deepest", g["bestRound"] == 100)
    check("target picked up from rows", g["target"] == 100)
    # running max is monotonic non-decreasing and ends at the best
    rm = g["runningMax"]
    check("running max monotonic", all(b >= a for a, b in zip(rm, rm[1:])))
    check("running max ends at best", rm[-1] == 100)
    check("running max after crash holds (no None)", rm[3] == 12)
    # death histogram buckets by 5; the crash (final_round None) is excluded
    hist = {d["bucket"]: d["count"] for d in g["deathHist"]}
    check("death bucket r5-9 has the round-8 run", hist.get(5) == 1)
    check("death bucket r10-14 has both round-12 runs", hist.get(10) == 2)
    check("round-41 run buckets at r40", hist.get(40) == 1)
    check("round-100 win buckets at r100", hist.get(100) == 1)
    check("crashed run (no round) absent from histogram",
          sum(hist.values()) == 5)
    # outcomes counted, most-common first, with severity tokens
    oc = {d["outcome"]: d for d in g["outcomes"]}
    check("defeat counted 3", oc["defeat"]["count"] == 3)
    check("victory tagged good", oc["victory"]["sev"] == "good")
    check("crashed tagged warning", oc["crashed"]["sev"] == "warning")
    # best run = the deepest, with its economy series parsed + sorted
    check("best run is the round-100 run", g["bestRun"]["final"] == 100)
    check("best run cash series parsed & sorted",
          [p["r"] for p in g["bestRun"]["cash"]] == [1, 10, 20])
    check("best run lives series parsed", len(g["bestRun"]["lives"]) == 3)

    # --- grouping: two slices, ordered by episode count -------------------
    mixed = rows + [run(7, "standard", "victory", 40, target=40)]
    payload = pp.build_payload(mixed, {})
    check("two map/mode slices", len(payload["groups"]) == 2)
    check("default slice is the busiest (chimps, 6 eps)",
          payload["default"] == "monkey_meadow · chimps")
    check("total episodes", payload["totalEpisodes"] == 7)

    # --- cmd_play rows (different, smaller schema) don't crash ------------
    play = [{"time": "t", "plan": "plans/mm_easy.json", "final_round": 40,
             "final_lives": 100, "outcome": "victory",
             "towers": []}]
    gp = pp.summarize_group(play)
    check("cmd_play row summarized without map/strategy", gp["episodes"] == 1)
    pl = pp.build_payload(play, {})
    check("cmd_play grouped under plan:<name>",
          any(k.startswith("plan:") for k in pl["groups"]))

    # --- ladder from progress.json ---------------------------------------
    prog = {"monkey_meadow|easy|standard": {"episodes": 5, "best_round": 40,
                                            "beaten": True, "beaten_at": "x"},
            "monkey_meadow|hard|chimps": {"episodes": 9, "best_round": 63,
                                          "beaten": False}}
    ladder = pp.build_ladder(prog)
    check("ladder built for the map", ladder and ladder[0]["map"] == "monkey_meadow")
    names = {r["name"]: r for r in ladder[0]["rungs"]}
    check("easy rung marked beaten", names["easy"]["beaten"] is True)
    check("chimps rung shows best 63 vs target",
          names["chimps"]["best"] == 63 and names["chimps"]["target"] == 100)

    # --- HTML emission is safe + complete --------------------------------
    html = pp.render_html(payload)
    check("emits a full HTML document", html.strip().startswith("<!DOCTYPE html>"))
    check("embeds the data payload", '"episodes":6' in html or '"episodes": 6' in html)
    # the JSON literal is `const DATA = {...};` on its own line, immediately
    # before its closing </script>. render_html escapes every "</" to "<\/", so
    # the literal itself must contain no "</" at all (no early script break-out).
    json_literal = html.split("const DATA =", 1)[1].split("\n</script>", 1)[0]
    check("no </ break-out inside the embedded JSON", "</" not in json_literal)
    check("carries a no-JS table twin", "<noscript>" in html and "<table" in html)
    check("references the validated accent blue", "#2a78d6" in html)
    check("is theme-aware (dark override present)",
          "data-theme=dark" in html and "prefers-color-scheme:dark" in html)

    # An untrusted-looking map name can't inject markup into the embed.
    evil = [run(1, "chimps", "defeat", 5)]
    evil[0]["map"] = "</script><script>alert(1)</script>"
    h2 = pp.render_html(pp.build_payload(evil, {}))
    check("script injection via map name is neutralized",
          "<script>alert(1)</script>" not in h2)

    # --- empty input: no crash, graceful empty state ---------------------
    empty = pp.build_payload([], {})
    check("empty payload has no default slice", empty["default"] is None)
    eh = pp.render_html(empty)
    check("empty dashboard still renders a document",
          eh.strip().startswith("<!DOCTYPE html>"))

    print()
    if _fails:
        print(f"FAILED {len(_fails)} case(s): {', '.join(_fails)}")
        sys.exit(1)
    print("all progress-dashboard cases passed")


if __name__ == "__main__":
    main()
