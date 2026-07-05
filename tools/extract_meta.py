"""Distill research/btd6_meta_research_v55.xlsx into meta_knowledge.json.

The spreadsheet is the human-readable source of truth for the meta; this
script turns it into the machine-readable knowledge base the bot loads.
Run it again whenever the spreadsheet gets a new version:

    pip install openpyxl        # only needed for this script, not the bot
    python tools/extract_meta.py [path/to/research.xlsx]

Two kinds of data get merged:

1. SHEET DATA -- scores, roles, partners, threat rounds, notes. Read
   straight from the workbook so a re-ranked spreadsheet re-ranks the bot.
2. CURATED MAPPINGS (the dicts below) -- the spreadsheet names upgrades
   the way humans do ("205 Tack Zone", "Glue Storm"); the bot needs path
   indices ([top, mid, bot] = [0, 1, 2]) and tier numbers. Those mappings
   are game facts that only change when Ninja Kiwi reworks a tower, so
   they live here in code rather than in the sheet.
"""

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_XLSX = REPO / "research" / "btd6_meta_research_v55.xlsx"
OUT_PATH = REPO / "meta_knowledge.json"

# Spreadsheet row label -> bot tower key (TOWER_HOTKEYS names in mk.py).
SHEET_TOWER_KEYS = {
    "dart": "dart", "boomerang": "boomerang", "bomb": "bomb",
    "tack": "tack", "ice": "ice", "glue": "glue", "sniper": "sniper",
    "sub": "sub", "buccaneer": "buccaneer", "heli": "heli",
    "mortar": "mortar", "dartling": "dartling", "wizard": "wizard",
    "super": "super", "ninja": "ninja", "alchemist": "alchemist",
    "druid": "druid", "village": "village", "engineer": "engineer",
    "beast": "beast",
}

# Meta build templates per tower: which main path to push and which
# crosspath to take, weighted by how central the build is in the sheet's
# Meta Matrix / Tower Roles. Path indices: 0=top, 1=middle, 2=bottom.
# These encode the sheet's shorthand ("205 Tack", "025 Wizard", "013
# Glue") as data the genome sampler can draw from.
BUILDS = {
    "dart":      [{"main": 2, "cross": 0, "weight": 0.5, "label": "Crossbow Master line (2-0-5)"},
                  {"main": 1, "cross": 0, "weight": 0.3, "label": "Fan Club line (2-5-0)"},
                  {"main": 0, "cross": 2, "weight": 0.2, "label": "Juggernaut line (5-0-2)"}],
    "boomerang": [{"main": 2, "cross": 1, "weight": 0.6, "label": "MOAB Press (0-2-4)"},
                  {"main": 1, "cross": 0, "weight": 0.4, "label": "Perma Charge line (2-5-0)"}],
    "bomb":      [{"main": 1, "cross": 0, "weight": 0.6, "label": "MOAB Assassin line (2-4-0)"},
                  {"main": 2, "cross": 0, "weight": 0.4, "label": "Recursive Cluster (2-0-4)"}],
    "tack":      [{"main": 2, "cross": 0, "weight": 0.7, "label": "The Tack Zone (2-0-5)"},
                  {"main": 0, "cross": 2, "weight": 0.3, "label": "Ring of Fire / Inferno (5-0-2)"}],
    "ice":       [{"main": 0, "cross": 1, "weight": 0.55, "label": "Embrittlement / Super Brittle (5-2-0)"},
                  {"main": 2, "cross": 1, "weight": 0.45, "label": "Icicle Impale (0-2-5)"}],
    "glue":      [{"main": 1, "cross": 0, "weight": 0.45, "label": "Glue Strike / Storm (2-5-0)"},
                  {"main": 2, "cross": 1, "weight": 0.35, "label": "MOAB Glue (0-1-3)"},
                  {"main": 0, "cross": 2, "weight": 0.20, "label": "Bloon Solver (5-0-2)"}],
    "sniper":    [{"main": 0, "cross": 1, "weight": 0.7, "label": "Maim / Cripple MOAB (4-2-0)"},
                  {"main": 2, "cross": 0, "weight": 0.3, "label": "Full Auto line (2-0-4)"}],
    "mortar":    [{"main": 1, "cross": 0, "weight": 0.5, "label": "Artillery Battery (2-4-0)"},
                  {"main": 2, "cross": 1, "weight": 0.5, "label": "Shattering Shells (0-2-4)"}],
    "wizard":    [{"main": 2, "cross": 1, "weight": 0.65, "label": "Prince of Darkness (0-2-5)"},
                  {"main": 1, "cross": 0, "weight": 0.35, "label": "Summon Phoenix (2-5-0)"}],
    "super":     [{"main": 0, "cross": 2, "weight": 0.6, "label": "Sun Avatar (3-0-2)"},
                  {"main": 0, "cross": 1, "weight": 0.4, "label": "Sun Avatar (3-2-0)"}],
    "ninja":     [{"main": 1, "cross": 0, "weight": 0.55, "label": "Bloon Sabotage / Shinobi (0-4-0)"},
                  {"main": 0, "cross": 2, "weight": 0.45, "label": "Bloonjitsu / Grandmaster (4-0-2)"}],
    "alchemist": [{"main": 0, "cross": 1, "weight": 0.6, "label": "Berserker Brew (4-2-0)"},
                  {"main": 0, "cross": 2, "weight": 0.4, "label": "Berserker Brew (4-0-1)"}],
    "druid":     [{"main": 2, "cross": 1, "weight": 0.7, "label": "Poplust / Avatar of Wrath (0-1-4/5)"},
                  {"main": 0, "cross": 2, "weight": 0.3, "label": "Ball Lightning (4-0-2)"}],
    "village":   [{"main": 1, "cross": 0, "weight": 0.8, "label": "Radar / MIB / Call to Arms (2-3-0)"},
                  {"main": 0, "cross": 1, "weight": 0.2, "label": "Primary Expertise (4-2-0)"}],
    "engineer":  [{"main": 1, "cross": 0, "weight": 0.7, "label": "Overclock (2-4-0)"},
                  {"main": 0, "cross": 1, "weight": 0.3, "label": "Sentry Expert line (4-2-0)"}],
    "spike":     [{"main": 1, "cross": 2, "weight": 0.55, "label": "Spike Storm (0-4-2)"},
                  {"main": 2, "cross": 1, "weight": 0.45, "label": "Perma-Spike (0-2-5)"}],
    "sub":       [{"main": 1, "cross": 0, "weight": 0.5, "label": "First Strike (2-4-0)"},
                  {"main": 0, "cross": 1, "weight": 0.5, "label": "Energizer (5-2-0)"}],
    "buccaneer": [{"main": 2, "cross": 0, "weight": 0.6, "label": "Trade Empire line (2-0-4)"},
                  {"main": 0, "cross": 2, "weight": 0.4, "label": "Pirate Lord line (4-0-2)"}],
    "heli":      [{"main": 0, "cross": 2, "weight": 0.6, "label": "Apache line (4-0-2)"},
                  {"main": 1, "cross": 0, "weight": 0.4, "label": "Downdraft (2-3-0)"}],
    "dartling":  [{"main": 0, "cross": 2, "weight": 0.5, "label": "Laser Cannon / MAD line (4-0-2)"},
                  {"main": 2, "cross": 0, "weight": 0.5, "label": "Ray of Doom line (2-0-4)"}],
}

# Placement profiles: how each tower wants to sit relative to the track
# and its teammates. "range" is the attack/buff radius as a fraction of
# screen WIDTH (BTD6 range units / 560 -- coarse, ranking is what
# matters). Styles:
#   coverage    maximize track cells in range (DPS: hit more bloons for
#               longer -- bends and long straights beat corners)
#   upstream    cover the stretch JUST BEFORE the carry's kill zone
#               (debuffers: glue applied too early wears off before the
#               DPS sees the bloons; applied downstream it does nothing)
#   buddy       sit within buff radius of teammates, the carry above all
#   downstream  cover late track to catch leaks (spikes near the exit)
#   offside     global range: stay OFF the prime real estate others need
PLACEMENT = {
    "dart":      {"range": 0.057, "style": "coverage"},
    "boomerang": {"range": 0.077, "style": "coverage"},
    "bomb":      {"range": 0.071, "style": "coverage"},
    "tack":      {"range": 0.041, "style": "coverage"},
    "ice":       {"range": 0.036, "style": "upstream"},
    "glue":      {"range": 0.082, "style": "upstream"},
    "sniper":    {"range": None,  "style": "offside"},
    "sub":       {"range": 0.075, "style": "coverage"},
    "buccaneer": {"range": 0.082, "style": "coverage"},
    "heli":      {"range": 0.080, "style": "coverage"},
    "mortar":    {"range": None,  "style": "offside"},
    "dartling":  {"range": None,  "style": "coverage"},
    "wizard":    {"range": 0.071, "style": "coverage"},
    "super":     {"range": 0.089, "style": "coverage"},
    "ninja":     {"range": 0.071, "style": "coverage"},
    "alchemist": {"range": 0.080, "style": "buddy"},
    "druid":     {"range": 0.062, "style": "coverage"},
    "village":   {"range": 0.071, "style": "buddy"},
    "engineer":  {"range": 0.071, "style": "coverage"},
    "spike":     {"range": 0.036, "style": "downstream"},
    "beast":     {"range": 0.060, "style": "coverage"},
}

# Role tags per tower for layout templating: every meta layout wants a
# carry, an amplifier, and some control (Dashboard: "carry + stall +
# debuff + cleanup"). "opener" = cheap towers that hold the early rounds.
ROLES = {
    "carry":     ["tack", "super", "wizard", "druid", "boomerang", "dartling"],
    "amplifier": ["alchemist", "village", "ice", "glue", "engineer"],
    "control":   ["ice", "glue", "sniper", "ninja", "bomb", "spike"],
    # Openers must be genuinely CHEAP -- they hold rounds 1-8 on start
    # cash. (Spike factory was here once: a $1000 "opener" that income
    # pacing correctly scheduled for round 5, i.e. not an opener.)
    "opener":    ["dart", "ninja", "sniper", "boomerang", "glue"],
}

# Threat solutions among land towers the bot can actually buy.
# tower -> null (base tower solves it) or [path_index, tier] required.
SOLUTIONS = {
    "camo": {"ninja": None, "wizard": [2, 2], "village": [1, 2],
             "sniper": [1, 1], "dart": [2, 2], "mortar": [2, 3]},
    "lead": {"bomb": None, "mortar": None, "wizard": [1, 1],
             "alchemist": [0, 2], "glue": [0, 2], "tack": [0, 3],
             "ice": [0, 2], "sniper": [0, 1], "boomerang": [2, 2],
             "druid": [0, 2], "super": [0, 2], "spike": [2, 2]},
    "moab": {"sniper": [0, 4], "bomb": [1, 3], "boomerang": [2, 4],
             "glue": [2, 3], "ninja": [1, 4], "ice": [2, 5],
             "wizard": [2, 5], "super": [0, 3], "tack": [2, 5],
             "druid": [2, 5], "spike": [1, 4]},
}

# Towers the sheet doesn't rank but the bot can use; scores inferred from
# the sheet's own mentions (Spike Storm shows up in three Round Threats
# rows) and flagged so nobody mistakes them for researched numbers.
INFERRED_TOWERS = {
    "spike": {"score": 83, "role": "cleanup/control",
              "partners": ["village", "alchemist", "engineer"],
              "notes": "Not ranked in the sheet; inferred from Round "
                       "Threats mentions (Spike Storm on R90-99)."},
}


def _norm(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _rows(ws):
    out = []
    for row in ws.iter_rows():
        vals = [c.value for c in row]
        while vals and vals[-1] in (None, ""):
            vals.pop()
        if vals:
            out.append(vals)
    return out


def _partners_list(cell):
    parts = re.split(r"[,/]| and ", str(cell or ""))
    keys = []
    for p in parts:
        n = _norm(p)
        for word, key in (("village", "village"), ("alch", "alchemist"),
                          ("glue", "glue"), ("ice", "ice"),
                          ("brittle", "ice"), ("sabo", "ninja"),
                          ("ninja", "ninja"), ("striker", "bomb"),
                          ("farm", "farm"), ("stall", "ice"),
                          ("overclock", "engineer"), ("sniper", "sniper"),
                          ("obyn", "hero:obyn"), ("geraldo", "hero:geraldo"),
                          ("pat", "hero:pat"), ("brickell", "hero:brickell"),
                          ("gwen", "hero:gwen")):
            if word in n and key not in keys:
                keys.append(key)
    return keys


def extract(xlsx_path):
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)

    towers = {}
    for row in _rows(wb["Tower Roles"])[1:]:
        if len(row) < 7:
            continue
        family, role, status, crosspath, partners, avoid, score = row[:7]
        notes = row[7] if len(row) > 7 else ""
        # Longest key first: "dartling" must win over "dart".
        key = next((v for k, v in sorted(SHEET_TOWER_KEYS.items(),
                                         key=lambda kv: -len(kv[0]))
                    if _norm(family).startswith(k)), None)
        if key is None or not isinstance(score, (int, float)):
            continue
        towers[key] = {
            "score": float(score),
            "role": str(role),
            "status": str(status),
            "crosspath_note": str(crosspath),
            "partners": [p for p in _partners_list(partners)
                         if not p.startswith("hero:")],
            "avoid_for": str(avoid),
            "notes": str(notes),
            "builds": BUILDS.get(key, []),
            "placement": PLACEMENT.get(key,
                                       {"range": 0.06, "style": "coverage"}),
        }
    for key, info in INFERRED_TOWERS.items():
        if key not in towers:
            towers[key] = {**info, "status": "inferred",
                           "builds": BUILDS.get(key, []),
                           "placement": PLACEMENT.get(
                               key, {"range": 0.06, "style": "coverage"}),
                           "inferred": True}

    heroes = {}
    for row in _rows(wb["Hero Matrix"])[1:]:
        if len(row) < 7 or not isinstance(row[5], (int, float)):
            continue
        heroes[str(row[0])] = {
            "role": str(row[1]), "best_for": str(row[2]),
            "strengths": str(row[3]), "caveats": str(row[4]),
            "score": float(row[5]), "micro": float(row[6]),
        }

    threats = []
    for row in _rows(wb["Round Threats"])[1:]:
        if len(row) < 4:
            continue
        rounds = [int(x) for x in re.findall(r"\d+", str(row[0]))]
        label = _norm(row[1])
        kind = ("ddt" if "ddt" in label else
                "bad" if label == "bad" else
                "ceramic" if "ceramic" in label else
                "camo" if "camo" in label else
                "lead" if "lead" in label else
                "moab" if "moab" in label or "zomg" in label else "other")
        threats.append({"rounds": rounds, "threat": str(row[1]),
                        "kind": kind, "tests": str(row[2]),
                        "answers": str(row[3]),
                        "notes": str(row[4]) if len(row) > 4 else ""})

    modes = {}
    for row in _rows(wb["Economy & Modes"])[1:]:
        if len(row) < 3:
            continue
        modes[_norm(row[0]).replace(" ", "_")] = {
            "priority": str(row[1]), "approach": str(row[2]),
            "avoid": str(row[3]) if len(row) > 3 else "",
            "notes": str(row[4]) if len(row) > 4 else "",
        }

    thesis = []
    for row in _rows(wb["Dashboard"]):
        if len(row) >= 2 and isinstance(row[0], (int, float)) \
                and isinstance(row[1], str) and len(row[1]) > 40:
            thesis.append(row[1])

    return {
        "version": "v55",
        "source": str(xlsx_path.relative_to(REPO))
                  if xlsx_path.is_relative_to(REPO) else str(xlsx_path),
        "generated_by": "tools/extract_meta.py",
        "thesis": thesis,
        "towers": towers,
        "heroes": heroes,
        "roles": ROLES,
        "solutions": SOLUTIONS,
        "threats": threats,
        "modes": modes,
    }


def main():
    xlsx = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_XLSX
    if not xlsx.exists():
        sys.exit(f"Spreadsheet not found: {xlsx}")
    knowledge = extract(xlsx)
    OUT_PATH.write_text(json.dumps(knowledge, indent=1) + "\n")
    n_towers = len(knowledge["towers"])
    n_threats = len(knowledge["threats"])
    print(f"Wrote {OUT_PATH.name}: {n_towers} towers, "
          f"{len(knowledge['heroes'])} heroes, {n_threats} threat rows, "
          f"{len(knowledge['modes'])} modes.")


if __name__ == "__main__":
    main()
