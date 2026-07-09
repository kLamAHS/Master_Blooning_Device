#!/usr/bin/env python3
"""
BTD6 Stage-1 bot -- screen capture, clicking, and round detection.

This is the "hands" of the ML project. No machine learning lives in this
file; it is the reliable pipeline that will later collect training data
(Stage 2) and execute the plans your genetic algorithm finds (Stage 3).

Subcommands
-----------
  locate               Print live mouse coordinates (pixels + normalized).
                       Use this to find coordinates for your plan file.
  watch                Continuously OCR the round counter and print what
                       the bot sees. Saves debug images so you can tune
                       "round_box" in config.json.
  play <plan.json>     Execute a gameplan against the running game.
  scan <map_name>      Sweep a tower ghost across the map and record every
                       placeable spot (the game paints invalid spots red).
                       Writes masks/<map>_<tower>.json -- this is what the
                       emergent layout generator samples from in Stage 2.
  farm <map_name>      STAGE 2: play random layouts unattended, restarting
                       between games, logging (layout, final_round,
                       outcome) to runs_log.jsonl -- the training dataset.
  solve <map_name>     STAGE 4: play to WIN the loaded rung (difficulty
                       and mode auto-detected, CHIMPS included) --
                       explore/attempt episodes until the final round is
                       actually beaten, then record it in progress.json.
  deploy <map_name>    Run the FINISHED trained model as a bot: replay
                       the champion layout straight to win the loaded
                       rung -- no exploration, no learning. Needs a
                       champion (train one with farm/solve first).
  campaign             Show the per-map ladder scoreboard (easy ->
                       medium -> hard -> CHIMPS) from progress.json.

Safety
------
  * pyautogui's failsafe is ON: slam the mouse into the TOP-LEFT corner
    of the screen to abort instantly.
  * Ctrl+C in the terminal also stops the bot.

Ninja Kiwi's terms of service forbid automation. Use this offline, in
single player only, ideally on a throwaway account. Never in races,
co-op, or events.
"""

import argparse
import csv
import ctypes
import json
import random
import re
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import mss
import pyautogui

BUILD = "2026-07-06.r12"  # printed at startup: ties every log to a build

DEBUG = True     # verbose decision logging; pass --quiet to farm/play


def dbg(msg):
    if DEBUG:
        print(f"      . {msg}")


# Provable-lower-bound cash guard (pure stdlib; see cashguard.py). It makes a
# low cash misread non-fatal: the bot keeps buying up to what it PROVABLY has
# instead of freezing behind a phantom-broke wallet and leaking the run.
from cashguard import CashFloor
# Pure placement geometry (see placement.py): never click a spot we know is
# already taken -- relocate to the nearest free mask point instead.
from placement import free_placement_spots

try:
    import pytesseract
except ImportError:
    pytesseract = None

# BTD6 (like most Unity games) reads keyboard input at the scan-code level
# and IGNORES the virtual-key events pyautogui sends -- keys typed by a
# human work, keys sent by pyautogui silently do nothing. pydirectinput
# sends real scan codes, so all key presses and clicks go through it.
# pyautogui is still used for cursor movement (games read the real OS
# cursor position) and for its corner-failsafe.
try:
    import pydirectinput
    pydirectinput.PAUSE = 0.05
    pydirectinput.FAILSAFE = True
except ImportError:
    pydirectinput = None
    if sys.platform == "win32":
        print("WARNING: pydirectinput is not installed, so BTD6 will most "
              "likely IGNORE every key press this bot sends.\n"
              "Fix: pip install pydirectinput\n")

pyautogui.FAILSAFE = True   # mouse to top-left corner = emergency stop
pyautogui.PAUSE = 0.05      # tiny pause after every pyautogui call


def press_key(key):
    """Send a key press the game will actually register."""
    (pydirectinput or pyautogui).press(key)


def click(button="left"):
    """Send a mouse click the game will actually register."""
    (pydirectinput or pyautogui).click(button=button)

if sys.platform == "win32":
    # Without this, Windows display scaling (125%, 150%...) puts clicks and
    # screenshots in different coordinate systems and everything misses.
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Configuration (per-machine settings live in config.json, created on first
# run; per-map strategy lives in the plan .json files)
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent / "config.json"

DEFAULT_CONFIG = {
    # How the bot finds the game on screen, in priority order:
    #   1. "region": [left, top, width, height] in pixels -- manual override.
    #      Leave null to use auto-detection (recommended).
    #   2. Auto-detect: the visible window whose title contains
    #      "window_title" (Windows only). Works windowed OR fullscreen,
    #      at any resolution -- 1920x1080, 1440p, whatever.
    #   3. Fallback: the whole primary monitor.
    "region": None,
    "window_title": "BloonsTD6",

    # Where the round counter ("13/40") sits, as fractions of the game
    # area: [x, y, width, height]. Tune with `watch` if OCR reads garbage.
    "round_box": [0.695, 0.012, 0.125, 0.052],

    # Where the cash number sits. null = auto-located (the bot finds the
    # gold coin icon and takes the box to its right).
    "cash_box": None,

    # Where the lives number sits. null = auto-located (red heart icon).
    # Lives reaching 0 is how the bot detects defeat -- the round counter
    # stays visible on the defeat screen, so it can't be the signal.
    "lives_box": None,

    # A patch of open ground with no towers on it. Clicking here closes
    # upgrade panels without selecting anything new.
    "deselect_point": [0.50, 0.92],

    # One-time calibration for `farm` (unattended restarts). Use `locate`:
    #   defeat_restart: the RESTART button on the defeat screen
    #                   (lose a game once on purpose to see it)
    #   pause_restart:  the RESTART button in the pause menu (press Esc)
    #   restart_confirm: the confirm/OK button on the "restart?" dialog
    "defeat_restart": None,
    "pause_restart": None,
    "restart_confirm": None,

    # Seconds to wait between the individual clicks/keys of one action.
    "action_delay": 0.40,

    # `scan`: how much redder (vs. the clean map) the ring around the
    # cursor must get before a grid point counts as NOT placeable. The
    # invalid-tint typically scores 20-60, valid ground scores near 0.
    # Tune only if the scan preview disagrees with the map.
    "scan_red_shift": 12.0,

    # Ability hotkeys pressed on threat rounds / leak emergencies by
    # farm and solve. Pressing them with no ability trained is a no-op.
    "ability_keys": ["1", "2", "3"],

    # Full path to the tesseract binary if it is not on PATH (Windows
    # usually needs this). null = try to auto-detect.
    "tesseract_cmd": None,
}

# Default BTD6 hotkeys. If you changed yours in Settings -> Hotkeys,
# update this dict to match. Newer towers may use other keys -- add them.
TOWER_HOTKEYS = {
    "dart": "q", "boomerang": "w", "bomb": "e", "tack": "r",
    "ice": "t", "glue": "y",
    "sniper": "z", "sub": "x", "buccaneer": "c", "ace": "v",
    "heli": "b", "mortar": "n", "dartling": "m",
    "wizard": "a", "super": "s", "ninja": "d", "alchemist": "f",
    "druid": "g",
    "farm": "h", "spike": "j", "village": "k", "engineer": "l",
    "beast": "i",
    "hero": "u",
}

# Upgrade hotkeys for [top path, middle path, bottom path]
UPGRADE_KEYS = [",", ".", "/"]


def load_config():
    if CONFIG_PATH.exists():
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(json.loads(CONFIG_PATH.read_text()))
        return cfg
    CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
    print(f"Created {CONFIG_PATH.name} with default settings -- edit it if "
          f"you play windowed or if `watch` can't read the round counter.\n")
    return dict(DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# Screen: capture + coordinate conversion
# ---------------------------------------------------------------------------
# All plan coordinates are *normalized*: fractions of the game area from
# 0.0 to 1.0. That way a plan written at 1920x1080 still works at 2560x1440.

class Screen:
    def __init__(self, region):
        self.sct = mss.MSS() if hasattr(mss, "MSS") else mss.mss()
        if region:
            self.left, self.top, self.w, self.h = region
        else:
            mon = self.sct.monitors[1]  # primary monitor
            self.left, self.top = mon["left"], mon["top"]
            self.w, self.h = mon["width"], mon["height"]

    def to_pixels(self, nx, ny):
        return self.left + int(nx * self.w), self.top + int(ny * self.h)

    def to_norm(self, px, py):
        return (px - self.left) / self.w, (py - self.top) / self.h

    def grab(self, norm_box=None):
        """Screenshot the whole game area, or a normalized sub-box, as BGR."""
        if norm_box is None:
            box = {"left": self.left, "top": self.top,
                   "width": self.w, "height": self.h}
        else:
            x, y, w, h = norm_box
            box = {"left": self.left + int(x * self.w),
                   "top": self.top + int(y * self.h),
                   "width": max(1, int(w * self.w)),
                   "height": max(1, int(h * self.h))}
        shot = self.sct.grab(box)
        return np.asarray(shot)[:, :, :3].copy()  # BGRA -> BGR


# ---------------------------------------------------------------------------
# Game window auto-detection (Windows)
# ---------------------------------------------------------------------------

class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def find_game_window(title_fragment):
    """Find the game window by title. Returns (hwnd, title, region) where
    region = [left, top, width, height] of the CLIENT area -- the actual
    game pixels, excluding the title bar and borders -- or None if no match.
    Windows only; on other platforms this quietly returns None."""
    if sys.platform != "win32":
        return None
    user32 = ctypes.windll.user32
    matches = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def _collect(hwnd, _lparam):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                if title_fragment.lower() in buf.value.lower():
                    matches.append((hwnd, buf.value))
        return True

    user32.EnumWindows(_collect, 0)
    if not matches:
        return None
    # Prefer an exact title match over "some browser tab mentioning bloons".
    matches.sort(key=lambda m: m[1].lower() != title_fragment.lower())
    hwnd, title = matches[0]

    def client_region():
        rect = _RECT()
        user32.GetClientRect(hwnd, ctypes.byref(rect))
        origin = _POINT(0, 0)
        user32.ClientToScreen(hwnd, ctypes.byref(origin))
        return [origin.x, origin.y,
                rect.right - rect.left, rect.bottom - rect.top]

    region = client_region()
    if region[2] <= 0 or region[3] <= 0:   # minimized -- restore and retry
        user32.ShowWindow(hwnd, 9)          # SW_RESTORE
        time.sleep(0.6)
        region = client_region()
    if region[2] <= 0 or region[3] <= 0:
        return None
    return hwnd, title, region


def focus_game_window(hwnd):
    """Best-effort: bring the game to the foreground so clicks land in it."""
    if hwnd is None or sys.platform != "win32":
        return False
    try:
        user32 = ctypes.windll.user32
        user32.ShowWindow(hwnd, 9)          # SW_RESTORE
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.4)
        return True
    except Exception:
        return False


def make_screen(cfg):
    """Work out where the game is (see DEFAULT_CONFIG for priority order),
    print what was decided, and return (Screen, hwnd_or_None)."""
    hwnd = None
    if cfg.get("region"):
        region = cfg["region"]
        source = 'config.json "region" override'
    else:
        found = find_game_window(cfg.get("window_title", "BloonsTD6"))
        if found:
            hwnd, title, region = found
            source = f"auto-detected window {title!r}"
        else:
            region = None
            source = ("full primary monitor (no game window found -- fine "
                      "for fullscreen)")
    screen = Screen(region)
    print(f"Game area: {source} -> ({screen.left}, {screen.top}, "
          f"{screen.w}x{screen.h})")
    return screen, hwnd


# ---------------------------------------------------------------------------
# Round detection (OCR)
# ---------------------------------------------------------------------------

def setup_tesseract(cfg):
    if pytesseract is None:
        sys.exit("pytesseract is not installed. Run: pip install pytesseract")
    if cfg.get("tesseract_cmd"):
        pytesseract.pytesseract.tesseract_cmd = cfg["tesseract_cmd"]
        return
    if shutil.which("tesseract"):
        return
    for candidate in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                      r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if Path(candidate).exists():
            pytesseract.pytesseract.tesseract_cmd = candidate
            return
    sys.exit("Tesseract binary not found. Install it (see README.md) or set "
             "\"tesseract_cmd\" in config.json.")


def preprocess_round_crop(img):
    """Make HUD text easy for tesseract: big, black-on-white. Threshold
    190: the text is pure white while grass-texture highlights sit around
    170-185 (visible as speckle in the user's debug dumps at 170)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_not(thresh)


def parse_round(text, total=None):
    """'13/40' -> 13. Returns None when the text doesn't look like a round.

    `total` (the map's final round, e.g. 100) rescues the common failure that
    makes a perfectly VISIBLE counter read as "dark": the '/' between the
    current round and the total is faint, so tesseract drops it or reads it as
    a digit and '34/100' fuses into '34100' / '341100'. That parses to a >200
    number -> None -> the bot thinks the HUD went dark and stops advancing its
    plan (the field complaint: it can't prep for late game because it "lost"
    the counter). When the fused digits end with the known total, strip it
    (and one stray separator glyph) and re-read the head as the current round."""
    text = text.strip()
    if "/" in text:
        text = text.split("/", 1)[0]
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    if 1 <= int(digits) <= 200:
        return int(digits)
    if total is not None:
        ts = str(int(total))
        if len(digits) > len(ts) and digits.endswith(ts):
            head = digits[:-len(ts)]
            for cand in (head, head[:-1]):     # allow one misread '/' glyph
                if cand and 1 <= int(cand) <= 200:
                    return int(cand)
    return None


def read_round(screen, cfg, total=None):
    """Returns (parsed_round_or_None, raw_text, crop_img, processed_img).
    `total` is the map's final round (e.g. 100); it lets parse_round recover a
    counter whose '/' faded and fused 'current/total' into one long number."""
    crop = screen.grab(cfg["round_box"])
    processed = preprocess_round_crop(crop)
    text = pytesseract.image_to_string(
        processed,
        config="--psm 7 -c tessedit_char_whitelist=0123456789/").strip()
    return parse_round(text, total=total), text, crop, processed


# Freeplay pushes past a beaten mode, so its round counter is a bare number
# (no "/total") that keeps climbing past 100. `solve --freeplay` sets this so
# the first-read sanity bound stops rejecting rounds above 100.
FREEPLAY = False


def plausible(new, last):
    """Rounds only move forward, and only a little at a time."""
    if new is None:
        return False
    if last is None:
        return 1 <= new <= (200 if FREEPLAY else 100)
    return last <= new <= last + 3


def save_config_value(key, value):
    """Persist one setting into config.json without touching the rest."""
    cfg = (json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists()
           else dict(DEFAULT_CONFIG))
    cfg[key] = value
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


# ---------------------------------------------------------------------------
# Self-learned price book. No hardcoded tables: BTD6 rebalances costs
# between versions and they differ per difficulty, so the bot records what
# every purchase ACTUALLY cost (cash before minus cash after) and reuses
# that knowledge on later runs -- to wait for exact amounts, and to
# estimate what a plan will need in total.
# ---------------------------------------------------------------------------

PRICE_DIFFICULTY = "medium"      # set from the plan's mode when playing
PRICES_SRC = {}                  # key -> "buy" | "seen" | "short" (session)

# Shipped, accurate BTD6 upgrade/base costs keyed EXACTLY like PRICES
# ("{difficulty}:{tower}[:{path}:{tier}]", e.g. "hard:dart:2:3"). Read-only and
# kept SEPARATE from the learned prices.json: it is a FALLBACK the game can
# never misread. Two uses -- (1) fill a price we never learned or that a panel
# failed to read (price_of / PRICES.get below), so an affordability/spend/
# reserve decision always has the real cost; (2) validate a live read
# (record_price below) so a misread can't poison the book. CHIMPS shares the
# "hard" column (1.08x).
SEED_PRICES_PATH = Path(__file__).parent / "tower_prices.json"
SEED_PRICES = (json.loads(SEED_PRICES_PATH.read_text())
               if SEED_PRICES_PATH.exists() else {})


class _PriceBook(dict):
    """The learned price book, with the shipped seed baked into .get() as a
    read-only fallback: every existing PRICES.get(price_key(...)) call now
    resolves learned -> seed -> default with NO call-site change, so a price a
    panel misread or never showed still comes back as the real BTD6 cost.
    Writes (record_price) and json.dumps still see only the learned entries, so
    seeds never leak into prices.json."""

    def get(self, key, default=None):
        v = dict.get(self, key)
        if v is not None:
            return v
        s = SEED_PRICES.get(key)
        return s if s is not None else default


PRICES_PATH = Path(__file__).parent / "prices.json"
PRICES = _PriceBook(json.loads(PRICES_PATH.read_text())
                    if PRICES_PATH.exists() else {})

# Upgrades you haven't unlocked with XP. The bot must NOT press these --
# doing so spends your limited XP. Auto-detected (an affordable press that
# moves neither cash nor tier is locked) and remembered across runs. You
# can also pre-seed this file by hand, e.g. {"easy:dartling:0:1": true} or
# broader "don't go past tier 2 on ninja" style entries you add yourself.
LOCKED_PATH = Path(__file__).parent / "locked.json"
LOCKED = json.loads(LOCKED_PATH.read_text()) if LOCKED_PATH.exists() else {}

# When you've unlocked every upgrade with XP, the auto-lock detector is pure
# downside: a single tier that fails to buy for some OTHER reason (a misread
# price, a panel that didn't open, a cash hiccup) gets mis-recorded as
# XP-locked and the bot then refuses that path forever. `--no-locks` turns the
# whole mechanism off -- nothing is ever treated as locked and nothing new is
# ever recorded -- for accounts that have everything unlocked.
IGNORE_LOCKS = False


def is_locked(ttype, path_i, tier):
    if IGNORE_LOCKS:
        return False
    return bool(LOCKED.get(price_key(ttype, path_i, tier)))


def mark_locked(ttype, path_i, tier):
    if IGNORE_LOCKS:
        return
    key = price_key(ttype, path_i, tier)
    if not LOCKED.get(key):
        LOCKED[key] = True
        LOCKED_PATH.write_text(json.dumps(LOCKED, indent=1, sort_keys=True))
        print(f"      (locked: {key} not unlocked -- won't try it again)")


def price_key(*parts):
    return ":".join([PRICE_DIFFICULTY, *map(str, parts)])


def price_of(*parts):
    """The price to ESTIMATE with: a verified learned value if we have one,
    else the shipped seed, else None (learned > seed > None). Spend tracking,
    reserve, and plan math read through here (and through PRICES.get), so a
    price the panel misread or never showed still resolves to the real BTD6
    cost. NOT for the press/no-press gates -- see learned_price."""
    key = price_key(*parts)
    v = PRICES.get(key)
    return v if v is not None else SEED_PRICES.get(key)


def learned_price(*parts):
    """The VERIFIED learned price ONLY (no seed fallback), for the press/no-
    press affordability GATES. The seed is an ESTIMATE and can OVER-state the
    real cost (a Monkey Village discount aura -- which is active in CHIMPS -- a
    stacked discount, or a stale/high shipped entry); gating a press on that
    over-statement would make the bot refuse to buy, and over-save for, a tier
    the panel would actually sell it. Unlearned -> None -> the gate is skipped
    and the tower panel verifies affordability by press-and-check, exactly as
    before seeds existed. The seed still sharpens spend/reserve/plan math."""
    return dict.get(PRICES, price_key(*parts))


# A verified purchase can legitimately differ from the shipped guide -- a
# Monkey Village discount aura (active in CHIMPS), a balance patch the guide
# predates -- so a cash-verified 'buy' is trusted to TEACH a new price within a
# wide sanity band (outside it, the delta is a corrupted cash read, not a real
# cost). An UNVERIFIED OCR read (green 'seen' / red 'short') is the opposite: it
# can never beat the guide, which by definition can't be misread, so when a
# seed exists an unverified read is ignored -- only a real purchase updates a
# seeded price. This is what keeps a clip/dupe/neighbor-row misread from ever
# shadowing the accurate guide, while still letting the book learn true costs.
SEED_BUY_LO = 0.5      # a verified buy below this * seed is a cash-misread delta
SEED_BUY_HI = 2.0      # ...above this too; between, trust it (discount/patch)


def record_price(key, cost, src="buy"):
    """Persist a learned price. Poison filters: a real BTD6 price is a multiple
    of 5 (a corrupted cash delta almost never is) and never absurdly huge. When
    a shipped seed exists for this key, an UNVERIFIED read (src != 'buy') is
    ignored -- the guide can't be misread, so only a cash-verified purchase may
    update it, and only within a wide sanity band (outside it the delta came
    from a bad cash read). src: 'buy' (cash-verified) / 'seen' (green panel) /
    'short' (red panel -- unverified, eligible for a re-check)."""
    if cost is None or cost <= 0 or cost > 120000:
        return
    if cost % 5 != 0:              # every real BTD6 price ends in 0 or 5
        return
    seed = SEED_PRICES.get(key)
    if seed:
        if src != "buy":
            return                 # unverified read never overrides the guide
        if not (SEED_BUY_LO * seed <= cost <= SEED_BUY_HI * seed):
            dbg(f"      (buy delta ${int(cost)} for {key} rejected: wildly "
                f"off the ${seed} guide -- corrupted cash read, keeping it)")
            return
    cost = int(cost)
    PRICES_SRC[key] = src
    if PRICES.get(key) != cost:
        PRICES[key] = cost
        PRICES_PATH.write_text(json.dumps(PRICES, indent=1, sort_keys=True))


def plan_cost_estimate(plan):
    """Total plan cost from learned prices; returns (known_total,
    unknown_purchase_count). Unknowns become known after one run."""
    tiers, total, unknown = {}, 0, 0
    for a in plan["actions"]:
        if a.get("do") == "place":
            tiers[tuple(a["at"])] = (a["tower"].lower(), [0, 0, 0])
            p = PRICES.get(price_key(a["tower"].lower()))
            total, unknown = total + (p or 0), unknown + (p is None)
        elif a.get("do") == "upgrade":
            at, entry = tuple(a["at"]), None
            if tiers:
                near = min(tiers, key=lambda q: (q[0] - at[0]) ** 2
                           + (q[1] - at[1]) ** 2)
                if ((near[0] - at[0]) ** 2
                        + (near[1] - at[1]) ** 2) ** 0.5 <= 0.03:
                    entry = tiers[near]
            if entry is None:
                unknown += sum(a["path"])
                continue
            ttype, cur = entry
            for i, count in enumerate(a["path"]):
                for _ in range(count):
                    cur[i] += 1
                    p = PRICES.get(price_key(ttype, i, cur[i]))
                    total, unknown = total + (p or 0), unknown + (p is None)
    return total, unknown


def _round_box_from_gear(screen):
    """Primary locator: find the blue settings-gear button in the top strip
    (a very distinctive color blob) and derive the round counter's box from
    it -- the number always sits at a fixed offset to the gear's left.
    Pure geometry, so it doesn't care that OCR struggles with the game
    font."""
    off_x = 0.40
    strip = screen.grab([off_x, 0.0, 1.0 - off_x, 0.14])
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (90, 120, 140), (112, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if area < 150 or w > 0.09 * screen.w or h > 0.11 * screen.h:
            continue
        if not (0.7 <= w / max(h, 1) <= 1.4):   # the button is square-ish
            continue
        if best is None or area > best[0]:
            best = (area, x, y, w, h)
    if best is None:
        return None
    _, x, y, w, h = best
    gcx = x + w / 2 + off_x * screen.w          # strip -> game-area pixels
    gcy = y + h / 2
    size = (w + h) / 2
    x0 = max(0.0, (gcx - 3.2 * size) / screen.w)
    y0 = max(0.0, (gcy - 0.18 * size) / screen.h)
    return [round(x0, 4), round(y0, 4),
            round(2.5 * size / screen.w, 4),
            round(0.70 * size / screen.h, 4)]


def _round_box_from_ocr(screen):
    """Fallback locator: OCR the whole top strip and look for a word shaped
    like 'N/NN'. Returns a padded normalized [x, y, w, h] box, or None."""
    strip = screen.grab([0.0, 0.0, 0.92, 0.12])   # skip the shop column
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    scale = 2
    big = cv2.resize(gray, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_CUBIC)
    _, thr = cv2.threshold(big, 170, 255, cv2.THRESH_BINARY)
    thr = cv2.bitwise_not(thr)
    data = pytesseract.image_to_data(
        thr, config="--psm 11 -c tessedit_char_whitelist=0123456789/",
        output_type=pytesseract.Output.DICT)
    for i, word in enumerate(data["text"]):
        if not re.fullmatch(r"\d{1,3}/\d{1,3}", word.strip()):
            continue
        if float(data["conf"][i]) < 30 or data["height"][i] < 10:
            continue
        x, y = data["left"][i] / scale, data["top"][i] / scale
        bw, bh = data["width"][i] / scale, data["height"][i] / scale
        pad_x, pad_y = 0.8 * bw, 0.6 * bh
        nx = max(0.0, (x - pad_x) / screen.w)
        ny = max(0.0, (y - pad_y) / screen.h)
        nw = min(1.0 - nx, (bw + 2 * pad_x) / screen.w)
        nh = min(1.0 - ny, (bh + 2 * pad_y) / screen.h)
        return [round(nx, 4), round(ny, 4), round(nw, 4), round(nh, 4)]
    debug_dir = Path(__file__).parent / "debug"
    debug_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(debug_dir / "round_search_strip.png"), thr)
    return None


def read_round_stable(screen, cfg, tries=4, total=None):
    """Return a round value only if two consecutive reads agree -- a
    misaligned crop produces flickery garbage that never repeats reliably,
    so this filters out the lucky-junk reads that fooled earlier versions.
    `total` (the HUD's "/100") is forwarded so a faded separator doesn't make
    a visible counter unreadable (see parse_round)."""
    prev = None
    for _ in range(tries):
        value, *_ = read_round(screen, cfg, total=total)
        if value is not None and value == prev:
            return value
        prev = value
        time.sleep(0.25)
    return None


def preflight_round_box(screen, cfg, recalibrate=False):
    """Guarantee a readable round counter before relying on it. Candidate
    boxes come from the gear geometry first, OCR search second; a candidate
    is only saved after an actual stable read verifies it. Requires a
    loaded map (the counter must be on screen)."""
    original = cfg["round_box"]
    if not recalibrate and read_round_stable(screen, cfg) is not None:
        return True
    print("Locating the round counter on screen...")
    candidates = []
    box = _round_box_from_gear(screen)
    if box:
        candidates.append(("settings-gear geometry", box))
    box = _round_box_from_ocr(screen)
    if box:
        candidates.append(("OCR pattern search", box))
    for how, cand in candidates:
        cfg["round_box"] = cand
        if read_round_stable(screen, cfg) is not None:
            if cand != original:
                save_config_value("round_box", cand)
                print(f"Locked on via {how}. Saved round_box={cand} to "
                      f"config.json.")
            return True
    cfg["round_box"] = original          # nothing verified; restore
    return read_round_stable(screen, cfg) is not None


def _hud_icon(screen, strip_box, hsv_lo, hsv_hi, extra_range=None):
    """Find a round HUD icon (coin/heart) in a strip. Returns
    (cx, cy, size) in game-area pixels, or None -- dumping the search
    strip and mask to debug/ on failure so misses can be diagnosed."""
    strip = screen.grab(strip_box)
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lo, hsv_hi)
    if extra_range:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, *extra_range))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if area < 200 or w > 0.06 * screen.w or h > 0.08 * screen.h:
            continue
        if not (0.7 <= w / max(h, 1) <= 1.4):
            continue
        if best is None or area > best[0]:
            best = (area, x, y, w, h)
    if best is None:
        debug_dir = Path(__file__).parent / "debug"
        debug_dir.mkdir(exist_ok=True)
        tag = f"{int(hsv_lo[0])}_{int(hsv_hi[0])}"
        cv2.imwrite(str(debug_dir / f"hud_search_{tag}_strip.png"), strip)
        cv2.imwrite(str(debug_dir / f"hud_search_{tag}_mask.png"), mask)
        return None
    _, x, y, w, h = best
    return (x + w / 2 + strip_box[0] * screen.w,
            y + h / 2 + strip_box[1] * screen.h, (w + h) / 2)


def _cash_box_from_coin(screen):
    """Find the gold coin icon in the top-left HUD; the cash number sits
    just to its right. Starting the box past the '$' sign keeps the symbol
    from being misread as a digit."""
    icon = _hud_icon(screen, [0.0, 0.0, 0.35, 0.12],
                     (12, 110, 140), (40, 255, 255))
    if icon is None:
        return None
    ccx, ccy, size = icon
    # Start BEFORE the '$' on purpose: read_cash anchors on it, and a box
    # that starts too far right clips the leading digit instead
    # ('$374' -> '74', seen in the field as impossible cash deltas).
    x0 = max(0.0, (ccx + 0.55 * size) / screen.w)
    y0 = max(0.0, (ccy - 0.55 * size) / screen.h)
    return [round(x0, 4), round(y0, 4),
            round(4.6 * size / screen.w, 4),
            round(1.1 * size / screen.h, 4)]


def _cash_box_from_text(screen, cfg=None):
    """Find the cash NUMBER itself via word bounding boxes. Guards against
    the lives counter masquerading as cash: candidate tokens must sit
    strictly RIGHT of the lives box (the coin separates them), and a token
    containing '$' beats rightmost-ness -- on frames where '$845' fails to
    tokenize, '200' (lives) must never win by default."""
    strip = screen.grab([0.0, 0.0, 0.35, 0.12])
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    scale = 2
    big = cv2.resize(gray, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_CUBIC)
    _, thr = cv2.threshold(big, 170, 255, cv2.THRESH_BINARY)
    thr = cv2.bitwise_not(thr)
    data = pytesseract.image_to_data(
        thr, config="--psm 11 -c tessedit_char_whitelist=0123456789$,",
        output_type=pytesseract.Output.DICT)
    min_x = 0.14 * screen.w                    # right of lives, always
    lb = (cfg or {}).get("lives_box")
    if lb:
        min_x = max(min_x, (lb[0] + lb[2]) * screen.w)
    tokens = []
    for i, word in enumerate(data["text"]):
        w = word.strip()
        digits = re.sub(r"\D", "", w)
        if len(digits) < 2:
            continue
        if float(data["conf"][i]) < 30 or data["height"][i] < 10:
            continue
        x = data["left"][i] / scale
        if x <= min_x:
            continue                           # that's lives territory
        tokens.append(("$" in w, x, data["top"][i] / scale,
                       data["width"][i] / scale, data["height"][i] / scale))
    if not tokens:
        return None
    has_dollar = [t for t in tokens if t[0]]
    pool = has_dollar or tokens
    _, x, y, bw, bh = max(pool, key=lambda t: t[1])
    x0 = max(0.0, (x - 0.4 * bh) / screen.w)
    y0 = max(0.0, (y - 0.5 * bh) / screen.h)
    return [round(x0, 4), round(y0, 4),
            round((bw + 2.4 * bh) / screen.w, 4),   # room to grow rightward
            round(2.0 * bh / screen.h, 4)]


def _plausible_cash_shape(box):
    """Shape sanity for a cash-counter box (screen fractions). The real
    counter is a short, wide strip pinned to the top HUD; bounds are
    generous, but a box a third of the screen tall can never pass. This
    is the brake on the snap's geometric growth: each session's snap
    widens its search window from the SAVED box, so one rogue tall
    'character' compounds session over session (0.055 -> 0.0845 ->
    0.1676 -> 0.3249 in the field) unless implausible shapes are
    refused both when produced and when loaded.

    The width cap is deliberately GENEROUS (0.22): a legit 6-digit run
    on a wide HUD measures ~0.148 (field: [0.2029, 0.0416, 0.1479,
    0.0651]), and a runaway 'junk $1' box measured ~0.151 -- the two are
    a hair apart, so width CANNOT discriminate good from broken. A cap
    tuned tight enough to reject the junk box (e.g. 0.13) also rejects
    the real counter, which blinds the economy for the whole run. So we
    only reject GROSSLY oversized boxes here (the 0.32 runaway tail) and
    let the stuck-detector (cash_floor.stuck() -> recalibrate) catch a
    box that is plausibly shaped but reads garbage."""
    if not box:
        return False
    x, y, w, h = box
    return (0.015 <= h <= 0.10          # digit strip, not a map chunk
            and w <= 0.22               # a 6-digit run is ~0.11-0.15 wide;
            #                             only reject the gross runaway tail
            #                             (0.32+); the stuck-detector, not
            #                             width, catches a garbage-reading box
            and 1.0 <= w / h <= 12.0    # wide strip (fractions, so a
            and y + h <= 0.16)          # 2-digit run is ~1.15) / top HUD


def _snap_cash_box_to_digits(screen, box, min_x=0.0):
    """Char-level refinement: given a roughly-right cash box, use
    tesseract's per-character boxes on a widened crop to trim it to the
    DIGITS ONLY -- excluding the '$', the glyph that keeps OCR-ing as a
    leading 5/3, and any coin sliver to its left. min_x fences the
    widened window so it can never grow leftward into the LIVES digits
    (which once produced a lives-overlapping 'refit'). Char heights are
    median-filtered so one tall non-digit blob (coin, glow) can't set
    the scale, and the result must pass _plausible_cash_shape or the
    snap returns nothing rather than an inflated box."""
    x, y, w, h = box
    wide = [max(min_x, x - 1.3 * h), max(0.0, y - 0.3 * h),
            w + 2.3 * h, h * 1.6]
    crop = screen.grab(wide)
    proc = preprocess_round_crop(crop)
    try:
        raw = pytesseract.image_to_boxes(
            proc, config="--psm 7 -c tessedit_char_whitelist=0123456789$")
    except Exception:
        return None
    ph = proc.shape[0]
    chars = []
    for line in raw.splitlines():
        p = line.split()
        if len(p) >= 5:
            try:
                c, x1, y1, x2, y2 = p[0], *map(int, p[1:5])
            except ValueError:
                continue
            chars.append((c, x1, ph - y2, x2, ph - y1))  # to top-origin
    last_dollar = max((i for i, ch in enumerate(chars) if ch[0] == "$"),
                      default=-1)
    digits = [ch for ch in chars[last_dollar + 1:] if ch[0].isdigit()]
    if len(digits) < 2:
        return None
    # One coin blob or glow misread as a tall 'digit' must not set the
    # vertical scale: keep only chars near the MEDIAN height, so the box
    # is sized by the actual digit run.
    heights = sorted(d[4] - d[2] for d in digits)
    med = max(heights[len(heights) // 2], 1)
    digits = [d for d in digits if 0.55 * med <= d[4] - d[2] <= 1.6 * med]
    if len(digits) < 2:
        return None
    scale = 3.0                                # preprocess upscales 3x
    dx1 = min(d[1] for d in digits) / scale
    dx2 = max(d[3] for d in digits) / scale
    dy1 = min(d[2] for d in digits) / scale
    dy2 = max(d[4] for d in digits) / scale
    bh = max(dy2 - dy1, 8)
    # Leave ~0.7 digit-widths of slack on the LEFT (was 0.25): cash grows a
    # digit or two over a run and, on this HUD, the number expands leftward,
    # so a box fitted tight to the starting digits clips the new leading
    # digit later. Still well clear of the '$' (~a full digit further left),
    # which the reader skips anyway. The width grows to match so the right
    # room is unchanged; the mid-run recheck heals any clip that still slips.
    nx0 = wide[0] + max(0.0, dx1 - 0.7 * bh) / screen.w
    ny0 = wide[1] + max(0.0, dy1 - 0.35 * bh) / screen.h
    snapped = [round(nx0, 4), round(ny0, 4),
               round((dx2 - dx1 + 2.65 * bh) / screen.w, 4),  # room L+R
               round(1.7 * bh / screen.h, 4)]
    return snapped if _plausible_cash_shape(snapped) else None


def _read_cash_box(screen, box):
    """Read cash from an explicit box without mutating cfg."""
    if not box:
        return None
    crop = screen.grab(box)
    value = _read_number_image(crop)
    if value is None:
        return None
    return value if value <= 150000 else None


def _read_cash_box_confirmed(screen, box, max_plausible=None):
    """Corroborated read from an EXPLICIT box (see read_cash_confirmed for
    the majority-of-three logic). Lets a candidate box be verified before
    it is adopted, without swapping it into cfg first."""
    def ok(v):
        return v is not None and (max_plausible is None or v <= max_plausible)

    a = _read_cash_box(screen, box)
    time.sleep(0.15)
    b = _read_cash_box(screen, box)
    if a == b:
        return a if ok(a) else None
    time.sleep(0.12)
    c = _read_cash_box(screen, box)
    for v in (a, b, c):
        if ok(v) and [a, b, c].count(v) >= 2:
            return v
    return None


def _repair_clipped_cash_box(screen, box, min_x=0.0):
    """If a saved/refit box starts too far right, it can read the visible
    suffix only ('$850' -> 50). Probe left-expanded versions and adopt one
    only when it cleanly contains the current reading as a suffix."""
    base = _read_cash_box(screen, box)
    x, y, w, h = box
    left_fence = min(max(min_x, 0.14), x)
    best = None
    for grow in (0.7, 1.0, 1.4, 1.8, 2.3):
        nx = max(left_fence, x - grow * h)
        cand = [round(nx, 4), y, round(w + (x - nx), 4), h]
        if cand == box or not _plausible_cash_shape(cand):
            continue
        value = _read_cash_box(screen, cand)
        if value is None:
            continue
        if base is None or (value > base and str(value).endswith(str(base))):
            if best is None or value > best[1]:
                best = (cand, value)
    return best[0] if best else None


def recheck_cash_box(screen, cfg, round_hint=None, mode="standard"):
    """Mid-run guard against a cash box that has started misreading -- the
    leading-digit clip that only appears once cash grows past the digit count
    the box was calibrated for (invisible on the small STARTING cash, so the
    preflight repair can't see it), plus HUD drift, a right-edge clip, or an
    UNCLEAN clip that the left-widen repair alone can't recover.

    Two passes, cheap first:
      1. Re-run the left-widen repair -- adopts a wider box only when it reads
         a strictly larger value containing the current read as a suffix (the
         clean-clip signature). Safe no-op when nothing is clipped.
      2. If that finds nothing, re-derive the WHOLE box from a live HUD
         landmark (the coin, whose fixed offset means the leading digit can
         never clip; then the heart as fallback). A re-anchored box is
         adopted only when it is shape-sane, clear of the lives counter, and
         reads a CORROBORATED value strictly LARGER than the current box
         (recovering a lost digit, never inventing one) within the round's
         plausible magnitude. This recovers drift / right-clip / unclean clip
         the left-widen misses, without risking a spurious box.

    Cheap enough to call once a round."""
    box = cfg.get("cash_box")
    if not box:
        return
    lb = cfg.get("lives_box")
    fence = (lb[0] + lb[2]) if lb else 0.0
    ceiling = None
    if round_hint is not None:
        try:
            from meta import earned_by
            ceiling = 2.0 * earned_by(round_hint, mode)   # generous per-round
        except Exception:                                 # magnitude cap
            ceiling = None

    def _adopt(cand, why):
        cfg["cash_box"] = cand
        save_config_value("cash_box", cand)
        dbg(f"cash box re-anchored mid-run ({why}): {box} -> {cand}")

    # Pass 1: cheap left-widen -- recovers a clean leading-digit clip.
    repaired = _repair_clipped_cash_box(screen, box, min_x=fence)
    if repaired and repaired != box and _plausible_cash_shape(repaired):
        _adopt(repaired, "clipped digit")
        return

    # Pass 2: landmark re-anchor -- recovers drift / right-clip / unclean clip,
    # AND heals a box that has started reading a truncated low value (e.g. a
    # single '$1'): a landmark that reads a strictly LARGER, corroborated value
    # is the real number. Only ever adopts a recovery over the CURRENT reading,
    # so it can never adopt a stray low-digit box. If the current box reads
    # nothing at all, leave it -- a dark box is handled safely upstream
    # (sane_cash returns None -> "buy anyway"), and adopting a landmark we
    # can't compare against risks locking onto a stray glyph.
    cur = _read_cash_box(screen, box)
    if cur is None:
        return
    candidates = []
    coin = _cash_box_from_coin(screen)
    if coin:
        snapped = _snap_cash_box_to_digits(screen, coin, min_x=fence)
        candidates.append(snapped or coin)
    candidates += _cash_boxes_from_heart(screen)
    for cand in candidates:
        if not cand or cand == box:
            continue
        if not _plausible_cash_shape(cand) or _overlaps_lives(cand, cfg):
            continue
        quick = _read_cash_box(screen, cand)              # cheap pre-filter
        if quick is None or quick <= cur \
                or (ceiling is not None and quick > ceiling):
            continue          # not a recovery -- only ever adopt MORE digits
        val = _read_cash_box_confirmed(screen, cand, ceiling)   # corroborate
        if val is None or val <= cur:
            continue
        _adopt(cand, "landmark re-anchor")
        return


def _cash_boxes_from_heart(screen):
    """Fallback anchor: the heart locator is rock-solid, and the coin sits
    at a fixed offset to its right, so cash can be derived even when the
    coin's own colors evade detection. Returns SEVERAL candidate boxes at
    slightly different horizontal offsets -- the digits' exact start
    varies with how the icon size was estimated -- for the caller to
    verify by actually reading each one."""
    icon = _hud_icon(screen, [0.0, 0.0, 0.20, 0.12],
                     (0, 140, 120), (8, 255, 255),
                     extra_range=((172, 140, 120), (180, 255, 255)))
    if icon is None:
        return []
    hcx, hcy, size = icon
    boxes = []
    for shift in (4.8, 4.4, 5.2, 4.0):        # land on/before the '$';
        x0 = max(0.0, (hcx + shift * size) / screen.w)   # parser anchors
        y0 = max(0.0, (hcy - 0.55 * size) / screen.h)
        boxes.append([round(x0, 4), round(y0, 4),
                      round(4.8 * size / screen.w, 4),
                      round(1.1 * size / screen.h, 4)])
    return boxes


# ---------------------------------------------------------------------------
# Exact-font HUD number reader. The templates below are rendered from the
# game's own font (LuckiestGuy) and embedded -- no font file and no OCR at
# runtime. Tesseract mangles cartoon display fonts (the source of every
# '$'->'5' prepend and dropped leading digit); shape-matching the true
# glyphs does not. Reads are exact or None, never "close".
# ---------------------------------------------------------------------------

_TEMPLATES_B64 = (
    "eNqVVj1vHDcQnc0C2k5MFdiBgDUCV0GKSxwnccW/cpXd+gcEIA2nV+vCgP9HGq2gIo0B"
    "t+m8hgqVpqHiVjDNyXtDrnIWYCQ64Z3Iub354Lx5vM3me8kiOol0WgQrnfmGZVBYveoi"
    "o6qKw9s84G3plHu+zYF7fqz8WM9UvV6p9lqCxqCvdHJ6oanXT3ATNBT4eqvS49mJbhID"
    "ZPMN4w4B8WSqTzo+OcA4d/CZhT7x+LHOo77TPGpi9MzohQ4ZUBkQeXHPgNV3EcYS1iGM"
    "HVlc5FLkx8238hJuo9c0mjfU+Jfqpe69doq0WBuiI7YuQYs70efI5QxxEAAOHMP+r83A"
    "MyzMYfyPjfj80+aeTBImFJKtAl8rnUI9f57CwlNsKOM+UO1Rx7KJcANmLwZG7RGAGNAB"
    "ByAFPIaWaERDYWL+WvSWALPKAyviK5KM/pWFxP0CGjJQ/B5GJHckoXwtHo5WBCadkRzB"
    "Pqf2n/tyjSdr4d6wBX4HAiKTKoy1gkxqWBqY1+yMmJgQN8nPm+/kNVmEHY6GLbXj8/pM"
    "36Afl5p81uzAk54GcCiFTE55kOpYT/UchgWGQoOD4UwT/rJH4xz+zpphCTiIt5V4xjzL"
    "dzVcv8iHiOOccaxLPZXPDDW5eDNbGibpRR7WglzmWHNEyKlE3tDwrA348gVD7DtqRYDf"
    "EQc0KLtrnSB7Na1O169wsPQ8cjD7mshkVCMl8KUtpQGZ7fDsdcmvgAvgShs36Oe4Ko3O"
    "bJJjSLXiJvllc1/+lq5K2Ey2kmz83BiXKAvMj9+y3FhsXXj9UDh9M09xsMKsKAgjnTtE"
    "jWQzv5m0ZUG66PtGdCxszyCQKCpWlSMsdmRSKB0WqZLvXCvDKSdVxCbybmgZQvMQ1lGr"
    "vCUzy682Q2JieftXCm1u2uwQnWExbhCDITUBmA2jXQL/gsJJtNHqDMU6Sgw23xXOKuD1"
    "QSRknuQ361CfpU6VmsxQX2x0bd6sH0YYNpmSRj3mmPIumXwJucMiYorQ2XMoavZQzVN8"
    "hMkCL3mFDGoasOfQm8Mr+MEz8DNqWNgPCDK1uZI2N8Ly6tqtJLz8rFWu3TRLryaKdh7o"
    "kJ/lkRU3lBvFDY0L17kkOnveKG3F5VBYXAyFft9D5NsieiPFjuwI6/yvSenYyGjjOXur"
    "+R2S2XrLoZEY0w46ZTd5G+HFKtC9KZjXym1uO7oZ1YS/jSp/ItzffCN/9Hfk8Kk8muRC"
    "+hdyuJW7UR5L90IObBkwYSjulD8HUNsHhQwqDsYuEMd8cY7QeYzd1OvWfbKCTjg+l5wo"
    "CNEEbhzc9PtEupdy8FTuTrD+sBnlT+mQ1SBGW3LMWw/5O4ANYrPtSkFNH283Jyf1ni/t"
    "YpqbKsb6kydbIG8qFyv/eUTJFMPBNFLYFptXVI2bT4zsA7aHyPhI5B8EdTWh")
_TEMPLATES = None


def _digit_templates():
    global _TEMPLATES
    if _TEMPLATES is None:
        import base64
        import zlib
        raw = zlib.decompress(base64.b64decode(_TEMPLATES_B64))
        t, i = {}, 0
        while i < len(raw):
            ch = chr(raw[i])
            h, w = raw[i + 1], raw[i + 2]
            n = int.from_bytes(raw[i + 3:i + 5], "big")
            bits = np.unpackbits(np.frombuffer(raw[i + 5:i + 5 + n],
                                               np.uint8))
            t[ch] = bits[:h * w].reshape(h, w).astype(bool)
            i += 5 + n
        _TEMPLATES = t
    return _TEMPLATES


def _shave_comma(glyph):
    """A fused comma leaves edge columns whose ink sits ONLY in the lower
    half of the line -- every digit column reaches the upper half. Trim
    such columns from the edges; the digit underneath emerges clean.
    Pure-comma blobs are left untouched (guarded) so they still classify
    as ',' and get skipped."""
    h = glyph.shape[0]
    has_ink = glyph.any(axis=0)
    tops = np.where(has_ink, np.argmax(glyph, axis=0), h)
    keep = has_ink & (tops < 0.5 * h)
    cols = np.where(keep)[0]
    if len(cols) >= 2 and len(cols) < glyph.shape[1]:
        glyph = glyph[:, cols[0]:cols[-1] + 1]
        rows = np.where(glyph.any(axis=1))[0]
        if len(rows) >= 2:
            glyph = glyph[rows[0]:rows[-1] + 1]
    return glyph


def _split_wide_glyph(glyph, h):
    """At small scales adjacent digit fills can touch and merge into one
    component. Every LuckiestGuy digit is taller than wide, so anything
    wider than its height is a fusion -- split it at the thinnest
    vertical columns (the junctions)."""
    w = glyph.shape[1]
    if w <= 0.92 * h:          # widest single glyph ('0') is 0.88 * h
        return [glyph]
    k = max(2, int(round(w / (0.72 * h))))
    proj = glyph.sum(axis=0)
    cuts = []
    for j in range(1, k):
        c = int(round(w * j / k))
        lo = max(1, c - max(2, w // (2 * k)))
        hi = min(w - 1, c + max(2, w // (2 * k)))
        cuts.append(lo + int(np.argmin(proj[lo:hi])))
    pieces, prev = [], 0
    for c in sorted(set(cuts)):
        if c - prev >= 3:
            pieces.append(glyph[:, prev:c])
            prev = c
    pieces.append(glyph[:, prev:])
    out = []
    for p in pieces:
        cols = np.where(p.any(axis=0))[0]
        rows = np.where(p.any(axis=1))[0]
        if len(cols) >= 2 and len(rows) >= 2:
            out.append(p[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1])
    return out or [glyph]


def _read_number_image(crop):
    """Read a white LuckiestGuy number from a BGR crop by template
    matching. The dark glyph outline separates characters into clean
    connected components; short components (commas, flowers, confetti)
    are filtered by relative height; '$' classifies as itself and is
    skipped. Returns int or None -- exact or nothing."""
    if crop.size == 0:
        return None
    white = (crop > 165).all(axis=2).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    comps = [(stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3], i)
             for i in range(1, n) if stats[i, 4] >= 12]
    if not comps:
        return None
    hmed = float(np.median([c[3] for c in comps]))
    templates = _digit_templates()
    digits = []
    for x, y, w, h, i in sorted(comps):
        if h < 0.55 * hmed or w < 2:
            continue
        glyph = (lab[y:y + h, x:x + w] == i)
        glyph = _shave_comma(glyph)
        pieces = _split_wide_glyph(glyph, glyph.shape[0])
        if len(pieces) > 1:
            # Insurance: a noise-widened single digit must not be halved.
            # If the UNSPLIT blob already matches a digit strongly, trust
            # that reading instead of the split.
            whole_best, whole_score = None, 0.0
            for ch, tpl in templates.items():
                g = cv2.resize(glyph.astype(np.uint8) * 255,
                               (tpl.shape[1], tpl.shape[0]),
                               interpolation=cv2.INTER_AREA) > 90
                union = np.logical_or(g, tpl).sum()
                s = (np.logical_and(g, tpl).sum() / union) if union else 0.0
                if s > whole_score:
                    whole_best, whole_score = ch, s
            if whole_best and whole_best.isdigit() and whole_score >= 0.70:
                pieces = [glyph]
        for piece in pieces:
            best, score, second = None, 0.0, 0.0
            for ch, tpl in templates.items():
                g = cv2.resize(piece.astype(np.uint8) * 255,
                               (tpl.shape[1], tpl.shape[0]),
                               interpolation=cv2.INTER_AREA) > 90
                union = np.logical_or(g, tpl).sum()
                s = (np.logical_and(g, tpl).sum() / union) if union else 0.0
                if s > score:
                    best, score, second = ch, s, score
                elif s > second:
                    second = s
            # Thin glyphs (4/7/9) score lower at small scales; accept a
            # lower score only when the winner clearly beats the runner-up.
            confident = score >= 0.70 or (score >= 0.52
                                          and score - second >= 0.05)
            if best and best.isdigit() and confident:
                digits.append((best, piece.shape[0], y + h / 2))
    if not digits:
        return None
    # Coherence: real digits share a height AND a baseline. Confetti
    # flecks fail the first; cross-row fragments fail the second.
    hm = float(np.median([h for _, h, _ in digits]))
    ym = float(np.median([yc for _, _, yc in digits]))
    kept = [c for c, h, yc in digits
            if abs(h - hm) <= 0.18 * hm and abs(yc - ym) <= 0.45 * hm]
    return int("".join(kept)) if kept else None


def read_cash(screen, cfg):
    """Current cash as an int, or None. Primary reader: exact-font
    template matching (see _read_number_image) -- reads are exact or
    None. Do not fall back to OCR for money values: a wrong cash read is
    worse than no read because it can poison timing, watermarks, and the
    learned price book. Implausibly huge readings return None."""
    box = cfg.get("cash_box")
    return _read_cash_box(screen, box)


def read_cash_confirmed(screen, cfg, max_plausible=None):
    """Corroborated cash read for decisions a single bad read would harm
    (watermarks, price recording). The fast path is unchanged -- two reads
    that agree exactly confirm the value. If they DISAGREE (fast-forward
    cash animation ticking between reads is the usual cause), a third read
    breaks the tie by majority instead of giving up. Affix pairs like
    950/50 or 600/6005 are contradictions, not confirmation.

    `max_plausible`, when the caller passes a ceiling (e.g. the income
    model's cap for the round), rejects any read above it: a clipped
    leading digit ('$2,851' -> '851') reads fluently but low while an
    inflated box reads high -- both are dropped rather than trusted."""
    return _read_cash_box_confirmed(screen, cfg.get("cash_box"), max_plausible)


def _overlaps_lives(box, cfg):
    """Does this candidate cash box intrude on the lives counter's zone?"""
    lb = cfg.get("lives_box")
    if not box or not lb:
        return False
    return (box[0] < lb[0] + lb[2] and lb[0] < box[0] + box[2]
            and box[1] < lb[1] + lb[3] and lb[1] < box[1] + box[3])


def preflight_cash_box(screen, cfg, recalibrate=False):
    """Locate/verify the cash counter: text search first, then coin
    anchor, then heart-derived candidates -- each verified by actually
    reading it. Any candidate (including a previously SAVED box) that
    overlaps the lives counter or fails cash-counter shape sanity is
    discarded on sight: a mispositioned or inflated box still reads
    'fluently' (lone noise digits), so it never self-corrects without
    these guards. All-fail dumps crops to debug/."""
    saved = cfg.get("cash_box")
    if saved and _overlaps_lives(saved, cfg):
        dbg("saved cash_box overlaps the LIVES counter -- discarding it")
        cfg["cash_box"] = None
        save_config_value("cash_box", None)
    elif saved and not _plausible_cash_shape(saved):
        dbg(f"saved cash_box {saved} has implausible cash-counter "
            f"proportions -- discarding it")
        cfg["cash_box"] = None
        save_config_value("cash_box", None)
    original = cfg.get("cash_box")
    if not recalibrate and original and read_cash(screen, cfg) is not None:
        # The box reads -- but a box that clips the leading digit ALSO
        # reads ('$2,851' -> 851) and would stay wrong forever. Always try
        # a left-widen repair first, then the digit snap.
        lb = cfg.get("lives_box")
        fence = (lb[0] + lb[2]) if lb else 0.0
        repaired = _repair_clipped_cash_box(screen, original, min_x=fence)
        if repaired and repaired != original:
            cfg["cash_box"] = repaired
            save_config_value("cash_box", repaired)
            dbg(f"cash box widened left to recover clipped digits: "
                f"{repaired}")
            return True
        original_value = read_cash(screen, cfg)
        snapped = _snap_cash_box_to_digits(screen, original, min_x=fence)
        if snapped and snapped != original:
            cfg["cash_box"] = snapped
            if read_cash(screen, cfg) == original_value:
                save_config_value("cash_box", snapped)
                dbg(f"cash box re-fitted to the digit run: {snapped}")
                return True
            cfg["cash_box"] = original
        return True
    candidates = []
    text_box = _cash_box_from_text(screen, cfg)
    if text_box:
        candidates.append(("cash text search", text_box))
    coin = _cash_box_from_coin(screen)
    if coin:
        candidates.append(("coin icon", coin))
    candidates += [("heart-relative geometry", b)
                   for b in _cash_boxes_from_heart(screen)]
    candidates = [(how, b) for how, b in candidates
                  if not _overlaps_lives(b, cfg)
                  and _plausible_cash_shape(b)]
    for how, box in candidates:
        cfg["cash_box"] = box
        ok = False
        for _ in range(3):
            if read_cash(screen, cfg) is not None:
                ok = True
                break
            time.sleep(0.15)
        if ok:
            lb = cfg.get("lives_box")
            fence = (lb[0] + lb[2]) if lb else 0.0
            repaired = _repair_clipped_cash_box(screen, box, min_x=fence)
            if repaired:
                cfg["cash_box"] = repaired
                if read_cash(screen, cfg) is not None:
                    box = repaired
                    how += " + left widen"
                else:
                    cfg["cash_box"] = box
            box_value = read_cash(screen, cfg)
            snapped = _snap_cash_box_to_digits(screen, box, min_x=fence)
            if snapped:
                cfg["cash_box"] = snapped
                if read_cash(screen, cfg) == box_value:
                    box = snapped
                    how += " + digit snap"
                else:
                    cfg["cash_box"] = box
            if box != original:
                save_config_value("cash_box", box)
                print(f"Cash counter located via {how}. Saved "
                      f"cash_box={box} to config.json.")
            return True
    cfg["cash_box"] = original
    if candidates:
        debug_dir = Path(__file__).parent / "debug"
        debug_dir.mkdir(exist_ok=True)
        for n, (how, box) in enumerate(candidates):
            crop = screen.grab(box)
            cv2.imwrite(str(debug_dir / f"cash_cand{n}.png"), crop)
            cv2.imwrite(str(debug_dir / f"cash_cand{n}_proc.png"),
                        preprocess_round_crop(crop))
        dbg(f"cash: {len(candidates)} candidate boxes all failed to read "
            f"-- crops dumped to debug/cash_cand*.png")
    return read_cash(screen, cfg) is not None



def wait_for_cash(screen, cfg, amount, timeout=120):
    """Block until cash >= amount (used by 'wait_cash' plan actions)."""
    t_end = time.time() + timeout
    while time.time() < t_end:
        cash = read_cash(screen, cfg)
        if cash is not None and cash >= amount:
            return True
        time.sleep(1.0)
    print(f"      !! waited {timeout}s for ${amount} without seeing it -- "
          f"continuing anyway")
    return False


def _lives_box_from_heart(screen):
    """Find the red heart icon in the top-left HUD; the lives number sits
    just to its right. Red wraps around hue 0, so two HSV ranges."""
    icon = _hud_icon(screen, [0.0, 0.0, 0.20, 0.12],
                     (0, 140, 120), (8, 255, 255),
                     extra_range=((172, 140, 120), (180, 255, 255)))
    if icon is None:
        return None
    hcx, hcy, size = icon
    x0 = max(0.0, (hcx + 0.7 * size) / screen.w)
    y0 = max(0.0, (hcy - 0.55 * size) / screen.h)
    return [round(x0, 4), round(y0, 4),
            round(2.9 * size / screen.w, 4),   # short of the coin's edge
            round(1.1 * size / screen.h, 4)]


def read_lives(screen, cfg):
    """Current lives as an int, or None. Same exact-font template reader
    as cash; tesseract fallback."""
    box = cfg.get("lives_box")
    if not box:
        return None
    crop = screen.grab(box)
    value = _read_number_image(crop)
    if value is None and pytesseract is not None:
        processed = preprocess_round_crop(crop)
        text = pytesseract.image_to_string(
            processed,
            config="--psm 7 -c tessedit_char_whitelist=0123456789").strip()
        digits = re.sub(r"\D", "", text)
        value = int(digits) if digits else None
    if value is None:
        return None
    return value if value <= 1000 else None    # '2001' = coin-sliver junk


def preflight_lives_box(screen, cfg, recalibrate=False):
    """Locate/verify the lives counter. Needed for defeat detection --
    the round counter stays visible on the defeat screen."""
    original = cfg.get("lives_box")
    if not recalibrate and read_lives(screen, cfg) is not None:
        return True
    box = _lives_box_from_heart(screen)
    if box:
        cfg["lives_box"] = box
        if read_lives(screen, cfg) is not None:
            if box != original:
                save_config_value("lives_box", box)
                print(f"Lives counter located. Saved lives_box={box} to "
                      f"config.json.")
            return True
        cfg["lives_box"] = original
    return read_lives(screen, cfg) is not None


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def click_norm(screen, pt, button="left"):
    x, y = screen.to_pixels(*pt)
    pyautogui.moveTo(x, y, duration=0.15)
    click(button)


def counter_steady(screen, cfg):
    """Two successful parses out of four reads -- a single OCR flicker
    while a ghost is still held must never fake a 'placed!' verdict."""
    hits = 0
    for _ in range(4):
        if read_round(screen, cfg)[0] is not None:
            hits += 1
            if hits >= 2:
                return True
        time.sleep(0.12)
    return False


def counter_visible(screen, cfg, tries=3):
    """The round counter is HIDDEN while a tower ghost is held (a big X
    replaces it) -- which turns it into a free placement sensor:
      hotkey pressed, counter still visible  -> no ghost = can't afford yet
      clicked, counter visible again         -> ghost gone = tower placed
      clicked, counter still hidden          -> ghost stuck = invalid spot"""
    for _ in range(tries):
        if read_round(screen, cfg)[0] is not None:
            return True
        time.sleep(0.12)
    return False


def ui_clear(screen, cfg):
    """True only when the HUD is readable AND no upgrade panel is open on
    EITHER side. A left-side panel leaves the round counter visible, so
    the counter alone is not enough -- that blind spot let left panels
    linger, covering the cash/lives HUD and left-half towers."""
    return (counter_visible(screen, cfg, tries=2)
            and detect_panel_side(screen) is None)


def clear_ui(screen, cfg):
    """Restore a clean state: no ghost, no panel (either side), no pause,
    no leftover RESTART? dialog. Escalation: dismiss dialog -> unpause ->
    right-click (cancels ghosts) -> the detected panel's own X button ->
    clicks on known-empty ground. The dialog check runs FIRST because the
    dialog is modal: while it's up every other recovery click lands on
    nothing, and the round counter behind it stays readable -- so without
    this the 'clean state' test can pass with the dialog still open."""
    if looks_restart_confirm(screen):
        dbg("stray RESTART? dialog -- dismissing via CANCEL")
        click_norm(screen, RESTART_GEOM["cancel"])
        time.sleep(0.6)
    unpause_if_needed(screen, cfg)
    if ui_clear(screen, cfg):
        return True
    click("right")
    time.sleep(0.3)
    if ui_clear(screen, cfg):
        return True
    side = detect_panel_side(screen)
    if side:
        dbg(f"closing {side} panel via its X button")
        click_norm(screen, PANEL_GEOM["close_x"][side])
        time.sleep(0.35)
        if ui_clear(screen, cfg):
            return True
    else:
        # Counter dark but no panel: a stuck GHOST is the likely blocker.
        # Esc cancels placement; if nothing was actually held it opens the
        # pause menu instead -- which the next line immediately heals.
        dbg("no panel detected -- pressing Esc to cancel a stuck ghost")
        press_key("esc")
        time.sleep(0.4)
        unpause_if_needed(screen, cfg)
        if ui_clear(screen, cfg):
            return True
    for pt in [cfg["deselect_point"]] + SAFE_CLICKS:
        click_norm(screen, pt)
        time.sleep(0.35)
        if ui_clear(screen, cfg):
            return True
    return False


## ---------------------------------------------------------------------------
# Upgrade-panel vision: read prices, XP locks, and closed paths by LOOKING,
# never by pressing. Geometry measured from the user's screenshots
# (normalized to the game client, so it scales with resolution).
# ---------------------------------------------------------------------------

PANEL_GEOM = {
    # All y-values re-measured from in-game screenshots with the window
    # TITLE BAR subtracted -- the original table skipped that subtraction
    # and sat ~31px (0.026) low, which made the price strips sample the
    # next row's NAME text and clip glyphs into garbage prices.
    "portrait": {"left": [0.017, 0.143, 0.198, 0.244],
                 "right": [0.640, 0.141, 0.200, 0.246]},
    "title_band": {"left": [0.031, 0.052, 0.165, 0.050],
                   "right": [0.655, 0.052, 0.165, 0.050]},
    "brown_strips": [(0.052, 0.050), (0.386, 0.018), (0.833, 0.050)],
    "button_x": {"left": [0.130, 0.092], "right": [0.752, 0.092]},
    "rows_y": [[0.402, 0.114], [0.539, 0.113], [0.677, 0.113]],
    "text_strip": [0.58, 0.40],
    "close_x": {"left": [0.2156, 0.0883], "right": [0.8417, 0.0883]},
    "pause_band": [0.44, 0.018, 0.12, 0.050],
    "pause_body": [0.30, 0.220, 0.40, 0.060],
    "continue_btn": [0.695, 0.776],
}


def _white_fraction(img):
    """Fraction of near-white pixels -- the XP button's big up-arrow."""
    return float((img > 185).all(axis=2).mean())


def _blue_fraction(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (90, 60, 150), (112, 255, 255))
    return float((mask > 0).mean())


def _lightblue_fraction(img):
    """The defeat dialog's BODY blue -- lighter and less saturated than
    the gear/portrait blues, so it gets its own range."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (90, 40, 120), (118, 220, 255))
    return float((mask > 0).mean())


def looks_paused(screen):
    """Is the pause menu open? Two independent cues must BOTH fire: the
    white 'PAUSE' header text AND the tan menu body below it. White
    flowers/confetti can fake the first; only the menu supplies both."""
    band = screen.grab(PANEL_GEOM["pause_band"])
    if _white_fraction(band) < 0.06:
        return False
    body = screen.grab(PANEL_GEOM["pause_body"])
    return _brown_fraction(body) > 0.45


def _brown_fraction(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (10, 60, 80), (28, 220, 230))
    return float((mask > 0).mean())


def unpause_if_needed(screen, cfg):
    if looks_paused(screen):
        dbg("pause menu detected -- clicking CONTINUE")
        click_norm(screen, PANEL_GEOM["continue_btn"])
        time.sleep(0.6)
        return True
    return False


# Defeat screen + RESTART? confirmation dialog geometry, normalized to
# the game client area (re-measured from the user's windowed screenshots).
DEFEAT_GEOM = {
    # Blue bands the DEFEAT dialog is checked by. body_a used to sit at
    # y 0.310-0.360 -- INSIDE the orange DEFEAT lettering (y 0.28-0.37),
    # so the >=45% blue test hovered at its threshold and the bot often
    # didn't know it was dead. It now spans the clean run from below the
    # lettering into the ROUND panel's top (both blues pass the filter).
    "body_a": [0.32, 0.377, 0.36, 0.043],
    "body_b": [0.32, 0.506, 0.36, 0.020],   # gap between inner panels
    "title": [0.38, 0.285, 0.24, 0.065],    # orange DEFEAT lettering
}
RESTART_GEOM = {
    # The green-ribboned "RESTART?" confirmation dialog (same layout over
    # the defeat screen and over the pause menu).
    "header": [0.28, 0.355, 0.44, 0.040],   # green ribbon + white text
    "body": [0.30, 0.440, 0.40, 0.045],     # blue body above the question
    "cancel": [0.398, 0.675],               # orange CANCEL button
}


def looks_defeated(screen):
    """Is the DEFEAT screen up? Two light-blue body strips (below the
    DEFEAT lettering and in the gap between the dialog's inner panels)
    plus the orange DEFEAT title. Decoupled from OCR: on the defeat
    screen lives is a lone '0' that reads unreliably while the round
    counter stays readable, so digit-based exits can all fail at once."""
    strip_a = screen.grab(DEFEAT_GEOM["body_a"])
    strip_b = screen.grab(DEFEAT_GEOM["body_b"])
    if _lightblue_fraction(strip_a) < 0.45 \
            or _lightblue_fraction(strip_b) < 0.45:
        return False
    title = screen.grab(DEFEAT_GEOM["title"])
    hsv = cv2.cvtColor(title, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, (3, 120, 120), (24, 255, 255))
    return float((orange > 0).mean()) > 0.08


VICTORY_GEOM = {
    # The VICTORY screen: a wide orange "VICTORY" ribbon banner ABOVE where
    # the DEFEAT lettering sits, and a green NEXT button below the stats card
    # (the defeat screen shows a golden-yellow RESTART button there instead).
    "banner": [0.30, 0.10, 0.40, 0.16],   # orange ribbon (y clear of DEFEAT)
    "next": [0.40, 0.83, 0.20, 0.10],     # green NEXT button, lower-centre
}


def looks_victorious(screen):
    """Is the VICTORY screen up? Two independent cues must fire: a GREEN NEXT
    button in the lower-centre (where the loss screen puts a golden RESTART,
    so the button colour is the clean discriminator) AND the orange VICTORY
    ribbon across the top. It also refuses if the DEFEAT screen is up, so a
    frame that briefly shows both can never be scored as a win. Callers gate
    this on the final round, so a mid-run popup can't end a run early."""
    nxt = screen.grab(VICTORY_GEOM["next"])
    if _green_fraction(nxt) < 0.15:           # no green NEXT -> not a win
        return False
    banner = screen.grab(VICTORY_GEOM["banner"])
    hsv = cv2.cvtColor(banner, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, (3, 120, 120), (24, 255, 255))
    if float((orange > 0).mean()) < 0.06:     # no orange VICTORY ribbon
        return False
    return not looks_defeated(screen)


# The defeat screen comes in two layouts -- Home/Restart and, more often,
# Home/Restart/Review Map -- which put the RESTART button in different
# places, so a single calibrated point misses one of them (and on the
# 3-button layout the old point can land on HOME and bounce us to the main
# menu). This band spans just the low button row: clear of the orange
# DEFEAT lettering above and the fast-forward control off to the side, and
# the two neighbouring buttons are cyan, so the golden-yellow RESTART
# button is the only warm blob in it.
DEFEAT_BUTTON_BAND = (0.30, 0.70, 0.42, 0.20)   # x, y, w, h (normalized)


def find_defeat_restart(screen, cfg):
    """Normalized centre of the defeat-screen RESTART button, found by its
    yellow colour so both button layouts work regardless of where it lands.
    Falls back to the calibrated cfg['defeat_restart'] point when the button
    can't be seen (wrong screen, odd theme)."""
    fallback = cfg.get("defeat_restart")
    try:
        bx, by, bw, bh = DEFEAT_BUTTON_BAND
        strip = screen.grab([bx, by, bw, bh])
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (14, 110, 140), (38, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in contours:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            if area < 0.0006 * screen.w * screen.h:   # skip text/specks
                continue
            if w > 0.20 * screen.w or h > 0.22 * screen.h:
                continue
            if not (0.6 <= w / max(h, 1) <= 2.4):      # a button-ish block
                continue
            if best is None or area > best[0]:
                best = (area, x, y, w, h)
        if best is None:
            dbg("defeat RESTART not found by colour -- using calibrated point")
            return fallback
        _, x, y, w, h = best
        nx = bx + (x + w / 2) / screen.w
        ny = by + (y + h / 2) / screen.h
        return [round(nx, 4), round(ny, 4)]
    except Exception:
        return fallback


def looks_restart_confirm(screen):
    """Is the 'RESTART?' confirmation dialog up? Green header ribbon
    carrying white text, with the dialog's light-blue body below it.
    This is the gate that makes restarts honest: a restart button press
    only counts once this dialog is SEEN, and its confirm click only
    counts once it's gone -- no more firing clicks into a screen that
    didn't have the expected buttons on it."""
    header = screen.grab(RESTART_GEOM["header"])
    if _green_fraction(header) < 0.35 or _white_fraction(header) < 0.04:
        return False
    return _lightblue_fraction(screen.grab(RESTART_GEOM["body"])) > 0.45


def _green_fraction(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (45, 100, 100), (80, 255, 255))
    return float((mask > 0).mean())


def _classify_price_text(text):
    """'$510' or '510' -> ('cash', 510); anything with X/P -> ('xp', None).
    The XP-safety no longer hinges on OCR: the white up-arrow pixel test
    runs BEFORE any text is read, so digits reaching this point are a
    price even when tesseract drops the stylized '$'."""
    t = text.upper()
    if "X" in t or "P" in t:
        return "xp", None
    digits = re.sub(r"\D", "", t)
    if digits:
        return "cash", int(digits)
    return "none", None




def expected_panel_side_for_tower(at):
    """BTD6 opens the upgrade panel on the side opposite the selected
    tower so the panel does not cover the tower. Towers left of the
    screen midpoint open the panel on the right; towers right of the
    midpoint open it on the left. This is only a prediction -- the bot
    still verifies the actual side by looking for the panel wood."""
    try:
        x = float(at[0])
    except Exception:
        return None
    return "right" if x < 0.5 else "left"


def _detect_panel_side(screen, preferred=None):
    """Detect which upgrade panel is visible. When a tower-position
    prediction is available, check that side first so a small stale piece
    of the old panel on the other side does not win a tie."""
    order = []
    if preferred in ("left", "right"):
        order.append(preferred)
    order.extend(side for side in ("left", "right") if side not in order)
    x_of = {"left": PANEL_GEOM["title_band"]["left"][:1]
            + [PANEL_GEOM["title_band"]["left"][2]],
            "right": PANEL_GEOM["title_band"]["right"][:1]
            + [PANEL_GEOM["title_band"]["right"][2]]}
    best = None
    for side in order:
        sx, sw = x_of[side]
        hits = 0
        score = 0.0
        for sy, sh in PANEL_GEOM["brown_strips"]:
            frac = _brown_fraction(screen.grab([sx, sy, sw, sh]))
            score += frac
            if frac >= 0.35:
                hits += 1
        if hits >= 2:
            if side == preferred:
                return side
            if best is None or (hits, score) > best[0]:
                best = ((hits, score), side)
    return best[1] if best else None


def _read_upgrade_price_box(screen, box):
    """Read an upgrade price from an explicit normalized box. Like cash,
    upgrade prices are exact-or-unknown: Tesseract is not trusted. If the
    visible box reads a suffix such as 50, probe a little farther left and
    only adopt a larger value when it cleanly ends with that suffix, e.g.
    850 -> 50. This prevents clipped prices from entering prices.json."""
    if not box:
        return None
    base = _read_number_image(screen.grab(box))
    x, y, w, h = box
    best = None
    for grow in (0.0, 0.35, 0.60, 0.90, 1.20):
        nx = max(0.0, x - grow * h)
        cand = [nx, y, w + (x - nx), h]
        val = _read_number_image(screen.grab(cand))
        if val is None:
            continue
        if base is not None and val != base:
            if not (val > base and str(val).endswith(str(base))):
                # Conflicting reads from nearby crops mean the row is not
                # trustworthy enough to save or press from.
                continue
        if best is None or val > best:
            best = val
    return best


def _panel_pixel_box_to_norm(parent_box, px_box, parent_shape):
    """Convert a pixel box inside a grabbed parent crop back into a
    normalized screen box."""
    px, py, pw, ph = px_box
    ih, iw = parent_shape[:2]
    return [round(parent_box[0] + px / iw * parent_box[2], 4),
            round(parent_box[1] + py / ih * parent_box[3], 4),
            round(pw / iw * parent_box[2], 4),
            round(ph / ih * parent_box[3], 4)]

def _vivid_nongreen_fraction(img):
    """Fraction of saturated, bright, NOT-green pixels. Every tower
    portrait background is a solid vivid color (blue, purple, ...) while
    the map underneath is grass green and desaturated gray path -- so this
    is high exactly when a portrait is present, whatever its category."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    vivid = (s > 90) & (v > 110)
    green = (h > 35) & (h < 85)
    return float((vivid & ~green).mean())


def detect_panel_side(screen, preferred=None):
    """Which side is an upgrade panel open on, if any? Portrait COLOR is
    useless: backgrounds are category-themed and Military's is GREEN --
    on a grass map, a sniper's panel was literally camouflaged from the
    old detector. Instead, triangulate the panel's WOOD: brown must show
    at 2 of 3 fixed heights (title band, under-portrait gap, sell bar).
    Grass/path/bloons can't fake brown; the HUD's gold coin covers far
    too little of one strip to matter.

    preferred may be the side predicted from tower position; it is checked
    first but the result is still vision-verified."""
    return _detect_panel_side(screen, preferred=preferred)



def _hsv_blobs(img, lo, hi, min_area):
    """Connected-component blobs of an HSV range. NOT contour-based:
    RETR_EXTERNAL hides a button whose surroundings (the grass field
    wrapping the panel) form an enclosing ring."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lo, hi)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            out.append((int(x), int(y), int(w), int(h)))
    return out


def scan_panel_rows(screen, side):
    """Judge each upgrade row's STATE by colored evidence at the KNOWN
    row positions (static geometry, measured from real screenshots).
    The previous version inferred a row grid from the SELL button plus
    pitch constants; those constants were wrong for the real panel and
    shifted every read by one slot -- rows are no longer inferred at all.
      green button blob at the row -> ('cash', price) or ('xp', None)
      red price text at the row    -> ('short', price)
      neither                      -> closed (absent from the dict)"""
    result = {}
    bx, bw = PANEL_GEOM["button_x"][side]
    for k, (ry, rh) in enumerate(PANEL_GEOM["rows_y"]):
        band_box = [max(0.0, bx - 0.010), max(0.0, ry - 0.30 * rh),
                    bw + 0.020, rh * 1.60]
        band = screen.grab(band_box)
        bh_px, bw_px = band.shape[:2]
        greens = [b for b in _hsv_blobs(band, (40, 90, 90), (85, 255, 255),
                                        min_area=0.10 * bh_px * bw_px)
                  if 0.45 <= b[2] / max(b[3], 1) <= 1.8]
        if greens:
            x, y, w, h = max(greens, key=lambda b: b[2] * b[3])
            btn = band[y:y + h, x:x + w]
            upper = btn[:max(1, int(h * 0.55))]
            if _white_fraction(upper) > 0.22:
                result[k] = ("xp", None)
                continue
            py = y + int(0.52 * h)
            ph = max(1, int(0.45 * h))
            price_box = _panel_pixel_box_to_norm(
                band_box, (x, py, w, ph), band.shape)
            price = _read_upgrade_price_box(screen, price_box)
            result[k] = ("cash", price) if price else ("unread", None)
            continue
        # RED price check, self-aligning: a fixed core slice decapitated
        # off-nominal rows into strings of '1's. Instead, find red
        # components across the FULL band, cluster them by baseline, and
        # read only the cluster nearest the row's own center -- neighbor
        # rows' fragments live in other clusters and are ignored.
        b, g, r = (band[..., 0].astype(int), band[..., 1].astype(int),
                   band[..., 2].astype(int))
        red_mask = ((r > 150) & (r > g + 90)
                    & (r > b + 90)).astype(np.uint8)
        if red_mask.mean() > 0.008:
            ncc, lab, stats, _ = cv2.connectedComponentsWithStats(red_mask, 8)
            comps = [(stats[i, 1] + stats[i, 3] / 2, i)
                     for i in range(1, ncc) if stats[i, 4] >= 10]
            if comps:
                comps.sort()
                clusters, cur = [], [comps[0]]
                for yc, i in comps[1:]:
                    if yc - cur[-1][0] <= 0.22 * bh_px:
                        cur.append((yc, i))
                    else:
                        clusters.append(cur)
                        cur = [(yc, i)]
                clusters.append(cur)
                row_center = 0.65 * bh_px      # where this row's text sits
                best = min(clusters, key=lambda cl: abs(
                    float(np.mean([yy for yy, _ in cl])) - row_center))
                keep = {i for _, i in best}
                iso = np.zeros_like(band)
                iso[np.isin(lab, list(keep))] = 255
                # Red shortfall prices are isolated by color. They are never
                # used to press an upgrade, but they can be saved as a price
                # hint, so keep them on the exact-font reader as well.
                result[k] = ("short", _read_number_image(iso))
    return result


def read_upgrade_row(screen, side, row_i):
    """Classify one upgrade row (static-geometry fallback path; the
    self-locating scanner is preferred). All price reading goes through
    the exact-font template reader -- tesseract read a red '$340' as
    '7340' and poisoned the price book, so it is out of every money path.
      ('cash', price) / ('short', price) / ('xp', None) /
      ('closed', None) / ('unread', None)"""
    bx, bw = PANEL_GEOM["button_x"][side]
    ry, rh = PANEL_GEOM["rows_y"][row_i]
    button = screen.grab([bx, ry, bw, rh])
    green = _green_fraction(button)
    ty, th = PANEL_GEOM["text_strip"]
    price_box = [bx, ry + ty * rh, bw, th * rh]
    crop = screen.grab(price_box)

    if green >= 0.10:
        upper = button[:max(1, int(button.shape[0] * 0.55))]
        if _white_fraction(upper) > 0.22:
            return "xp", None            # the big white unlock arrow
        price = _read_upgrade_price_box(screen, price_box)
        if price:
            return "cash", price

    # No (readable) green: a RED price means a real-but-unaffordable row.
    b, g, r = (crop[..., 0].astype(int), crop[..., 1].astype(int),
               crop[..., 2].astype(int))
    red_mask = (r > 150) & (r > g + 90) & (r > b + 90)
    if red_mask.mean() > 0.02:
        iso = np.zeros_like(crop)
        iso[red_mask] = 255
        return "short", _read_number_image(iso)

    if green < 0.10:
        return "closed", None
    debug_dir = Path(__file__).parent / "debug"
    debug_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(debug_dir / f"panel_row{row_i}_unread.png"), crop)
    return "unread", None


def read_upgrade_panel(screen, cfg, tower=None):
    """With a tower selected, read all three rows in one pass. Also feeds
    everything it sees into the books: visible prices are recorded WITHOUT
    buying, visible XP locks are recorded WITHOUT pressing. Returns
    (side, rows) or (None, None) if the panel wasn't found."""
    side = detect_panel_side(screen)
    if side is None:
        return None, None
    rows = [read_upgrade_row(screen, side, i) for i in range(3)]
    if tower:
        ttype = tower["tower"].lower()
        for i, (state, price) in enumerate(rows):
            tier = tower["path"][i] + 1
            if state == "cash" and price:
                record_price(price_key(ttype, i, tier), price)
            elif state == "xp":
                mark_locked(ttype, i, tier)
    return side, rows


# Fallback offsets used only when NO mask is loaded. Two rings: small
# nudges, then wider jumps that can actually clear a path arm.
NUDGES = [(0, 0), (0.012, 0), (-0.012, 0), (0, 0.02), (0, -0.02),
          (0.012, 0.02), (-0.012, 0.02), (0.012, -0.02), (-0.012, -0.02),
          (0.03, 0), (-0.03, 0), (0, 0.045), (0, -0.045),
          (0.03, 0.045), (-0.03, 0.045), (0.03, -0.045), (-0.03, -0.045)]


def placement_candidates(spot, tower=None, limit=12):
    """Where to try placing, in order. First the planned spot itself; then
    the nearest known-good mask points. Large-footprint towers try the 6
    nearest ROOMY points, then the 6 nearest strict points -- the game is
    the final judge, and a rejected click only costs a couple seconds."""
    candidates = [list(spot)]
    seen = {(round(spot[0], 4), round(spot[1], 4))}
    if tower in LARGE_TOWERS:
        pools = [(MASK_ROOMY, 6), (MASK_POINTS, 6)]
    elif MASK_POINTS:
        pools = [(MASK_POINTS, limit)]
    else:
        pools = []
    for pool, take in pools:
        for q in sorted(pool, key=lambda p: (p[0] - spot[0]) ** 2
                        + (p[1] - spot[1]) ** 2)[:take]:
            key = (round(q[0], 4), round(q[1], 4))
            if key not in seen:
                seen.add(key)
                candidates.append([q[0], q[1]])
    if len(candidates) == 1:                   # no mask -> blind nudges
        candidates += [[spot[0] + dx, spot[1] + dy] for dx, dy in NUDGES[1:]]
    return candidates


def _probe_occupied(screen, cfg, spot):
    """Is a monkey sitting on `spot`? With NO ghost held, a bare left click on
    an occupied spot opens that tower's upgrade panel, while a click on empty
    placeable ground does nothing. That is how the executor turns a mystery
    placement failure into a KNOWN-occupied point it can avoid from then on.
    Drops any held ghost first and closes the panel it opens, so the UI is
    left clean; if a ghost/overlay can't be cleared it declines to probe
    rather than risk a stray click."""
    click("right")                        # drop any held ghost
    time.sleep(0.2)
    if not counter_visible(screen, cfg, tries=2):
        return False                      # ghost/overlay stuck -- don't risk it
    click_norm(screen, spot)
    time.sleep(0.3)
    if detect_panel_side(screen):
        clear_ui(screen, cfg)             # close the panel the probe opened
        return True
    return False


def act_place(screen, cfg, action):
    """Try to place a tower. Returns (status, landed):
      ("placed", [x, y])  -- tower is down at that coordinate
      ("broke", None)     -- no ghost appeared: can't afford it YET. One
                             press, one check, immediate return -- the
                             caller waits for income; this never counts
                             as a placement failure.
      ("no_spot", None)   -- ghost held but every candidate click was
                             rejected: genuinely unplaceable around here."""
    tl = action["tower"].lower()
    large = tl in LARGE_TOWERS
    key = TOWER_HOTKEYS[tl]
    spot = action["at"]
    candidates = placement_candidates(spot, tl)
    # Never click ON a spot we KNOW is taken -- a tower this run already
    # placed (`avoid`), or one a probe caught a monkey sitting on (`occupied`,
    # a persistent per-episode blacklist). Those clicks can only fail or select
    # the neighbor; watching the bot try to stack glue on glue is silly.
    occupied = action.get("occupied")            # mutable list, grown on probe
    avoid = list(action.get("avoid") or [])
    if occupied:
        avoid += list(occupied)
    min_d = 0.05 if large else 0.03
    candidates = free_placement_spots(
        candidates, avoid, min_d, MASK_ROOMY if large else MASK_POINTS, spot)
    deadline = time.time() + action.get("timeout", 60)

    cash_before = read_cash(screen, cfg)
    press_key(key)                            # try to pick up the ghost
    time.sleep(0.3)
    if counter_visible(screen, cfg, tries=2):
        return "broke", None                  # no ghost -> just money

    attempt = 0
    max_tries = min(len(candidates), 12)
    baseline = cash_before
    target = [round(spot[0], 4), round(spot[1], 4)]
    while attempt < max_tries and time.time() < deadline:
        target = candidates[attempt]
        target = [round(target[0], 4), round(target[1], 4)]
        fresh = read_cash_confirmed(screen, cfg)
        if fresh is not None:
            baseline = fresh                  # per-click cash baseline
        click_norm(screen, target)
        time.sleep(0.35)
        if counter_steady(screen, cfg):       # ghost truly gone -> placed
            note = ""
            cash_after = read_cash(screen, cfg)
            if cash_before is not None and cash_after is not None \
                    and cash_after < cash_before:
                record_price(price_key(action["tower"].lower()),
                             cash_before - cash_after)
                note = f"  (${cash_before} -> ${cash_after})"
            if attempt:
                note += f"  [moved to {target}]"
            print(f"      placed "
                  f"{action.get('name') or action['tower']}{note}")
            return "placed", target
        # The counter says nothing landed -- but CASH is the ground
        # truth. If money left the wallet right after our click, the
        # tower IS down (a panel or hover UI was hiding the counter) and
        # every further "retry" would try to stack a copy on top of it.
        spent = read_cash_confirmed(screen, cfg)
        if baseline is not None and spent is not None \
                and spent <= baseline - 100:
            record_price(price_key(action["tower"].lower()),
                         baseline - spent)
            print(f"      placed {action.get('name') or action['tower']}"
                  f"  (cash-verified ${baseline} -> ${spent})")
            clear_ui(screen, cfg)     # whatever hid the counter, clear it
            return "placed", target
        # Rejected: cancel the ghost and VERIFY it's gone -- some game
        # states (e.g. nudge mode) leave a stuck ghost that right-click
        # doesn't clear, and re-pressing the hotkey then does nothing.
        click("right")
        time.sleep(0.25)
        if not counter_visible(screen, cfg, tries=2):
            dbg("ghost stuck after right-click -- pressing Esc")
            press_key("esc")
            time.sleep(0.4)
            unpause_if_needed(screen, cfg)
        attempt += 1
        if attempt < max_tries:
            press_key(key)
            time.sleep(0.25)
            if counter_visible(screen, cfg, tries=1):
                return "broke", None          # affordability flapped
    click("right")                            # tidy up any held ghost
    spent = read_cash_confirmed(screen, cfg)
    if baseline is not None and spent is not None \
            and spent <= baseline - 100:
        # Late catch: a placement landed during the attempts without
        # either sensor seeing it at the time. Better an approximate
        # position than a phantom retry stacking towers.
        print(f"      placed {action.get('name') or action['tower']}"
              f"  (cash-verified late, ${baseline} -> ${spent})")
        clear_ui(screen, cfg)
        return "placed", target
    # Still nothing placed. Before giving up, find out WHY the planned spot
    # refused us: a bare click there (no ghost held) that opens an upgrade
    # panel means a monkey is sitting on it. Blacklist that point so this
    # item's own retries -- and every later buy -- relocate instead of
    # clicking the same dead spot again.
    if occupied is not None and _probe_occupied(screen, cfg, spot):
        pt = [round(spot[0], 4), round(spot[1], 4)]
        if pt not in occupied:
            occupied.append(pt)
            dbg(f"{tl}: {pt} is occupied by a monkey -- blacklisted this run")
    return "no_spot", None


def act_upgrade(screen, cfg, action, tower=None, timeout=8):
    """Buy the requested upgrade tiers, deciding by SIGHT first. After the
    panel opens, the bot reads each row: '$price' = buyable, 'XP' = locked
    (recorded, never pressed -- your XP is yours), no green button = path
    closed/maxed. A key is only pressed on a row visually confirmed as a
    cash upgrade we can afford. Returns a status string:
      "bought" / "locked" / "closed" / "broke" / "no_select" / "unread"."""
    ttype = tower["tower"].lower() if tower else None

    # Skip tiers already known to be XP-locked, before opening anything.
    if tower:
        for i, count in enumerate(action["path"]):
            if count and is_locked(ttype, i, tower["path"][i] + 1):
                return "locked"

    # A leftover panel from a previous action physically COVERS towers
    # underneath it -- clicks would hit UI, not the map. Clean state first.
    if not ui_clear(screen, cfg):
        clear_ui(screen, cfg)

    # Selection success = a panel APPEARED (either side). The old check --
    # "did the round counter hide?" -- only worked for right-side panels;
    # left panels leave the counter visible and were invisible to it.
    expected_side = expected_panel_side_for_tower(action.get("at"))
    side = None
    for _ in range(3):
        click_norm(screen, action["at"])
        time.sleep(cfg["action_delay"])
        side = detect_panel_side(screen, preferred=expected_side)
        if side:
            if expected_side and side != expected_side:
                dbg(f"panel opened on {side}; expected {expected_side} "
                    f"from tower x={action['at'][0]:.3f}")
            break
    if not side:
        dbg(f"select failed at {action['at']} (no panel appeared)")
        if tower is not None:
            tower["_noselect"] = tower.get("_noselect", 0) + 1
            if tower["_noselect"] == 1:
                debug_dir = Path(__file__).parent / "debug"
                debug_dir.mkdir(exist_ok=True)
                shot = debug_dir / (f"no_select_{ttype}_"
                                    f"{int(time.time()) % 100000}.png")
                cv2.imwrite(str(shot), screen.grab())
                dbg(f"full-frame screenshot saved: {shot.name}")
        clear_ui(screen, cfg)      # close anything half-open before leaving
        return "no_select"

    # One full harvest per tower (prices + locks recorded); the
    # SELF-LOCATING scanner runs first (aspect/resolution independent),
    # static geometry only as fallback.
    if tower is not None and not tower.get("_scanned"):
        scan = scan_panel_rows(screen, side)
        if scan is not None:
            rows = [scan.get(i, ("closed", None)) for i in range(3)]
        else:
            rows = [read_upgrade_row(screen, side, i) for i in range(3)]
        ttype_l = tower["tower"].lower()
        for i, (state, price) in enumerate(rows):
            t = tower["path"][i] + 1
            k = price_key(ttype_l, i, t)
            if state == "cash" and price:
                record_price(k, price, src="seen")   # green: cleanest
            elif state == "short" and price and k not in PRICES:
                record_price(k, price, src="short")  # red: unverified
            elif state == "xp":
                mark_locked(ttype_l, i, t)
        tower["_side"] = side
        tower["_scanned"] = True
        dbg(f"panel[{side}] {ttype}: " + " / ".join(
            f"{s}{'' if p is None else ' $' + str(p)}"
            for s, p in rows))
    else:
        rows = [None, None, None]     # lazy: read only the row being bought

    def broke_status(have=None, need=None):
        if have is not None and need is not None:
            return f"broke (${have}/${need})"
        if need is not None:
            return f"broke (cash unreadable/${need})"
        if have is not None:
            return f"broke (${have}/?)"
        return "broke"

    status = "bought"
    for i, count in enumerate(action["path"]):
        for _ in range(count):
            tier = tower["path"][i] + 1 if tower else None
            pkey = (price_key(ttype, i, tier) if tower else None)
            if tower and is_locked(ttype, i, tier):
                status = "locked"
                break
            if rows is not None and rows[i] is None and side:
                scan = scan_panel_rows(screen, side)
                if scan is not None:
                    rows[i] = scan.get(i, ("closed", None))
                else:
                    rows[i] = read_upgrade_row(screen, side, i)
                if tower:                       # book the lazy read too
                    state, seen_price = rows[i]
                    if state == "cash" and seen_price:
                        record_price(pkey, seen_price, src="seen")
                    elif state == "short" and seen_price \
                            and pkey not in PRICES:
                        record_price(pkey, seen_price, src="short")
                    elif state == "xp":
                        mark_locked(ttype, i, tier)
            info = rows[i] if rows else None
            if info:
                state, seen_price = info
                if state == "xp":
                    if IGNORE_LOCKS:
                        # Everything is unlocked, so an 'xp' row can only be a
                        # MISREAD -- don't block the path, just retry it.
                        status = "unread"
                        break
                    status = "locked"          # already recorded by reader
                    break
                if state == "short":
                    # Greyed button + red price: a real upgrade we can't
                    # afford yet. Price is booked; the caller saves up.
                    status = broke_status(read_cash(screen, cfg), seen_price)
                    break
                if state == "closed":
                    if tower:
                        closed = tower.setdefault("closed_paths", [])
                        if i not in closed:
                            closed.append(i)
                    status = "closed"
                    break
                if state == "unread":
                    status = "unread"          # can't be sure: don't press
                    break
            known = dict.get(PRICES, pkey) if pkey else None   # gate: learned
            cash = read_cash(screen, cfg)           # raw: for the buy delta
            # Affordability is decided on the FLOOR-protected value, not the
            # raw read: a clipped '$1' misread would otherwise fake a "broke"
            # and, retried, get the whole upgrade dropped. When the read is
            # implausibly low the gate falls back to "unreadable" (proceed and
            # let the row-based verification below judge), never a false broke.
            sc = action.get("sane_cash")
            gate_cash = sc() if sc is not None else cash
            if known is not None and gate_cash is not None and gate_cash < known:
                status = broke_status(gate_cash, known)  # caller waits, closed
                break
            press_key(UPGRADE_KEYS[i])
            time.sleep(0.5)
            after = read_cash(screen, cfg)
            if cash is None or after is None:
                if tower:
                    tower["path"][i] += 1      # can't verify; trust
            elif after <= cash - 10:           # verified purchase
                if pkey:
                    record_price(pkey, cash - after)
                if tower:
                    tower["path"][i] += 1
            else:
                if info and info[0] == "cash":
                    # A GREEN row is the game itself saying "affordable",
                    # and the no-move verdict rests on the same cash reads
                    # that misread in the first place. Ask the ROW before
                    # judging: if its price/state moved on, the purchase
                    # actually landed and the cash reads were noise --
                    # missing that would desync the tier tracking AND
                    # poison the watermark.
                    re_read = None
                    if side:
                        scan = scan_panel_rows(screen, side)
                        re_read = (scan or {}).get(i) \
                            or read_upgrade_row(screen, side, i)
                    if re_read is not None and re_read[0] != "unread" \
                            and re_read != info:
                        dbg(f"cash read missed a landed buy (path {i + 1}:"
                            f" row moved {info} -> {re_read})")
                        if pkey and info[1]:
                            record_price(pkey, info[1], src="seen")
                        if tower:
                            tower["path"][i] += 1
                        if rows is not None:
                            rows[i] = re_read
                        continue
                    # Row unchanged: the press really didn't take (stale
                    # green after cash dropped, or an input hiccup) --
                    # NOT an XP lock; marking it locked would be poison.
                    dbg(f"press didn't take on a cash row (path {i + 1}) "
                        f"-- treating as broke")
                    status = broke_status(after if after is not None else cash,
                                           info[1] if info else known)
                    break
                # No visual info and an affordable press moved nothing: the
                # safety net. Normally that IS an XP lock -- record it and stop
                # touching the path. But with --no-locks the account has
                # everything unlocked, so a no-move press is a transient hiccup
                # (a cash misread, a dropped keypress), NOT a lock: report it as
                # broke so the caller retries later instead of blocking the path
                # for the whole run (the "randomly blocks it for no reason"
                # complaint).
                if IGNORE_LOCKS:
                    status = broke_status(
                        after if after is not None else cash,
                        info[1] if info else known)
                else:
                    if tower:
                        mark_locked(ttype, i, tier)
                    status = "locked"
                break
            # Multi-tier buys: refresh this row so the NEXT tier's state
            # (new price / new XP wall) is read before another press.
            if rows is not None and count > 1 and side:
                rows[i] = read_upgrade_row(screen, side, i)
        if status != "bought":
            break
    if not clear_ui(screen, cfg):
        print("      !! panel would not close -- HUD blocked")
    return status


def act_press(screen, cfg, action):
    """Generic key press -- e.g. abilities on '1', '2', '3'."""
    press_key(action["key"])


ACTION_FUNCS = {"place": act_place, "upgrade": act_upgrade, "press": act_press}


def describe(a):
    if a["do"] == "place":
        return f"place {a['tower']} at {a['at']}"
    if a["do"] == "upgrade":
        return f"upgrade tower at {a['at']} by {a['path']}"
    if a["do"] == "press":
        return f"press '{a['key']}'"
    return str(a)


def validate_plan(plan):
    for a in plan["actions"]:
        if a.get("do") not in ACTION_FUNCS:
            sys.exit(f"Unknown action type in plan: {a}")
        if a["do"] == "place" and a["tower"].lower() not in TOWER_HOTKEYS:
            sys.exit(f"Unknown tower '{a['tower']}' -- add its hotkey to "
                     f"TOWER_HOTKEYS in btd6_bot.py")
        if "round" not in a:
            sys.exit(f"Action is missing a \"round\": {a}")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_locate(args):
    cfg = load_config()
    screen, _ = make_screen(cfg)
    print("Hover the mouse over the game. Copy the norm=[x, y] values into "
          "your plan file. Ctrl+C to stop.\n")
    try:
        while True:
            x, y = pyautogui.position()
            nx, ny = screen.to_norm(x, y)
            print(f"\r  pixel=({x:>5},{y:>5})   norm=[{nx:0.3f}, {ny:0.3f}]   ",
                  end="", flush=True)
            time.sleep(0.15)
    except KeyboardInterrupt:
        print("\nDone.")


def cmd_watch(args):
    cfg = load_config()
    setup_tesseract(cfg)
    screen, _ = make_screen(cfg)
    debug_dir = Path(__file__).parent / "debug"
    debug_dir.mkdir(exist_ok=True)
    if not preflight_round_box(screen, cfg, recalibrate=True):
        print("Auto-locate failed. Is a map loaded with the round counter "
              "visible? debug/round_search_strip.png shows what was "
              "searched. You can still set \"round_box\" by hand.\n")
    if not preflight_cash_box(screen, cfg, recalibrate=True):
        print("Couldn't locate the cash counter (non-fatal: placement "
              "still self-verifies, but cash gating is off).\n")
    if not preflight_lives_box(screen, cfg, recalibrate=True):
        print("Couldn't locate the lives counter (needed for defeat "
              "detection in farm/play).\n")
    print("Reading the round counter once per second. With a game loaded, "
          "'parsed' should match the round on screen.")
    print(f"Debug images -> {debug_dir}/round_crop.png and "
          f"round_processed.png (adjust \"round_box\" in config.json until "
          f"the crop shows just the round text). Ctrl+C to stop.\n")
    try:
        while True:
            value, text, crop, processed = read_round(screen, cfg)
            cv2.imwrite(str(debug_dir / "round_crop.png"), crop)
            cv2.imwrite(str(debug_dir / "round_processed.png"), processed)
            print(f"  raw={text!r:<12} parsed={value}   "
                  f"cash={read_cash(screen, cfg)}   "
                  f"lives={read_lives(screen, cfg)}")
            la = _lightblue_fraction(screen.grab(DEFEAT_GEOM["body_a"]))
            lb2 = _lightblue_fraction(screen.grab(DEFEAT_GEOM["body_b"]))
            tt = screen.grab(DEFEAT_GEOM["title"])
            oo = float((cv2.inRange(cv2.cvtColor(tt, cv2.COLOR_BGR2HSV),
                                    (3, 120, 120),
                                    (24, 255, 255)) > 0).mean())
            print(f"    defeat-signals: blueA={la:.2f} blueB={lb2:.2f} "
                  f"orange={oo:.2f} (need A,B>=0.45, orange>0.08) -> "
                  f"{looks_defeated(screen)}   "
                  f"restart-dialog={looks_restart_confirm(screen)}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nDone.")


def ring_red_shift(cur, clean, cx, cy, r_in, r_out):
    """How much redder did a ring around (cx, cy) get compared to the clean
    map? The ring sits inside the ghost's range circle but OUTSIDE the
    monkey's body -- monkey fur is brown (red-heavy!), which is exactly what
    fooled the first version of this detector. Comparing against the clean
    frame cancels out map art (red flowers, brown crates, ...) too.

    Valid spot: gray translucent circle -> all channels shift equally -> ~0.
    Invalid spot: red circle -> red channel shifts far more -> 20-60."""
    h, w = cur.shape[:2]
    y0, y1 = max(cy - r_out, 0), min(cy + r_out + 1, h)
    x0, x1 = max(cx - r_out, 0), min(cx + r_out + 1, w)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    ring = (d2 >= r_in ** 2) & (d2 <= r_out ** 2)
    if not ring.any():
        return 0.0
    diff = cur[y0:y1, x0:x1].astype(int) - clean[y0:y1, x0:x1].astype(int)
    diff = diff[ring]                       # N x 3, BGR
    # Per-pixel red shift, then the MEDIAN: a genuine invalid-tint covers
    # the whole ring so the median is high, while transient red things --
    # falling confetti, firework flashes (hi, seasonal skins), a sliver of
    # brown monkey fur -- only cover a fraction of pixels and can't move it.
    shift_px = diff[:, 2] - (diff[:, 0] + diff[:, 1]) / 2.0
    return float(np.median(shift_px))


def measure_tower_range(screen, cfg, tower, debug_dir=None):
    """Measure a tower's ACTUAL range as a fraction of screen width, by
    hovering its ghost over an UNPLACEABLE spot -- where BTD6 paints the whole
    range circle RED -- and walking a thin ring outward until that red tint
    ends. That radius is the range. Reuses the very red-shift detector the
    scanner uses for placeability, so it needs no new calibration idea; the
    result replaces the rough hardcoded `range` the brain guesses coverage
    from (a tack modelled at 1.9% of track is why placements looked blind).
    Returns the range fraction, or None if no red circle could be sized (the
    ghost never appeared, or no on-track hover spot was found)."""
    r_in = max(6, int(0.020 * screen.h))          # scan's placeability ring
    r_out = max(r_in + 4, int(0.050 * screen.h))
    clean = screen.grab()
    pyautogui.moveTo(*screen.to_pixels(0.45, 0.50))
    time.sleep(0.2)
    press_key(TOWER_HOTKEYS[tower])                # pick up the ghost
    time.sleep(0.45)
    cx0, cy0 = int(0.45 * screen.w), int(0.50 * screen.h)
    held = screen.grab()
    if np.abs(held[cy0 - r_out:cy0 + r_out, cx0 - r_out:cx0 + r_out].astype(int)
              - clean[cy0 - r_out:cy0 + r_out, cx0 - r_out:cx0 + r_out]
              .astype(int)).mean() < 2.0:
        click("right")
        return None                                # no ghost (unaffordable?)

    # Find the reddest (most clearly on-track) hover spot: the range circle is
    # only red where the cursor itself sits on an unplaceable cell.
    best = None
    for hx in (0.34, 0.42, 0.50, 0.26, 0.58, 0.38, 0.46, 0.30):
        for hy in (0.50, 0.42, 0.58, 0.35, 0.66):
            pyautogui.moveTo(*screen.to_pixels(hx, hy))
            time.sleep(0.10)
            cur = screen.grab()
            cx, cy = int(hx * screen.w), int(hy * screen.h)
            red = ring_red_shift(cur, clean, cx, cy, r_in, r_out)
            if best is None or red > best[0]:
                best = (red, hx, hy)
    red0, hx, hy = best
    if red0 < 10.0:                                # never landed on the track
        click("right")
        return None
    pyautogui.moveTo(*screen.to_pixels(hx, hy))
    time.sleep(0.15)
    cur = screen.grab()
    click("right")                                 # done: drop the ghost
    cx, cy = int(hx * screen.w), int(hy * screen.h)

    # Radial red profile: thin rings from just outside the monkey body out to a
    # quarter-screen. The red tint fills the circle, so it stays high to the
    # edge and then falls off -- the farthest ring still clearly red is R.
    step = max(2, int(0.0035 * screen.w))
    prof = []
    r = max(6, int(0.016 * screen.h))
    while r < int(0.26 * screen.w):
        prof.append((r, ring_red_shift(cur, clean, cx, cy,
                                       max(2, r - step), r + step)))
        r += step
    peak = max((v for _, v in prof), default=0.0)
    if peak < 10.0:
        return None
    thr = max(6.0, 0.45 * peak)                    # half the fill's red shift
    strong = [rr for rr, v in prof if v >= thr]
    if not strong:
        return None
    R = max(strong)
    if debug_dir is not None:
        shot = cur.copy()
        cv2.circle(shot, (cx, cy), R, (0, 255, 0), 2)
        cv2.circle(shot, (cx, cy), 3, (0, 255, 0), -1)
        cv2.imwrite(str(Path(debug_dir) / f"range_{tower}.png"), shot)
    return round(R / screen.w, 4)


def cmd_scan(args):
    """Machine-generated placement knowledge: no human hovering required.
    Hold a tower ghost, sweep the cursor over a grid, and ask the game
    (via the red invalid-tint on the range circle) whether each point is
    placeable."""
    cfg = load_config()
    screen, hwnd = make_screen(cfg)
    tower = args.tower.lower()
    if tower not in TOWER_HOTKEYS:
        sys.exit(f"Unknown tower '{args.tower}' -- see TOWER_HOTKEYS.")
    threshold = float(cfg.get("scan_red_shift", 12.0))

    masks_dir = Path(__file__).parent / "masks"
    debug_dir = Path(__file__).parent / "debug"
    masks_dir.mkdir(exist_ok=True)
    debug_dir.mkdir(exist_ok=True)

    # Grid over the play area only: skip the top HUD bar and the tower
    # shop column on the right.
    xs = np.arange(0.03, 0.84, args.step)
    ys = np.arange(0.10, 0.95, args.step)
    total = len(xs) * len(ys)

    # Ring radii in pixels: inside the range circle, outside the monkey.
    r_in = max(6, int(0.020 * screen.h))
    r_out = max(r_in + 4, int(0.050 * screen.h))

    print(f"Scanning {total} grid points with a {tower} ghost "
          f"(step {args.step}). Takes a couple of minutes.")
    print("Have the map LOADED but round 1 NOT started. Starting in 3 "
          "seconds...")
    focus_game_window(hwnd)
    time.sleep(3)

    clean = screen.grab()        # ghost-free reference frame + preview base

    # Pick up the ghost, then VERIFY the game actually got the key press.
    # (If it didn't, every point would silently read as "placeable" --
    # that's the all-green half-map failure mode.)
    check_pt = (0.45, 0.50)
    pyautogui.moveTo(*screen.to_pixels(*check_pt))
    time.sleep(0.2)
    press_key(TOWER_HOTKEYS[tower])
    time.sleep(0.5)
    cx, cy = int(check_pt[0] * screen.w), int(check_pt[1] * screen.h)
    probe = screen.grab()
    changed = np.abs(
        probe[cy - r_out:cy + r_out, cx - r_out:cx + r_out].astype(int) -
        clean[cy - r_out:cy + r_out, cx - r_out:cx + r_out].astype(int)
    ).mean()
    if changed < 2.0:
        sys.exit(
            "\nNo tower ghost appeared after pressing "
            f"'{TOWER_HOTKEYS[tower]}'.\n"
            "The game did not receive the key press. Checklist:\n"
            "  1. pip install pydirectinput  (pyautogui keys don't reach "
            "Unity games)\n"
            "  2. The BTD6 window must be focused (it should be -- the bot "
            "focuses it)\n"
            "  3. In-game hotkeys must be the BTD6 defaults (Settings -> "
            "Hotkeys).\n     NOTE: BTD4/BTD5 hotkey lists on the wiki do "
            "NOT apply to BTD6.")
    print(f"Ghost confirmed (frame change {changed:.1f}). Sweeping...")

    valid, invalid = [], []
    done = 0
    try:
        for ny in ys:
            for nx in xs:
                pyautogui.moveTo(*screen.to_pixels(nx, ny))
                time.sleep(0.08)              # let the tint render
                cur = screen.grab()
                px, py = int(nx * screen.w), int(ny * screen.h)
                shift = ring_red_shift(cur, clean, px, py, r_in, r_out)
                if 6.0 <= shift <= 30.0:
                    # Borderline: could be a firework flash or confetti
                    # cluster. A transient fades between frames; a real
                    # invalid-tint doesn't. Resample and keep the lower.
                    time.sleep(0.22)
                    cur = screen.grab()
                    shift = min(shift, ring_red_shift(cur, clean,
                                                      px, py, r_in, r_out))
                point = [round(float(nx), 3), round(float(ny), 3)]
                (invalid if shift > threshold else valid).append(point)
                done += 1
            print(f"\r  {done}/{total} points checked", end="", flush=True)
    except KeyboardInterrupt:
        print("\nScan aborted.")
    finally:
        click("right")                        # drop the ghost
    print()

    # "Strict" points have all four grid neighbors valid too -- these sit
    # comfortably inside placeable regions, away from track edges where
    # detection is fuzzy and tower footprints overhang. Stage 2 samples
    # from these; edge points are kept separately for reference.
    step = args.step
    valid_set = {(p[0], p[1]) for p in valid}
    strict = [p for p in valid
              if all((round(p[0] + dx, 3), round(p[1] + dy, 3)) in valid_set
                     for dx, dy in ((step, 0), (-step, 0),
                                    (0, step), (0, -step)))]

    mask_path = masks_dir / f"{args.name}_{tower}.json"
    mask_path.write_text(json.dumps(
        {"map": args.name, "tower": tower, "step": args.step,
         "game_area": [screen.w, screen.h],
         "valid_strict": strict, "valid": valid}, indent=1))

    preview = clean.copy()
    for nx, ny in invalid:
        cv2.circle(preview, (int(nx * screen.w), int(ny * screen.h)),
                   2, (0, 0, 255), -1)
    edge = valid_set - {(p[0], p[1]) for p in strict}
    for nx, ny in edge:                       # valid but near an edge
        cv2.circle(preview, (int(nx * screen.w), int(ny * screen.h)),
                   3, (0, 165, 255), -1)
    for nx, ny in strict:                     # safely placeable
        cv2.circle(preview, (int(nx * screen.w), int(ny * screen.h)),
                   4, (0, 255, 0), -1)
    preview_path = debug_dir / f"scan_{args.name}_{tower}_preview.png"
    cv2.imwrite(str(preview_path), preview)

    print(f"{len(strict)} safely placeable (green) + {len(edge)} edge "
          f"(orange) / {done} checked.")
    print(f"Mask    -> {mask_path}")
    print(f"Preview -> {preview_path}")
    print("Green = safe interior spots (use these for plans). Orange = "
          "valid but hugging an edge. If it disagrees with the map, tune "
          "\"scan_red_shift\" in config.json and rescan.")


MEASURED_RANGES_PATH = Path(__file__).parent / "measured_ranges.json"


def cmd_measure_ranges(args):
    """Measure each tower's TRUE range from the in-game range circle and save
    it, so the brain reasons about lane coverage with real numbers instead of
    the rough hardcoded guesses (a tack it thought covered ~2% of track). Load
    the map (round NOT started) and walk away; each tower's ghost is hovered
    over the track, where its range circle turns red, and the red disc is
    measured. Writes measured_ranges.json (merged), which meta.py loads."""
    cfg = load_config()
    screen, hwnd = make_screen(cfg)
    debug_dir = Path(__file__).parent / "debug"
    debug_dir.mkdir(exist_ok=True)
    if args.tower:
        towers = [args.tower.lower()]
    else:
        # "hero" measures whichever hero is EQUIPPED (its range circle is the
        # same red overlay) -- so a Sauda opener is sized from her real reach,
        # not the default guess. It's placed via its own hotkey like any tower.
        towers = [t for t in ("dart", "tack", "boomerang", "sniper", "ninja",
                              "bomb", "glue", "ice", "wizard", "druid",
                              "alchemist", "engineer", "mortar", "super",
                              "village", "hero") if t in TOWER_HOTKEYS]
    print("Measuring tower ranges from the in-game range circle. Load the map "
          "(round NOT started) and don't touch the mouse.")
    print("Starting in 4 seconds...")
    focus_game_window(hwnd)
    time.sleep(4)
    out = {}
    for tower in towers:
        try:
            r = measure_tower_range(screen, cfg, tower, debug_dir)
        except Exception:
            _log_crash(f"measure-range {tower}")
            r = None
        try:
            clear_ui(screen, cfg)
        except Exception:
            pass
        if r:
            out[tower] = r
        print(f"   {tower:10} range = "
              f"{('%.4f (%.0f%% of screen width)' % (r, 100 * r)) if r else 'FAILED -- see debug/range_*.png'}")
    existing = {}
    if MEASURED_RANGES_PATH.exists():
        try:
            existing = json.loads(MEASURED_RANGES_PATH.read_text())
        except Exception:
            existing = {}
    existing.update(out)
    MEASURED_RANGES_PATH.write_text(json.dumps(existing, indent=1) + "\n")
    print(f"\nSaved {len(out)}/{len(towers)} ranges to "
          f"{MEASURED_RANGES_PATH.name}. The brain now sizes coverage with "
          f"these. Check debug/range_<tower>.png -- the green circle should "
          f"trace the real range; if it's off, rerun a single tower with "
          f"`measure-ranges --tower <name>`.")


MASK_POINTS = []     # strict points from the loaded mask; retries use these
MASK_ROOMY = []      # strict points eroded once more: room for BIG towers
MASK_NEAR = []       # strict points within 2 steps of the track
MASK_MID = []        # strict points 3-4 steps from the track
SAFE_CLICKS = []     # mask points far from every tower -- safe to deselect on

# Towers with footprints larger than the dart used for scanning. They only
# snap to / retry on ROOMY points (extra clearance on all sides).
LARGE_TOWERS = {"super", "village", "farm", "ace"}

# Tower pool for random layouts in `farm` (land towers only, since the
# mask is scanned with a dart ghost).
FARM_TOWERS = ["dart", "tack", "bomb", "sniper", "ninja", "wizard",
               "druid", "alchemist", "glue", "ice"]


def load_mask(mask_path):
    """Load a scan mask and derive the placement pools. The mask already
    encodes the TRACK: it's the largest connected blob of INVALID lattice
    cells in the interior (the path rejects placement; trees hug the
    border and are excluded). Each strict point's distance to that blob
    feeds MASK_NEAR/MASK_MID so sampling can prefer spots that can
    actually hit bloons instead of decorating corners."""
    global MASK_POINTS, MASK_ROOMY, MASK_NEAR, MASK_MID
    data = json.loads(mask_path.read_text())
    points = data.get("valid_strict") or data.get("valid") or []
    step = data.get("step", 0.025)
    have = {(p[0], p[1]) for p in points}
    MASK_ROOMY = [p for p in points
                  if all((round(p[0] + dx, 3), round(p[1] + dy, 3)) in have
                         for dx, dy in ((step, 0), (-step, 0),
                                        (0, step), (0, -step)))] or points
    MASK_POINTS = points

    every = {(round(p[0], 3), round(p[1], 3))
             for p in (data.get("valid") or points)}
    strict = {(round(p[0], 3), round(p[1], 3)) for p in points}

    def nb(c):
        return [(round(c[0] + dx, 3), round(c[1] + dy, 3))
                for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step))]

    MASK_NEAR, MASK_MID = [], []
    if every:
        xs = sorted({c[0] for c in every})
        ys = sorted({c[1] for c in every})
        lattice = {(round(float(x), 3), round(float(y), 3))
                   for x in np.arange(xs[0], xs[-1] + step / 2, step)
                   for y in np.arange(ys[0], ys[-1] + step / 2, step)}
        invalid = lattice - every
        ring = {c for c in lattice
                if c[0] < xs[0] + 1.5 * step or c[0] > xs[-1] - 1.5 * step
                or c[1] < ys[0] + 1.5 * step or c[1] > ys[-1] - 1.5 * step}
        interior = invalid - ring
        seen, track = set(), set()
        for c in interior:
            if c in seen:
                continue
            comp, stack = set(), [c]
            while stack:
                q = stack.pop()
                if q in comp:
                    continue
                comp.add(q)
                stack += [n for n in nb(q) if n in interior and n not in comp]
            seen |= comp
            if len(comp) > len(track):
                track = comp
        dist = {c: 0 for c in track}
        frontier, d = list(track), 0
        while frontier and d < 5:
            d += 1
            nxt = []
            for c in frontier:
                for n in nb(c):
                    if n in lattice and n not in dist:
                        dist[n] = d
                        nxt.append(n)
            frontier = nxt
        MASK_NEAR = [list(c) for c in strict if dist.get(c, 99) <= 2]
        MASK_MID = [list(c) for c in strict if 2 < dist.get(c, 99) <= 4]
        if track:
            dbg(f"mask: track blob {len(track)} cells; pools "
                f"{len(MASK_NEAR)} near / {len(MASK_MID)} mid / "
                f"{len(points)} strict")
    return points


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")


def find_mask_path(plan, plan_path):
    """Resolve the plan's mask file. Order: the plan's explicit "mask"
    field; then auto-discovery -- any masks/*.json whose name contains the
    plan's map name; then a lone masks/*.json if there's exactly one."""
    bases = (Path.cwd(), plan_path.parent, Path(__file__).parent)
    name = plan.get("mask")
    if name:
        for base in bases:
            if (base / name).exists():
                return base / name
        print(f"Mask '{name}' not found where expected -- trying "
              f"auto-discovery instead.")
    found = []
    for base in bases:
        d = base / "masks"
        if d.is_dir():
            for p in sorted(d.glob("*.json")):
                if p.resolve() not in [f.resolve() for f in found]:
                    found.append(p)
    slug = _slug(plan.get("map"))
    if slug:
        matched = [p for p in found if slug in p.stem]
        if matched:
            return matched[0]
    if len(found) == 1:
        return found[0]
    return None


def snap_plan_to_mask(plan, plan_path):
    """Resolve every 'place' coordinate to the nearest safe (strict) point
    from the map's scan mask, and keep the mask points around so placement
    retries can use them too. Plan coordinates are HINTS."""
    global MASK_POINTS
    mask_path = find_mask_path(plan, plan_path)
    if mask_path is None:
        if any(a.get("do") == "place" for a in plan["actions"]):
            print("!" * 64)
            print("!! No scan mask found. Placements will use RAW plan")
            print("!! coordinates and retries will be blind nudges -- this")
            print("!! is how towers end up wrestling with the path.")
            print("!! Fix: run `scan <map_name>` once for this map.")
            print("!" * 64)
        return
    data_points = load_mask(mask_path)
    if not data_points:
        print(f"Mask {mask_path.name} has no placeable points -- rescan.")
        return
    points = data_points
    moves = []
    for a in plan["actions"]:
        if a.get("do") != "place":
            continue
        large = a["tower"].lower() in LARGE_TOWERS
        pool = MASK_ROOMY if large else points
        # Big towers get a longer leash: cramped map centers may put the
        # nearest roomy spot far away, and relocating a super beats
        # stranding it on an unplaceable hint.
        cap = 0.20 if large else 0.08
        ax, ay = a["at"]
        nx, ny = min(pool, key=lambda q: (q[0] - ax) ** 2 + (q[1] - ay) ** 2)
        dist = ((nx - ax) ** 2 + (ny - ay) ** 2) ** 0.5
        if dist > cap:
            print(f"   !! no safe point within reach of {a['at']} "
                  f"(nearest is {dist:.2f} away) -- leaving it as-is")
            continue
        if dist > 0.001:
            print(f"   snapped {a['tower']} {a['at']} -> [{nx}, {ny}]")
            moves.append((tuple(a["at"]), [nx, ny]))
            a["at"] = [nx, ny]
    # Second pass: upgrades written against the ORIGINAL coordinates must
    # follow their tower to its snapped position, or they'd click grass.
    for a in plan["actions"]:
        if a.get("do") != "upgrade" or not moves:
            continue
        ax, ay = a["at"]
        (ox, oy), new = min(
            moves, key=lambda m: (m[0][0] - ax) ** 2 + (m[0][1] - ay) ** 2)
        if ((ox - ax) ** 2 + (oy - ay) ** 2) ** 0.5 <= 0.03:
            a["at"] = list(new)
    # Safe deselect spots: mask points far from every planned tower, so a
    # panel-closing click can never land on a monkey and open a new panel.
    place_pts = [a["at"] for a in plan["actions"] if a.get("do") == "place"]
    if place_pts:
        def _min_d(q):
            return min((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2
                       for p in place_pts)
        SAFE_CLICKS[:] = [list(q) for q in
                          sorted(points, key=_min_d, reverse=True)[:3]]
    print(f"Placements snapped to safe points from {mask_path.name}.")


def sense_flow_entry(screen, clean, track, timeout=7.0):
    """Which end of the track do bloons come from? Pixels of a still map
    can't say -- but the round just started on an EMPTY map, so the first
    thing that MOVES near the track is the bloons entering. Frame-diff
    against the pre-start clean frame, masked to a corridor around the
    track cells so HUD animation, confetti, and map decorations can't
    vote. Returns the normalized [x, y] of the first motion, or None."""
    h, w = clean.shape[:2]
    corridor = np.zeros((h, w), np.uint8)
    rad = max(6, int(track.step * 0.9 * w))
    for cx, cy in track.cells:
        cv2.circle(corridor, (int(cx * w), int(cy * h)), rad, 255, -1)
    corridor[:int(0.14 * h)] = 0          # HUD strip animates every round
    t_end = time.time() + timeout
    while time.time() < t_end:
        cur = screen.grab()
        diff = cv2.absdiff(cur[..., :3], clean[..., :3])
        moving = ((diff.max(axis=2) > 45) & (corridor > 0)).astype(np.uint8)
        n, _lab, stats, cent = cv2.connectedComponentsWithStats(moving, 8)
        blobs = [(stats[i, 4], cent[i]) for i in range(1, n)
                 if stats[i, 4] >= 30]
        if blobs:
            _area, (cx, cy) = max(blobs, key=lambda b: b[0])
            return [round(cx / w, 4), round(cy / h, 4)]
        time.sleep(0.12)
    return None


# ---------------------------------------------------------------------------
# STAGE 2: the data farm. Random layouts -> unattended episodes ->
# (layout, final_round, outcome) rows in runs_log.jsonl.
# ---------------------------------------------------------------------------

def random_genome(rng, n_towers, hero=False):
    """An ordered buy list: towers on random mask points, then shuffled
    single-tier upgrades (a random main path to <=4, a crosspath to <=2).
    Executed greedily as cash allows -- timing emerges from the economy.
    Tiers already known to be XP-locked are never generated."""
    genome = []
    spots = []
    for _ in range(min(n_towers + (1 if hero else 0), len(MASK_POINTS))):
        roll = rng.random()
        if roll < 0.70 and MASK_NEAR:
            pool = MASK_NEAR
        elif roll < 0.95 and (MASK_MID or MASK_NEAR):
            pool = MASK_MID or MASK_NEAR
        else:
            pool = MASK_POINTS
        for _ in range(40):
            cand = rng.choice(pool)
            if cand not in spots:
                spots.append(cand)
                break
    types = [rng.choice(FARM_TOWERS) for _ in spots]
    if hero and types:
        types[0] = "hero"          # levels on its own; no upgrade rolls
    for ref, (spot, ttype) in enumerate(zip(spots, types)):
        genome.append({"do": "place", "tower": ttype,
                       "at": [spot[0], spot[1]], "ref": ref})
    ups = []
    for ref, ttype in enumerate(types):
        if ttype == "hero":
            continue
        main = rng.randrange(3)
        cross = rng.choice([p for p in range(3) if p != main])
        # At least tier 1 on the main path: near-empty genomes finish
        # buying by round 7 and spend the rest of the run just observing.
        want = ([main] * rng.randint(1, 4)) + ([cross] * rng.randint(0, 2))
        tiers = {main: 0, cross: 0}
        for path_i in want:
            tiers[path_i] += 1
            if is_locked(ttype, path_i, tiers[path_i]):
                tiers[path_i] -= 1             # locked: stop this path here
                continue
            ups.append((ref, path_i))
    rng.shuffle(ups)
    for ref, path_i in ups:
        vec = [0, 0, 0]
        vec[path_i] = 1
        genome.append({"do": "upgrade", "ref": ref, "path": vec})
    return genome


def run_episode(screen, cfg, genome, final_round, abort_lives=50,
                flow_sensor=None, play_out=False, danger_rounds=None,
                abilities=False, one_life=False, freeplay=False,
                mode="standard"):
    """Play one layout to survival or defeat. Returns (outcome,
    final_round_reached, towers, lives_by_round, cash_by_round,
    spent_by_round).
    flow_sensor, when given, is called once with a pre-start frame right
    after the round starts -- its window to watch where the first bloons
    appear. Opening towers due on the starting round are placed before
    Space is pressed, so the run does not leak while waiting for OCR.
    play_out: don't declare survival at the START of the final round --
    play THROUGH it and only call it survived once the victory screen
    covers the HUD (or the final round has sat finished for minutes).
    That is the difference between farming a target and actually
    BEATING the game, where round 100's BAD dies or you do.
    danger_rounds + abilities: on known threat rounds (and during leak
    emergencies) press the ability hotkeys every few seconds -- a free
    no-op when nothing is trained, exactly how a player dumps abilities
    on r24/r63/r90+."""
    # (Re)calibrate sensors on the clean, un-started map -- the best frame
    # we'll ever get. Cash preflight runs UNCONDITIONALLY: a box that
    # clips the leading digit still 'reads', and only the digit snap
    # inside the preflight can catch and re-fit it.
    if not preflight_cash_box(screen, cfg):
        dbg("cash unreadable -- recalibrating from scratch")
        preflight_cash_box(screen, cfg, recalibrate=True)
    if read_lives(screen, cfg) is None:
        dbg("lives unreadable -- recalibrating from the heart icon")
        preflight_lives_box(screen, cfg, recalibrate=True)
    dbg(f"sensors: cash={read_cash(screen, cfg)} "
        f"lives={read_lives(screen, cfg)}")

    try:
        from meta import choose_buy, earned_by, round_supported_by_cash
    except ImportError:                       # no brain: old greedy rule
        def choose_buy(q, _round, _cash, _cost, now, emergency=False,
                       gate_ok=None):
            return next((j for j, it in enumerate(q)
                         if it.get("_wake", 0) <= now), None)

        def earned_by(_r, _mode="standard"):  # no income model available
            return 0

        def round_supported_by_cash(*_a, **_k):   # no curve: never veto
            return True

    queue = list(genome)
    landed_by_ref, towers = {}, {}
    # ref -> a human tower tag ("druid#2(carry)"), from the genome's place
    # entries, so an upgrade buy (whose queue item carries only ref+path) can
    # still name its tower in the saving logs even before it is placed.
    ref_name = {g["ref"]: (g.get("name") or g.get("tower", "?"))
                for g in genome if g.get("do") == "place" and "ref" in g}
    # Spots a probe caught a monkey sitting on (a tower we didn't track, or
    # one whose real position drifted from where we recorded it). Grown by
    # act_place and fed back into every placement so no buy keeps clicking a
    # spot that's already taken.
    occupied_spots = []
    lives_by_round = {}
    cash_by_round = {}       # cash at the start of each round (telemetry
    spent_by_round = {}      # for the learned income curve)
    broke_at = [None, 0.0]   # cash watermark: level + time of last set
    emergency_until = [0.0]  # leak emergency: reserves off until this time
    # Provable lower bound on current cash: it only falls through purchases
    # (all reported via record_spend), so a read far below it is a misread.
    cash_floor = CashFloor(income_model=lambda r: earned_by(r, mode))
    prev_round_lives = [None]
    last_ping = [0.0]        # ability-hotkey pinger cooldown
    final_seen = None        # play_out: when the final round was reached
    last_round = prev_read = None

    def record_spend(amount):
        if amount:
            key = str(last_round if last_round is not None else 0)
            spent_by_round[key] = spent_by_round.get(key, 0) + amount
            cash_floor.spend(amount)      # keep the provable floor valid

    def confirm_floor():
        """Raise the cash floor from a corroborated read (see CashFloor).
        Called at episode start (seed) and at each round change / successful
        buy (re-anchor to the true level so the floor never goes stale-low).
        A per-round magnitude cap keeps a spurious-high read from INFLATING
        the floor (which would make the bot try to over-buy); too-tight a cap
        merely under-raises it, which is the safe direction."""
        cap = None
        if last_round is not None:
            try:
                cap = 2.0 * earned_by(last_round, mode)
            except Exception:
                cap = None
        cash_floor.confirm(read_cash_confirmed(screen, cfg, max_plausible=cap))

    def sane_cash():
        """read_cash, but a read implausibly far below what we PROVABLY have
        is a clipped/misread number, not a real drop -- corroborate it and,
        if it stays low, substitute the floor so a low misread can never
        freeze the buy plan and leak the run. A correct read passes through
        unchanged. (The floor logic itself lives in cashguard.CashFloor.)"""
        v = read_cash(screen, cfg)
        out = cash_floor.sane(
            v, confirm_fn=lambda: read_cash_confirmed(screen, cfg),
            round_hint=last_round)
        if v is not None and out is not None and out != v:
            # `out` is the higher of the spend-tracked floor and the CHIMPS
            # earned-minus-spent estimate; either way it's cash we can PROVE
            # we have, so a read below it is a clipped/garbled OCR misread.
            dbg(f"sane_cash: read ${v} is below the ${out} we can prove we "
                f"have by round {last_round} (earned minus spend) -- "
                f"using ${out}, not reading it as broke")
        return out

    def recalibrate_cash_if_stuck():
        """A cash box that has drifted onto junk reads a CONSTANT low value
        (the field bug: a '$1' at round 27 while cash is really in the
        thousands), which the floor then substitutes on every buy -- the whole
        plan freezes behind a phantom-low wallet until the run is lost. When
        sane_cash has substituted many times running with no good read, the
        BOX is broken, not the game: re-find the counter from scratch off the
        coin/heart HUD landmark and re-seed the floor from the fresh read."""
        if not cash_floor.stuck():
            return
        dbg("cash reads stuck low -- recalibrating the counter from the "
            "HUD landmark")
        preflight_cash_box(screen, cfg, recalibrate=True)
        cash_floor.reset_stuck()
        confirm_floor()

    def set_watermark(price=None):
        """Record the cash level of a money-failure. Being broke for a
        KNOWN price means cash < price BY DEFINITION, so the watermark is
        capped at price-5 -- a junk read like $6005 (real $600) can then
        never freeze the whole build behind an unreachable level."""
        lvl = read_cash_confirmed(screen, cfg)
        if price and (lvl is None or lvl >= price):
            lvl = price - 5
        broke_at[0] = lvl
        broke_at[1] = time.time()

    def watermark_holds(cash):
        """Should this buy wait for income to grow past the watermark?
        Watermarks EXPIRE after 40s: income only rises, so a hold that
        old has done its job -- and if the level itself was a misread,
        expiry is what un-sticks the run (each item also has its own
        _wake cooldown, so retries stay cheap)."""
        if broke_at[0] is None or cash is None:
            return False
        if time.time() - broke_at[1] > 40:
            broke_at[0] = None
            return False
        return cash <= broke_at[0] + 100
    round_seen_at = time.time()   # for the frozen-defeat-screen net
    observing = False             # queue empty: just watching the run
    # Safe deselect spots for THIS episode: mask points farthest from every
    # tower this genome will place, so a panel-closing click can't select
    # a monkey. (The genome differs per episode, so recompute each time.)
    spots = [g["at"] for g in genome if g["do"] == "place"]
    pool = MASK_ROOMY or MASK_POINTS
    if pool and spots:
        SAFE_CLICKS[:] = sorted(
            pool, key=lambda q: -min((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2
                                     for p in spots))[:3]
    place_i = 0

    def place_prestart_openers():
        """Put down any due opening towers before pressing start. The old
        flow started the round first, then waited for OCR to say which round
        we were on before buying; on hard/CHIMPS that can leak before the
        hero/carry is even placed."""
        nonlocal place_i
        start_round = read_round_stable(
            screen, cfg, tries=3, total=None if FREEPLAY else 100) or 1
        # Openers we can't place THIS pass (unaffordable now, or a transient
        # miss) are set aside, not allowed to block the rest: a free hero
        # must never be stranded behind a carry we can't yet afford -- that
        # leaks the very opening rounds CHIMPS never forgives. Set-aside
        # items stay queued; the main loop buys them once cash arrives.
        skipped = set()
        while True:
            idx = next((j for j, it in enumerate(queue)
                        if it.get("do") == "place"
                        and it.get("round", 0) <= start_round
                        and id(it) not in skipped), None)
            if idx is None:
                break
            item = queue[idx]
            ttype = item["tower"].lower()
            base = learned_price(ttype)        # gate: verified price only
            cash = sane_cash()
            if base and cash is not None and cash < base:
                skipped.add(id(item))          # can't afford yet: try others
                continue
            st, landed = act_place(
                screen, cfg,
                {**item, "timeout": 10,
                 "avoid": [t["at"] for t in towers.values()],
                 "occupied": occupied_spots})
            if landed is None:
                if st == "no_spot":
                    print(f"   pre-start {ttype}: no placeable spot near "
                          f"{item['at']} -- skipping")
                    landed_by_ref[item.get("ref", place_i)] = None
                    queue.pop(idx)
                    continue
                skipped.add(id(item))          # transient miss: leave queued
                continue
            landed_by_ref[item.get("ref", place_i)] = landed
            towers[tuple(landed)] = {"tower": item["tower"], "at": landed,
                                     "path": [0, 0, 0],
                                     **({"name": item["name"]}
                                        if item.get("name") else {})}
            place_i += 1
            queue.pop(idx)
            record_spend(PRICES.get(price_key(ttype)) or item.get("est") or 0)
            clear_ui(screen, cfg)
        return start_round

    def upgrade_prestart_openers(start_round):
        """Spend the leftover opening cash UPGRADING the just-placed towers
        before the round starts. A bare 0-0-0 tower leaks round 6 on one
        life -- the wave hits before its first tier can be bought mid-round
        -- but the pre-round phase has NO clock, so buy each opener's due
        tiers now and start the round with teeth already out. Only tiers the
        plan already scheduled for the start round are pulled (the opener's
        early_opener teeth); the carry's pricey upgrades are gated later, so
        this never drains the wallet, it just front-loads the cheap defence
        that has to hold round 6."""
        guard = 0
        while guard < 12:
            guard += 1
            idx = next((j for j, it in enumerate(queue)
                        if it.get("do") == "upgrade"
                        and it.get("round", 0) <= start_round
                        and it.get("ref") in landed_by_ref
                        and landed_by_ref.get(it["ref"]) is not None), None)
            if idx is None:
                break
            item = queue[idx]
            landed = landed_by_ref[item["ref"]]
            entry = towers.get(tuple(landed))
            if entry is None:
                queue.pop(idx)
                continue
            ttype = entry["tower"].lower()
            pi = item["path"].index(1) if 1 in item["path"] else 0
            tier = entry["path"][pi] + 1
            if is_locked(ttype, pi, tier):
                queue.pop(idx)
                continue
            known = learned_price(ttype, pi, tier)   # gate: verified price only
            cash = sane_cash()
            if known is not None and cash is not None and cash < known:
                break          # can't afford the cheapest due tier: start now
            st = act_upgrade(
                screen, cfg,
                {**item, "at": landed, "sane_cash": sane_cash}, entry)
            if st == "bought":
                queue.pop(idx)
                record_spend(PRICES.get(price_key(ttype, pi, tier))
                             or item.get("est") or 0)
                confirm_floor()
            elif st in ("locked", "closed"):
                queue.pop(idx)
            else:
                break          # broke / no_select: stop, get the round going

    confirm_floor()                # seed the floor from starting cash before
    _start_round = place_prestart_openers()   # any pre-start buy lowers it
    upgrade_prestart_openers(_start_round)     # give the opener teeth pre-wave
    clean = screen.grab() if flow_sensor else None
    press_key("space")
    time.sleep(0.3)
    press_key("space")                        # fast-forward
    if flow_sensor:
        flow_sensor(clean)
    attempts = 0
    lives = prev_lives = None
    zero_streak = 0
    dumped_defeat_check = False
    misreads = 0
    blind_since = None            # when the counter went continuously dark
    hud_dark_since = None         # when the WHOLE HUD (lives too) went dark
    last_blind_recovery = 0.0     # throttle for clear/recalibrate attempts
    blind_building = [False]      # keep-building-while-blind latch (log once)
    saving_toward = [None]        # last "saving toward X" note (log once)
    outcome = "hud_lost"
    while True:
        unpause_if_needed(screen, cfg)        # cheap; a paused game shows
        recalibrate_cash_if_stuck()           # broken box -> re-find, not freeze
        # Pass the HUD total (BTD6 shows "round/100" for standard & CHIMPS) so
        # a faded '/' that fuses "34/100" into "34100" is recovered instead of
        # reading as a dark counter and freezing the plan. Freeplay's counter
        # is a bare number past 100, so no total there.
        value, *_ = read_round(screen, cfg,
                               total=None if FREEPLAY else 100)
        prev_lives = lives
        lives = read_lives(screen, cfg)
        if lives is not None and lives > 0:
            hud_dark_since = None      # a live lives count: HUD isn't dark
        accepted = None
        if value is not None:
            if value == last_round:
                accepted = value
            elif plausible(value, last_round) and value == prev_read:
                accepted = value
            elif (misreads >= 6 and value == prev_read
                  and last_round is not None
                  and last_round < value <= last_round + 40
                  and value <= (200 if FREEPLAY else 100)):
                # Re-sync after a blind stretch. BTD6 rounds auto-chain, so
                # while the counter was unreadable (a long buy-spree in the
                # tower panels, an overlay) the game kept advancing. A read
                # past last+3 is then NOT a misread but the true, larger
                # round -- rejecting it forever (as the tight plausible()
                # jump does) is exactly what made a live round-27 game read
                # as a lost HUD. Accept it once two consecutive frames agree
                # (a one-frame garbage read won't repeat) and it is within a
                # bounded forward jump under the mode cap, so the bot catches
                # back up instead of declaring hud_lost on a readable counter.
                #
                # But a BIG jump is cross-checked against the wallet: in CHIMPS
                # you can only be at round r if you've earned its cumulative
                # pop-cash, so a jump the money provably can't support is a
                # COUNTER misread, not real progress (the field failure:
                # 27 -> 50 on ~$1k, which then inflated the income floor and
                # coasted the run to death). Reject it and keep the old round.
                if value - last_round >= 8 and not round_supported_by_cash(
                        value, read_cash_confirmed(screen, cfg),
                        cash_floor.spent, mode):
                    dbg(f"counter read {value} rejected: the cash on hand "
                        f"can't support round {value} in CHIMPS -- likely a "
                        f"misread, holding round {last_round}")
                else:
                    dbg(f"counter re-synced after {misreads} misses: "
                        f"round {last_round} -> {value}")
                    accepted = value
            prev_read = value
        if accepted is None:
            if detect_panel_side(screen):
                # A counter hidden behind an OPEN PANEL isn't a lost HUD
                # -- close the panel instead of counting toward hud_lost.
                clear_ui(screen, cfg)
            else:
                misreads += 1
        else:
            misreads = 0
            blind_since = None                # counter is readable again
            hud_dark_since = None             # ...so the HUD is not dark
            if accepted != last_round:
                last_round = accepted
                # Cash grows a digit or two over a run; re-widen the box if
                # its leading digit has started clipping (the preflight fit
                # couldn't see a clip that didn't exist yet on starting cash).
                recheck_cash_box(screen, cfg, round_hint=last_round, mode=mode)
                # Re-anchor the floor to the true level each round so it never
                # goes stale-low as income outgrows the last confirmed read.
                confirm_floor()
                if lives is not None:
                    lives_by_round[str(last_round)] = lives
                cash_now = sane_cash()
                if cash_now is not None:
                    cash_by_round[str(last_round)] = cash_now
                print(f"   [round {last_round:>3}]  lives={lives}  "
                      f"({len(queue)} buys pending)")
                if lives is not None and prev_round_lives[0] is not None \
                        and prev_round_lives[0] - lives >= 8:
                    # Leaking hard: a good player dumps savings on any
                    # defense NOW instead of hoarding for the big buy.
                    if time.time() >= emergency_until[0]:
                        print(f"   leaking ({prev_round_lives[0]} -> "
                              f"{lives}): reserves off, buying any "
                              f"affordable defense")
                    emergency_until[0] = time.time() + 45
                if lives is not None:
                    prev_round_lives[0] = lives

        if lives == 0:
            zero_streak += 1
            if zero_streak == 1 and not looks_defeated(screen) \
                    and not dumped_defeat_check:
                # Detectors disagree: photograph the moment so a missed
                # defeat diagnoses itself from the debug folder.
                dumped_defeat_check = True
                ddir = Path(__file__).parent / "debug"
                ddir.mkdir(exist_ok=True)
                cv2.imwrite(str(ddir / f"defeat_check_{int(time.time()) % 100000}.png"),
                            screen.grab())
        elif lives is not None:
            zero_streak = 0
        digit_defeat = (lives == 0 and prev_lives == 0) or zero_streak >= 3
        if digit_defeat and (not one_life or looks_defeated(screen)):
            # A one-life run cannot afford a false defeat: a burst of lives
            # MISREADS (the CHIMPS "1" flickering to 0, coin/heart noise)
            # would otherwise restart a game that is still alive. There the
            # digit reading must be backed by the DEFEAT screen itself.
            # Multi-life rungs keep the digit-only exit (a leak to 0 there
            # is unambiguous and the HUD stays up).
            outcome = "defeat"                # defeat screen (HUD stays!)
            break
        near_final = (play_out and last_round is not None
                      and last_round >= final_round - 1)
        if abort_lives and lives is not None and 0 < lives < abort_lives \
                and not near_final:
            # A lost cause: abort while the game is LIVE, so the restart
            # takes the pause route -- the one with a proven field
            # record. But NEVER abort on (or one short of) the final
            # round of a play-out: a champion that reaches round 80 with
            # 40 lives is about to WIN, not to be written off -- let it
            # finish so the victory actually registers.
            print(f"   lives {lives} < {abort_lives}: aborting this "
                  f"episode early")
            outcome = "defeat"
            break
        if accepted is not None and accepted != last_round:
            pass
        if accepted is not None:
            round_seen_at = time.time()
        buyable_now = any(it.get("round", 0) <= (last_round or 0)
                          for it in queue)
        # "Critical lives" for the frozen-round net: on a normal rung, a
        # low life count during a minute-long stall means we quietly leaked
        # to death behind a covered HUD. On a ONE-LIFE rung that test is
        # meaningless -- lives is permanently 1, so `lives < 40` is always
        # true and any stall (a BAD/ZOMG round, a UI overlay, a drifted
        # crop) got misread as a defeat and restarted a live game. There,
        # only an actual leaked-out (lives 0) counts.
        crit = (lives == 0) if one_life else (lives is None or lives < 40)
        if not buyable_now and time.time() - round_seen_at > 60 and crit:
            # Round frozen for a minute, nothing buyable at this round,
            # critical lives: that is the defeat screen, whatever the
            # pixels say. (Checked against ELIGIBLE buys, not the raw
            # queue -- future-scheduled items must not keep a dead
            # episode waiting forever.)
            dbg("round frozen 60s with critical lives -- treating as "
                "defeat")
            outcome = "defeat"
            break
        if looks_defeated(screen):
            time.sleep(0.4)
            if looks_defeated(screen):        # confirmed on two samples
                dbg("DEFEAT screen recognized visually")
                outcome = "defeat"
                break

        # Abilities: on known threat rounds and during leak emergencies,
        # ping the ability hotkeys. Untrained abilities make the presses
        # no-ops; trained ones fire exactly when a player would use them.
        if abilities and last_round is not None:
            hot = (danger_rounds and last_round in danger_rounds) \
                or time.time() < emergency_until[0]
            if hot and time.time() - last_ping[0] > 5.0:
                last_ping[0] = time.time()
                for key in (cfg.get("ability_keys") or ["1", "2", "3"]):
                    press_key(str(key))

        # Try the next buy (one attempt per loop, so rounds keep reading).
        if queue and last_round is not None:
            def _cost_of(it):
                """Learned price first, genome estimate second -- feeds
                the reserve math in choose_buy."""
                if it.get("do") == "place":
                    known = PRICES.get(price_key(it["tower"].lower()))
                else:
                    known = None
                    lnd = landed_by_ref.get(it.get("ref"))
                    ent = towers.get(tuple(lnd)) if lnd else None
                    if ent:
                        pi = it["path"].index(1) if 1 in it["path"] else 0
                        known = PRICES.get(price_key(
                            ent["tower"].lower(), pi,
                            ent["path"][pi] + 1))
                return known or it.get("est")

            def _saving_desc(it):
                """Human tag for a pending buy that ALWAYS names the tower: a
                placement reads as the tower name ('tack#1(carry)'), an upgrade
                as 'druid#2(carry) path3 t4'. Upgrade queue items carry only
                ref+path, so the tower name comes from the placed entry when it
                is down, else from the genome's ref->name map -- never a bare
                'path3 t4' with no tower."""
                if it.get("do") == "place":
                    return it.get("name") or it.get("tower", "?")
                pi = it["path"].index(1) if 1 in it.get("path", []) else 0
                lnd = landed_by_ref.get(it.get("ref"))
                ent = towers.get(tuple(lnd)) if lnd else None
                name = (ent.get("name") if ent else None) \
                    or ref_name.get(it.get("ref"), "?")
                tier = (ent["path"][pi] + 1) if ent else "?"
                return f"{name} path{pi + 1} t{tier}"

            def _gate_ok(it):
                """Conditional buys: support bases gated on the carry
                being stable (main path at the gate tier). Overrides
                that keep the gate from becoming a deadlock: a threat
                date ('by') close at hand, the carry placement having
                been skipped, or the item running 6+ rounds past its
                own schedule -- support arrives when NEEDED, and a
                struggling carry can't strand it forever."""
                g = it.get("gate")
                if not g:
                    return True
                if it.get("by") and last_round >= it["by"] - 1:
                    return True
                if last_round >= it.get("round", 0) + 6:
                    return True
                if g["ref"] in landed_by_ref \
                        and landed_by_ref[g["ref"]] is None:
                    return True          # carry skipped: gate is void
                lnd = landed_by_ref.get(g["ref"])
                ent = towers.get(tuple(lnd)) if lnd else None
                return bool(ent) and max(ent["path"]) >= g["tier"]

            def do_batched_upgrade(item, entry, landed, pi, tier,
                                   ttype, tname):
                """Buy every DUE tier for THIS tower in ONE panel session:
                open once, buy the whole affordable run, close once --
                instead of open/buy/close per queued tier (act_upgrade
                already buys a multi-count path vector in one panel; the
                queue just split every tier into its own item). A reservation
                for the priciest OTHER due buy keeps a tower's cheap tiers
                from starving the carry we are saving toward, so this never
                spends money choose_buy would not have released this round;
                worst case it degrades to the single selected tier (no
                regression). Returns how many tiers were bought."""
                now = time.time()
                reserve = max(
                    [_cost_of(it) or 0 for it in queue
                     if it.get("ref") != item["ref"]
                     and it.get("round", 0) <= (last_round or 0)],
                    default=0)
                cashb = sane_cash()
                budget = None if cashb is None else max(0, cashb - reserve)
                proj = list(entry["path"])
                vec = [0, 0, 0]
                members = []                       # (queue_item, path_i)
                spent_known = 0
                ordered = [item] + [
                    it for it in queue
                    if it is not item and it.get("ref") == item["ref"]
                    and it.get("do") == "upgrade"]
                for it in ordered:
                    p = it["path"].index(1) if 1 in it["path"] else 0
                    t = proj[p] + 1
                    if is_locked(ttype, p, t) \
                            or p in entry.get("closed_paths", []):
                        continue
                    if it is not item and it.get("_wake", 0) > now:
                        continue
                    price = PRICES.get(price_key(ttype, p, t))
                    if it is not item and budget is not None \
                            and price is not None \
                            and spent_known + price > budget:
                        continue                   # would dip into reserve
                    if price is not None:
                        spent_known += price
                    vec[p] += 1
                    proj[p] += 1
                    members.append((it, p))
                before = list(entry["path"])
                st = act_upgrade(
                    screen, cfg,
                    {**item, "at": landed, "path": vec,
                     "sane_cash": sane_cash}, entry)
                bought = [entry["path"][k] - before[k] for k in range(3)]
                nb = sum(bought)
                giveup = 4 if st in ("no_select", "unread") else 6
                need = list(bought)
                for it, p in members:
                    if need[p] > 0:                # this tier landed
                        need[p] -= 1
                        try:
                            queue.remove(it)
                        except ValueError:
                            pass
                    else:                          # didn't land: keep it
                        t = entry["path"][p] + 1
                        if p in entry.get("closed_paths", []) \
                                or is_locked(ttype, p, t):
                            try:
                                queue.remove(it)   # path closed / XP-locked
                            except ValueError:
                                pass
                        else:
                            it["_fails"] = it.get("_fails", 0) + 1
                            it["_wake"] = time.time() + (
                                min(10 * 2 ** (it["_fails"] - 1), 60)
                                if st == "broke" else 10)
                            if it["_fails"] >= giveup:
                                try:
                                    queue.remove(it)
                                except ValueError:
                                    pass
                if nb > 0:
                    broke_at[0] = None
                    for k in range(3):
                        for t in range(before[k] + 1, entry["path"][k] + 1):
                            record_spend(
                                PRICES.get(price_key(ttype, k, t))
                                or item.get("est") or 0)
                    confirm_floor()
                    if nb > 1:
                        dbg(f"{tname}: bought {nb} tiers in one panel "
                            f"{bought}")
                    else:
                        dbg(f"{tname} path{pi + 1} t{tier}: bought")
                else:
                    dbg(f"{tname} path{pi + 1} t{tier}: {st}")
                    if st == "broke":
                        set_watermark(PRICES.get(price_key(ttype, pi, tier)))
                return nb

            cash_now = sane_cash()
            # Blind-but-alive: the counter is stuck so last_round is frozen,
            # and the round-gated schedule would FREEZE the whole build for the
            # blind stretch -- the field failure was reading cleanly to ~r27,
            # going blind, and reaching round 50 with only ~12 buys made and a
            # bare board that the MOAB rounds walk through. So while blind keep
            # BUILDING on the provable cash floor: drop the round gate and buy
            # any affordable pending defense in schedule order, so the board
            # grows through the blind stretch instead of stalling.
            blind_build = (misreads >= 8 and lives is not None and lives > 0)
            if blind_build and not blind_building[0]:
                dbg("counter blind but alive -- building on the cash floor, "
                    "not freezing the plan")
            blind_building[0] = blind_build
            rush = time.time() < emergency_until[0] or blind_build
            try:
                idx = choose_buy(queue, last_round, cash_now, _cost_of,
                                 time.time(), emergency=rush,
                                 gate_ok=_gate_ok)
            except TypeError:
                # A meta.py from a different version than mk.py (mixed
                # zip extractions, stale __pycache__). Never crash the
                # episode over it -- run without gates.
                dbg("stale meta.choose_buy without gate support -- "
                    "running ungated (re-download / clear __pycache__)")
                idx = choose_buy(queue, last_round, cash_now, _cost_of,
                                 time.time(), emergency=rush)
            item = queue[idx] if idx is not None else None
            if item is None:
                # Every wallet held: choose_buy is RESERVING cash for the
                # scheduled head buy (or waiting for its round). Say what we're
                # saving toward so a held wallet is never silent about its goal.
                head = next((it for it in queue
                             if it.get("round", 0) <= (last_round or 0)), None) \
                    or min(queue, key=lambda it: (it.get("round", 0),
                                                  it.get("prio", 1)),
                           default=None)
                goal = _cost_of(head) if head is not None else None
                msg = None
                if head is not None and cash_now is not None and goal:
                    if cash_now < goal:
                        msg = (f"saving toward {_saving_desc(head)} "
                               f"(${cash_now}/${goal})")
                    elif head.get("round", 0) > (last_round or 0):
                        msg = (f"holding ${cash_now} -- next buy "
                               f"{_saving_desc(head)} opens round "
                               f"{head['round']}")
                if msg and saving_toward[0] != msg:
                    saving_toward[0] = msg
                    dbg(msg)
            elif item["do"] == "place":
                ttype = item["tower"].lower()
                base = learned_price(ttype)        # gate: verified price only
                cash = sane_cash()
                if base and cash is not None and cash < base:
                    msg = f"{ttype}: saving up (${cash}/${base})"
                    if item.get("_dbg") != msg:
                        item["_dbg"] = msg
                        dbg(msg)
                    item["_wake"] = time.time() + 3    # saving up: silent
                elif watermark_holds(cash):
                    msg = f"{ttype}: watermark hold (${cash} <= " \
                          f"${broke_at[0]}+100)"
                    if item.get("_dbg") != msg:
                        item["_dbg"] = msg
                        dbg(msg)
                    item["_wake"] = time.time() + 4    # watermark gate
                else:
                    st, landed = act_place(
                        screen, cfg,
                        {**item, "timeout": 10,
                         "avoid": [t["at"] for t in towers.values()],
                         "occupied": occupied_spots})
                    if landed is not None:
                        # Key by the genome's own ref when present: place
                        # items can execute out of order (broke/no-spot
                        # items sleep on _wake), and pop-order keying
                        # would then wire upgrades to the wrong tower.
                        landed_by_ref[item.get("ref", place_i)] = landed
                        towers[tuple(landed)] = {"tower": item["tower"],
                                                 "at": landed,
                                                 "path": [0, 0, 0],
                                                 **({"name": item["name"]}
                                                    if item.get("name")
                                                    else {})}
                        place_i += 1
                        queue.pop(idx)
                        broke_at[0] = None
                        record_spend(PRICES.get(price_key(ttype))
                                     or item.get("est") or 0)
                        confirm_floor()        # re-anchor to true post-buy cash
                    elif st == "broke":
                        est_c = base or item.get("est")
                        lvl = read_cash_confirmed(screen, cfg)
                        if est_c and lvl is not None and lvl >= est_c:
                            # "Can't afford" while holding MORE than the
                            # price: the hotkey isn't producing a ghost
                            # at all (classic case: hero key with no
                            # hero equipped). Money won't fix this --
                            # stop letting it anchor the buy plan.
                            item["_ghost_fails"] = \
                                item.get("_ghost_fails", 0) + 1
                            if item["_ghost_fails"] >= 4:
                                print(f"   giving up on {item['tower']}: "
                                      f"affordable (${lvl}) but no ghost "
                                      f"ever appears"
                                      + (" -- is a hero equipped?"
                                         if item["tower"] == "hero"
                                         else ""))
                                landed_by_ref[item.get("ref", place_i)] \
                                    = None
                                place_i += 1
                                queue.pop(idx)
                            else:
                                item["_wake"] = time.time() + 5
                        else:
                            set_watermark(base)
                            item["_wake"] = time.time() + 5
                    else:                              # no_spot: real fail
                        item["_fails"] = item.get("_fails", 0) + 1
                        if item["_fails"] >= 3:
                            print(f"   skipping {item['tower']} at "
                                  f"{item['at']} -- no placeable spot "
                                  f"nearby (rescan if this looks wrong)")
                            landed_by_ref[item.get("ref", place_i)] = None
                            place_i += 1
                            queue.pop(idx)
                        else:
                            item["_wake"] = time.time() + 4
            else:                              # upgrade
                landed = landed_by_ref.get(item["ref"])
                if item["ref"] not in landed_by_ref:
                    # Its tower isn't DOWN yet (scheduled for later, or
                    # still saving up): wait. Only a ref explicitly
                    # marked None -- a skipped placement -- may kill its
                    # upgrades.
                    item["_wake"] = time.time() + 3
                elif landed is None:           # its tower was skipped
                    queue.pop(idx)
                else:
                    entry = towers.get(tuple(landed))
                    pi = item["path"].index(1) if 1 in item["path"] else 0
                    ttype = entry["tower"].lower() if entry else ""
                    tname = (entry.get("name") if entry else None) \
                        or ttype
                    tier = (entry["path"][pi] + 1) if entry else 1
                    # Decide WITHOUT opening the menu where possible.
                    if entry and entry.get("_noselect", 0) >= 3:
                        dbg(f"{tname}: unselectable tower -- dropping "
                            f"this upgrade")
                        queue.pop(idx)
                    elif entry and is_locked(ttype, pi, tier):
                        dbg(f"{tname} path{pi + 1} t{tier}: XP-locked, skip")
                        queue.pop(idx)         # XP-locked: never touch it
                    elif entry and pi in entry.get("closed_paths", []):
                        dbg(f"{tname} path{pi + 1}: closed, skip")
                        queue.pop(idx)         # path closed on this tower
                    elif (known := learned_price(ttype, pi, tier)) \
                            and (cash := sane_cash()) is not None \
                            and cash < known:
                        # VERIFIED price (gate: learned only, never a possibly
                        # over-stated seed), can't afford: no menu, just sleep
                        # this item and let income build.
                        msg = (f"{tname} path{pi + 1} t{tier}: saving "
                               f"(${cash}/${known})")
                        if item.get("_dbg") != msg:
                            item["_dbg"] = msg
                            dbg(msg)
                        item["_saving"] = item.get("_saving", time.time())
                        pk = price_key(ttype, pi, tier)
                        if PRICES_SRC.get(pk) not in ("seen", "buy") \
                                and time.time() - item["_saving"] > 45 \
                                and time.time() >= item.get("_recheck_at",
                                                            0):
                            # This price was never verified THIS session
                            # (red-row read, or loaded from prices.json --
                            # possibly poisoned, e.g. $210 recorded as
                            # $2105 gates the buy forever). One menu open
                            # re-reads it; a green sighting overwrites and
                            # heals. Repeats every 60s while still stuck.
                            item["_recheck_at"] = time.time() + 60
                            dbg(f"{tname} path{pi + 1}: re-checking "
                                f"unverified ${known}")
                            st = act_upgrade(
                                screen, cfg,
                                {**item, "at": landed,
                                 "sane_cash": sane_cash}, entry)
                            dbg(f"{tname} path{pi + 1} t{tier}: {st}")
                            if st == "bought":
                                queue.pop(idx)
                                broke_at[0] = None
                                record_spend(
                                    PRICES.get(price_key(ttype, pi, tier))
                                    or item.get("est") or 0)
                                confirm_floor()
                            elif st in ("locked", "closed"):
                                queue.pop(idx)
                            else:
                                item["_wake"] = time.time() + 5
                        elif time.time() - item["_saving"] > 120:
                            print(f"   skipping ${known} upgrade -- income "
                                  f"too slow")
                            queue.pop(idx)
                        else:
                            item["_wake"] = time.time() + 3
                    elif broke_at[0] is not None \
                            and (cash := sane_cash()) is not None \
                            and watermark_holds(cash):
                        msg = (f"{tname} path{pi + 1}: watermark hold "
                               f"(${cash} <= ${broke_at[0]}+100)")
                        if item.get("_dbg") != msg:
                            item["_dbg"] = msg
                            dbg(msg)
                        item["_wake"] = time.time() + 5
                    else:
                        # One panel session buys every DUE tier for this
                        # tower (open/buy.../close), not one tier per open.
                        # The helper pops the tiers that landed, sleeps or
                        # drops the rest, and books the spend.
                        if do_batched_upgrade(item, entry, landed, pi, tier,
                                              ttype, tname) > 0:
                            attempts = 0

        endgame = (play_out and last_round is not None
                   and last_round >= final_round)
        if endgame and looks_victorious(screen):
            time.sleep(0.4)
            if looks_victorious(screen):      # confirmed on two samples
                dbg("VICTORY screen recognized visually")
                outcome = "survived"
                break
        if misreads == 12:                    # HUD covered a while
            if looks_defeated(screen):
                dbg("DEFEAT screen recognized (via dark counter)")
                outcome = "defeat"
                break
            dbg("counter dark for a while -- clearing UI")
            restored = clear_ui(screen, cfg)
            if endgame and not restored:
                # The HUD stayed covered THROUGH a clear attempt on the
                # final round, and it is NOT the defeat screen. A
                # transient overlay (a stuck ghost, a level-up/reward
                # popup) would have been dismissed by clear_ui and the
                # HUD would be back. Only the victory screen -- the game
                # is over, there is no live HUD left to restore --
                # survives this. That is the win. (Requiring the clear
                # to FAIL is what stops a dismissible popup from faking
                # a victory the way a bare misread count would.)
                dbg("HUD stayed covered through a clear on the final "
                    "round with no defeat screen -- victory screen")
                outcome = "survived"
                break
        if last_round is not None and last_round >= final_round:
            if not play_out or freeplay:
                # Freeplay is endless: there is NO victory screen at the
                # target, so REACHING it IS the win -- don't wait for a
                # screen that will never come. (Non-play_out farm runs also
                # stop here, as before.)
                outcome = "survived"
                break
            # Safety net for a win the covered-HUD path above missed:
            # the final round is reached, every planned buy is done, and
            # the counter has sat unchanged for minutes with lives never
            # hitting 0 (a real leak trips the defeat/lives checks
            # first). Gated on an empty queue so it can never fire
            # mid-plan while the round is legitimately still grinding
            # through the build.
            if not queue:
                final_seen = final_seen or time.time()
                if time.time() - final_seen > 240:
                    dbg("final round + build complete, counter stable "
                        "4 min, lives up -- counting as survived")
                    outcome = "survived"
                    break
            else:
                final_seen = None
        else:
            final_seen = None
        if misreads >= 25:
            # The round counter has been unreadable for a long stretch.
            # Before doing anything drastic, ask the OTHER half of the
            # HUD: readable, positive LIVES mean the game is ALIVE and
            # almost certainly winning -- only the round-counter OCR is
            # struggling (a MOAB or effect over the box, a drifted
            # crop). Restarting here throws away a live game, and that
            # false mid-round restart is exactly the "it randomly thinks
            # it lost the HUD and restarts" failure. So DON'T give up
            # while lives are alive: try to HEAL the counter (clear
            # stray UI, re-locate the box from the gear), and keep
            # playing. Only conclude the episode when a build-complete
            # defense has held blind for minutes (it effectively won) --
            # or when lives go unreadable too (the whole HUD is gone).
            alive = (lives is not None and lives > 0
                     and not looks_defeated(screen))
            if alive:
                blind_since = blind_since or time.time()
                now = time.time()
                blind_for = now - blind_since
                if now - last_blind_recovery > 12:
                    last_blind_recovery = now
                    dbg(f"round counter unreadable ({misreads}x) but "
                        f"lives={lives}: game is alive -- NOT "
                        f"restarting. Clearing UI and re-locating the "
                        f"counter.")
                    clear_ui(screen, cfg)
                    try:
                        preflight_round_box(screen, cfg, recalibrate=True)
                    except Exception:
                        pass          # recalibration is best-effort
                # A defense holding blind with lives still up has
                # effectively survived. End a build-complete run after
                # 3 min, ANY run after 6 min, so a permanently dead
                # counter can't hang the episode. (A blind run whose
                # lives FALL is ended as a defeat by the frozen-round
                # net above, which triggers on lives < 40.)
                if (not queue and blind_for > 180) or blind_for > 360:
                    dbg("counter dark for minutes but lives held -- the "
                        "defense survived")
                    outcome = "survived"
                    break
            else:
                # Whole HUD gone (lives unreadable too), or lives at 0.
                # In a play-out at/after the final round that is the
                # victory screen covering everything (the defeat screen
                # keeps the counter visible), not a lost window.
                if play_out and last_round is not None \
                        and last_round >= final_round \
                        and lives is None and not looks_defeated(screen):
                    dbg("whole HUD covered at the final round -- victory")
                    outcome = "survived"
                    break
                # A one-life game that was alive a moment ago must NOT be
                # thrown away on a transient dark HUD: an ability/effect can
                # blank both the counter and the lives digit for a few
                # frames. Give the dark spell time to clear -- keep healing
                # the HUD -- and only conclude hud_lost if the WHOLE HUD
                # stays unreadable for a stretch. A real leak-out shows the
                # defeat screen (caught above), so unreadable-lives here is
                # a misread, not a loss. (Multi-life runs keep the old
                # immediate exit: a leak to 0 there is unambiguous.)
                if one_life and not looks_defeated(screen):
                    hud_dark_since = hud_dark_since or time.time()
                    if time.time() - hud_dark_since < 45:
                        if time.time() - last_blind_recovery > 12:
                            last_blind_recovery = time.time()
                            dbg("whole HUD dark but was alive -- healing, "
                                "not giving up yet")
                            clear_ui(screen, cfg)
                            try:
                                preflight_round_box(screen, cfg,
                                                    recalibrate=True)
                            except Exception:
                                pass
                    else:
                        dbg("whole HUD dark 45s with no defeat screen -- "
                            "window lost")
                        break
                else:
                    break             # genuinely unknown/dead state
        if not queue and not observing:
            observing = True
            print("   build complete -- observing until survive/abort")
        time.sleep(1.2 if observing else 0.6)
    clean = [{k: v for k, v in t.items() if not k.startswith("_")}
             for t in towers.values()]
    return (outcome, last_round, clean, lives_by_round,
            cash_by_round, spent_by_round)

def _wait_until(cond, timeout, poll=0.3):
    """Poll cond() until it's truthy or timeout seconds pass."""
    t_end = time.time() + timeout
    while True:
        if cond():
            return True
        if time.time() >= t_end:
            return False
        time.sleep(poll)


def _confirm_restart_dialog(screen, cfg):
    """The RESTART? dialog is up: press its green RESTART until the
    dialog actually closes. A click that leaves the dialog standing did
    not restart anything and must not count as one."""
    for _ in range(3):
        click_norm(screen, cfg["restart_confirm"])
        time.sleep(0.9)
        if not looks_restart_confirm(screen):
            return True
    return False


def _fresh_game_verified(screen, cfg, start_round, timeout=25):
    """After a CONFIRMED restart click: wait out the map reload until the
    HUD reads like a NEW game -- round at/below the start round AND lives
    alive again, on two consecutive samples. A readable low round alone
    (the old acceptance test) is NOT proof: a live game in its first
    minutes reads exactly the same, which is how completely failed
    restarts got declared done. One concession: if the lives reader is
    dead this run (it never returns anything -- recalibration only
    happens at the NEXT episode's preflight), round-only evidence is
    accepted at double the consecutive-sample count rather than failing
    a restart that in fact worked."""
    t_end = time.time() + timeout
    good = 0
    lives_ever_read = False
    while time.time() < t_end:
        if looks_defeated(screen) or looks_restart_confirm(screen):
            good = 0
        else:
            value = read_round(screen, cfg)[0]
            lives = read_lives(screen, cfg)
            lives_ever_read = lives_ever_read or lives is not None
            fresh_round = value is not None and value <= max(1, start_round)
            if fresh_round and lives is not None and lives > 0:
                good += 2
            elif fresh_round and not lives_ever_read:
                good += 1
            else:
                good = 0        # includes lives reading 0: still defeated
            if good >= 4:
                return True
        time.sleep(0.8)
    return False


def restart_game(screen, cfg, outcome, start_round=1):
    """Get back to a fresh game, verifying EVERY step by looking. Each
    attempt re-reads the screen and takes the route that matches what is
    actually there (leftover RESTART? dialog > defeat screen > pause
    menu > Esc to raise the pause menu): a restart button is only pressed
    on the screen it belongs to, the press only counts once the RESTART?
    dialog is seen, the confirm only counts once that dialog closes, and
    success is only declared by _fresh_game_verified. The previous
    version fired all three clicks blind and accepted any readable round
    <= start_round -- on a live early-round game a failed restart
    'verified' instantly (the bot moved on convinced it had restarted),
    and on an unrecognized defeat screen the blind confirm clicked into
    a dialog that had never opened."""
    for attempt in range(4):
        if attempt:
            time.sleep(1.2)
        if looks_restart_confirm(screen):
            how = "leftover RESTART? dialog"
        elif looks_defeated(screen):
            spot = find_defeat_restart(screen, cfg)
            how = f"defeat screen (RESTART at {spot[0]:.3f},{spot[1]:.3f})"
            click_norm(screen, spot)
        elif looks_paused(screen):
            how = "pause menu"
            click_norm(screen, cfg["pause_restart"])
        else:
            clear_ui(screen, cfg)
            press_key("esc")
            if _wait_until(lambda: looks_paused(screen), 3.0):
                how = "pause menu (via Esc)"
                click_norm(screen, cfg["pause_restart"])
            else:
                # Esc raised nothing. Likeliest: an end screen the defeat
                # detector missed -- and round-readability proves nothing,
                # since the round counter stays visible on the defeat
                # screen. One press of the defeat RESTART, kept honest by
                # the dialog gate below (on a live game it just clicks
                # map and the missing dialog sends us around again).
                how = "unrecognized screen -- trying the defeat button"
                click_norm(screen, find_defeat_restart(screen, cfg))
        dbg(f"restart attempt {attempt + 1}: {how}")
        if not _wait_until(lambda: looks_restart_confirm(screen), 4.0):
            dbg("restart: RESTART? dialog did not appear -- re-evaluating")
            continue
        if not _confirm_restart_dialog(screen, cfg):
            dbg("restart: RESTART? dialog would not close -- re-evaluating")
            continue
        if _fresh_game_verified(screen, cfg, start_round):
            dbg("restart verified: fresh HUD (round back at start, "
                "lives repopulated)")
            return True
        dbg("restart: dialog confirmed but no fresh HUD appeared")
    return False


def _log_crash(where):
    """Print AND append the current exception to crash_log.txt, so a
    cut-off console paste can never lose the evidence again."""
    err = traceback.format_exc()
    print(err)
    try:
        with open(Path(__file__).parent / "crash_log.txt", "a") as f:
            f.write(f"\n=== {datetime.now().isoformat()} {where} "
                    f"(build {BUILD})\n")
            f.write(err)
    except Exception:
        pass
    print(f"Crash in {where} -- logged to crash_log.txt.")


def make_flow_sensor(screen, track, mask_path):
    """One-shot sensor for the bloon entry direction, or None when the
    track is already oriented (the result is saved into the mask, so it
    only ever runs on a map's first episode)."""
    if track is None or track.oriented:
        return None

    def sensor(clean):
        pt = sense_flow_entry(screen, clean, track)
        if pt:
            track.orient(pt)
            try:
                d = json.loads(mask_path.read_text())
                d["flow_entry"] = pt
                mask_path.write_text(json.dumps(d))
            except Exception:
                _log_crash("flow_entry save")
            print(f"   flow sensed: bloons enter near "
                  f"[{pt[0]:.2f}, {pt[1]:.2f}] -- saved to "
                  f"{mask_path.name}")
        else:
            print("   flow not sensed this episode -- placement "
                  "stays direction-agnostic, will retry")
    return sensor


def _stable_round(screen, cfg, tries=6):
    """A round value confirmed by two consecutive equal reads, retried
    up to `tries` times. One flaky read used to collapse the starting
    round to 1 and misclassify CHIMPS (starts r6) as impoppable (r3)."""
    prev = None
    for _ in range(tries):
        v = read_round(screen, cfg)[0]
        if v is not None and v == prev:
            return v
        prev = v
        time.sleep(0.25)
    return prev


def detect_loaded_rung(screen, cfg, flag_difficulty=None,
                       flag_mode=None):
    """What game is actually loaded? Starting lives pin the difficulty
    (200/150/100 = easy/medium/hard, 1 = a one-life mode) and the
    starting round separates CHIMPS (starts at round 6) from impoppable
    (round 3). Returns (lives, start_round, difficulty, game_mode) --
    explicit flags always win over detection.

    The observed round only SEEDS detection; once the rung is known
    (detected or forced by flags), its canonical start round from
    campaign.RUNG_INFO is authoritative, so a single flaky OCR read at
    detection time can't send the whole run down the wrong ladder."""
    import campaign
    lv = read_lives(screen, cfg)
    seen = _stable_round(screen, cfg)
    seed_round = seen if (seen is not None and seen <= 6) else 1
    rung = campaign.detect_rung(lv, seed_round)
    difficulty = flag_difficulty or (rung[0] if rung else None)
    game_mode = flag_mode or (rung[1] if rung else "standard")
    # Canonical start round for the resolved rung wins over the read.
    start_round = campaign.rung_start(difficulty, game_mode) \
        if difficulty else seed_round
    if rung and flag_difficulty and rung[0] != flag_difficulty:
        print(f"NOTE: HUD ({lv} lives, start r{seed_round}) suggests "
              f"'{rung[0]}/{rung[1]}' but --difficulty "
              f"{flag_difficulty} was given -- using the flag.")
    return lv, start_round, difficulty, game_mode


def cmd_farm(args):
    cfg = load_config()
    setup_tesseract(cfg)
    screen, hwnd = make_screen(cfg)

    missing = [k for k in ("defeat_restart", "pause_restart",
                           "restart_confirm") if not cfg.get(k)]
    if missing:
        sys.exit(
            "farm needs one-time restart calibration. Missing in "
            f"config.json: {', '.join(missing)}.\n"
            "Use `locate` and hover: the defeat screen's RESTART button "
            "(lose once on purpose),\nthe pause menu's RESTART (press "
            "Esc), and the confirm dialog's OK button.")

    mask_path = find_mask_path({"map": args.name}, Path.cwd() / "_")
    if mask_path is None:
        sys.exit(f"No mask found for '{args.name}' -- run `scan "
                 f"{args.name}` first.")
    load_mask(mask_path)
    # (safe deselect spots are computed per episode, from each genome)

    global PRICE_DIFFICULTY
    PRICE_DIFFICULTY = args.difficulty or "easy"
    rng = random.Random(args.seed)
    runs_path = Path(__file__).parent / "runs_log.jsonl"

    print(f"Farming {args.episodes} episodes on '{args.name}' "
          f"({args.towers} towers, to round "
          f"{args.final_round or 'auto (difficulty final round)'}, "
          f"difficulty {args.difficulty or 'auto-detect'}).")
    print("Load the map fresh (round 1, no towers). Starting in 5 "
          "seconds...")
    focus_game_window(hwnd)
    time.sleep(5)
    if not preflight_round_box(screen, cfg):
        sys.exit("Round counter unreadable -- fix with `watch` first.")
    if not preflight_cash_box(screen, cfg):
        print("Cash not readable yet -- will recalibrate at each episode "
              "start (make sure the map is loaded).")
    if not preflight_lives_box(screen, cfg):
        print("Lives not readable yet -- will recalibrate at each episode "
              "start. Defeat detection needs it, so if episode 1 still "
              "can't read lives, stop and debug with `watch`.")

    import campaign
    lv, start_round, difficulty, game_mode = detect_loaded_rung(
        screen, cfg, flag_difficulty=args.difficulty)
    if args.difficulty is None:
        args.difficulty = difficulty or "easy"
        print(f"Rung auto-detected from {lv} starting lives / start "
              f"round {start_round}: {args.difficulty}/{game_mode}")
    PRICE_DIFFICULTY = args.difficulty

    if args.final_round is None:
        # Learn the WHOLE game, not the easy half: survival means the
        # rung's real final round. Stopping at 40 on hard scored
        # half-finished runs as perfect wins and never scheduled a
        # single end-game buy.
        args.final_round = campaign.rung_target(args.difficulty,
                                                game_mode)
        print(f"Target round auto-set to {args.final_round} for "
              f"{args.difficulty}/{game_mode} (--final-round "
              f"overrides).")

    rung_lives = campaign.RUNG_INFO.get(
        (args.difficulty, game_mode), {}).get("lives")
    one_life = (game_mode == "chimps" or args.difficulty == "impoppable"
                or (rung_lives is not None and rung_lives <= 3)
                or (lv is not None and lv <= 3))
    if args.abort_lives and one_life:
        # A 1-life mode with --abort-lives 50 would abort every episode
        # on sight ("0 < 1 < 50"). Any leak already ends those games.
        # Keyed on the RUNG, not just the lives read, so an unreadable
        # startup lives count on a chimps/impoppable rung still disables
        # it instead of insta-aborting once per-episode calibration
        # recovers the "1".
        print(f"one-life rung ({args.difficulty}/{game_mode}) -- "
              f"disabling the early-abort (defeat itself is the signal "
              f"here).")
        args.abort_lives = 0

    if start_round != 1:
        print(f"Game starts at round {start_round} on this rung -- "
              f"restart verification will expect it.")

    # STAGE 3: the meta brain. Spreadsheet-derived priors + everything in
    # runs_log.jsonl -> Thompson-sampled layouts that start meta-informed
    # and drift toward whatever actually survives on THIS map. Missing
    # knowledge file or --no-meta degrades cleanly to uniform random.
    brain, tower_pool, track = None, FARM_TOWERS, None
    if not args.no_meta:
        try:
            import meta as meta_mod
            api = getattr(meta_mod, "META_API", 0)
            if api != 4:
                print("!" * 64)
                print(f"!! meta.py reports API {api}, this mk.py needs 4.")
                print("!! You are mixing files from different versions --")
                print("!! re-download the whole branch and delete any")
                print("!! __pycache__ folders. Farming WITHOUT the meta")
                print("!! brain so nothing crashes.")
                print("!" * 64)
                raise ImportError(f"meta API {api} != 4")
            brain = meta_mod.MetaBrain(args.name, args.difficulty,
                                       target_round=args.final_round,
                                       explore=args.explore,
                                       evolve=not args.no_evolve,
                                       mode=game_mode,
                                       start_round=start_round)
            if args.pool == "full":
                tower_pool = FARM_TOWERS + meta_mod.META_EXTRA_TOWERS
            n_map = sum(1 for r in brain.history if brain.usable(r))
            print(f"Meta brain ON: {len(brain.towers)} tower priors, "
                  f"{n_map} usable past episodes on this map, "
                  f"explore={brain.explore:.2f}, "
                  f"evolution={'on' if not args.no_evolve else 'off'}, "
                  f"pool={len(tower_pool)} towers.")
            mask_data = json.loads(mask_path.read_text())
            track = meta_mod.TrackModel(mask_data)
            if track.ok:
                if mask_data.get("flow_entry"):
                    track.orient(mask_data["flow_entry"])
                print(f"Track model: {len(track.cells)} path cells; flow "
                      + ("known (entry saved in mask)" if track.oriented
                         else "unknown -- will watch where bloons enter "
                              "on episode 1"))
            else:
                track = None
                print("Track model unavailable (no track blob in mask) "
                      "-- placement falls back to distance pools.")
        except Exception as e:
            print(f"Meta brain unavailable ({e}) -- falling back to "
                  f"uniform random layouts.")
            brain = None

    for ep in range(1, args.episodes + 1):
        if brain:
            pools = {"near": MASK_NEAR, "mid": MASK_MID,
                     "all": MASK_POINTS, "roomy": MASK_ROOMY}
            genome = brain.next_genome(
                rng, args.towers, pools, is_locked=is_locked,
                large_towers=LARGE_TOWERS, tower_pool=tower_pool,
                price_of=lambda t, p=None, tr=None: PRICES.get(
                    price_key(t) if p is None else price_key(t, p, tr)),
                track=track, hero=not args.no_hero)
        else:
            genome = random_genome(rng, args.towers,
                                   hero=not args.no_hero)
        sensor = make_flow_sensor(screen, track, mask_path)
        print(f"\n=== Episode {ep}/{args.episodes}: "
              f"{sum(1 for g in genome if g['do'] == 'place')} towers, "
              f"{sum(1 for g in genome if g['do'] == 'upgrade')} upgrades")
        if brain:
            print(brain.describe_genome(genome))
        danger = {r for t in (brain.threats if brain else [])
                  for r in t.get("rounds", [])}
        try:
            (outcome, reached, towers, lives_by_round, cash_by_round,
             spent_by_round) = run_episode(
                screen, cfg, genome, args.final_round,
                abort_lives=args.abort_lives, flow_sensor=sensor,
                danger_rounds=danger, abilities=not args.no_abilities,
                one_life=one_life, mode=game_mode)
        except KeyboardInterrupt:
            raise
        except Exception:
            _log_crash(f"episode {ep}")
            print("Attempting to recover and continue.")
            (outcome, reached, towers, lives_by_round, cash_by_round,
             spent_by_round) = ("crashed", None, [], {}, {}, {})
            try:
                clear_ui(screen, cfg)
            except Exception:
                _log_crash(f"episode {ep} (recovery clear_ui)")
        print(f"=== Episode {ep}: {outcome} at round {reached}")
        row = {"time": datetime.now().isoformat(timespec="seconds"),
               "mode": "farm", "map": args.name,
               "difficulty": args.difficulty,
               "game_mode": game_mode,
               "target_round": args.final_round,
               "start_round": start_round,
               "final_round": reached, "outcome": outcome,
               "strategy": brain.last_strategy if brain
               else {"kind": "uniform"},
               "lives_by_round": lives_by_round,
               "cash_by_round": cash_by_round,
               "spent_by_round": spent_by_round,
               "towers": towers}
        try:
            with open(runs_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        except Exception:
            _log_crash(f"episode {ep} (dataset write)")
        if brain:
            brain.observe(row)     # posteriors sharpen mid-session too
        if ep < args.episodes:
            try:
                # Clicks land wherever the OS focus is -- re-assert it, or
                # a stolen focus turns the whole restart into no-ops.
                focus_game_window(hwnd)
                ok = restart_game(screen, cfg, outcome,
                                  start_round=start_round)
            except Exception:
                _log_crash(f"episode {ep} (restart)")
                ok = False
            if not ok:
                sys.exit("Couldn't restart into a fresh game -- check the "
                         "restart calibration points in config.json.")
    print(f"\nDone. {args.episodes} labeled episodes appended to "
          f"{runs_path.name}.")


def cmd_solve(args):
    """STAGE 4: play the loaded rung to WIN it. Detects what is loaded
    (difficulty AND mode -- CHIMPS is one life starting at round 6),
    then lets the campaign policy alternate exploration episodes (which
    feed the posteriors, the outcome model, and the income curve) with
    attempt episodes (the repaired champion played straight, zero
    roulette) until the final round is actually survived. A victory is
    recorded in progress.json and the per-map ladder advances."""
    import campaign
    try:
        import meta as meta_mod
    except ImportError:
        sys.exit("solve needs meta.py next to mk.py.")
    api = getattr(meta_mod, "META_API", 0)
    if api != 4:
        sys.exit(f"meta.py reports API {api}, this mk.py needs 4 -- "
                 "re-download the whole branch and delete __pycache__.")

    cfg = load_config()
    setup_tesseract(cfg)
    screen, hwnd = make_screen(cfg)
    missing = [k for k in ("defeat_restart", "pause_restart",
                           "restart_confirm") if not cfg.get(k)]
    if missing:
        sys.exit("solve needs the same one-time restart calibration as "
                 f"farm. Missing in config.json: {', '.join(missing)}.\n"
                 "See the README's farm section (three `locate` points).")
    mask_path = find_mask_path({"map": args.name}, Path.cwd() / "_")
    if mask_path is None:
        sys.exit(f"No mask found for '{args.name}' -- run `scan "
                 f"{args.name}` first.")
    load_mask(mask_path)

    global PRICE_DIFFICULTY, FREEPLAY
    FREEPLAY = args.freeplay          # relax the round cap before preflight
    rng = random.Random(args.seed)
    runs_path = Path(__file__).parent / "runs_log.jsonl"

    print(f"Solving '{args.name}': playing until the loaded rung is "
          f"beaten (budget {args.episodes} episodes this session).")
    print("Load the map fresh (round not started) and walk away. "
          "Starting in 5 seconds...")
    focus_game_window(hwnd)
    time.sleep(5)
    if not preflight_round_box(screen, cfg):
        sys.exit("Round counter unreadable -- fix with `watch` first.")
    if not preflight_cash_box(screen, cfg):
        print("Cash not readable yet -- will recalibrate at each "
              "episode start.")
    if not preflight_lives_box(screen, cfg):
        print("Lives not readable yet -- will recalibrate at each "
              "episode start. Rung detection needs it, so pass "
              "--difficulty/--mode if detection fails.")

    if args.freeplay:
        # Freeplay: the game continued past a beaten mode, so lives/round
        # match no rung and the counter is a bare number that climbs past
        # 100. Skip detection -- the user names the difficulty (for the
        # right price book / income; freeplay keeps that mode's economy)
        # and the target round to push to.
        difficulty = args.difficulty or "hard"
        game_mode = "standard"
        lv = read_lives(screen, cfg)
        start_round = _stable_round(screen, cfg) or 1
        target = args.final_round or 100
        print(f"Freeplay: '{difficulty}' economy, currently ~round "
              f"{start_round} (lives {lv}); reading the bare round counter "
              f"and pushing to round {target}.")
    else:
        lv, start_round, difficulty, game_mode = detect_loaded_rung(
            screen, cfg, flag_difficulty=args.difficulty,
            flag_mode=args.mode)
        if difficulty is None:
            sys.exit(f"Couldn't detect the loaded difficulty (lives read "
                     f"{lv}, start round {start_round}) -- pass "
                     f"--difficulty (and --mode chimps if applicable, or "
                     f"--freeplay for a freeplay game).")
        target = args.final_round or campaign.rung_target(difficulty,
                                                          game_mode)
    PRICE_DIFFICULTY = difficulty
    # One-life is a property of the RUNG, not of one startup lives read.
    # Deriving it from `lv` alone meant an unreadable-lives launch (the
    # very case the preflight tells the user to fix with --mode chimps)
    # kept abort_lives=50, and the per-episode recalibration then read
    # 1 life and insta-aborted every episode as "0 < 1 < 50".
    rung_lives = campaign.RUNG_INFO.get((difficulty, game_mode), {}) \
        .get("lives")
    one_life = (game_mode == "chimps" or difficulty == "impoppable"
                or (rung_lives is not None and rung_lives <= 3)
                or (lv is not None and lv <= 3))
    abort_lives = 0 if one_life else args.abort_lives
    rung_name = difficulty if game_mode == "standard" else game_mode
    print(f"Rung: {args.name} / {rung_name} -- survive round {target} "
          f"(start r{start_round}, {lv} lives"
          + (", early-abort off: one-life mode" if one_life else "")
          + ").")
    if game_mode == "chimps":
        print("CHIMPS: pops-only income (paced by the learned income "
              "curve), one life, no selling -- the run only counts "
              "once round 100 actually ENDS.")

    brain = meta_mod.MetaBrain(args.name, difficulty,
                               target_round=target,
                               explore=args.explore, evolve=True,
                               mode=game_mode, start_round=start_round)
    tower_pool = FARM_TOWERS + (meta_mod.META_EXTRA_TOWERS
                                if args.pool == "full" else [])
    mask_data = json.loads(mask_path.read_text())
    track = meta_mod.TrackModel(mask_data)
    if track.ok:
        if mask_data.get("flow_entry"):
            track.orient(mask_data["flow_entry"])
    else:
        track = None

    progress = campaign.Progress()
    policy = campaign.EpisodePolicy()
    for row in brain.history:
        was_attempt = str((row.get("strategy") or {})
                          .get("kind", "")).startswith("attempt")
        policy.update(brain._reward(row), was_attempt=was_attempt)
    ol = brain._outcome_learner(track)
    gate = ol.gate() if ol is not None else {}
    print(f"Brain: {len(brain.history)} past episodes on this rung, "
          f"best progress {policy.best:.2f}; outcome model "
          f"{'OPEN' if gate.get('open') else 'closed'}"
          + (f" (AUC {gate['auc']:.2f})" if gate.get("auc") else "")
          + f"; elites {len(brain.elites())}.")

    pools = {"near": MASK_NEAR, "mid": MASK_MID,
             "all": MASK_POINTS, "roomy": MASK_ROOMY}
    danger = {r for t in brain.threats for r in t.get("rounds", [])}

    def price_of(t, p=None, tr=None):
        return PRICES.get(price_key(t) if p is None
                          else price_key(t, p, tr))

    victory = False
    for ep in range(1, args.episodes + 1):
        decision = policy.decide(rng)
        brain.explore = decision["explore"]
        genome = None
        if decision["kind"] == "attempt":
            genome = brain.attempt_genome(
                rng, pools, is_locked=is_locked,
                large_towers=LARGE_TOWERS, tower_pool=tower_pool,
                price_of=price_of, track=track, hero=not args.no_hero)
        if genome is None:
            brain.explore = decision["explore"]
            genome = brain.next_genome(
                rng, args.towers, pools, is_locked=is_locked,
                large_towers=LARGE_TOWERS, tower_pool=tower_pool,
                price_of=price_of, track=track, hero=not args.no_hero,
                novelty=decision.get("novelty", False))
        sensor = make_flow_sensor(screen, track, mask_path)
        print(f"\n=== Episode {ep}/{args.episodes} "
              f"[{decision['kind']}, explore "
              f"{decision['explore']:.2f}]")
        print(brain.describe_genome(genome))
        try:
            (outcome, reached, towers, lives_by_round, cash_by_round,
             spent_by_round) = run_episode(
                screen, cfg, genome, target, abort_lives=abort_lives,
                flow_sensor=sensor, play_out=True,
                danger_rounds=danger,
                abilities=not args.no_abilities, one_life=one_life,
                freeplay=getattr(args, "freeplay", False), mode=game_mode)
        except KeyboardInterrupt:
            raise
        except Exception:
            _log_crash(f"solve episode {ep}")
            print("Attempting to recover and continue.")
            (outcome, reached, towers, lives_by_round, cash_by_round,
             spent_by_round) = ("crashed", None, [], {}, {}, {})
            try:
                clear_ui(screen, cfg)
            except Exception:
                _log_crash(f"solve episode {ep} (recovery clear_ui)")
        if outcome == "survived":
            outcome = "victory"   # play_out=True means round N ENDED
        print(f"=== Episode {ep}: {outcome} at round {reached}")
        row = {"time": datetime.now().isoformat(timespec="seconds"),
               "mode": "solve", "map": args.name,
               "difficulty": difficulty, "game_mode": game_mode,
               "target_round": target, "start_round": start_round,
               "final_round": reached, "outcome": outcome,
               "strategy": brain.last_strategy,
               "lives_by_round": lives_by_round,
               "cash_by_round": cash_by_round,
               "spent_by_round": spent_by_round,
               "towers": towers}
        try:
            with open(runs_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        except Exception:
            _log_crash(f"solve episode {ep} (dataset write)")
        brain.observe(row)
        policy.update(brain._reward(row),
                      was_attempt=decision["kind"] == "attempt")
        try:
            progress.record_episode(
                args.name, difficulty, game_mode, outcome, reached,
                was_attempt=decision["kind"] == "attempt")
        except Exception:
            # progress.json is a convenience scoreboard, not the run --
            # a transient file lock (Windows AV/editor/OneDrive) must
            # not crash an unattended session or skip the restart.
            _log_crash(f"solve episode {ep} (progress write)")
        if outcome == "victory":
            victory = True
            break
        if ep < args.episodes:
            try:
                focus_game_window(hwnd)
                ok = restart_game(screen, cfg, outcome,
                                  start_round=start_round)
            except Exception:
                _log_crash(f"solve episode {ep} (restart)")
                ok = False
            if not ok:
                sys.exit("Couldn't restart into a fresh game -- all "
                         "progress is saved; check the restart "
                         "calibration points in config.json.")

    print()
    if victory:
        print(f"*** {args.name} / {rung_name} BEATEN -- round {target} "
              f"survived. ***")
        nxt = progress.next_rung(args.name)
        if nxt:
            nxt_name = nxt[0] if nxt[1] == "standard" else nxt[1]
            print(f"Next rung: load {args.name} on '{nxt_name}' and "
                  f"run `python mk.py solve {args.name}` again -- "
                  f"everything learned carries over.")
        else:
            print(f"{args.name} is COMPLETE -- every rung up through "
                  f"CHIMPS beaten. Scan a new map and keep climbing.")
    else:
        best = progress.rung(args.name, difficulty,
                             game_mode)["best_round"]
        print(f"Budget exhausted without a win -- deepest round so far "
              f"{best}/{target}. Everything learned is saved: run "
              f"`solve` again to keep going, or `learn {args.name}` to "
              f"see what the brain believes.")
    print()
    for line in progress.board([args.name]):
        print(line)


def cmd_deploy(args):
    """Run the FINISHED, trained model as a bot. Where `solve` keeps
    exploring -- throwing fresh and evolved layouts to gather data --
    `deploy` does none of that: it loads everything `farm`/`solve`
    already learned (tower posteriors, elite layouts, income curve, the
    price book -- all reconstructed from runs_log.jsonl) and plays the
    CHAMPION straight. The single best layout found for the loaded rung,
    repaired for full threat coverage, played through to an actual win.
    Pure exploitation: no roulette, no mutation, and -- unless --log is
    passed -- nothing written back to the training set, so re-running the
    same winner never skews the posteriors that `solve` depends on. If no
    champion has been trained for this rung yet, it says so plainly
    instead of improvising a random layout."""
    import campaign
    try:
        import meta as meta_mod
    except ImportError:
        sys.exit("deploy needs meta.py next to mk.py.")
    api = getattr(meta_mod, "META_API", 0)
    if api != 4:
        sys.exit(f"meta.py reports API {api}, this mk.py needs 4 -- "
                 "re-download the whole branch and delete __pycache__.")
    if args.games < 1:
        sys.exit("--games must be at least 1.")

    cfg = load_config()
    setup_tesseract(cfg)
    screen, hwnd = make_screen(cfg)
    if args.games > 1:
        missing = [k for k in ("defeat_restart", "pause_restart",
                               "restart_confirm") if not cfg.get(k)]
        if missing:
            sys.exit("deploy --games >1 restarts between games, which "
                     "needs the same one-time calibration as farm. "
                     f"Missing in config.json: {', '.join(missing)}.\n"
                     "See the README's farm section (three `locate` "
                     "points), or play a single game with --games 1.")
    mask_path = find_mask_path({"map": args.name}, Path.cwd() / "_")
    if mask_path is None:
        sys.exit(f"No mask found for '{args.name}' -- run `scan "
                 f"{args.name}` first.")
    load_mask(mask_path)

    global PRICE_DIFFICULTY
    rng = random.Random(args.seed)
    runs_path = Path(__file__).parent / "runs_log.jsonl"

    print(f"Deploying the trained bot on '{args.name}': the champion "
          f"strategy, played to win ({args.games} "
          + ("game" if args.games == 1 else "games") + " this session).")
    print("Load the map fresh (round not started) and walk away. "
          "Starting in 5 seconds...")
    focus_game_window(hwnd)
    time.sleep(5)
    if not preflight_round_box(screen, cfg):
        sys.exit("Round counter unreadable -- fix with `watch` first.")
    if not preflight_cash_box(screen, cfg):
        print("Cash not readable yet -- will recalibrate at each "
              "game start.")
    if not preflight_lives_box(screen, cfg):
        print("Lives not readable yet -- will recalibrate at each "
              "game start. Rung detection needs it, so pass "
              "--difficulty/--mode if detection fails.")

    lv, start_round, difficulty, game_mode = detect_loaded_rung(
        screen, cfg, flag_difficulty=args.difficulty,
        flag_mode=args.mode)
    if difficulty is None:
        sys.exit(f"Couldn't detect the loaded difficulty (lives read "
                 f"{lv}, start round {start_round}) -- pass "
                 f"--difficulty (and --mode chimps if applicable).")
    PRICE_DIFFICULTY = difficulty
    target = args.final_round or campaign.rung_target(difficulty,
                                                      game_mode)
    # One-life is a property of the RUNG (see the note in cmd_solve): a
    # flaky startup lives read must not leave abort armed on CHIMPS.
    rung_lives = campaign.RUNG_INFO.get((difficulty, game_mode), {}) \
        .get("lives")
    one_life = (game_mode == "chimps" or difficulty == "impoppable"
                or (rung_lives is not None and rung_lives <= 3)
                or (lv is not None and lv <= 3))
    # Deploy plays every game to its real conclusion -- no early abort by
    # default (default 0), so a champion that starts to leak still
    # finishes the round it dies on and reports an honest final_round.
    abort_lives = 0 if one_life else args.abort_lives
    rung_name = difficulty if game_mode == "standard" else game_mode
    print(f"Rung: {args.name} / {rung_name} -- survive round {target} "
          f"(start r{start_round}, {lv} lives).")

    # explore=0.0 is the whole point: the finished model, exploited, not
    # explored. attempt_genome carries no exploration roulette anyway.
    brain = meta_mod.MetaBrain(args.name, difficulty,
                               target_round=target, explore=0.0,
                               evolve=True, mode=game_mode,
                               start_round=start_round)
    tower_pool = FARM_TOWERS + (meta_mod.META_EXTRA_TOWERS
                                if args.pool == "full" else [])
    mask_data = json.loads(mask_path.read_text())
    track = meta_mod.TrackModel(mask_data)
    if track.ok:
        if mask_data.get("flow_entry"):
            track.orient(mask_data["flow_entry"])
    else:
        track = None

    champion = brain.elites(top=1)
    if not champion:
        sys.exit(
            f"No trained champion for {args.name} / {rung_name} yet -- "
            f"the model has nothing to deploy.\n"
            f"Train one first: `python mk.py solve {args.name}` (or "
            f"`farm`) until a layout survives, then deploy it. "
            f"`python mk.py learn {args.name}` shows what the brain "
            f"currently believes.")
    ch_reward, ch_row = champion[0]
    ch_from = ("this rung" if brain.usable(ch_row)
               else f"transferred from {ch_row.get('difficulty')}/"
                    f"{ch_row.get('game_mode', 'standard')}")
    ch_layout = ", ".join(
        f"{t.get('tower')}{t.get('path') or [0, 0, 0]}"
        for t in ch_row.get("towers", []) if t.get("tower"))
    print(f"Champion (reward {ch_reward:.2f}, {ch_from}): {ch_layout}")
    if ch_reward < 0.999:
        # The best layout on record has never actually SURVIVED the
        # target -- deploy will play its best, but be honest that the
        # model isn't proven on this rung yet.
        print(f"   NOTE: this champion has not yet won {args.name} / "
              f"{rung_name} (its best reached round "
              f"{ch_row.get('final_round')}/{target}). It plays the "
              f"strongest layout learned so far; run `solve "
              f"{args.name}` if it can't quite close the game.")

    progress = campaign.Progress()
    pools = {"near": MASK_NEAR, "mid": MASK_MID,
             "all": MASK_POINTS, "roomy": MASK_ROOMY}
    danger = {r for t in brain.threats for r in t.get("rounds", [])}

    def price_of(t, p=None, tr=None):
        return PRICES.get(price_key(t) if p is None
                          else price_key(t, p, tr))

    wins = 0
    for g in range(1, args.games + 1):
        genome = brain.attempt_genome(
            rng, pools, is_locked=is_locked,
            large_towers=LARGE_TOWERS, tower_pool=tower_pool,
            price_of=price_of, track=track, hero=not args.no_hero)
        if genome is None:
            # elites() said there is a champion, so this only fires if the
            # champion row has no placeable towers -- a corrupt record.
            sys.exit("The champion layout has no placeable towers -- the "
                     f"runs_log.jsonl record for {args.name} is corrupt.")
        sensor = make_flow_sensor(screen, track, mask_path)
        print(f"\n=== Game {g}/{args.games} [deploy champion]")
        print(brain.describe_genome(genome))
        try:
            (outcome, reached, towers, lives_by_round, cash_by_round,
             spent_by_round) = run_episode(
                screen, cfg, genome, target, abort_lives=abort_lives,
                flow_sensor=sensor, play_out=True,
                danger_rounds=danger,
                abilities=not args.no_abilities, one_life=one_life,
                freeplay=getattr(args, "freeplay", False), mode=game_mode)
        except KeyboardInterrupt:
            raise
        except Exception:
            _log_crash(f"deploy game {g}")
            print("Attempting to recover and continue.")
            (outcome, reached, towers, lives_by_round, cash_by_round,
             spent_by_round) = ("crashed", None, [], {}, {}, {})
            try:
                clear_ui(screen, cfg)
            except Exception:
                _log_crash(f"deploy game {g} (recovery clear_ui)")
        if outcome == "survived":
            outcome = "victory"   # play_out=True means the final round ENDED
        won = outcome == "victory"
        wins += 1 if won else 0
        print(f"=== Game {g}: {outcome} at round {reached}"
              + (" -- WIN" if won else ""))
        if args.log:
            # Opt-in: treat this deploy game as fresh evidence and let the
            # champion adapt in-session. Off by default so deployment
            # leaves the training distribution untouched.
            row = {"time": datetime.now().isoformat(timespec="seconds"),
                   "mode": "solve", "map": args.name,
                   "difficulty": difficulty, "game_mode": game_mode,
                   "target_round": target, "start_round": start_round,
                   "final_round": reached, "outcome": outcome,
                   "strategy": brain.last_strategy,
                   "lives_by_round": lives_by_round,
                   "cash_by_round": cash_by_round,
                   "spent_by_round": spent_by_round,
                   "towers": towers}
            try:
                with open(runs_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                brain.observe(row)
            except Exception:
                _log_crash(f"deploy game {g} (dataset write)")
        try:
            progress.record_episode(
                args.name, difficulty, game_mode, outcome, reached,
                was_attempt=True)
        except Exception:
            _log_crash(f"deploy game {g} (progress write)")
        if g < args.games:
            try:
                focus_game_window(hwnd)
                ok = restart_game(screen, cfg, outcome,
                                  start_round=start_round)
            except Exception:
                _log_crash(f"deploy game {g} (restart)")
                ok = False
            if not ok:
                sys.exit("Couldn't restart into a fresh game -- all "
                         "progress is saved; check the restart "
                         "calibration points in config.json.")

    print()
    print(f"Deploy session done: {wins}/{args.games} "
          + ("game" if args.games == 1 else "games")
          + f" won on {args.name} / {rung_name}.")
    for line in progress.board([args.name]):
        print(line)


def cmd_campaign(args):
    """Offline: the ladder scoreboard. Maps come from masks/ (one scan
    per map = the bot can play it) plus anything already recorded in
    progress.json."""
    import campaign
    progress = campaign.Progress()
    maps = set()
    d = Path(__file__).parent / "masks"
    if d.is_dir():
        for p in sorted(d.glob("*.json")):
            stem = p.stem
            for suffix in ("_dart", "_sub"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
            maps.add(stem)
    maps |= {key.split("|")[0] for key in progress.data["rungs"]}
    if not maps:
        print("No maps known yet -- `scan <map>` one, then `solve` it.")
        return
    for line in progress.board(sorted(maps)):
        print(line)


def cmd_learn(args):
    """Offline: show the meta brain's current beliefs for a rung -- which
    towers its own episodes confirm or contradict the spreadsheet on,
    the elite layouts evolution draws from, where defenses die, whether
    the outcome model has earned its vote, and the map's prime real
    estate if a scan mask exists."""
    import campaign
    import meta as meta_mod
    target = args.final_round or campaign.rung_target(args.difficulty,
                                                      args.mode)
    brain = meta_mod.MetaBrain(args.name, args.difficulty,
                               target_round=target, mode=args.mode,
                               start_round=campaign.rung_start(
                                   args.difficulty, args.mode))
    track = mask_pts = None
    mask_path = find_mask_path({"map": args.name}, Path.cwd() / "_")
    if mask_path:
        try:
            data = json.loads(mask_path.read_text())
            track = meta_mod.TrackModel(data)
            if track.ok and data.get("flow_entry"):
                track.orient(data["flow_entry"])
            mask_pts = data.get("valid_strict") or data.get("valid")
        except Exception:
            track = None
    print(brain.report(track=track, mask_points=mask_pts))


def cmd_graph(args):
    """Render training progress (runs_log.jsonl + progress.json) to a
    self-contained HTML dashboard. Pure-stdlib generator in tools/, so it
    needs no game and no extra dependencies."""
    tools_dir = Path(__file__).parent / "tools"
    sys.path.insert(0, str(tools_dir))
    import plot_progress
    argv = []
    if getattr(args, "out", None):
        argv += ["--out", args.out]
    if getattr(args, "open", False):
        argv += ["--open"]
    return plot_progress.main(argv)


def cmd_play(args):
    cfg = load_config()
    setup_tesseract(cfg)
    screen, hwnd = make_screen(cfg)

    plan = json.loads(Path(args.plan).read_text())
    validate_plan(plan)
    snap_plan_to_mask(plan, Path(args.plan).resolve())
    global PRICE_DIFFICULTY
    mode_slug = _slug(str(plan.get("difficulty", "")) + " "
                      + str(plan.get("mode", "")))
    PRICE_DIFFICULTY = next((d for d in ("easy", "hard", "impoppable")
                             if d in mode_slug), "medium")
    est, unknown = plan_cost_estimate(plan)
    line = f"Plan cost ({PRICE_DIFFICULTY}, learned prices): ${est}"
    if unknown:
        line += (f" + {unknown} purchase(s) not yet learned -- they'll be "
                 f"recorded this run")
    print(line)
    queue = sorted(plan["actions"], key=lambda a: a["round"])
    final_round = plan.get("final_round", 40)

    # Round log -- this is the seed of your Stage 2 dataset.
    log_path = Path(__file__).parent / "rounds_log.csv"
    new_log = not log_path.exists()
    log_file = open(log_path, "a", newline="")
    log = csv.writer(log_file)
    if new_log:
        log.writerow(["timestamp", "elapsed_s", "round", "lives"])

    print(f"Plan: {args.plan}  ({len(queue)} actions, final round "
          f"{final_round})")
    print("EMERGENCY STOP: slam the mouse into the top-left corner.")
    if focus_game_window(hwnd):
        print("Focused the game window automatically -- have the map loaded. "
              "Starting in 3 seconds.\n")
        time.sleep(3)
    else:
        print("Couldn't auto-focus the game -- click on the BTD6 window! "
              "Starting in 5 seconds.\n")
        time.sleep(5)

    if not preflight_round_box(screen, cfg):
        sys.exit("Could not read or locate the round counter, so this run "
                 "would be flying blind. Load the map and debug with "
                 "`watch` first.")
    if not preflight_cash_box(screen, cfg):
        print("Cash counter not located -- placement still self-verifies "
              "via the hidden-counter trick, but wait_cash gating is off "
              "this run.")
    if not preflight_lives_box(screen, cfg):
        print("Lives counter not located -- defeat detection is off this "
              "run (the bot may idle on a defeat screen).")

    t0 = time.time()
    press_key("space")                   # start round 1
    if plan.get("fast_forward", True):
        time.sleep(0.3)
        press_key("space")               # second press toggles fast-forward

    last_round = None
    prev_read = None
    misreads = 0
    lives = prev_lives = None
    final_seen = None
    placements = {}      # planned coordinate -> where the tower really is
    towers = {}          # actual position -> {tower, at, path[t,m,b]}
    outcome = "stopped"
    try:
        while True:
            unpause_if_needed(screen, cfg)
            value, *_ = read_round(screen, cfg)
            prev_lives = lives
            lives = read_lives(screen, cfg)

            # Accept a new round only after seeing the same value twice in
            # a row -- one-frame OCR glitches then can't trigger actions.
            accepted = None
            if value is not None:
                if value == last_round:
                    accepted = value
                elif plausible(value, last_round) and value == prev_read:
                    accepted = value
                prev_read = value

            if accepted is None:
                misreads += 1
            else:
                misreads = 0
                if accepted != last_round:
                    last_round = accepted
                    print(f"[round {last_round:>3}]  lives={lives}")
                    log.writerow([datetime.now().isoformat(timespec='seconds'),
                                  round(time.time() - t0, 1), last_round,
                                  lives if lives is not None else ""])
                    log_file.flush()

            if lives == 0 and prev_lives == 0:
                # Two consecutive zero reads: that's the defeat screen.
                # (The round counter stays visible on it, so it can't be
                # the signal.)
                outcome = "defeat"
                print(f"\nDefeat at round {last_round}.")
                break
            if looks_defeated(screen):
                time.sleep(0.4)
                if looks_defeated(screen):
                    outcome = "defeat"
                    print(f"\nDefeat at round {last_round} (recognized "
                          f"visually).")
                    break

            while queue and last_round is not None \
                    and queue[0]["round"] <= last_round:
                action = queue.pop(0)
                print(f"   -> {describe(action)}")
                if "wait_cash" in action:
                    wait_for_cash(screen, cfg, action["wait_cash"])
                if action["do"] == "place":
                    st, landed = act_place(screen, cfg, action)
                    t_end = time.time() + 90
                    while st == "broke" and time.time() < t_end:
                        time.sleep(2.0)        # plan order holds: save up
                        st, landed = act_place(screen, cfg, action)
                    if landed is not None:
                        placements[tuple(action["at"])] = landed
                        towers[tuple(landed)] = {"tower": action["tower"],
                                                 "at": landed,
                                                 "path": [0, 0, 0]}
                    elif st == "no_spot":
                        print(f"      !! no placeable spot near "
                              f"{action['at']} -- move this hint")
                elif action["do"] == "upgrade" and placements:
                    at = tuple(action["at"])
                    near = min(placements, key=lambda p: (p[0] - at[0]) ** 2
                               + (p[1] - at[1]) ** 2)
                    if ((near[0] - at[0]) ** 2
                            + (near[1] - at[1]) ** 2) ** 0.5 <= 0.03:
                        action = {**action, "at": list(placements[near])}
                    entry = towers.get(tuple(action["at"]))
                    st = act_upgrade(screen, cfg, action, entry)
                    if st == "broke" and entry and 1 in action["path"]:
                        pi = action["path"].index(1)
                        need = learned_price(          # wait target: verified
                            entry["tower"].lower(), pi, entry["path"][pi] + 1)
                        if need:
                            wait_for_cash(screen, cfg, need, timeout=90)
                            act_upgrade(screen, cfg, action, entry)
                    elif st in ("no_select", "locked", "closed", "unread"):
                        print(f"      upgrade {st} -- moving on")
                else:
                    ACTION_FUNCS[action["do"]](screen, cfg, action)

            if misreads in (25, 55):
                # Something (a stuck panel, a lingering ghost) is hiding
                # the HUD. Try to clear it instead of counting down to
                # termination.
                print("   (round counter blocked -- clearing any open UI)")
                clear_ui(screen, cfg)

            if last_round is not None and last_round >= final_round \
                    and not queue:
                final_seen = final_seen or time.time()
                if misreads >= 8 or time.time() - final_seen > 120:
                    # HUD gone after the last round, or the round has sat
                    # at the final number for two minutes: victory.
                    outcome = "victory"
                    print("\nFinal round done -- looks like a win. GG!")
                    break
            else:
                final_seen = None
            if misreads >= 90:
                outcome = "counter_lost"
                print("\nCouldn't read the round counter for a long time "
                      "even after clearing UI -- stopping.")
                break
            time.sleep(0.7)
    except KeyboardInterrupt:
        outcome = "interrupted"
        print("\nStopped by user.")
    finally:
        log_file.close()
        if towers:
            print("\nFinal layout:")
            for t in towers.values():
                print(f"   {t['tower']:<10} {t['path']}  at {t['at']}")
        runs_path = Path(__file__).parent / "runs_log.jsonl"
        clean_towers = [{k: v for k, v in t.items()
                         if not k.startswith("_")} for t in towers.values()]
        with open(runs_path, "a") as f:
            f.write(json.dumps({
                "time": datetime.now().isoformat(timespec="seconds"),
                "plan": str(args.plan),
                "difficulty": PRICE_DIFFICULTY,
                "final_round": last_round,
                "final_lives": lives,
                "outcome": outcome,
                "towers": clean_towers}) + "\n")
        print(f"Round log -> {log_path.name}; run summary -> "
              f"{runs_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="BTD6 Stage-1 bot (see README.md)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("locate", help="print live mouse coordinates")
    sub.add_parser("watch", help="debug the round-counter OCR")
    p_play = sub.add_parser("play", help="execute a gameplan")
    p_play.add_argument("plan", help="path to a plan .json file")
    p_play.add_argument("--no-locks", action="store_true", dest="no_locks",
                        help="you've unlocked every upgrade with XP: never "
                             "treat a tier as XP-locked")
    p_play.add_argument("-q", "--quiet", action="store_true",
                        help="suppress per-decision debug output")
    p_scan = sub.add_parser(
        "scan", help="auto-detect every placeable spot on the current map")
    p_scan.add_argument("name", help="map name for output files, "
                                     "e.g. monkey_meadow")
    p_scan.add_argument("--tower", default="dart",
                        help="ghost to sweep with: dart=land, sub=water "
                             "(default: dart)")
    p_scan.add_argument("--step", type=float, default=0.025,
                        help="grid spacing as a fraction of the game area "
                             "(default: 0.025)")
    p_farm = sub.add_parser(
        "farm", help="STAGE 2: play random layouts unattended and log the "
                     "training dataset")
    p_farm.add_argument("name", help="map name matching your mask, "
                                     "e.g. monkey_meadow")
    p_farm.add_argument("--episodes", type=int, default=10)
    p_farm.add_argument("--towers", type=int, default=4,
                        help="towers per random layout (default: 4)")
    p_farm.add_argument("--final-round", type=int, default=None,
                        dest="final_round",
                        help="round that counts as survival; default: the "
                             "detected difficulty's real final round "
                             "(easy 40, medium 60, hard 80, impoppable "
                             "100), so the bot learns the END game too")
    p_farm.add_argument("--difficulty", default=None,
                        choices=["easy", "medium", "hard", "impoppable"],
                        help="price-book key; auto-detected from starting "
                             "lives when omitted")
    p_farm.add_argument("--abort-lives", type=int, default=50,
                        help="abort an episode once lives drop below "
                             "this (0 disables); live-game restarts are "
                             "the reliable ones")
    p_farm.add_argument("--seed", type=int, default=None,
                        help="RNG seed for reproducible layouts")
    p_farm.add_argument("--no-meta", action="store_true", dest="no_meta",
                        help="ignore meta_knowledge.json and play uniform "
                             "random layouts (the old Stage-2 behavior)")
    p_farm.add_argument("--explore", type=float, default=0.20,
                        help="fraction of decisions that ignore the meta "
                             "and go uniform random; 1.0 = pure random, "
                             "0.0 = pure exploit (default: 0.20)")
    p_farm.add_argument("--no-evolve", action="store_true", dest="no_evolve",
                        help="disable the genetic layer that mutates and "
                             "crosses over the best layouts found so far")
    p_farm.add_argument("--no-hero", action="store_true", dest="no_hero",
                        help="don't place the equipped hero (hotkey u) "
                             "as the early anchor of each episode")
    p_farm.add_argument("--pool", choices=["classic", "full"],
                        default="full",
                        help="tower pool for meta layouts: classic = the "
                             "original 10 land towers; full also allows "
                             "boomerang/mortar/spike/village/super/"
                             "engineer (default: full)")
    p_farm.add_argument("--no-abilities", action="store_true",
                        dest="no_abilities",
                        help="don't press ability hotkeys on threat "
                             "rounds")
    p_farm.add_argument("--no-locks", action="store_true", dest="no_locks",
                        help="you've unlocked every upgrade with XP: never "
                             "treat a tier as XP-locked")
    p_farm.add_argument("-q", "--quiet", action="store_true",
                        help="suppress per-decision debug output")
    p_solve = sub.add_parser(
        "solve", help="STAGE 4: play the loaded rung until it is "
                      "actually BEATEN (difficulty + mode auto-"
                      "detected, CHIMPS included)")
    p_solve.add_argument("name", help="map name matching your mask, "
                                      "e.g. monkey_meadow")
    p_solve.add_argument("--episodes", type=int, default=40,
                         help="episode budget for this session "
                              "(default: 40); progress persists either "
                              "way")
    p_solve.add_argument("--towers", type=int, default=5,
                         help="towers per fresh layout (default: 5)")
    p_solve.add_argument("--difficulty", default=None,
                         choices=["easy", "medium", "hard",
                                  "impoppable"],
                         help="override the lives-based detection")
    p_solve.add_argument("--mode", default=None,
                         choices=["standard", "chimps"],
                         help="override the mode detection (chimps = "
                              "1 life starting at round 6)")
    p_solve.add_argument("--final-round", type=int, default=None,
                         dest="final_round",
                         help="override the rung's final round")
    p_solve.add_argument("--freeplay", action="store_true",
                         help="the loaded game is in FREEPLAY (past a "
                              "beaten mode): skip lives-based rung "
                              "detection, read the bare round counter, and "
                              "push to --final-round. Pass --difficulty for "
                              "the right price book (defaults to hard); "
                              "set --final-round for the target (default "
                              "100)")
    p_solve.add_argument("--explore", type=float, default=0.20,
                         help="base exploration rate (the campaign "
                              "policy adjusts it per episode)")
    p_solve.add_argument("--abort-lives", type=int, default=50,
                         help="abort exploration episodes below this "
                              "many lives (auto-disabled on one-life "
                              "modes; 0 disables)")
    p_solve.add_argument("--seed", type=int, default=None)
    p_solve.add_argument("--no-hero", action="store_true",
                         dest="no_hero",
                         help="don't place the equipped hero")
    p_solve.add_argument("--no-abilities", action="store_true",
                         dest="no_abilities",
                         help="don't press ability hotkeys on threat "
                              "rounds")
    p_solve.add_argument("--pool", choices=["classic", "full"],
                         default="full")
    p_solve.add_argument("--no-locks", action="store_true", dest="no_locks",
                         help="you've unlocked every upgrade with XP: never "
                              "treat a tier as XP-locked (stops a stray buy "
                              "failure from blocking a path for the run)")
    p_solve.add_argument("-q", "--quiet", action="store_true",
                         help="suppress per-decision debug output")
    p_deploy = sub.add_parser(
        "deploy", help="run the FINISHED trained model as a bot: play "
                       "the champion straight to win the loaded rung "
                       "(no exploration, no learning)")
    p_deploy.add_argument("name", help="map name matching your mask, "
                                       "e.g. monkey_meadow")
    p_deploy.add_argument("--games", type=int, default=1,
                          help="how many games to play this session; "
                               ">1 auto-restarts between them (needs the "
                               "farm restart calibration) (default: 1)")
    p_deploy.add_argument("--difficulty", default=None,
                          choices=["easy", "medium", "hard",
                                   "impoppable"],
                          help="override the lives-based detection")
    p_deploy.add_argument("--mode", default=None,
                          choices=["standard", "chimps"],
                          help="override the mode detection (chimps = "
                               "1 life starting at round 6)")
    p_deploy.add_argument("--final-round", type=int, default=None,
                          dest="final_round",
                          help="override the rung's final round")
    p_deploy.add_argument("--abort-lives", type=int, default=0,
                          help="abort a game once lives drop below this "
                               "(default 0 = play every game to its "
                               "real end; one-life modes force it off)")
    p_deploy.add_argument("--log", action="store_true",
                          help="append deploy games to runs_log.jsonl as "
                               "evidence and let the champion adapt "
                               "in-session (default: off -- deployment "
                               "leaves the training set untouched)")
    p_deploy.add_argument("--seed", type=int, default=None)
    p_deploy.add_argument("--no-hero", action="store_true",
                          dest="no_hero",
                          help="don't place the equipped hero")
    p_deploy.add_argument("--no-abilities", action="store_true",
                          dest="no_abilities",
                          help="don't press ability hotkeys on threat "
                               "rounds")
    p_deploy.add_argument("--pool", choices=["classic", "full"],
                          default="full")
    p_deploy.add_argument("--no-locks", action="store_true", dest="no_locks",
                          help="you've unlocked every upgrade with XP: never "
                               "treat a tier as XP-locked")
    p_deploy.add_argument("-q", "--quiet", action="store_true",
                          help="suppress per-decision debug output")
    sub.add_parser(
        "campaign", help="show the per-map ladder scoreboard "
                         "(easy -> medium -> hard -> CHIMPS)")
    p_learn = sub.add_parser(
        "learn", help="STAGE 3: report what the meta brain has learned "
                      "from runs_log.jsonl (no game needed)")
    p_learn.add_argument("name", help="map name, e.g. monkey_meadow")
    p_learn.add_argument("--difficulty", default="easy")
    p_learn.add_argument("--mode", default="standard",
                         choices=["standard", "chimps"],
                         help="which rung's beliefs to report")
    p_learn.add_argument("--final-round", type=int, default=None,
                         dest="final_round",
                         help="round that counts as survival (default: "
                              "the rung's final round)")
    p_range = sub.add_parser(
        "measure-ranges", help="measure each tower's TRUE range from the "
                               "in-game range circle -> measured_ranges.json "
                               "(load the map, round not started, walk away)")
    p_range.add_argument("--tower", default=None,
                         help="measure just this tower (default: all)")
    p_graph = sub.add_parser(
        "graph", help="render training progress from runs_log.jsonl to a "
                      "self-contained progress.html you can open in a browser")
    p_graph.add_argument("--out", default=None,
                         help="output HTML path (default: progress.html)")
    p_graph.add_argument("--open", action="store_true",
                         help="open the dashboard in a browser when done")
    args = parser.parse_args()
    print(f"btd6_bot build {BUILD}")
    global DEBUG, IGNORE_LOCKS
    DEBUG = not getattr(args, "quiet", False)
    IGNORE_LOCKS = getattr(args, "no_locks", False)
    if IGNORE_LOCKS:
        print("      (--no-locks: XP-lock detection OFF -- every upgrade "
              "treated as unlocked)")
    {"locate": cmd_locate, "watch": cmd_watch,
     "play": cmd_play, "scan": cmd_scan, "farm": cmd_farm,
     "solve": cmd_solve, "deploy": cmd_deploy, "campaign": cmd_campaign,
     "learn": cmd_learn, "graph": cmd_graph,
     "measure-ranges": cmd_measure_ranges}[args.command](args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except SystemExit:
        raise
    except Exception:
        _log_crash("main")
        raise
