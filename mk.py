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

BUILD = "2026-07-06.r10"  # printed at startup: ties every log to a build

DEBUG = True     # verbose decision logging; pass --quiet to farm/play


def dbg(msg):
    if DEBUG:
        print(f"      . {msg}")

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


def parse_round(text):
    """'13/40' -> 13. Returns None when the text doesn't look like a round."""
    text = text.strip()
    if "/" in text:
        text = text.split("/", 1)[0]
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits and 1 <= int(digits) <= 200:
        return int(digits)
    return None


def read_round(screen, cfg):
    """Returns (parsed_round_or_None, raw_text, crop_img, processed_img)."""
    crop = screen.grab(cfg["round_box"])
    processed = preprocess_round_crop(crop)
    text = pytesseract.image_to_string(
        processed,
        config="--psm 7 -c tessedit_char_whitelist=0123456789/").strip()
    return parse_round(text), text, crop, processed


def plausible(new, last):
    """Rounds only move forward, and only a little at a time."""
    if new is None:
        return False
    if last is None:
        return 1 <= new <= 100
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

PRICES_PATH = Path(__file__).parent / "prices.json"
PRICES = json.loads(PRICES_PATH.read_text()) if PRICES_PATH.exists() else {}
PRICE_DIFFICULTY = "medium"      # set from the plan's mode when playing
PRICES_SRC = {}                  # key -> "buy" | "seen" | "short" (session)

# Upgrades you haven't unlocked with XP. The bot must NOT press these --
# doing so spends your limited XP. Auto-detected (an affordable press that
# moves neither cash nor tier is locked) and remembered across runs. You
# can also pre-seed this file by hand, e.g. {"easy:dartling:0:1": true} or
# broader "don't go past tier 2 on ninja" style entries you add yourself.
LOCKED_PATH = Path(__file__).parent / "locked.json"
LOCKED = json.loads(LOCKED_PATH.read_text()) if LOCKED_PATH.exists() else {}


def is_locked(ttype, path_i, tier):
    return bool(LOCKED.get(price_key(ttype, path_i, tier)))


def mark_locked(ttype, path_i, tier):
    key = price_key(ttype, path_i, tier)
    if not LOCKED.get(key):
        LOCKED[key] = True
        LOCKED_PATH.write_text(json.dumps(LOCKED, indent=1, sort_keys=True))
        print(f"      (locked: {key} not unlocked -- won't try it again)")


def price_key(*parts):
    return ":".join([PRICE_DIFFICULTY, *map(str, parts)])


def record_price(key, cost, src="buy"):
    """Persist a learned price -- with poison filters: every real BTD6
    price is a multiple of 5, and deltas computed from corrupted cash
    reads almost never are. Implausibly huge costs are rejected too.
    src tags provenance: 'buy' (cash-verified) / 'seen' (green panel) /
    'short' (red panel -- unverified, eligible for a re-check)."""
    if cost is None or cost <= 0 or cost > 120000:
        return
    if cost % 5 != 0:              # every real BTD6 price ends in 0 or 5
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


def read_round_stable(screen, cfg, tries=4):
    """Return a round value only if two consecutive reads agree -- a
    misaligned crop produces flickery garbage that never repeats reliably,
    so this filters out the lucky-junk reads that fooled earlier versions."""
    prev = None
    for _ in range(tries):
        value, *_ = read_round(screen, cfg)
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
    refused both when produced and when loaded."""
    if not box:
        return False
    x, y, w, h = box
    return (0.015 <= h <= 0.10          # digit strip, not a map chunk
            and w <= 0.35               # never wider than the HUD zone
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
    wide = [max(min_x, x - h), max(0.0, y - 0.3 * h), w + 2 * h, h * 1.6]
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
    nx0 = wide[0] + max(0.0, dx1 - 0.25 * bh) / screen.w
    ny0 = wide[1] + max(0.0, dy1 - 0.35 * bh) / screen.h
    snapped = [round(nx0, 4), round(ny0, 4),
               round((dx2 - dx1 + 2.2 * bh) / screen.w, 4),  # right room
               round(1.7 * bh / screen.h, 4)]
    return snapped if _plausible_cash_shape(snapped) else None


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
    None. Tesseract remains only as a fallback, with '$'-anchored
    parsing. Implausibly huge readings return None."""
    box = cfg.get("cash_box")
    if not box:
        return None
    crop = screen.grab(box)
    value = _read_number_image(crop)
    if value is None and pytesseract is not None:
        processed = preprocess_round_crop(crop)
        text = pytesseract.image_to_string(
            processed,
            config="--psm 7 -c tessedit_char_whitelist=0123456789$,"
        ).strip()
        if "$" in text:
            text = text.rsplit("$", 1)[1]
        digits = re.sub(r"\D", "", text)
        value = int(digits) if digits else None
    if value is None:
        return None
    return value if value <= 150000 else None


def read_cash_confirmed(screen, cfg):
    """Two reads that must corroborate: equal -> that value; one a clean
    AFFIX of the other -> the shorter. A junk glyph can attach at either
    end of the digit run ('851' inside '6851' = prepended, '600' inside
    '6005' = appended), and both signatures poison watermarks the same
    way, so both resolve to the shorter read. Else None. Used where a
    single bad read does lasting damage (watermarks, price recording)."""
    a = read_cash(screen, cfg)
    time.sleep(0.15)
    b = read_cash(screen, cfg)
    if a is None or b is None:
        return a if a == b else None
    if a == b:
        return a
    sa, sb = str(a), str(b)
    if sa.endswith(sb) or sa.startswith(sb):
        return b
    if sb.endswith(sa) or sb.startswith(sa):
        return a
    return None


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
        # the digit snap; adopt it when it reads and differs.
        lb = cfg.get("lives_box")
        fence = (lb[0] + lb[2]) if lb else 0.0
        snapped = _snap_cash_box_to_digits(screen, original, min_x=fence)
        if snapped and snapped != original:
            cfg["cash_box"] = snapped
            if read_cash(screen, cfg) is not None:
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
            snapped = _snap_cash_box_to_digits(screen, box, min_x=fence)
            if snapped:
                cfg["cash_box"] = snapped
                if read_cash(screen, cfg) is not None:
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


# ---------------------------------------------------------------------------
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


def detect_panel_side(screen):
    """Which side is an upgrade panel open on, if any? Portrait COLOR is
    useless: backgrounds are category-themed and Military's is GREEN --
    on a grass map, a sniper's panel was literally camouflaged from the
    old detector. Instead, triangulate the panel's WOOD: brown must show
    at 2 of 3 fixed heights (title band, under-portrait gap, sell bar).
    Grass/path/bloons can't fake brown; the HUD's gold coin covers far
    too little of one strip to matter."""
    x_of = {"left": PANEL_GEOM["title_band"]["left"][:1]
            + [PANEL_GEOM["title_band"]["left"][2]],
            "right": PANEL_GEOM["title_band"]["right"][:1]
            + [PANEL_GEOM["title_band"]["right"][2]]}
    for side in ("left", "right"):
        sx, sw = x_of[side]
        hits = 0
        for sy, sh in PANEL_GEOM["brown_strips"]:
            if _brown_fraction(screen.grab([sx, sy, sw, sh])) >= 0.35:
                hits += 1
        if hits >= 2:
            return side
    return None



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
            strip = band[y + int(0.52 * h):y + int(0.97 * h), x:x + w]
            price = _read_number_image(strip)
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
    crop = screen.grab([bx, ry + ty * rh, bw, th * rh])

    if green >= 0.10:
        upper = button[:max(1, int(button.shape[0] * 0.55))]
        if _white_fraction(upper) > 0.22:
            return "xp", None            # the big white unlock arrow
        price = _read_number_image(crop)
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


def act_place(screen, cfg, action):
    """Try to place a tower. Returns (status, landed):
      ("placed", [x, y])  -- tower is down at that coordinate
      ("broke", None)     -- no ghost appeared: can't afford it YET. One
                             press, one check, immediate return -- the
                             caller waits for income; this never counts
                             as a placement failure.
      ("no_spot", None)   -- ghost held but every candidate click was
                             rejected: genuinely unplaceable around here."""
    key = TOWER_HOTKEYS[action["tower"].lower()]
    spot = action["at"]
    candidates = placement_candidates(spot, action["tower"].lower())
    # Never retry ON another tower this run already placed: those clicks
    # can only fail (or select the neighbor), and watching the bot try
    # to stack glue on glue for a minute is silly.
    avoid = action.get("avoid") or []
    if avoid:
        min_d = 0.05 if action["tower"].lower() in LARGE_TOWERS else 0.03
        far = [c for c in candidates
               if all((c[0] - a[0]) ** 2 + (c[1] - a[1]) ** 2
                      >= min_d ** 2 for a in avoid)]
        candidates = far or candidates
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
    side = None
    for _ in range(3):
        click_norm(screen, action["at"])
        time.sleep(cfg["action_delay"])
        side = detect_panel_side(screen)
        if side:
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
                    status = "locked"          # already recorded by reader
                    break
                if state == "short":
                    # Greyed button + red price: a real upgrade we can't
                    # afford yet. Price is booked; the caller saves up.
                    status = "broke"
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
            known = PRICES.get(pkey) if pkey else None
            cash = read_cash(screen, cfg)
            if known is not None and cash is not None and cash < known:
                status = "broke"               # caller waits, menu closed
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
                    status = "broke"
                    break
                # No visual info and an affordable press moved nothing:
                # the safety net -- treat as locked and stop pressing.
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
                abilities=False):
    """Play one layout to survival or defeat. Returns (outcome,
    final_round_reached, towers, lives_by_round, cash_by_round,
    spent_by_round).
    flow_sensor, when given, is called once with the pre-start clean
    frame right after the round starts -- its window to watch where the
    first bloons appear (nothing has been bought yet, so leaking a few
    seconds of round 1 costs nothing).
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
        from meta import choose_buy
    except ImportError:                       # no brain: old greedy rule
        def choose_buy(q, _round, _cash, _cost, now, emergency=False,
                       gate_ok=None):
            return next((j for j, it in enumerate(q)
                         if it.get("_wake", 0) <= now), None)

    clean = screen.grab() if flow_sensor else None
    press_key("space")
    time.sleep(0.3)
    press_key("space")                        # fast-forward
    if flow_sensor:
        flow_sensor(clean)
    queue = list(genome)
    landed_by_ref, towers = {}, {}
    lives_by_round = {}
    cash_by_round = {}       # cash at the start of each round (telemetry
    spent_by_round = {}      # for the learned income curve)
    broke_at = [None, 0.0]   # cash watermark: level + time of last set
    emergency_until = [0.0]  # leak emergency: reserves off until this time
    prev_round_lives = [None]
    last_ping = [0.0]        # ability-hotkey pinger cooldown
    final_seen = None        # play_out: when the final round was reached

    def record_spend(amount):
        if amount:
            key = str(last_round if last_round is not None else 0)
            spent_by_round[key] = spent_by_round.get(key, 0) + amount

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
    attempts = 0
    last_round = prev_read = None
    lives = prev_lives = None
    zero_streak = 0
    dumped_defeat_check = False
    misreads = 0
    outcome = "hud_lost"
    while True:
        unpause_if_needed(screen, cfg)        # cheap; a paused game shows
        value, *_ = read_round(screen, cfg)   # a frozen-but-visible counter
        prev_lives = lives
        lives = read_lives(screen, cfg)
        accepted = None
        if value is not None:
            if value == last_round:
                accepted = value
            elif plausible(value, last_round) and value == prev_read:
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
            if accepted != last_round:
                last_round = accepted
                if lives is not None:
                    lives_by_round[str(last_round)] = lives
                cash_now = read_cash(screen, cfg)
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
        if (lives == 0 and prev_lives == 0) or zero_streak >= 3:
            outcome = "defeat"                # defeat screen (HUD stays!)
            break
        if abort_lives and lives is not None and 0 < lives < abort_lives:
            # A lost cause: abort while the game is LIVE, so the restart
            # takes the pause route -- the one with a proven field record.
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
        if not buyable_now and time.time() - round_seen_at > 60 \
                and (lives is None or lives < 40):
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

            cash_now = read_cash(screen, cfg)
            rush = time.time() < emergency_until[0]
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
                pass                           # all pending buys cooling down
            elif item["do"] == "place":
                ttype = item["tower"].lower()
                base = PRICES.get(price_key(ttype))
                cash = read_cash(screen, cfg)
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
                         "avoid": [t["at"] for t in towers.values()]})
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
                    elif (known := PRICES.get(price_key(ttype, pi, tier))) \
                            and (cash := read_cash(screen, cfg)) is not None \
                            and cash < known:
                        # Known price, can't afford: no menu, just sleep
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
                            st = act_upgrade(screen, cfg,
                                             {**item, "at": landed}, entry)
                            dbg(f"{tname} path{pi + 1} t{tier}: {st}")
                            if st == "bought":
                                queue.pop(idx)
                                broke_at[0] = None
                                record_spend(
                                    PRICES.get(price_key(ttype, pi, tier))
                                    or item.get("est") or 0)
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
                            and (cash := read_cash(screen, cfg)) is not None \
                            and watermark_holds(cash):
                        msg = (f"{tname} path{pi + 1}: watermark hold "
                               f"(${cash} <= ${broke_at[0]}+100)")
                        if item.get("_dbg") != msg:
                            item["_dbg"] = msg
                            dbg(msg)
                        item["_wake"] = time.time() + 5
                    else:
                        st = act_upgrade(screen, cfg,
                                         {**item, "at": landed}, entry)
                        dbg(f"{tname} path{pi + 1} t{tier}: {st}")
                        if st == "bought":
                            queue.pop(idx)
                            broke_at[0] = None
                            attempts = 0
                            record_spend(
                                PRICES.get(price_key(ttype, pi, tier))
                                or item.get("est") or 0)
                        elif st in ("locked", "closed"):
                            queue.pop(idx)     # recorded; done with it
                        elif st == "broke":
                            set_watermark(
                                PRICES.get(price_key(ttype, pi, tier)))
                            item["_fails"] = item.get("_fails", 0) + 1
                            item["_wake"] = time.time() + min(
                                10 * 2 ** (item["_fails"] - 1), 60)
                            if item["_fails"] >= 6:
                                queue.pop(idx)
                        else:                  # no_select / unread
                            item["_fails"] = item.get("_fails", 0) + 1
                            item["_wake"] = time.time() + 10
                            if item["_fails"] >= 4:
                                print(f"   giving up on an upgrade ({st})")
                                queue.pop(idx)

        endgame_watch = (play_out and last_round is not None
                         and last_round >= final_round)
        if misreads == 12 and not endgame_watch:  # stuck panel? clear it
            if looks_defeated(screen):
                dbg("DEFEAT screen recognized (via dark counter)")
                outcome = "defeat"
                break
            dbg("counter dark for a while -- clearing UI")
            clear_ui(screen, cfg)
        if last_round is not None and last_round >= final_round:
            if not play_out:
                outcome = "survived"
                break
            # Play THROUGH the final round: the victory screen covers
            # the round counter (misreads climb while no defeat screen
            # shows), or the counter just sits on the final number for
            # minutes after the last buy. Both mean the game is WON --
            # a leak on the final round still hits the defeat/lives
            # checks above first. No UI-clearing in this state: poking
            # at a victory screen navigates menus blindly.
            final_seen = final_seen or time.time()
            if misreads >= 8:
                dbg("HUD covered after the final round with no defeat "
                    "screen -- that's the victory screen")
                outcome = "survived"
                break
            if time.time() - final_seen > 240:
                dbg("final round stable for 4 minutes with lives up -- "
                    "counting it as survived")
                outcome = "survived"
                break
        else:
            final_seen = None
        if misreads >= 25:                    # HUD truly gone: unknown state
            break
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
            how = "defeat screen"
            click_norm(screen, cfg["defeat_restart"])
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
                click_norm(screen, cfg["defeat_restart"])
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


def detect_loaded_rung(screen, cfg, flag_difficulty=None,
                       flag_mode=None):
    """What game is actually loaded? Starting lives pin the difficulty
    (200/150/100 = easy/medium/hard, 1 = a one-life mode) and the
    starting round separates CHIMPS (starts at round 6) from impoppable
    (round 3). Returns (lives, start_round, difficulty, game_mode) --
    explicit flags always win over detection."""
    import campaign
    lv = read_lives(screen, cfg)
    r1 = read_round(screen, cfg)[0]
    time.sleep(0.3)
    r2 = read_round(screen, cfg)[0]
    start_round = r1 if (r1 is not None and r1 == r2 and r1 <= 6) else 1
    rung = campaign.detect_rung(lv, start_round)
    difficulty = flag_difficulty or (rung[0] if rung else None)
    game_mode = flag_mode or (rung[1] if rung else "standard")
    if rung and flag_difficulty and rung[0] != flag_difficulty:
        print(f"NOTE: HUD ({lv} lives, start r{start_round}) suggests "
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

    if lv is not None and args.abort_lives and lv <= args.abort_lives:
        # A 1-life mode with --abort-lives 50 would abort every episode
        # on sight ("0 < 1 < 50"). Any leak already ends those games.
        print(f"abort-lives {args.abort_lives} >= starting lives {lv} "
              f"-- disabling the early-abort (defeat itself is the "
              f"signal here).")
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
                danger_rounds=danger, abilities=not args.no_abilities)
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

    global PRICE_DIFFICULTY
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
    one_life = lv is not None and lv <= 3
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
                abilities=not args.no_abilities)
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
        progress.record_episode(
            args.name, difficulty, game_mode, outcome, reached,
            was_attempt=decision["kind"] == "attempt")
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
                        need = PRICES.get(price_key(
                            entry["tower"].lower(), pi, entry["path"][pi] + 1))
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
    p_solve.add_argument("-q", "--quiet", action="store_true",
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
    args = parser.parse_args()
    print(f"btd6_bot build {BUILD}")
    global DEBUG
    DEBUG = not getattr(args, "quiet", False)
    {"locate": cmd_locate, "watch": cmd_watch,
     "play": cmd_play, "scan": cmd_scan, "farm": cmd_farm,
     "solve": cmd_solve, "campaign": cmd_campaign,
     "learn": cmd_learn}[args.command](args)


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
