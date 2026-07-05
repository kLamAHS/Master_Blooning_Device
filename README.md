# BTD6 bot — Stage 1

The "hands" of the ML project: a Python bot that watches the screen, reads the
round counter with OCR, and clicks towers into place from a gameplan file.
There is deliberately **no machine learning in here yet** — the milestone is a
script that beats Monkey Meadow on easy, unattended. Once that works, this same
script becomes your data collector for Stage 2 (it already logs every round to
`rounds_log.csv`).

> **Heads up:** automation is against Ninja Kiwi's terms of service and can get
> an account flagged or banned. Use this **offline, single player only**,
> ideally on a throwaway account. Never in races, co-op, or events.

## 1. Install

You need Python 3.10+.

```
pip install -r requirements.txt
```

Then install the Tesseract OCR engine (a separate program that reads text from
images — `pytesseract` is just the Python wrapper for it):

- **Windows:** installer from https://github.com/UB-Mannheim/tesseract/wiki
  (the default install path is auto-detected by the bot)
- **macOS:** `brew install tesseract`
- **Linux:** `sudo apt install tesseract-ocr`

macOS note: the first run will ask you to grant the terminal
Accessibility + Screen Recording permissions so it can click and screenshot.

## 2. Game settings

1. Run BTD6 **fullscreen on your primary monitor** (simplest). If you play
   windowed instead, put the window's `[left, top, width, height]` in pixels
   into `"region"` in `config.json` (use `locate` to measure the corners).
2. Settings → Gameplay: turn **Auto Start ON** so rounds flow without the bot
   needing to press anything between them.
3. Keep default hotkeys (or edit `TOWER_HOTKEYS` in `btd6_bot.py` to match
   yours).
4. Don't move or resize the window after calibrating — every coordinate
   depends on it.

## 3. Workflow

**Step 1 — find your coordinates.** With the game open on your map:

```
python btd6_bot.py locate
```

Hover over each spot where you want a tower and write down the
`norm=[x, y]` values. Normalized coordinates (fractions of the game area)
mean your plan keeps working if you later change resolution.

**Step 2 — edit the plan.** Open `plans/monkey_meadow_easy.json` and replace
the placeholder coordinates with yours. The strategy notes in the file explain
the intended build.

**Step 3 — check the round OCR.** Load into the map (Monkey Meadow → Easy →
Standard) but don't start the round, then run:

```
python btd6_bot.py watch
```

You want `parsed=1`. If it prints garbage, open `debug/round_crop.png` — if
the crop doesn't show exactly the round text, nudge the `"round_box"` values
in `config.json` (`[x, y, width, height]` as fractions of the game area) and
re-run until it locks on.

**Step 4 — let it play.**

```
python btd6_bot.py play plans/monkey_meadow_easy.json
```

Click onto the game window during the 5-second countdown, then hands off.

**Emergency stop:** slam the mouse into the **top-left corner** of the screen
(pyautogui's failsafe), or Ctrl+C in the terminal.

## Plan file format

Each entry in `"actions"` fires as soon as the round counter reaches its
`"round"`:

| action | fields | what it does |
|---|---|---|
| `place` | `tower`, `at` | presses the tower's hotkey, clicks at `at` |
| `upgrade` | `at`, `path` | clicks the tower, buys `[top, mid, bot]` upgrade tiers |
| `press` | `key` | presses any key (e.g. `"1"` for an ability) |

`"fast_forward": true` makes the bot toggle triple-speed at the start.

## Known limitations (a.k.a. why Stage 2 exists)

- **The bot is blind to cash.** If an action fires before you can afford it,
  the purchase silently fails (the bot right-clicks to clear the stuck tower
  ghost). Fix: schedule actions a round or two later. Reading the cash number
  with the same OCR trick is a great first extension.
- **No defeat detection.** If the defense leaks out, the bot just sits there.
  Detecting the defeat screen (its colors are very distinctive — compare
  screenshots) is exactly the label you need for Stage 2's
  "did this layout leak?" dataset.
- OCR occasionally misreads a frame; the bot requires the same round twice in
  a row before acting, so glitches can't trigger anything.

## What Stage 2 will add

Randomized tower layouts from a fixed menu of spots, hundreds of unattended
games, and a log of `(round, layout, leaked?)` — the training set for the
neural net. `rounds_log.csv` is already accumulating timing data every run.
