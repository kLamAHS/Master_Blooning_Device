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

## Usage guide — every command, in order

Each step's details live in the section noted. Setup steps (1–5) happen
once; after that your daily loop is just steps 6–8.

1. **Install** (once per machine, §1):

   ```
   pip install -r requirements.txt      # plus the Tesseract OCR engine
   ```

2. **Set the game up** (once, §2): Settings → Gameplay → **Auto Start ON**
   and **Disable Nudge Mode ON**; play windowed or fullscreen and don't
   move the window afterwards.

3. **Verify the round OCR** (once per machine, §3 step 3). Load into the
   map — Easy → Standard — but don't start round 1, then:

   ```
   python mk.py watch                   # want: parsed=1 on every read
   ```

4. **Scan the map** (once per map, "Emergent mode" section). Same clean,
   un-started map:

   ```
   python mk.py scan monkey_meadow                # land spots
   python mk.py scan monkey_meadow --tower sub    # optional water pass
   ```

5. **Calibrate the three restart buttons** (once, Stage 2 section) so
   `farm` can chain episodes unattended. Use:

   ```
   python mk.py locate                  # prints live mouse coordinates
   ```

   Lose a game on purpose and hover the defeat screen's RESTART, press
   Esc mid-game and hover the pause menu's RESTART, then hover the
   confirm dialog's OK — and put the three values in `config.json` as
   `defeat_restart`, `pause_restart`, `restart_confirm`.

6. **(Optional) prove the pipeline** with the hand-written control plan
   (§3):

   ```
   python mk.py play plans/monkey_meadow_easy.json
   ```

7. **Farm learning episodes** — the main loop (Stage 2 + Stage 3
   sections). Load the map fresh at round 1 and walk away:

   ```
   python mk.py farm monkey_meadow --episodes 15 --towers 4
   ```

   Meta-guided layouts are the default, and the equipped hero is placed
   as the early anchor — equip one (Sauda recommended) in the hero menu
   first. Useful flags: `--explore 0.5` (more randomness), `--no-meta`
   (pure random, the old Stage 2), `--no-evolve` (no genetic layer),
   `--no-hero` (skip hero placement), `--pool classic` (original 10
   towers only), `--final-round 40`, `--abort-lives 50`, `--seed N`.

8. **Review what it learned** (Stage 3 section, no game needed):

   ```
   python mk.py learn monkey_meadow
   ```

9. **When the research spreadsheet gets a new version**, regenerate the
   knowledge base (needs `pip install openpyxl`):

   ```
   python tools/extract_meta.py
   ```

10. **After touching the code**, run the offline sanity checks:

    ```
    python meta.py selftest
    ```

**Emergency stop, anytime:** slam the mouse into the top-left corner of
the screen, or Ctrl+C in the terminal.

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

1. **The bot finds the game window automatically on Windows** — windowed or
   fullscreen, any resolution (1920×1080 is perfect). It looks for a window
   whose title contains `BloonsTD6` and locks onto its client area (the game
   pixels, excluding the title bar). Every command prints what it detected,
   e.g. `Game area: auto-detected window 'BloonsTD6' -> (0, 0, 1920x1080)` —
   glance at that line to confirm it grabbed the right thing. If your window
   title differs (check the title bar or Task Manager), change
   `"window_title"` in `config.json`. On macOS, or if detection ever fails,
   play fullscreen or set `"region": [left, top, width, height]` in pixels
   as a manual override.
2. Settings → Gameplay: turn **Auto Start ON**, and turn **Disable Nudge
   Mode ON** — with nudge mode active, a failed placement leaves the ghost
   in a stuck confirm-state that ignores the cursor, which wrecks the
   bot's retry-at-the-next-spot behavior (worst for big towers like the
   super monkey).
3. Keep default hotkeys (or edit `TOWER_HOTKEYS` in `mk.py` to match
   yours).
4. Don't move or resize the window after calibrating — every coordinate
   depends on it.

## 3. Workflow

**Step 1 — plan coordinates are just hints.** With a mask scanned, `play`
snaps every placement to the nearest safe green point automatically, and
upgrades follow their tower — so the template's rough coordinates work
as-is. To move a tower, nudge its hint toward where you want it (eyeball
fractions from the preview image, or use `python mk.py locate` to
hover for exact values). `locate` is also handy for one-off points like
`"deselect_point"` in config.json.

**Step 2 — check the plan.** Open `plans/monkey_meadow_easy.json`. The
`"mask"` field points at your scanned mask, but it's optional: if it's
missing, the bot auto-discovers a `masks/*.json` matching the plan's
`"map"` name (or the only mask present). If no mask can be found at all,
`play` prints an unmissable warning, because raw un-snapped coordinates
are how towers end up wrestling with the path. Placement *retries* also
come from the mask — every retry is a spot the game itself approved
during the scan.

**Step 3 — check the round OCR.** Load into the map (Monkey Meadow → Easy →
Standard) but don't start the round, then run:

```
python mk.py watch
```

`watch` now **recalibrates every time it starts**: it finds the blue
settings-gear button (an unmistakable color blob) and derives the counter's
position from pure geometry — the number always sits at a fixed offset to
the gear's left — with an OCR pattern search as fallback. Every candidate
box is verified by an actual stable read (two consecutive matching parses)
before being saved to `config.json`, so a lucky garbage read can't fake a
pass. `play` runs the same check before starting, so a run can never launch
blind. You want `parsed=1`; manual `"round_box"` tweaking should never be
needed anymore.

**Step 4 — let it play.**

```
python mk.py play plans/monkey_meadow_easy.json
```

On Windows the bot brings the game window to the front by itself and starts
after a 3-second countdown — completely hands-off. (If auto-focus ever fails,
it falls back to a 5-second countdown and asks you to click the game.)

**Emergency stop:** slam the mouse into the **top-left corner** of the screen
(pyautogui's failsafe), or Ctrl+C in the terminal.

## Emergent mode: `scan` finds the spots itself

For the emergent pipeline, no human should be choosing tower spots — and the
machine doesn't need to "understand" the map to know where it can build,
because the game already knows. While the bot holds a tower ghost, BTD6
tints invalid locations red. `scan` exploits that: it takes a clean
reference screenshot, picks up a ghost (and verifies the game actually
received the key press — it aborts with a checklist if not), then sweeps
the cursor across a ~1,000-point grid. At each point it measures how much
*redder* a ring around the cursor got compared to the clean frame. The
ring sits outside the monkey's body — fur is brown, which is red-heavy and
fools naive color checks — and the before/after comparison cancels out red
map decorations too. Every legal placement gets written out.

```
python mk.py scan monkey_meadow                # land spots (dart ghost)
python mk.py scan monkey_meadow --tower sub    # add a water pass
```

Run it with the map loaded and round 1 not yet started (empty map, no bloons
to confuse the colors). It takes a couple of minutes and produces:

- `masks/monkey_meadow_dart.json` — the list of placeable `[x, y]` points.
  This is exactly what Stage 2's random-layout generator and the genetic
  algorithm sample from. One scan per map = full placement knowledge.
- `debug/scan_monkey_meadow_dart_preview.png` — your screenshot with green
  dots on placeable points, red on blocked ones. Eyeball it once; if it
  disagrees with the map, adjust `"scan_red_shift"` in `config.json`
  (invalid spots typically score 20–60, valid ground near 0; default
  threshold 12) and rescan — rescanning overwrites the old mask. That
  30-second glance replaces all the hovering.

The red-tint check is a heuristic, so a stray dot or two near map edges is
normal — the GA doesn't care, since bad spots just evolve away.

Where does that leave `locate`? It's only needed for hand-written baseline
plans (your control experiment that proves the clicking pipeline works) and
for one-off points like `deselect_point`. The emergent pipeline never uses it.

## Stage 2: `farm` — the bot starts learning

`farm` plays **random layouts** end-to-end, unattended: towers on random
mask points (large-footprint towers like the super monkey only use extra-
roomy points), random upgrade paths, bought greedily as cash allows so
timing emerges from the economy. Every episode appends one labeled row to
`runs_log.jsonl` — layout, final round reached, survived-or-died. That
file is the training set for the outcome model.

One-time calibration (three clicks worth of `locate`): lose a game on
purpose and hover the defeat screen's **RESTART** button, press Esc in a
game and hover the pause menu's **RESTART**, and hover the confirm
dialog's OK. Put them in `config.json` as `defeat_restart`,
`pause_restart`, `restart_confirm`. Then:

```
python mk.py farm monkey_meadow --episodes 15 --towers 4
```

Load the map fresh and walk away. Random layouts mostly *die* — that's
the point: the model needs both classes. Expect early rounds to survive
and leads/MOABs to filter the weak. After a few dozen episodes across an
evening or two, the dataset is ready for training, and learned prices
accumulate as a free side effect.

## Stage 3: the meta brain — priors from research, tactics from experience

A lot of BTD6 is already figured out. `research/btd6_meta_research_v55.xlsx`
holds that community meta distilled into rankings — tower scores, roles,
synergy shells, crosspath builds, round threats, mode priorities — and
`tools/extract_meta.py` compiles it into the machine-readable
`meta_knowledge.json` (rerun it whenever the spreadsheet gets a new
version; it needs `pip install openpyxl`, the bot itself doesn't).

The design rule in `meta.py`: **the meta is a prior, never a rulebook.**
Every choice — which tower, which spot, which upgrade path — is a
Thompson-sampling draw from a Beta posterior whose pseudo-counts *start*
at the spreadsheet's score and are updated by the bot's own episodes in
`runs_log.jsonl`. With no data the bot plays roughly what the research
says is good (carry + amplifier + control + opener, camo answered before
round 24, lead before 28, buys ordered by threat deadline). After ~6
episodes featuring a tower, its own results outweigh the spreadsheet — a
meta darling that keeps dying on *this* map gets sampled less, an
off-meta pick that keeps surviving gets sampled more.

Emergence is protected two ways on top of that:

- **The explore knob.** Every decision has an `--explore` chance
  (default 0.30) of ignoring the meta entirely and going uniform random,
  so no tower/path/spot ever starves and the dataset keeps both classes.
- **Evolution.** Once a few episodes have survived deep, layouts start
  being bred from the best ones found so far — mutated (move a tower,
  swap its species, push a build deeper, add/drop a tower) and crossed
  over between two elites. Parents are the bot's own discoveries, and
  mutations are free to wander off-meta. `--no-evolve` disables it.

```
python mk.py farm monkey_meadow --episodes 15 --towers 4          # meta on
python mk.py farm monkey_meadow --episodes 15 --explore 1.0      # pure random
python mk.py farm monkey_meadow --episodes 15 --no-meta          # old Stage 2
```

The meta pool also unlocks towers the random farm never used —
boomerang, mortar, spike, village, super, engineer (`--pool classic`
restricts to the original ten). Every logged episode now records a
`"strategy"` field (meta / evolve / crossover, roles, mutations), so the
dataset itself shows *how* each layout was conceived.

### Placement is geometry-aware

The scan mask contains more than "where can I build" — the track itself
is the big blob of interior cells that *refuse* placement. The brain
builds a **track model** from it: a 0→1 progress coordinate along the
path, and per-spot **coverage** (how much track a tower's range actually
touches — bends and long straights beat corner decorations, which is
why some spots hit more bloons for longer). Each tower then gets placed
by what it wants:

- **DPS carries** claim the highest-coverage real estate first.
- **Alchemist / Village** sit inside buff radius of teammates, the
  carry above all — a brew that reaches nobody buffs nobody.
- **Glue / Ice** cover the stretch *just upstream* of the carry's kill
  zone: glue applied too early wears off before the DPS sees the
  bloons, and glue applied downstream of it does nothing.
- **Spike factories** favor late track, where leaks go to die.
- **Snipers / mortars** (global range) stay *off* prime spots that
  range-limited towers need.

Towers keep real spacing: planned spots stay a footprint apart (more
for supers/villages), placement retries never target a spot another
tower already holds, and duplicate tower types get sharply diminishing
sampling weight — a second glue is occasionally right, a third 000 glue
never is. Coverage towers also prefer *early* track when the flow is
known: damage near the entry leaves room for error, a defense camped at
the exit pops with zero margin. Every episode prints its reasoning per
tower —

```
glue#0(control) @ [0.70,0.55]  covers path 0.31-0.43 (4% of track)
boomerang#1(carry) @ [0.58,0.60]  covers path 0.21-0.64 (12% of track)
```

— so a misplaced tower is visible at a glance instead of a mystery.

One thing pixels can't tell: which end of the track is the entry. So
`farm` senses it — on its first episode the map is empty, and the first
thing that moves near the track *is* the bloons entering. The result is
saved into the mask file (`"flow_entry"`), so it's sensed once per map.
Until it's known, placement runs direction-agnostic (debuffers co-locate
with the carry instead of aiming upstream). The learned per-region
posteriors still multiply every score, and the `--explore` fraction of
placements stays fully random, so position learning and emergence both
survive the geometry.

### Buying is scheduled, reserved, and leak-reactive

The old farm bought greedily — four towers as fast as possible, then
whatever upgrade happened to be affordable. Meta thesis #5 says the
opposite: *money efficiency and save-up windows matter more than
theoretical DPS.* So every buy now carries a **round** (when it should
happen), a **priority**, and a **cost estimate**:

- **Paced by income, upgrade-first.** Buys are scheduled along a rough
  income curve so the plan never wants more money than the game can
  have produced — and in the order a good player buys: the hero
  anchors (with a hero placed there is **no separate opener** — that
  would just split cash away from the carry), the carry base follows,
  then the **carry's first tiers come before any more bases**. A
  $2,500 super is *planned* for ~round 15 instead of being dribbled
  away on trinkets. Threat answers keep hard dates (camo before 24,
  lead before 28) that cap both the upgrade *and* its tower's
  placement, whatever the curve says. When cash runs ahead of the
  model, the next scheduled buy unlocks early — estimates pace,
  reality decides.
- **Support is conditional, not on a timer.** Amplifier/control/extra
  bases are *gated on the carry being stable* (main path at tier 3):
  glue and buffs arrive when the core can use them, not because a
  clock ticked. Gates yield to threat dates (camo coverage never waits
  for a struggling carry), to a leak emergency, and to running 6+
  rounds late — support arrives when needed either way. Every tower
  gets an identity label (`boomerang#0(carry)`), so logs and the
  dataset say exactly which tower each upgrade landed on.
- **Reservation.** The most important due purchase reserves its price.
  Lower-priority buys (crosspaths, luxuries) only spend the *surplus*
  above the reservation — being efficient now is what makes the big
  thing affordable later.
- **Leak emergency.** If a round costs 8+ lives, reserves come off for
  45 seconds and the bot buys any affordable defense immediately, like
  a player dumping savings when the defense cracks.

The executor still verifies everything against real cash — estimates
pace the plan, reality decides the purchase.

### The hero plays too

Each episode opens by placing your equipped hero (hotkey `u`) as the
early anchor — free scaling value the meta guide rates highly, with
**Sauda** the recommended low-micro pick (equip her in the hero menu
before farming; the bot can't choose heroes, only place them). Heroes
level on their own, so no upgrade buys are ever attempted on one, and
placement uses a short-range coverage profile that suits Sauda's melee
reach. If no hero is equipped, the `u` press produces no ghost — the
bot notices ("affordable but no ghost ever appears"), drops the hero
from the plan after a few tries, and plays on. `--no-hero` skips hero
placement entirely.

To see what the bot currently believes — where its experience confirms
or contradicts the research, which elite layouts evolution is breeding
from, and which round bucket kills it (annotated with the nearest known
threat, e.g. "deaths cluster near r24 — camo"):

```
python mk.py learn monkey_meadow            # or: python meta.py report monkey_meadow
python meta.py selftest                     # offline sanity checks, no game needed
```

## Plan file format

Each entry in `"actions"` fires as soon as the round counter reaches its
`"round"`:

| action | fields | what it does |
|---|---|---|
| `place` | `tower`, `at` | self-verifying placement: waits until affordable, confirms the tower landed, auto-nudges nearby if the exact spot is invalid |
| `upgrade` | `at`, `path` | clicks the tower, buys `[top, mid, bot]` upgrade tiers |
| `press` | `key` | presses any key (e.g. `"1"` for an ability) |

Optional fields on any action: `"wait_cash": 2000` blocks until cash reaches
that amount first (useful before expensive upgrades); `"timeout": 90` changes
how long a `place` keeps retrying (default 60 s).

`"fast_forward": true` makes the bot toggle triple-speed at the start.

## Troubleshooting

**Keys you press work, keys the bot presses don't.** Unity games (BTD6
included) read keyboard input at the scan-code level and ignore the
virtual-key events `pyautogui` sends — the failure is completely silent.
That's why all input now goes through `pydirectinput`, which sends real
scan codes. It installs automatically from `requirements.txt` on Windows;
if you set up before it was added, run `pip install pydirectinput`. The
bot prints a loud warning at startup if it's missing.

**Hotkeys: BTD6 only.** The wiki's BTD4/BTD5 hotkey tables do **not**
apply to BTD6 (in BTD5, W is Tack Shooter; in BTD6, W is Boomerang). The
`TOWER_HOTKEYS` dict already matches BTD6 defaults — verify against your
own Settings → Hotkeys screen, not old wiki pages.

**Scan shows green dots on the track / everywhere.** That means no ghost
was held during the sweep — with nothing on the cursor there's no red tint
anywhere, so everything reads "placeable." The scan now checks for the
ghost right after pressing the hotkey and aborts with a checklist instead
of producing a garbage mask.

**Scan shows red dots everywhere (even open grass).** The old detector
counted red-ish pixels near the cursor, and monkey fur is brown — which is
red-heavy — so merely holding a monkey looked "invalid." The current
detector measures the red *shift* on a ring around the cursor versus the
clean frame, which is immune to fur, red flowers, and crates.

**Scan is speckled with random red/orange on open grass.** Seasonal map
skins (holiday events) add falling confetti and firework flashes — moving
red things that pollute frame comparisons. The detector judges each point
by the *median* per-pixel red-shift and resamples borderline readings, so
transients can't flip a verdict; if a skin still causes trouble, check the
game's settings for an option to disable seasonal decorations and rescan.

## Bookkeeping the bot does for you

- **`prices.json` — a self-learned price book.** Every purchase records
  what it actually cost (cash before minus after), keyed by
  difficulty/tower/path/tier. Nothing is hardcoded, so game rebalances
  can't make it wrong. Once a price is learned, upgrades wait for the
  exact amount before pressing, and `play` prints your plan's total
  estimated cost at startup with a count of not-yet-learned purchases.
  **Poisoned prices heal themselves:** any price that wasn't verified in
  the current session (loaded from disk, or read off a red row) gets
  re-checked visually after ~45 s of blocking a buy — one menu open, and
  a green-row sighting overwrites the bad value (a `$210` recorded as
  `$2105` no longer gates the upgrade forever).
- **Money-failure watermarks are misread-proof.** When a buy fails on
  cash, the bot notes the cash level and holds spending until income
  clears it — but the noted level is capped at the item's known price
  (being broke for a `$110` upgrade *means* cash < $110, whatever the
  counter OCR claims), and every watermark expires after 40 s. A junk
  read like `$6005` when the real cash is `$600` can stall buys for
  seconds, not rounds. A press that "didn't take" on a green (affordable)
  row also re-reads the row itself before judging: if the row moved on to
  the next tier, the purchase actually landed and the cash reads were
  noise.
- **`runs_log.jsonl` — one line per run**: final round reached, outcome
  (`victory` / `defeat` / `interrupted`), and the full tower layout
  with each tower's real position and upgrade tiers. Farm episodes also
  record `lives_by_round` — lives at the start of every round, so the
  model can learn *which round* a layout leaks on, not just whether it
  died. Defeat is detected by the lives counter (auto-located from the
  red heart icon) hitting 0, because the round counter stays visible on
  the defeat screen and can't be the signal.
- **Stuck-panel recovery.** Upgrades verify the tower got selected and the
  panel closed afterwards, deselect clicks go to mask points far from
  every tower, and if the HUD ever stays dark mid-run the main loop
  actively clears UI at ~20 s and ~40 s before giving up at ~65 s.

- **`locked.json` — upgrades the bot must not buy.** The bot now reads the
  upgrade panel *visually* before pressing anything: a green button with a
  `$` price is buyable (and the price is recorded — no purchase needed), a
  button showing `XP` is an unlock you haven't bought — recorded here and
  never pressed, so **your XP is never spent** — and a padlocked
  `PATH CLOSED` row is remembered per tower. Text the OCR can't confidently
  read is treated as unpressable, erring on the side of your XP. You can
  also **pre-seed this file by hand** to reserve XP decisions for yourself:
  add `"easy:dartling:0:1": true` or any `difficulty:tower:path:tier` key.
  Because every panel-open harvests all three visible prices, the price
  book fills fast and menus soon open only to actually buy.

## Known limitations (a.k.a. why Stage 2 exists)

- **Placement is now self-verifying.** The round counter is hidden whenever
  a ghost is held, which the bot uses as a sensor: hotkey pressed but the
  counter is still visible → no ghost → can't afford yet, so it waits;
  clicked but the counter stays hidden → ghost stuck → invalid spot, so it
  cancels and retries a small spiral of nearby offsets. Cash is also read
  directly (auto-located from the gold coin icon) for `wait_cash` gating
  and richer logs. **Pick plan coordinates from the GREEN dots** in the
  scan preview — orange dots are valid but hug an edge, where the tower's
  footprint may overhang the track.
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
