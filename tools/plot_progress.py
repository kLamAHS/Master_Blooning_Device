"""Turn the bot's training log into a progress dashboard you can look at.

Reads runs_log.jsonl (one row per episode, written by farm/solve/deploy) and
progress.json (the campaign ladder) and writes a single self-contained HTML
file -- open it in any browser, no server, no dependencies. Pure stdlib, so it
runs without cv2/numpy (unlike the rest of the bot), and every number it draws
is computed by plain functions that are unit-tested in tools/test_plot_progress.py.

    python tools/plot_progress.py                 # -> progress.html
    python tools/plot_progress.py --open          # ...and open it
    python mk.py graph                            # same thing, from the CLI

Charts (per map + mode, switchable):
  - a hero number + KPI tiles (episodes, wins, best round, recent form)
  - PROGRESS OVER EPISODES: each run's deepest round, the running personal
    best, wins marked, the target line -- the "am I learning?" picture
  - WHERE RUNS END: a histogram of the round each run reached (the wall)
  - OUTCOMES: how episodes ended (victory / defeat / lost track / crashed)
  - BEST RUN ECONOMY: lives and cash by round for the deepest run so far
  - CAMPAIGN LADDER: best round vs target per rung, from progress.json
"""

import argparse
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Optional, guarded reuse of the pure-stdlib brain modules for nicer context.
try:
    sys.path.insert(0, str(REPO))
    from meta import earned_by            # income prior for the economy chart
except Exception:                          # pragma: no cover - defensive
    earned_by = None
try:
    from campaign import LADDER, RUNG_INFO, rung_key
except Exception:                          # pragma: no cover - defensive
    LADDER = [("easy", "standard"), ("medium", "standard"),
              ("hard", "standard"), ("hard", "chimps")]
    RUNG_INFO = {("easy", "standard"): {"target": 40},
                 ("medium", "standard"): {"target": 60},
                 ("hard", "standard"): {"target": 80},
                 ("hard", "chimps"): {"target": 100}}

    def rung_key(m, d, mode):
        return f"{m}|{d}|{mode}"


WIN_OUTCOMES = ("victory", "survived")
# outcome -> (display label, severity token used for the status color)
OUTCOME_META = OrderedDict([
    ("victory", ("victory", "good")),
    ("survived", ("survived", "good")),
    ("defeat", ("defeat", "critical")),
    ("hud_lost", ("lost track", "serious")),
    ("counter_lost", ("lost track", "serious")),
    ("crashed", ("crashed", "warning")),
    ("interrupted", ("stopped", "muted")),
    ("stopped", ("stopped", "muted")),
])


def is_win(outcome):
    return outcome in WIN_OUTCOMES


def load_runs(path):
    """Parse runs_log.jsonl leniently -- a half-written final line (the bot was
    killed mid-write) or a stray blank line must never lose the whole history."""
    rows = []
    try:
        text = Path(path).read_text()
    except (OSError, ValueError):
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def load_progress(path):
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    rungs = data.get("rungs")
    return rungs if isinstance(rungs, dict) else {}


def group_of(row):
    """(map, mode, label) for a row. cmd_play rows carry a 'plan' not a 'map'."""
    m = row.get("map")
    mode = row.get("game_mode")
    if not m and row.get("plan"):
        m = "plan:" + Path(str(row["plan"])).stem
        mode = mode or "plan"
    m = m or "unknown"
    mode = mode or "standard"
    return m, mode, f"{m} · {mode}"


def _int_or_none(v):
    return v if isinstance(v, int) else None


def _round_series(d):
    """A {str(round): value} dict -> a sorted [{'r': int, 'v': number}] list."""
    out = []
    if isinstance(d, dict):
        for k, v in d.items():
            try:
                out.append({"r": int(k), "v": v})
            except (TypeError, ValueError):
                continue
    out.sort(key=lambda p: p["r"])
    return out


def summarize_group(rows):
    """All the numbers the dashboard draws for one (map, mode) slice. Rows are
    in file order == chronological, which is the episode axis."""
    series, running, deaths = [], [], Counter()
    outcomes = Counter()
    best_round = 0
    best_run_row = None
    run_max = 0
    for i, row in enumerate(rows, 1):
        outcome = row.get("outcome") or "unknown"
        fr = _int_or_none(row.get("final_round"))
        win = is_win(outcome)
        outcomes[outcome] += 1
        kind = ((row.get("strategy") or {}).get("kind")
                if isinstance(row.get("strategy"), dict) else None)
        series.append({"i": i, "round": fr, "outcome": outcome, "win": win,
                       "kind": kind or "—", "time": row.get("time") or ""})
        if fr is not None:
            run_max = max(run_max, fr)
            deaths[5 * (fr // 5)] += 1
            best_round = max(best_round, fr)
            # deepest run wins; ties resolve to the later (>=) run
            cur = (_int_or_none(best_run_row.get("final_round"))
                   if best_run_row else None)
            if cur is None or fr >= cur:
                best_run_row = row
        running.append(run_max)

    targets = [row.get("target_round") for row in rows
               if isinstance(row.get("target_round"), int)]
    target = max(targets) if targets else None

    death_hist = [{"bucket": b, "count": deaths[b]} for b in sorted(deaths)]
    outcome_rows = []
    for oc, n in outcomes.most_common():
        label, sev = OUTCOME_META.get(oc, (oc, "muted"))
        outcome_rows.append({"outcome": oc, "count": n, "label": label,
                             "sev": sev})

    wins = sum(1 for s in series if s["win"])
    n = len(series)
    # recent form: last 10 vs the first 10, so a delta means something
    tail = series[-10:]
    head = series[:10]
    recent_wins = sum(1 for s in tail if s["win"])
    recent_best = max((s["round"] for s in tail if s["round"] is not None),
                      default=0)
    head_best = max((s["round"] for s in head if s["round"] is not None),
                    default=0)

    best_run = None
    if best_run_row is not None:
        mode = group_of(best_run_row)[1]
        cash = _round_series(best_run_row.get("cash_by_round"))
        prior = []
        if earned_by is not None and cash:
            try:
                prior = [{"r": p["r"], "v": round(earned_by(p["r"], mode))}
                         for p in cash]
            except Exception:
                prior = []
        best_run = {
            "final": _int_or_none(best_run_row.get("final_round")),
            "lives": _round_series(best_run_row.get("lives_by_round")),
            "cash": cash,
            "prior": prior,
        }

    return {
        "episodes": n,
        "wins": wins,
        "winRate": (wins / n) if n else 0.0,
        "bestRound": best_round,
        "target": target,
        "recentWins": recent_wins,
        "recentN": len(tail),
        "recentWinRate": (recent_wins / len(tail)) if tail else 0.0,
        "recentBest": recent_best,
        "headBest": head_best,
        "series": series,
        "runningMax": running,
        "deathHist": death_hist,
        "outcomes": outcome_rows,
        "bestRun": best_run,
    }


def build_ladder(progress):
    """progress.json rungs -> per-map ladders in canonical order."""
    maps = OrderedDict()
    for key in progress:
        parts = key.split("|")
        if len(parts) == 3:
            maps.setdefault(parts[0], True)
    ladders = []
    for m in maps:
        rungs = []
        for diff, mode in LADDER:
            rec = progress.get(rung_key(m, diff, mode))
            if not isinstance(rec, dict):
                continue
            target = (RUNG_INFO.get((diff, mode)) or {}).get("target") or 0
            rungs.append({
                "name": mode if mode == "chimps" else diff,
                "diff": diff, "mode": mode,
                "best": int(rec.get("best_round") or 0),
                "target": target,
                "episodes": int(rec.get("episodes") or 0),
                "beaten": bool(rec.get("beaten")),
            })
        if rungs:
            ladders.append({"map": m, "rungs": rungs})
    return ladders


def build_payload(runs, progress, generated_at=""):
    groups = OrderedDict()
    ordered = OrderedDict()
    for row in runs:
        m, mode, label = group_of(row)
        ordered.setdefault(label, []).append(row)
    for label, rows in ordered.items():
        m, mode, _ = group_of(rows[0])
        g = summarize_group(rows)
        g["label"] = label
        g["map"] = m
        g["mode"] = mode
        groups[label] = g
    # default to the slice with the most episodes -- the one being trained
    order = sorted(groups, key=lambda k: -groups[k]["episodes"])
    return {
        "groups": groups,
        "order": order,
        "default": order[0] if order else None,
        "ladder": build_ladder(progress),
        "generatedAt": generated_at,
        "totalEpisodes": sum(g["episodes"] for g in groups.values()),
    }


# --------------------------------------------------------------------------
# Rendering. Python emits the shell + the embedded data + a no-JS table twin;
# the JS below draws the SVG charts, the hover layer, the filter, and rebuilds
# the table per slice. Colors are the validated data-viz reference palette.
# --------------------------------------------------------------------------

def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _noscript_table(payload):
    key = payload.get("default")
    if not key:
        return "<p>No episodes logged yet.</p>"
    g = payload["groups"][key]
    head = ("<tr><th>episode</th><th>round reached</th><th>outcome</th>"
            "<th>strategy</th></tr>")
    body = []
    for s in g["series"]:
        body.append(
            "<tr><td>{i}</td><td>{r}</td><td>{o}</td><td>{k}</td></tr>".format(
                i=s["i"], r=("—" if s["round"] is None else s["round"]),
                o=_esc(s["outcome"]), k=_esc(s["kind"])))
    return (f"<p>Showing <strong>{_esc(key)}</strong> "
            f"({g['episodes']} episodes). Enable JavaScript for charts.</p>"
            f"<table class='tbl'>{head}{''.join(body)}</table>")


def render_html(payload):
    data_json = json.dumps(payload, separators=(",", ":"))
    data_json = data_json.replace("</", "<\\/")     # can't break out of <script>
    noscript = _noscript_table(payload)
    return (_TEMPLATE
            .replace("/*__DATA__*/", data_json)
            .replace("<!--__NOSCRIPT__-->", noscript))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default=str(REPO / "runs_log.jsonl"),
                    help="path to runs_log.jsonl (default: repo root)")
    ap.add_argument("--progress", default=str(REPO / "progress.json"),
                    help="path to progress.json (default: repo root)")
    ap.add_argument("--out", default=str(REPO / "progress.html"),
                    help="output HTML file (default: progress.html)")
    ap.add_argument("--open", action="store_true",
                    help="open the file in a browser when done")
    args = ap.parse_args(argv)

    runs = load_runs(args.runs)
    progress = load_progress(args.progress)
    payload = build_payload(runs, progress)
    html = render_html(payload)
    Path(args.out).write_text(html)
    n = payload["totalEpisodes"]
    where = Path(args.out).resolve()
    if not runs:
        print(f"No episodes in {args.runs} yet -- wrote an empty dashboard to "
              f"{where}.\nTrain first (e.g. python mk.py solve <map>), then "
              f"re-run this.")
    else:
        print(f"Wrote {where} ({n} episodes across "
              f"{len(payload['groups'])} map/mode slice(s)).")
    if args.open:
        import webbrowser
        webbrowser.open(where.as_uri())
    return 0


# The single-file dashboard shell. {DATA} is spliced in; the JS renderer draws
# everything from it. Kept as a plain string (no f-string) so CSS/JS braces are
# literal.
_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BTD6 bot -- training progress</title>
<style>
:root{
  --surface:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
  --accent:#2a78d6; --good:#0ca30c; --critical:#d03b3b; --serious:#ec835a;
  --warning:#fab219; --neutral:#898781; --lives:#e34948;
}
@media (prefers-color-scheme:dark){:root{
  --surface:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --accent:#3987e5; --good:#0ca30c; --critical:#d03b3b; --serious:#ec835a;
  --warning:#fab219; --lives:#e66767;
}}
:root[data-theme=light]{
  --surface:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10); --accent:#2a78d6;
  --lives:#e34948;
}
:root[data-theme=dark]{
  --surface:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink2:#c3c2b7;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10); --accent:#3987e5;
  --lives:#e66767;
}
*{box-sizing:border-box}
html,body{margin:0}
body{background:var(--plane);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:14px;line-height:1.45;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:28px 20px 64px}
header.top{display:flex;align-items:baseline;justify-content:space-between;
  gap:16px;flex-wrap:wrap;margin-bottom:6px}
h1{font-size:20px;font-weight:650;margin:0}
.sub{color:var(--ink2);font-size:13px}
.toggle{border:1px solid var(--border);background:var(--surface);color:var(--ink2);
  border-radius:999px;padding:5px 12px;font:inherit;font-size:12px;cursor:pointer}
.toggle:hover{color:var(--ink)}
#filters{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0 8px}
.chip{border:1px solid var(--border);background:var(--surface);color:var(--ink2);
  border-radius:999px;padding:6px 13px;font:inherit;font-size:13px;cursor:pointer}
.chip[aria-pressed=true]{border-color:var(--accent);color:var(--ink);
  box-shadow:inset 0 0 0 1px var(--accent)}
.hero{display:flex;gap:28px;align-items:flex-end;flex-wrap:wrap;
  margin:16px 0 22px}
.hero .num{font-size:52px;font-weight:680;line-height:1;letter-spacing:-.01em}
.hero .of{font-size:20px;color:var(--muted);font-weight:500}
.hero .cap{color:var(--ink2);font-size:13px;margin-top:6px}
.badge{display:inline-flex;align-items:center;gap:6px;border-radius:999px;
  padding:4px 11px;font-size:12px;font-weight:600;border:1px solid transparent}
.badge.win{color:var(--good);border-color:var(--good)}
.badge.grind{color:var(--ink2);border-color:var(--border)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;
  margin-bottom:22px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;
  padding:13px 15px}
.tile .k{color:var(--ink2);font-size:12px}
.tile .v{font-size:24px;font-weight:640;margin-top:3px}
.tile .d{font-size:12px;margin-top:2px;color:var(--muted)}
.tile .d.up{color:var(--good)} .tile .d.dn{color:var(--critical)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;
  padding:16px 16px 10px;margin-bottom:16px}
.card h2{font-size:14px;font-weight:620;margin:0 0 2px}
.card .note{color:var(--ink2);font-size:12.5px;margin:0 0 8px}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin:2px 0 6px;font-size:12px;
  color:var(--ink2)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.lk{width:16px;height:2px;border-radius:2px;display:inline-block}
.ld{width:9px;height:9px;border-radius:50%;display:inline-block}
svg{display:block;width:100%;height:auto;overflow:visible}
.axis text{fill:var(--muted);font-size:11px}
.axis line,.axis path{stroke:var(--axis)}
.grid line{stroke:var(--grid);stroke-width:1}
.lbl{fill:var(--ink2);font-size:11px;font-weight:600}
.lbl.strong{fill:var(--ink)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media (max-width:720px){.grid2{grid-template-columns:1fr}}
.ladder{display:flex;flex-direction:column;gap:10px}
.rung{display:grid;grid-template-columns:120px 1fr auto;gap:12px;align-items:center}
.rung .nm{font-size:13px;color:var(--ink2)}
.track{height:12px;border-radius:999px;background:color-mix(in srgb,var(--accent) 16%,transparent);
  overflow:hidden}
.fill{height:100%;border-radius:999px;background:var(--accent)}
.fill.beat{background:var(--good)}
.rung .rt{font-size:12px;color:var(--ink2);font-variant-numeric:tabular-nums}
.tbl{border-collapse:collapse;width:100%;font-size:12.5px;
  font-variant-numeric:tabular-nums}
.tbl th,.tbl td{text-align:left;padding:5px 10px;border-bottom:1px solid var(--border)}
.tbl th{color:var(--ink2);font-weight:600}
details.tv{margin-top:8px}
details.tv summary{cursor:pointer;color:var(--ink2);font-size:12.5px}
.tip{position:fixed;pointer-events:none;z-index:10;background:var(--surface);
  border:1px solid var(--border);border-radius:9px;padding:8px 10px;font-size:12px;
  box-shadow:0 6px 24px rgba(0,0,0,.16);opacity:0;transition:opacity .08s;max-width:240px}
.tip .tv{font-weight:660;font-size:13px}
.tip .tr{display:flex;align-items:center;gap:7px;color:var(--ink2);margin-top:3px}
.empty{color:var(--ink2);padding:30px 4px}
.foot{color:var(--muted);font-size:12px;margin-top:26px}
.hit{fill:transparent;cursor:crosshair}
</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <div>
      <h1>BTD6 bot &mdash; training progress</h1>
      <div class="sub" id="subtitle">&nbsp;</div>
    </div>
    <button class="toggle" id="themeBtn" type="button">theme</button>
  </header>
  <div id="filters"></div>
  <div id="app"></div>
  <noscript><div class="card"><!--__NOSCRIPT__--></div></noscript>
  <div class="foot" id="foot"></div>
</div>
<div class="tip" id="tip"></div>
<script>
const DATA = /*__DATA__*/;
</script>
<script>
(function(){
"use strict";
const $=(id)=>document.getElementById(id);
const svgNS="http://www.w3.org/2000/svg";
const tip=$("tip");

// ---- theme (manual toggle wins over prefers-color-scheme) ----
function initTheme(){
  const b=$("themeBtn");
  b.addEventListener("click",()=>{
    const cur=document.documentElement.getAttribute("data-theme");
    const sysDark=matchMedia("(prefers-color-scheme:dark)").matches;
    const next=cur?(cur==="dark"?"light":"dark"):(sysDark?"light":"dark");
    document.documentElement.setAttribute("data-theme",next);
    if(state.key) render(state.key);           // repaint marks in new theme
  });
}
function cssvar(n){return getComputedStyle(document.body).getPropertyValue(n).trim();}
const SEV={good:"--good",critical:"--critical",serious:"--serious",
  warning:"--warning",muted:"--muted"};
const ICON={good:"✓",critical:"✕",serious:"?",warning:"!",muted:"•"};

// ---- tiny helpers ----
function fmt(n){
  if(n==null)return "—";
  if(Math.abs(n)>=1e6)return (n/1e6).toFixed(1).replace(/\.0$/,"")+"M";
  if(Math.abs(n)>=1e4)return (n/1e3).toFixed(1).replace(/\.0$/,"")+"K";
  return (Math.round(n*100)/100).toLocaleString();
}
function el(tag,attrs,kids){
  const e=document.createElement(tag);
  if(attrs)for(const k in attrs){if(k==="text")e.textContent=attrs[k];
    else if(k==="html")e.innerHTML=attrs[k]; else e.setAttribute(k,attrs[k]);}
  (kids||[]).forEach(c=>e.appendChild(c));
  return e;
}
function S(tag,attrs){
  const e=document.createElementNS(svgNS,tag);
  for(const k in (attrs||{}))e.setAttribute(k,attrs[k]);
  return e;
}
function niceTicks(lo,hi,n){
  if(hi<=lo)hi=lo+1;
  const raw=(hi-lo)/n, mag=Math.pow(10,Math.floor(Math.log10(raw)));
  const norm=raw/mag, step=(norm<1.5?1:norm<3?2:norm<7?5:10)*mag;
  const out=[]; let t=Math.ceil(lo/step)*step;
  for(;t<=hi+1e-9;t+=step)out.push(Math.round(t*1e6)/1e6);
  return out;
}

// ---- generic axes into a plot group ----
function plot(host,W,H,pad){
  const svg=S("svg",{viewBox:`0 0 ${W} ${H}`,role:"img"});
  host.appendChild(svg);
  return {svg,W,H,pad,
    x0:pad.l, x1:W-pad.r, y0:H-pad.b, y1:pad.t,
    sx(v,lo,hi){return pad.l+(v-lo)/((hi-lo)||1)*(W-pad.l-pad.r);},
    sy(v,lo,hi){return (H-pad.b)-(v-lo)/((hi-lo)||1)*(H-pad.b-pad.t);}};
}
function yGrid(p,lo,hi,ticks,fmtT){
  const g=S("g",{class:"grid"}),a=S("g",{class:"axis"});
  ticks.forEach(t=>{
    const y=p.sy(t,lo,hi);
    g.appendChild(S("line",{x1:p.x0,x2:p.x1,y1:y,y2:y}));
    const tx=S("text",{x:p.x0-8,y:y+3,"text-anchor":"end"});
    tx.textContent=(fmtT||fmt)(t); a.appendChild(tx);
  });
  p.svg.appendChild(g); p.svg.appendChild(a);
}
function xTicks(p,vals,fmtT){
  const a=S("g",{class:"axis"});
  a.appendChild(S("line",{x1:p.x0,x2:p.x1,y1:p.y0,y2:p.y0}));
  vals.forEach(v=>{const tx=S("text",{x:v.x,y:p.y0+16,"text-anchor":"middle"});
    tx.textContent=v.t; a.appendChild(tx);});
  p.svg.appendChild(a);
}

// ---- state ----
const state={key:null};

// ======================= CHARTS =======================
function progressChart(host,g){
  const s=g.series; if(!s.length){host.appendChild(el("div",{class:"empty",
    text:"No episodes yet."}));return;}
  const W=1040,H=340,pad={l:46,r:54,t:14,b:30};
  const p=plot(host,W,H,pad);
  const maxR=Math.max(g.target||0,g.bestRound,
    ...s.map(d=>d.round||0),10);
  const yhi=Math.ceil(maxR/10)*10;
  const ticks=niceTicks(0,yhi,5);
  yGrid(p,0,yhi,ticks);
  const n=s.length, xlo=1, xhi=Math.max(n,2);
  const xv=niceTicks(1,xhi,Math.min(n,8)).filter(v=>v>=1&&v<=xhi);
  xTicks(p,xv.map(v=>({x:p.sx(v,xlo,xhi),t:String(v)})));
  // target line
  if(g.target){const y=p.sy(g.target,0,yhi);
    p.svg.appendChild(S("line",{x1:p.x0,x2:p.x1,y1:y,y2:y,
      stroke:cssvar("--axis"),"stroke-width":1}));
    const t=S("text",{x:p.x1+6,y:y+3,class:"lbl"});t.textContent="target "+g.target;
    p.svg.appendChild(t);}
  // running-best line (accent) -- the story
  let d="";
  g.runningMax.forEach((v,i)=>{const x=p.sx(i+1,xlo,xhi),y=p.sy(v,0,yhi);
    d+=(i?"L":"M")+x+" "+y+" ";});
  p.svg.appendChild(S("path",{d:d,fill:"none",stroke:cssvar("--accent"),
    "stroke-width":2,"stroke-linejoin":"round","stroke-linecap":"round"}));
  // episode dots: muted context, wins highlighted (status good) + ring
  s.forEach(d=>{if(d.round==null)return;
    const x=p.sx(d.i,xlo,xhi),y=p.sy(d.round,0,yhi);
    if(d.win){
      p.svg.appendChild(S("circle",{cx:x,cy:y,r:5.5,fill:cssvar("--good"),
        stroke:cssvar("--surface"),"stroke-width":2}));
    }else{
      p.svg.appendChild(S("circle",{cx:x,cy:y,r:3,fill:cssvar("--muted"),
        opacity:.7}));
    }});
  // end label: current personal best
  const last=g.runningMax[g.runningMax.length-1];
  if(last!=null){const x=p.sx(n,xlo,xhi),y=p.sy(last,0,yhi);
    p.svg.appendChild(S("circle",{cx:x,cy:y,r:4,fill:cssvar("--accent"),
      stroke:cssvar("--surface"),"stroke-width":2}));
    const t=S("text",{x:Math.min(x+8,p.x1+6),y:y-8,class:"lbl strong"});
    t.textContent="best "+last; p.svg.appendChild(t);}
  // crosshair + hover
  const cross=S("line",{x1:0,x2:0,y1:p.y1,y2:p.y0,stroke:cssvar("--axis"),
    "stroke-width":1,opacity:0}); p.svg.appendChild(cross);
  const hit=S("rect",{x:p.x0,y:p.y1,width:p.x1-p.x0,height:p.y0-p.y1,
    class:"hit"}); p.svg.appendChild(hit);
  function move(ev){
    const r=p.svg.getBoundingClientRect(), sc=W/r.width;
    const mx=(ev.clientX-r.left)*sc;
    const idx=Math.max(0,Math.min(n-1,Math.round(
      (mx-p.x0)/((p.x1-p.x0)||1)*(xhi-xlo)+xlo)-1));
    const d=s[idx], x=p.sx(d.i,xlo,xhi);
    cross.setAttribute("x1",x);cross.setAttribute("x2",x);cross.setAttribute("opacity",1);
    const rows=[["episode",String(d.i)],
      ["round reached",d.round==null?"—":String(d.round)],
      ["outcome",d.outcome],["strategy",d.kind]];
    showTip(ev,"best so far "+g.runningMax[idx],rows,d.win?"good":null);
  }
  hit.addEventListener("pointermove",move);
  hit.addEventListener("pointerleave",()=>{cross.setAttribute("opacity",0);hideTip();});
}

function deathsChart(host,g){
  const h=g.deathHist; if(!h.length){host.appendChild(el("div",{class:"empty",
    text:"No finished runs yet."}));return;}
  const W=506,H=250,pad={l:40,r:14,t:12,b:30};
  const p=plot(host,W,H,pad);
  const maxC=Math.max(...h.map(d=>d.count),1);
  const yhi=Math.max(maxC,1);
  yGrid(p,0,yhi,niceTicks(0,yhi,Math.min(yhi,4)));
  const xlo=Math.min(...h.map(d=>d.bucket));
  const xhi=Math.max(...h.map(d=>d.bucket))+5;
  const bw=Math.min(24,(p.x1-p.x0)/((xhi-xlo)/5)-6);
  h.forEach(d=>{
    const x=p.sx(d.bucket+2.5,xlo,xhi), y=p.sy(d.count,0,yhi);
    const bh=p.y0-y; if(bh<=0)return;
    const rx=Math.min(4,bw/2,bh/2);        // clamp by height too, or a short
                                           // bar's rounded top dips past the base
    // rounded top, square base
    const path=`M${x-bw/2} ${p.y0} L${x-bw/2} ${y+rx}
      Q${x-bw/2} ${y} ${x-bw/2+rx} ${y} L${x+bw/2-rx} ${y}
      Q${x+bw/2} ${y} ${x+bw/2} ${y+rx} L${x+bw/2} ${p.y0} Z`;
    p.svg.appendChild(S("path",{d:path,fill:cssvar("--accent")}));
    const hit=S("rect",{x:x-bw/2-1,y:p.y1,width:bw+2,height:p.y0-p.y1,class:"hit"});
    hit.addEventListener("pointermove",ev=>showTip(ev,d.count+" run"+(d.count!=1?"s":""),
      [["ended at","round "+d.bucket+"–"+(d.bucket+4)]]));
    hit.addEventListener("pointerleave",hideTip);
    p.svg.appendChild(hit);
  });
  const xv=[xlo,xlo+Math.round((xhi-xlo)/2/5)*5,xhi].filter((v,i,a)=>a.indexOf(v)===i);
  xTicks(p,xv.map(v=>({x:p.sx(v,xlo,xhi),t:String(v)})));
}

function outcomesChart(host,g){
  const o=g.outcomes; if(!o.length){host.appendChild(el("div",{class:"empty",
    text:"No outcomes yet."}));return;}
  const total=o.reduce((a,d)=>a+d.count,0);
  const W=506,H=Math.max(120,o.length*40+20),pad={l:96,r:44,t:8,b:8};
  const p=plot(host,W,H,pad);
  const maxC=Math.max(...o.map(d=>d.count),1);
  const rowH=(H-pad.t-pad.b)/o.length, bh=Math.min(22,rowH-12);
  o.forEach((d,i)=>{
    const cy=pad.t+i*rowH+rowH/2, col=cssvar(SEV[d.sev]||"--muted");
    const w=(d.count/maxC)*(p.x1-p.x0), rx=Math.min(4,bh/2);
    const path=`M${p.x0} ${cy-bh/2} L${p.x0+Math.max(w-rx,0)} ${cy-bh/2}
      Q${p.x0+w} ${cy-bh/2} ${p.x0+w} ${cy-bh/2+rx} L${p.x0+w} ${cy+bh/2-rx}
      Q${p.x0+w} ${cy+bh/2} ${p.x0+Math.max(w-rx,0)} ${cy+bh/2} L${p.x0} ${cy+bh/2} Z`;
    p.svg.appendChild(S("path",{d:path,fill:col}));
    const lab=S("text",{x:p.x0-10,y:cy+4,"text-anchor":"end",class:"lbl"});
    lab.textContent=ICON[d.sev]+" "+d.label; p.svg.appendChild(lab);
    const val=S("text",{x:p.x0+w+7,y:cy+4,class:"lbl strong"});
    val.textContent=d.count+" ("+Math.round(100*d.count/total)+"%)";
    p.svg.appendChild(val);
    const hit=S("rect",{x:0,y:cy-rowH/2,width:W,height:rowH,class:"hit"});
    hit.addEventListener("pointermove",ev=>showTip(ev,d.count+" episode"+(d.count!=1?"s":""),
      [[d.label,Math.round(100*d.count/total)+"% of runs"]]));
    hit.addEventListener("pointerleave",hideTip);
    p.svg.appendChild(hit);
  });
}

function lineChart(host,pts,color,label,unit,extra){
  if(!pts.length){host.appendChild(el("div",{class:"empty",text:"No data."}));return;}
  const W=506,H=230,pad={l:48,r:16,t:14,b:28};
  const p=plot(host,W,H,pad);
  const all=pts.concat(extra||[]);
  const xlo=Math.min(...all.map(d=>d.r)), xhi=Math.max(...all.map(d=>d.r));
  const yhi=Math.max(...all.map(d=>d.v),1), ylo=0;
  yGrid(p,ylo,yhi,niceTicks(ylo,yhi,4));
  const xv=niceTicks(xlo,xhi,5).filter(v=>v>=xlo&&v<=xhi);
  xTicks(p,xv.map(v=>({x:p.sx(v,xlo,xhi),t:String(v)})));
  function line(data,col,dash){
    let d=""; data.forEach((q,i)=>{d+=(i?"L":"M")+p.sx(q.r,xlo,xhi)+" "+p.sy(q.v,ylo,yhi)+" ";});
    const path=S("path",{d:d,fill:"none",stroke:col,"stroke-width":2,
      "stroke-linejoin":"round","stroke-linecap":"round"});
    if(dash)path.setAttribute("opacity",.5);
    p.svg.appendChild(path);
  }
  if(extra&&extra.length)line(extra,cssvar("--muted"),true);
  line(pts,color,false);
  // hover crosshair over rounds
  const cross=S("line",{x1:0,x2:0,y1:p.y1,y2:p.y0,stroke:cssvar("--axis"),
    "stroke-width":1,opacity:0}); p.svg.appendChild(cross);
  const hit=S("rect",{x:p.x0,y:p.y1,width:p.x1-p.x0,height:p.y0-p.y1,class:"hit"});
  p.svg.appendChild(hit);
  hit.addEventListener("pointermove",ev=>{
    const r=p.svg.getBoundingClientRect(),sc=W/r.width,mx=(ev.clientX-r.left)*sc;
    let best=pts[0],bd=1e9;
    pts.forEach(q=>{const dx=Math.abs(p.sx(q.r,xlo,xhi)-mx);if(dx<bd){bd=dx;best=q;}});
    const x=p.sx(best.r,xlo,xhi);
    cross.setAttribute("x1",x);cross.setAttribute("x2",x);cross.setAttribute("opacity",1);
    const rows=[[label,fmt(best.v)+(unit||"")]];
    const pr=(extra||[]).find(q=>q.r===best.r);
    if(pr)rows.push(["model expects",fmt(pr.v)+(unit||"")]);
    showTip(ev,"round "+best.r,rows);
  });
  hit.addEventListener("pointerleave",()=>{cross.setAttribute("opacity",0);hideTip();});
}

// ---- tooltip ----
function showTip(ev,value,rows,sev){
  tip.textContent="";
  tip.appendChild(el("div",{class:"tv",text:value}));
  (rows||[]).forEach(r=>{
    const line=el("div",{class:"tr"});
    line.appendChild(el("span",{text:r[0]+": "}));
    line.appendChild(el("strong",{text:r[1]}));
    tip.appendChild(line);
  });
  tip.style.opacity=1;
  const pad=14,w=tip.offsetWidth,h=tip.offsetHeight;
  let x=ev.clientX+pad,y=ev.clientY+pad;
  if(x+w>innerWidth)x=ev.clientX-w-pad;
  if(y+h>innerHeight)y=ev.clientY-h-pad;
  tip.style.left=x+"px"; tip.style.top=y+"px";
}
function hideTip(){tip.style.opacity=0;}

// ---- KPI + hero + ladder + table ----
function kpiTile(k,v,d,cls){
  const t=el("div",{class:"tile"});
  t.appendChild(el("div",{class:"k",text:k}));
  t.appendChild(el("div",{class:"v",text:v}));
  if(d)t.appendChild(el("div",{class:"d "+(cls||""),text:d}));
  return t;
}
function renderHero(host,g){
  const h=el("div",{class:"hero"});
  const beaten=g.wins>0;
  const num=el("div");
  const big=el("span",{class:"num",text:String(g.bestRound)});
  num.appendChild(big);
  if(g.target)num.appendChild(el("span",{class:"of",text:" / "+g.target}));
  const wrap=el("div");
  wrap.appendChild(num);
  wrap.appendChild(el("div",{class:"cap",text:"deepest round reached"}));
  h.appendChild(wrap);
  const badge=el("div",{class:"badge "+(beaten?"win":"grind"),
    text:beaten?("✓ beaten ×"+g.wins):"not beaten yet"});
  const bwrap=el("div"); bwrap.appendChild(badge);
  h.appendChild(bwrap);
  host.appendChild(h);
}
function renderKPIs(host,g){
  const k=el("div",{class:"kpis"});
  k.appendChild(kpiTile("episodes trained",fmt(g.episodes)));
  k.appendChild(kpiTile("wins",fmt(g.wins),
    Math.round(g.winRate*100)+"% win rate"));
  const dB=g.recentBest-g.headBest;
  k.appendChild(kpiTile("best round (last "+g.recentN+")",fmt(g.recentBest),
    (dB>0?"+":"")+dB+" vs first 10",dB>0?"up":(dB<0?"dn":"")));
  k.appendChild(kpiTile("recent win rate",Math.round(g.recentWinRate*100)+"%",
    g.recentWins+" of last "+g.recentN));
  host.appendChild(k);
}
function card(title,note){
  const c=el("div",{class:"card"});
  c.appendChild(el("h2",{text:title}));
  if(note)c.appendChild(el("p",{class:"note",text:note}));
  return c;
}
function legend(items){
  const l=el("div",{class:"legend"});
  items.forEach(it=>{
    const s=el("span");
    const m=el("span",{class:it.line?"lk":"ld"});
    m.style.background=cssvar(it.color);
    s.appendChild(m); s.appendChild(el("span",{text:it.text}));
    l.appendChild(s);
  });
  return l;
}
function renderLadder(host){
  if(!DATA.ladder||!DATA.ladder.length)return;
  DATA.ladder.forEach(ld=>{
    const c=card("Campaign ladder — "+ld.map,
      "best round reached vs the round that beats each rung");
    const box=el("div",{class:"ladder"});
    ld.rungs.forEach(r=>{
      const row=el("div",{class:"rung"});
      row.appendChild(el("div",{class:"nm",text:r.name+(r.beaten?" ✓":"")}));
      const track=el("div",{class:"track"});
      const frac=r.target?Math.min(1,r.best/r.target):0;
      const fill=el("div",{class:"fill"+(r.beaten?" beat":"")});
      fill.style.width=Math.round(frac*100)+"%";
      track.appendChild(fill); row.appendChild(track);
      row.appendChild(el("div",{class:"rt",text:r.best+" / "+r.target}));
      box.appendChild(row);
    });
    c.appendChild(box); host.appendChild(c);
  });
}
function renderTable(host,g){
  const d=el("details",{class:"tv"});
  d.appendChild(el("summary",{text:"table view (every episode)"}));
  const t=el("table",{class:"tbl"});
  const hd=el("tr");
  ["episode","round reached","outcome","strategy"].forEach(h=>hd.appendChild(el("th",{text:h})));
  t.appendChild(hd);
  g.series.slice().reverse().forEach(s=>{
    const tr=el("tr");
    tr.appendChild(el("td",{text:String(s.i)}));
    tr.appendChild(el("td",{text:s.round==null?"—":String(s.round)}));
    tr.appendChild(el("td",{text:s.outcome}));
    tr.appendChild(el("td",{text:s.kind}));
    t.appendChild(tr);
  });
  d.appendChild(t); host.appendChild(d);
}

// ======================= RENDER =======================
function render(key){
  state.key=key;
  const g=DATA.groups[key];
  $("subtitle").textContent=g.label+" — "+g.episodes+" episodes";
  document.querySelectorAll("#filters .chip").forEach(c=>
    c.setAttribute("aria-pressed",String(c.dataset.key===key)));
  const app=$("app"); app.textContent="";
  renderHero(app,g);
  renderKPIs(app,g);

  const c1=card("Progress over episodes",
    "each run's deepest round (gray), the running personal best (blue), wins in green");
  c1.appendChild(legend([{color:"--accent",line:true,text:"personal best"},
    {color:"--good",text:"victory"},{color:"--muted",text:"episode"}]));
  progressChart(c1,g); renderTable(c1,g); app.appendChild(c1);

  const g2=el("div",{class:"grid2"});
  const c2=card("Where runs end","how many runs finished in each 5-round band");
  deathsChart(c2,g); g2.appendChild(c2);
  const c3=card("Outcomes","how the episodes ended");
  outcomesChart(c3,g); g2.appendChild(c3);
  app.appendChild(g2);

  if(g.bestRun&&(g.bestRun.lives.length||g.bestRun.cash.length)){
    const c4=card("Best run so far — round "+(g.bestRun.final||"?"),
      "lives and cash through the deepest run; the model's expected cash is the faint line");
    const gr=el("div",{class:"grid2"});
    const a=el("div"); a.appendChild(el("div",{class:"note",html:"&nbsp;lives"}));
    lineChart(a,g.bestRun.lives,cssvar("--lives"),"lives","");
    const b=el("div"); b.appendChild(el("div",{class:"note",html:"&nbsp;cash"}));
    lineChart(b,g.bestRun.cash,cssvar("--accent"),"cash","$",g.bestRun.prior);
    gr.appendChild(a); gr.appendChild(b); c4.appendChild(gr); app.appendChild(c4);
  }
  renderLadder(app);
}

function initFilters(){
  const f=$("filters");
  if(DATA.order.length<2)return;              // one slice: no filter row needed
  DATA.order.forEach(k=>{
    const g=DATA.groups[k];
    const c=el("button",{class:"chip",type:"button",text:g.label+" ("+g.episodes+")"});
    c.dataset.key=k;
    c.addEventListener("click",()=>render(k));
    f.appendChild(c);
  });
}

function boot(){
  initTheme();
  $("foot").textContent="Generated from runs_log.jsonl"+
    (DATA.generatedAt?(" · "+DATA.generatedAt):"")+
    " · "+DATA.totalEpisodes+" episodes total";
  if(!DATA.default){
    $("app").appendChild(el("div",{class:"card"},[
      el("div",{class:"empty",text:"No episodes logged yet. Train the bot "+
        "(e.g. python mk.py solve <map>), then regenerate this page."})]));
    return;
  }
  initFilters();
  render(DATA.default);
}
boot();
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
