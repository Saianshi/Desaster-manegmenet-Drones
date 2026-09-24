"""
Builds output/player.html: a self-contained, single-file canvas animation
player -- baseline (left) vs our engine (right), same clock, same seed.
Reads ONLY output/run_log.json (does not import /engine or /simulation).
All data the page needs is downsampled and embedded directly in the HTML
as a JSON blob, so it loads and plays instantly with no fetch, no
external JS library, and works as a static file on GitHub Pages.

What it renders, per panel, per frame:
  - flood water as a translucent layer that visibly rises (coarse
    elevation grid vs current water_level, both already public World
    state -- nothing here re-derives simulation physics)
  - buildings (collapsed = darker)
  - victims: grey = undetected, colour-graded green->red by remaining
    survival "urgency" once first detected, black X once dead, green
    marker once rescued
  - drones: triangle, green if backhaul-connected to base else red, a
    faint sensor-coverage disc, and dashed backhaul links (green/red
    matching the same connectivity)
  - resources: marker per type, visibly travelling (their logged
    position already interpolates during the travel state)
  - a one-time callout at the first tick the two strategies' cumulative
    rescued-people counts stop being equal, quoting the engine's own
    Assignment.reason for the dispatch responsible

Run standalone with:  python -m viz.player
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import params
from viz import logdata

OUT_PATH = ROOT / "output" / "player.html"

STRIDE = 4                # downsample: every 4th tick = every 2 sim-minutes
FLOOD_GRID_POOL = 5        # average-pool the 200x200 elevation map by this factor


def _pool_elevation(elevation_map, factor: int):
    n = len(elevation_map)
    m = n // factor
    pooled = []
    for gy in range(m):
        row = []
        for gx in range(m):
            vals = [
                elevation_map[gy * factor + dy][gx * factor + dx]
                for dy in range(factor) for dx in range(factor)
            ]
            row.append(round(sum(vals) / len(vals), 2))
        pooled.append(row)
    return pooled


def _victim_state_char(status: str, urgency: float) -> str:
    if status == "rescued":
        return "r"
    if status == "dead":
        return "x"
    if status == "undetected":
        return "u"
    decile = min(9, max(0, int(urgency * 10)))
    return str(decile)


def _build_run_data(log: dict, run_name: str, victim_ids: list, group_size: dict) -> dict:
    entries = log["runs"][run_name]["entries"]
    gt_timesteps = log["runs"][run_name]["ground_truth"]["timesteps"]
    n_full = len(entries)
    first_detected = logdata.first_detected_times(log, run_name)

    # per-victim rescued/dead time, so per-frame status is a cheap lookup
    rescued_t = {}
    dead_t = {}
    for e in log["runs"][run_name]["ground_truth"]["rescue_events"]:
        if e["outcome"] == "rescued":
            rescued_t[e["victim_id"]] = e["t"]
    for e in log["runs"][run_name]["ground_truth"]["death_events"]:
        dead_t.setdefault(e["victim_id"], e["t"])

    frame_indices = list(range(0, n_full, STRIDE))
    frames = []
    cum_rescued_people = []
    cum_dead_people = []
    for i in frame_indices:
        entry = entries[i]
        t = entry["input"]["t"]
        gt = gt_timesteps[i]
        water_level = entry["input"]["environment"]["water_level"]

        drones = []
        for d in entry["input"]["drones"]:
            drones.append({"id": d["drone_id"], "x": round(d["position"][0], 1), "y": round(d["position"][1], 1),
                            "linked": list(d["linked_to"])})
        has_backhaul = logdata.compute_backhaul_from_linked_to(entry["input"]["drones"])
        for d in drones:
            d["conn"] = bool(has_backhaul.get(d["id"], False))

        resources = [
            {"id": r["resource_id"], "type": r["type"], "x": round(r["position"][0], 1),
             "y": round(r["position"][1], 1), "status": r["status"]}
            for r in entry["input"]["resources"]
        ]

        urgency_by_vid = {v["victim_id"]: v.get("urgency", 0.0) for v in gt["victims"]}
        alive_by_vid = {v["victim_id"]: v["alive"] for v in gt["victims"]}

        state_chars = []
        rescued_now = 0
        dead_now = 0
        for vid in victim_ids:
            gs = group_size[vid]
            if vid in rescued_t and rescued_t[vid] <= t:
                state_chars.append("r")
                rescued_now += gs
            elif vid in dead_t and dead_t[vid] <= t:
                state_chars.append("x")
                dead_now += gs
            elif first_detected.get(vid, float("inf")) <= t and alive_by_vid.get(vid, True):
                state_chars.append(_victim_state_char("alive", urgency_by_vid.get(vid, 0.0)))
            else:
                state_chars.append("u")

        frames.append({
            "t": t, "wl": round(water_level, 2), "d": drones, "r": resources,
            "v": "".join(state_chars),
        })
        cum_rescued_people.append(rescued_now)
        cum_dead_people.append(dead_now)

    return {
        "frames": frames,
        "cumRescued": cum_rescued_people,
        "cumDead": cum_dead_people,
    }


def _find_divergence(log: dict, victim_ids: list, group_size: dict) -> dict:
    """First FULL-resolution tick where cumulative rescued-people counts
    differ between the two runs, plus the reason string for the dispatch
    that produced the winning run's first contributing rescue."""
    n_full = len(log["runs"]["engine"]["entries"])
    rescued_events = {
        run: sorted((e for e in log["runs"][run]["ground_truth"]["rescue_events"] if e["outcome"] == "rescued"),
                    key=lambda e: e["t"])
        for run in ("engine", "baseline")
    }

    def cum_at(run, t):
        return sum(group_size[e["victim_id"]] for e in rescued_events[run] if e["t"] <= t)

    entries_engine = log["runs"]["engine"]["entries"]
    divergence_t = None
    winner = None
    for i in range(n_full):
        t = entries_engine[i]["input"]["t"]
        pe = cum_at("engine", t)
        pb = cum_at("baseline", t)
        if pe != pb:
            divergence_t = t
            winner = "engine" if pe > pb else "baseline"
            break

    if divergence_t is None:
        return {"found": False}

    # the first rescue event, in the winning run, at or before divergence_t
    win_event = next((e for e in rescued_events[winner] if e["t"] <= divergence_t), None)
    if win_event is None:
        return {"found": False}

    reason, dispatch_t, resource_type = None, None, None
    for entry in log["runs"][winner]["entries"]:
        if entry["input"]["t"] > win_event["t"]:
            break
        for a in entry["output"]["assignments"]:
            if a["resource_id"] == win_event["resource_id"] and a["target_belief_id"] == win_event["belief_id"]:
                reason = a["reason"]
                dispatch_t = entry["input"]["t"]
    for r in log["runs"][winner]["entries"][0]["input"]["resources"]:
        if r["resource_id"] == win_event["resource_id"]:
            resource_type = r["type"]

    # locate the divergence in DOWNSAMPLED frame space (both runs share the same frame grid)
    downsampled_indices = list(range(0, n_full, STRIDE))
    frame_idx = min(range(len(downsampled_indices)), key=lambda k: abs(downsampled_indices[k] - int(divergence_t // params.DT_SECONDS)))

    victim = win_event.get("victim_id")
    return {
        "found": True,
        "run": winner,
        "t": divergence_t,
        "frameIdx": frame_idx,
        "reason": reason or "(reason not found -- dispatch may have occurred before logging window)",
        "resourceType": resource_type or "?",
        "resourceId": win_event["resource_id"],
        "groupSize": group_size.get(victim, 1),
    }


def build_player_data(log: dict) -> dict:
    victim_ids = list(log["victims_static"].keys())
    group_size = logdata.group_size_lookup(log)
    env_type = logdata.env_type_lookup(log)
    positions = logdata.victim_true_positions(log, "engine")  # identical for both runs, same seed

    elevation_pooled = _pool_elevation(log["world_static"]["elevation_map"], FLOOD_GRID_POOL)
    pool_cell_m = log["world_static"]["cell_size_m"] * FLOOD_GRID_POOL

    buildings = [
        {"x0": b["x0"], "y0": b["y0"], "x1": b["x1"], "y1": b["y1"], "c": b["collapsed"]}
        for b in log["world_static"]["buildings"]
    ]

    runs = {
        run: _build_run_data(log, run, victim_ids, group_size)
        for run in ("engine", "baseline")
    }

    divergence = _find_divergence(log, victim_ids, group_size)

    total_people = sum(group_size.values())

    return {
        "meta": {
            "mapSize": log["meta"]["map_size_m"],
            "T": log["meta"]["T"],
            "dt": log["meta"]["dt"],
            "stride": STRIDE,
            "nFrames": len(runs["engine"]["frames"]),
            "base": list(params.BASE_STATION_POSITION),
            "coverageRadius": params.RF_RANGE_M,
            "totalPeople": total_people,
            "poolCellM": pool_cell_m,
        },
        "elevation": elevation_pooled,
        "buildings": buildings,
        "victims": {
            "ids": victim_ids,
            "x": [round(positions[v][0], 1) for v in victim_ids],
            "y": [round(positions[v][1], 1) for v in victim_ids],
            "gs": [group_size[v] for v in victim_ids],
            "env": [env_type[v] for v in victim_ids],
        },
        "runs": {
            "engine": {"frames": runs["engine"]["frames"], "cumRescued": runs["engine"]["cumRescued"], "cumDead": runs["engine"]["cumDead"]},
            "baseline": {"frames": runs["baseline"]["frames"], "cumRescued": runs["baseline"]["cumRescued"], "cumDead": runs["baseline"]["cumDead"]},
        },
        "divergence": divergence,
    }


PAGE_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Response Player</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { background:#0d1117; color:#c9d1d9; font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin:0; padding:18px 20px 40px; }
  .wrap { max-width:1300px; margin:0 auto; }
  h1 { font-size:19px; color:#e6edf3; margin:0 0 4px; }
  .sub { color:#8b949e; font-size:12.5px; margin:0 0 16px; }
  .transport { display:flex; align-items:center; gap:14px; background:#161b22; border:1px solid #30363d;
               border-radius:10px; padding:12px 16px; margin-bottom:14px; flex-wrap:wrap; }
  .clock { font-size:28px; font-weight:700; color:#58a6ff; font-variant-numeric:tabular-nums; min-width:118px; }
  button.playbtn { background:#238636; color:white; border:none; border-radius:6px; padding:9px 16px;
              font-size:14px; font-weight:600; cursor:pointer; min-width:88px; }
  button.playbtn:hover { background:#2ea043; }
  .timeline { flex:1; min-width:220px; position:relative; }
  .timeline input[type=range] { width:100%; accent-color:#58a6ff; }
  .divergence-tick { position:absolute; top:-2px; width:2px; height:22px; background:#f0883e; pointer-events:none; }
  .speeds { display:flex; gap:4px; }
  .speeds button { background:#21262d; border:1px solid #30363d; color:#c9d1d9; border-radius:5px;
                   padding:6px 10px; font-size:12px; cursor:pointer; }
  .speeds button.active { background:#1f6feb; border-color:#1f6feb; color:white; font-weight:600; }
  .jumpbtn { background:#21262d; border:1px solid #f0883e; color:#f0883e; border-radius:5px; padding:6px 10px; font-size:12px; cursor:pointer; }
  .panels { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
  .panel { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:12px; position:relative; }
  .panel h2 { font-size:13px; text-transform:uppercase; letter-spacing:0.5px; margin:0 0 8px; color:#8b949e; }
  .panel.engine h2 { color:#58a6ff; }
  .panel.baseline h2 { color:#f85149; }
  canvas { width:100%; height:auto; display:block; background:#0b0e14; border-radius:6px; }
  .counters { display:flex; gap:10px; margin-top:10px; }
  .counter { flex:1; background:#0d1117; border:1px solid #21262d; border-radius:8px; padding:8px 6px; text-align:center; }
  .counter .n { font-size:22px; font-weight:700; font-variant-numeric:tabular-nums; }
  .counter .l { font-size:10px; color:#8b949e; text-transform:uppercase; letter-spacing:0.4px; }
  .counter.rescued .n { color:#3fb950; }
  .counter.dead .n { color:#8b949e; }
  .counter.atrisk .n { color:#f0883e; }
  .callout { position:absolute; max-width:78%; background:#1c2333; border:1.5px solid #f0883e; border-radius:8px;
             padding:9px 12px; font-size:11.5px; line-height:1.5; color:#e6edf3; box-shadow:0 4px 18px rgba(0,0,0,0.5);
             pointer-events:none; opacity:0; transition:opacity 0.25s; z-index:5; }
  .callout.show { opacity:1; }
  .callout .tag { color:#f0883e; font-weight:700; font-size:10px; text-transform:uppercase; letter-spacing:0.4px; display:block; margin-bottom:3px; }
  .legend { display:flex; flex-wrap:wrap; gap:14px; margin-top:14px; font-size:11px; color:#8b949e; }
  .legend span.sw { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; vertical-align:middle; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Response player</h1>
  <p class="sub">Baseline (nearest-first) left, our engine right, identical seed, identical clock. Grey victims are undetected; colour runs green (safe) &rarr; red (imminent) once found.</p>

  <div class="transport">
    <div class="clock" id="clock">T+00:00</div>
    <button class="playbtn" id="playBtn">&#9654; Play</button>
    <div class="timeline">
      <input type="range" id="scrub" min="0" max="0" value="0" step="1">
      <div class="divergence-tick" id="divTick" title="first divergence"></div>
    </div>
    <div class="speeds" id="speeds">
      <button data-speed="0.5">0.5x</button>
      <button data-speed="1" class="active">1x</button>
      <button data-speed="2">2x</button>
      <button data-speed="4">4x</button>
      <button data-speed="8">8x</button>
    </div>
    <button class="jumpbtn" id="jumpDiv">&#9873; Jump to divergence</button>
  </div>

  <div class="panels">
    <div class="panel baseline">
      <h2>Baseline (nearest-first)</h2>
      <canvas id="cv-baseline" width="620" height="620"></canvas>
      <div class="callout" id="callout-baseline"></div>
      <div class="counters">
        <div class="counter rescued"><div class="n" id="n-rescued-baseline">0</div><div class="l">Rescued</div></div>
        <div class="counter dead"><div class="n" id="n-dead-baseline">0</div><div class="l">Lost</div></div>
        <div class="counter atrisk"><div class="n" id="n-atrisk-baseline">0</div><div class="l">At risk</div></div>
      </div>
    </div>
    <div class="panel engine">
      <h2>Our engine</h2>
      <canvas id="cv-engine" width="620" height="620"></canvas>
      <div class="callout" id="callout-engine"></div>
      <div class="counters">
        <div class="counter rescued"><div class="n" id="n-rescued-engine">0</div><div class="l">Rescued</div></div>
        <div class="counter dead"><div class="n" id="n-dead-engine">0</div><div class="l">Lost</div></div>
        <div class="counter atrisk"><div class="n" id="n-atrisk-engine">0</div><div class="l">At risk</div></div>
      </div>
    </div>
  </div>

  <div class="legend">
    <span><span class="sw" style="background:#6e7681"></span>undetected</span>
    <span><span class="sw" style="background:#3fb950"></span>detected, safe</span>
    <span><span class="sw" style="background:#f85149"></span>detected, imminent</span>
    <span><span class="sw" style="background:#222" ></span>&times; dead</span>
    <span><span class="sw" style="background:#2ecc71"></span>&#9679; rescued</span>
    <span>&#9650; drone (green=backhaul connected, red=broken)</span>
    <span>&#9632;/&#43;/&#9670; boat / excavator / medical</span>
    <span>&#9733; base station</span>
  </div>
</div>

<script id="player-data" type="application/json">__PLAYER_DATA__</script>
<script>
(function () {
  const DATA = JSON.parse(document.getElementById("player-data").textContent);
  const { meta, elevation, buildings, victims, runs, divergence } = DATA;
  const mapSize = meta.mapSize;
  const nFrames = meta.nFrames;
  const RESOURCE_GLYPH = { boat: "square", excavator: "plus", medical: "diamond" };

  function urgencyColor(decile) {
    // green (0=safe) -> red (9=imminent), matches viz/animate.py's RdYlGn_r
    const stops = [
      [63,185,80],[126,196,80],[189,207,80],[224,196,69],[240,171,63],
      [240,142,63],[236,110,72],[229,83,75],[224,58,64],[248,81,73]
    ];
    const c = stops[Math.max(0, Math.min(9, decile))];
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }

  function fmtClock(tSeconds) {
    const h = Math.floor(tSeconds / 3600);
    const m = Math.floor((tSeconds % 3600) / 60);
    return `T+${String(h).padStart(2,"0")}:${String(m).padStart(2,"0")}`;
  }

  function backhaulEdgeColor(fromConn, toConn) {
    return (fromConn && toConn) ? "rgba(63,185,80,0.55)" : "rgba(248,81,73,0.55)";
  }

  class Panel {
    constructor(runName) {
      this.runName = runName;
      this.canvas = document.getElementById("cv-" + runName);
      this.ctx = this.canvas.getContext("2d");
      this.data = runs[runName];
      this.scale = this.canvas.width / mapSize;
      this.calloutEl = document.getElementById("callout-" + runName);
    }

    worldToCanvas(x, y) {
      return [x * this.scale, this.canvas.height - y * this.scale];
    }

    render(frameIdx) {
      const ctx = this.ctx;
      const W = this.canvas.width, H = this.canvas.height;
      const frame = this.data.frames[frameIdx];
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = "#0b0e14";
      ctx.fillRect(0, 0, W, H);

      // flood: coarse elevation grid, filled where under current water level,
      // alpha scaled by depth so it visibly deepens as water_level rises
      const poolCell = meta.poolCellM;
      const wl = frame.wl;
      ctx.save();
      for (let gy = 0; gy < elevation.length; gy++) {
        const row = elevation[gy];
        for (let gx = 0; gx < row.length; gx++) {
          const depth = wl - row[gx];
          if (depth <= 0) continue;
          const alpha = Math.min(0.62, 0.16 + depth * 0.12);
          ctx.fillStyle = `rgba(46,117,196,${alpha})`;
          const x0 = gx * poolCell, y0 = gy * poolCell;
          const [cx, cy] = this.worldToCanvas(x0, y0 + poolCell);
          ctx.fillRect(cx, cy, poolCell * this.scale, poolCell * this.scale);
        }
      }
      ctx.restore();

      // buildings
      for (const b of buildings) {
        ctx.fillStyle = b.c ? "#33383f" : "#4a5361";
        ctx.globalAlpha = 0.65;
        const [cx, cy] = this.worldToCanvas(b.x0, b.y1);
        ctx.fillRect(cx, cy, (b.x1 - b.x0) * this.scale, (b.y1 - b.y0) * this.scale);
      }
      ctx.globalAlpha = 1;

      // victims
      const states = frame.v;
      for (let i = 0; i < victims.ids.length; i++) {
        const st = states[i];
        const [cx, cy] = this.worldToCanvas(victims.x[i], victims.y[i]);
        if (st === "r") {
          ctx.fillStyle = "#2ecc71";
          ctx.strokeStyle = "#0b0e14"; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.arc(cx, cy, 4.2, 0, 7); ctx.fill(); ctx.stroke();
        } else if (st === "x") {
          ctx.strokeStyle = "#e6edf3"; ctx.lineWidth = 1.6;
          ctx.beginPath();
          ctx.moveTo(cx - 3.5, cy - 3.5); ctx.lineTo(cx + 3.5, cy + 3.5);
          ctx.moveTo(cx - 3.5, cy + 3.5); ctx.lineTo(cx + 3.5, cy - 3.5);
          ctx.stroke();
        } else if (st === "u") {
          ctx.fillStyle = "rgba(139,148,158,0.55)";
          ctx.beginPath(); ctx.arc(cx, cy, 2.4, 0, 7); ctx.fill();
        } else {
          ctx.fillStyle = urgencyColor(parseInt(st, 10));
          ctx.strokeStyle = "#0b0e14"; ctx.lineWidth = 0.6;
          ctx.beginPath(); ctx.arc(cx, cy, 3.6, 0, 7); ctx.fill(); ctx.stroke();
        }
      }

      // base station
      const [bx, by] = this.worldToCanvas(meta.base[0], meta.base[1]);
      ctx.fillStyle = "gold";
      drawStar(ctx, bx, by, 7, 5);

      // drones: backhaul links first (under markers), then coverage disc + triangle
      const drones = frame.d;
      const byId = {};
      for (const d of drones) byId[d.id] = d;
      for (const d of drones) {
        const [dx, dy] = this.worldToCanvas(d.x, d.y);
        for (const nb of d.linked) {
          let other, otherConn;
          if (nb === "base") { other = [bx, by]; otherConn = true; }
          else if (byId[nb]) { other = this.worldToCanvas(byId[nb].x, byId[nb].y); otherConn = byId[nb].conn; }
          else continue;
          ctx.strokeStyle = backhaulEdgeColor(d.conn, otherConn);
          ctx.lineWidth = 1.3;
          ctx.setLineDash([5, 4]);
          ctx.beginPath(); ctx.moveTo(dx, dy); ctx.lineTo(other[0], other[1]); ctx.stroke();
          ctx.setLineDash([]);
        }
      }
      for (const d of drones) {
        const [dx, dy] = this.worldToCanvas(d.x, d.y);
        const color = d.conn ? "#3fb950" : "#f85149";
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.16;
        ctx.beginPath(); ctx.arc(dx, dy, meta.coverageRadius * this.scale, 0, 7); ctx.stroke();
        ctx.globalAlpha = 1;
        drawTriangle(ctx, dx, dy, 7, color);
      }

      // resources
      for (const r of frame.r) {
        const [rx, ry] = this.worldToCanvas(r.x, r.y);
        drawResource(ctx, rx, ry, RESOURCE_GLYPH[r.type] || "circle", r.status);
      }

      this.updateCounters(frameIdx);
      this.updateCallout(frameIdx);
    }

    updateCounters(frameIdx) {
      const rescued = this.data.cumRescued[frameIdx];
      const dead = this.data.cumDead[frameIdx];
      const atRisk = meta.totalPeople - rescued - dead;
      document.getElementById("n-rescued-" + this.runName).textContent = rescued;
      document.getElementById("n-dead-" + this.runName).textContent = dead;
      document.getElementById("n-atrisk-" + this.runName).textContent = atRisk;
    }

    updateCallout(frameIdx) {
      if (!divergence.found || divergence.run !== this.runName) { this.calloutEl.classList.remove("show"); return; }
      const window = 10; // stay visible for ~10 downsampled frames after the moment
      if (frameIdx < divergence.frameIdx || frameIdx > divergence.frameIdx + window) {
        this.calloutEl.classList.remove("show");
        return;
      }
      this.calloutEl.innerHTML = `<span class="tag">First divergence &middot; ${fmtClock(divergence.t)}</span>` +
        `${divergence.reason}`;
      const pct = 100 * (frameIdx >= divergence.frameIdx ? 1 : 0);
      this.calloutEl.style.left = "8%";
      this.calloutEl.style.top = "8%";
      this.calloutEl.classList.add("show");
    }
  }

  function drawStar(ctx, cx, cy, r, points) {
    ctx.beginPath();
    for (let i = 0; i < points * 2; i++) {
      const rad = i % 2 === 0 ? r : r / 2.4;
      const ang = (Math.PI / points) * i - Math.PI / 2;
      const x = cx + Math.cos(ang) * rad, y = cy + Math.sin(ang) * rad;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
  }

  function drawTriangle(ctx, cx, cy, r, color) {
    ctx.fillStyle = color;
    ctx.strokeStyle = "#0b0e14";
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    ctx.moveTo(cx, cy - r);
    ctx.lineTo(cx - r * 0.87, cy + r * 0.6);
    ctx.lineTo(cx + r * 0.87, cy + r * 0.6);
    ctx.closePath();
    ctx.fill(); ctx.stroke();
  }

  function drawResource(ctx, cx, cy, glyph, status) {
    const active = status !== "idle";
    ctx.fillStyle = active ? "#f1c40f" : "rgba(241,196,15,0.45)";
    ctx.strokeStyle = "#0b0e14";
    ctx.lineWidth = 0.7;
    const s = 5;
    if (glyph === "square") {
      ctx.beginPath(); ctx.rect(cx - s, cy - s, s * 2, s * 2); ctx.fill(); ctx.stroke();
    } else if (glyph === "plus") {
      ctx.fillRect(cx - s, cy - 1.6, s * 2, 3.2);
      ctx.fillRect(cx - 1.6, cy - s, 3.2, s * 2);
    } else if (glyph === "diamond") {
      ctx.beginPath();
      ctx.moveTo(cx, cy - s); ctx.lineTo(cx + s, cy); ctx.lineTo(cx, cy + s); ctx.lineTo(cx - s, cy);
      ctx.closePath(); ctx.fill(); ctx.stroke();
    } else {
      ctx.beginPath(); ctx.arc(cx, cy, s, 0, 7); ctx.fill(); ctx.stroke();
    }
  }

  // -- transport controls --------------------------------------------------
  const panels = { baseline: new Panel("baseline"), engine: new Panel("engine") };
  const scrub = document.getElementById("scrub");
  const clockEl = document.getElementById("clock");
  const playBtn = document.getElementById("playBtn");
  scrub.max = nFrames - 1;

  let frameIdx = 0, playing = false, speed = 1, lastTs = null, acc = 0;
  const FRAME_MS = 380; // base real-ms per downsampled frame at 1x

  function renderFrame(idx) {
    frameIdx = idx;
    panels.baseline.render(idx);
    panels.engine.render(idx);
    clockEl.textContent = fmtClock(runs.engine.frames[idx].t);
    scrub.value = idx;
  }

  function tick(ts) {
    if (!playing) { lastTs = null; return; }
    if (lastTs === null) lastTs = ts;
    acc += (ts - lastTs);
    lastTs = ts;
    const step = FRAME_MS / speed;
    let idx = frameIdx;
    let advanced = false;
    while (acc >= step) {
      acc -= step;
      idx++;
      advanced = true;
      if (idx >= nFrames) { idx = nFrames - 1; playing = false; playBtn.innerHTML = "&#9654; Play"; break; }
    }
    if (advanced) renderFrame(idx);
    if (playing) requestAnimationFrame(tick);
  }

  playBtn.addEventListener("click", () => {
    playing = !playing;
    playBtn.innerHTML = playing ? "&#10074;&#10074; Pause" : "&#9654; Play";
    if (playing) { lastTs = null; requestAnimationFrame(tick); }
  });

  scrub.addEventListener("input", () => { playing = false; playBtn.innerHTML = "&#9654; Play"; renderFrame(parseInt(scrub.value, 10)); });

  document.getElementById("speeds").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-speed]");
    if (!btn) return;
    speed = parseFloat(btn.dataset.speed);
    document.querySelectorAll("#speeds button").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
  });

  document.getElementById("jumpDiv").addEventListener("click", () => {
    if (!divergence.found) return;
    playing = false; playBtn.innerHTML = "&#9654; Play";
    renderFrame(divergence.frameIdx);
  });

  if (divergence.found) {
    const pct = 100 * divergence.frameIdx / (nFrames - 1);
    document.getElementById("divTick").style.left = pct + "%";
  } else {
    document.getElementById("divTick").style.display = "none";
  }

  renderFrame(0);
})();
</script>
</body>
</html>
"""


def main():
    print("Loading run_log.json...")
    log = logdata.load_log()
    print("Building downsampled player data...")
    data = build_player_data(log)
    payload = json.dumps(data, separators=(",", ":"))
    html = PAGE_TEMPLATE.replace("__PLAYER_DATA__", payload)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html, encoding="utf-8")
    size_kb = len(html.encode("utf-8")) / 1024
    print(f"Player written to {OUT_PATH} ({size_kb:.0f} KB, {data['meta']['nFrames']} frames/run)")
    if data["divergence"]["found"]:
        d = data["divergence"]
        print(f"Divergence: {d['run']} pulls ahead at T+{d['t']/3600:.2f}h via {d['resourceId']} ({d['resourceType']})")
    else:
        print("No divergence found (strategies tied throughout -- reporting plainly, not concealing it).")


if __name__ == "__main__":
    main()
