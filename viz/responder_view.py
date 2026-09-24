"""
Generates a single static HTML page mocking up what a first responder at
the perimeter would see at a chosen timestamp: a map of detected
clusters and drone coverage, a ranked action list (team, target, ETA,
group size, survival window, required equipment, and the engine's own
`reason` string for each assignment), and a panel of suspected silent
zones. Reads output/run_log.json only -- no /engine or /simulation import.

Run standalone with:  python -m viz.responder_view [frame_index]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import params
from viz import logdata
from viz.animate import find_divergence_frame, _urgency_color

RUN_NAME = "engine"  # the responder is watching OUR engine's feed


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def render_map(log: dict, frame_idx: int, out_path: Path) -> None:
    entry = log["runs"][RUN_NAME]["entries"][frame_idx]
    map_size = log["meta"]["map_size_m"]
    water_level = entry["input"]["environment"]["water_level"]
    beliefs = entry.get("beliefs", [])
    silent_zones = entry["output"]["suspected_silent_zones"]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    for b in log["world_static"]["buildings"]:
        color = "#3a3f47" if b["collapsed"] else "#22262c"
        ax.add_patch(Rectangle((b["x0"], b["y0"]), b["x1"] - b["x0"], b["y1"] - b["y0"],
                                facecolor=color, edgecolor="none", zorder=1))

    elevation_map = np.array(log["world_static"]["elevation_map"])
    flood_mask = elevation_map <= water_level
    from matplotlib.colors import ListedColormap
    ax.imshow(np.where(flood_mask, 1.0, np.nan), extent=(0, map_size, 0, map_size), origin="lower",
              cmap=ListedColormap(["#1f6feb"]), vmin=0, vmax=1, alpha=0.35, zorder=2)

    for bel in beliefs:
        pos = bel["position"]
        color = _urgency_color(1 - bel["confidence"])  # low confidence rendered like "uncertain/red-ish" for visibility
        ax.scatter(*pos, s=90 + 12 * bel["est_group_size"], c=[color], edgecolors="white",
                   linewidths=0.8, zorder=5, alpha=0.9)
        ax.add_patch(Circle(pos, bel["uncertainty"], fill=False, edgecolor="#f0f0f0", alpha=0.3, linewidth=0.7, zorder=4))

    for x, y, risk in silent_zones:
        ax.scatter(x, y, marker="s", s=70, facecolors="none", edgecolors="#ff6b6b",
                   linewidths=1.6, zorder=6, alpha=0.5 + 0.5 * risk)

    drones = entry["input"]["drones"]
    has_backhaul = logdata.compute_backhaul_from_linked_to(drones)
    base_pos = tuple(params.BASE_STATION_POSITION)
    for d in drones:
        pos = d["position"]
        color = "#3fb950" if has_backhaul.get(d["drone_id"], False) else "#f85149"
        ax.add_patch(Circle(pos, params.RF_RANGE_M, fill=False, edgecolor=color, alpha=0.25, linewidth=0.9, zorder=3))
        ax.scatter(*pos, marker="^", s=110, c=color, edgecolors="white", linewidths=0.8, zorder=7)
    ax.scatter(*base_pos, marker="*", s=260, c="#ffd43b", edgecolors="white", linewidths=0.8, zorder=8)

    for r in entry["input"]["resources"]:
        marker = {"boat": "s", "excavator": "P", "medical": "D"}.get(r["type"], "o")
        ax.scatter(*r["position"], marker=marker, s=80, c="#ffd43b", edgecolors="#0d1117", linewidths=0.6, zorder=7)

    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=150, facecolor="#0d1117")
    plt.close(fig)


def _format_minutes(m: float) -> str:
    if m == float("inf"):
        return "no active deadline"
    if m < 60:
        return f"{m:.0f} min"
    return f"{m / 60:.1f} h"


def build_action_list(log: dict, frame_idx: int) -> list:
    """
    EngineOutput.assignments only lists dispatches issued THIS tick --
    most ticks, most resources are already busy from an earlier dispatch
    and so don't reappear there. A responder's action list needs the
    full current picture: what is every currently-busy resource actually
    doing right now. Reconstructed by scanning backward for the most
    recent assignment issued to each resource still non-idle now.
    """
    entries = log["runs"][RUN_NAME]["entries"]
    entry = entries[frame_idx]
    resources_by_id = {r["resource_id"]: r for r in entry["input"]["resources"]}
    beliefs_by_id = {b["belief_id"]: b for b in entry.get("beliefs", [])}

    active_assignment = {}
    for j in range(frame_idx, -1, -1):
        for a in entries[j]["output"]["assignments"]:
            rid = a["resource_id"]
            if rid not in active_assignment:
                active_assignment[rid] = a
        if len(active_assignment) >= len(resources_by_id):
            break

    rows = []
    for rid, resource in resources_by_id.items():
        if resource["status"] == "idle":
            continue
        a = active_assignment.get(rid)
        if a is None:
            continue
        belief = beliefs_by_id.get(a["target_belief_id"])
        if belief is None:
            continue  # belief has since decayed/pruned -- skip rather than guess at its state
        eta_min = _dist(resource["position"], belief["position"]) / max(resource["speed"], 1e-6)
        rows.append({
            "team": f"{resource['type'].capitalize()} {resource['resource_id']}",
            "target": belief["belief_id"],
            "eta_min": eta_min,
            "group_size": belief["est_group_size"],
            "survival_window": _format_minutes(belief["survival_deadline"]),
            "equipment": ", ".join(belief["required_resources"]) or "none",
            "confidence": belief["confidence"],
            "env_type": belief["env_type"],
            "reason": a["reason"],
            "value": a["expected_value"],
        })

    # A buried rescue is excavator+medical dispatched TOGETHER (see
    # scheduler.py) -- two Assignment records, one logical mission. Two
    # rows here always share the same target_belief_id in that case
    # (single-resource dispatches never share a target, by construction
    # of schedule()'s used_belief_ids exclusivity), so merging on target
    # is safe and turns the duplicate pair into one "Team A + Team B" row.
    merged: dict = {}
    order = []
    for row in rows:
        key = row["target"]
        if key in merged:
            merged[key]["team"] = f"{merged[key]['team']} + {row['team']}"
        else:
            merged[key] = row
            order.append(key)
    rows = [merged[k] for k in order]
    rows.sort(key=lambda r: -r["value"])
    return rows


ENV_ICON = {"buried": "\U0001F3DA", "rooftop": "\U0001F30A", "street": "\U0001F6B6", "unknown": "❓"}


def render_html(log: dict, frame_idx: int, action_rows: list, out_path: Path, map_rel_path: str) -> None:
    entry = log["runs"][RUN_NAME]["entries"][frame_idx]
    t = entry["input"]["t"]
    h, m = int(t // 3600), int((t % 3600) // 60)
    silent_zones = sorted(entry["output"]["suspected_silent_zones"], key=lambda z: -z[2])

    rescued = log["runs"][RUN_NAME]["ground_truth"]["timesteps"]
    group_size = logdata.group_size_lookup(log)
    rescued_ids = logdata.rescued_ids_by_frame(log, RUN_NAME, frame_idx + 1)[-1]
    gt = log["runs"][RUN_NAME]["ground_truth"]["timesteps"][frame_idx]
    n_alive = sum(group_size.get(v["victim_id"], 1) for v in gt["victims"] if v["alive"])
    n_dead = sum(group_size.get(v["victim_id"], 1) for v in gt["victims"] if not v["alive"])
    n_rescued = sum(group_size.get(vid, 1) for vid in rescued_ids)

    rows_html = ""
    for i, r in enumerate(action_rows, 1):
        icon = ENV_ICON.get(r["env_type"], "?")
        rows_html += f"""
        <tr>
          <td class="rank">{i}</td>
          <td class="team">{r['team']}</td>
          <td>{icon} {r['target']}</td>
          <td>{r['eta_min']:.0f} min</td>
          <td>{r['group_size']}</td>
          <td class="{'urgent' if r['survival_window'] != 'no active deadline' and 'min' in r['survival_window'] and float(r['survival_window'].split()[0]) < 30 else ''}">{r['survival_window']}</td>
          <td>{r['equipment']}</td>
          <td>{r['confidence']*100:.0f}%</td>
        </tr>
        <tr class="reason-row"><td></td><td colspan="7" class="reason">&#8618; {r['reason']}</td></tr>
        """

    zones_html = ""
    for x, y, risk in silent_zones[:12]:
        bar_width = int(risk * 100)
        zones_html += f"""
        <div class="zone-row">
          <span class="zone-pos">({x:.0f}, {y:.0f})</span>
          <div class="zone-bar-bg"><div class="zone-bar" style="width:{bar_width}%"></div></div>
          <span class="zone-risk">{risk:.2f}</span>
        </div>
        """

    html = f"""<meta charset="utf-8">
<title>Perimeter Ops View</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ background:#0d1117; color:#c9d1d9; font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin:0; padding:20px; }}
  .header {{ display:flex; justify-content:space-between; align-items:center; border-bottom:2px solid #30363d; padding-bottom:14px; margin-bottom:18px; }}
  .header h1 {{ font-size:20px; margin:0; letter-spacing:0.5px; color:#e6edf3; }}
  .clock {{ font-size:28px; font-weight:700; color:#58a6ff; font-variant-numeric:tabular-nums; }}
  .stat-row {{ display:flex; gap:18px; }}
  .stat {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:8px 16px; text-align:center; min-width:90px; }}
  .stat .n {{ font-size:22px; font-weight:700; }}
  .stat .l {{ font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:0.5px; }}
  .stat.rescued .n {{ color:#3fb950; }}
  .stat.dead .n {{ color:#f85149; }}
  .stat.alive .n {{ color:#d29922; }}
  .grid {{ display:grid; grid-template-columns:520px 1fr; gap:20px; }}
  .panel {{ background:#161b22; border:1px solid #30363d; border-radius:10px; padding:16px; }}
  .panel h2 {{ font-size:13px; text-transform:uppercase; letter-spacing:0.8px; color:#8b949e; margin:0 0 12px 0; }}
  .map-img {{ width:100%; border-radius:6px; display:block; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ text-align:left; color:#8b949e; font-size:11px; text-transform:uppercase; letter-spacing:0.4px; padding:6px 8px; border-bottom:1px solid #30363d; }}
  td {{ padding:7px 8px; border-bottom:1px solid #1c2128; }}
  td.rank {{ color:#58a6ff; font-weight:700; }}
  td.team {{ font-weight:600; color:#e6edf3; }}
  td.urgent {{ color:#f85149; font-weight:700; }}
  tr.reason-row td.reason {{ color:#8b949e; font-size:11.5px; font-style:italic; padding:0 8px 10px 8px; border-bottom:1px solid #1c2128; }}
  .zone-row {{ display:flex; align-items:center; gap:10px; margin-bottom:8px; font-size:12.5px; }}
  .zone-pos {{ width:110px; color:#8b949e; font-variant-numeric:tabular-nums; }}
  .zone-bar-bg {{ flex:1; height:10px; background:#21262d; border-radius:5px; overflow:hidden; }}
  .zone-bar {{ height:100%; background:linear-gradient(90deg,#f0883e,#f85149); }}
  .zone-risk {{ width:36px; text-align:right; color:#f85149; font-weight:600; }}
  .legend {{ font-size:11px; color:#8b949e; margin-top:10px; line-height:1.6; }}
  .footer-note {{ font-size:11px; color:#6e7681; margin-top:18px; text-align:center; }}
</style>
<div class="header">
  <div>
    <h1>DISASTER RESPONSE - PERIMETER OPERATIONS VIEW</h1>
    <div class="stat-row" style="margin-top:10px;">
      <div class="stat rescued"><div class="n">{n_rescued}</div><div class="l">Rescued</div></div>
      <div class="stat alive"><div class="n">{n_alive}</div><div class="l">At risk</div></div>
      <div class="stat dead"><div class="n">{n_dead}</div><div class="l">Lost</div></div>
    </div>
  </div>
  <div class="clock">T+{h:02d}:{m:02d}</div>
</div>
<div class="grid">
  <div class="panel">
    <h2>Live Situational Map</h2>
    <img class="map-img" src="{map_rel_path}" alt="situational map">
    <div class="legend">
      &#9650; drone (green = has backhaul, red = disconnected) &nbsp;&bull;&nbsp;
      &#9733; base station &nbsp;&bull;&nbsp; &#9632;&#9670;&#10010; resource units<br>
      Coloured dot = detected cluster (colour = confidence) &nbsp;&bull;&nbsp;
      Red square = suspected silent zone &nbsp;&bull;&nbsp; Blue tint = flooded
    </div>
  </div>
  <div class="panel">
    <h2>Ranked Action List</h2>
    <table>
      <tr><th>#</th><th>Team</th><th>Target</th><th>ETA</th><th>Grp</th><th>Survival Window</th><th>Equipment</th><th>Conf.</th></tr>
      {rows_html if rows_html else '<tr><td colspan="8" style="color:#8b949e;">No active assignments this tick.</td></tr>'}
    </table>
  </div>
</div>
<div class="grid" style="grid-template-columns:520px 1fr; margin-top:20px;">
  <div class="panel">
    <h2>Suspected Silent Zones ({len(silent_zones)})</h2>
    {zones_html if zones_html else '<div style="color:#8b949e;font-size:12px;">None flagged this tick.</div>'}
    <div class="legend">High prior population, scanned, zero detections: either empty, or everyone there is too trapped to signal.</div>
  </div>
  <div class="panel">
    <h2>Notes</h2>
    <div class="legend" style="font-size:12px; line-height:1.8;">
      Icons: {ENV_ICON['buried']} buried &nbsp; {ENV_ICON['rooftop']} rooftop/flood &nbsp; {ENV_ICON['street']} street<br>
      "Reason" lines are the engine's own comparative justification for each dispatch, generated at decision time -- not added after the fact.<br>
      This view reflects the engine's OWN belief state: noisy, uncertain, and occasionally wrong. Ground truth is never shown here, matching what the engine itself has to work with.
    </div>
  </div>
</div>
<div class="footer-note">Generated by viz/responder_view.py from output/run_log.json, frame {frame_idx}, seed {log['meta']['seed']}</div>
"""
    out_path.write_text(html, encoding="utf-8")


def main():
    log = logdata.load_log()
    if len(sys.argv) > 1:
        frame_idx = int(sys.argv[1])
    else:
        frame_idx = find_divergence_frame(log)
        print(f"No frame given -- using the divergence frame ({frame_idx}) as a representative, interesting moment.")

    out_dir = ROOT / "output"
    map_path = out_dir / "charts" / "responder_view_map.png"
    map_path.parent.mkdir(parents=True, exist_ok=True)
    print("Rendering map panel...")
    render_map(log, frame_idx, map_path)

    print("Building ranked action list...")
    rows = build_action_list(log, frame_idx)

    html_path = out_dir / "responder_view.html"
    render_html(log, frame_idx, rows, html_path, map_rel_path="charts/responder_view_map.png")
    print(f"Responder view written to {html_path} (frame {frame_idx}, t={log['runs']['engine']['entries'][frame_idx]['input']['t']/3600:.2f}h)")


if __name__ == "__main__":
    main()
