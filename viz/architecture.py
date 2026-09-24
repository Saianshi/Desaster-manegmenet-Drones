"""
Writes one static SVG architecture diagram to output/charts/architecture.svg:
the data path from a victim's phone to a dispatched responder. Pure
presentation -- no simulation or engine data, so it's hand-authored
rather than plotted, but lives under /viz and writes only to output/
like every other artifact here, for the same "everything traceable"
reason.

Run standalone with:  python -m viz.architecture
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "output" / "charts" / "architecture.svg"

BG = "#0d1117"
PANEL = "#161b22"
BORDER = "#30363d"
TEXT = "#e6edf3"
MUTED = "#8b949e"
BLUE = "#58a6ff"
GREEN = "#3fb950"
ORANGE = "#f0883e"
PURPLE = "#bc8cff"

NODES = [
    # (title, subtitle, x, y, w, h, accent)
    ("Victim BLE mesh", "phone/wearable beacon, no cell network needed", 20, 170, 210, 100, GREEN),
    ("Drone portable cell", "thermal + UWB + RF pick up the beacon", 270, 170, 210, 100, BLUE),
    ("Drone-to-drone backhaul", "multi-hop relay when no direct line to base", 520, 170, 230, 100, BLUE),
    ("Base station", "aggregates all drone traffic, fixed position", 790, 170, 200, 100, BLUE),
    ("Belief fusion", "engine/belief.py -- noisy detections -> one\nconfidence-weighted VictimBelief per victim", 1030, 170, 240, 100, PURPLE),
    ("Triage scheduler", "engine/scheduler.py -- survival-aware value,\ncascade bonus, rescuer risk, greedy assignment", 1310, 170, 250, 100, ORANGE),
    ("Responder view", "dispatched resource + human-readable\ndecision trace for the crew on scene", 1600, 170, 220, 100, GREEN),
]

WIDTH = 1860
HEIGHT = 420


def _node_svg(title, subtitle, x, y, w, h, accent):
    lines = subtitle.split("\n")
    sub_svg = "".join(
        f'<tspan x="{x + w/2}" dy="{"0" if i == 0 else "16"}">{ln}</tspan>'
        for i, ln in enumerate(lines)
    )
    return f'''
  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{PANEL}" stroke="{accent}" stroke-width="2"/>
  <rect x="{x}" y="{y}" width="{w}" height="6" rx="3" fill="{accent}"/>
  <text x="{x + w/2}" y="{y + 40}" text-anchor="middle" font-size="17" font-weight="700" fill="{TEXT}" font-family="Segoe UI, Arial, sans-serif">{title}</text>
  <text x="{x + w/2}" y="{y + 64}" text-anchor="middle" font-size="11.5" fill="{MUTED}" font-family="Segoe UI, Arial, sans-serif">{sub_svg}</text>
'''


def _arrow_svg(x1, y1, x2, y2, label=""):
    mid_x = (x1 + x2) / 2
    label_svg = (
        f'<text x="{mid_x}" y="{y1 - 10}" text-anchor="middle" font-size="11" fill="{MUTED}" '
        f'font-family="Segoe UI, Arial, sans-serif" font-style="italic">{label}</text>'
        if label else ""
    )
    return f'''
  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{MUTED}" stroke-width="2.5" marker-end="url(#arrow)"/>
  {label_svg}
'''


def build_svg() -> str:
    parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="100%" height="auto">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="{MUTED}"/>
    </marker>
  </defs>
  <rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="{BG}"/>
  <text x="{WIDTH/2}" y="45" text-anchor="middle" font-size="22" font-weight="700" fill="{TEXT}" font-family="Segoe UI, Arial, sans-serif">From a victim's phone to a dispatched rescuer -- the whole data path</text>
  <text x="{WIDTH/2}" y="72" text-anchor="middle" font-size="13" fill="{MUTED}" font-family="Segoe UI, Arial, sans-serif">Every hop after "Victim BLE mesh" is comms-denied and unreliable by design -- the engine only ever sees what survives this chain.</text>
''']

    for title, subtitle, x, y, w, h, accent in NODES:
        parts.append(_node_svg(title, subtitle, x, y, w, h, accent))

    for i in range(len(NODES) - 1):
        _, _, x1, y1, w1, h1, _ = NODES[i]
        _, _, x2, y2, w2, h2, _ = NODES[i + 1]
        parts.append(_arrow_svg(x1 + w1, y1 + h1 / 2, x2, y2 + h2 / 2))

    # feedback loop: responder view -> field report -> belief fusion (on-scene correction)
    fb_y = 170 + 100 + 55
    resp_x = NODES[6][2] + NODES[6][3] / 2
    fusion_x = NODES[4][2] + NODES[4][3] / 2
    parts.append(f'''
  <path d="M {resp_x} {170 + 100} C {resp_x} {fb_y}, {fusion_x} {fb_y}, {fusion_x} {170 + 100}"
        fill="none" stroke="{ORANGE}" stroke-width="2" stroke-dasharray="6,4" marker-end="url(#arrow)"/>
  <text x="{(resp_x + fusion_x) / 2}" y="{fb_y + 18}" text-anchor="middle" font-size="11.5" fill="{ORANGE}"
        font-family="Segoe UI, Arial, sans-serif" font-style="italic">field report on wrong-resource-type: crew radios back what they actually found -- treated as authoritative, overrides remote-sensor classification</text>
''')

    parts.append(f'''
  <text x="30" y="{HEIGHT - 15}" font-size="11" fill="{MUTED}" font-family="Segoe UI, Arial, sans-serif">/engine never imports /simulation and never sees ground truth -- only Detection objects that made it through this chain.</text>
</svg>''')
    return "".join(parts)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(build_svg(), encoding="utf-8")
    print(f"Architecture diagram written to {OUT_PATH}")


if __name__ == "__main__":
    main()
