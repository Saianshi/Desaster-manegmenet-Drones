"""
Human-readable chronological trace of the most consequential decisions
in the engine's run: every successful rescue with its own dispatch
reason, and any "cascade" pattern where a rescue was followed shortly
after by nearby new detections. Reads output/run_log.json only -- no
/engine or /simulation import.

Output: output/decision_trace.md and output/decision_trace.pdf.

Run standalone with:  python -m viz.decision_trace
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from viz import logdata

CASCADE_WINDOW_S = 30 * 60  # look for new detections within this long after a rescue
CASCADE_RADIUS_M = 300.0    # matches engine.config's ENGINE_CASCADE_RADIUS_M / DRONE_CELL_RANGE_M
                             # -- restated here (not imported) so /viz stays decoupled from /engine


def find_assignment_for_resource(log: dict, run_name: str, frame_idx: int, resource_id: str):
    entries = log["runs"][run_name]["entries"]
    for j in range(frame_idx, -1, -1):
        for a in entries[j]["output"]["assignments"]:
            if a["resource_id"] == resource_id:
                return a
    return None


def rescue_narrative(log: dict, run_name: str = "engine") -> list:
    events = log["runs"][run_name]["ground_truth"]["rescue_events"]
    group_size = logdata.group_size_lookup(log)
    env_type = logdata.env_type_lookup(log)
    entries = log["runs"][run_name]["entries"]
    dt = log["meta"]["dt"]
    n = len(entries)

    narrative = []
    for e in sorted((ev for ev in events if ev["outcome"] == "rescued"), key=lambda x: x["t"]):
        frame_idx = min(int(e["t"] // dt), n - 1)
        a = find_assignment_for_resource(log, run_name, frame_idx, e["resource_id"])
        vid = e["victim_id"]
        narrative.append({
            "t": e["t"],
            "victim_id": vid,
            "group_size": group_size.get(vid, 1),
            "env_type": env_type.get(vid, "unknown"),
            "resource_id": e["resource_id"],
            "reason": a["reason"] if a else "(dispatch record not found -- resource may have been reassigned)",
        })
    return narrative


def find_cascade_events(log: dict, run_name: str = "engine") -> list:
    """A rescue is flagged if, within CASCADE_WINDOW_S afterward, one or
    more previously-undetected victims got their first detection within
    CASCADE_RADIUS_M of the rescue site. Proximity + timing only -- not
    a proven causal link (a drone could have been passing through
    anyway), and reported with that caveat."""
    positions = logdata.victim_true_positions(log, run_name)
    first_detected = logdata.first_detected_times(log, run_name)
    rescues = [e for e in log["runs"][run_name]["ground_truth"]["rescue_events"] if e["outcome"] == "rescued"]

    cascades = []
    for r in rescues:
        rescue_pos = positions.get(r["victim_id"])
        if rescue_pos is None:
            continue
        revealed = [
            vid for vid, dt_first in first_detected.items()
            if vid != r["victim_id"] and r["t"] < dt_first <= r["t"] + CASCADE_WINDOW_S
            and logdata.dist(positions[vid], rescue_pos) <= CASCADE_RADIUS_M
        ]
        if revealed:
            cascades.append({"t": r["t"], "trigger_victim": r["victim_id"], "revealed": revealed})
    return cascades


def _clock(t: float) -> str:
    h, m = int(t // 3600), int((t % 3600) // 60)
    return f"T+{h:02d}:{m:02d}"


def build_markdown(log: dict) -> str:
    seed = log["meta"]["seed"]
    group_size = logdata.group_size_lookup(log)
    total_people = sum(group_size.values())

    engine_rescued = [e for e in log["runs"]["engine"]["ground_truth"]["rescue_events"] if e["outcome"] == "rescued"]
    baseline_rescued = [e for e in log["runs"]["baseline"]["ground_truth"]["rescue_events"] if e["outcome"] == "rescued"]
    engine_people = sum(group_size.get(e["victim_id"], 1) for e in engine_rescued)
    baseline_people = sum(group_size.get(e["victim_id"], 1) for e in baseline_rescued)

    lines = [
        f"# Decision Trace -- seed {seed}",
        "",
        f"Scenario: {total_people} people across {len(group_size)} victim groups, 8-hour response window.",
        "",
        f"**Outcome**: engine rescued **{engine_people}** people ({len(engine_rescued)} missions); "
        f"baseline (nearest-first) rescued **{baseline_people}** people ({len(baseline_rescued)} missions).",
        "",
    ]
    if baseline_people == 0:
        lines += [
            "> Note: this specific seed is an extreme case for the baseline. Across the full "
            "30-seed evaluation (Phase 7) the baseline's mean was around 4-5 people rescued, not "
            "literally zero -- zero here reflects this one run, not the general finding.",
            "",
        ]

    lines.append("## Engine rescue timeline")
    lines.append("")
    narrative = rescue_narrative(log, "engine")
    if not narrative:
        lines.append("_No successful rescues this run._")
    for n in narrative:
        person_word = "person" if n["group_size"] == 1 else "people"
        lines.append(f"**{_clock(n['t'])}** -- rescued {n['group_size']} {person_word} ({n['env_type']}) via {n['resource_id']}.")
        lines.append(f"> {n['reason']}")
        lines.append("")

    lines.append("## Possible cascade events")
    lines.append("")
    lines.append(f"_A rescue is flagged here if, within {CASCADE_WINDOW_S // 60} minutes afterward, one or "
                  f"more previously-undetected victims were first detected within {CASCADE_RADIUS_M:.0f}m of "
                  f"the rescue site. This is a proximity/timing heuristic, not a proven causal link -- a drone "
                  f"could simply have been passing through anyway._")
    lines.append("")
    cascades = find_cascade_events(log, "engine")
    if not cascades:
        lines.append("_No cascade-pattern events detected this run._")
        lines.append("")
    for c in cascades:
        lines.append(f"**{_clock(c['t'])}** -- rescue near `{c['trigger_victim']}` preceded first detection of "
                      f"{len(c['revealed'])} new victim(s) nearby within {CASCADE_WINDOW_S // 60} min.")
        lines.append("")

    lines.append("## Baseline attempts (for contrast)")
    lines.append("")
    total_baseline_missions = len(log["runs"]["baseline"]["ground_truth"]["rescue_events"])
    lines.append(f"Baseline made {total_baseline_missions} total mission attempts, {len(baseline_rescued)} successful "
                 f"({100 * len(baseline_rescued) / max(total_baseline_missions, 1):.1f}%).")
    lines.append("")

    return "\n".join(lines)


def _strip_inline_markdown(text: str) -> str:
    return text.replace("**", "").replace("`", "")


def markdown_to_pdf(md_text: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    wrapped = []
    for line in md_text.split("\n"):
        line = _strip_inline_markdown(line)
        if line.startswith("_") and line.endswith("_") and len(line) > 1:
            line = line[1:-1]  # whole-line italic emphasis, e.g. "_no events this run_"
        if line.startswith("# "):
            wrapped.append(("title", line[2:]))
        elif line.startswith("## "):
            wrapped.append(("heading", line[3:]))
        elif line.startswith("> "):
            for wl in textwrap.wrap(line[2:], 95) or [""]:
                wrapped.append(("quote", wl))
        elif line.strip() == "":
            wrapped.append(("blank", ""))
        else:
            for wl in textwrap.wrap(line, 100) or [""]:
                wrapped.append(("body", wl))

    LINES_PER_PAGE = 46
    with PdfPages(out_path) as pdf:
        page_lines = []

        def flush():
            if not page_lines:
                return
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")
            y = 0.97
            for kind, text in page_lines:
                if kind == "title":
                    ax.text(0.06, y, text, fontsize=18, fontweight="bold", va="top", transform=ax.transAxes)
                    y -= 0.045
                elif kind == "heading":
                    ax.text(0.06, y, text, fontsize=14, fontweight="bold", va="top", transform=ax.transAxes, color="#1E6FB5")
                    y -= 0.035
                elif kind == "quote":
                    ax.text(0.09, y, text, fontsize=9.5, style="italic", va="top", transform=ax.transAxes, color="#555555")
                    y -= 0.022
                elif kind == "blank":
                    y -= 0.015
                else:
                    ax.text(0.06, y, text, fontsize=10.5, va="top", transform=ax.transAxes)
                    y -= 0.022
            pdf.savefig(fig)
            plt.close(fig)

        for item in wrapped:
            page_lines.append(item)
            if len(page_lines) >= LINES_PER_PAGE:
                flush()
                page_lines = []
        flush()


def main():
    log = logdata.load_log()
    md = build_markdown(log)

    out_md = ROOT / "output" / "decision_trace.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"Markdown written to {out_md}")

    out_pdf = ROOT / "output" / "decision_trace.pdf"
    markdown_to_pdf(md, out_pdf)
    print(f"PDF written to {out_pdf}")


if __name__ == "__main__":
    main()
