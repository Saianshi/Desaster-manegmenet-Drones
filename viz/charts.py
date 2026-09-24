"""
Regenerates all 6 presentation charts at 300 DPI into output/charts/.
Reads output/results.json (Phase 7 ablation/sensitivity) and
output/run_log.json (coverage curve, network graph). The one exception
to "log-only": the no-rescue mortality curve is inherently a simulation
question (what happens with zero intervention), so it re-runs a short,
read-only no-rescue simulation via the EXISTING, unmodified
simulation.world/victims machinery -- same as
simulation/phase2_checkpoint.py already did, just refreshed at 300 DPI
here. This does not change /engine or /simulation, it only calls them.

Dark theme matches docs/index.html's palette (bg #0d1117, panel #161b22,
accent #58a6ff) so charts don't look like a separate paper figure pasted
under the site. Each chart carries one direct on-figure annotation of
the specific number a reader should take away -- the same number stated
in the chart's caption in docs/index.html, not a separate claim.

Run standalone with:  python -m viz.charts
"""

from __future__ import annotations

import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from viz import logdata

DPI = 300
OUT_DIR = ROOT / "output" / "charts"

# -- dark theme, matching docs/index.html -----------------------------------
BG = "#0d1117"
PANEL = "#161b22"
GRID = "#30363d"
TEXT = "#e6edf3"
MUTED = "#8b949e"
BLUE = "#58a6ff"      # our engine
RED = "#f85149"       # pure baseline
ORANGE = "#f0883e"    # informed operator
GREEN = "#3fb950"     # positive / connected
PURPLE = "#bc8cff"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT,
    "text.color": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "legend.facecolor": PANEL,
    "legend.edgecolor": GRID,
    "legend.labelcolor": TEXT,
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
})


def _savefig(fig, out_path: Path) -> None:
    fig.savefig(out_path, dpi=DPI, facecolor=BG)
    plt.close(fig)


def _paired_stats(results: dict, cond_a: str, cond_b: str) -> dict:
    """Same-seed paired difference stats between two ablation conditions
    (matches evaluation's own paired analysis -- not a separate metric)."""
    ab = results["ablation"]
    a, b = ab[cond_a], ab[cond_b]
    map_a = dict(zip(a["seeds"], a["raw_rescued_people"]))
    map_b = dict(zip(b["seeds"], b["raw_rescued_people"]))
    common = sorted(set(map_a) & set(map_b))
    diffs = [map_a[s] - map_b[s] for s in common]
    n = len(diffs)
    mean_diff = statistics.mean(diffs)
    std_diff = statistics.stdev(diffs) if n > 1 else 0.0
    se = std_diff / math.sqrt(n) if n > 0 else float("nan")
    t = mean_diff / se if se > 0 else float("nan")
    return {"n": n, "mean_diff": mean_diff, "t": t}


def chart_lives_saved(results: dict, out_path: Path) -> None:
    ablation = results["ablation"]
    order = ["pure_baseline", "informed_operator", "full_engine"]
    labels = ["Baseline\n(nearest-first)", "Informed\noperator", "Our engine"]
    colors = [RED, ORANGE, BLUE]
    means = [ablation[k]["rescued_people_mean"] for k in order]
    stds = [ablation[k]["rescued_people_std"] for k in order]
    n = results["config"]["n_seeds"]
    ratio = means[2] / means[0] if means[0] else float("inf")

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    bars = ax.bar(labels, means, yerr=stds, capsize=8, color=colors, edgecolor=BG, linewidth=1.2, width=0.6)
    ax.set_ylabel("People rescued (mean ± std)")
    ax.set_title(f"Lives saved: baseline vs our engine\n(N={n} seeds, identical scenarios)")
    ax.grid(axis="y", alpha=0.25)
    y_top = max(m + s for m, s in zip(means, stds))
    ax.set_ylim(0, y_top * 1.2)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + y_top * 0.03, f"{m:.1f}", ha="center", fontsize=12, fontweight="bold", color=TEXT)

    # direct annotation: the headline ratio, as an arrow spanning baseline -> engine
    y_arrow = max(means[0], means[2]) * 0.55
    ax.annotate("", xy=(2, y_arrow), xytext=(0, y_arrow),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=2))
    ax.text(1, y_arrow + max(means) * 0.03, f"~{ratio:.0f}x", ha="center", fontsize=15,
            fontweight="bold", color=GREEN)
    fig.tight_layout()
    _savefig(fig, out_path)


def chart_ablation(results: dict, out_path: Path) -> None:
    ablation = results["ablation"]
    order = ["full_engine", "minus_cascade_bonus", "minus_silence_as_signal", "minus_rescuer_risk", "informed_operator", "pure_baseline"]
    labels = ["Full\nengine", "− cascade\nbonus", "− silence-\nas-signal", "− rescuer\nrisk", "Informed\noperator", "Pure\nbaseline"]
    colors = [BLUE, MUTED, MUTED, MUTED, ORANGE, RED]
    means = [ablation[k]["rescued_people_mean"] for k in order]
    stds = [ablation[k]["rescued_people_std"] for k in order]
    n = results["config"]["n_seeds"]

    stats = {
        "minus_cascade_bonus": _paired_stats(results, "full_engine", "minus_cascade_bonus"),
        "minus_silence_as_signal": _paired_stats(results, "full_engine", "minus_silence_as_signal"),
        "minus_rescuer_risk": _paired_stats(results, "full_engine", "minus_rescuer_risk"),
    }
    verdict = {
        "minus_cascade_bonus": "null",
        "minus_silence_as_signal": "null",
        "minus_rescuer_risk": "borderline",
    }
    verdict_color = {"null": MUTED, "borderline": ORANGE}

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(labels, means, yerr=stds, capsize=6, color=colors, edgecolor=BG, linewidth=1.0)
    ax.set_ylabel(f"People rescued (mean ± std, N={n} seeds)")
    ax.set_title("Ablation: contribution of each engine feature (paired t-test vs full engine)")
    ax.grid(axis="y", alpha=0.25)

    y_top = max(m + s for m, s in zip(means, stds))
    ax.set_ylim(0, y_top * 1.22)
    for i, key in enumerate(["minus_cascade_bonus", "minus_silence_as_signal", "minus_rescuer_risk"], start=1):
        s = stats[key]
        v = verdict[key]
        ax.text(i, means[i] + stds[i] + y_top * 0.04, f"t={s['t']:+.2f}\n({v})",
                ha="center", fontsize=10.5, fontweight="bold", color=verdict_color[v])
    fig.tight_layout()
    _savefig(fig, out_path)


def chart_sensitivity(results: dict, out_path: Path) -> None:
    sensitivity = results["sensitivity"]
    resource_counts = results["config"]["resource_counts"]
    n = results["config"]["sensitivity_n_seeds"]

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    series = {}
    for strat_name, color, label in [
        ("full_engine", BLUE, "Our engine"),
        ("informed_operator", ORANGE, "Informed operator"),
        ("pure_baseline", RED, "Pure baseline (nearest-first)"),
    ]:
        means = [sensitivity[strat_name][str(c)]["rescued_people_mean"] for c in resource_counts]
        stds = [sensitivity[strat_name][str(c)]["rescued_people_std"] for c in resource_counts]
        series[strat_name] = means
        ax.errorbar(resource_counts, means, yerr=stds, marker="o", capsize=5, color=color, label=label, linewidth=2.4, markersize=6)
    ax.set_xlabel("Number of resources (scarce → plentiful)")
    ax.set_ylabel(f"People rescued (mean ± std, N={n} seeds)")
    ax.set_title("Sensitivity: rescue advantage vs resource scarcity")
    ax.legend()
    ax.grid(alpha=0.25)

    # annotate the widest engine-vs-baseline gap directly on the chart
    gaps = [e - b for e, b in zip(series["full_engine"], series["pure_baseline"])]
    idx = int(np.argmax(gaps))
    x0 = resource_counts[idx]
    y_e, y_b = series["full_engine"][idx], series["pure_baseline"][idx]
    ax.annotate("", xy=(x0, y_e), xytext=(x0, y_b), arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.8))
    ax.text(x0 + max(resource_counts) * 0.02, (y_e + y_b) / 2, f"widest gap\n+{gaps[idx]:.0f} at N={x0}",
            fontsize=10, color=GREEN, fontweight="bold", va="center")
    fig.tight_layout()
    _savefig(fig, out_path)


def chart_coverage_curve(log: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    total_victims = len(log["victims_static"])
    curves = {}
    for run_name, color, label in [("engine", BLUE, "Our engine"), ("baseline", RED, "Baseline (lawnmower + nearest-first)")]:
        first_detected = logdata.first_detected_times(log, run_name)
        times = sorted(t for t in first_detected.values() if t < float("inf"))
        t_hours = [t / 3600.0 for t in times]
        cum_frac = [100 * (i + 1) / total_victims for i in range(len(times))]
        T = log["meta"]["T"] / 3600.0
        t_hours = [0.0] + t_hours + [T]
        cum_frac = [0.0] + cum_frac + [cum_frac[-1] if cum_frac else 0.0]
        curves[run_name] = (t_hours, cum_frac)
        ax.step(t_hours, cum_frac, where="post", color=color, linewidth=2.4, label=label)
    ax.set_xlabel("Time since disaster (hours)")
    ax.set_ylabel("Victim groups discovered (%)")
    ax.set_title("Discovery coverage over time: our engine vs naive lawnmower")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)

    # annotate time-to-90%-coverage for the engine directly on the chart
    et, ef = curves["engine"]
    target = 90.0
    t90 = next((t for t, f in zip(et, ef) if f >= target), None)
    if t90 is not None:
        ax.axhline(target, color=MUTED, linestyle=":", linewidth=1)
        ax.axvline(t90, color=BLUE, linestyle=":", linewidth=1)
        ax.plot([t90], [target], "o", color=BLUE, markersize=7, zorder=5)
        ax.text(t90 + 0.15, target - 8, f"engine: 90% found\nby T+{t90:.1f}h", color=BLUE, fontsize=10, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, out_path)


def chart_mortality_curve(out_path: Path) -> None:
    """The one chart that's inherently a fresh-simulation question (what
    happens with zero intervention) rather than something derivable from
    an already-strategy-driven run_log.json. Calls simulation.world /
    simulation.victims exactly as simulation/phase2_checkpoint.py does --
    read-only use of existing, unmodified code, not a change to it."""
    from config import params as sim_params
    from simulation.world import World

    rng = random.Random(sim_params.RANDOM_SEED)
    world = World(rng)

    n_steps = int(sim_params.T_TOTAL_SECONDS // sim_params.DT_SECONDS)
    t_hours, alive_people = [], {"buried": [], "rooftop": [], "street": []}
    for _ in range(n_steps):
        world.step(sim_params.DT_SECONDS)
        t_hours.append(world.t / 3600.0)
        counts = Counter()
        for v in world.victims:
            if v.alive:
                counts[v.env_type] += v.group_size
        for kind in ("buried", "rooftop", "street"):
            alive_people[kind].append(counts[kind])

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    colors = {"buried": "#d29922", "rooftop": BLUE, "street": GREEN}
    labels = {
        "buried": "Buried (crush syndrome, hours-scale decline)",
        "rooftop": "Rooftop in flood zone (steep drop as water arrives)",
        "street": "Street level (no active hazard modelled)",
    }
    for kind in ("buried", "rooftop", "street"):
        ax.plot(t_hours, alive_people[kind], label=labels[kind], color=colors[kind], linewidth=2.4)
    ax.set_xlabel("Time since disaster (hours)")
    ax.set_ylabel("People still alive (no rescue)")
    ax.set_title("No-rescue mortality curves by victim population\n(the two populations die on very different timescales)")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.25)

    # annotate the steepest point of the rooftop curve = flood arrival
    rooftop = np.array(alive_people["rooftop"])
    if len(rooftop) > 2:
        d = np.diff(rooftop)
        steepest = int(np.argmin(d))
        ax.axvline(t_hours[steepest], color=BLUE, linestyle=":", linewidth=1)
        ax.annotate("flood water\narrives", xy=(t_hours[steepest], rooftop[steepest]),
                    xytext=(t_hours[steepest] + 0.4, rooftop[steepest] + max(rooftop) * 0.12),
                    color=BLUE, fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5))
    fig.tight_layout()
    _savefig(fig, out_path)


def chart_network_graph(log: dict, out_path: Path) -> None:
    """Detected-cluster + drone-backhaul network, early vs late in the
    run (the engine run) -- log-driven reinterpretation of "mesh clusters
    before/after coverage": shows how many detected clusters exist and
    how connected the drone relay chain is, at two points in time."""
    from matplotlib.patches import Rectangle
    from config import params as sim_params

    entries = log["runs"]["engine"]["entries"]
    n = len(entries)
    early_idx, late_idx = n // 12, n - 1  # ~40 min in vs end of run
    map_size = log["meta"]["map_size_m"]
    base_pos = tuple(sim_params.BASE_STATION_POSITION)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    conn_pct = {}
    for ax, idx, title in zip(axes, (early_idx, late_idx), ("Early (~40 min in)", f"Late (T+{entries[late_idx]['input']['t']/3600:.1f}h)")):
        entry = entries[idx]
        for b in log["world_static"]["buildings"]:
            color = "#30363d" if b["collapsed"] else "#21262d"
            ax.add_patch(Rectangle((b["x0"], b["y0"]), b["x1"] - b["x0"], b["y1"] - b["y0"], facecolor=color, edgecolor="none", alpha=0.8, zorder=1))

        beliefs = entry.get("beliefs", [])
        for bel in beliefs:
            ax.scatter(*bel["position"], s=30 + 8 * bel["est_group_size"], c=ORANGE, edgecolors=BG, linewidths=0.5, alpha=0.9, zorder=4)

        drones = entry["input"]["drones"]
        has_backhaul = logdata.compute_backhaul_from_linked_to(drones)
        for d in drones:
            color = GREEN if has_backhaul.get(d["drone_id"], False) else RED
            ax.scatter(*d["position"], marker="^", s=120, c=color, edgecolors=BG, linewidths=0.8, zorder=6)
            for nb in d["linked_to"]:
                other = base_pos if nb == "base" else next((x["position"] for x in drones if x["drone_id"] == nb), None)
                if other is not None:
                    ax.plot([d["position"][0], other[0]], [d["position"][1], other[1]], linestyle="--", color=MUTED, linewidth=1.1, alpha=0.7, zorder=3)
        ax.scatter(*base_pos, marker="*", s=280, c="gold", edgecolors=BG, linewidths=0.8, zorder=7)

        n_connected = sum(has_backhaul.values())
        conn_pct[title] = 100 * n_connected / len(drones) if drones else 0
        ax.set_title(f"{title}\n{len(beliefs)} clusters CURRENTLY tracked, {n_connected}/{len(drones)} drones connected", fontsize=12)
        ax.set_xlim(0, map_size)
        ax.set_ylim(0, map_size)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Detected clusters & drone backhaul network: early vs late in the run\n"
                 "(cluster count can DROP over time -- unconfirmed clusters decay and are pruned if never re-detected)",
                 fontsize=12, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _savefig(fig, out_path)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading results.json and run_log.json...")
    results = logdata.load_results()
    log = logdata.load_log()

    print("Chart 1/6: lives saved, baseline vs ours...")
    chart_lives_saved(results, OUT_DIR / "lives_saved_baseline_vs_ours.png")

    print("Chart 2/6: ablation...")
    chart_ablation(results, OUT_DIR / "ablation.png")

    print("Chart 3/6: sensitivity...")
    chart_sensitivity(results, OUT_DIR / "sensitivity.png")

    print("Chart 4/6: discovery coverage curve...")
    chart_coverage_curve(log, OUT_DIR / "coverage_curve.png")

    print("Chart 5/6: no-rescue mortality curve (re-simulating)...")
    chart_mortality_curve(OUT_DIR / "mortality_curve_no_rescue.png")

    print("Chart 6/6: network graph, before/after...")
    chart_network_graph(log, OUT_DIR / "network_graph_before_after.png")

    print(f"\nAll 6 charts written to {OUT_DIR} at {DPI} DPI.")


if __name__ == "__main__":
    main()
