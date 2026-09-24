"""
Phase 6 checkpoint: engine.orchestrator vs a naive lawnmower sweep,
under the IDENTICAL physical constraint (DRONE_SPEED_M_PER_MIN) -- same
hardware, different flight logic. Plots drone paths for the informed run
and a population-weighted coverage curve for both strategies.

Both runs use a fresh BeliefTracker fed only drone positions (no real
detections needed for a pure coverage comparison) -- this reuses the
exact scan-marking machinery under test, rather than duplicating it.

Run standalone with:  python -m simulation.phase6_checkpoint
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baseline.lawnmower import lawnmower_position
from config import params
from engine import orchestrator
from engine.belief import BeliefTracker
from engine.contracts import Drone, EngineInput
from simulation.world import World


def weighted_coverage(tracker: BeliefTracker, prior_map: np.ndarray) -> float:
    """
    Coverage weighted the SAME way the orchestrator values information
    gain (ORCHESTRATOR_SENSOR_INFO_WEIGHT) rather than raw "scanned by
    anything". This matters: rf's 300m passive range covers nearly the
    whole 2km map within minutes regardless of flight strategy, so a
    naive "scanned by any sensor" metric saturates near-instantly for
    BOTH strategies and shows no difference at all. Weighting by scarcity
    (uwb >> thermal >> rf) surfaces the real, much slower-growing signal:
    who actually got the hard-to-get coverage that matters.
    """
    total = float(prior_map.sum())
    weight_sum = sum(params.ORCHESTRATOR_SENSOR_INFO_WEIGHT.values())
    if total <= 0 or weight_sum <= 0:
        return 0.0
    covered = sum(
        w * float(prior_map[tracker._scanned_by_sensor[s]].sum())
        for s, w in params.ORCHESTRATOR_SENSOR_INFO_WEIGHT.items()
    )
    return covered / (total * weight_sum)


def sensor_coverage(tracker: BeliefTracker, prior_map: np.ndarray, sensor: str) -> float:
    total = float(prior_map.sum())
    if total <= 0:
        return 0.0
    return float(prior_map[tracker._scanned_by_sensor[sensor]].sum()) / total


def run_informed(world: World) -> tuple:
    """Real orchestrator, driven by a real BeliefTracker that also sees
    real detections (so its info_gain reflects genuine fusion state, not
    just a coverage-only proxy)."""
    tracker = BeliefTracker(seed=params.RANDOM_SEED)
    drones = [Drone(drone_id=d.drone_id, position=d.position, battery_pct=100.0,
                     sensors=list(d.sensors), comm_range=d.comm_range, linked_to=[]) for d in world.drones]
    map_size = world.map_size
    n_steps = int(params.T_TOTAL_SECONDS // params.DT_SECONDS)

    t_hours, coverage, uwb_coverage, paths = [], [], [], {d.drone_id: [d.position] for d in drones}
    rng = random.Random(params.RANDOM_SEED)

    for step in range(n_steps):
        t = step * params.DT_SECONDS
        from simulation import sensors as sensors_mod
        raw_by_drone = sensors_mod.scan(drones, world.victims, t, rng)
        dets = [d for dl in raw_by_drone.values() for d in dl]

        inp = EngineInput(t=t, detections=dets, drones=drones, resources=[],
                           environment={"water_level": world.water_level, "rise_rate": params.DEFAULT_RISE_RATE_M_PER_HR},
                           prior_map=world.prior_map)
        tracker.update(inp)
        silent_zones = tracker.get_silent_zones(world.prior_map)
        waypoints = orchestrator.decide_waypoints(drones, tracker, world.prior_map, silent_zones, map_size)
        for d in drones:
            d.position = waypoints[d.drone_id]
            paths[d.drone_id].append(d.position)

        t_hours.append(t / 3600.0)
        coverage.append(weighted_coverage(tracker, world.prior_map))
        uwb_coverage.append(sensor_coverage(tracker, world.prior_map, "uwb"))

    return t_hours, coverage, uwb_coverage, paths


def run_lawnmower(world: World) -> tuple:
    tracker = BeliefTracker(seed=params.RANDOM_SEED)
    n_drones = len(world.drones)
    map_size = world.map_size
    n_steps = int(params.T_TOTAL_SECONDS // params.DT_SECONDS)

    t_hours, coverage, uwb_coverage = [], [], []
    for step in range(n_steps):
        t = step * params.DT_SECONDS
        drones = [
            Drone(drone_id=f"d{i}", position=lawnmower_position(i, n_drones, t, map_size),
                  battery_pct=100.0, sensors=["thermal", "uwb", "rf"], comm_range=300.0, linked_to=[])
            for i in range(n_drones)
        ]
        inp = EngineInput(t=t, detections=[], drones=drones, resources=[],
                           environment={"water_level": 0.0, "rise_rate": 0.0}, prior_map=world.prior_map)
        tracker.update(inp)

        t_hours.append(t / 3600.0)
        coverage.append(weighted_coverage(tracker, world.prior_map))
        uwb_coverage.append(sensor_coverage(tracker, world.prior_map, "uwb"))

    return t_hours, coverage, uwb_coverage


def plot_paths(world: World, paths: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    for b in world.buildings:
        color = "#555555" if b.collapsed else "#cccccc"
        ax.add_patch(Rectangle((b.x0, b.y0), b.x1 - b.x0, b.y1 - b.y0, facecolor=color, edgecolor="none", alpha=0.5, zorder=1))

    colors = plt.cm.tab10.colors
    for i, (drone_id, path) in enumerate(paths.items()):
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color=colors[i % len(colors)], linewidth=1.0, alpha=0.8, label=drone_id, zorder=2)
        ax.scatter([xs[0]], [ys[0]], color=colors[i % len(colors)], marker="o", s=60, zorder=3, edgecolors="black")
        ax.scatter([xs[-1]], [ys[-1]], color=colors[i % len(colors)], marker="s", s=60, zorder=3, edgecolors="black")

    base = params.BASE_STATION_POSITION
    ax.scatter([base[0]], [base[1]], marker="*", s=400, c="gold", edgecolors="black", label="base", zorder=4)

    ax.set_xlim(0, world.map_size)
    ax.set_ylim(0, world.map_size)
    ax.set_aspect("equal")
    ax.set_title("Orchestrator drone paths over 8h (o = start, sq = end)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)


def plot_coverage(t_informed, cov_informed, uwb_informed, t_lawn, cov_lawn, uwb_lawn, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1.plot(t_informed, [100 * c for c in cov_informed], label="orchestrator", color="#1E6FB5", linewidth=2.2)
    ax1.plot(t_lawn, [100 * c for c in cov_lawn], label="naive lawnmower", color="#c0392b", linewidth=2.2, linestyle="--")
    ax1.set_xlabel("Time since disaster (hours)")
    ax1.set_ylabel("Scarcity-weighted coverage (%)")
    ax1.set_title("Composite information gain\n(uwb x5, thermal x1, rf x0.3)")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.plot(t_informed, [100 * c for c in uwb_informed], label="orchestrator", color="#1E6FB5", linewidth=2.2)
    ax2.plot(t_lawn, [100 * c for c in uwb_lawn], label="naive lawnmower", color="#c0392b", linewidth=2.2, linestyle="--")
    ax2.set_xlabel("Time since disaster (hours)")
    ax2.set_ylabel("UWB-only coverage (%)")
    ax2.set_title("UWB coverage specifically\n(the only sensor that finds buried victims)")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle("Coverage over time: orchestrator vs naive lawnmower (identical drone speed, identical scenario)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)


def main():
    rng = random.Random(params.RANDOM_SEED)
    world = World(rng)

    print("=== Running informed (orchestrator) pass ===")
    t_informed, cov_informed, uwb_informed, paths = run_informed(world)

    print("=== Running naive lawnmower pass (same scenario, same drone speed) ===")
    t_lawn, cov_lawn, uwb_lawn = run_lawnmower(world)

    print(f"\nfinal scarcity-weighted coverage:")
    print(f"  orchestrator: {100*cov_informed[-1]:.1f}%   lawnmower: {100*cov_lawn[-1]:.1f}%")
    print(f"final UWB-only coverage (buried-victim-finding capability):")
    print(f"  orchestrator: {100*uwb_informed[-1]:.1f}%   lawnmower: {100*uwb_lawn[-1]:.1f}%")

    def time_to(cov_series, t_series, target):
        for t, c in zip(t_series, cov_series):
            if c >= target:
                return t
        return None

    print("\ntime to reach coverage thresholds (composite):")
    for target in (0.25, 0.5, 0.75):
        ti = time_to(cov_informed, t_informed, target)
        tl = time_to(cov_lawn, t_lawn, target)
        ti_s = f"{ti:.2f}h" if ti is not None else "never"
        tl_s = f"{tl:.2f}h" if tl is not None else "never"
        print(f"  {int(target*100)}%: orchestrator={ti_s}   lawnmower={tl_s}")

    print("\ntime to reach UWB-only coverage thresholds:")
    for target in (0.10, 0.25, 0.5):
        ti = time_to(uwb_informed, t_informed, target)
        tl = time_to(uwb_lawn, t_lawn, target)
        ti_s = f"{ti:.2f}h" if ti is not None else "never"
        tl_s = f"{tl:.2f}h" if tl is not None else "never"
        print(f"  {int(target*100)}%: orchestrator={ti_s}   lawnmower={tl_s}")

    paths_out = ROOT / "output" / "charts" / "orchestrator_paths.png"
    plot_paths(world, paths, paths_out)
    print(f"\nDrone paths chart saved to: {paths_out}")

    coverage_out = ROOT / "output" / "charts" / "orchestrator_vs_lawnmower.png"
    plot_coverage(t_informed, cov_informed, uwb_informed, t_lawn, cov_lawn, uwb_lawn, coverage_out)
    print(f"Coverage comparison chart saved to: {coverage_out}")


if __name__ == "__main__":
    main()
