"""
Phase 3 checkpoint: plot the BLE mesh graph at t=0, and plot reachability
+ backhaul coverage over the full run.

Run standalone with:  python -m simulation.phase3_checkpoint
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import params
from simulation import comms
from simulation.world import World

ENV_COLOR = {"buried": "#8B4513", "rooftop": "#1E6FB5", "street": "#4C9A4C"}


def plot_mesh_graph(world: World, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))

    for b in world.buildings:
        color = "#555555" if b.collapsed else "#bbbbbb"
        ax.add_patch(Rectangle((b.x0, b.y0), b.x1 - b.x0, b.y1 - b.y0, facecolor=color, edgecolor="none", alpha=0.6, zorder=1))

    edges = comms.build_ble_edges(world.victims)
    pos = {v.victim_id: v.true_position for v in world.victims}
    for a, b in edges:
        xa, ya = pos[a]
        xb, yb = pos[b]
        ax.plot([xa, xb], [ya, yb], color="#e6a817", linewidth=1.0, alpha=0.8, zorder=2)

    for kind in ("buried", "rooftop", "street"):
        xs = [v.true_position[0] for v in world.victims if v.env_type == kind]
        ys = [v.true_position[1] for v in world.victims if v.env_type == kind]
        sizes = [18 + 6 * v.group_size for v in world.victims if v.env_type == kind]
        ax.scatter(xs, ys, s=sizes, c=ENV_COLOR[kind], label=f"{kind} victims", zorder=3, edgecolors="black", linewidths=0.3)

    has_backhaul, linked_to = comms.compute_backhaul(world.drones)
    base = params.BASE_STATION_POSITION
    ax.scatter([base[0]], [base[1]], marker="*", s=400, c="black", label="base station", zorder=5)

    for d in world.drones:
        color = "#00b050" if has_backhaul[d.drone_id] else "#e00000"
        ax.scatter([d.position[0]], [d.position[1]], marker="^", s=180, c=color, edgecolors="black", zorder=5)
        ax.annotate(d.drone_id, d.position, textcoords="offset points", xytext=(6, 6), fontsize=8, zorder=5)
        for nb in linked_to[d.drone_id]:
            if nb == "base":
                ax.plot([d.position[0], base[0]], [d.position[1], base[1]], "k--", linewidth=1.2, alpha=0.7, zorder=4)
            else:
                other = next(x for x in world.drones if x.drone_id == nb)
                ax.plot([d.position[0], other.position[0]], [d.position[1], other.position[1]], "k--", linewidth=1.2, alpha=0.7, zorder=4)

    ax.set_xlim(0, world.map_size)
    ax.set_ylim(0, world.map_size)
    ax.set_aspect("equal")
    ax.set_title("Comms mesh at t=0: BLE clusters (orange), backhaul chain (dashed),\ngreen triangle = drone has backhaul, red = buffering")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)


def run_coverage_over_time(seed: int) -> tuple:
    rng = random.Random(seed)
    world = World(rng)
    relay = comms.CommsRelay()

    n_steps = int(params.T_TOTAL_SECONDS // params.DT_SECONDS)
    t_hours, frac_reachable, frac_backhaul = [], [], []

    for _ in range(n_steps):
        world.step(params.DT_SECONDS)

        alive_people = sum(v.group_size for v in world.victims if v.alive)
        reachable_ids = comms.reachable_victim_ids(world.victims, world.drones)
        reachable_people = sum(v.group_size for v in world.victims if v.alive and v.victim_id in reachable_ids)
        frac_reachable.append(reachable_people / alive_people if alive_people else 0.0)

        has_backhaul, linked_to = comms.compute_backhaul(world.drones)
        for d in world.drones:
            d.linked_to = linked_to[d.drone_id]
        frac_backhaul.append(sum(has_backhaul.values()) / len(world.drones))

        t_hours.append(world.t / 3600.0)

    return t_hours, frac_reachable, frac_backhaul


def plot_coverage(t_hours, frac_reachable, frac_backhaul, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_hours, [100 * x for x in frac_reachable], label="% of alive people currently reachable (Layer A+B)", color="#1E6FB5", linewidth=2)
    ax.plot(t_hours, [100 * x for x in frac_backhaul], label="% of drones with backhaul to base (Layer C)", color="#c0392b", linewidth=2)
    ax.set_xlabel("Time since disaster (hours)")
    ax.set_ylabel("%")
    ax.set_ylim(-2, 102)
    ax.set_title("Comms coverage over time")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)


def main():
    rng = random.Random(params.RANDOM_SEED)
    world = World(rng)

    clusters = comms.build_ble_clusters(world.victims)
    reachable = comms.reachable_victim_ids(world.victims, world.drones)
    has_backhaul, linked_to = comms.compute_backhaul(world.drones)

    print("=== Mesh snapshot at t=0 ===")
    print(f"victims with a live phone: {sum(len(c) for c in clusters)}")
    print(f"BLE clusters: {len(clusters)}   sizes: {sorted(len(c) for c in clusters)}")
    print(f"victims reachable right now (Layer A+B): {len(reachable)}")
    print(f"drones with backhaul to base: {sum(has_backhaul.values())}/{len(world.drones)}  -> {has_backhaul}")

    mesh_path = ROOT / "output" / "charts" / "mesh_graph.png"
    plot_mesh_graph(world, mesh_path)
    print(f"\nMesh graph saved to: {mesh_path}")

    print("\n=== Running full 8h scenario for coverage-over-time ===")
    t_hours, frac_reachable, frac_backhaul = run_coverage_over_time(params.RANDOM_SEED)
    print(f"mean reachability: {100*sum(frac_reachable)/len(frac_reachable):.1f}%")
    print(f"mean backhaul uptime: {100*sum(frac_backhaul)/len(frac_backhaul):.1f}%")

    coverage_path = ROOT / "output" / "charts" / "comms_coverage.png"
    plot_coverage(t_hours, frac_reachable, frac_backhaul, coverage_path)
    print(f"Coverage chart saved to: {coverage_path}")


if __name__ == "__main__":
    main()
