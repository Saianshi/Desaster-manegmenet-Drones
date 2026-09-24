"""
Phase 2 checkpoint: victim counts, cluster size distribution, and the
no-rescue mortality curve.

Runs the world with NO engine, NO resources, NO drones doing anything
useful -- just World.step() advancing time and killing victims per the
two survival curves. This isolates the ground-truth death process from
any scheduling decisions, which is the point: it's the baseline hazard
the engine is trying to beat.

Run standalone with:  python -m simulation.phase2_checkpoint
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import params
from simulation.world import World


def main():
    rng = random.Random(params.RANDOM_SEED)
    world = World(rng)

    # --- victim counts by type ------------------------------------------------
    groups_by_type = Counter(v.env_type for v in world.victims)
    people_by_type = Counter()
    for v in world.victims:
        people_by_type[v.env_type] += v.group_size

    print("=== Victim counts ===")
    for kind in ("buried", "rooftop", "street"):
        print(f"  {kind:8s}: {groups_by_type[kind]:3d} groups, {people_by_type[kind]:4d} people")
    print(f"  {'TOTAL':8s}: {len(world.victims):3d} groups, {sum(people_by_type.values()):4d} people")

    # --- cluster (group) size distribution -------------------------------------
    sizes = [v.group_size for v in world.victims]
    size_counts = Counter(sizes)
    print("\n=== Group size distribution (1-12 people per group) ===")
    for size in range(1, 13):
        n = size_counts.get(size, 0)
        bar = "#" * n
        print(f"  size {size:2d}: {n:3d} {bar}")
    print(f"  mean group size: {sum(sizes)/len(sizes):.2f}  max: {max(sizes)}  min: {min(sizes)}")

    # --- no-rescue mortality curve ----------------------------------------------
    n_steps = int(params.T_TOTAL_SECONDS // params.DT_SECONDS)
    t_hours = []
    alive_people = {"buried": [], "rooftop": [], "street": []}

    for step in range(n_steps):
        world.step(params.DT_SECONDS)
        t_hours.append(world.t / 3600.0)
        counts = Counter()
        for v in world.victims:
            if v.alive:
                counts[v.env_type] += v.group_size
        for kind in ("buried", "rooftop", "street"):
            alive_people[kind].append(counts[kind])

    print("\n=== No-rescue outcome after {:.1f}h ===".format(params.T_TOTAL_SECONDS / 3600.0))
    for kind in ("buried", "rooftop", "street"):
        start = people_by_type[kind]
        end = alive_people[kind][-1] if alive_people[kind] else start
        print(f"  {kind:8s}: {start:4d} -> {end:4d} alive  ({start-end} deaths, {100*(start-end)/max(start,1):.0f}%)")

    # --- plot --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"buried": "#8B4513", "rooftop": "#1E6FB5", "street": "#4C9A4C"}
    labels = {
        "buried": "Buried (crush syndrome, hours-scale decline)",
        "rooftop": "Rooftop in flood zone (steep drop as water arrives)",
        "street": "Street level (no active hazard modelled)",
    }
    for kind in ("buried", "rooftop", "street"):
        ax.plot(t_hours, alive_people[kind], label=labels[kind], color=colors[kind], linewidth=2.2)

    ax.set_xlabel("Time since disaster (hours)")
    ax.set_ylabel("People still alive (no rescue)")
    ax.set_title("No-rescue mortality curves by victim population")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = ROOT / "output" / "charts" / "mortality_curve_no_rescue.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"\nChart saved to: {out_path}")


if __name__ == "__main__":
    main()
