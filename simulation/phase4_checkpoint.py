"""
Phase 4 checkpoint: run the real sensor/comms pipeline through the real
Engine (belief fusion + survival/extraction models) and show:
  - how many raw detections collapsed into how many fused beliefs
  - a few individual beliefs' fields (env_type, confidence, group size
    guess, survival_deadline, extraction_minutes, required_resources)
  - the suspected silent zones the belief tracker found

This script only touches simulation/* to DRIVE the demo (generate a
world, produce detections) -- the actual fusion logic under test lives
entirely in /engine and never imports simulation.

Run standalone with:  python -m simulation.phase4_checkpoint
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import params
from engine.contracts import EngineInput
from engine.engine import Engine
from simulation import comms, sensors
from simulation.world import World


def main():
    rng = random.Random(params.RANDOM_SEED)
    world = World(rng)
    engine = Engine(seed=params.RANDOM_SEED)
    relay = comms.CommsRelay()

    n_steps = int(params.T_TOTAL_SECONDS // params.DT_SECONDS)
    total_raw = 0

    for step in range(n_steps):
        t = step * params.DT_SECONDS
        world.step(params.DT_SECONDS)

        raw_by_drone = sensors.scan(world.drones, world.victims, t, rng, rescued_ids=world.rescued_ids)
        has_backhaul, linked_to = comms.compute_backhaul(world.drones)
        for d in world.drones:
            d.linked_to = linked_to[d.drone_id]
        dets = relay.route(raw_by_drone, has_backhaul)
        total_raw += sum(len(v) for v in raw_by_drone.values())

        inp = EngineInput(
            t=t, detections=dets, drones=world.drones, resources=world.resources,
            environment={"water_level": world.water_level, "rise_rate": params.DEFAULT_RISE_RATE_M_PER_HR},
            prior_map=world.prior_map,
        )
        out = engine.decide(inp)

    beliefs = engine._belief_tracker.get_beliefs()
    silent_zones = engine._belief_tracker.get_silent_zones(world.prior_map)
    stats = engine._belief_tracker.get_stats()

    print("=== Detection fusion ===")
    print(f"total raw detections over {n_steps} steps: {total_raw}")
    print(f"total distinct belief clusters ever created: {stats['total_created']}")
    print(f"  (fusion ratio: {total_raw / max(stats['total_created'],1):.1f} detections per cluster)")
    print(f"currently alive (not yet pruned by confidence decay): {stats['currently_alive']}")
    print("  Note: with drones still doing Phase-1-stub RANDOM flight (not yet")
    print("  connectivity/information-aware -- that's Phase 6), most clusters go")
    print("  a long time without being re-seen and decay below the prune threshold.")
    print("  This is a real, honest limitation of undirected flight, not a fusion bug --")
    print("  Phase 6's orchestrator is what keeps beliefs alive by revisiting them.")

    print("\n=== Sample beliefs (first 8, by belief_id) ===")
    header = f"{'belief_id':12s} {'env_type':8s} {'conf':5s} {'grp':4s} {'sensors':16s} {'deadline_min':12s} {'extract_min':11s} {'resources'}"
    print(header)
    for b in list(beliefs)[:8]:
        deadline = "inf" if b.survival_deadline == float("inf") else f"{b.survival_deadline:.0f}"
        print(f"{b.belief_id:12s} {b.env_type:8s} {b.confidence:.2f} {b.est_group_size:4d} "
              f"{','.join(b.detected_by):16s} {deadline:12s} {b.extraction_minutes:11.1f} {b.required_resources}")

    env_counts = {}
    for b in beliefs:
        env_counts[b.env_type] = env_counts.get(b.env_type, 0) + 1
    print(f"\nbelief env_type breakdown: {env_counts}")

    print(f"\n=== Suspected silent zones (found {len(silent_zones)}) ===")
    for x, y, risk in silent_zones[:15]:
        print(f"  ({x:7.1f}, {y:7.1f})  risk={risk:.2f}")
    if len(silent_zones) > 15:
        print(f"  ... and {len(silent_zones) - 15} more")


if __name__ == "__main__":
    main()
