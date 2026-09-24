"""
Phase 5 checkpoint: run the full pipeline (world + sensors + comms +
engine, including the real triage scheduler and rescue resolution) and
report lives saved plus a decision trace -- a sample of real, comparative
Assignment.reason strings from across the run.

Run standalone with:  python -m simulation.phase5_checkpoint
"""

from __future__ import annotations

import random
import sys
from collections import Counter
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
    all_reasons = []  # (t, reason, expected_value)
    rescue_events_all = []

    for step in range(n_steps):
        t = step * params.DT_SECONDS
        world.step(params.DT_SECONDS)

        raw_by_drone = sensors.scan(world.drones, world.victims, t, rng, rescued_ids=world.rescued_ids)
        has_backhaul, linked_to = comms.compute_backhaul(world.drones)
        for d in world.drones:
            d.linked_to = linked_to[d.drone_id]
        dets = relay.route(raw_by_drone, has_backhaul)
        dets = dets + world.pop_field_report_detections()

        inp = EngineInput(
            t=t, detections=dets, drones=world.drones, resources=world.resources,
            environment={"water_level": world.water_level, "rise_rate": params.DEFAULT_RISE_RATE_M_PER_HR},
            prior_map=world.prior_map,
        )
        out = engine.decide(inp)
        world.apply(out, engine.get_belief_positions(), engine.get_belief_uncertainties())

        for a in out.assignments:
            all_reasons.append((t, a.reason, a.expected_value))
        rescue_events_all.extend(world.pop_rescue_events())
        world.pop_death_events()  # drain, not needed here

    n_total_people = sum(v.group_size for v in world.victims)
    n_alive_people = sum(v.group_size for v in world.victims if v.alive)
    rescued_people = sum(v.group_size for v in world.victims if v.victim_id in world.rescued_ids)

    print("=== Lives saved ===")
    print(f"total people in scenario: {n_total_people}")
    print(f"alive at end (8h): {n_alive_people}")
    print(f"RESCUED (extracted by a resource): {len(world.rescued_ids)} groups, {rescued_people} people")
    print(f"alive but not yet reached: {n_alive_people - rescued_people} people")
    print(f"dead: {n_total_people - n_alive_people} people")

    outcome_counts = Counter(e["outcome"] for e in rescue_events_all)
    print(f"\n=== Rescue mission outcomes (all {len(rescue_events_all)} completed dispatches) ===")
    for outcome, n in outcome_counts.most_common():
        print(f"  {outcome:30s} {n}")
    print("  Note: 'wrong_resource_type' and 'no_victim_found' are the direct cost of the")
    print("  engine's own uncertainty -- an unconfirmed env_type guess, a false-positive")
    print("  sensor reading, or a victim who died/was rescued before this dispatch arrived.")
    print("  This is expected, not a bug: it's exactly the cost the scheduler is trying to")
    print("  minimise by weighting confidence into base_value.")

    print(f"\n=== Decision trace (12 sample assignments across the run) ===")
    sample_idx = [int(i * (len(all_reasons) - 1) / 11) for i in range(12)] if len(all_reasons) >= 12 else range(len(all_reasons))
    seen = set()
    shown = 0
    for idx in sample_idx:
        if idx in seen or idx >= len(all_reasons):
            continue
        seen.add(idx)
        t, reason, value = all_reasons[idx]
        print(f"  [t={t/60:6.1f} min] {reason}")
        shown += 1

    print(f"\ntotal assignment decisions made over the run: {len(all_reasons)}")


if __name__ == "__main__":
    main()
