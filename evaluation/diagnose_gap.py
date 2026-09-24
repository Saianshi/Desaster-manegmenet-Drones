"""
Diagnostic-only comparison of full_engine vs informed_operator on a
single seed. Read-only instrumentation of the existing pipeline -- no
scoring logic is touched here. Answers, with numbers:

  1. Mission outcome breakdown (rescued / no_victim_found /
     wrong_resource_type / died_during_extraction_no_medical /
     partner_timeout / victim_lost_during_extraction) per strategy.
  2. Completed rescue missions and mean travel time per NEW dispatch
     (a dispatch counted only the tick a resource goes idle->enroute).
  3. Distribution of belief.confidence for every dispatched target.
  4. rescued_people split by TRUE env_type (rooftop/buried/street).
  5. How often a resource sits idle while a type-compatible belief
     exists in the tracker but wasn't assigned to it that tick.

Run standalone with:  python -m evaluation.diagnose_gap [seed]
"""

from __future__ import annotations

import math
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import params
from baseline.informed_operator import InformedOperatorEngine
from engine.engine import Engine
from simulation import comms, sensors
from simulation.harness import build_engine_input
from simulation.world import World


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


RESOURCE_COMPATIBLE_ENV = {
    "boat": {"rooftop"},
    "medical": {"street", "buried"},
    "excavator": {"buried"},
}


def run_diagnostic(strategy_name: str, engine, seed: int) -> dict:
    import random
    rng = random.Random(seed)
    world = World(rng, n_resources=params.N_RESOURCES_STUB)
    relay = comms.CommsRelay()
    n_steps = int(params.T_TOTAL_SECONDS // params.DT_SECONDS)

    dispatch_log = []          # one entry per NEW dispatch (idle -> enroute this tick)
    all_rescue_events = []
    idle_serviceable_ticks = Counter()   # resource_type -> count of ticks left idle despite a compatible belief existing
    idle_ticks_total = Counter()         # resource_type -> total idle ticks (denominator)

    for step in range(n_steps):
        t = step * params.DT_SECONDS
        world.step(params.DT_SECONDS)

        raw_by_drone = sensors.scan(world.drones, world.victims, t, rng, rescued_ids=world.rescued_ids)
        has_backhaul, linked_to = comms.compute_backhaul(world.drones)
        for d in world.drones:
            d.linked_to = linked_to[d.drone_id]
        dets = relay.route(raw_by_drone, has_backhaul)
        dets = dets + world.pop_field_report_detections()
        inp = build_engine_input(t, dets, world)

        idle_before = {r.resource_id: (r.status, r.type, r.position, r.speed) for r in world.resources}

        out = engine.decide(inp)
        belief_positions = engine.get_belief_positions() if hasattr(engine, "get_belief_positions") else {}
        belief_uncertainties = engine.get_belief_uncertainties() if hasattr(engine, "get_belief_uncertainties") else {}

        belief_info = {}
        if hasattr(engine, "_belief_tracker"):
            for b in engine._belief_tracker.get_beliefs():
                belief_info[b.belief_id] = (b.confidence, b.env_type)

        assigned_this_tick = set()
        for a in out.assignments:
            info = idle_before.get(a.resource_id)
            if info is None:
                continue
            status, rtype, pos, speed = info
            assigned_this_tick.add(a.resource_id)
            if status == "idle":
                target_pos = belief_positions.get(a.target_belief_id)
                travel_min = _dist(pos, target_pos) / max(speed, 1e-6) if target_pos is not None else None
                conf, env_type = belief_info.get(a.target_belief_id, (None, None))
                dispatch_log.append({
                    "t": t, "resource_type": rtype, "confidence": conf,
                    "belief_env_type": env_type, "travel_minutes": travel_min,
                })

        # idle-but-serviceable: resources idle BEFORE decide(), not touched this tick,
        # while a type-compatible belief exists in the tracker right now.
        env_types_present = {env for (_, env) in belief_info.values()}
        for rid, (status, rtype, pos, speed) in idle_before.items():
            if status != "idle":
                continue
            idle_ticks_total[rtype] += 1
            if rid in assigned_this_tick:
                continue
            compatible_envs = RESOURCE_COMPATIBLE_ENV.get(rtype, set())
            if compatible_envs & env_types_present:
                idle_serviceable_ticks[rtype] += 1

        world.apply(out, belief_positions, belief_uncertainties)
        all_rescue_events.extend(world.pop_rescue_events())
        world.pop_death_events()

    n_total_people = sum(v.group_size for v in world.victims)
    n_alive_people = sum(v.group_size for v in world.victims if v.alive)
    rescued_people = sum(v.group_size for v in world.victims if v.victim_id in world.rescued_ids)

    rescued_by_env = Counter()
    for v in world.victims:
        if v.victim_id in world.rescued_ids:
            rescued_by_env[v.env_type] += v.group_size

    outcome_counts = Counter(e["outcome"] for e in all_rescue_events)

    return {
        "strategy": strategy_name,
        "seed": seed,
        "n_total_people": n_total_people,
        "n_alive_people": n_alive_people,
        "rescued_people": rescued_people,
        "rescued_groups": len(world.rescued_ids),
        "outcome_counts": dict(outcome_counts),
        "dispatch_log": dispatch_log,
        "rescued_by_env": dict(rescued_by_env),
        "idle_serviceable_ticks": dict(idle_serviceable_ticks),
        "idle_ticks_total": dict(idle_ticks_total),
    }


def report(result: dict) -> None:
    name = result["strategy"]
    print(f"\n{'=' * 20} {name} (seed {result['seed']}) {'=' * 20}")
    print(f"rescued_people: {result['rescued_people']} / {result['n_total_people']}   "
          f"(alive at end: {result['n_alive_people']})")

    print("\n[1] Mission outcome breakdown:")
    total_missions = sum(result["outcome_counts"].values())
    for outcome, n in sorted(result["outcome_counts"].items(), key=lambda kv: -kv[1]):
        pct = 100 * n / total_missions if total_missions else 0
        print(f"    {outcome:32s} {n:5d}  ({pct:5.1f}%)")
    print(f"    {'TOTAL completed missions':32s} {total_missions:5d}")

    dispatches = result["dispatch_log"]
    print(f"\n[2] New dispatches (mission starts): {len(dispatches)}")
    travel_times = [d["travel_minutes"] for d in dispatches if d["travel_minutes"] is not None]
    if travel_times:
        print(f"    mean travel time:   {statistics.mean(travel_times):6.1f} min")
        print(f"    median travel time: {statistics.median(travel_times):6.1f} min")
        print(f"    max travel time:    {max(travel_times):6.1f} min")

    print(f"\n[3] belief.confidence of dispatched targets:")
    confs = [d["confidence"] for d in dispatches if d["confidence"] is not None]
    if confs:
        print(f"    n={len(confs)}  mean={statistics.mean(confs):.3f}  median={statistics.median(confs):.3f}  "
              f"min={min(confs):.3f}  max={max(confs):.3f}")
        buckets = Counter()
        for c in confs:
            buckets[int(c * 10) / 10] += 1
        for b in sorted(buckets):
            print(f"      [{b:.1f}-{b + 0.1:.1f}): {'#' * buckets[b]} ({buckets[b]})")

    print(f"\n[4] rescued_people by TRUE env_type:")
    for env in ("rooftop", "buried", "street"):
        print(f"    {env:10s}: {result['rescued_by_env'].get(env, 0)}")

    print(f"\n[5] Resource idle-while-serviceable-belief-exists rate:")
    for rtype in ("boat", "excavator", "medical"):
        idle_total = result["idle_ticks_total"].get(rtype, 0)
        idle_serviceable = result["idle_serviceable_ticks"].get(rtype, 0)
        pct = 100 * idle_serviceable / idle_total if idle_total else 0
        print(f"    {rtype:10s}: {idle_serviceable:4d} / {idle_total:4d} idle ticks ({pct:5.1f}%) had a compatible belief available but unused")

    # dispatch mix by resource type
    rt_counts = Counter(d["resource_type"] for d in dispatches)
    print(f"\n    dispatch count by resource type: {dict(rt_counts)}")


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else params.EVAL_BASE_SEED

    results = []
    for name, engine in [
        ("full_engine", Engine(seed=seed)),
        ("informed_operator", InformedOperatorEngine(seed=seed)),
    ]:
        print(f"Running {name}...", flush=True)
        results.append(run_diagnostic(name, engine, seed))

    for r in results:
        report(r)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    full, informed = results
    print(f"rescued_people: full_engine={full['rescued_people']}  informed_operator={informed['rescued_people']}")
    print(f"total missions completed: full_engine={sum(full['outcome_counts'].values())}  "
          f"informed_operator={sum(informed['outcome_counts'].values())}")
    print(f"rescued missions: full_engine={full['outcome_counts'].get('rescued', 0)}  "
          f"informed_operator={informed['outcome_counts'].get('rescued', 0)}")


if __name__ == "__main__":
    main()
