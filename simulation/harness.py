"""
Shared simulation run loop -- the SAME harness drives both the demo
script (simulation/runner.py) and the evaluation suite
(evaluation/compare.py). Fairness for Phase 7's comparisons depends on
this: the full engine and the naive baseline must be driven through
IDENTICAL world generation, sensor physics, and comms routing, with only
the decision-maker swapped out.

Anything passed as `engine` just needs a `.decide(EngineInput) ->
EngineOutput` method, and optionally `.get_belief_positions() ->
Dict[str, Tuple[float, float]]` (used to dispatch resources toward
wherever the engine's targets actually are -- see engine.belief's
get_belief_positions docstring for why this exists outside the
EngineOutput contract). Both engine.engine.Engine and
baseline.nearest_first.NearestFirstEngine satisfy this.
"""

from __future__ import annotations

import random
from typing import Optional

from config import params
from engine.contracts import EngineInput
from simulation import comms, sensors
from simulation.world import World


def build_engine_input(t: float, dets, world: World) -> EngineInput:
    return EngineInput(
        t=t,
        detections=dets,
        drones=world.drones,
        resources=world.resources,
        environment={"water_level": world.water_level, "rise_rate": params.DEFAULT_RISE_RATE_M_PER_HR},
        prior_map=world.prior_map,
    )


def run_scenario(engine, seed: int, T: float = params.T_TOTAL_SECONDS, dt: float = params.DT_SECONDS,
                  n_resources: int = params.N_RESOURCES_STUB, run_log: Optional["object"] = None) -> dict:
    """
    Runs one full scenario with the given engine/seed. If run_log is
    provided (a simulation.runner.RunLog instance), full per-timestep
    logging happens too -- omitted by default because JSON serialisation
    is measurably expensive and evaluation runs don't need it (see
    Phase 4's profiling note in engine/belief.py's git history).

    Returns a metrics dict; see the keys below.
    """
    rng = random.Random(seed)
    world = World(rng, n_resources=n_resources)
    comms_relay = comms.CommsRelay()
    if run_log is not None:
        run_log.set_world_static(world)

    n_steps = int(T // dt)
    total_raw_detections = 0
    total_delivered_detections = 0
    total_assignments = 0

    for step in range(n_steps):
        t = step * dt
        world.step(dt)

        raw_by_drone = sensors.scan(world.drones, world.victims, t, rng, rescued_ids=world.rescued_ids)
        has_backhaul, linked_to = comms.compute_backhaul(world.drones)
        for d in world.drones:
            d.linked_to = linked_to[d.drone_id]
        dets = comms_relay.route(raw_by_drone, has_backhaul)
        # Field reports (crew radios back a corrected env_type after a
        # wrong_resource_type failure) are a direct channel, independent
        # of drone backhaul -- delivered every tick, not comms-routed.
        dets = dets + world.pop_field_report_detections()

        inp = build_engine_input(t, dets, world)
        out = engine.decide(inp)
        belief_positions = engine.get_belief_positions() if hasattr(engine, "get_belief_positions") else {}
        belief_uncertainties = engine.get_belief_uncertainties() if hasattr(engine, "get_belief_uncertainties") else {}
        world.apply(out, belief_positions, belief_uncertainties)

        if run_log is not None:
            run_log.append(inp, out, world)

        total_raw_detections += sum(len(v) for v in raw_by_drone.values())
        total_delivered_detections += len(dets)
        total_assignments += len(out.assignments)

    n_total_people = sum(v.group_size for v in world.victims)
    n_alive_people = sum(v.group_size for v in world.victims if v.alive)
    rescued_people = sum(v.group_size for v in world.victims if v.victim_id in world.rescued_ids)

    return {
        "seed": seed,
        "n_resources": n_resources,
        "n_total_people": n_total_people,
        "n_alive_people": n_alive_people,
        "rescued_people": rescued_people,
        "rescued_groups": len(world.rescued_ids),
        "dead_people": n_total_people - n_alive_people,
        "total_raw_detections": total_raw_detections,
        "total_delivered_detections": total_delivered_detections,
        "total_assignments": total_assignments,
        "final_water_level": world.water_level,
        "buffered_at_end": comms_relay.buffered_count(),
    }
