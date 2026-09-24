"""
Produces the DUAL-STRATEGY log the /viz layer reads (output/run_log.json).
Runs the full engine and the pure_baseline (nearest-first) strategy on
the SAME seed, so both panels of viz/animate.py see the identical
starting world/buildings/victims/terrain, diverging only through their
own decisions from there on -- same fairness principle as
simulation/harness.py's run_scenario.

This is a driver script: it does not modify /engine or /simulation, it
only calls their existing, unmodified public interfaces (World, Engine,
NearestFirstEngine, sensors, comms) twice and writes an extended log.

Log schema (differs from simulation/runner.py's single-strategy log):
{
  "meta": {seed, dt, T, map_size_m, cell_size_m},
  "world_static": {...}       -- shared: identical for both runs (same seed)
  "prior_map": [...]          -- shared
  "runs": {
    "engine":   {"entries": [...], "ground_truth": {...}},
    "baseline": {"entries": [...], "ground_truth": {...}}
  }
}

ground_truth.timesteps carries an extra per-victim "urgency" field (0-1)
beyond simulation/world.py's own snapshot_ground_truth() -- a rendering
aid computed here by READING already-public World state (victim_meta,
water_level), not by editing World's own method. It approximates how
close each living victim is to their hazard threshold, using the same
physical quantities (median_survival_hours, elevation_m vs water_level)
the ground-truth hazard model in simulation/victims.py already tracks,
so the animation can colour victims by urgency without needing a
seperate simulation change.

Run standalone with:  python -m simulation.generate_viz_log [seed]
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import params
from baseline.nearest_first import NearestFirstEngine
from engine.engine import Engine
from simulation import comms, sensors
from simulation.harness import build_engine_input
from simulation.world import World


def _victim_urgency(world: World) -> dict:
    """0 (safe) - 1 (imminent) per living victim, from public World state
    only. Buried: inverse of assumed severity via median_survival_hours
    (shorter median = more severe = more urgent). Rooftop: how close the
    current water_level is to this victim's true elevation. Street: 0
    (no active hazard in this model, matching victims.update_victims)."""
    urgency = {}
    for v in world.victims:
        if not v.alive:
            urgency[v.victim_id] = 0.0
            continue
        meta = world.victim_meta.get(v.victim_id, {})
        if v.env_type == "buried":
            median_h = meta.get("median_survival_hours", params.BURIED_MEDIAN_SURVIVAL_HOURS_BASE)
            urgency[v.victim_id] = max(0.0, min(1.0, 1 - median_h / params.BURIED_MEDIAN_SURVIVAL_HOURS_BASE))
        elif v.env_type == "rooftop":
            elevation = meta.get("elevation_m", 999.0)
            margin = elevation - world.water_level
            urgency[v.victim_id] = max(0.0, min(1.0, 1 - margin / 5.0))
        else:
            urgency[v.victim_id] = 0.0
    return urgency


def run_one(engine, seed: int, T: float, dt: float) -> dict:
    rng = random.Random(seed)
    world = World(rng, n_resources=params.N_RESOURCES_STUB)
    relay = comms.CommsRelay()

    entries = []
    gt_timesteps = []
    death_events = []
    rescue_events = []
    prior_map = None
    world_static = world.to_static_dict()

    n_steps = int(T // dt)
    for step in range(n_steps):
        t = step * dt
        world.step(dt)

        raw_by_drone = sensors.scan(world.drones, world.victims, t, rng, rescued_ids=world.rescued_ids)
        has_backhaul, linked_to = comms.compute_backhaul(world.drones)
        for d in world.drones:
            d.linked_to = linked_to[d.drone_id]
        dets = relay.route(raw_by_drone, has_backhaul)
        dets = dets + world.pop_field_report_detections()

        inp = build_engine_input(t, dets, world)
        out = engine.decide(inp)
        belief_positions = engine.get_belief_positions() if hasattr(engine, "get_belief_positions") else {}
        belief_uncertainties = engine.get_belief_uncertainties() if hasattr(engine, "get_belief_uncertainties") else {}
        world.apply(out, belief_positions, belief_uncertainties)

        if prior_map is None:
            prior_map = inp.prior_map.tolist()
        input_dict = inp.to_dict()
        del input_dict["prior_map"]
        entry = {"input": input_dict, "output": out.to_dict()}
        # Full belief snapshot (survival_deadline, extraction_minutes,
        # required_resources, est_group_size, confidence) -- EngineOutput
        # itself doesn't carry this (only assignments/priority_list/
        # silent_zones), and viz/responder_view.py needs it for the
        # ranked action list. This reads the engine's OWN already-public
        # VictimBelief.to_dict() output, nothing engine.py doesn't already
        # expose to its own caller -- not a change to /engine.
        if hasattr(engine, "_belief_tracker"):
            entry["beliefs"] = [b.to_dict() for b in engine._belief_tracker.get_beliefs()]
        entries.append(entry)

        urgency = _victim_urgency(world)
        gt = world.snapshot_ground_truth()
        for vdict in gt["victims"]:
            vdict["urgency"] = urgency.get(vdict["victim_id"], 0.0)
        gt_timesteps.append(gt)

        death_events.extend(world.pop_death_events())
        rescue_events.extend(world.pop_rescue_events())

    n_total = sum(v.group_size for v in world.victims)
    n_alive = sum(v.group_size for v in world.victims if v.alive)
    n_rescued = sum(v.group_size for v in world.victims if v.victim_id in world.rescued_ids)
    print(f"  rescued {n_rescued}/{n_total}, alive at end {n_alive}/{n_total}")

    victims_static = {v.victim_id: {"group_size": v.group_size, "env_type": v.env_type} for v in world.victims}

    return {
        "entries": entries,
        "ground_truth": {"timesteps": gt_timesteps, "death_events": death_events, "rescue_events": rescue_events},
        "_prior_map": prior_map,
        "_world_static": world_static,
        "_victims_static": victims_static,
    }


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else params.RANDOM_SEED
    T, dt = params.T_TOTAL_SECONDS, params.DT_SECONDS

    print(f"Running full engine (seed={seed})...")
    engine_run = run_one(Engine(seed=seed), seed, T, dt)
    print(f"Running baseline (seed={seed})...")
    baseline_run = run_one(NearestFirstEngine(seed=seed), seed, T, dt)

    combined = {
        "meta": {"seed": seed, "dt": dt, "T": T, "map_size_m": params.MAP_SIZE_M, "cell_size_m": params.CELL_SIZE_M},
        "world_static": engine_run["_world_static"],  # identical for both -- same seed generates the same terrain/buildings
        "prior_map": engine_run["_prior_map"],
        "victims_static": engine_run["_victims_static"],  # group_size/env_type per victim_id, identical for both runs
        "runs": {
            "engine": {"entries": engine_run["entries"], "ground_truth": engine_run["ground_truth"]},
            "baseline": {"entries": baseline_run["entries"], "ground_truth": baseline_run["ground_truth"]},
        },
    }

    out_path = ROOT / "output" / "run_log.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(combined, f)
    print(f"\nDual-strategy log written to {out_path}")


if __name__ == "__main__":
    main()
