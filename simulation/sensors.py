"""
Phase 3 sensor models: thermal, uwb, rf.

Each sensor has a real physical detection limit, not just noise on top of
omniscience:
  - thermal sees surface heat signatures (rooftop/street) but is BLIND to
    anyone under rubble -- infrared cannot see through debris.
  - uwb (a ground-penetrating/through-rubble radar-like sensor) is the
    only thing that can find buried victims, but it is short range and
    "slow": only one uwb-equipped drone in the whole fleet gets to run it
    per timestep (see UWB comment in config/params.py).
  - rf picks up any phone that is genuinely broadcasting (phone_state ==
    "alive"); a dead battery or a phone with no app is silent to RF,
    regardless of range.

Every scan can miss (MISS_PROB per sensor) and can produce an occasional
false positive (FALSE_POSITIVE_PROB_PER_SCAN), because that's what real
sensors do.

/simulation is allowed to import from engine.contracts (the shared wire
format). It must never import from engine.engine.

Detections are returned grouped by the producing drone
(Dict[drone_id, List[Detection]]) so simulation/comms.py can decide, per
drone, whether that drone currently has backhaul to base -- a drone
without backhaul BUFFERS what it found instead of delivering it.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List

from config import params
from engine.contracts import Detection, Drone, Victim


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _thermal_scan(drone: Drone, victims: List[Victim], t: float, rng: random.Random, counter: List[int]) -> List[Detection]:
    dets: List[Detection] = []
    for v in victims:
        if not v.alive or v.env_type == "buried":
            continue  # thermal IR cannot see through rubble
        if _dist(drone.position, v.true_position) > params.THERMAL_RANGE_M:
            continue
        if rng.random() < params.THERMAL_MISS_PROB:
            continue
        pos = (
            v.true_position[0] + rng.gauss(0, params.THERMAL_SIGMA_M),
            v.true_position[1] + rng.gauss(0, params.THERMAL_SIGMA_M),
        )
        dets.append(Detection(
            detection_id=f"th_{t}_{counter[0]}", est_position=pos,
            position_sigma=params.THERMAL_SIGMA_M, sensor="thermal",
            confidence=params.THERMAL_CONFIDENCE, timestamp=t, env_hint="surface",
        ))
        counter[0] += 1

    if rng.random() < params.FALSE_POSITIVE_PROB_PER_SCAN:
        pos = (
            drone.position[0] + rng.uniform(-params.THERMAL_RANGE_M, params.THERMAL_RANGE_M),
            drone.position[1] + rng.uniform(-params.THERMAL_RANGE_M, params.THERMAL_RANGE_M),
        )
        dets.append(Detection(
            detection_id=f"th_{t}_{counter[0]}_fp", est_position=pos,
            position_sigma=params.THERMAL_SIGMA_M, sensor="thermal",
            confidence=params.THERMAL_CONFIDENCE * 0.5, timestamp=t, env_hint="surface",
        ))
        counter[0] += 1
    return dets


def _uwb_scan(drone: Drone, victims: List[Victim], t: float, rng: random.Random, counter: List[int]) -> List[Detection]:
    dets: List[Detection] = []
    for v in victims:
        if not v.alive or v.env_type != "buried":
            continue  # uwb in this model only looks for buried victims
        if _dist(drone.position, v.true_position) > params.UWB_RANGE_M:
            continue
        if rng.random() < params.UWB_MISS_PROB:
            continue
        pos = (
            v.true_position[0] + rng.gauss(0, params.UWB_SIGMA_M),
            v.true_position[1] + rng.gauss(0, params.UWB_SIGMA_M),
        )
        dets.append(Detection(
            detection_id=f"uwb_{t}_{counter[0]}", est_position=pos,
            position_sigma=params.UWB_SIGMA_M, sensor="uwb",
            confidence=params.UWB_CONFIDENCE, timestamp=t, env_hint="buried",
        ))
        counter[0] += 1

    if rng.random() < params.FALSE_POSITIVE_PROB_PER_SCAN:
        pos = (
            drone.position[0] + rng.uniform(-params.UWB_RANGE_M, params.UWB_RANGE_M),
            drone.position[1] + rng.uniform(-params.UWB_RANGE_M, params.UWB_RANGE_M),
        )
        dets.append(Detection(
            detection_id=f"uwb_{t}_{counter[0]}_fp", est_position=pos,
            position_sigma=params.UWB_SIGMA_M, sensor="uwb",
            confidence=params.UWB_CONFIDENCE * 0.5, timestamp=t, env_hint="buried",
        ))
        counter[0] += 1
    return dets


def _rf_scan(drone: Drone, victims: List[Victim], t: float, rng: random.Random, counter: List[int]) -> List[Detection]:
    dets: List[Detection] = []
    for v in victims:
        if not v.alive or v.phone_state != "alive":
            continue  # dead battery or no app: silent to RF regardless of range
        if _dist(drone.position, v.true_position) > params.RF_RANGE_M:
            continue
        if rng.random() < params.RF_MISS_PROB:
            continue
        pos = (
            v.true_position[0] + rng.gauss(0, params.RF_SIGMA_M),
            v.true_position[1] + rng.gauss(0, params.RF_SIGMA_M),
        )
        # RF just picks up a phone signal -- it cannot tell buried from surface.
        dets.append(Detection(
            detection_id=f"rf_{t}_{counter[0]}", est_position=pos,
            position_sigma=params.RF_SIGMA_M, sensor="rf",
            confidence=params.RF_CONFIDENCE, timestamp=t, env_hint="unknown",
        ))
        counter[0] += 1

    if rng.random() < params.FALSE_POSITIVE_PROB_PER_SCAN:
        pos = (
            drone.position[0] + rng.uniform(-params.RF_RANGE_M, params.RF_RANGE_M),
            drone.position[1] + rng.uniform(-params.RF_RANGE_M, params.RF_RANGE_M),
        )
        dets.append(Detection(
            detection_id=f"rf_{t}_{counter[0]}_fp", est_position=pos,
            position_sigma=params.RF_SIGMA_M, sensor="rf",
            confidence=params.RF_CONFIDENCE * 0.5, timestamp=t, env_hint="unknown",
        ))
        counter[0] += 1
    return dets


def scan(drones: List[Drone], victims: List[Victim], t: float, rng: random.Random,
         rescued_ids: frozenset = frozenset()) -> Dict[str, List[Detection]]:
    """
    Returns detections grouped by producing drone_id.

    rescued_ids: victims already extracted are no longer physically at
    that location for a sensor to find -- without this, a rescued victim
    (still alive=True, just safe) keeps generating detections forever,
    which keeps reinforcing the belief and wastes further dispatches on
    an already-resolved location.
    """
    counter = [0]
    out: Dict[str, List[Detection]] = {d.drone_id: [] for d in drones}
    active_victims = [v for v in victims if v.victim_id not in rescued_ids]

    uwb_drones = [d for d in drones if "uwb" in d.sensors]
    active_uwb_id = None
    if uwb_drones:
        idx = int(t // params.DT_SECONDS) % len(uwb_drones)
        active_uwb_id = uwb_drones[idx].drone_id

    for d in drones:
        if "thermal" in d.sensors:
            out[d.drone_id].extend(_thermal_scan(d, active_victims, t, rng, counter))
        if "rf" in d.sensors:
            out[d.drone_id].extend(_rf_scan(d, active_victims, t, rng, counter))
        if "uwb" in d.sensors and d.drone_id == active_uwb_id:
            out[d.drone_id].extend(_uwb_scan(d, active_victims, t, rng, counter))

    return out


if __name__ == "__main__":
    import random as _random
    from simulation.world import World

    rng = _random.Random(params.RANDOM_SEED)
    world = World(rng)

    totals = {"thermal": 0, "uwb": 0, "rf": 0}
    for step in range(60):  # 30 minutes
        world.step(params.DT_SECONDS)
        by_drone = scan(world.drones, world.victims, world.t, rng)
        for dets in by_drone.values():
            for det in dets:
                totals[det.sensor] += 1

    print("detections over first 30 min, by sensor:", totals)
    print("(uwb should be much lower -- one drone, one 60m circle, per timestep)")
