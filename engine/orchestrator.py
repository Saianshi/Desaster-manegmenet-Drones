"""
engine/orchestrator.py -- decides where each drone flies next.

For each drone, generates a small ring of candidate waypoints (compass
directions at two step radii, plus "stay put") and scores each:

    + information_gain: never-scanned prior population reachable from
      that position (reusing BeliefTracker.unexplored_population_within,
      the same machinery the scheduler's cascade bonus uses), plus extra
      weight for candidates near a suspected silent zone -- actively
      investigate flagged risk, not just any empty cell.
    - connectivity_penalty: a LARGE penalty if flying there would break
      THIS drone's backhaul path to base. Computed using only the
      drones' own exact positions (self-reported telemetry, not victim
      ground truth) and their own known hardware backhaul range --
      legitimate for the engine to reason about.
    - travel_cost: distance, with a battery-awareness multiplier that's
      currently dormant (drone battery is static in this simulation) but
      wired for when it isn't.

Picks the single best candidate per drone, independently. Recomputed
fresh every timestep -- no path planning, no multi-step lookahead, per
spec ("keep it greedy -- do not build a complex planner"). One
consequence of that simplicity, worth naming: if a drone is ALREADY
disconnected and no single-step candidate can reconnect it, the
connectivity penalty applies equally to every candidate that tick and
stops discriminating -- the drone will drift on pure information_gain
until (over several ticks) it happens back into range. A real planner
would route it home directly; this one doesn't, by design.

/engine must never import from /simulation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from config import params
from engine.belief import BeliefTracker
from engine.contracts import Drone

_SENSOR_RANGE_M = {
    "thermal": params.THERMAL_RANGE_M,
    "uwb": params.UWB_RANGE_M,
    "rf": params.RF_RANGE_M,
}


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _max_sensor_range(drone: Drone) -> float:
    ranges = [_SENSOR_RANGE_M[s] for s in drone.sensors if s in _SENSOR_RANGE_M]
    return max(ranges) if ranges else params.RF_RANGE_M


def _silent_zone_reach(drone: Drone) -> float:
    # For the silent-zone proximity bonus (sensor-agnostic), use the
    # drone's broadest sensor reach -- being anywhere within that range
    # of a flagged zone is what lets it get investigated next.
    return _max_sensor_range(drone)


def _candidate_waypoints(drone: Drone, map_size: float) -> List[Tuple[float, float]]:
    # Two step radii (half and full DRONE_SPEED_M_PER_MIN reach) plus
    # "stay put". The half-step option turned out to matter more than
    # expected: it's what lets a drone hover right at the edge of
    # backhaul range instead of only being able to overshoot past it in
    # one full step -- dropping it (tried during Phase 7 performance
    # tuning) caused a lot more disconnection and buffered-not-delivered
    # detections. ORCHESTRATOR_N_ANGLES was trimmed instead, which is a
    # cheaper way to cut candidate count.
    step = params.DRONE_SPEED_M_PER_MIN * (params.DT_SECONDS / 60.0)
    candidates = [drone.position]  # "stay put" is always an option
    for radius_frac in (0.5, 1.0):
        r = step * radius_frac
        for i in range(params.ORCHESTRATOR_N_ANGLES):
            angle = 2 * math.pi * i / params.ORCHESTRATOR_N_ANGLES
            x = min(max(drone.position[0] + r * math.cos(angle), 0.0), map_size)
            y = min(max(drone.position[1] + r * math.sin(angle), 0.0), map_size)
            candidates.append((x, y))
    return candidates


def _would_have_backhaul(candidate_pos: Tuple[float, float], drone_id: str, drones: List[Drone]) -> bool:
    """Same BFS comms.compute_backhaul does in simulation, but run here on
    the engine's own exactly-known drone positions (self-telemetry) --
    everyone else stays put, this one drone moves to candidate_pos."""
    base = params.BASE_STATION_POSITION
    positions = {d.drone_id: (candidate_pos if d.drone_id == drone_id else d.position) for d in drones}
    ids = list(positions.keys())

    adjacency: Dict[str, set] = {did: set() for did in ids}
    linked_to_base = set()
    for i, id1 in enumerate(ids):
        if _dist(positions[id1], base) <= params.DRONE_BACKHAUL_RANGE_M:
            linked_to_base.add(id1)
        for id2 in ids[i + 1:]:
            if _dist(positions[id1], positions[id2]) <= params.DRONE_BACKHAUL_RANGE_M:
                adjacency[id1].add(id2)
                adjacency[id2].add(id1)

    visited = set(linked_to_base)
    frontier = list(linked_to_base)
    while frontier:
        nxt = []
        for did in frontier:
            for nb in adjacency[did]:
                if nb not in visited:
                    visited.add(nb)
                    nxt.append(nb)
        frontier = nxt

    return drone_id in visited


def _information_gain(candidate_pos: Tuple[float, float], drone: Drone, tracker: BeliefTracker,
                       integral_images: Dict, silent_zones: List[Tuple[float, float, float]]) -> float:
    # Summed per sensor the drone actually carries, each using ITS OWN
    # range and ITS OWN scan history -- see ORCHESTRATOR_SENSOR_INFO_WEIGHT
    # for why this matters (rf's cheap huge-range coverage must not drown
    # out the much scarcer, much more valuable uwb coverage). Uses the
    # O(1) integral-image query (query_unexplored_rect), not the exact
    # circular one -- this runs ~17 candidates x 3 sensors x every drone x
    # every timestep, so the meshgrid-per-query version dominated runtime.
    unexplored = 0.0
    for sensor in drone.sensors:
        weight = params.ORCHESTRATOR_SENSOR_INFO_WEIGHT.get(sensor)
        sensor_range = _SENSOR_RANGE_M.get(sensor)
        if weight is None or sensor_range is None:
            continue
        unexplored += weight * tracker.query_unexplored_rect(integral_images[sensor], candidate_pos, sensor_range)

    reach = _silent_zone_reach(drone)
    silent_bonus = sum(
        risk for x, y, risk in silent_zones if _dist(candidate_pos, (x, y)) <= reach
    ) * params.ORCHESTRATOR_SILENT_ZONE_WEIGHT
    return unexplored + silent_bonus


def decide_waypoints(drones: List[Drone], tracker: BeliefTracker, prior_map: np.ndarray,
                      silent_zones: List[Tuple[float, float, float]], map_size: float) -> Dict[str, Tuple[float, float]]:
    # Built ONCE per decision, not once per candidate -- see
    # BeliefTracker.build_unexplored_integral_images.
    integral_images = tracker.build_unexplored_integral_images(prior_map)

    waypoints: Dict[str, Tuple[float, float]] = {}
    for drone in drones:
        best_pos = drone.position
        best_value = float("-inf")
        for candidate in _candidate_waypoints(drone, map_size):
            info_gain = _information_gain(candidate, drone, tracker, integral_images, silent_zones)
            connectivity_penalty = (
                0.0 if _would_have_backhaul(candidate, drone.drone_id, drones)
                else params.ORCHESTRATOR_CONNECTIVITY_PENALTY
            )
            battery_mult = (
                params.ORCHESTRATOR_LOW_BATTERY_COST_MULTIPLIER
                if drone.battery_pct < params.ORCHESTRATOR_LOW_BATTERY_PCT else 1.0
            )
            travel_cost = params.ORCHESTRATOR_TRAVEL_COST_PER_M * _dist(drone.position, candidate) * battery_mult

            value = info_gain - connectivity_penalty - travel_cost
            if value > best_value:
                best_value = value
                best_pos = candidate
        waypoints[drone.drone_id] = best_pos
    return waypoints
