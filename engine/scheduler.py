"""
engine/scheduler.py -- THE CORE of the decision engine.

For every (idle resource, belief) pairing a resource is physically
capable of servicing, computes:

    arrival_minutes = travel_time(resource.position, belief.position)
    p_survive       = models.survival_probability(belief, arrival_minutes, ...)
    base_value      = p_survive * belief.est_group_size * belief.confidence
    final_value     = base_value + cascade_bonus - rescuer_risk

CASCADE BONUS: a rescue is not just one life -- if the belief has a live
phone (proof: 'rf' in belief.detected_by, which only ever fires for
phone_state=="alive" victims -- no ground truth needed) and sits near a
suspected silent zone OR is currently out of range of every drone (the
engine's own inferable proxy for "not on the mesh right now"), physically
reaching it extends coverage. The bonus is proportional to how much
never-scanned prior population sits within that reach.

RESCUER RISK PENALTY: subtracted from value. Fast-rising water is
dangerous for a boat crew; a collapsed structure (env_type == "buried"
already IS that signal) is dangerous for excavator/medical crews.

Assignment is GREEDY by final_value, respecting resource-type
constraints: boat -> rooftop only; medical alone -> street only;
excavator + medical dispatched TOGETHER, simultaneously -> buried only
(both must arrive before extraction can start -- arrival_minutes for a
bundle is the SLOWER of the two).

Every committed Assignment's `reason` names the runner-up candidate that
resource could have served instead, and whether that runner-up remains
reachable by some other still-idle resource afterward.

/engine must never import from /simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config import params
from engine import connectivity, models
from engine.belief import BeliefTracker
from engine.contracts import Assignment, Drone, Resource, VictimBelief

import numpy as np


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _travel_minutes(pos_a: Tuple[float, float], pos_b: Tuple[float, float], speed_m_per_min: float) -> float:
    return _dist(pos_a, pos_b) / max(speed_m_per_min, 1e-6)


@dataclass
class Candidate:
    resources: Tuple[Resource, ...]  # 1 for boat/medical, 2 (excavator, medical) for a buried bundle
    belief: VictimBelief
    arrival_minutes: float
    p_survive: float
    base_value: float
    cascade_bonus: float
    rescuer_risk: float
    final_value: float

    @property
    def resource_ids(self) -> Tuple[str, ...]:
        return tuple(r.resource_id for r in self.resources)


def _rescuer_risk(belief: VictimBelief, environment: Dict[str, float]) -> float:
    if belief.env_type == "rooftop":
        return params.RESCUER_RISK_BOAT_BASE + params.RESCUER_RISK_WATER_COEFF * environment.get("rise_rate", 0.0)
    if belief.env_type == "buried":
        return params.RESCUER_RISK_BURIED_BASE
    return params.RESCUER_RISK_STREET_BASE


def _local_unexplored_scale(prior_map: np.ndarray, radius: float) -> float:
    """
    Mean prior population expected within a disc of this radius, if the
    map's density were uniform -- the LOCAL yardstick trigger-A's bonus
    is normalised against, instead of the map's total population.

    PROBLEM 2(a) fix: the original version divided by prior_map.sum()
    (the ENTIRE 2km map), which makes any local neighbourhood's
    contribution vanishingly small by construction -- a 300m disc is a
    tiny fraction of a 2000x2000m map regardless of how rich it is.
    Normalising locally means a genuinely dense, fully-unexplored pocket
    can reach close to ENGINE_CASCADE_BONUS_MAX, the same order of
    magnitude as base_value, as intended.
    """
    disc_area_cells = math.pi * (radius / params.CELL_SIZE_M) ** 2
    mean_prior_per_cell = float(prior_map.mean())
    return max(mean_prior_per_cell * disc_area_cells, 1e-9)


def _cascade_bonus(belief: VictimBelief, drones: List[Drone], silent_zones: List[Tuple[float, float, float]],
                    tracker: BeliefTracker, prior_map: np.ndarray, has_backhaul: Dict[str, bool]) -> float:
    if "rf" not in belief.detected_by:
        return 0.0  # spec: cascade requires "live phones"

    bonus = 0.0

    # Trigger A: near a suspected silent zone, or currently out of range
    # of every drone (the engine's own inferable proxy for "not on the
    # mesh right now"). Rewards reaching never-scanned population nearby.
    # This pool shrinks as exploration completes (Phase 6's orchestrator
    # reaches ~97% coverage by hour 5) -- see trigger B below for the
    # PROBLEM 2(b) fix, a longer-lived complement.
    near_silent_zone = any(
        _dist(belief.position, (x, y)) <= params.ENGINE_CASCADE_RADIUS_M for x, y, _ in silent_zones
    )
    nearest_drone_dist = min((_dist(belief.position, d.position) for d in drones), default=float("inf"))
    is_unconnected_to_drone = nearest_drone_dist > params.DRONE_CELL_RANGE_M
    if near_silent_zone or is_unconnected_to_drone:
        unexplored = tracker.unexplored_population_within(belief.position, params.ENGINE_CASCADE_RADIUS_M, prior_map)
        local_scale = _local_unexplored_scale(prior_map, params.ENGINE_CASCADE_RADIUS_M)
        bonus += params.ENGINE_CASCADE_BONUS_MAX * min(unexplored / local_scale, 1.0)

    # Trigger B (PROBLEM 2(b)): proximity to a drone that currently has NO
    # multi-hop path to base. has_backhaul is reconstructed by
    # engine.connectivity from Drone.linked_to alone (self-reported
    # one-hop telemetry) -- no ground truth, no import from /simulation.
    # This condition doesn't evaporate once exploration completes: drones
    # can lose backhaul at any point in the run as they range further out,
    # so unlike trigger A it stays a live incentive for the whole 8h.
    near_disconnected_drone = any(
        (not has_backhaul.get(d.drone_id, True)) and _dist(belief.position, d.position) <= params.ENGINE_CASCADE_RADIUS_M
        for d in drones
    )
    if near_disconnected_drone:
        bonus += params.ENGINE_CASCADE_DISCONNECTED_BONUS

    return bonus


def _score(resources: Tuple[Resource, ...], belief: VictimBelief, arrival_minutes: float,
           medical_present: bool, environment: Dict[str, float], drones: List[Drone],
           silent_zones: List[Tuple[float, float, float]], tracker: BeliefTracker,
           prior_map: np.ndarray, has_backhaul: Dict[str, bool],
           use_cascade_bonus: bool = True, use_rescuer_risk: bool = True) -> Candidate:
    water_at_first_seen = tracker.get_water_level_at_first_seen(belief.belief_id)
    # [REALISM FIX] survival must be evaluated at the time the victim is
    # actually SAFE, not merely when the resource arrives -- arrival is
    # followed by an uncertainty-scaled search and then extraction (see
    # models.extraction_minutes / models.search_minutes). Using travel
    # time alone here would let the scheduler ignore the real cost of
    # dispatching to a vague, high-uncertainty belief.
    time_to_safety = arrival_minutes + belief.extraction_minutes
    p_survive = models.survival_probability(
        belief, time_to_safety, environment,
        medical_team_present=medical_present, water_level_at_first_seen=water_at_first_seen,
    )
    base_value = p_survive * belief.est_group_size * belief.confidence
    cascade = _cascade_bonus(belief, drones, silent_zones, tracker, prior_map, has_backhaul) if use_cascade_bonus else 0.0
    risk = _rescuer_risk(belief, environment) if use_rescuer_risk else 0.0
    return Candidate(resources, belief, arrival_minutes, p_survive, base_value, cascade, risk, base_value + cascade - risk)


def build_candidates(beliefs: List[VictimBelief], resources: List[Resource], environment: Dict[str, float],
                      drones: List[Drone], silent_zones: List[Tuple[float, float, float]],
                      tracker: BeliefTracker, prior_map: np.ndarray,
                      use_cascade_bonus: bool = True, use_rescuer_risk: bool = True) -> List[Candidate]:
    idle = [r for r in resources if r.status == "idle"]
    boats = [r for r in idle if r.type == "boat"]
    excavators = [r for r in idle if r.type == "excavator"]
    medicals = [r for r in idle if r.type == "medical"]
    # Computed ONCE per schedule() call, not once per candidate --
    # cascade bonus's trigger B needs this, and it's the same for every
    # candidate this tick regardless of which belief/resource is being scored.
    has_backhaul = connectivity.compute_backhaul(drones)

    candidates: List[Candidate] = []
    for belief in beliefs:
        if belief.env_type == "rooftop":
            for r in boats:
                arrival = _travel_minutes(r.position, belief.position, r.speed)
                candidates.append(_score((r,), belief, arrival, False, environment, drones, silent_zones, tracker, prior_map, has_backhaul, use_cascade_bonus, use_rescuer_risk))

        elif belief.env_type == "street":
            for r in medicals:
                arrival = _travel_minutes(r.position, belief.position, r.speed)
                candidates.append(_score((r,), belief, arrival, True, environment, drones, silent_zones, tracker, prior_map, has_backhaul, use_cascade_bonus, use_rescuer_risk))

        elif belief.env_type == "buried":
            # Bundle only -- a lone excavator can never rescue a buried
            # belief in this model, per spec ("requires excavator AND
            # medical simultaneously"). arrival_minutes is the slower of
            # the two, since work can't start until both are on scene.
            for exc in excavators:
                for med in medicals:
                    arrival = max(
                        _travel_minutes(exc.position, belief.position, exc.speed),
                        _travel_minutes(med.position, belief.position, med.speed),
                    )
                    candidates.append(_score((exc, med), belief, arrival, True, environment, drones, silent_zones, tracker, prior_map, has_backhaul, use_cascade_bonus, use_rescuer_risk))

        # env_type == "unknown": no resource type applies with confidence, skip.
    return candidates


def _format_window(minutes: float) -> str:
    if minutes == float("inf"):
        return "no deadline"
    return f"{minutes:.0f} min window"


def _resource_label(resources: Tuple[Resource, ...]) -> str:
    if len(resources) == 1:
        r = resources[0]
        return f"{r.type.capitalize()} {r.resource_id}"
    return " + ".join(f"{r.type.capitalize()} {r.resource_id}" for r in resources)


def _build_reason(chosen: Candidate, runner_up: Optional[Candidate], used_resource_ids_after: set,
                   all_candidates: List[Candidate]) -> str:
    resource_label = _resource_label(chosen.resources)
    chosen_window = _format_window(chosen.belief.survival_deadline)
    base = (f"{resource_label} -> {chosen.belief.belief_id} ({chosen_window}, "
            f"{chosen.arrival_minutes:.0f} min travel, value={chosen.final_value:.2f})")

    if runner_up is None:
        return base + " -- no comparable alternative for this resource this round."

    alt = runner_up.belief
    comparison = "nearer" if runner_up.arrival_minutes < chosen.arrival_minutes else "farther"
    alt_window = _format_window(alt.survival_deadline)

    still_reachable = any(
        cand.belief.belief_id == alt.belief_id and not any(rid in used_resource_ids_after for rid in cand.resource_ids)
        for cand in all_candidates
    )
    reach_note = "remains reachable by another resource" if still_reachable else "would no longer be reachable this round"

    return (f"{base} over {comparison} {alt.belief_id} ({alt_window}, value={runner_up.final_value:.2f}); "
            f"{alt.belief_id} {reach_note}.")


def schedule(beliefs: List[VictimBelief], resources: List[Resource], environment: Dict[str, float],
             drones: List[Drone], silent_zones: List[Tuple[float, float, float]],
             tracker: BeliefTracker, prior_map: np.ndarray,
             use_cascade_bonus: bool = True, use_rescuer_risk: bool = True) -> List[Assignment]:
    candidates = build_candidates(beliefs, resources, environment, drones, silent_zones, tracker, prior_map,
                                   use_cascade_bonus, use_rescuer_risk)
    candidates.sort(key=lambda c: -c.final_value)

    by_resource: Dict[str, List[Candidate]] = {}
    for c in candidates:
        for rid in c.resource_ids:
            by_resource.setdefault(rid, []).append(c)

    used_resource_ids: set = set()
    used_belief_ids: set = set()
    assignments: List[Assignment] = []

    for c in candidates:
        if c.belief.belief_id in used_belief_ids:
            continue
        if any(rid in used_resource_ids for rid in c.resource_ids):
            continue

        runner_up: Optional[Candidate] = None
        for alt in by_resource.get(c.resource_ids[0], []):
            if alt.belief.belief_id == c.belief.belief_id:
                continue
            if any(rid in used_resource_ids for rid in alt.resource_ids):
                continue
            if runner_up is None or alt.final_value > runner_up.final_value:
                runner_up = alt

        used_after = used_resource_ids | set(c.resource_ids)
        reason = _build_reason(c, runner_up, used_after, candidates)

        for r in c.resources:
            assignments.append(Assignment(
                resource_id=r.resource_id,
                target_belief_id=c.belief.belief_id,
                expected_value=c.final_value,
                reason=reason,
            ))

        used_belief_ids.add(c.belief.belief_id)
        used_resource_ids |= set(c.resource_ids)

    return assignments
