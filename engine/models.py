"""
Survival and extraction-time models -- ENGINE'S OWN ESTIMATES, not ground
truth. /engine must never import from /simulation, so everything here is
built from what a VictimBelief and the current EngineInput.environment
actually contain: a fused position, an env_type classification the
engine derived itself, a confidence score, and the current water level /
rise rate. It never sees a victim's true injury_severity or true building
elevation -- see config/params.py's "ENGINE-SIDE MODEL ASSUMPTIONS"
section for exactly which numbers are guesses and why.

Both hazard curves use the same math as the simulation's ground-truth
process (constant-rate/exponential hazard, see simulation/victims.py),
because that discretisation is just the standard way to turn a "median
survival time" into a per-minute death probability -- reusing the
technique isn't reusing the answer. The actual TIME CONSTANTS the engine
plugs in are its own best guesses, and can be wrong.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from config import params
from engine.contracts import VictimBelief

LOG2 = math.log(2.0)


def _buried_median_survival_hours() -> float:
    # Engine has no per-victim injury_severity, so it uses the
    # population-average assumption (ENGINE_ASSUMED_INJURY_SEVERITY) in
    # the same formula the simulation itself uses to generate ground
    # truth (median survival hours falls linearly with assumed severity).
    return params.BURIED_MEDIAN_SURVIVAL_HOURS_BASE * (
        1 - params.BURIED_SEVERITY_SURVIVAL_PENALTY * params.ENGINE_ASSUMED_INJURY_SEVERITY
    )


def _waiting_survival_prob_buried(t_hours: float) -> float:
    """P(still alive) after waiting t_hours, from crush syndrome alone,
    with no rescue yet. t_hours: hours from now. Exponential hazard:
    hazard_per_hour = ln(2) / median_survival_hours."""
    median_hours = _buried_median_survival_hours()
    hazard_per_hour = LOG2 / max(median_hours, 0.01)
    return math.exp(-hazard_per_hour * max(t_hours, 0.0))


def _time_to_submersion_minutes(environment: Dict[str, float], water_level_at_first_seen: float) -> Optional[float]:
    """
    Minutes from now until the predicted water surface reaches this
    belief's ASSUMED elevation (water_level_at_first_seen +
    ENGINE_ASSUMED_ROOFTOP_CLEARANCE_M), given the current water_level
    and rise_rate (m/hour) from EngineInput.environment.

    Returns None if the water is not rising and hasn't reached the
    victim yet (never reached, under current conditions).
    """
    assumed_elevation_m = water_level_at_first_seen + params.ENGINE_ASSUMED_ROOFTOP_CLEARANCE_M
    threshold_m = assumed_elevation_m - params.SUBMERSION_BUFFER_M
    current_water_m = environment["water_level"]
    rise_rate_m_per_hr = environment["rise_rate"]

    if current_water_m >= threshold_m:
        return 0.0
    if rise_rate_m_per_hr <= 0:
        return None
    hours_to_cross = (threshold_m - current_water_m) / rise_rate_m_per_hr
    return hours_to_cross * 60.0


def _waiting_survival_prob_rooftop(arrival_minutes: float, environment: Dict[str, float], water_level_at_first_seen: float) -> float:
    """P(still alive) at arrival_minutes from now, from the flood hazard
    alone. Safe (p=1) until the predicted water surface crosses the
    belief's assumed elevation; a short, steep hazard window afterward
    (drowning risk), matching the "scale of minutes once water is close"
    behaviour requested for the ground-truth model."""
    t_cross = _time_to_submersion_minutes(environment, water_level_at_first_seen)
    if t_cross is None or arrival_minutes <= t_cross:
        return 1.0
    exposure_hours = (arrival_minutes - t_cross) / 60.0
    median_hours_after = params.ROOFTOP_MEDIAN_SURVIVAL_MIN_AFTER_SUBMERSION / 60.0
    hazard_per_hour = LOG2 / max(median_hours_after, 0.001)
    return math.exp(-hazard_per_hour * exposure_hours)


def survival_probability(
    belief: VictimBelief,
    arrival_minutes: float,
    environment: Dict[str, float],
    medical_team_present: bool = False,
    water_level_at_first_seen: Optional[float] = None,
) -> float:
    """
    P(at least one member of this belief's group is still alive) if a
    rescue arrives arrival_minutes from now. This is the number the
    Phase 5 scheduler scores candidate assignments with.

    For buried beliefs, medical_team_present controls whether the
    reperfusion-injury penalty applies AT THE MOMENT OF EXTRACTION --
    it does not affect the waiting curve itself (that hazard runs
    whether or not a medic will eventually be present).
    """
    if belief.env_type == "buried":
        p_wait = _waiting_survival_prob_buried(arrival_minutes / 60.0)
        if not medical_team_present:
            p_wait *= (1 - params.ENGINE_MEDICAL_PENALTY_NO_TEAM)
        return p_wait

    if belief.env_type == "rooftop":
        if water_level_at_first_seen is None:
            water_level_at_first_seen = environment["water_level"]
        return _waiting_survival_prob_rooftop(arrival_minutes, environment, water_level_at_first_seen)

    # street / unknown: no active hazard curve modelled -- see
    # ENGINE_DEFAULT_SURVIVAL_PROB_UNMODELED for why this is a flat
    # constant rather than a curve.
    return params.ENGINE_DEFAULT_SURVIVAL_PROB_UNMODELED


def survival_deadline_minutes(
    belief: VictimBelief,
    environment: Dict[str, float],
    water_level_at_first_seen: Optional[float] = None,
) -> float:
    """
    Minutes from now at which the WAITING survival curve (no rescue,
    no extraction outcome) crosses 50%. This is a summary number for
    display/priority sorting (VictimBelief.survival_deadline) -- the
    scheduler should call survival_probability() directly for actual
    arrival-time-dependent scoring, not re-derive it from this deadline.
    """
    if belief.env_type == "buried":
        return _buried_median_survival_hours() * 60.0

    if belief.env_type == "rooftop":
        if water_level_at_first_seen is None:
            water_level_at_first_seen = environment["water_level"]
        t_cross = _time_to_submersion_minutes(environment, water_level_at_first_seen)
        if t_cross is None:
            return float("inf")
        return t_cross + params.ROOFTOP_MEDIAN_SURVIVAL_MIN_AFTER_SUBMERSION

    return float("inf")  # street / unknown: no modelled deadline


def search_minutes(belief: VictimBelief) -> float:
    """
    [REALISM FIX, post-Phase-7 diagnostic] Expected time a team spends
    physically searching once on scene, before extraction can even start.
    A fixed search radius against an uncertain (e.g. RF-derived,
    position_sigma up to 40m) belief was the dominant cause of mission
    failure -- teams don't search a fixed disc regardless of how good the
    fix is; they scale the search to the reported uncertainty, and it
    costs them time to do it.

    search_radius_m: ~2x the belief's own reported uncertainty, floored
    at 50m (never search less than that even for a tight fix) and capped
    at 150m (beyond that a team calls it a false lead, not an ever-wider
    search). search_minutes scales with the AREA searched (radius^2), not
    the radius itself, so a vague belief costs meaningfully more responder
    time, not just a wider miss chance -- this is what lets the scheduler
    price localisation quality into arrival_minutes via extraction_minutes
    (see engine/scheduler.py's use of belief.extraction_minutes).
    """
    radius = min(
        max(params.RESCUE_SEARCH_RADIUS_UNCERTAINTY_MULTIPLIER * belief.uncertainty, params.RESCUE_SEARCH_RADIUS_FLOOR_M),
        params.RESCUE_SEARCH_RADIUS_CAP_M,
    )
    return params.SEARCH_TIME_AT_FLOOR_MIN * (radius / params.RESCUE_SEARCH_RADIUS_FLOOR_M) ** 2


def extraction_minutes(belief: VictimBelief) -> float:
    """
    Estimated time from arrival on scene to the victim actually being
    safe: search time (see search_minutes -- always incurred, since a
    team has to find someone before extracting them) plus the physical
    extraction itself. Units: minutes.
      - rooftop (boat rescue): ~flat, per spec.
      - buried (debris extraction): scales with group_size, 45min for a
        single person up to 90min for a large group (linear interpolation
        between group_size=1 and a reference "large group" of 12).
      - street: no specialised extraction, just evacuation assistance.
    """
    if belief.env_type == "rooftop":
        base = params.EXTRACTION_BOAT_MINUTES
    elif belief.env_type == "buried":
        gs = max(belief.est_group_size, 1)
        span = params.DEBRIS_EXTRACTION_MAX_MIN - params.DEBRIS_EXTRACTION_BASE_MIN
        frac = min(gs - 1, 11) / 11.0  # reference large group = 12 people
        base = params.DEBRIS_EXTRACTION_BASE_MIN + span * frac
    else:
        base = params.STREET_EXTRACTION_MINUTES

    return base + search_minutes(belief)


def required_resources(belief: VictimBelief) -> List[str]:
    """Resource types needed to complete a rescue of this belief. Buried
    victims need excavator AND medical SIMULTANEOUSLY (medical is there
    for the reperfusion-injury risk at the moment of extraction, not just
    transport) -- see Phase 5 for how the scheduler enforces "simultaneously"."""
    if belief.env_type == "rooftop":
        return ["boat"]
    if belief.env_type == "buried":
        return ["excavator", "medical"]
    return ["medical"]  # street: assume any survivors found need at least a medical check
