"""
Victim generation and per-timestep victim lifecycle (ground truth).

This is the one module most responsible for the project's central
narrative: two populations dying on very different timescales. Buried
victims (crush syndrome) decline smoothly over hours; rooftop victims in
the flood zone are safe until the rising water reaches their elevation,
then face a short, steep hazard window (minutes). Street-level victims
carry no active death timer in this model -- see the ASSUMPTION note in
update_victims().

Every Victim here is really a GROUP (group_size 1-12, see
config.params.GROUP_SIZE_WEIGHTS): a household, a floor of an apartment
block, a cluster waiting at a corner. This is what gives the triage
scheduler real trade-offs later (Phase 5) between "few people, urgent"
and "many people, less urgent".

Extra ground-truth fields that don't belong in the fixed Victim contract
(elevation, hazard parameters, individual battery drain rate) are kept in
a parallel `meta` dict keyed by victim_id, entirely on the simulation
side. The engine never sees Victim or this meta dict -- only Detections.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import numpy as np

from config import params
from engine.contracts import Victim

LOG2 = math.log(2.0)


def _sample_group_size(rng: random.Random) -> int:
    sizes = list(range(1, len(params.GROUP_SIZE_WEIGHTS) + 1))
    return rng.choices(sizes, weights=params.GROUP_SIZE_WEIGHTS)[0]


def _sample_phone_state(rng: random.Random) -> str:
    return rng.choices(
        ["alive", "dead", "no_app"],
        weights=[params.FRAC_PHONE_ALIVE, params.FRAC_PHONE_DEAD_BATTERY, params.FRAC_PHONE_NO_APP],
    )[0]


def _elevation_at(elevation_map: np.ndarray, x: float, y: float, cell_size: float) -> float:
    n = elevation_map.shape[0]
    gx = min(int(x // cell_size), n - 1)
    gy = min(int(y // cell_size), n - 1)
    return float(elevation_map[gy, gx])


def _building_in_flood_zone(b, flood_zone_mask: np.ndarray, cell_size: float) -> bool:
    cx, cy = b.centroid
    n = flood_zone_mask.shape[0]
    gx = min(int(cx // cell_size), n - 1)
    gy = min(int(cy // cell_size), n - 1)
    return bool(flood_zone_mask[gy, gx])


def _make_street_sites(rng: random.Random, buildings, map_size: float, n: int) -> List[Tuple[float, float]]:
    sites = []
    tries = 0
    while len(sites) < n and tries < n * 30:
        x, y = rng.uniform(0, map_size), rng.uniform(0, map_size)
        tries += 1
        if any(b.x0 <= x <= b.x1 and b.y0 <= y <= b.y1 for b in buildings):
            continue
        sites.append((x, y))
    if not sites:
        sites = [(map_size / 2, map_size / 2)]
    return sites


def _make_group(rng: random.Random, counter: int, kind: str, site_center: Tuple[float, float],
                 cell_size: float, elevation_map: np.ndarray, flood_threshold_m: float,
                 building=None) -> Tuple[Victim, dict]:
    group_size = _sample_group_size(rng)
    map_size = elevation_map.shape[0] * cell_size
    x = min(max(site_center[0] + rng.gauss(0, params.GROUP_JITTER_SIGMA_M), 0), map_size - 0.01)
    y = min(max(site_center[1] + rng.gauss(0, params.GROUP_JITTER_SIGMA_M), 0), map_size - 0.01)

    phone_state = _sample_phone_state(rng)
    battery_pct = 0.0 if phone_state == "dead" else rng.uniform(10, 100)
    injury_severity = rng.uniform(0, 1)

    victim = Victim(
        victim_id=f"v{counter}",
        true_position=(x, y),
        env_type=kind,
        phone_state=phone_state,
        battery_pct=battery_pct,
        injury_severity=injury_severity,
        group_size=group_size,
        alive=True,
    )

    meta: Dict = {
        "kind": kind,
        "battery_drain_pct_per_hr": rng.uniform(
            params.BATTERY_DRAIN_PCT_PER_HOUR_MIN, params.BATTERY_DRAIN_PCT_PER_HOUR_MAX
        ),
    }
    if kind == "buried":
        # median_survival_hours: time at which 50% of victims at this
        # injury_severity have died from crush syndrome if never rescued.
        meta["median_survival_hours"] = params.BURIED_MEDIAN_SURVIVAL_HOURS_BASE * (
            1 - params.BURIED_SEVERITY_SURVIVAL_PENALTY * injury_severity
        )
    elif kind == "rooftop":
        floors = building.floors if building is not None else 1
        raw_terrain = building.elevation_m if building is not None else _elevation_at(elevation_map, x, y, cell_size)
        # Rescale raw terrain (0..flood_threshold_m within the flood zone,
        # by construction) down to a small physical relief band -- see
        # FLOOD_ZONE_RELIEF_M in config/params.py for why.
        ground_component = (raw_terrain / max(flood_threshold_m, 1e-6)) * params.FLOOD_ZONE_RELIEF_M
        meta["elevation_m"] = ground_component + floors * params.FLOOR_HEIGHT_M
    else:  # street
        meta["elevation_m"] = _elevation_at(elevation_map, x, y, cell_size)

    return victim, meta


def _generate_type(rng, kind, target_people, sites, cell_size, victims, meta, counter, elevation_map,
                    flood_threshold_m):
    total = 0
    is_building_site = kind in ("buried", "rooftop")
    while total < target_people:
        site = rng.choice(sites)
        building = site if is_building_site else None
        center = site.centroid if is_building_site else site
        victim, m = _make_group(rng, counter, kind, center, cell_size, elevation_map, flood_threshold_m, building=building)
        victims.append(victim)
        meta[victim.victim_id] = m
        total += victim.group_size
        counter += 1
    return counter


def generate_victims(rng: random.Random, buildings, elevation_map: np.ndarray,
                      flood_zone_mask: np.ndarray, cell_size: float):
    """
    Returns (victims: List[Victim], meta: Dict[str, dict]).

    Targets are in PEOPLE, not groups -- FRAC_BURIED/ROOFTOP/STREET of
    N_VICTIMS_FULL people are distributed across groups of size 1-12, so
    the actual number of Victim (group) records is smaller than
    N_VICTIMS_FULL and varies run to run. The last group of each type can
    overshoot its target by up to (max group size - 1) people; this is
    accepted rather than truncated so group sizes stay honest.
    """
    victims: List[Victim] = []
    meta: Dict[str, dict] = {}
    counter = 0

    collapsed_buildings = [b for b in buildings if b.collapsed] or list(buildings)
    flood_buildings = [
        b for b in buildings if not b.collapsed and _building_in_flood_zone(b, flood_zone_mask, cell_size)
    ] or list(buildings)
    map_size = elevation_map.shape[0] * cell_size
    street_sites = _make_street_sites(rng, buildings, map_size, params.N_STREET_CLUSTER_SITES)
    flood_threshold_m = float(np.percentile(elevation_map, params.FLOOD_ZONE_PERCENTILE * 100))

    target_buried = round(params.N_VICTIMS_FULL * params.FRAC_BURIED)
    target_rooftop = round(params.N_VICTIMS_FULL * params.FRAC_ROOFTOP)
    target_street = params.N_VICTIMS_FULL - target_buried - target_rooftop

    counter = _generate_type(rng, "buried", target_buried, collapsed_buildings, cell_size, victims, meta, counter, elevation_map, flood_threshold_m)
    counter = _generate_type(rng, "rooftop", target_rooftop, flood_buildings, cell_size, victims, meta, counter, elevation_map, flood_threshold_m)
    counter = _generate_type(rng, "street", target_street, street_sites, cell_size, victims, meta, counter, elevation_map, flood_threshold_m)

    return victims, meta


def update_victims(victims: List[Victim], meta: Dict[str, dict], t: float, dt: float,
                    water_level: float, rng: random.Random, rescued_ids: frozenset = frozenset()) -> List[dict]:
    """
    Advances battery state and evaluates death for every living,
    not-yet-rescued victim. Returns a list of death events:
    {"victim_id", "t", "cause"}.

    rescued_ids: victims already extracted (see simulation/rescue.py) are
    safe -- they no longer face the ongoing hazard, whether or not they
    have since died from a separate extraction-moment risk (that's
    resolved in rescue.py, not here).

    Both hazards below are modelled as a constant-rate (exponential)
    process over the timestep: given a hazard rate h (deaths per hour),
    the probability of death within dt_hours is 1 - exp(-h * dt_hours).
    This is the standard discretisation of a Poisson death process and
    is what makes "median survival hours" a well-defined input: for a
    pure exponential, h = ln(2) / median_hours.
    """
    dt_hours = dt / 3600.0
    events: List[dict] = []

    for v in victims:
        if not v.alive or v.victim_id in rescued_ids:
            continue
        m = meta[v.victim_id]

        if v.phone_state == "alive":
            v.battery_pct -= m["battery_drain_pct_per_hr"] * dt_hours
            if v.battery_pct <= 0:
                v.battery_pct = 0.0
                v.phone_state = "dead"

        if v.env_type == "buried":
            hazard_per_hour = LOG2 / max(m["median_survival_hours"], 0.01)
            p_death = 1.0 - math.exp(-hazard_per_hour * dt_hours)
            if rng.random() < p_death:
                v.alive = False
                events.append({"victim_id": v.victim_id, "t": t, "cause": "crush_syndrome"})

        elif v.env_type == "rooftop":
            elevation = m["elevation_m"]
            if water_level >= elevation - params.SUBMERSION_BUFFER_M:
                median_hours_after = params.ROOFTOP_MEDIAN_SURVIVAL_MIN_AFTER_SUBMERSION / 60.0
                hazard_per_hour = LOG2 / max(median_hours_after, 0.001)
                p_death = 1.0 - math.exp(-hazard_per_hour * dt_hours)
                if rng.random() < p_death:
                    v.alive = False
                    events.append({"victim_id": v.victim_id, "t": t, "cause": "drowning"})

        else:
            # street: [ASSUMPTION] no active over-time death hazard modelled.
            # Street-level victims are exposed but neither buried nor
            # submerged; the spec defines survival curves only for the
            # buried and rooftop-in-flood populations. Secondary hazards
            # (aftershocks, exposure) are out of scope for this model.
            pass

    return events


def true_extraction_minutes(victim: Victim) -> float:
    """
    Ground-truth extraction time, using this victim's REAL env_type/
    group_size -- used by simulation/rescue.py to resolve how long a
    dispatched resource actually takes once it has found the real
    victim(s). Mirrors engine.models.extraction_minutes exactly, but the
    engine's version necessarily runs on its own ESTIMATED group_size;
    the two will often disagree, which is the point (see
    ENGINE_GROUP_SIZE_MAX_GUESS's docstring in config/params.py).
    These extraction-time constants (EXTRACTION_BOAT_MINUTES etc) are
    public operational parameters, not ground truth about any victim --
    the same category as sensor range, safe for both sides to share.
    """
    if victim.env_type == "rooftop":
        return params.EXTRACTION_BOAT_MINUTES
    if victim.env_type == "buried":
        span = params.DEBRIS_EXTRACTION_MAX_MIN - params.DEBRIS_EXTRACTION_BASE_MIN
        frac = min(max(victim.group_size - 1, 0), 11) / 11.0
        return params.DEBRIS_EXTRACTION_BASE_MIN + span * frac
    return params.STREET_EXTRACTION_MINUTES


if __name__ == "__main__":
    from simulation.world import World

    rng = random.Random(params.RANDOM_SEED)
    world = World(rng)

    by_type: Dict[str, int] = {}
    people_by_type: Dict[str, int] = {}
    for v in world.victims:
        by_type[v.env_type] = by_type.get(v.env_type, 0) + 1
        people_by_type[v.env_type] = people_by_type.get(v.env_type, 0) + v.group_size

    print("victim GROUPS by type:", by_type)
    print("PEOPLE by type:", people_by_type, " total people:", sum(people_by_type.values()))

    sizes = sorted(v.group_size for v in world.victims)
    print("group size distribution (sorted):", sizes)
