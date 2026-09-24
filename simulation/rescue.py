"""
Resolves what ACTUALLY happens when the engine dispatches a resource to
an ESTIMATED belief position -- this is where engine uncertainty has
real consequences. A misclassified env_type sends the wrong resource
type; a belief position that's off searches right past the real victim
unless the search is wide (and slow) enough to cover the actual error.

State machine per dispatched resource:
    traveling -> searching -> [waiting_for_partner, buried only] -> working -> done

[REALISM FIX, post-Phase-7 diagnostic] Two changes from the original
version, both about making mission RESOLUTION realistic rather than
instantaneous:

  1. On arrival, a resource doesn't resolve success/failure on the spot.
     It SEARCHES a radius scaled to the belief's own reported uncertainty
     (frozen at dispatch time, like target_pos already was), and that
     search takes time proportional to the area covered -- see
     engine.models.search_minutes for the formula both this module and
     the scheduler's own value calculation are built from.
  2. On a wrong_resource_type failure, the crew is physically standing at
     the site and radios back what they found: a high-confidence,
     well-localised Detection carrying the CORRECTED env_type. Real
     responders don't just walk away and let the same wrong dispatch
     repeat forever. Queued via pop_field_reports() -- the harness routes
     it through EngineInput like any other detection, on the very next
     tick, exactly as a real radio report would arrive.

Owned by World (composition), kept in its own module because the
travel/search/pairing/extraction logic would otherwise clutter world.py.

/simulation is allowed to import from engine.contracts (Assignment,
Detection) -- it must never import engine.engine / engine.belief /
engine.models. The resource-type-compatibility rule and extraction-time
constants used here are the same PUBLIC, real-world operational
parameters the engine's own estimate is built from (see
victims.true_extraction_minutes) -- using them here is not a
ground-truth leak into the engine, it's the reverse: the simulation
drawing on publicly-known operational timings, same as it already does
for sensor ranges in sensors.py.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Set, Tuple

from config import params
from engine.contracts import Assignment, Detection, Resource, Victim
from simulation.victims import true_extraction_minutes

RESOURCE_ENV_COMPAT = {
    "boat": {"rooftop"},
    "excavator": {"buried"},
    "medical": {"buried", "street"},
}


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _search_radius_and_minutes(uncertainty: float) -> Tuple[float, float]:
    radius = min(
        max(params.RESCUE_SEARCH_RADIUS_UNCERTAINTY_MULTIPLIER * uncertainty, params.RESCUE_SEARCH_RADIUS_FLOOR_M),
        params.RESCUE_SEARCH_RADIUS_CAP_M,
    )
    minutes = params.SEARCH_TIME_AT_FLOOR_MIN * (radius / params.RESCUE_SEARCH_RADIUS_FLOOR_M) ** 2
    return radius, minutes


class RescueResolver:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.rescued_ids: Set[str] = set()
        self._active: Dict[str, dict] = {}  # resource_id -> state
        self._pending_field_reports: List[Detection] = []
        self._field_report_counter = 0

    # -- dispatch: called right after engine.decide(), like world.apply() ---
    def dispatch(self, assignments: List[Assignment], resources: List[Resource],
                 belief_positions: Dict[str, Tuple[float, float]],
                 belief_uncertainties: Dict[str, float], t: float) -> None:
        by_id = {r.resource_id: r for r in resources}
        for a in assignments:
            resource = by_id.get(a.resource_id)
            if resource is None or resource.status != "idle":
                continue
            target_pos = belief_positions.get(a.target_belief_id)
            if target_pos is None:
                continue
            # Frozen at dispatch time, same principle as target_pos: this
            # is the information the team had when they left, not
            # whatever the belief has since been refined to.
            uncertainty = belief_uncertainties.get(a.target_belief_id, params.RESCUE_SEARCH_RADIUS_FLOOR_M)
            search_radius, search_minutes = _search_radius_and_minutes(uncertainty)

            travel_min = _dist(resource.position, target_pos) / max(resource.speed, 1e-6)
            resource.status = "enroute"
            self._active[resource.resource_id] = {
                "belief_id": a.target_belief_id,
                "start_pos": resource.position,
                "target_pos": target_pos,
                "start_t": t,
                "eta": t + travel_min * 60.0,
                "phase": "traveling",
                "resource_type": resource.type,
                "search_radius": search_radius,
                "search_minutes": search_minutes,
            }

    # -- advance: called from World.step(), like the victim hazard update --
    def step(self, resources: List[Resource], victims: List[Victim], t: float) -> List[dict]:
        by_id = {r.resource_id: r for r in resources}
        victims_by_id = {v.victim_id: v for v in victims}
        events: List[dict] = []

        for rid, state in list(self._active.items()):
            resource = by_id[rid]

            if state["phase"] == "traveling":
                if t >= state["eta"]:
                    resource.position = state["target_pos"]
                    state["phase"] = "searching"
                    state["search_done_t"] = t + state["search_minutes"] * 60.0
                else:
                    span = max(state["eta"] - state["start_t"], 1e-6)
                    frac = (t - state["start_t"]) / span
                    sx, sy = state["start_pos"]
                    tx, ty = state["target_pos"]
                    resource.position = (sx + (tx - sx) * frac, sy + (ty - sy) * frac)

            elif state["phase"] == "searching":
                if t >= state["search_done_t"]:
                    self._resolve_search(rid, resource, state, victims, t, events)

            elif state["phase"] == "waiting_for_partner":
                if t - state["waiting_since"] > params.PARTNER_WAIT_TIMEOUT_MIN * 60.0:
                    resource.status = "idle"
                    resource.free_at = t
                    events.append({"t": t, "resource_id": rid, "outcome": "partner_timeout", "belief_id": state["belief_id"]})
                    del self._active[rid]

            elif state["phase"] == "working":
                if t >= state["work_done_t"]:
                    self._complete(rid, resource, state, victims_by_id, t, events)

        return events

    def pop_field_reports(self) -> List[Detection]:
        out = self._pending_field_reports
        self._pending_field_reports = []
        return out

    # -- internal state transitions ------------------------------------
    def _resolve_search(self, rid, resource, state, victims, t, events) -> None:
        candidates = [
            v for v in victims
            if v.alive and v.victim_id not in self.rescued_ids
            and _dist(v.true_position, state["target_pos"]) <= state["search_radius"]
        ]
        if not candidates:
            resource.status = "idle"
            resource.free_at = t
            events.append({"t": t, "resource_id": rid, "outcome": "no_victim_found", "belief_id": state["belief_id"]})
            del self._active[rid]
            return

        victim = min(candidates, key=lambda v: _dist(v.true_position, state["target_pos"]))
        if victim.env_type not in RESOURCE_ENV_COMPAT.get(state["resource_type"], set()):
            resource.status = "idle"
            resource.free_at = t
            events.append({
                "t": t, "resource_id": rid, "outcome": "wrong_resource_type",
                "belief_id": state["belief_id"], "true_env_type": victim.env_type,
            })
            self._emit_field_report(state["belief_id"], victim, t)
            del self._active[rid]
            return

        state["victim_id"] = victim.victim_id
        if victim.env_type == "buried":
            partner_rid = self._find_waiting_partner(state)
            if partner_rid is not None:
                self._begin_work(rid, state, victim, t, partner_rid=partner_rid)
            else:
                state["phase"] = "waiting_for_partner"
                state["waiting_since"] = t
        else:
            self._begin_work(rid, state, victim, t)

    def _emit_field_report(self, belief_id: str, victim: Victim, t: float) -> None:
        """The crew is physically standing at the site -- they radio back
        what they actually found. Small position_sigma (they're right
        there), high confidence (direct human observation, not a
        probabilistic remote sensor read), env_hint = the TRUE type.
        Queued for the harness to deliver on the next tick, like a real
        radio report; NOT injected via the drone/comms pipeline, since a
        field radio is a direct channel independent of drone backhaul."""
        self._field_report_counter += 1
        self._pending_field_reports.append(Detection(
            detection_id=f"field_{belief_id}_{self._field_report_counter}",
            est_position=victim.true_position,
            position_sigma=params.FIELD_REPORT_SIGMA_M,
            sensor="field_report",
            confidence=params.FIELD_REPORT_CONFIDENCE,
            timestamp=t,
            env_hint=victim.env_type,
        ))

    def _find_waiting_partner(self, state: dict) -> Optional[str]:
        need = "medical" if state["resource_type"] == "excavator" else "excavator"
        for other_rid, other in self._active.items():
            if (other["belief_id"] == state["belief_id"] and other["resource_type"] == need
                    and other["phase"] == "waiting_for_partner"):
                return other_rid
        return None

    def _begin_work(self, rid: str, state: dict, victim: Victim, t: float, partner_rid: Optional[str] = None) -> None:
        extraction_min = true_extraction_minutes(victim)
        work_done_t = t + extraction_min * 60.0
        state["phase"] = "working"
        state["work_done_t"] = work_done_t
        state["victim_id"] = victim.victim_id
        state["medical_present"] = (state["resource_type"] == "medical") or (partner_rid is not None)
        if partner_rid is not None:
            partner = self._active[partner_rid]
            partner["phase"] = "working"
            partner["work_done_t"] = work_done_t
            partner["victim_id"] = victim.victim_id
            partner["medical_present"] = True

    def _complete(self, rid: str, resource: Resource, state: dict, victims_by_id: Dict[str, Victim], t: float, events: List[dict]) -> None:
        resource.status = "idle"
        resource.free_at = t
        victim = victims_by_id.get(state["victim_id"])
        del self._active[rid]

        if victim is None or not victim.alive or victim.victim_id in self.rescued_ids:
            events.append({"t": t, "resource_id": rid, "outcome": "victim_lost_during_extraction", "belief_id": state["belief_id"]})
            return

        if victim.env_type == "buried" and not state.get("medical_present", False):
            # Reperfusion injury risk (see ENGINE_MEDICAL_PENALTY_NO_TEAM --
            # this is real physiology, not an engine-side guess, so it's
            # legitimate to reuse the same constant for the true outcome).
            if self.rng.random() >= (1 - params.ENGINE_MEDICAL_PENALTY_NO_TEAM):
                victim.alive = False
                events.append({
                    "t": t, "resource_id": rid, "outcome": "died_during_extraction_no_medical",
                    "belief_id": state["belief_id"], "victim_id": victim.victim_id,
                })
                return

        self.rescued_ids.add(victim.victim_id)
        events.append({
            "t": t, "resource_id": rid, "outcome": "rescued",
            "belief_id": state["belief_id"], "victim_id": victim.victim_id,
        })
