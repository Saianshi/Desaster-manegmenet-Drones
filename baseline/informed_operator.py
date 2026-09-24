"""
baseline/informed_operator.py -- a competent human operator with a
decent situational map.

Sits strictly between engine.engine.Engine (full smarts) and
baseline.nearest_first.NearestFirstEngine (no smarts at all). It exists
to isolate what belief fusion + correct resource-type matching +
no-double-dispatch buy on their own, separate from survival-aware
scoring, cascade bonus, rescuer risk, and connectivity-aware drone
routing -- without those, the pure_baseline vs full_engine gap conflates
four independent weaknesses (no fusion, wrong-type dispatch, resource
contention, blind flight) into one number, which reads as a strawman.

GETS:
  - real belief fusion: this operator uses engine.belief.BeliefTracker
    DIRECTLY, the same class the full engine uses. Fusion is part of the
    SENSING stack (turning noisy detections into a coherent situational
    picture), not the decision logic under test -- reusing it here is
    what makes the comparison fair: this operator sees exactly what the
    full engine sees, and still only makes much simpler decisions with it.
  - correct resource-type matching: boat -> rooftop, excavator + medical
    bundle -> buried (dispatched together, never a lone excavator),
    medical -> street.
  - no double-assignment: each idle resource and each belief is used at
    most once per tick.

DOES NOT GET:
  - survival modelling or any arrival-time-aware scoring
  - cascade bonus
  - rescuer risk penalty
  - connectivity-aware drone routing (flies the fixed lawnmower pattern)

Rule: for each idle resource, dispatch to the NEAREST valid belief of a
type it can service. Buried beliefs are handled first as excavator+
medical pairs (nearest available excavator, nearest available medical);
remaining idle resources then go single-handed to the nearest compatible
rooftop/street belief. No sorting by confidence, deadline, or anything
else -- that would be a step toward the scoring this operator explicitly
lacks.

Implements the same EngineInput -> EngineOutput interface as
engine.engine.Engine, for a drop-in swap in simulation/harness.py.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from baseline.lawnmower import lawnmower_position
from config import params
from engine.belief import BeliefTracker
from engine.contracts import Assignment, EngineInput, EngineOutput, VictimBelief


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class InformedOperatorEngine:
    def __init__(self, seed: int = 0):
        self._belief_tracker = BeliefTracker(seed=seed)

    def get_belief_positions(self) -> Dict[str, Tuple[float, float]]:
        return self._belief_tracker.get_belief_positions()

    def get_belief_uncertainties(self) -> Dict[str, float]:
        return self._belief_tracker.get_belief_uncertainties()

    def decide(self, inp: EngineInput) -> EngineOutput:
        self._belief_tracker.update(inp)
        beliefs = self._belief_tracker.get_beliefs()

        n_drones = len(inp.drones)
        map_size = inp.prior_map.shape[0] * params.CELL_SIZE_M
        drone_waypoints = {
            d.drone_id: lawnmower_position(i, n_drones, inp.t, map_size)
            for i, d in enumerate(inp.drones)
        }

        assignments = self._dispatch(beliefs, inp.resources)

        return EngineOutput(
            t=inp.t,
            drone_waypoints=drone_waypoints,
            assignments=assignments,
            priority_list=[b.belief_id for b in beliefs],
            suspected_silent_zones=[],
        )

    def _dispatch(self, beliefs: List[VictimBelief], resources) -> List[Assignment]:
        idle = [r for r in resources if r.status == "idle"]
        used_belief_ids: set = set()
        used_resource_ids: set = set()
        assignments: List[Assignment] = []

        # Buried first: needs excavator + medical simultaneously. A lone
        # excavator (or lone medical) can never complete a buried rescue
        # in this model, so both-or-nothing per belief.
        idle_excavators = [r for r in idle if r.type == "excavator"]
        idle_medicals = [r for r in idle if r.type == "medical"]
        for belief in beliefs:
            if belief.env_type != "buried" or belief.belief_id in used_belief_ids:
                continue
            avail_exc = [r for r in idle_excavators if r.resource_id not in used_resource_ids]
            avail_med = [r for r in idle_medicals if r.resource_id not in used_resource_ids]
            if not avail_exc or not avail_med:
                continue
            exc = min(avail_exc, key=lambda r: _dist(r.position, belief.position))
            med = min(avail_med, key=lambda r: _dist(r.position, belief.position))
            reason = (f"informed operator: {exc.resource_id} + {med.resource_id} -> {belief.belief_id} "
                      f"(buried: nearest available excavator+medical pair, no survival scoring)")
            assignments.append(Assignment(resource_id=exc.resource_id, target_belief_id=belief.belief_id, expected_value=0.0, reason=reason))
            assignments.append(Assignment(resource_id=med.resource_id, target_belief_id=belief.belief_id, expected_value=0.0, reason=reason))
            used_resource_ids.add(exc.resource_id)
            used_resource_ids.add(med.resource_id)
            used_belief_ids.add(belief.belief_id)

        # Single-resource dispatch: boat -> rooftop, medical -> street.
        # Leftover excavators (no medical partner available this tick) do
        # nothing -- an informed operator wouldn't send one alone.
        for r in idle:
            if r.resource_id in used_resource_ids or r.type == "excavator":
                continue
            valid_env_types = {"rooftop"} if r.type == "boat" else {"street"} if r.type == "medical" else set()
            candidates = [b for b in beliefs if b.env_type in valid_env_types and b.belief_id not in used_belief_ids]
            if not candidates:
                continue
            nearest = min(candidates, key=lambda b: _dist(r.position, b.position))
            assignments.append(Assignment(
                resource_id=r.resource_id, target_belief_id=nearest.belief_id, expected_value=0.0,
                reason=f"informed operator: {r.resource_id} -> {nearest.belief_id} (nearest valid {r.type} target, no survival/risk scoring)",
            ))
            used_belief_ids.add(nearest.belief_id)
            used_resource_ids.add(r.resource_id)

        return assignments


if __name__ == "__main__":
    from simulation.harness import run_scenario

    m = run_scenario(InformedOperatorEngine(seed=1), seed=params.RANDOM_SEED)
    print("informed operator, one run:")
    for k, v in m.items():
        print(f"  {k}: {v}")
