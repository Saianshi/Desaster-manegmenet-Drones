"""
Naive baseline: what a human operator watching a screen would do.

Drones fly a fixed lawnmower pattern (baseline.lawnmower), no matter what
they've found. Every idle resource is dispatched to the nearest
currently-known detection, with NO resource-type reasoning, NO survival
modelling, NO confidence weighting, NO multi-sensor fusion, NO cascade
bonus, NO connectivity awareness, and NO coordination between resources
(two resources can and do chase the same visible target).

The only "fusion" here is merging detections within
BASELINE_TARGET_MERGE_RADIUS_M into one marker -- the minimum a human
watching blips on a screen would naturally do, not real fusion.

Implements the SAME EngineInput -> EngineOutput interface as
engine.engine.Engine (a `.decide()` method, plus `.get_belief_positions()`
for the same reason the real engine has one -- see
engine/belief.py's docstring), so it's a drop-in replacement in
simulation/harness.py's run_scenario for a fair, apples-to-apples
comparison. This is the "pure baseline" arm of Phase 7's evaluation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from config import params
from baseline.lawnmower import lawnmower_position
from engine.contracts import Assignment, EngineInput, EngineOutput


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class NearestFirstEngine:
    def __init__(self, seed: int = 0):
        self._targets: Dict[str, Tuple[float, float]] = {}
        self._uncertainties: Dict[str, float] = {}  # latest position_sigma per target, same crude "overwrite" tracking as _targets
        self._counter = 0

    def get_belief_positions(self) -> Dict[str, Tuple[float, float]]:
        return dict(self._targets)

    def get_belief_uncertainties(self) -> Dict[str, float]:
        # Fairness: the baseline gets the same uncertainty-aware search
        # radius the real engine gets (simulation/rescue.py applies this
        # identically to both) -- it just has no smarter way to USE the
        # number than the real engine's belief fusion does.
        return dict(self._uncertainties)

    def decide(self, inp: EngineInput) -> EngineOutput:
        # -- "fusion": just merge nearby detections into one marker --
        for det in inp.detections:
            matched = None
            for tid, pos in self._targets.items():
                if _dist(pos, det.est_position) <= params.BASELINE_TARGET_MERGE_RADIUS_M:
                    matched = tid
                    break
            if matched is not None:
                self._targets[matched] = det.est_position  # overwrite with latest reading, no smoothing
                self._uncertainties[matched] = det.position_sigma
            else:
                self._counter += 1
                tid = f"target_{self._counter}"
                self._targets[tid] = det.est_position
                self._uncertainties[tid] = det.position_sigma

        # -- drones: fixed lawnmower sweep, no information-gain reasoning --
        n_drones = len(inp.drones)
        map_size = inp.prior_map.shape[0] * params.CELL_SIZE_M
        drone_waypoints = {
            d.drone_id: lawnmower_position(i, n_drones, inp.t, map_size)
            for i, d in enumerate(inp.drones)
        }

        # -- resources: nearest idle resource -> nearest known target, no
        # type/survival/coordination reasoning. Multiple resources CAN
        # converge on the same target -- that's the point of "naive".
        assignments: List[Assignment] = []
        idle_resources = [r for r in inp.resources if r.status == "idle"]
        for r in idle_resources:
            if not self._targets:
                break
            nearest_tid = min(self._targets, key=lambda tid: _dist(r.position, self._targets[tid]))
            assignments.append(Assignment(
                resource_id=r.resource_id,
                target_belief_id=nearest_tid,
                expected_value=0.0,
                reason=(f"nearest-first baseline: {r.resource_id} sent to closest known "
                        f"detection ({nearest_tid}) -- no resource-type check, no survival "
                        f"estimate, no coordination with other resources."),
            ))

        return EngineOutput(
            t=inp.t,
            drone_waypoints=drone_waypoints,
            assignments=assignments,
            priority_list=list(self._targets.keys()),
            suspected_silent_zones=[],
        )


if __name__ == "__main__":
    import random

    from simulation.harness import run_scenario

    m = run_scenario(NearestFirstEngine(seed=1), seed=params.RANDOM_SEED)
    print("nearest-first baseline, one run:")
    for k, v in m.items():
        print(f"  {k}: {v}")
