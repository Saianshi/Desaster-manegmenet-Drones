"""
Engine entry point: EngineInput -> EngineOutput.

Phase 4 built real belief fusion (engine.belief.BeliefTracker) and real
survival/extraction estimates (engine.models). Phase 5 added the
decision core: engine.scheduler assigns idle resources to beliefs by
value (survival x group_size x confidence, plus a cascade bonus for
information gain, minus rescuer risk). Phase 6 adds engine.orchestrator,
which routes drones by information gain while protecting backhaul
connectivity. None of this changes the module's public interface:

    Engine().decide(EngineInput) -> EngineOutput

/engine must never import from /simulation. This module only depends on
engine.contracts, engine.belief, engine.models, engine.scheduler,
engine.orchestrator, and the standard library.
"""

from __future__ import annotations

import random

from config import params
from engine import orchestrator, scheduler
from engine.belief import BeliefTracker
from engine.contracts import EngineInput, EngineOutput


class Engine:
    def __init__(self, seed: int = 0, use_cascade_bonus: bool = True,
                 use_silence_as_signal: bool = True, use_rescuer_risk: bool = True):
        """
        The three ablation flags exist for Phase 7's evaluation
        (evaluation/compare.py) to isolate each feature's contribution.
        use_silence_as_signal gates the WHOLE feature (silent-zone
        detection feeds both the scheduler's cascade bonus and the
        orchestrator's exploration bonus, plus EngineOutput's own
        suspected_silent_zones field) -- turning it off means beliefs
        still get fused and scheduled, they just never get the "this
        rescue also unlocks a suspected trapped population nearby" signal.
        """
        self._rng = random.Random(seed)
        self._belief_tracker = BeliefTracker(seed=seed)
        self._use_cascade_bonus = use_cascade_bonus
        self._use_silence_as_signal = use_silence_as_signal
        self._use_rescuer_risk = use_rescuer_risk

    def get_belief_positions(self):
        """The engine's own current belief positions -- see
        BeliefTracker.get_belief_positions for why the simulation needs
        this (EngineOutput.Assignment only carries a belief_id, not a
        position, so something has to tell dispatch where to actually go)."""
        return self._belief_tracker.get_belief_positions()

    def get_belief_uncertainties(self):
        """See BeliefTracker.get_belief_uncertainties -- lets the
        simulation size a realistic on-scene search radius per belief."""
        return self._belief_tracker.get_belief_uncertainties()

    def decide(self, inp: EngineInput) -> EngineOutput:
        self._belief_tracker.update(inp)
        beliefs = self._belief_tracker.get_beliefs()
        silent_zones = self._belief_tracker.get_silent_zones(inp.prior_map) if self._use_silence_as_signal else []

        assignments = scheduler.schedule(
            beliefs=beliefs,
            resources=inp.resources,
            environment=inp.environment,
            drones=inp.drones,
            silent_zones=silent_zones,
            tracker=self._belief_tracker,
            prior_map=inp.prior_map,
            use_cascade_bonus=self._use_cascade_bonus,
            use_rescuer_risk=self._use_rescuer_risk,
        )

        map_size = inp.prior_map.shape[0] * params.CELL_SIZE_M
        drone_waypoints = orchestrator.decide_waypoints(
            drones=inp.drones,
            tracker=self._belief_tracker,
            prior_map=inp.prior_map,
            silent_zones=silent_zones,
            map_size=map_size,
        )

        priority_list = [b.belief_id for b in sorted(beliefs, key=lambda b: b.survival_deadline)]

        return EngineOutput(
            t=inp.t,
            drone_waypoints=drone_waypoints,
            assignments=assignments,
            priority_list=priority_list,
            suspected_silent_zones=silent_zones,
        )
