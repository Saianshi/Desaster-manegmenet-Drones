"""
Demo driver script -- runs the real engine through simulation/harness.py
with full per-timestep logging to output/run_log.json for later
visualisation. Phase 7's evaluation/compare.py uses the same harness
without logging (see harness.py's docstring for why that's fast enough
to run 100+ times where this script only needs to run once).

Run standalone with:  python -m simulation.runner
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import params
from engine.contracts import EngineInput
from engine.engine import Engine
from simulation.harness import run_scenario
from simulation.world import World


class RunLog:
    """
    Three independent sections, kept strictly separate:

    - prior_map / entries: the sim<->engine wire traffic (EngineInput /
      EngineOutput pairs). This is the ONLY thing the engine ever sees.
    - world_static: buildings, flood zone, elevation -- generated once,
      never changes during a run, so logged once instead of per timestep.
    - ground_truth: per-timestep victim positions/alive state plus death
      events. This is ground truth for scoring and visualisation ONLY.
      The engine never receives this section, in any form.
    """

    def __init__(self):
        self.entries = []
        self._prior_map = None
        self._world_static = None
        self._ground_truth_timesteps = []
        self._death_events = []
        self._rescue_events = []

    def set_world_static(self, world: World) -> None:
        self._world_static = world.to_static_dict()

    def append(self, inp: EngineInput, out, world: World) -> None:
        if self._prior_map is None:
            self._prior_map = inp.prior_map.tolist()
        input_dict = inp.to_dict()
        del input_dict["prior_map"]
        self.entries.append({"input": input_dict, "output": out.to_dict()})

        self._ground_truth_timesteps.append(world.snapshot_ground_truth())
        self._death_events.extend(world.pop_death_events())
        self._rescue_events.extend(world.pop_rescue_events())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "prior_map": self._prior_map,
                    "entries": self.entries,
                    "world_static": self._world_static,
                    "ground_truth": {
                        "timesteps": self._ground_truth_timesteps,
                        "death_events": self._death_events,
                        "rescue_events": self._rescue_events,
                    },
                },
                f,
            )


def run(seed: int = params.RANDOM_SEED, T: float = params.T_TOTAL_SECONDS, dt: float = params.DT_SECONDS):
    engine = Engine(seed=seed)
    log = RunLog()

    metrics = run_scenario(engine, seed=seed, T=T, dt=dt, run_log=log)
    log.save(ROOT / "output" / "run_log.json")

    n_steps = int(T // dt)
    print("=== Phase 7 run complete (full engine) ===")
    print(f"seed={seed}  steps={n_steps}  dt={dt}s  T={T/3600:.1f}h")
    print(f"total people={metrics['n_total_people']}  drones={params.N_DRONES_STUB}  resources={metrics['n_resources']}")
    print(f"people alive at end: {metrics['n_alive_people']}/{metrics['n_total_people']}")
    print(f"LIVES SAVED (rescued groups): {metrics['rescued_groups']}   (people): {metrics['rescued_people']}")
    print(f"raw detections captured: {metrics['total_raw_detections']}   delivered to engine: {metrics['total_delivered_detections']}   still buffered at end: {metrics['buffered_at_end']}")
    print(f"total assignments made: {metrics['total_assignments']}")
    print(f"final water level: {metrics['final_water_level']:.2f}m")
    print(f"log written to: {ROOT / 'output' / 'run_log.json'}")
    return log


if __name__ == "__main__":
    run()
