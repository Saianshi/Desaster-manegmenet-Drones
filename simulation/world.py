"""
Phase 2 world: geometry, buildings, flood, and the pre-disaster population
prior. Victim generation and per-step victim lifecycle (battery, death)
live in simulation/victims.py and are invoked from World.step().

/simulation is allowed to import from engine.contracts (the shared wire
format for Detection/Drone/Resource/etc). It must never import from
engine.engine or any other engine internals -- this module only produces
ground truth, it never makes a decision.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from config import params
from engine.contracts import Drone, Resource
from simulation import victims as victims_mod
from simulation.rescue import RescueResolver


@dataclass
class Building:
    """Simulation-only ground truth. The engine never sees this object."""

    building_id: str
    x0: float  # footprint bounds, metres
    y0: float
    x1: float
    y1: float
    floors: int
    collapsed: bool
    elevation_m: float  # ground elevation at building centroid

    @property
    def centroid(self) -> Tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def to_dict(self):
        return {
            "building_id": self.building_id,
            "x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1,
            "floors": self.floors,
            "collapsed": self.collapsed,
            "elevation_m": self.elevation_m,
        }


def _box_blur_2d(a: np.ndarray, kernel_size: int, iters: int) -> np.ndarray:
    """
    Poor-man's Gaussian smoothing using repeated 1D box blurs (numpy +
    convolve only -- no scipy). Used to turn white noise into smooth,
    hill-like terrain for the elevation map.
    """
    kernel = np.ones(kernel_size) / kernel_size
    out = a.copy()
    for _ in range(iters):
        out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=out)
        out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=out)
    return out


class World:
    def __init__(self, rng: random.Random, n_resources: int = params.N_RESOURCES_STUB):
        self.rng = rng
        self.t = 0.0
        self.map_size = params.MAP_SIZE_M
        self.n_cells = params.GRID_N
        self.cell_size = params.CELL_SIZE_M

        self._np_rng = np.random.RandomState(self.rng.randint(0, 2**31 - 1))

        self.elevation_map = self._make_elevation_map()
        self.flood_zone_mask = self._make_flood_zone_mask(self.elevation_map)
        self.buildings: List[Building] = self._make_buildings()
        self.prior_map = self._make_prior_map(self.buildings)

        self.water_level = params.INITIAL_WATER_LEVEL_M

        self.victims, self.victim_meta = victims_mod.generate_victims(
            self.rng, self.buildings, self.elevation_map, self.flood_zone_mask, self.cell_size
        )

        self.drones: List[Drone] = self._make_drones(params.N_DRONES_STUB)
        self.resources: List[Resource] = self._make_resources(n_resources)

        self._pending_death_events: List[dict] = []
        self._pending_rescue_events: List[dict] = []
        self._rescue_resolver = RescueResolver(self.rng)

    @property
    def rescued_ids(self):
        return self._rescue_resolver.rescued_ids

    # -- terrain --------------------------------------------------------
    def _make_elevation_map(self) -> np.ndarray:
        raw = self._np_rng.uniform(0, 1, size=(self.n_cells, self.n_cells))
        smooth = _box_blur_2d(raw, params.ELEVATION_SMOOTH_KERNEL, params.ELEVATION_SMOOTH_ITERS)
        smooth -= smooth.min()
        smooth /= max(smooth.max(), 1e-9)
        return smooth * params.ELEVATION_MAX_M

    def _make_flood_zone_mask(self, elevation_map: np.ndarray) -> np.ndarray:
        threshold = np.percentile(elevation_map, params.FLOOD_ZONE_PERCENTILE * 100)
        return elevation_map <= threshold

    def elevation_at(self, x: float, y: float) -> float:
        gx = min(int(x // self.cell_size), self.n_cells - 1)
        gy = min(int(y // self.cell_size), self.n_cells - 1)
        return float(self.elevation_map[gy, gx])

    def in_flood_zone(self, x: float, y: float) -> bool:
        gx = min(int(x // self.cell_size), self.n_cells - 1)
        gy = min(int(y // self.cell_size), self.n_cells - 1)
        return bool(self.flood_zone_mask[gy, gx])

    # -- buildings --------------------------------------------------------
    def _make_buildings(self) -> List[Building]:
        occupied = np.zeros((self.n_cells, self.n_cells), dtype=bool)
        buildings: List[Building] = []
        for i in range(params.N_BUILDINGS):
            placed = False
            for _try in range(params.BUILDING_PLACEMENT_MAX_TRIES):
                w = self.rng.randint(params.BUILDING_MIN_SIZE_CELLS, params.BUILDING_MAX_SIZE_CELLS)
                h = self.rng.randint(params.BUILDING_MIN_SIZE_CELLS, params.BUILDING_MAX_SIZE_CELLS)
                gx0 = self.rng.randint(0, self.n_cells - w - 1)
                gy0 = self.rng.randint(0, self.n_cells - h - 1)
                if occupied[gy0:gy0 + h, gx0:gx0 + w].any():
                    continue
                occupied[gy0:gy0 + h, gx0:gx0 + w] = True
                x0, y0 = gx0 * self.cell_size, gy0 * self.cell_size
                x1, y1 = (gx0 + w) * self.cell_size, (gy0 + h) * self.cell_size
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                collapsed = self.rng.random() < params.COLLAPSED_FRACTION
                floors = self.rng.randint(params.BUILDING_MIN_FLOORS, params.BUILDING_MAX_FLOORS)
                buildings.append(
                    Building(
                        building_id=f"b{i}",
                        x0=x0, y0=y0, x1=x1, y1=y1,
                        floors=floors,
                        collapsed=collapsed,
                        elevation_m=self.elevation_at(cx, cy),
                    )
                )
                placed = True
                break
            if not placed:
                continue  # grid too full, skip this building (rare)
        self._occupied_mask = occupied
        return buildings

    # -- population prior --------------------------------------------------
    def _make_prior_map(self, buildings: List[Building]) -> np.ndarray:
        n = self.n_cells
        baseline = self._np_rng.uniform(0, 1, size=(n, n)) * params.PRIOR_BASELINE_WEIGHT
        prior = baseline
        yy, xx = np.mgrid[0:n, 0:n]
        sigma = params.PRIOR_BUILDING_SIGMA_CELLS
        for b in buildings:
            cx_cell = ((b.x0 + b.x1) / 2) / self.cell_size
            cy_cell = ((b.y0 + b.y1) / 2) / self.cell_size
            bump = np.exp(-(((xx - cx_cell) ** 2 + (yy - cy_cell) ** 2) / (2 * sigma ** 2)))
            prior += bump * b.floors * params.PRIOR_BUILDING_WEIGHT
        return prior

    # -- drones / resources (unchanged from Phase 1 stub) -------------------
    def _make_drones(self, n: int) -> List[Drone]:
        drones = []
        for i in range(n):
            pos = (self.rng.uniform(0, self.map_size), self.rng.uniform(0, self.map_size))
            drones.append(
                Drone(drone_id=f"d{i}", position=pos, battery_pct=100.0,
                      sensors=["thermal", "uwb", "rf"], comm_range=300.0, linked_to=[])
            )
        return drones

    def _make_resources(self, n: int) -> List[Resource]:
        resources = []
        types = ["boat", "excavator", "medical"]
        for i in range(n):
            pos = (self.rng.uniform(0, self.map_size), self.rng.uniform(0, self.map_size))
            resources.append(
                Resource(resource_id=f"r{i}", type=types[i % len(types)], position=pos,
                          speed=200.0, status="idle", free_at=0.0)
            )
        return resources

    # -- simulation step ------------------------------------------------
    def step(self, dt: float, rise_rate_m_per_hr: float = params.DEFAULT_RISE_RATE_M_PER_HR) -> None:
        self.t += dt
        dt_hours = dt / 3600.0
        self.water_level += rise_rate_m_per_hr * dt_hours

        # Drone movement: no drift here -- World.apply() sets drone.position
        # directly to whatever engine.orchestrator commanded last decision
        # cycle (Phase 6). There's nothing for step() to do between commands.

        events = victims_mod.update_victims(
            self.victims, self.victim_meta, self.t, dt, self.water_level, self.rng,
            rescued_ids=self._rescue_resolver.rescued_ids,
        )
        self._pending_death_events.extend(events)

        rescue_events = self._rescue_resolver.step(self.resources, self.victims, self.t)
        self._pending_rescue_events.extend(rescue_events)

    def pop_death_events(self) -> List[dict]:
        events = self._pending_death_events
        self._pending_death_events = []
        return events

    def pop_rescue_events(self) -> List[dict]:
        events = self._pending_rescue_events
        self._pending_rescue_events = []
        return events

    def pop_field_report_detections(self) -> list:
        """Radio reports from crews physically on scene (see
        RescueResolver._emit_field_report) -- deliver these into the NEXT
        EngineInput.detections like any other detection. Caller's
        responsibility (not routed through comms/backhaul: a field radio
        is a direct channel, independent of drone relay)."""
        return self._rescue_resolver.pop_field_reports()

    def apply(self, engine_output, belief_positions: dict, belief_uncertainties: dict) -> None:
        # Drone waypoints: move-instantly to whatever engine.orchestrator
        # picked (candidate waypoints are already sized to one timestep's
        # worth of travel at DRONE_SPEED_M_PER_MIN, so this is a reasonable
        # approximation rather than a full flight-dynamics simulation).
        # Resource dispatch is real: RescueResolver drives
        # travel -> search -> [pairing] -> extraction -> outcome, resolved
        # against ground truth at the estimated belief position.
        for drone_id, wp in engine_output.drone_waypoints.items():
            for d in self.drones:
                if d.drone_id == drone_id:
                    d.position = wp
        self._rescue_resolver.dispatch(engine_output.assignments, self.resources, belief_positions, belief_uncertainties, self.t)

    # -- static ground-truth export (for logging / visualisation only) -----
    def to_static_dict(self) -> dict:
        return {
            "map_size_m": self.map_size,
            "cell_size_m": self.cell_size,
            "elevation_map": self.elevation_map.tolist(),
            "flood_zone_cells": [[int(x), int(y)] for y, x in zip(*np.where(self.flood_zone_mask))],
            "buildings": [b.to_dict() for b in self.buildings],
        }

    def snapshot_ground_truth(self) -> dict:
        return {
            "t": self.t,
            "water_level": self.water_level,
            "victims": [
                {"victim_id": v.victim_id, "position": [v.true_position[0], v.true_position[1]], "alive": v.alive}
                for v in self.victims
            ],
        }


if __name__ == "__main__":
    rng = random.Random(params.RANDOM_SEED)
    w = World(rng)
    print(f"buildings={len(w.buildings)}  collapsed={sum(b.collapsed for b in w.buildings)}")
    print(f"flood zone cells={int(w.flood_zone_mask.sum())} / {w.n_cells * w.n_cells}")
    print(f"victims={len(w.victims)}  prior_map shape={w.prior_map.shape}")
    for _ in range(5):
        w.step(params.DT_SECONDS)
    print(f"after 5 steps: t={w.t}s water_level={w.water_level:.3f}m")
