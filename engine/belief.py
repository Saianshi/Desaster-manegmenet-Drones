"""
Belief fusion: turns raw, noisy Detections into a stateful set of
VictimBelief objects -- the engine's actual world model. This is the
only place in /engine that is allowed to accumulate state across
timesteps (everything else can be a pure function of the current
EngineInput).

/engine must never import from /simulation. Everything here operates on
Detection/VictimBelief/EngineInput only.

Design notes on the harder judgment calls (see config/params.py's
"ENGINE-SIDE MODEL ASSUMPTIONS" section for the actual numbers):

- FUSION: a new detection is matched to an existing belief if it falls
  within ENGINE_FUSION_SIGMA_MULTIPLIER sigma of it (spec: "~2 sigma").
  Matching updates position via inverse-variance weighting (a cheap
  1-step Kalman-style fusion) and shrinks uncertainty; a NEW sensor type
  showing up raises confidence more than a repeat hit from a sensor
  that's already vouching for this belief.

- ENV_TYPE CLASSIFICATION is a real sensing limitation, not an oversight:
  thermal and rf cannot physically distinguish "rooftop" from "street"
  (see Detection.env_hint's own enum -- it only has "surface", not a
  rooftop/street split). A uwb hit is conclusive for "buried" (uwb only
  fires on buried victims in this sensor model). An ambiguous "surface"
  or "unknown" hint is resolved using the POPULATION-LEVEL base rate
  (FRAC_ROOFTOP vs FRAC_STREET) as a Bayesian prior -- the same category
  of knowledge as prior_map (aggregate historical data), never a peek at
  this specific victim's ground truth.

- GROUP SIZE cannot be observed directly (each detection is one sensor
  hit, not a headcount). The engine nudges its group-size guess up for
  every few independent re-detections of the same cluster, capped well
  below the true possible max -- a weak, honestly-labelled heuristic.

- SILENCE AS SIGNAL: a per-cell scan history is kept (which sensors have
  ever swept a cell, and whether any of those scans ever produced a
  detection there). A cell with high prior population, at least one scan,
  and zero detections ever is flagged as a suspected silent zone --
  either genuinely empty, or everyone there is too trapped to signal,
  which is the worse case and exactly why this gets surfaced rather than
  silently ignored.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import params
from engine import models
from engine.contracts import Detection, EngineInput, VictimBelief

# Sensor ranges, duplicated here only as scan radii for the silent-zone
# grid (this is public sensor spec, not simulation ground truth -- the
# engine knows its own drones' sensor ranges).
_SENSOR_RANGE_M = {
    "thermal": params.THERMAL_RANGE_M,
    "uwb": params.UWB_RANGE_M,
    "rf": params.RF_RANGE_M,
}


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _to_integral_image(arr: np.ndarray) -> np.ndarray:
    """2D cumulative sum with a leading zero row/col, so that the sum over
    rows [y0,y1) and cols [x0,x1) is ii[y1,x1]-ii[y0,x1]-ii[y1,x0]+ii[y0,x0]."""
    ii = np.zeros((arr.shape[0] + 1, arr.shape[1] + 1), dtype=arr.dtype)
    ii[1:, 1:] = np.cumsum(np.cumsum(arr, axis=0), axis=1)
    return ii


class BeliefTracker:
    def __init__(self, seed: int = 0, grid_n: int = params.GRID_N, cell_size: float = params.CELL_SIZE_M):
        self._rng = random.Random(seed)
        self.beliefs: Dict[str, VictimBelief] = {}
        self._meta: Dict[str, dict] = {}  # belief_id -> {detection_count, water_level_at_first_seen}
        self._counter = 0
        self._last_t: float = None

        self._grid_n = grid_n
        self._cell_size = cell_size
        # Tracked PER SENSOR, not just "scanned by anything" -- this matters
        # a lot: rf's 300m passive range marks huge areas "scanned" within
        # the first few timesteps regardless of whether it found anyone,
        # while uwb (the ONLY sensor that can find buried victims) only
        # ever covers a 60m circle, once per timestep, fleet-wide. Treating
        # "scanned by rf" as equivalent to "scanned by uwb" made the whole
        # map look information-exhausted almost immediately, which starved
        # the Phase 6 orchestrator of any signal to steer by. See
        # unexplored_population_within()'s `sensor` parameter.
        self._scanned_by_sensor: Dict[str, np.ndarray] = {
            s: np.zeros((grid_n, grid_n), dtype=bool) for s in _SENSOR_RANGE_M
        }
        self._scan_count_by_sensor: Dict[str, np.ndarray] = {
            s: np.zeros((grid_n, grid_n), dtype=int) for s in _SENSOR_RANGE_M
        }
        self._has_detection = np.zeros((grid_n, grid_n), dtype=bool)  # ever produced a detection

    # -- public API ------------------------------------------------------
    def update(self, inp: EngineInput) -> None:
        dt_minutes = 0.0 if self._last_t is None else max(inp.t - self._last_t, 0.0) / 60.0
        self._last_t = inp.t

        matched_ids = set()
        for det in inp.detections:
            belief_id = self._fuse(det, inp.environment)
            matched_ids.add(belief_id)

        self._mark_scan_coverage(inp)
        self._decay_unmatched(matched_ids, dt_minutes)
        self._refresh_derived_fields(inp.environment)

    def get_beliefs(self) -> List[VictimBelief]:
        return list(self.beliefs.values())

    def get_stats(self) -> dict:
        """Diagnostic counters -- total_created includes beliefs that have
        since decayed below ENGINE_BELIEF_PRUNE_THRESHOLD and were dropped;
        currently_alive is len(self.beliefs) right now."""
        return {"total_created": self._counter, "currently_alive": len(self.beliefs)}

    def get_water_level_at_first_seen(self, belief_id: str) -> float:
        """Only meaningful for rooftop beliefs -- see models.py's
        ENGINE_ASSUMED_ROOFTOP_CLEARANCE_M for why this is the anchor
        point instead of a (never-observed) true elevation."""
        return self._meta.get(belief_id, {}).get("water_level_at_first_seen")

    def get_belief_positions(self) -> Dict[str, Tuple[float, float]]:
        """The engine's OWN already-computed belief positions, exposed so
        the simulation can dispatch a resource somewhere -- EngineOutput
        itself only carries belief_ids, not positions. This is not a
        ground-truth leak: it's the engine handing back its own estimate,
        the same way a real dispatch system would receive coordinates."""
        return {bid: b.position for bid, b in self.beliefs.items()}

    def get_belief_uncertainties(self) -> Dict[str, float]:
        """Same rationale as get_belief_positions(): the engine's own
        already-computed uncertainty, exposed so the simulation can size a
        realistic on-scene search radius -- a real dispatcher tells the
        team how confident the fix is, not just where to go."""
        return {bid: b.uncertainty for bid, b in self.beliefs.items()}

    def _scanned_any(self) -> np.ndarray:
        """OR across all sensors -- "has this cell been scanned by
        anything at all", used where the specific sensor doesn't matter
        (e.g. silent-zone detection: a cell that's genuinely silent looks
        the same regardless of which sensor swept it)."""
        arrays = list(self._scanned_by_sensor.values())
        return np.logical_or.reduce(arrays) if arrays else np.zeros((self._grid_n, self._grid_n), dtype=bool)

    def _scan_count_any(self) -> np.ndarray:
        arrays = list(self._scan_count_by_sensor.values())
        return np.sum(arrays, axis=0) if arrays else np.zeros((self._grid_n, self._grid_n), dtype=int)

    def build_unexplored_integral_images(self, prior_map: np.ndarray) -> Dict[Optional[str], np.ndarray]:
        """
        Summed-area tables (2D cumulative sums, one leading zero row/col)
        of (unscanned * prior_map), one per sensor plus one for "any
        sensor" (key None). Lets a caller answer MANY "how much
        unexplored population near point X" queries in O(1) each via
        query_unexplored_rect(), instead of rebuilding a meshgrid per
        query -- engine.orchestrator asks this question ~17 candidates x
        up to 3 sensors x every drone x every timestep, and the meshgrid
        version of that was the dominant cost of a full run.
        Call once per engine.decide(), not once per candidate.
        """
        images: Dict[Optional[str], np.ndarray] = {}
        for sensor, scanned in self._scanned_by_sensor.items():
            images[sensor] = _to_integral_image(np.where(scanned, 0.0, prior_map))
        images[None] = _to_integral_image(np.where(self._scanned_any(), 0.0, prior_map))
        return images

    def query_unexplored_rect(self, integral_image: np.ndarray, position: Tuple[float, float], radius: float) -> float:
        """O(1) SQUARE-bounding-box approximation of the circular query
        unexplored_population_within() does exactly -- overestimates
        slightly (counts the circle's bounding square, corners included),
        which is an acceptable trade for a greedy per-candidate heuristic
        since the bias is roughly uniform across candidates and doesn't
        distort their RELATIVE ranking much. Use the exact method instead
        for one-off, non-hot-path queries (e.g. scheduler's cascade bonus,
        called only a handful of times per timestep)."""
        cx, cy = self._cell_of(position)
        r_cells = int(math.ceil(radius / self._cell_size))
        y0, y1 = max(cy - r_cells, 0), min(cy + r_cells + 1, self._grid_n)
        x0, x1 = max(cx - r_cells, 0), min(cx + r_cells + 1, self._grid_n)
        if y0 >= y1 or x0 >= x1:
            return 0.0
        return float(integral_image[y1, x1] - integral_image[y0, x1] - integral_image[y1, x0] + integral_image[y0, x0])

    def unexplored_population_within(self, position: Tuple[float, float], radius: float,
                                      prior_map: np.ndarray, sensor: str = None) -> float:
        """
        Sum of prior_map over cells within radius of position that have
        NEVER been scanned -- used by the scheduler's cascade bonus (any
        sensor, sensor=None) to value physically reaching a location for
        the information it would unlock, and by the Phase 6 orchestrator
        (sensor=<specific>) to value what THAT sensor specifically hasn't
        covered yet -- see the sensor-tracking note in __init__.
        """
        scanned = self._scanned_by_sensor[sensor] if sensor is not None else self._scanned_any()

        cx, cy = self._cell_of(position)
        r_cells = int(math.ceil(radius / self._cell_size))
        y0, y1 = max(cy - r_cells, 0), min(cy + r_cells + 1, self._grid_n)
        x0, x1 = max(cx - r_cells, 0), min(cx + r_cells + 1, self._grid_n)
        if y0 >= y1 or x0 >= x1:
            return 0.0

        yy, xx = np.mgrid[y0:y1, x0:x1]
        cell_cx = (xx + 0.5) * self._cell_size
        cell_cy = (yy + 0.5) * self._cell_size
        in_circle = (cell_cx - position[0]) ** 2 + (cell_cy - position[1]) ** 2 <= radius ** 2

        unscanned = ~scanned[y0:y1, x0:x1]
        mask = in_circle & unscanned
        return float(prior_map[y0:y1, x0:x1][mask].sum())

    def get_silent_zones(self, prior_map: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        Returns the top ENGINE_SILENT_ZONE_MAX_REPORTED (x, y, risk) cells
        that are scanned, high-prior, and have never produced a detection.
        risk in [0, 1]. Fully vectorised: this runs once per engine
        decision, and a Python-level loop over every qualifying 10m cell
        (there can be thousands once scanning covers a lot of ground) was
        by far the dominant cost of a full run.
        """
        threshold = np.percentile(prior_map, params.ENGINE_SILENT_ZONE_PRIOR_PERCENTILE)
        prior_max = max(float(prior_map.max()), 1e-9)

        silent_mask = self._scanned_any() & (~self._has_detection) & (prior_map >= threshold)
        if not silent_mask.any():
            return []

        prior_norm = prior_map / prior_max
        scan_confidence = 1 - 0.5 ** np.maximum(self._scan_count_any(), 1)
        risk = np.where(silent_mask, prior_norm * scan_confidence, 0.0)

        flat = risk.ravel()
        # Pull a larger candidate pool than we need, since non-max
        # suppression below will discard cells too close to an
        # already-chosen zone (one large contiguous unscanned blob would
        # otherwise report as dozens of adjacent 10m cells).
        pool_size = min(10 * params.ENGINE_SILENT_ZONE_MAX_REPORTED, int(silent_mask.sum()))
        pool_idx = np.argpartition(flat, -pool_size)[-pool_size:]
        pool_idx = pool_idx[np.argsort(-flat[pool_idx])]

        zones: List[Tuple[float, float, float]] = []
        min_sep2 = params.ENGINE_SILENT_ZONE_MIN_SEPARATION_M ** 2
        for idx in pool_idx:
            r = float(flat[idx])
            if r < params.ENGINE_SILENT_ZONE_MIN_RISK:
                break
            gy, gx = divmod(int(idx), self._grid_n)
            cx = (gx + 0.5) * self._cell_size
            cy = (gy + 0.5) * self._cell_size
            if any((cx - zx) ** 2 + (cy - zy) ** 2 < min_sep2 for zx, zy, _ in zones):
                continue
            zones.append((cx, cy, r))
            if len(zones) >= params.ENGINE_SILENT_ZONE_MAX_REPORTED:
                break
        return zones

    # -- fusion ------------------------------------------------------
    def _fuse(self, det: Detection, environment: Dict[str, float]) -> str:
        best_id = None
        best_dist = None
        for bid, belief in self.beliefs.items():
            if det.sensor == "field_report":
                # A field report's own position_sigma is small (the crew
                # is standing there -- that's a precise fix, not an
                # uncertain one), but that's the wrong number to use for
                # MATCHING which belief it corrects: the crew searched up
                # to RESCUE_SEARCH_RADIUS_CAP_M around the belief's
                # estimated position before finding them, so the true
                # position can legitimately be that far from where the
                # dispatch thought it was. Matching on the tiny sigma
                # would usually miss the belief entirely and spawn a
                # disconnected new one instead of correcting the original
                # -- exactly what this feedback path exists to prevent.
                match_radius = params.RESCUE_SEARCH_RADIUS_CAP_M
            else:
                match_radius = params.ENGINE_FUSION_SIGMA_MULTIPLIER * max(det.position_sigma, belief.uncertainty)
            d = _dist(det.est_position, belief.position)
            if d <= match_radius and (best_dist is None or d < best_dist):
                best_id, best_dist = bid, d

        if best_id is None:
            return self._create_belief(det, environment)

        self._merge_detection(best_id, det)
        return best_id

    def _create_belief(self, det: Detection, environment: Dict[str, float]) -> str:
        self._counter += 1
        belief_id = f"belief_{self._counter}"
        env_type = self._classify_env_type(det, prior_env_type=None)

        belief = VictimBelief(
            belief_id=belief_id,
            position=det.est_position,
            uncertainty=det.position_sigma,
            env_type=env_type,
            est_group_size=1,
            confidence=det.confidence,
            survival_deadline=0.0,
            extraction_minutes=0.0,
            required_resources=[],
            detected_by=[det.sensor],
            first_seen_t=det.timestamp,
        )
        self.beliefs[belief_id] = belief
        self._meta[belief_id] = {
            "detection_count": 1,
            "water_level_at_first_seen": environment["water_level"],
        }
        return belief_id

    def _merge_detection(self, belief_id: str, det: Detection) -> None:
        belief = self.beliefs[belief_id]
        meta = self._meta[belief_id]

        # Inverse-variance weighted position fusion (1-step Kalman update).
        var_old = belief.uncertainty ** 2
        var_new = det.position_sigma ** 2
        w_old = 1.0 / max(var_old, 1e-6)
        w_new = 1.0 / max(var_new, 1e-6)
        fused_x = (belief.position[0] * w_old + det.est_position[0] * w_new) / (w_old + w_new)
        fused_y = (belief.position[1] * w_old + det.est_position[1] * w_new) / (w_old + w_new)
        fused_var = 1.0 / (w_old + w_new)
        belief.position = (fused_x, fused_y)
        belief.uncertainty = math.sqrt(fused_var)

        is_new_sensor = det.sensor not in belief.detected_by
        if is_new_sensor:
            belief.detected_by.append(det.sensor)
            # Multi-sensor agreement raises confidence more than a repeat hit.
            belief.confidence = min(1.0, belief.confidence + (1 - belief.confidence) * 0.5)
        else:
            belief.confidence = min(1.0, belief.confidence + (1 - belief.confidence) * 0.15)

        belief.env_type = self._classify_env_type(det, prior_env_type=belief.env_type)

        meta["detection_count"] += 1
        belief.est_group_size = min(
            1 + meta["detection_count"] // params.ENGINE_GROUP_SIZE_DETECTIONS_PER_STEP,
            params.ENGINE_GROUP_SIZE_MAX_GUESS,
        )

    def _classify_env_type(self, det: Detection, prior_env_type) -> str:
        # [REALISM FIX] An on-scene field report (a crew physically
        # standing at the site after a wrong_resource_type failure -- see
        # simulation/rescue.py._emit_field_report) is direct human
        # observation. It overrides EVERYTHING else unconditionally,
        # including a prior uwb-confirmed classification: a uwb hit is
        # itself just a strong probabilistic inference (this sensor model
        # only fires for buried victims), while a field report is ground
        # truth relayed by a person who looked. Checked first, on purpose.
        if det.sensor == "field_report":
            return det.env_hint

        # uwb is conclusive: this sensor model only ever fires for buried
        # victims, so a uwb hit overrides any prior ambiguous classification.
        if det.sensor == "uwb" or det.env_hint == "buried":
            return "buried"
        if prior_env_type == "buried":
            # Don't let a later ambiguous surface/rf hit downgrade a
            # uwb-confirmed buried belief (rare fusion-radius overlap).
            return "buried"
        if prior_env_type in ("rooftop", "street"):
            return prior_env_type  # already resolved, keep it

        # Ambiguous "surface" (thermal) or "unknown" (rf) hint, and no
        # prior classification: resolve using the population base rate
        # (see module docstring) rather than leaving it permanently
        # unclassified -- a real triage system has to commit to a working
        # hypothesis about flood exposure, or it will silently deprioritise
        # every stranded rooftop victim it can't visually confirm.
        p_rooftop = params.FRAC_ROOFTOP / (params.FRAC_ROOFTOP + params.FRAC_STREET)
        return "rooftop" if self._rng.random() < p_rooftop else "street"

    # -- scan coverage / silence as signal --------------------------------
    def _mark_scan_coverage(self, inp: EngineInput) -> None:
        # uwb is throttled fleet-wide to ONE drone per timestep in the real
        # sensor model (simulation/sensors.py's round-robin) -- replicated
        # here identically (same formula, same inputs: t, DT_SECONDS, and
        # which drones carry uwb -- all public equipment/schedule info, not
        # ground truth) so the tracker doesn't believe uwb coverage is far
        # more plentiful than it really is. Getting this wrong quietly
        # undermines the whole point of weighting uwb coverage so heavily
        # in engine.orchestrator's information_gain.
        uwb_drones = [d for d in inp.drones if "uwb" in d.sensors]
        active_uwb_id = None
        if uwb_drones:
            idx = int(inp.t // params.DT_SECONDS) % len(uwb_drones)
            active_uwb_id = uwb_drones[idx].drone_id

        for drone in inp.drones:
            for sensor in drone.sensors:
                if sensor == "uwb" and drone.drone_id != active_uwb_id:
                    continue
                r = _SENSOR_RANGE_M.get(sensor)
                if r is None:
                    continue
                self._mark_circle_scanned(sensor, drone.position, r)

        for det in inp.detections:
            gx, gy = self._cell_of(det.est_position)
            self._has_detection[gy, gx] = True

    def _mark_circle_scanned(self, sensor: str, center: Tuple[float, float], radius: float) -> None:
        # Vectorised over just the bounding box of the circle -- this runs
        # per drone per sensor per timestep, so a Python-level double loop
        # over every cell in range was the dominant cost of a full run.
        cx, cy = self._cell_of(center)
        r_cells = int(math.ceil(radius / self._cell_size))
        y0, y1 = max(cy - r_cells, 0), min(cy + r_cells + 1, self._grid_n)
        x0, x1 = max(cx - r_cells, 0), min(cx + r_cells + 1, self._grid_n)
        if y0 >= y1 or x0 >= x1:
            return

        yy, xx = np.mgrid[y0:y1, x0:x1]
        cell_cx = (xx + 0.5) * self._cell_size
        cell_cy = (yy + 0.5) * self._cell_size
        mask = (cell_cx - center[0]) ** 2 + (cell_cy - center[1]) ** 2 <= radius ** 2

        self._scanned_by_sensor[sensor][y0:y1, x0:x1] |= mask
        self._scan_count_by_sensor[sensor][y0:y1, x0:x1] += mask

    def _cell_of(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        gx = min(max(int(pos[0] // self._cell_size), 0), self._grid_n - 1)
        gy = min(max(int(pos[1] // self._cell_size), 0), self._grid_n - 1)
        return gx, gy

    # -- confidence decay --------------------------------------------------
    def _decay_unmatched(self, matched_ids: set, dt_minutes: float) -> None:
        if dt_minutes <= 0:
            return
        to_drop = []
        for bid, belief in self.beliefs.items():
            if bid in matched_ids:
                continue
            belief.confidence = max(0.0, belief.confidence - params.ENGINE_BELIEF_CONFIDENCE_DECAY_PER_MIN * dt_minutes)
            if belief.confidence < params.ENGINE_BELIEF_PRUNE_THRESHOLD:
                to_drop.append(bid)
        for bid in to_drop:
            del self.beliefs[bid]
            del self._meta[bid]

    # -- derived fields (survival_deadline, extraction_minutes, resources) --
    def _refresh_derived_fields(self, environment: Dict[str, float]) -> None:
        for bid, belief in self.beliefs.items():
            water_at_first_seen = self._meta[bid].get("water_level_at_first_seen")
            belief.survival_deadline = models.survival_deadline_minutes(belief, environment, water_at_first_seen)
            belief.extraction_minutes = models.extraction_minutes(belief)
            belief.required_resources = models.required_resources(belief)


if __name__ == "__main__":
    # Synthetic, hand-crafted detections (no simulation involved) to show
    # the fusion mechanics in isolation: two detections close together
    # should merge into one higher-confidence belief; a far-away one
    # should not.
    env = {"water_level": 4.0, "rise_rate": 0.4}

    det_a1 = Detection(detection_id="a1", est_position=(100.0, 100.0), position_sigma=8.0,
                        sensor="thermal", confidence=0.85, timestamp=0.0, env_hint="surface")
    det_a2 = Detection(detection_id="a2", est_position=(105.0, 98.0), position_sigma=40.0,
                        sensor="rf", confidence=0.5, timestamp=30.0, env_hint="unknown")
    det_b1 = Detection(detection_id="b1", est_position=(1500.0, 1500.0), position_sigma=15.0,
                        sensor="uwb", confidence=0.6, timestamp=30.0, env_hint="buried")

    tracker = BeliefTracker(seed=1)
    inp1 = EngineInput(t=0.0, detections=[det_a1], drones=[], resources=[], environment=env,
                        prior_map=np.ones((params.GRID_N, params.GRID_N)))
    tracker.update(inp1)
    print("After 1 thermal detection:")
    for b in tracker.get_beliefs():
        print(f"  {b.belief_id}: env={b.env_type} conf={b.confidence:.2f} sensors={b.detected_by}")

    inp2 = EngineInput(t=30.0, detections=[det_a2, det_b1], drones=[], resources=[], environment=env,
                        prior_map=np.ones((params.GRID_N, params.GRID_N)))
    tracker.update(inp2)
    print("\nAfter a nearby RF hit (should FUSE, confidence should rise) + a distant uwb hit (should be NEW belief):")
    for b in tracker.get_beliefs():
        print(f"  {b.belief_id}: env={b.env_type} conf={b.confidence:.2f} sensors={b.detected_by} pos={b.position}")

    assert len(tracker.get_beliefs()) == 2, "expected exactly 2 beliefs: one fused, one distinct"
    fused = [b for b in tracker.get_beliefs() if len(b.detected_by) == 2][0]
    assert fused.confidence > 0.85, "multi-sensor fusion should raise confidence above the single-sensor reading"
    print("\nOK: nearby detection fused and raised confidence; distant detection stayed separate.")
