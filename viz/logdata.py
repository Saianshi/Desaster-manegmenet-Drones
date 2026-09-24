"""
Shared log-loading and derived-data helpers for the /viz layer.

Every module under /viz reads ONLY from output/*.json (run_log.json,
produced by simulation/generate_viz_log.py, and results.json, produced
by evaluation/compare.py). Nothing here imports /engine or /simulation --
per the visualisation layer's own rule, it is a passive consumer of
already-produced data, never a participant in the sim or the decision
loop.

run_log.json's schema (see simulation/generate_viz_log.py's docstring):
{
  "meta": {seed, dt, T, map_size_m, cell_size_m},
  "world_static": {buildings, elevation_map, flood_zone_cells, ...},
  "prior_map": [...],
  "victims_static": {victim_id: {group_size, env_type}},
  "runs": {
    "engine":   {"entries": [{"input":..., "output":...}, ...], "ground_truth": {...}},
    "baseline": {"entries": [...], "ground_truth": {...}}
  }
}
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_PATH = ROOT / "output" / "run_log.json"
DEFAULT_RESULTS_PATH = ROOT / "output" / "results.json"

RUN_LABELS = {"engine": "Our engine", "baseline": "Baseline (nearest-first)"}


def load_log(path: Path = DEFAULT_LOG_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def load_results(path: Path = DEFAULT_RESULTS_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def victim_true_positions(log: dict, run_name: str) -> Dict[str, Tuple[float, float]]:
    """A victim's true position never changes over their lifetime; grab it once from t=0."""
    ts0 = log["runs"][run_name]["ground_truth"]["timesteps"][0]
    return {v["victim_id"]: tuple(v["position"]) for v in ts0["victims"]}


def first_detected_times(log: dict, run_name: str, radius: float = 50.0) -> Dict[str, float]:
    """
    For each TRUE victim, the first sim-time (seconds) at which some
    DELIVERED detection landed within `radius` of their true position.
    This is a ground-truth-informed proxy used ONLY for rendering
    (grey/undetected vs coloured/detected victims) -- it is never
    something the engine itself used to decide anything; the engine only
    ever saw the noisy detections, never this cross-reference against
    true positions.
    """
    positions = victim_true_positions(log, run_name)
    vids = list(positions.keys())
    vxy = np.array([positions[v] for v in vids])

    first_t = {vid: float("inf") for vid in vids}
    for entry in log["runs"][run_name]["entries"]:
        dets = entry["input"]["detections"]
        if not dets:
            continue
        t = entry["input"]["t"]
        dxy = np.array([d["est_position"] for d in dets])
        d2 = ((dxy[:, None, :] - vxy[None, :, :]) ** 2).sum(axis=2)
        within = (d2 <= radius ** 2).any(axis=0)
        for i, vid in enumerate(vids):
            if within[i] and t < first_t[vid]:
                first_t[vid] = t
    return first_t


def rescued_events_sorted(log: dict, run_name: str) -> List[Tuple[float, str]]:
    """(t, victim_id) for every successful rescue, chronological."""
    events = log["runs"][run_name]["ground_truth"]["rescue_events"]
    rescued = [(e["t"], e["victim_id"]) for e in events if e["outcome"] == "rescued"]
    rescued.sort(key=lambda x: x[0])
    return rescued


def rescued_ids_by_frame(log: dict, run_name: str, n_frames: int) -> List[Set[str]]:
    """rescued_ids_by_frame[i] = victim_ids rescued by (and including) entries[i]'s timestep."""
    entries = log["runs"][run_name]["entries"]
    pairs = rescued_events_sorted(log, run_name)
    out = []
    idx = 0
    current: Set[str] = set()
    for i in range(n_frames):
        t = entries[i]["input"]["t"]
        while idx < len(pairs) and pairs[idx][0] <= t:
            current.add(pairs[idx][1])
            idx += 1
        out.append(set(current))
    return out


def group_size_lookup(log: dict) -> Dict[str, int]:
    return {vid: info["group_size"] for vid, info in log["victims_static"].items()}


def env_type_lookup(log: dict) -> Dict[str, str]:
    return {vid: info["env_type"] for vid, info in log["victims_static"].items()}


def compute_backhaul_from_linked_to(drones: List[dict]) -> Dict[str, bool]:
    """Same multi-hop BFS engine/connectivity.py does, reimplemented
    standalone here so /viz never imports /engine. drones: list of
    Drone.to_dict()-shaped dicts (has 'drone_id' and 'linked_to')."""
    by_id = {d["drone_id"]: d for d in drones}
    adjacency: Dict[str, set] = {d["drone_id"]: set() for d in drones}
    linked_to_base = set()
    for d in drones:
        for nb in d["linked_to"]:
            if nb == "base":
                linked_to_base.add(d["drone_id"])
            elif nb in by_id:
                adjacency[d["drone_id"]].add(nb)

    visited = set(linked_to_base)
    frontier = list(linked_to_base)
    while frontier:
        nxt = []
        for did in frontier:
            for nb in adjacency[did]:
                if nb not in visited:
                    visited.add(nb)
                    nxt.append(nb)
        frontier = nxt
    return {d["drone_id"]: (d["drone_id"] in visited) for d in drones}


def n_frames(log: dict, run_name: str) -> int:
    return len(log["runs"][run_name]["entries"])


def frame_t(log: dict, run_name: str, i: int) -> float:
    return log["runs"][run_name]["entries"][i]["input"]["t"]
