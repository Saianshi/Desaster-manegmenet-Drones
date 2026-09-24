"""
Three-layer comms model (Phase 3).

Layer A: BLE mesh directly between victim phones. Short range, badly
         attenuated by rubble. Victims who mesh together share whichever
         member has the best connectivity out.
Layer B: each drone carries a portable cell. Any BLE cluster (or lone
         phone) with at least one member within cell range of a drone is
         "reachable" -- that drone can pull the whole cluster's data.
Layer C: drone-to-drone backhaul. A drone only actually DELIVERS what it
         collected if it has a (possibly multi-hop) radio path back to
         the fixed base station. No path -> the data is BUFFERED, not
         lost, and flushed later once connectivity is restored.

This entire module is simulation-side ground truth (it reads Victim.
true_position and phone_state directly) and must never be imported by
/engine. What the engine is allowed to know about connectivity is
whatever the simulation truthfully reports in Drone.linked_to -- the
engine has to reconstruct multi-hop backhaul chains itself from that,
the same way real drone telemetry would only report each drone's direct
neighbours.
"""

from __future__ import annotations

import math
from typing import Dict, List, Set, Tuple

from config import params
from engine.contracts import Detection, Drone, Victim


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_ble_edges(victims: List[Victim]) -> List[Tuple[str, str]]:
    """
    Pairwise BLE links between victims with a functioning phone (alive
    person, phone battery on, app installed -- phone_state == "alive").
    An edge exists if the pair is within BLE range: 40m in open air, or
    10m if either endpoint is a buried victim (signal has to fight
    through rubble -- see BLE_RANGE_THROUGH_COLLAPSED_M in config for the
    simplification this uses instead of raycasting against building
    polygons). Exposed separately from build_ble_clusters so the actual
    edge list is available for drawing the mesh graph.
    """
    nodes = [v for v in victims if v.alive and v.phone_state == "alive"]
    edges: List[Tuple[str, str]] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            v1, v2 = nodes[i], nodes[j]
            d = _dist(v1.true_position, v2.true_position)
            max_range = (
                params.BLE_RANGE_THROUGH_COLLAPSED_M
                if (v1.env_type == "buried" or v2.env_type == "buried")
                else params.BLE_RANGE_OPEN_M
            )
            if d <= max_range:
                edges.append((v1.victim_id, v2.victim_id))
    return edges


def build_ble_clusters(victims: List[Victim]) -> List[Set[str]]:
    """Union-find over build_ble_edges() -- connected components of the BLE mesh."""
    nodes = [v.victim_id for v in victims if v.alive and v.phone_state == "alive"]
    parent = {vid: vid for vid in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for a, b in build_ble_edges(victims):
        union(a, b)

    clusters: Dict[str, Set[str]] = {}
    for vid in nodes:
        root = find(vid)
        clusters.setdefault(root, set()).add(vid)
    return list(clusters.values())


def reachable_victim_ids(victims: List[Victim], drones: List[Drone]) -> Set[str]:
    """Layer A + B: victim_ids whose BLE cluster has >=1 member within
    DRONE_CELL_RANGE_M of >=1 drone."""
    clusters = build_ble_clusters(victims)
    pos = {v.victim_id: v.true_position for v in victims}
    reachable: Set[str] = set()
    for cluster in clusters:
        if any(
            _dist(pos[vid], d.position) <= params.DRONE_CELL_RANGE_M
            for vid in cluster
            for d in drones
        ):
            reachable |= cluster
    return reachable


def compute_backhaul(drones: List[Drone]) -> Tuple[Dict[str, bool], Dict[str, List[str]]]:
    """
    Layer C. Returns (has_backhaul, linked_to):
      has_backhaul[drone_id] -- True if there is ANY path (possibly via
        other drones) from this drone back to the base station.
      linked_to[drone_id] -- this drone's DIRECT (one-hop) neighbours:
        other drone_ids within DRONE_BACKHAUL_RANGE_M, plus the literal
        string "base" if the base station itself is directly in range.
        This is exactly what Drone.linked_to should be set to -- it's
        real telemetry a drone could report about itself, not a
        privileged computation only the sim gets to see.
    """
    base = params.BASE_STATION_POSITION
    linked_to: Dict[str, List[str]] = {d.drone_id: [] for d in drones}
    adjacency: Dict[str, Set[str]] = {d.drone_id: set() for d in drones}

    for i, d1 in enumerate(drones):
        if _dist(d1.position, base) <= params.DRONE_BACKHAUL_RANGE_M:
            linked_to[d1.drone_id].append("base")
        for d2 in drones[i + 1:]:
            if _dist(d1.position, d2.position) <= params.DRONE_BACKHAUL_RANGE_M:
                linked_to[d1.drone_id].append(d2.drone_id)
                linked_to[d2.drone_id].append(d1.drone_id)
                adjacency[d1.drone_id].add(d2.drone_id)
                adjacency[d2.drone_id].add(d1.drone_id)

    has_backhaul = {d.drone_id: False for d in drones}
    frontier = [d.drone_id for d in drones if "base" in linked_to[d.drone_id]]
    for did in frontier:
        has_backhaul[did] = True
    visited = set(frontier)
    while frontier:
        nxt = []
        for did in frontier:
            for nb in adjacency[did]:
                if nb not in visited:
                    visited.add(nb)
                    has_backhaul[nb] = True
                    nxt.append(nb)
        frontier = nxt

    return has_backhaul, linked_to


class CommsRelay:
    """
    Stateful buffering across timesteps. A drone without backhaul this
    tick keeps whatever it collected in its own buffer; once it (or a
    relay path) regains a path to base, the whole buffer flushes at once.
    This is the mechanism that makes "buffered, not delivered" real
    rather than just a status label.
    """

    def __init__(self):
        self._buffer: Dict[str, List[Detection]] = {}

    def route(self, detections_by_drone: Dict[str, List[Detection]], has_backhaul: Dict[str, bool]) -> List[Detection]:
        delivered: List[Detection] = []
        for drone_id, dets in detections_by_drone.items():
            self._buffer.setdefault(drone_id, []).extend(dets)
            if has_backhaul.get(drone_id, False):
                delivered.extend(self._buffer[drone_id])
                self._buffer[drone_id] = []
        return delivered

    def buffered_count(self) -> int:
        return sum(len(v) for v in self._buffer.values())


if __name__ == "__main__":
    import random

    from simulation.world import World

    rng = random.Random(params.RANDOM_SEED)
    world = World(rng)

    clusters = build_ble_clusters(world.victims)
    reachable = reachable_victim_ids(world.victims, world.drones)
    has_backhaul, linked_to = compute_backhaul(world.drones)

    n_mesh_nodes = sum(len(c) for c in clusters)
    print(f"BLE clusters: {len(clusters)}  (covering {n_mesh_nodes} victims with a live phone)")
    print(f"cluster sizes: {sorted(len(c) for c in clusters)}")
    print(f"victims reachable via drone cell pickup: {len(reachable)}")
    print(f"drone backhaul status: {has_backhaul}")
    print(f"drone linked_to: {linked_to}")
