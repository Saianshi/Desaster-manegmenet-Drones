"""
Reconstructs drone backhaul-to-base connectivity from Drone.linked_to --
each drone's own one-hop neighbours (other drone_ids, plus the literal
string "base" if base is directly in range), self-reported telemetry
that simulation/comms.py populates truthfully every tick. No ground
truth is read here; this is the same multi-hop BFS
simulation/comms.py's compute_backhaul does, run on the engine side
using only what the drones themselves report.

Shared by engine.scheduler (cascade bonus's "near a disconnected drone"
trigger -- see PROBLEM 2b) and available to engine.orchestrator, though
orchestrator's own connectivity_penalty needs a different, HYPOTHETICAL
question ("if this drone moved here, would it still have backhaul?")
that this module's straight telemetry read can't answer -- that one
still needs geometry (see orchestrator._would_have_backhaul).
"""

from __future__ import annotations

from typing import Dict, List, Set

from engine.contracts import Drone


def compute_backhaul(drones: List[Drone]) -> Dict[str, bool]:
    """Current backhaul status per drone, from each drone's own
    self-reported linked_to list -- no positions, no distances, just the
    adjacency graph the drones already told us about themselves."""
    by_id = {d.drone_id: d for d in drones}
    adjacency: Dict[str, Set[str]] = {d.drone_id: set() for d in drones}
    linked_to_base: Set[str] = set()

    for d in drones:
        for nb in d.linked_to:
            if nb == "base":
                linked_to_base.add(d.drone_id)
            elif nb in by_id:
                adjacency[d.drone_id].add(nb)

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

    return {d.drone_id: (d.drone_id in visited) for d in drones}
