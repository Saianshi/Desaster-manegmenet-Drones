"""
Data contracts shared across the whole project.

IMPORTANT (architecture rule): this module lives under /engine because the
engine is the thing that must be deployable against real hardware, and real
hardware needs to agree on a wire format. /simulation is ALLOWED to import
from here (it has to produce Detections, Drones, etc. in this exact shape).
/engine is NEVER allowed to import from /simulation.

`Victim` is the one exception worth calling out: it is ground truth, and it
only exists so /simulation can track what's "really" happening in the world.
The engine must never receive a Victim -- it only ever sees Detection and
VictimBelief. If you find yourself importing Victim into /engine, stop.

Every class has to_dict()/from_dict() so a full EngineInput/EngineOutput pair
can be logged to /output/run_log.json for later visualisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

Position = Tuple[float, float]


def _pos_to_list(p: Position) -> List[float]:
    return [float(p[0]), float(p[1])]


def _pos_from_list(v: Any) -> Position:
    return (float(v[0]), float(v[1]))


# ---------------------------------------------------------------------------
# Simulation-side ground truth (NEVER passed to the engine)
# ---------------------------------------------------------------------------
@dataclass
class Victim:
    victim_id: str
    true_position: Position
    env_type: str  # "buried" | "rooftop" | "street"
    phone_state: str  # "alive" | "dead" | "no_app"
    battery_pct: float
    injury_severity: float  # 0-1
    group_size: int
    alive: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "victim_id": self.victim_id,
            "true_position": _pos_to_list(self.true_position),
            "env_type": self.env_type,
            "phone_state": self.phone_state,
            "battery_pct": self.battery_pct,
            "injury_severity": self.injury_severity,
            "group_size": self.group_size,
            "alive": self.alive,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Victim":
        return cls(
            victim_id=d["victim_id"],
            true_position=_pos_from_list(d["true_position"]),
            env_type=d["env_type"],
            phone_state=d["phone_state"],
            battery_pct=d["battery_pct"],
            injury_severity=d["injury_severity"],
            group_size=d["group_size"],
            alive=d["alive"],
        )


# ---------------------------------------------------------------------------
# Simulation -> Engine
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    detection_id: str
    est_position: Position
    position_sigma: float  # metres, 1-sigma uncertainty radius
    sensor: str  # "thermal" | "uwb" | "rf"
    confidence: float  # 0-1
    timestamp: float  # seconds since scenario start
    env_hint: str  # "buried" | "surface" | "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "est_position": _pos_to_list(self.est_position),
            "position_sigma": self.position_sigma,
            "sensor": self.sensor,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "env_hint": self.env_hint,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Detection":
        return cls(
            detection_id=d["detection_id"],
            est_position=_pos_from_list(d["est_position"]),
            position_sigma=d["position_sigma"],
            sensor=d["sensor"],
            confidence=d["confidence"],
            timestamp=d["timestamp"],
            env_hint=d["env_hint"],
        )


# ---------------------------------------------------------------------------
# Engine-internal world model
# ---------------------------------------------------------------------------
@dataclass
class VictimBelief:
    belief_id: str
    position: Position
    uncertainty: float  # metres, 1-sigma
    env_type: str  # "buried" | "rooftop" | "street" | "unknown"
    est_group_size: int
    confidence: float  # 0-1
    survival_deadline: float  # minutes from now until p_survive -> ~0
    extraction_minutes: float  # estimated time to physically extract
    required_resources: List[str] = field(default_factory=list)
    detected_by: List[str] = field(default_factory=list)  # sensor names
    first_seen_t: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "belief_id": self.belief_id,
            "position": _pos_to_list(self.position),
            "uncertainty": self.uncertainty,
            "env_type": self.env_type,
            "est_group_size": self.est_group_size,
            "confidence": self.confidence,
            "survival_deadline": self.survival_deadline,
            "extraction_minutes": self.extraction_minutes,
            "required_resources": list(self.required_resources),
            "detected_by": list(self.detected_by),
            "first_seen_t": self.first_seen_t,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VictimBelief":
        return cls(
            belief_id=d["belief_id"],
            position=_pos_from_list(d["position"]),
            uncertainty=d["uncertainty"],
            env_type=d["env_type"],
            est_group_size=d["est_group_size"],
            confidence=d["confidence"],
            survival_deadline=d["survival_deadline"],
            extraction_minutes=d["extraction_minutes"],
            required_resources=list(d.get("required_resources", [])),
            detected_by=list(d.get("detected_by", [])),
            first_seen_t=d.get("first_seen_t", 0.0),
        )


@dataclass
class Resource:
    resource_id: str
    type: str  # "boat" | "excavator" | "medical"
    position: Position
    speed: float  # metres / minute
    status: str  # "idle" | "enroute" | "working"
    free_at: float  # sim time (seconds) at which this resource becomes free

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "type": self.type,
            "position": _pos_to_list(self.position),
            "speed": self.speed,
            "status": self.status,
            "free_at": self.free_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Resource":
        return cls(
            resource_id=d["resource_id"],
            type=d["type"],
            position=_pos_from_list(d["position"]),
            speed=d["speed"],
            status=d["status"],
            free_at=d["free_at"],
        )


@dataclass
class Drone:
    drone_id: str
    position: Position
    battery_pct: float
    sensors: List[str] = field(default_factory=list)
    comm_range: float = 300.0  # metres
    linked_to: List[str] = field(default_factory=list)  # drone_ids or "base"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drone_id": self.drone_id,
            "position": _pos_to_list(self.position),
            "battery_pct": self.battery_pct,
            "sensors": list(self.sensors),
            "comm_range": self.comm_range,
            "linked_to": list(self.linked_to),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Drone":
        return cls(
            drone_id=d["drone_id"],
            position=_pos_from_list(d["position"]),
            battery_pct=d["battery_pct"],
            sensors=list(d.get("sensors", [])),
            comm_range=d.get("comm_range", 300.0),
            linked_to=list(d.get("linked_to", [])),
        )


@dataclass
class EngineInput:
    t: float
    detections: List[Detection]
    drones: List[Drone]
    resources: List[Resource]
    environment: Dict[str, float]  # {"water_level": m, "rise_rate": m/hr}
    prior_map: np.ndarray  # 2D array, pre-disaster population density

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": self.t,
            "detections": [d.to_dict() for d in self.detections],
            "drones": [d.to_dict() for d in self.drones],
            "resources": [r.to_dict() for r in self.resources],
            "environment": dict(self.environment),
            "prior_map": self.prior_map.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngineInput":
        return cls(
            t=d["t"],
            detections=[Detection.from_dict(x) for x in d["detections"]],
            drones=[Drone.from_dict(x) for x in d["drones"]],
            resources=[Resource.from_dict(x) for x in d["resources"]],
            environment=dict(d["environment"]),
            prior_map=np.array(d["prior_map"], dtype=float),
        )


@dataclass
class Assignment:
    resource_id: str
    target_belief_id: str
    expected_value: float
    reason: str  # human-readable justification, must reference alternatives

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "target_belief_id": self.target_belief_id,
            "expected_value": self.expected_value,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Assignment":
        return cls(
            resource_id=d["resource_id"],
            target_belief_id=d["target_belief_id"],
            expected_value=d["expected_value"],
            reason=d["reason"],
        )


@dataclass
class EngineOutput:
    t: float
    drone_waypoints: Dict[str, Position]  # drone_id -> (x, y)
    assignments: List[Assignment]
    priority_list: List[str]  # belief_ids, ranked
    suspected_silent_zones: List[Tuple[float, float, float]]  # (x, y, risk)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": self.t,
            "drone_waypoints": {k: _pos_to_list(v) for k, v in self.drone_waypoints.items()},
            "assignments": [a.to_dict() for a in self.assignments],
            "priority_list": list(self.priority_list),
            "suspected_silent_zones": [
                [float(x), float(y), float(r)] for (x, y, r) in self.suspected_silent_zones
            ],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngineOutput":
        return cls(
            t=d["t"],
            drone_waypoints={k: _pos_from_list(v) for k, v in d["drone_waypoints"].items()},
            assignments=[Assignment.from_dict(x) for x in d["assignments"]],
            priority_list=list(d["priority_list"]),
            suspected_silent_zones=[(x, y, r) for (x, y, r) in d["suspected_silent_zones"]],
        )
