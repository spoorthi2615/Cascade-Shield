from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any

class NodeType(str, Enum):
    POWER_BUS = "PowerBus"
    GRID_INTERCONNECT = "GridInterconnect"
    WATER_PUMP = "WaterPump"
    TRAFFIC_CONTROLLER = "TrafficController"
    TRANSIT_NODE = "TransitNode"
    COMM_GATEWAY = "CommGateway"

class EdgeType(str, Enum):
    PHYSICAL_DEPENDENCY = "PHYSICAL_DEPENDENCY"
    LOGICAL_DEPENDENCY = "LOGICAL_DEPENDENCY"
    INFORMATIONAL_DEPENDENCY = "INFORMATIONAL_DEPENDENCY"

@dataclass
class Node:
    id: str
    type: NodeType
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Edge:
    source: str
    target: str
    type: EdgeType
    attributes: Dict[str, Any] = field(default_factory=dict)
