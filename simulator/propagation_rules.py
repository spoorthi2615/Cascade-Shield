from enum import Enum
from dataclasses import dataclass

class DependencyType(str, Enum):
    PHYSICAL = "PHYSICAL_DEPENDENCY"
    LOGICAL = "LOGICAL_DEPENDENCY"
    INFORMATIONAL = "INFORMATIONAL_DEPENDENCY"

@dataclass
class PropagationRule:
    dependency_type: DependencyType
    base_probability: float
    latency_hops: int

RULES = {
    DependencyType.PHYSICAL: PropagationRule(DependencyType.PHYSICAL, base_probability=0.65, latency_hops=2),
    DependencyType.LOGICAL: PropagationRule(DependencyType.LOGICAL, base_probability=0.80, latency_hops=1),
    DependencyType.INFORMATIONAL: PropagationRule(DependencyType.INFORMATIONAL, base_probability=0.95, latency_hops=1),
}

def get_propagation_rule(edge_type: str) -> PropagationRule:
    """Safely get rule based on string edge type from NetworkX graph."""
    for dep_type in DependencyType:
        if dep_type.value == edge_type:
            return RULES[dep_type]
    # Fallback default
    return PropagationRule(DependencyType.LOGICAL, base_probability=0.1, latency_hops=1)

def propagation_probability(rule: PropagationRule, node_criticality_weight: float) -> float:
    """Combines base rule probability with target-node criticality/coupling strength."""
    return min(1.0, rule.base_probability * node_criticality_weight)
