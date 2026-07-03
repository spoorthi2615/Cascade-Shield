import random
from dataclasses import dataclass
from typing import Optional

@dataclass
class AttackScenario:
    origin_node: str
    attack_type: str
    seed: int
    injected_at_timestep: int = 0

def generate_scenario(graph, strategy: str = "random", seed: Optional[int] = None) -> AttackScenario:
    """
    Generate an attack scenario by selecting an origin node based on the strategy.
    Strategies: "random", "degree_weighted", "subsystem_stratified"
    """
    rng = random.Random(seed)
    nodes = list(graph.nodes(data=True))
    
    if not nodes:
        raise ValueError("Graph is empty")

    if strategy == "random":
        origin_node = rng.choice(nodes)[0]
    elif strategy == "degree_weighted":
        degrees = dict(graph.degree())
        total_degree = sum(degrees.values())
        if total_degree == 0:
            origin_node = rng.choice(nodes)[0]
        else:
            probs = [degrees[n] / total_degree for n, _ in nodes]
            origin_node = rng.choices([n for n, _ in nodes], weights=probs, k=1)[0]
    elif strategy == "subsystem_stratified":
        # Group by node type
        subsystems = {}
        for n, data in nodes:
            ntype = data.get("type", "unknown")
            subsystems.setdefault(ntype, []).append(n)
        
        # Pick a random subsystem, then random node within it
        chosen_subsystem = rng.choice(list(subsystems.keys()))
        origin_node = rng.choice(subsystems[chosen_subsystem])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    attack_types = ["credential_theft", "firmware_exploit", "network_pivot"]
    return AttackScenario(
        origin_node=origin_node,
        attack_type=rng.choice(attack_types),
        seed=seed if seed is not None else rng.randint(0, 999999)
    )
