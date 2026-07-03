import simpy
import networkx as nx
import random
import json
from enum import Enum
from simulator.attack_generator import AttackScenario
from simulator.propagation_rules import get_propagation_rule, propagation_probability

class NodeState(str, Enum):
    SUSCEPTIBLE = "S"
    EXPOSED = "E"
    INFECTED = "I"
    RECOVERED = "R"
    FAILED = "F"

class CascadeSim:
    def __init__(self, env: simpy.Environment, graph: nx.Graph, scenario: AttackScenario, recovery_threshold: float = 0.5):
        self.env = env
        self.graph = graph
        self.scenario = scenario
        self.recovery_threshold = recovery_threshold
        self.rng = random.Random(scenario.seed)
        
        self.node_states = {n: NodeState.SUSCEPTIBLE for n in graph.nodes()}
        self.logs = []
        
    def log_event(self, node: str, state: NodeState, origin: str = None):
        self.logs.append({
            "timestamp": self.env.now,
            "node": node,
            "state": state.value,
            "origin_node": origin
        })

    def infect_node(self, node: str, origin: str):
        if self.node_states[node] not in (NodeState.SUSCEPTIBLE, NodeState.EXPOSED):
            return
            
        self.node_states[node] = NodeState.INFECTED
        self.log_event(node, NodeState.INFECTED, origin)
        
        # Start propagation and recovery processes
        self.env.process(self.propagate_from(node))
        self.env.process(self.recover_or_fail(node))

    def propagate_from(self, node: str):
        """Attempt propagation to all healthy neighbors periodically while infected."""
        while self.node_states[node] == NodeState.INFECTED:
            for neighbor in self.graph.neighbors(node):
                if self.node_states[neighbor] == NodeState.SUSCEPTIBLE:
                    edge_data = self.graph.get_edge_data(node, neighbor)
                    edge_type = edge_data.get("type", "LOGICAL_DEPENDENCY")
                    
                    rule = get_propagation_rule(edge_type)
                    criticality = self.graph.nodes[neighbor].get("criticality", 1.0)
                    weight = edge_data.get("weight", 1.0)
                    prob = propagation_probability(rule, criticality) * weight
                    
                    if self.rng.random() < prob:
                        # Transition to EXPOSED, waiting for latency_hops ticks
                        self.node_states[neighbor] = NodeState.EXPOSED
                        self.log_event(neighbor, NodeState.EXPOSED, node)
                        
                        # Schedule infection after latency
                        self.env.process(self.delayed_infection(neighbor, node, rule.latency_hops))
            yield self.env.timeout(1)

    def delayed_infection(self, node: str, origin: str, latency: int):
        yield self.env.timeout(latency)
        if self.node_states[node] == NodeState.EXPOSED:
            self.infect_node(node, origin)

    def recover_or_fail(self, node: str):
        """Determine if node recovers or fails permanently."""
        # Wait a fixed recovery time, e.g., 5 ticks
        yield self.env.timeout(5)
        
        if self.node_states[node] == NodeState.INFECTED:
            criticality = self.graph.nodes[node].get("criticality", 0.3)
            if criticality >= self.recovery_threshold:
                self.node_states[node] = NodeState.FAILED
                self.log_event(node, NodeState.FAILED)
            else:
                self.node_states[node] = NodeState.RECOVERED
                self.log_event(node, NodeState.RECOVERED)

    def run(self):
        # Inject initial compromise
        yield self.env.timeout(self.scenario.injected_at_timestep)
        self.infect_node(self.scenario.origin_node, self.scenario.origin_node)

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from graph.build_topology import parse_opendss_feeder, build_networkx_graph
    from simulator.attack_generator import generate_scenario

    feeder_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "13Bus", "IEEE13Nodeckt.dss")
    nodes, edges = parse_opendss_feeder(feeder_path)
    G = build_networkx_graph(nodes, edges)
    
    # Assign high criticality to ensure recovery_threshold (0.5) is exceeded -> FAILED
    for n in G.nodes():
        G.nodes[n]["criticality"] = 1.0



    scenario = generate_scenario(G, strategy="degree_weighted", seed=42)
    print(f"Generated Scenario: Origin={scenario.origin_node}, Type={scenario.attack_type}")

    env = simpy.Environment()
    sim = CascadeSim(env, G, scenario)
    
    # Start the simulation process
    env.process(sim.run())
    env.run(until=20)  # run for 20 discrete ticks

    print(f"\\nSimulation completed with {len(sim.logs)} events.")
    print("Sample logs:")
    for log in sim.logs:
        print(json.dumps(log))
