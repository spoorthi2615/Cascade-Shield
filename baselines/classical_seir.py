import os
import sys
import random
import simpy
import networkx as nx
import time
from typing import Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.shared import BaselinePredictor
from simulator.discrete_event_sim import CascadeSim, NodeState
from simulator.attack_generator import AttackScenario
from graph.build_topology import parse_opendss_feeder, build_networkx_graph, generate_synthetic_city

class UniformCascadeSim(CascadeSim):
    def propagate_from(self, node: str):
        """Attempt propagation with strictly uniform probabilities and latency, ignoring edge types and node criticalities."""
        while self.node_states[node] == NodeState.INFECTED:
            for neighbor in self.graph.neighbors(node):
                if self.node_states[neighbor] == NodeState.SUSCEPTIBLE:
                    # Uniform probability
                    prob = 0.35
                    
                    if self.rng.random() < prob:
                        # Transition to EXPOSED, uniform latency of 1 hop
                        self.node_states[neighbor] = NodeState.EXPOSED
                        self.log_event(neighbor, NodeState.EXPOSED, node)
                        
                        # Schedule infection after latency = 1
                        self.env.process(self.delayed_infection(neighbor, node, 1))
            yield self.env.timeout(1)

class ClassicalSEIRPredictor(BaselinePredictor):
    def __init__(self, num_trials: int = 20):
        super().__init__(name="ClassicalSEIR")
        self.num_trials = num_trials

    def predict(self, graph: nx.Graph, origin_node: str = None) -> Dict[str, float]:
        if not origin_node:
            raise ValueError("ClassicalSEIR requires an origin_node to simulate.")

        node_infection_counts = {n: 0 for n in graph.nodes()}

        for i in range(self.num_trials):
            # We must use a unique seed per trial to explore the stochastic space
            scenario = AttackScenario(origin_node=origin_node, attack_type="classical_seir", seed=i)
            env = simpy.Environment()
            sim = UniformCascadeSim(env, graph, scenario)
            
            env.process(sim.run())
            env.run(until=50) # Same tick horizon as generate_dataset.py

            # Track which nodes got infected in this trial
            infected_this_run = set()
            for log in sim.logs:
                if log["state"] in [NodeState.EXPOSED.value, NodeState.INFECTED.value, NodeState.FAILED.value]:
                    infected_this_run.add(log["node"])

            for n in infected_this_run:
                node_infection_counts[n] += 1

        # Return the fraction of trials each node was infected
        risk_scores = {n: count / self.num_trials for n, count in node_infection_counts.items()}
        return risk_scores

    def predict_time(self, graph: nx.Graph, origin_node: str = None) -> Dict[str, float]:
        if not origin_node:
            raise ValueError("ClassicalSEIR requires an origin_node to simulate.")

        node_infection_times = {n: [] for n in graph.nodes()}

        for i in range(self.num_trials):
            scenario = AttackScenario(origin_node=origin_node, attack_type="classical_seir", seed=i)
            env = simpy.Environment()
            sim = UniformCascadeSim(env, graph, scenario)
            
            env.process(sim.run())
            env.run(until=50)

            for log in sim.logs:
                if log["state"] in [NodeState.EXPOSED.value, NodeState.INFECTED.value, NodeState.FAILED.value]:
                    node_infection_times[log["node"]].append(float(log["timestamp"]))

        # Average time of infection across trials where the node was infected
        # If never infected, default to 50.0 (max horizon)
        time_predictions = {}
        for n, times in node_infection_times.items():
            if len(times) > 0:
                time_predictions[n] = sum(times) / len(times)
            else:
                time_predictions[n] = 50.0
                
        return time_predictions

if __name__ == "__main__":
    print("--- Running Timing Test for ClassicalSEIR (N=20 Monte Carlo) ---")
    feeder_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "123Bus", "IEEE123Master.dss")
    nodes, edges = parse_opendss_feeder(feeder_path)
    base_power_G = build_networkx_graph(nodes, edges)
    for n in base_power_G.nodes():
        base_power_G.nodes[n]["subsystem"] = "power"
        
    city_G = generate_synthetic_city(base_power_G, density_param=0.01, cross_weight=0.30, seed=42)
    
    # Pick a random origin
    origin = list(city_G.nodes())[0]
    
    predictor = ClassicalSEIRPredictor(num_trials=20)
    
    start_time = time.time()
    predictions = predictor.predict(city_G, origin_node=origin)
    end_time = time.time()
    
    print(f"Elapsed time for N=20 trials on 246-node graph: {end_time - start_time:.4f} seconds")
    
    infected_nodes = {k: v for k, v in predictions.items() if v > 0.0}
    print(f"Nodes ever infected across 20 trials: {len(infected_nodes)} / {len(city_G.nodes())}")
    
    # Show top 5 highest probability nodes
    top_nodes = sorted(infected_nodes.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"Top 5 highest risk nodes for origin {origin}:")
    for n, p in top_nodes:
        print(f"  {n}: {p:.2f}")

class BlindClassicalSEIRPredictor(BaselinePredictor):
    def __init__(self, num_trials: int = 10):
        super().__init__(name="BlindClassicalSEIR")
        self.num_trials = num_trials
        self._last_graph_id = None
        self._cached_risk = None
        self._cached_time = None
        self.is_temporal = True

    def _ensure_simulations(self, graph: nx.Graph):
        if id(graph) == self._last_graph_id:
            return
            
        node_infection_counts = {n: 0 for n in graph.nodes()}
        node_infection_times = {n: [] for n in graph.nodes()}
        nodes = list(graph.nodes())
        
        for origin in nodes:
            for i in range(self.num_trials):
                scenario = AttackScenario(origin_node=origin, attack_type="classical_seir", seed=i)
                env = simpy.Environment()
                sim = UniformCascadeSim(env, graph, scenario)
                env.process(sim.run())
                env.run(until=50)
                
                infected_this_run = set()
                for log in sim.logs:
                    if log["state"] in [NodeState.EXPOSED.value, NodeState.INFECTED.value, NodeState.FAILED.value]:
                        infected_this_run.add(log["node"])
                        node_infection_times[log["node"]].append(float(log["timestamp"]))
                for n in infected_this_run:
                    node_infection_counts[n] += 1
                    
        total_runs = len(nodes) * self.num_trials
        self._cached_risk = {n: count / total_runs for n, count in node_infection_counts.items()}
        
        time_predictions = {}
        for n, times in node_infection_times.items():
            if len(times) > 0:
                time_predictions[n] = sum(times) / len(times)
            else:
                time_predictions[n] = 50.0
        self._cached_time = time_predictions
        self._last_graph_id = id(graph)

    def predict(self, graph: nx.Graph, origin_node: str = None) -> Dict[str, float]:
        self._ensure_simulations(graph)
        return self._cached_risk

    def predict_time(self, graph: nx.Graph, origin_node: str = None) -> Dict[str, float]:
        self._ensure_simulations(graph)
        return self._cached_time
