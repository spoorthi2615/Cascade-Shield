import os
import sys
import torch
import simpy
import networkx as nx
from typing import Dict, List
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.shared import BaselinePredictor
from simulator.discrete_event_sim import CascadeSim, NodeState
from simulator.attack_generator import AttackScenario

class HybridCascadeSim(CascadeSim):
    def __init__(self, env, graph, scenario, edge_prob_map):
        super().__init__(env, graph, scenario)
        self.edge_prob_map = edge_prob_map
        
    def propagate_from(self, node: str):
        """Propagation driven by GNN-predicted edge probabilities."""
        while self.node_states[node] == NodeState.INFECTED:
            for neighbor in self.graph.neighbors(node):
                if self.node_states[neighbor] == NodeState.SUSCEPTIBLE:
                    # Look up custom edge probability, default to 0.35 if missing
                    prob = self.edge_prob_map.get((node, neighbor), 0.35)
                    
                    if self.rng.random() < prob:
                        self.node_states[neighbor] = NodeState.EXPOSED
                        self.log_event(neighbor, NodeState.EXPOSED, node)
                        self.env.process(self.delayed_infection(neighbor, node, 1))
            yield self.env.timeout(1)

class HybridSEIRPredictor(BaselinePredictor):
    def __init__(self, model, num_trials: int = 10):
        super().__init__(name="GNN-SEIR Hybrid")
        self.model = model
        self.num_trials = num_trials
        
        # Diagnostic tracking lists
        self.diag_origin_infected: List[bool] = []
        self.diag_origin_matched: List[bool] = []
        
        # Caching
        self._last_graph_id = None
        self._cached_risk = None
        self._cached_time = None
        self.is_temporal = True

    def _ensure_simulations(self, graph: nx.Graph, origin_node: str = None):
        # We need the current batch to perform GNN forward pass
        batch = getattr(self, "current_batch", None)
        if batch is None:
            raise ValueError("HybridSEIRPredictor requires current_batch to be set prior to evaluation.")
            
        if id(graph) == self._last_graph_id:
            return
            
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # 1. Run GNN forward pass to predict node logits, times, and edge transmission probabilities
        with torch.no_grad():
            logits_clf, _, edge_probs = self.model(
                batch.x.to(device), 
                batch.edge_index.to(device), 
                batch.edge_attr.to(device), 
                getattr(batch, 'batch', None), 
                return_edge_probs=True
            )
            
        # 2. Identify predicted origin node.
        # If origin_node is passed by the evaluator, use it to ensure aligned simulations.
        # Otherwise, fall back to guessing from GNN logits.
        N = batch.num_nodes
        origin_idx = torch.argmax(logits_clf[:N]).item()
        
        if origin_node is not None:
            origin_name = origin_node
            # Find the idx of the provided origin_node for diagnostic logging
            if hasattr(self, 'node_to_idx') and origin_name in self.node_to_idx:
                origin_idx = self.node_to_idx[origin_name]
        else:
            origin_name = self.idx_to_node[origin_idx]
        
        # 3. Log diagnostic metrics for origin pick accuracy
        true_y = batch.y.cpu().numpy()
        valid_times = batch.y_time.clone()
        valid_times[batch.y_time < 0] = float('inf')
        true_origin_idx = torch.argmin(valid_times).item()
        
        self.diag_origin_infected.append(bool(true_y[origin_idx] == 1.0))
        self.diag_origin_matched.append(bool(origin_idx == true_origin_idx))
        
        # 4. Map GNN edge predictions to the graph edges
        edge_probs_numpy = edge_probs.cpu().numpy()
        edge_index_numpy = batch.edge_index.cpu().numpy()
        
        edge_prob_map = {}
        for idx in range(edge_index_numpy.shape[1]):
            u_idx = edge_index_numpy[0, idx]
            v_idx = edge_index_numpy[1, idx]
            if u_idx < N and v_idx < N:
                u_name = self.idx_to_node[u_idx]
                v_name = self.idx_to_node[v_idx]
                edge_prob_map[(u_name, v_name)] = float(edge_probs_numpy[idx])
                
        # 5. Run Monte Carlo simulations starting from GNN predicted origin
        node_infection_counts = {n: 0 for n in graph.nodes()}
        node_infection_times = {n: [] for n in graph.nodes()}
        
        for i in range(self.num_trials):
            scenario = AttackScenario(origin_node=origin_name, attack_type="classical_seir", seed=i)
            env = simpy.Environment()
            sim = HybridCascadeSim(env, graph, scenario, edge_prob_map)
            env.process(sim.run())
            env.run(until=50)
            
            infected_this_run = set()
            for log in sim.logs:
                if log["state"] in [NodeState.EXPOSED.value, NodeState.INFECTED.value, NodeState.FAILED.value]:
                    infected_this_run.add(log["node"])
                    node_infection_times[log["node"]].append(float(log["timestamp"]))
            for n in infected_this_run:
                node_infection_counts[n] += 1
                
        total_runs = self.num_trials
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
        self._ensure_simulations(graph, origin_node)
        return self._cached_risk

    def predict_time(self, graph: nx.Graph, origin_node: str = None) -> Dict[str, float]:
        self._ensure_simulations(graph, origin_node)
        return self._cached_time
