import os
import sys
import random
import argparse
from typing import Dict, List, Tuple
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.build_topology import parse_opendss_feeder, build_networkx_graph, generate_synthetic_city
from simulator.attack_generator import AttackScenario, generate_scenario
from simulator.discrete_event_sim import CascadeSim, NodeState
from graph.schema import EdgeType
import simpy
import networkx as nx

# Conditional imports for PyTorch (fail gracefully if not installed yet)
try:
    import torch
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def create_base_tensors(G: nx.Graph) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
    """Converts the static graph topology and features into PyTorch tensors."""
    node_to_idx = {n: i for i, n in enumerate(G.nodes())}
    N = len(G.nodes())
    
    # x: [criticality, is_power, is_water, is_traffic, is_transit, in_degree, out_degree, is_compromised]
    x = torch.zeros((N, 8), dtype=torch.float)
    
    subsystems = ["power", "water", "traffic", "transit"]
    
    for n, data in G.nodes(data=True):
        idx = node_to_idx[n]
        x[idx, 0] = data.get("criticality", 0.5)
        
        sub_idx = subsystems.index(data.get("subsystem", "power"))
        x[idx, 1 + sub_idx] = 1.0
        
        x[idx, 5] = G.in_degree(n) if G.is_directed() else G.degree(n)
        x[idx, 6] = G.out_degree(n) if G.is_directed() else G.degree(n)
        
    edges = list(G.edges(data=True))
    edge_index = torch.zeros((2, len(edges)), dtype=torch.long)
    edge_attr = torch.zeros((len(edges), 4), dtype=torch.float)
    
    edge_types = [EdgeType.PHYSICAL_DEPENDENCY.value, EdgeType.LOGICAL_DEPENDENCY.value, EdgeType.INFORMATIONAL_DEPENDENCY.value]
    
    for i, (u, v, data) in enumerate(edges):
        edge_index[0, i] = node_to_idx[u]
        edge_index[1, i] = node_to_idx[v]
        
        e_type = data.get("type", EdgeType.PHYSICAL_DEPENDENCY.value)
        try:
            type_idx = edge_types.index(e_type)
            edge_attr[i, type_idx] = 1.0
        except ValueError:
            edge_attr[i, 0] = 1.0 # fallback physical
            
        edge_attr[i, 3] = data.get("weight", 1.0)
            
    return x, edge_index, edge_attr, node_to_idx

def generate_splits(G: nx.Graph, seed: int = 42) -> Dict[str, List[str]]:
    """Splits origin nodes into 70/15/15 stratified by subsystem."""
    rng = random.Random(seed)
    subsystem_nodes = {}
    
    for n, data in G.nodes(data=True):
        sub = data.get("subsystem", "power")
        subsystem_nodes.setdefault(sub, []).append(n)
        
    split = {"train": [], "val": [], "test": []}
    
    for sub, nodes in subsystem_nodes.items():
        rng.shuffle(nodes)
        n_train = int(len(nodes) * 0.7)
        n_val = int(len(nodes) * 0.15)
        
        split["train"].extend(nodes[:n_train])
        split["val"].extend(nodes[n_train:n_train+n_val])
        split["test"].extend(nodes[n_train+n_val:])
        
    return split

def run_dataset_generation(check_balance: bool = False):
    if not TORCH_AVAILABLE:
        print("PyTorch or torch_geometric not installed. Exiting.")
        return

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Topology
    feeder_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "123Bus", "IEEE123Master.dss")
    nodes, edges = parse_opendss_feeder(feeder_path)
    base_power_G = build_networkx_graph(nodes, edges)
    for n in base_power_G.nodes():
        base_power_G.nodes[n]["subsystem"] = "power"
        
    # Density 0.01 provides ~3 cross edges total on this graph, yielding ~42% cross-fraction.
    city_G = generate_synthetic_city(base_power_G, density_param=0.01, seed=42)
    
    # Assign FIXED criticalities to the topology so GNN can actually learn static vulnerabilities
    rng = random.Random(99)
    for n in city_G.nodes():
        city_G.nodes[n]["criticality"] = rng.random()

    # 2. Tensors & Splits
    x, edge_index, edge_attr, node_to_idx = create_base_tensors(city_G)
    splits = generate_splits(city_G, seed=42)
    
    num_scenarios = 100 if check_balance else 1000
    
    # For a balanced sampling, we pick randomly from the defined origin split sets.
    # In check_balance mode, we just run 100 random from train.
    
    total_infected = 0
    total_nodes = len(city_G.nodes())
    
    print(f"Generating {num_scenarios} scenarios... {'(BALANCE CHECK MODE)' if check_balance else ''}")
    
    for i in range(num_scenarios):
        # Decide which split this scenario belongs to
        if check_balance:
            split_name = "train"
        else:
            # roughly 70/15/15 scenario distribution
            rand_val = rng.random()
            if rand_val < 0.7: split_name = "train"
            elif rand_val < 0.85: split_name = "val"
            else: split_name = "test"
            
        origin = rng.choice(splits[split_name])
        
        scenario = AttackScenario(origin_node=origin, attack_type="credential_theft", seed=i)
        env = simpy.Environment()
        sim = CascadeSim(env, city_G, scenario)
        
        env.process(sim.run())
        env.run(until=50) # tick horizon
        
        y = torch.zeros(total_nodes, dtype=torch.float)
        y_time = torch.full((total_nodes,), -1.0, dtype=torch.float)
        
        # We define ground-truth INFECTED/FAILED as 1. 
        # For time, we take the earliest timestamp it hit INFECTED or EXPOSED.
        # CascadeSim logs: [timestamp, node, state, origin]
        infected_nodes = set()
        
        for log in sim.logs:
            if log["state"] in [NodeState.EXPOSED.value, NodeState.INFECTED.value, NodeState.FAILED.value]:
                idx = node_to_idx[log["node"]]
                y[idx] = 1.0
                if y_time[idx] == -1.0:
                    y_time[idx] = float(log["timestamp"])
                infected_nodes.add(log["node"])
                
        total_infected += len(infected_nodes)
        
        if not check_balance:
            scenario_x = x.clone()
            origin_idx = node_to_idx[origin]
            scenario_x[origin_idx, 7] = 1.0
            
            # Sanity check: ensure the compromised node is indeed the cascade origin
            assert y_time[origin_idx] == 0.0, f"Origin node {origin} does not have y_time == 0.0!"
            
            data = Data(x=scenario_x, edge_index=edge_index, edge_attr=edge_attr, y=y, y_time=y_time)
            data.split = split_name
            torch.save(data, os.path.join(out_dir, f"scenario_{i}.pt"))
            
    if check_balance:
        overall_fraction = total_infected / (total_nodes * num_scenarios)
        print("--- Class Balance Report ---")
        print(f"Total Scenarios: {num_scenarios}")
        print(f"Total Nodes per Scenario: {total_nodes}")
        print(f"Global Fraction Infected (y=1): {overall_fraction*100:.2f}%")
        print(f"Global Fraction Clean (y=0): {(1.0 - overall_fraction)*100:.2f}%")
        if overall_fraction < 0.05:
            print("WARNING: Highly skewed toward 0. Consider loss-weighting or subsampling in training.")
        else:
            print("Balance is acceptable (target 5-25%).")
    else:
        print(f"Successfully generated and saved {num_scenarios} PyTorch Geometric graphs to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-balance", action="store_true", help="Run 100 scenarios and print class balance")
    args = parser.parse_args()
    
    run_dataset_generation(check_balance=args.check_balance)
