import sys
import os
import pickle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "graph-engine"))

import json
import random
from typing import List, Dict, Any
import numpy as np
from engine import SimPyCascadeSimulator
from ingest import build_fused_graph
import simpy

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")

def compute_chokepoint(cascade_path: List[str], edges: List[Dict[str, Any]], origin: str) -> str:
    if len(cascade_path) <= 1:
        return "None"
    candidates = [node for node in cascade_path if node != origin]
    if not candidates:
        return "None"
    dependency_count = {}
    for node in candidates:
        direct_targets = [
            e["target"] for e in edges 
            if e["source"] == node and e["target"] in cascade_path
        ]
        dependency_count[node] = len(direct_targets)
    best_candidate = max(dependency_count, key=dependency_count.get)
    return best_candidate

def nx_to_sim_dicts(G, cross_weight=0.5):
    node_dicts = {}
    for n, d in G.nodes(data=True):
        sub = d.get('subsystem', 'road')
        crit = {'power': 0.80, 'water': 0.70, 'road': 0.40}.get(sub, 0.40)
        node_dicts[n] = {
            'id': n, 'name': d.get('name', n), 'label': d.get('type', sub),
            'criticality': crit, 'compromised': False, 'ip_address': None,
            'extra_properties': {'subsystem': sub, 'coord_synthetic': d.get('coord_synthetic', False)}
        }
    edge_dicts = []
    for u, v, d in G.edges(data=True):
        w = float(d.get('weight', 1.0))
        t = d.get('type', 'PHYSICAL')
        if t == 'cross_layer': w = cross_weight
        edge_dicts.append({
            'source': u, 'target': v, 'type': t,
            'weight': w, 'description': d.get('description', '')
        })
    return node_dicts, edge_dicts

def generate_and_save_dataset():
    thresholds = [1000]
    total_scenarios_per_threshold = 1000
    until_time = 1500.0

    for threshold in thresholds:
        print(f"\n========================================")
        print(f"Generating dataset for Threshold: {threshold}m")
        print(f"========================================")
        G = build_fused_graph(threshold)
        
        # Serialize the master topology so convert_dataset doesn't have to rebuild it
        threshold_dir = os.path.join(DATASET_DIR, f"{threshold}m")
        os.makedirs(threshold_dir, exist_ok=True)
        topo_path = os.path.join(threshold_dir, "base_topology.gpickle")
        with open(topo_path, 'wb') as f:
            pickle.dump(G, f)
            
        node_dicts, edge_dicts = nx_to_sim_dicts(G, cross_weight=0.5)
        num_nodes = G.number_of_nodes()
        
        scenarios = []
        target_nodes = list(node_dicts.keys())
        
        random.seed(threshold)
        np.random.seed(threshold)
        
        # Load ICS events to use as ground-truth origins
        import sys
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "graph-engine", "parsers"))
        from ics_parser import parse_ics_events
        ics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "01-Basic", "01-Basic", "raw", "test", "events.jsonl")
        ics_events = parse_ics_events(ics_path)
        valid_ics_origins = [e['target_node'] for e in ics_events if e['target_node'] in target_nodes]
        if not valid_ics_origins:
            print("WARNING: No valid ICS origins found. Falling back to random targets.")
            valid_ics_origins = target_nodes
            
        # Draw 1000 origins from the ICS target set
        origins = random.choices(valid_ics_origins, k=total_scenarios_per_threshold)
        
        print(f"Running {total_scenarios_per_threshold} scenarios (until={until_time})...")
        
        for i, origin_id in enumerate(origins):
            if (i+1) % 100 == 0:
                print(f"  Processed {i+1}/{total_scenarios_per_threshold}...")
                
            env = simpy.Environment()
            simulator = SimPyCascadeSimulator(env, node_dicts, edge_dicts, logger=None)
            
            simulator.infect_node(origin_id, parent_id=None, propagation_type="Initial Injection")
            detection_delay = random.uniform(5.0, 20.0)
            env.process(simulator.run_mitigation_loop(detection_delay))
            
            env.run(until=until_time)
            
            compromised_nodes = [
                nid for nid, state in simulator.states.items() 
                if state in ["I", "R"] and nid in simulator.compromised_times
            ]
            cascade_path = sorted(compromised_nodes, key=lambda x: simulator.compromised_times[x])
            
            time_to_impact = -1.0
            for nid in cascade_path:
                if node_dicts[nid]["criticality"] >= 0.8 and nid != origin_id:
                    time_to_impact = float(simulator.compromised_times[nid])
                    break
            
            chokepoint = compute_chokepoint(cascade_path, edge_dicts, origin_id)
            confidence = float(min(1.0, len(cascade_path) / (num_nodes * 0.5) + 0.1))
            
            scenarios.append({
                "origin_node": origin_id,
                "cascade_path": cascade_path,
                "time_to_impact": time_to_impact,
                "chokepoint_recommendation": chokepoint,
                "confidence": confidence,
                "metrics": {
                    "compromised_count": len(cascade_path),
                    "total_nodes": num_nodes,
                    "fraction_compromised": len(cascade_path) / num_nodes
                }
            })

        print(f"Generated {len(scenarios)} scenarios for {threshold}m threshold.")

        # Partition dataset into train (70%), val (15%), test (15%) splits
        random.shuffle(scenarios)
        n_total = len(scenarios)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)
        
        splits = {
            "train": scenarios[:n_train],
            "val": scenarios[n_train:n_train + n_val],
            "test": scenarios[n_train + n_val:]
        }
        
        # Save to threshold-specific directory
        threshold_dir = os.path.join(DATASET_DIR, f"{threshold}m")
        os.makedirs(threshold_dir, exist_ok=True)
        
        for split_name, split_data in splits.items():
            file_path = os.path.join(threshold_dir, f"{split_name}.json")
            with open(file_path, "w") as f:
                json.dump(split_data, f, indent=2)
            print(f"Saved {len(split_data)} scenarios to {file_path}")

if __name__ == "__main__":
    generate_and_save_dataset()
