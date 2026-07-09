import sys
import os
import random
import numpy as np
import simpy
from scipy.stats import skew

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "simulator"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "graph-engine"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "graph-engine", "parsers"))

from ingest import build_fused_graph
from generate_dataset import nx_to_sim_dicts
from engine import SimPyCascadeSimulator
from ics_parser import parse_ics_events

def get_stats(sizes):
    if not sizes: return 0,0,0
    sizes = np.array(sizes)
    med = np.median(sizes)
    s = skew(sizes)
    runaway = np.mean(sizes > (sizes.max() * 0.5)) if sizes.max() > 0 else 0
    return med, s, runaway

def main():
    print("Building topology...")
    G = build_fused_graph(1000)
    node_dicts, edge_dicts = nx_to_sim_dicts(G, cross_weight=0.5)
    num_nodes = G.number_of_nodes()
    
    # Load ICS origins
    ics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "01-Basic", "01-Basic", "raw", "test", "events.jsonl")
    ics_events = parse_ics_events(ics_path)
    
    target_nodes = [n for n, d in G.nodes(data=True) if d.get('subsystem') in ['power', 'water']]
    valid_ics_origins = [e['target_node'] for e in ics_events if e['target_node'] in target_nodes]
    print(f"Found {len(valid_ics_origins)} valid ICS origins out of {len(target_nodes)} target nodes.")
    
    if not valid_ics_origins:
        print("No valid ICS origins found! Cannot compare.")
        return
        
    def run_sim(origins):
        sizes = []
        for i, origin_id in enumerate(origins):
            env = simpy.Environment()
            sim = SimPyCascadeSimulator(env, node_dicts, edge_dicts, logger=None)
            sim.infect_node(origin_id, parent_id=None, propagation_type="Initial Injection")
            detection_delay = random.uniform(5.0, 20.0)
            env.process(sim.run_mitigation_loop(detection_delay))
            env.run(until=1500.0)
            
            infected = sum(1 for state in sim.states.values() if state in ["I", "R"])
            sizes.append(infected / num_nodes)
        return sizes

    print("Running 300 ICS-driven scenarios...")
    ics_origins = random.choices(valid_ics_origins, k=300)
    ics_sizes = run_sim(ics_origins)
    
    print("Running 300 random SEIR scenarios...")
    random_origins = random.choices(target_nodes, k=300)
    rnd_sizes = run_sim(random_origins)
    
    ics_med, ics_skew, ics_runaway = get_stats(ics_sizes)
    rnd_med, rnd_skew, rnd_runaway = get_stats(rnd_sizes)
    
    print("\n--- Cascade Size Profiles ---")
    print(f"Random SEIR: Median={rnd_med*100:.1f}%, Skew={rnd_skew:.2f}, Runaway Rate={rnd_runaway*100:.1f}%")
    print(f"ICS-Driven : Median={ics_med*100:.1f}%, Skew={ics_skew:.2f}, Runaway Rate={ics_runaway*100:.1f}%")

if __name__ == "__main__":
    main()
