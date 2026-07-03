import os
import sys
import random
from collections import Counter
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.build_topology import parse_opendss_feeder, build_networkx_graph
from simulator.attack_generator import generate_scenario
from simulator.discrete_event_sim import CascadeSim, NodeState
import simpy

def run_calibration():
    feeder_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "123Bus", "IEEE123Master.dss")
    nodes, edges = parse_opendss_feeder(feeder_path)
    base_G = build_networkx_graph(nodes, edges)

    num_scenarios = 50
    tick_horizon = 50
    
    cascade_sizes = []
    found_mixed_recovery = False
    example_mixed_log = []

    for i in range(num_scenarios):
        # Fresh graph copy and random criticalities for each run
        G = base_G.copy()
        rng = random.Random(i)
        
        # Mixed criticality to ensure both FAILED and RECOVERED fire
        for n in G.nodes():
            G.nodes[n]["criticality"] = rng.random()

        scenario = generate_scenario(G, strategy="random", seed=i)
        env = simpy.Environment()
        sim = CascadeSim(env, G, scenario)
        
        env.process(sim.run())
        env.run(until=tick_horizon)

        infected_nodes = set()
        failed_count = 0
        recovered_count = 0
        
        for log in sim.logs:
            if log["state"] == NodeState.INFECTED.value:
                infected_nodes.add(log["node"])
            elif log["state"] == NodeState.FAILED.value:
                failed_count += 1
            elif log["state"] == NodeState.RECOVERED.value:
                recovered_count += 1

        cascade_sizes.append(len(infected_nodes))
        
        # Save a log that demonstrates both recovery and failure branches working
        if failed_count > 0 and recovered_count > 0 and not found_mixed_recovery:
            found_mixed_recovery = True
            example_mixed_log = sim.logs

    # Print Distribution
    print("--- Cascade Size Distribution (Total Infected Nodes) ---")
    size_counts = Counter(cascade_sizes)
    for size in sorted(size_counts.keys()):
        print(f"Size {size}: {size_counts[size]} scenarios")
        
    print(f"\nAverage cascade size: {sum(cascade_sizes)/len(cascade_sizes):.1f} nodes (Graph size: {len(base_G.nodes())})")
    
    if found_mixed_recovery:
        print("\n--- Example Log proving both FAILED and RECOVERED states trigger ---")
        for log in example_mixed_log:
            if log["state"] in ["F", "R"]:
                print(json.dumps(log))
    else:
        print("\nWARNING: No scenarios triggered both FAILED and RECOVERED.")

if __name__ == "__main__":
    run_calibration()
