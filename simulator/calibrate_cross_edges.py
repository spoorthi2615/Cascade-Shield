import os
import sys
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.build_topology import parse_opendss_feeder, build_networkx_graph, generate_synthetic_city
from simulator.attack_generator import generate_scenario
from simulator.discrete_event_sim import CascadeSim, NodeState
import simpy

def get_node_subsystem(G, node_id):
    node_data = G.nodes[node_id]
    if "subsystem" in node_data:
        return node_data["subsystem"]
    # Power nodes don't have a subsystem explicitly set in generate_synthetic_city yet, default to "power"
    return "power"

def run_calibration():
    feeder_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "123Bus", "IEEE123Master.dss")
    nodes, edges = parse_opendss_feeder(feeder_path)
    base_power_G = build_networkx_graph(nodes, edges)

    # Ensure power nodes have subsystem="power"
    for n in base_power_G.nodes():
        base_power_G.nodes[n]["subsystem"] = "power"

    weights = [0.1, 0.3, 0.5, 0.7, 1.0]
    num_scenarios = 100
    tick_horizon = 50

    print("--- Cross-Subsystem Edge Weight Calibration (Density fixed at 0.01 / 3 edges) ---")

    for weight in weights:
        # Generate full city graph with exactly 3 cross-edges, but parameterized weight
        city_G = generate_synthetic_city(base_power_G, density_param=0.01, cross_weight=weight, seed=42)
        
        cross_boundary_count = 0
        total_cascades = 0

        for i in range(num_scenarios):
            G = city_G.copy()
            rng = random.Random(i)
            
            # Assign mixed criticality
            for n in G.nodes():
                G.nodes[n]["criticality"] = rng.random()

            # Generate scenario using stratified sampling to test all origin types
            scenario = generate_scenario(G, strategy="subsystem_stratified", seed=i)
            origin_subsystem = get_node_subsystem(G, scenario.origin_node)

            env = simpy.Environment()
            sim = CascadeSim(env, G, scenario)
            
            env.process(sim.run())
            env.run(until=tick_horizon)

            infected_nodes = set()
            for log in sim.logs:
                if log["state"] == NodeState.INFECTED.value:
                    infected_nodes.add(log["node"])
            
            # Did it cascade beyond the origin?
            if len(infected_nodes) > 1:
                total_cascades += 1
                # Did it cross a boundary?
                crossed = any(get_node_subsystem(G, n) != origin_subsystem for n in infected_nodes)
                if crossed:
                    cross_boundary_count += 1

        cross_fraction = (cross_boundary_count / total_cascades) if total_cascades > 0 else 0
        print(f"Weight {weight:.2f}: {cross_fraction*100:.1f}% of cascades crossed a subsystem boundary (out of {total_cascades} multi-node cascades).")

if __name__ == "__main__":
    run_calibration()
