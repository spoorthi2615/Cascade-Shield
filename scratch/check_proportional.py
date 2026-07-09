import sys, random, numpy as np
sys.path.insert(0, r'd:\projects\cascade sheild\data\graph-engine')
sys.path.insert(0, r'd:\projects\cascade sheild\data\simulator')
from ingest import build_fused_graph
from generate_dataset import nx_to_sim_dicts
import simpy
from engine import SimPyCascadeSimulator

G = build_fused_graph(1000)
num_nodes = G.number_of_nodes()
nd, ed = nx_to_sim_dicts(G, cross_weight=0.5)

random.seed(42)
all_nodes = list(G.nodes())
origins = random.choices(all_nodes, k=300)

sizes = []
for origin_id in origins:
    env = simpy.Environment()
    sim = SimPyCascadeSimulator(env, nd, ed, logger=None)
    sim.infect_node(origin_id, parent_id=None, propagation_type='Initial Injection')
    env.process(sim.run_mitigation_loop(random.uniform(5.0, 20.0)))
    env.run(until=1500.0)
    sizes.append(sum(1 for s in sim.states.values() if s in ['I', 'R']) / num_nodes)

print(f"Proportional Median: {np.median(sizes)*100:.2f}%")
print(f"Proportional Mean: {np.mean(sizes)*100:.2f}%")
