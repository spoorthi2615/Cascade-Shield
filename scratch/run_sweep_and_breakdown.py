import sys, os, random, time
import numpy as np
sys.path.insert(0, r'd:\projects\cascade sheild\data\graph-engine')
sys.path.insert(0, r'd:\projects\cascade sheild\data\graph-engine\parsers')
sys.path.insert(0, r'd:\projects\cascade sheild\data\simulator')
import simpy
from engine import SimPyCascadeSimulator
from ingest import build_fused_graph

UNTIL_TIME = 1500 # Approx 1.0x radius traversal time (1540)
RUNAWAY_PCT = 0.20

G = build_fused_graph(1000)
N = G.number_of_nodes()
RUNAWAY_THRESHOLD = int(N * RUNAWAY_PCT)

def run_scenarios(node_dicts, edge_dicts, origins, until_val):
    sizes = []
    for origin_id in origins:
        env = simpy.Environment()
        sim = SimPyCascadeSimulator(env, node_dicts, edge_dicts, logger=None)
        sim.infect_node(origin_id, parent_id=None, propagation_type='Initial Injection')
        env.process(sim.run_mitigation_loop(random.uniform(5.0, 20.0)))
        env.run(until=until_val)
        comp = [n for n, s in sim.states.items() if s in ['I', 'R'] and n in sim.compromised_times]
        sizes.append(len(comp))
    return sizes

def build_sim_dicts(G, cross_weight=0.5):
    node_dicts = {}
    for n, d in G.nodes(data=True):
        sub = d.get('subsystem', 'road')
        crit = {'power': 0.80, 'water': 0.70, 'road': 0.40}.get(sub, 0.40)
        node_dicts[n] = {'id': n, 'name': d.get('name', n), 'label': d.get('type', sub),
                         'criticality': crit, 'compromised': False, 'ip_address': None,
                         'extra_properties': {'subsystem': sub}}
    edge_dicts = []
    for u, v, d in G.edges(data=True):
        w = float(d.get('weight', 1.0))
        t = d.get('type', 'PHYSICAL')
        if t == 'cross_layer': w = cross_weight
        edge_dicts.append({'source': u, 'target': v, 'type': t,
                           'weight': w, 'description': d.get('description', '')})
    return node_dicts, edge_dicts

all_nodes = list(G.nodes())
pwr_nodes = [n for n in all_nodes if n.startswith('PWR_')]
water_nodes = [n for n in all_nodes if n.startswith('W_')]
road_nodes = [n for n in all_nodes if n.startswith('R_INT_')]

origins_by_sub = {
    'power': random.choices(pwr_nodes, k=50),
    'water': random.choices(water_nodes, k=200),
    'road': random.choices(road_nodes, k=50)
}
mixed_origins = origins_by_sub['power'] + origins_by_sub['water'] + origins_by_sub['road']

print("=" * 60)
print("PART 1: CROSS-LAYER WEIGHT SENSITIVITY (until=1500, n=300)")
print("=" * 60)
for w in [0.3, 0.5, 0.7]:
    random.seed(42)  # Reset seed so stochastic propagation is identical for weight comparison
    nd, ed = build_sim_dicts(G, cross_weight=w)
    sizes = run_scenarios(nd, ed, mixed_origins, UNTIL_TIME)
    print(f"Weight {w:.1f} | Mean size: {np.mean(sizes):.1f} ({np.mean(sizes)/N*100:.1f}%), Median: {np.median(sizes):.1f} ({np.median(sizes)/N*100:.1f}%)")

print("\n" + "=" * 60)
print("PART 2: SUBSYSTEM BREAKDOWN (until=1500, cross_weight=0.5, Power=50, Water=200, Road=50)")
print("=" * 60)
nd, ed = build_sim_dicts(G, cross_weight=0.5)

random.seed(42)  # Reset seed so Part 2 matches the w=0.5 run from Part 1 bit-for-bit
results = {}
for sub, origins in origins_by_sub.items():
    sizes = run_scenarios(nd, ed, origins, UNTIL_TIME)
    arr = np.array(sizes)
    runaways = [s for s in sizes if s > RUNAWAY_THRESHOLD]
    print(f"[{sub.upper()}] n={len(sizes)}")
    print(f"  Mean: {arr.mean():.1f} ({arr.mean()/N*100:.1f}%), Median: {np.median(arr):.1f} ({np.median(arr)/N*100:.1f}%)")
    print(f"  Min: {arr.min()}, Max: {arr.max()} ({arr.max()/N*100:.1f}%)")
    print(f"  Runaways (>{RUNAWAY_THRESHOLD}): {len(runaways)}/{len(sizes)} ({100*len(runaways)/len(sizes):.1f}%)")
    if runaways:
        print(f"  Runaway sizes: {sorted(runaways)}")
    print()
    results[sub] = sizes

all_sizes = np.array(results['power'] + results['water'] + results['road'])
runaways_all = [s for s in all_sizes if s > RUNAWAY_THRESHOLD]
print(f"[OVERALL] n={len(all_sizes)}")
print(f"  Mean: {all_sizes.mean():.1f} ({all_sizes.mean()/N*100:.1f}%), Median: {np.median(all_sizes):.1f} ({np.median(all_sizes)/N*100:.1f}%)")
print(f"  Runaways: {len(runaways_all)}/{len(all_sizes)} ({100*len(runaways_all)/len(all_sizes):.1f}%)")
