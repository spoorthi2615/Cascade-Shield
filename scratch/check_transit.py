import pickle
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "simulator"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "graph-engine"))

from ingest import build_fused_graph
G = build_fused_graph(1000)

transit_nodes = [n for n, d in G.nodes(data=True) if d.get('subsystem') == 'transit']
print("Transit nodes count:", len(transit_nodes))
transit_edges = [(u,v) for u,v,d in G.edges(data=True) if G.nodes[u].get('subsystem') == 'transit' or G.nodes[v].get('subsystem') == 'transit']
print("Transit edges count:", len(transit_edges))
