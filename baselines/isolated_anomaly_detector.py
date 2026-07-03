import networkx as nx
import numpy as np
from typing import Dict
from baselines.shared import BaselinePredictor

try:
    from scipy.stats import norm
except ImportError:
    # Fallback to a simple sigmoid if scipy is not installed
    def norm_cdf(z):
        return 1 / (1 + np.exp(-z))
else:
    norm_cdf = norm.cdf

class IsolatedAnomalyPredictor(BaselinePredictor):
    def __init__(self):
        super().__init__(name="IsolatedAnomalyDetector")
        self._cached_graph = None
        self._cached_scores = None

    def predict(self, graph: nx.Graph, origin_node: str = None) -> Dict[str, float]:
        """
        Simulates siloed IDS by severing cross-subsystem edges and computing degree Z-scores
        within each isolated subsystem. 
        Origin node is ignored by design.
        """
        if self._cached_graph is graph and self._cached_scores is not None:
            return self._cached_scores

        # 1. Sever cross-subsystem edges
        isolated_G = nx.Graph()
        isolated_G.add_nodes_from(graph.nodes(data=True))
        
        for u, v, data in graph.edges(data=True):
            sub_u = graph.nodes[u].get("subsystem")
            sub_v = graph.nodes[v].get("subsystem")
            if sub_u == sub_v:
                isolated_G.add_edge(u, v, **data)

        # 2. Compute degree Z-scores per connected component (or subsystem)
        scores = {}
        for component in nx.connected_components(isolated_G):
            comp_nodes = list(component)
            if len(comp_nodes) < 2:
                for n in comp_nodes:
                    scores[n] = 0.0
                continue
                
            degrees = [isolated_G.degree(n) for n in comp_nodes]
            mu = np.mean(degrees)
            sigma = np.std(degrees)
            
            for n, deg in zip(comp_nodes, degrees):
                if sigma > 0:
                    z = (deg - mu) / sigma
                else:
                    z = 0.0
                    
                # Convert Z-score to a 0-1 anomaly score
                # A Z-score of 2.0 maps to ~0.977, meaning highly anomalous.
                scores[n] = float(norm_cdf(z))

        self._cached_graph = graph
        self._cached_scores = scores
        return scores

if __name__ == "__main__":
    import os
    import sys
    import time
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from graph.build_topology import parse_opendss_feeder, build_networkx_graph, generate_synthetic_city
    
    print("--- Running Test for IsolatedAnomalyDetector ---")
    feeder_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "123Bus", "IEEE123Master.dss")
    nodes, edges = parse_opendss_feeder(feeder_path)
    base_power_G = build_networkx_graph(nodes, edges)
    for n in base_power_G.nodes():
        base_power_G.nodes[n]["subsystem"] = "power"
        
    city_G = generate_synthetic_city(base_power_G, density_param=0.01, cross_weight=0.30, seed=42)
    
    predictor = IsolatedAnomalyPredictor()
    
    start_time = time.time()
    predictions = predictor.predict(city_G)
    end_time = time.time()
    
    print(f"Elapsed time on 246-node graph: {end_time - start_time:.4f} seconds")
    
    anomalies = {k: v for k, v in predictions.items() if v > 0.9} # roughly Z > 1.3
    print(f"Nodes with anomaly score > 0.9: {len(anomalies)} / {len(city_G.nodes())}")
    
    top_nodes = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"Top 5 most anomalous nodes:")
    for n, p in top_nodes:
        print(f"  {n}: {p:.4f}")
