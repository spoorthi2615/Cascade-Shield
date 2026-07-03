import networkx as nx
from typing import Dict
from baselines.shared import BaselinePredictor

class StaticCentralityPredictor(BaselinePredictor):
    def __init__(self):
        super().__init__(name="StaticCentrality")
        # Cache betweenness to avoid recomputing since it's static per graph
        self._cached_graph = None
        self._cached_scores = None

    def predict(self, graph: nx.Graph, origin_node: str = None) -> Dict[str, float]:
        """
        Returns normalized betweenness centrality.
        Origin node is ignored by design, demonstrating the limitation of static topology metrics.
        """
        # Simple caching mechanism (assumes graph topology doesn't mutate between calls)
        if self._cached_graph is not graph or self._cached_scores is None:
            centrality = nx.betweenness_centrality(graph, normalized=True)
            self._cached_graph = graph
            self._cached_scores = centrality
            
        return self._cached_scores
