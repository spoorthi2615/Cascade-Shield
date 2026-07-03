from typing import Dict
import networkx as nx

class BaselinePredictor:
    """
    Shared evaluation contract for all baseline models.
    """
    def __init__(self, name: str):
        self.name = name

    def predict(self, graph: nx.Graph, origin_node: str = None) -> Dict[str, float]:
        """
        Takes the topology graph and optionally the attack origin (if the baseline is dynamic),
        and returns a dictionary mapping node_id -> risk_score (0.0 to 1.0).
        """
        raise NotImplementedError("Baselines must implement the predict method.")
