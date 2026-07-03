import os
import sys
import torch
import glob
from torch_geometric.data import Data
from typing import Dict, List, Any
import networkx as nx

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.cascade_net import CascadeNet
from baselines.classical_seir import ClassicalSEIRPredictor

from models.train import add_distance_feature

class ModelService:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_scenarios: Dict[int, Data] = {}
        self.gnn_model = None
        self.seir_model = ClassicalSEIRPredictor(num_trials=50) # Use 50 trials for smoother UI
        
        self.load_data()
        self.load_model()
        
    def load_data(self):
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "processed")
        files = glob.glob(os.path.join(data_dir, "scenario_*.pt"))
        for f in files:
            try:
                data = torch.load(f, map_location=self.device, weights_only=False)
                # Only load TEST split
                if getattr(data, 'split', '') == 'test':
                    # Add distance feature (Phase 3)
                    add_distance_feature(data)
                    # Extract scenario ID from filename
                    basename = os.path.basename(f)
                    sid = int(basename.split('_')[1].split('.')[0])
                    self.test_scenarios[sid] = data
            except Exception as e:
                print(f"Error loading {f}: {e}")
        print(f"Loaded {len(self.test_scenarios)} test scenarios.")
        
    def load_model(self):
        # We assume one of the test scenarios has the right shape
        if not self.test_scenarios:
            raise RuntimeError("No test scenarios loaded.")
            
        sample_data = list(self.test_scenarios.values())[0]
        in_channels = sample_data.x.shape[1]
        
        # SAFETY CHECK from Phase 3
        assert in_channels == 9, f"Expected 9-dim distance-enriched features, got {in_channels}"
        
        self.gnn_model = CascadeNet(
            in_channels=in_channels,
            edge_dim=4,
            hidden_dim=64,
            num_layers=3,
            heads=4,
            use_supernode=False
        ).to(self.device)
        
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "checkpoints", "width_64_seed_0", "best_model.pth")
        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
        self.gnn_model.load_state_dict(ckpt['model_state_dict'], strict=False)
        self.gnn_model.eval()
        print("Loaded Champion GNN (3L Rank).")

    def _build_networkx(self, data: Data) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        edges = data.edge_index.t().tolist()
        G.add_edges_from(edges)
        return G

    def get_scenarios(self) -> List[Dict[str, Any]]:
        result = []
        for sid, data in self.test_scenarios.items():
            num_infected = int(data.y.sum().item())
            result.append({
                "id": sid,
                "num_nodes": data.num_nodes,
                "num_edges": data.edge_index.shape[1] // 2,
                "cascade_size": num_infected
            })
        return sorted(result, key=lambda x: x["id"])

    def get_scenario(self, sid: int) -> Dict[str, Any]:
        if sid not in self.test_scenarios:
            return None
        data = self.test_scenarios[sid]
        
        # Nodes
        nodes = []
        for i in range(data.num_nodes):
            is_origin = (data.y_time[i].item() == 0.0)
            is_infected = (data.y[i].item() == 1.0)
            # Find origin node explicitly (column 7 is is_compromised)
            is_compromised_feature = (data.x[i, 7].item() == 1.0)
            
            nodes.append({
                "id": i,
                "is_origin": is_compromised_feature,
                "is_infected_gt": is_infected
            })
            
        # Edges
        edges = []
        # Undirected graph in PyG has bidirectional edges, we only want unique pairs
        seen = set()
        for i in range(data.edge_index.shape[1]):
            u = data.edge_index[0, i].item()
            v = data.edge_index[1, i].item()
            if u < v:
                edges.append({"source": u, "target": v})
                
        return {
            "id": sid,
            "nodes": nodes,
            "links": edges
        }

    def predict(self, sid: int) -> Dict[str, Any]:
        if sid not in self.test_scenarios:
            return None
        data = self.test_scenarios[sid]
        
        # GNN Inference
        with torch.no_grad():
            logits, _ = self.gnn_model(data.x, data.edge_index, data.edge_attr)
            gnn_probs = torch.sigmoid(logits).cpu().numpy().flatten().tolist()
            
        # SEIR Inference
        nx_graph = self._build_networkx(data)
        # Find origin node (first node with y_time == 0.0)
        origin_idx = (data.y_time == 0.0).nonzero(as_tuple=True)[0]
        if len(origin_idx) > 0:
            origin_idx = origin_idx[0].item()
        else:
            origin_idx = None
            
        # Both models get the origin (Fair comparison!)
        if origin_idx is not None:
            # We map idx to string because ClassicalSEIR uses strings/ints internally?
            # Actually, `build_networkx` used integer nodes.
            seir_risk_dict = self.seir_model.predict(nx_graph, origin_idx)
            seir_probs = [0.0] * data.num_nodes
            for node_id, risk in seir_risk_dict.items():
                seir_probs[node_id] = risk
        else:
            seir_probs = [0.0] * data.num_nodes
            
        # True cascade size
        K = int(data.y.sum().item())
        
        return {
            "gnn_probs": gnn_probs,
            "seir_probs": seir_probs,
            "ground_truth": data.y.cpu().numpy().flatten().tolist(),
            "K": K
        }

# Global singleton
model_service = ModelService()
