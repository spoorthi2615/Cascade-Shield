"""
Synthesizes a heterogeneous city graph by combining IEEE test feeders
(power), SCADA reference architectures (water/traffic), and IoT datasets.
Loads the resulting topology into Neo4j.
"""

import os
import networkx as nx
import opendssdirect as dss
import random
import math
from typing import List, Tuple
from graph.schema import Node, Edge, NodeType, EdgeType

def parse_opendss_feeder(master_dss_path: str) -> Tuple[List[Node], List[Edge]]:
    """
    Loads an OpenDSS master file and extracts the bus connectivity
    into Cascade Shield schema Nodes and Edges.
    """
    if not os.path.exists(master_dss_path):
        raise FileNotFoundError(f"DSS file not found: {master_dss_path}")
        
    # Compile compiles the circuit
    # Quotes are needed around the path if there are spaces
    dss.Text.Command(f'Compile "{master_dss_path}"')
    
    nodes = []
    edges = []
    
    # Extract Buses as Nodes
    for bus_name in dss.Circuit.AllBusNames():
        base_bus = bus_name.split(".")[0].lower() # strip phase info if present
        node_id = f"bus_{base_bus}"
        
        # Prevent duplicates if phases were returned separately
        if any(n.id == node_id for n in nodes):
            continue
            
        node_type = NodeType.GRID_INTERCONNECT if base_bus == "sourcebus" else NodeType.POWER_BUS
        
        nodes.append(Node(
            id=node_id, 
            type=node_type,
            attributes={"original_name": base_bus}
        ))
        
    # Extract Lines as Edges
    for line_name in dss.Lines.AllNames():
        dss.Lines.Name(line_name)
        bus1 = dss.Lines.Bus1().split(".")[0].lower() # strip phase info
        bus2 = dss.Lines.Bus2().split(".")[0].lower()
        
        source_id = f"bus_{bus1}"
        target_id = f"bus_{bus2}"
        
        edges.append(Edge(
            source=source_id,
            target=target_id,
            type=EdgeType.PHYSICAL_DEPENDENCY,
            attributes={"length": dss.Lines.Length(), "line_name": line_name}
        ))
        
    # Transformers also connect buses
    for tx_name in dss.Transformers.AllNames():
        dss.Transformers.Name(tx_name)
        # OpenDSS direct doesn't have a simple Bus1/Bus2 for Transformers in the basic API,
        # but we can get it from the CktElement interface
        dss.Circuit.SetActiveElement(f"Transformer.{tx_name}")
        buses = dss.CktElement.BusNames()
        if len(buses) >= 2:
            bus1 = buses[0].split(".")[0].lower()
            bus2 = buses[1].split(".")[0].lower()
            edges.append(Edge(
                source=f"bus_{bus1}",
                target=f"bus_{bus2}",
                type=EdgeType.PHYSICAL_DEPENDENCY,
                attributes={"tx_name": tx_name}
            ))
            
    return nodes, edges

def build_networkx_graph(nodes: List[Node], edges: List[Edge]) -> nx.Graph:
    """
    Converts schema Nodes and Edges into a NetworkX graph for analysis.
    """
    G = nx.Graph()
    for n in nodes:
        G.add_node(n.id, type=n.type.value, **n.attributes)
    for e in edges:
        G.add_edge(e.source, e.target, type=e.type.value, **e.attributes)
    return G

def generate_synthetic_city(power_G: nx.Graph, density_param: float = 0.01, cross_weight: float = 0.30, seed: int = 42) -> nx.Graph:
    """
    Extends the base OpenDSS power graph with synthetic Water, Traffic, and Transit subgraphs,
    and hand-places cross-subsystem edges based on the density_param.
    """
    G = power_G.copy()
    rng = random.Random(seed)
    
    num_power_nodes = len(power_G.nodes())
    
    # 1. Water Network (~30% size, random geometric graph)
    num_water = max(1, int(num_power_nodes * 0.3))
    water_sub = nx.random_geometric_graph(num_water, radius=0.3, seed=seed)
    water_mapping = {n: f"water_{n}" for n in water_sub.nodes()}
    nx.relabel_nodes(water_sub, water_mapping, copy=False)
    for n in water_sub.nodes():
        G.add_node(n, type=NodeType.WATER_PUMP.value, subsystem="water")
    for u, v in water_sub.edges():
        G.add_edge(u, v, type=EdgeType.PHYSICAL_DEPENDENCY.value, weight=1.0)

    # 2. Traffic Network (~40% size, 2D grid graph)
    # find closest grid dimensions
    grid_side = max(1, int(math.sqrt(num_power_nodes * 0.4)))
    traffic_sub = nx.grid_2d_graph(grid_side, grid_side)
    traffic_mapping = {n: f"traffic_{n[0]}_{n[1]}" for n in traffic_sub.nodes()}
    nx.relabel_nodes(traffic_sub, traffic_mapping, copy=False)
    for n in traffic_sub.nodes():
        G.add_node(n, type=NodeType.TRAFFIC_CONTROLLER.value, subsystem="traffic")
    for u, v in traffic_sub.edges():
        G.add_edge(u, v, type=EdgeType.LOGICAL_DEPENDENCY.value, weight=1.0)

    # 3. Transit Network (~20% size, Watts-Strogatz small world)
    num_transit = max(4, int(num_power_nodes * 0.2)) # WS needs n > k
    transit_sub = nx.watts_strogatz_graph(n=num_transit, k=2, p=0.1, seed=seed)
    transit_mapping = {n: f"transit_{n}" for n in transit_sub.nodes()}
    nx.relabel_nodes(transit_sub, transit_mapping, copy=False)
    for n in transit_sub.nodes():
        G.add_node(n, type=NodeType.TRANSIT_NODE.value, subsystem="transit")
    for u, v in transit_sub.edges():
        G.add_edge(u, v, type=EdgeType.LOGICAL_DEPENDENCY.value, weight=1.0)

    # --- Cross-Subsystem Edge Placement ---
    # add cross-edges equal to density_param * num_power_nodes
    target_cross_edges = int(num_power_nodes * density_param)
    
    power_nodes = list(power_G.nodes())
    
    # 1. Power-to-Water (PHYSICAL): connect highest degree water nodes to random power nodes
    water_degrees = sorted(water_sub.degree, key=lambda x: x[1], reverse=True)
    for i in range(max(1, target_cross_edges // 3)):
        if i < len(water_degrees):
            w_node = water_degrees[i][0]
            p_node = rng.choice(power_nodes)
            G.add_edge(p_node, w_node, type=EdgeType.PHYSICAL_DEPENDENCY.value, weight=cross_weight)

    # 2. Power-to-Traffic (PHYSICAL)
    traffic_nodes = list(traffic_sub.nodes())
    for i in range(max(1, target_cross_edges // 3)):
        if i < len(traffic_nodes):
            t_node = rng.choice(traffic_nodes)
            p_node = rng.choice(power_nodes)
            G.add_edge(p_node, t_node, type=EdgeType.PHYSICAL_DEPENDENCY.value, weight=cross_weight)

    # 3. Transit-to-Power (LOGICAL/INFORMATIONAL)
    transit_nodes = list(transit_sub.nodes())
    for i in range(max(1, target_cross_edges // 3)):
        if i < len(transit_nodes):
            tr_node = rng.choice(transit_nodes)
            p_node = rng.choice(power_nodes)
            G.add_edge(tr_node, p_node, type=rng.choice([EdgeType.LOGICAL_DEPENDENCY.value, EdgeType.INFORMATIONAL_DEPENDENCY.value]), weight=cross_weight)
            
    return G

if __name__ == "__main__":
    # Example usage for the 13-bus feeder
    feeder_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "13Bus", "IEEE13Nodeckt.dss")
    if os.path.exists(feeder_path):
        nodes, edges = parse_opendss_feeder(feeder_path)
        G = build_networkx_graph(nodes, edges)
        print(f"Extracted {len(nodes)} buses and {len(edges)} connections from {feeder_path}")
        print(f"NetworkX graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    else:
        print(f"Skipping test, {feeder_path} not found.")
