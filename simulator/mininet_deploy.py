import os
import sys
import networkx as nx

# Conditional import so it doesn't crash if run on Windows without Mininet
try:
    from mininet.net import Mininet
    from mininet.node import OVSController, OVSKernelSwitch
    from mininet.cli import CLI
    from mininet.log import setLogLevel, info
    MININET_AVAILABLE = True
except ImportError:
    MININET_AVAILABLE = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph.build_topology import parse_opendss_feeder, build_networkx_graph

def deploy_mininet(graph: nx.Graph):
    if not MININET_AVAILABLE:
        print("Mininet is not available in this environment. Please run inside a Linux VM or WSL2 with Mininet installed.")
        sys.exit(1)

    setLogLevel('info')
    
    info('*** Creating network\n')
    net = Mininet(controller=OVSController, switch=OVSKernelSwitch)
    
    info('*** Adding controller\n')
    net.addController('c0')
    
    info('*** Adding hosts and switches\n')
    
    # We use switches to represent our nodes, since switches can easily have many links.
    # We can also attach one host per switch to represent the compute element (e.g. SCADA RTU).
    node_map = {}
    
    for i, n in enumerate(graph.nodes()):
        # Mininet switch names must be alphanumeric and short, e.g. s1, s2
        s_name = f's{i}'
        h_name = f'h{i}'
        
        switch = net.addSwitch(s_name)
        host = net.addHost(h_name, ip=f'10.0.0.{i%254 + 1}') # basic IP assignment
        
        net.addLink(host, switch)
        node_map[n] = switch
        
    info('*** Adding links\n')
    for u, v in graph.edges():
        net.addLink(node_map[u], node_map[v])
        
    info('*** Starting network\n')
    net.start()
    
    info('*** Running smoke test (pingall)\n')
    net.pingAll()
    
    info('*** Stopping network\n')
    net.stop()

if __name__ == '__main__':
    print("--- Mininet Digital-Twin Smoke Test ---")
    
    feeder_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "13Bus", "IEEE13Nodeckt.dss")
    nodes, edges = parse_opendss_feeder(feeder_path)
    G = build_networkx_graph(nodes, edges)
    
    print(f"Topology loaded: {len(G.nodes())} nodes, {len(G.edges())} edges.")
    
    if os.name == 'nt':
        print("Running on Windows. Mininet requires Linux (e.g., WSL2).")
        print("To test, run: wsl -d Ubuntu -e sudo python3 simulator/mininet_deploy.py")
    else:
        deploy_mininet(G)
