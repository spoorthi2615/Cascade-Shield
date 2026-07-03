# Related Work: Cascade Shield

## 1. Single-subsystem ICS/SCADA IDS
*(Papers focusing on intrusion detection in isolated industrial control systems)*

- **[Hink et al., 2014] "Machine learning for power system disturbance and cyber-attack discrimination."** (DOI: 10.1109/ISRCS.2014.6900095)
  - *Summary:* Distinguishes physical faults from cyber-attacks using SVMs on relay logs.
  - *Gap relative to Cascade Shield:* Does not capture topological node dependencies, treating each relay log independently rather than as part of a connected graph.
- **[Mathur & Tippenhauer, 2016] "SWaT: A water treatment testbed for research and training on ICS security."** (DOI: 10.1109/CySWater.2016.7469060)
  - *Summary:* Introduces the SWaT testbed and discusses approaches to detecting anomalies in water treatment processes.
  - *Gap relative to Cascade Shield:* Only considers intra-PLC dependencies; assumes the attacker enters directly via the water control network, ignoring IT/OT shared credential propagation.
- **[Goh et al., 2017] "Anomaly detection in cyber physical systems using recurrent neural networks."** (DOI: 10.1109/hase.2017.36)
  - *Summary:* An unsupervised learning approach using LSTMs to detect deviations in water tank levels and pump states.
  - *Gap relative to Cascade Shield:* Single-subsystem focus; lacks a graph-theoretic foundation to rank the most critical points of failure before an attack occurs.
- **"Explainable Hybrid Intrusion Detection for SCADA/ICS: A Review and Research Agenda"** (DOI: 10.1109/BigData55660.2022.10020248)
  - *Summary:* Reviews hybrid machine learning approaches for explaining anomalies within industrial control environments.
  - *Gap Analysis:* Limits focus to singular SCADA environments without modeling multi-infrastructure interdependencies or epidemic-style failure propagation.
- **"Intrusion Detection in SCADA Networks: From Traditional Approaches to Graph Convolutional Networks"** (DOI: 10.5120/ijca2026926292)
  - *Summary:* Applies Graph Convolutional Networks (GCNs) for detecting anomalous behavior in single-subsystem SCADA network topologies.
  - *Gap Analysis:* Analyzes static SCADA topologies rather than cross-infrastructure dynamic linkages, lacking SEIR-based state progression.
- **"Real-Time Intrusion Detection of Insider Threats in Industrial Control System Workstations Through File Integrity Monitoring"** (DOI: 10.14569/IJACSA.2023.0140636)
  - *Summary:* Utilizes file integrity monitoring for real-time intrusion detection against insider threats on ICS workstations.
  - *Gap Analysis:* Focuses strictly on host-based indicators within a single ICS tier, lacking the network-level cascading failure modeling across domains.
- **"SCADA Intrusion Detection Using Deep Factorization Machines"** (DOI: 10.1038/s41598-025-20625-2)
  - *Summary:* Proposes deep factorization machines to identify complex anomaly patterns in SCADA systems.
  - *Gap Analysis:* Solves detection as a discrete classification problem in isolated grids, rather than fusing detection with cross-domain propagation dynamics.
- **"Digital Twin-Driven Intrusion Detection for Industrial SCADA: A Cyber-Physical Case Study"** (DOI: 10.3390/s24247883)
  - *Summary:* Leverages a digital twin framework for real-time anomaly detection in industrial cyber-physical SCADA systems.
  - *Gap Analysis:* Models the cyber-physical state of a single plant, unable to evaluate failures spreading to external interdependent infrastructures.
- **"A Review of Research Work on Network-Based SCADA Intrusion Detection Systems"** (DOI: 10.1109/ACCESS.2020.2994961)
  - *Summary:* Surveys network-based IDS implementations specifically designed for SCADA protocols and architectures.
  - *Gap Analysis:* Evaluates rule-based and ML detections for SCADA only, omitting the heterogeneous edge probabilities needed for multi-domain failure modeling.

## 2. Epidemiological Spread Models on Networks
*(SEIR variants applied to networks)*

- **[Pastor-Satorras & Vespignani, 2001] "Epidemic spreading in scale-free networks."** (DOI: 10.1103/PhysRevLett.86.3200)
  - *Summary:* Foundational work demonstrating that scale-free networks lack an epidemic threshold for SIS models.
  - *Gap relative to Cascade Shield:* Assumes uniform, fixed transmission rates across all edges, whereas infrastructural malware spread depends on specific edge types (e.g., shared credentials vs. physical dependency).
- **[Wang et al., 2003] "Epidemic spreading in real networks: An eigenvalue viewpoint."** (DOI: 10.1109/reldis.2003.1238052)
  - *Summary:* Links the epidemic threshold of a network to the spectral radius of its adjacency matrix.
  - *Gap relative to Cascade Shield:* Assumes a homogeneous spreading process; does not account for the heterogeneous nature of IT vs. OT propagation.
- **[Gomez et al., 2010] "Discrete-time Markov chain approach to contact-based disease spreading."** (DOI: 10.1209/0295-5075/89/38009)
  - *Summary:* Proposes a microscopic Markov chain approach for modeling SEIR states at the individual node level.
  - *Gap relative to Cascade Shield:* Edge weights are static; it does not dynamically learn transmission probabilities from a simulated dataset.
- **[De Domenico et al., 2016] "The physics of spreading processes in multilayer networks."** (DOI: 10.1038/nphys3865)
  - *Summary:* Formulates the tensor mathematics for how pathogens cross between different but coupled networks.
  - *Gap relative to Cascade Shield:* Analytical and theoretical focus; lacks a data-driven predictive model for actionable chokepoint ranking.
- **"Epidemic processes in complex networks"** (arXiv:1408.2701)
  - *Summary:* Provides a comprehensive framework for theoretical models of epidemic spreading in heterogeneous network structures.
  - *Gap Analysis:* Assumes uniform or statically distributed transmission rates instead of dynamically learned, state-dependent edge probabilities across diverse domains.
- **"Unification of theoretical approaches for epidemic spreading on complex networks"** (DOI: 10.1088/1361-6633/aa5398)
  - *Summary:* Unifies mean-field, dynamical message-passing, and link percolation approaches for analyzing epidemic diffusion.
  - *Gap Analysis:* Focuses on classical epidemic models with fixed parameters, lacking integration with GNN-based anomaly detection mechanisms for infrastructure.
- **"Exponential rate of epidemic spreading on complex networks"** (DOI: 10.1103/PhysRevE.111.044311)
  - *Summary:* Predicts early exponential epidemic spreading rates based on structural properties like degree distribution and clustering.
  - *Gap Analysis:* Evaluates unmitigated transmission topology, whereas Cascade Shield learns heterogeneous edge weights adapted from multi-modal infrastructure data.
- **"Epidemic spreading on complex networks with community structures"** (DOI: 10.1038/srep29748)
  - *Summary:* Examines how strongly connected community structures in complex networks either facilitate or restrict epidemic spread.
  - *Gap Analysis:* Treats network communities as homogeneous, whereas interdependent infrastructure requires distinct, domain-specific transition dynamics between cyber and physical nodes.

## 3. GNN-based Infrastructure & Cascading Failure
*(Applying Graph Neural Networks to critical infrastructure)*

- **[Gorka et al., 2024] "Cascading Blackout Severity Prediction with Statistically-Augmented Graph Neural Networks."** (arXiv:2403.15363)
  - *Summary:* Uses statistically-augmented GNNs to predict the severity of cascading blackouts in power grids.
  - *Gap relative to Cascade Shield:* Focuses strictly on predicting physical transmission line failure severities (power), lacking cross-domain logic for IT/OT cyber-propagation.
- **"Prediction and mitigation of nonlocal cascading failures using graph neural networks"** (DOI: 10.1063/5.0107420)
  - *Summary:* Utilizes GNNs to predict avalanche sizes and nonlocal cascading failures in large interdependent power grids.
  - *Gap Analysis:* Concentrates on physical load redistribution cascades rather than fusing malware/intrusion propagation with physical degradation via SEIR models.
- **"Physics-Informed Graph Neural Jump ODEs for Cascading Failure Prediction in Power Grids"** (arXiv:2603.14175)
  - *Summary:* Combines GNNs with Neural ODEs and Kirchhoff’s laws to model cascading failures in continuous and discrete power systems.
  - *Gap Analysis:* Strongly tailored to electrical physics within a single grid domain rather than cross-domain cyber-physical contagion.
- **"Power Failure Cascade Prediction using Graph Neural Networks"** (arXiv:2404.16104)
  - *Summary:* Develops a flow-free GNN model to predict physical grid states during failure cascades without traditional power-flow calculations.
  - *Gap Analysis:* Solves only for physical power failure cascades, lacking the capability to model cascading cyber-threat infections across varying infrastructures.
- **"Geometric deep learning for online prediction of cascading failures in power grids"** (DOI: 10.1016/j.ress.2023.109287)
  - *Summary:* Utilizes geometric deep learning for the real-time prediction of physical failure cascade progression.
  - *Gap Analysis:* Focuses strictly on predicting physical transmission line failures without incorporating an SEIR-driven cyber contagion component.
- **"Predicting Cascade Failures in Interdependent Urban Infrastructure Networks"** (arXiv:2502.15582)
  - *Summary:* Introduces a dual Graph Autoencoder approach to predict physical failure cascades across integrated urban infrastructure networks.
  - *Gap Analysis:* Models physical interdependencies well but fails to incorporate SEIR-based epidemic cyber-threat dynamics that can trigger cross-domain cascades.

## Synthesis & Gap Analysis
Most existing research falls into three distinct silos: (1) single-subsystem IDS models that ignore cross-infrastructure dependencies; (2) epidemiological network models that assume fixed, static transmission rates; and (3) GNN applications constrained to predicting physical faults or continuous metrics within isolated grids. 

Cascade Shield bridges these disciplines. Unlike classical SEIR models, Cascade Shield does not assume fixed transmission rates; instead, it uses a GNN to learn heterogeneous edge transmission probabilities dynamically based on edge types (physical, logical, informational). Unlike existing infrastructure GNNs, it evaluates cross-subsystem cascading failures, providing a predictive framework that outputs actionable, ranked chokepoints to halt cross-domain cyber-physical attacks before they compromise the broader smart city ecosystem.
