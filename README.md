# Game-Thoery-Projects
## 1. Evolutionary Social Dynamics - Overview of the Code Functionality
🧬 Evolutionary Dynamics in Social Systems
This project provides a simulation framework for modeling and analyzing evolutionary game dynamics in large-scale, networked social systems. It integrates models for opinion dynamics, migration patterns, and the emergence of social conventions, using realistic social networks and continuous strategy spaces.

The framework is modular, extensible, and designed to support both theoretical explorations and data-driven validation of complex adaptive systems.

🔍 Key Features
📈 Opinion Dynamics
Simulate how individual opinions evolve in a social network under the pressures of conformity, mutation, and selection using replicator dynamics.

🌍 Migration Model
Capture movement patterns of populations across locations based on adaptive migration propensities and resource availability.

🤝 Social Convention Evolution
Model the emergence and stabilization of social norms through local coordination on networks.

🧠 Evolutionary Analysis & Stability
Evaluate system convergence, strategy variance, and identify stable strategies using kernel density estimation.

🔬 Synthetic Network Generator
Build realistic hybrid networks combining scale-free and small-world properties for agent-based simulations.
## 2. 🎮 Optimal Control and Evolution in Networked Games
This repository provides a modular simulation and analysis framework for networked game theory, allowing users to model, optimize, and visualize both information diffusion and epidemic control problems across diverse network topologies. The system also includes evolutionary dynamics analysis for strategies and Nash equilibrium computation.

🚦 Main Features
Flexible Game Models:

Information diffusion and epidemic (SIR) games are implemented, controllable by each agent/node in the network.

Optimal Control Methods:

Includes Linear Quadratic Regulator (LQR), Model Predictive Control (MPC), gradient-based, genetic algorithm, and heuristic strategies.

Network Generation & Analysis:

Easily create and analyze small-world, scale-free, random, and complete networks, including centrality metrics and Laplacian matrices.

Evolutionary Strategy Dynamics:

Replicator dynamics and evolutionary stable strategy computations for multi-dimensional strategy spaces.

Nash Equilibrium Analysis:

Automated, iterative best-response procedure to identify Nash equilibria and compute efficiency loss (price of anarchy).

Impact of Network Topology:

Comparative evaluation of how network structure affects outcomes like diffusion efficiency and epidemic suppression.

Comprehensive Visualization:

Multi-panel figure outputs summarizing mean trajectories, strategy payoffs, infection spread, centrality structure, and convergence.

📦 Usage Guide
1. Initialize Parameters
Set simulation parameters such as player count, network type (small_world, scale_free, random), control constraints, and time horizon via the GameParameters dataclass.

2. Run Analysis
Call the main execution function (e.g., run_comprehensive_analysis()) to:

Generate the specified network.

Run information diffusion analysis (LQR, MPC, heuristic control).

Run epidemic control analysis (gradient, genetic, heuristic).

Analyze evolutionary stability of strategies.

Visualize all outcomes with Matplotlib.

3. Output & Insights
Prints tabulated summaries for key strategies and their payoffs (information mean/variance, infection minimization, control costs).

Nash equilibrium existence, social welfare, and price of anarchy.

Comparative results across network topologies (info diffusion efficiency, epidemic effectiveness, cost ratios).

Visualizations include:

Time-evolution of average information and infection

Bar charts of strategy performance

Node-colored network layouts (degree, final info, final infection)

Evolutionary convergence trajectories

🏗️ Modular Structure
NetworkedGame: Base class—holds network, computes Laplacian, centralities.

InformationDiffusionGame / EpidemicControlGame: Specialized game models.

GameEvolution: Evolves population strategies using replicator dynamics.

NetworkGameAnalyzer: Ties together simulation, strategy, and topology comparison logic.

Support for Extension: Add more network types, payoff functions, control mechanisms, or evolutionary models.

🔍 Example Use Cases
Design and testing of targeted information or immunization campaigns in real-world social networks.

Academic investigations of price of anarchy and network resilience.

Educational demos for game theory, control, and epidemiology classes.

📊 Empirical Validation Tools
Compare simulation outcomes with empirical (or synthetic) datasets using error metrics and correlation analysis.

📉 Visualization Dashboard
Intuitive plots to observe time evolution, distributions, and network states for all models.
