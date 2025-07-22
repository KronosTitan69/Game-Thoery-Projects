# Game-Thoery-Projects
## 1. Evolutionary Social Dynamics - Overview of the Code Functionality
This code implements a comprehensive simulation framework for evolutionary dynamics in large-scale social systems, integrating models for opinion formation, migration patterns, and the emergence of social conventions on complex networks. It combines evolutionary game theory, continuous strategy spaces, and validation routines, outputting results with both summary statistics and visual plots.

Key Components
1. Model Parameterization
EvolutionaryParams Dataclass:
Stores parameters for simulations, such as mutation rate, selection strength, network influence, simulation time, and time step size.

2. Simulation Modules
ContinuousStrategySpace:
Manages variables that represent strategies or opinions on a continuous scale (e.g., [-2, 2]). Supports normalization, mutation, and range transformations.

A. Opinion Dynamics Model
OpinionDynamicsModel:

Agents are nodes in a social network.

Each agent holds a continuous-valued opinion.

Payoff increases when opinions are similar (conformity pressure).

Update mechanism uses continuous-time replicator dynamics, allowing agent opinions to adapt by imitating higher-fitness neighbors and random mutations.

B. Migration Model
MigrationModel:

Models several locations (nodes) with resources and populations.

Each location has a 'migration propensity' determining readiness for people to move.

Agents (populations) migrate between locations based on comparative payoffs, resources, and population pressure.

Migration propensities mutate and evolve over time, enabling the study of population flows and adaptive strategies.

C. Social Convention Model
SocialConventionModel:

Simulates how social conventions (norm adoption levels between 0 and 1) spread via network coordination.

Agents update based on fitness (how well their conventions match their neighbors).

Acceptance of mutations (adopting new conventions) depends on improvement in fitness, with some randomness for exploration.

3. Network Generation
Uses combined Barabási-Albert and Watts-Strogatz models to create a realistic synthetic social network that exhibits both scale-free and small-world properties.

Adds random interconnections to integrate both network structures.

4. Validation and Stability Analysis
DataValidator:

Compares model predictions to (synthetic) empirical data, calculating metrics such as Mean Squared Error, opinion polarization, clustering errors, and correlation.

Analyzes evolutionary stability: Evaluates if the final strategy distribution is convergent, or if multiple stable outcomes (equilibria) are present.

Simulation Execution
run_comprehensive_analysis() Function:

Sets up simulation parameters.

Creates a synthetic network and initializes models for opinions, migration, and social conventions.

Runs each model, recording the evolution of agent states over time.

Validates the opinion model against generated "empirical" data and analyzes the stability of the outcome.

Plots:

Opinion dynamics for selected agents.

Distribution of final opinions.

Migration population/proclivity trends across locations.

Social convention levels.

Final network state (nodes colored by final opinion).

Outputs
After running, the code:

Displays plots showing the temporal dynamics of opinions, population migration, and social convention evolution.

Shows the structure and state of the final network visually.

Prints a detailed results summary, including:

Final average and variance of opinions, migration, and conventions.

Validation error statistics (e.g., MSE between model and empirical opinions).

Evolutionary stability metrics such as number of stable points and convergence measures.

Final population/composition of locations and uniformity of conventions.

Network-level metrics: clustering coefficient, path length, and density.
