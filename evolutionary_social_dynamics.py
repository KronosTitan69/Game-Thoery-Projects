import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.integrate import odeint
from scipy.optimize import minimize
import pandas as pd
from sklearn.metrics import mean_squared_error
from typing import Tuple, List, Callable, Dict
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class EvolutionaryParams:
    """Parameters for evolutionary dynamics models"""
    mutation_rate: float = 0.01
    selection_strength: float = 1.0
    network_influence: float = 0.5
    time_horizon: float = 100.0
    dt: float = 0.1

class ContinuousStrategySpace:
    """Handles continuous strategy spaces for evolutionary dynamics"""
    
    def __init__(self, bounds: Tuple[float, float] = (-1.0, 1.0)):
        self.min_val, self.max_val = bounds
        self.range = self.max_val - self.min_val
    
    def normalize(self, value: float) -> float:
        """Normalize strategy to [0,1]"""
        return (value - self.min_val) / self.range
    
    def denormalize(self, value: float) -> float:
        """Convert from [0,1] back to original range"""
        return value * self.range + self.min_val
    
    def mutate(self, strategy: float, mutation_rate: float) -> float:
        """Apply mutation to strategy"""
        mutation = np.random.normal(0, mutation_rate)
        new_strategy = strategy + mutation
        return np.clip(new_strategy, self.min_val, self.max_val)

class OpinionDynamicsModel:
    """Model for opinion dynamics with continuous opinions"""
    
    def __init__(self, network: nx.Graph, strategy_space: ContinuousStrategySpace):
        self.network = network
        self.strategy_space = strategy_space
        self.n_agents = len(network.nodes())
        self.opinions = np.random.uniform(
            strategy_space.min_val, 
            strategy_space.max_val, 
            self.n_agents
        )
    
    def payoff_function(self, opinion_i: float, opinion_j: float) -> float:
        """Payoff for interaction between two opinions"""
        # Higher payoff for similar opinions (conformity pressure)
        similarity = 1.0 - abs(opinion_i - opinion_j) / self.strategy_space.range
        return similarity ** 2
    
    def fitness(self, agent_id: int) -> float:
        """Calculate fitness for an agent based on network interactions"""
        agent_opinion = self.opinions[agent_id]
        total_payoff = 0.0
        degree = self.network.degree(agent_id)
        
        if degree == 0:
            return 0.0
        
        for neighbor in self.network.neighbors(agent_id):
            neighbor_opinion = self.opinions[neighbor]
            total_payoff += self.payoff_function(agent_opinion, neighbor_opinion)
        
        return total_payoff / degree
    
    def replicator_dynamics(self, state: np.ndarray, t: float, params: EvolutionaryParams) -> np.ndarray:
        """Continuous-time replicator dynamics for opinion evolution"""
        self.opinions = state
        dsdt = np.zeros(self.n_agents)
        
        for i in range(self.n_agents):
            fitness_i = self.fitness(i)
            # Average fitness of neighbors (local reference)
            neighbor_fitness = []
            for neighbor in self.network.neighbors(i):
                neighbor_fitness.append(self.fitness(neighbor))
            
            if neighbor_fitness:
                avg_neighbor_fitness = np.mean(neighbor_fitness)
                # Opinion change based on relative fitness
                dsdt[i] = params.selection_strength * (fitness_i - avg_neighbor_fitness)
                # Add mutation/exploration
                dsdt[i] += params.mutation_rate * np.random.normal(0, 0.1)
        
        return dsdt
    
    def evolve(self, params: EvolutionaryParams) -> Tuple[np.ndarray, np.ndarray]:
        """Run evolutionary dynamics simulation"""
        t = np.arange(0, params.time_horizon, params.dt)
        trajectory = odeint(self.replicator_dynamics, self.opinions, t, args=(params,))
        return t, trajectory

class MigrationModel:
    """Model for migration patterns as evolutionary strategy"""
    
    def __init__(self, n_locations: int, strategy_space: ContinuousStrategySpace):
        self.n_locations = n_locations
        self.strategy_space = strategy_space
        # Migration propensity as continuous strategy
        self.migration_propensity = np.random.uniform(
            strategy_space.min_val, 
            strategy_space.max_val, 
            n_locations
        )
        # Resource levels at each location
        self.resources = np.random.exponential(1.0, n_locations)
        # Population at each location
        self.population = np.random.poisson(100, n_locations).astype(float)
    
    def carrying_capacity_payoff(self, location: int, propensity: float) -> float:
        """Payoff based on resource availability and population pressure"""
        resource_factor = self.resources[location]
        competition_factor = 1.0 / (1.0 + self.population[location] / 100.0)
        migration_cost = abs(propensity) * 0.1  # Cost of migration tendency
        return resource_factor * competition_factor - migration_cost
    
    def migration_flow(self, from_loc: int, to_loc: int, dt: float) -> float:
        """Calculate migration flow between locations"""
        propensity_from = self.migration_propensity[from_loc]
        payoff_diff = self.carrying_capacity_payoff(to_loc, propensity_from) - \
                     self.carrying_capacity_payoff(from_loc, propensity_from)
        
        if payoff_diff > 0 and propensity_from > 0:
            flow_rate = propensity_from * payoff_diff * self.population[from_loc]
            return min(flow_rate * dt, self.population[from_loc] * 0.1)  # Max 10% per time step
        return 0.0
    
    def evolve_migration(self, params: EvolutionaryParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate migration dynamics"""
        t = np.arange(0, params.time_horizon, params.dt)
        pop_history = np.zeros((len(t), self.n_locations))
        prop_history = np.zeros((len(t), self.n_locations))
        
        for i, time in enumerate(t):
            # Record current state
            pop_history[i] = self.population.copy()
            prop_history[i] = self.migration_propensity.copy()
            
            # Calculate migration flows
            new_population = self.population.copy()
            for from_loc in range(self.n_locations):
                for to_loc in range(self.n_locations):
                    if from_loc != to_loc:
                        flow = self.migration_flow(from_loc, to_loc, params.dt)
                        new_population[from_loc] -= flow
                        new_population[to_loc] += flow
            
            self.population = np.maximum(new_population, 0)  # Prevent negative population
            
            # Evolve migration propensities based on success
            for loc in range(self.n_locations):
                current_payoff = self.carrying_capacity_payoff(loc, self.migration_propensity[loc])
                # Mutate strategy
                new_propensity = self.strategy_space.mutate(
                    self.migration_propensity[loc], 
                    params.mutation_rate
                )
                new_payoff = self.carrying_capacity_payoff(loc, new_propensity)
                
                # Accept if better (with some probability for exploration)
                if new_payoff > current_payoff or np.random.random() < 0.1:
                    self.migration_propensity[loc] = new_propensity
        
        return t, pop_history, prop_history

class SocialConventionModel:
    """Model for emergence of social conventions"""
    
    def __init__(self, network: nx.Graph, strategy_space: ContinuousStrategySpace):
        self.network = network
        self.strategy_space = strategy_space
        self.n_agents = len(network.nodes())
        # Convention adoption level (0 = old convention, 1 = new convention)
        self.convention_level = np.random.uniform(0, 1, self.n_agents)
    
    def coordination_payoff(self, level_i: float, level_j: float) -> float:
        """Payoff for coordination between two agents"""
        # Higher payoff for coordination (similar levels)
        coordination = 1.0 - abs(level_i - level_j)
        # Network effects: more coordination = higher payoff
        return coordination ** 2
    
    def fitness(self, agent_id: int) -> float:
        """Calculate fitness based on coordination with neighbors"""
        agent_level = self.convention_level[agent_id]
        total_payoff = 0.0
        degree = self.network.degree(agent_id)
        
        if degree == 0:
            return 0.0
        
        for neighbor in self.network.neighbors(agent_id):
            neighbor_level = self.convention_level[neighbor]
            total_payoff += self.coordination_payoff(agent_level, neighbor_level)
        
        return total_payoff / degree
    
    def evolve_conventions(self, params: EvolutionaryParams) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate convention evolution"""
        t = np.arange(0, params.time_horizon, params.dt)
        level_history = np.zeros((len(t), self.n_agents))
        
        for i, time in enumerate(t):
            level_history[i] = self.convention_level.copy()
            
            # Update each agent's convention level
            for agent in range(self.n_agents):
                current_fitness = self.fitness(agent)
                
                # Try mutation
                new_level = np.clip(
                    self.convention_level[agent] + np.random.normal(0, params.mutation_rate),
                    0, 1
                )
                
                # Temporarily update to calculate new fitness
                old_level = self.convention_level[agent]
                self.convention_level[agent] = new_level
                new_fitness = self.fitness(agent)
                
                # Accept/reject based on fitness difference
                if new_fitness > current_fitness or \
                   np.random.random() < np.exp((new_fitness - current_fitness) * params.selection_strength):
                    # Accept the change
                    pass
                else:
                    # Reject the change
                    self.convention_level[agent] = old_level
        
        return t, level_history

class DataValidator:
    """Validates theoretical predictions against empirical data"""
    
    def __init__(self):
        self.empirical_data = {}
    
    def generate_synthetic_social_network(self, n_nodes: int = 500) -> nx.Graph:
        """Generate realistic social network for testing"""
        # Combination of preferential attachment and small-world properties
        G1 = nx.barabasi_albert_graph(n_nodes // 2, 3)
        G2 = nx.watts_strogatz_graph(n_nodes // 2, 6, 0.3)
        
        # Combine graphs
        G = nx.union(G1, G2, rename=("BA-", "WS-"))
        
        # Add some random edges between the two components
        ba_nodes = [n for n in G.nodes() if n.startswith("BA-")]
        ws_nodes = [n for n in G.nodes() if n.startswith("WS-")]
        
        for _ in range(20):
            ba_node = np.random.choice(ba_nodes)
            ws_node = np.random.choice(ws_nodes)
            G.add_edge(ba_node, ws_node)
        
        return G
    
    def validate_opinion_dynamics(self, model_predictions: np.ndarray, 
                                empirical_opinions: np.ndarray) -> Dict[str, float]:
        """Compare model predictions with empirical opinion data"""
        # Calculate various metrics
        mse = mean_squared_error(empirical_opinions, model_predictions)
        
        # Opinion polarization measure
        model_polarization = np.std(model_predictions)
        empirical_polarization = np.std(empirical_opinions)
        polarization_error = abs(model_polarization - empirical_polarization)
        
        # Opinion clustering (simplified)
        model_clusters = len(np.unique(np.round(model_predictions, 1)))
        empirical_clusters = len(np.unique(np.round(empirical_opinions, 1)))
        cluster_error = abs(model_clusters - empirical_clusters)
        
        return {
            'mse': mse,
            'polarization_error': polarization_error,
            'cluster_error': cluster_error,
            'correlation': np.corrcoef(model_predictions, empirical_opinions)[0, 1]
        }
    
    def evolutionary_stability_analysis(self, final_strategies: np.ndarray, 
                                      strategy_space: ContinuousStrategySpace) -> Dict[str, float]:
        """Analyze evolutionary stability of final strategy distribution"""
        # Check for evolutionary stable strategies (ESS)
        mean_strategy = np.mean(final_strategies)
        strategy_variance = np.var(final_strategies)
        
        # Stability indicators
        convergence_measure = 1.0 / (1.0 + strategy_variance)  # Higher = more convergent
        
        # Check for multiple stable points (multimodality)
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(final_strategies)
        test_points = np.linspace(strategy_space.min_val, strategy_space.max_val, 100)
        density = kde(test_points)
        
        # Count peaks (local maxima)
        peaks = []
        for i in range(1, len(density) - 1):
            if density[i] > density[i-1] and density[i] > density[i+1]:
                peaks.append(test_points[i])
        
        return {
            'mean_strategy': mean_strategy,
            'strategy_variance': strategy_variance,
            'convergence_measure': convergence_measure,
            'n_stable_points': len(peaks),
            'stable_points': peaks
        }

def run_comprehensive_analysis():
    """Run complete evolutionary dynamics analysis"""
    print("🧬 Evolutionary Dynamics in Large-Scale Social Systems")
    print("=" * 60)
    
    # Initialize parameters
    params = EvolutionaryParams(
        mutation_rate=0.02,
        selection_strength=2.0,
        network_influence=0.7,
        time_horizon=50.0,
        dt=0.1
    )
    
    strategy_space = ContinuousStrategySpace(bounds=(-2.0, 2.0))
    validator = DataValidator()
    
    # Generate social network
    print("\n📊 Generating synthetic social network...")
    network = validator.generate_synthetic_social_network(300)
    print(f"Network: {len(network.nodes())} nodes, {len(network.edges())} edges")
    print(f"Average clustering: {nx.average_clustering(network):.3f}")
    print(f"Average path length: {nx.average_shortest_path_length(network):.3f}")
    
    # 1. Opinion Dynamics Analysis
    print("\n💭 Analyzing Opinion Dynamics...")
    opinion_model = OpinionDynamicsModel(network, strategy_space)
    t_op, opinion_trajectory = opinion_model.evolve(params)
    
    # 2. Migration Model Analysis
    print("\n🌍 Analyzing Migration Patterns...")
    migration_model = MigrationModel(10, strategy_space)
    t_mig, pop_history, prop_history = migration_model.evolve_migration(params)
    
    # 3. Social Convention Analysis
    print("\n🤝 Analyzing Social Convention Emergence...")
    convention_model = SocialConventionModel(network, strategy_space)
    t_conv, convention_history = convention_model.evolve_conventions(params)
    
    # Validation and Analysis
    print("\n🔍 Validating Models...")
    
    # Generate synthetic empirical data for validation
    empirical_opinions = opinion_trajectory[-1] + np.random.normal(0, 0.1, len(opinion_trajectory[-1]))
    
    validation_results = validator.validate_opinion_dynamics(
        opinion_trajectory[-1], empirical_opinions
    )
    
    stability_analysis = validator.evolutionary_stability_analysis(
        opinion_trajectory[-1], strategy_space
    )
    
    # Visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Evolutionary Dynamics in Social Systems', fontsize=16, fontweight='bold')
    
    # Opinion dynamics
    axes[0, 0].plot(t_op, opinion_trajectory[:, :20])  # Show first 20 agents
    axes[0, 0].set_title('Opinion Dynamics Evolution')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Opinion Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Final opinion distribution
    axes[0, 1].hist(opinion_trajectory[-1], bins=30, alpha=0.7, color='skyblue')
    axes[0, 1].axvline(np.mean(opinion_trajectory[-1]), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(opinion_trajectory[-1]):.2f}')
    axes[0, 1].set_title('Final Opinion Distribution')
    axes[0, 1].set_xlabel('Opinion Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Migration patterns
    for i in range(min(5, migration_model.n_locations)):
        axes[1, 0].plot(t_mig, pop_history[:, i], label=f'Location {i+1}')
    axes[1, 0].set_title('Population Migration Dynamics')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Population')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Migration propensities
    for i in range(min(5, migration_model.n_locations)):
        axes[1, 1].plot(t_mig, prop_history[:, i], label=f'Location {i+1}')
    axes[1, 1].set_title('Migration Propensity Evolution')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Migration Propensity')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Convention dynamics
    axes[2, 0].plot(t_conv, convention_history[:, :20])  # Show first 20 agents
    axes[2, 0].set_title('Social Convention Evolution')
    axes[2, 0].set_xlabel('Time')
    axes[2, 0].set_ylabel('Convention Level')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Network structure
    pos = nx.spring_layout(network, k=0.5, iterations=50)
    node_colors = opinion_trajectory[-1][:len(network.nodes())]
    nx.draw(network, pos, ax=axes[2, 1], node_color=node_colors, 
            node_size=30, cmap='coolwarm', alpha=0.8)
    axes[2, 1].set_title('Final Network State\n(Node color = Final opinion)')
    
    plt.tight_layout()
    plt.show()
    
    # Results Summary
    print("\n📈 ANALYSIS RESULTS")
    print("=" * 40)
    print(f"Opinion Dynamics:")
    print(f"  • Final mean opinion: {np.mean(opinion_trajectory[-1]):.3f}")
    print(f"  • Opinion variance: {np.var(opinion_trajectory[-1]):.3f}")
    print(f"  • Validation MSE: {validation_results['mse']:.4f}")
    print(f"  • Model-data correlation: {validation_results.get('correlation', 0):.3f}")
    
    print(f"\nEvolutionary Stability:")
    print(f"  • Convergence measure: {stability_analysis['convergence_measure']:.3f}")
    print(f"  • Number of stable points: {stability_analysis['n_stable_points']}")
    if stability_analysis['stable_points']:
        print(f"  • Stable strategies: {[f'{x:.2f}' for x in stability_analysis['stable_points']]}")
    
    print(f"\nMigration Dynamics:")
    print(f"  • Final population distribution: {[f'{int(x)}' for x in migration_model.population]}")
    print(f"  • Population variance: {np.var(migration_model.population):.1f}")
    
    print(f"\nSocial Conventions:")
    print(f"  • Final convention adoption: {np.mean(convention_model.convention_level):.3f}")
    print(f"  • Convention uniformity: {1.0 - np.var(convention_model.convention_level):.3f}")
    
    print(f"\nNetwork Properties:")
    print(f"  • Clustering coefficient: {nx.average_clustering(network):.3f}")
    print(f"  • Average path length: {nx.average_shortest_path_length(network):.3f}")
    print(f"  • Network density: {nx.density(network):.4f}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    run_comprehensive_analysis()