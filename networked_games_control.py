import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp, odeint
from scipy.linalg import solve_continuous_are, solve_discrete_are
import pandas as pd
from typing import Tuple, List, Callable, Dict, Optional
from dataclasses import dataclass
import seaborn as sns
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

@dataclass
class GameParameters:
    """Parameters for networked games"""
    n_players: int = 100
    network_type: str = "small_world"  # "small_world", "scale_free", "random"
    control_cost: float = 0.1
    diffusion_rate: float = 0.05
    recovery_rate: float = 0.02
    infection_rate: float = 0.03
    control_budget: float = 10.0
    time_horizon: float = 20.0
    discount_factor: float = 0.95

class NetworkedGame:
    """Base class for games played over networks"""
    
    def __init__(self, network: nx.Graph, params: GameParameters):
        self.network = network
        self.params = params
        self.n_nodes = len(network.nodes())
        self.adjacency = nx.adjacency_matrix(network).toarray()
        self.degree_matrix = np.diag([network.degree(node) for node in network.nodes()])
        self.laplacian = self.degree_matrix - self.adjacency
        
    def generate_network(self) -> nx.Graph:
        """Generate network based on type specification"""
        n = self.params.n_players
        
        if self.params.network_type == "small_world":
            return nx.watts_strogatz_graph(n, k=6, p=0.3)
        elif self.params.network_type == "scale_free":
            return nx.barabasi_albert_graph(n, m=3)
        elif self.params.network_type == "random":
            return nx.erdos_renyi_graph(n, p=0.1)
        else:
            return nx.complete_graph(n)
    
    def compute_centrality_measures(self) -> Dict[str, np.ndarray]:
        """Compute various centrality measures for strategic importance"""
        centralities = {
            'degree': np.array(list(dict(nx.degree_centrality(self.network)).values())),
            'betweenness': np.array(list(nx.betweenness_centrality(self.network).values())),
            'closeness': np.array(list(nx.closeness_centrality(self.network).values())),
            'eigenvector': np.array(list(nx.eigenvector_centrality(self.network).values()))
        }
        return centralities

class InformationDiffusionGame(NetworkedGame):
    """Game model for optimal control of information diffusion"""
    
    def __init__(self, network: nx.Graph, params: GameParameters):
        super().__init__(network, params)
        # State: information level at each node [0, 1]
        self.information_state = np.random.uniform(0, 0.1, self.n_nodes)
        # Control: information injection rate at each node
        self.control_actions = np.zeros(self.n_nodes)
        
    def diffusion_dynamics(self, t: float, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Information diffusion dynamics with control"""
        # Network diffusion effect
        diffusion_term = -self.params.diffusion_rate * self.laplacian @ state
        
        # Natural decay
        decay_term = -0.01 * state
        
        # Control input (information injection)
        control_term = control
        
        # Saturation effects (diminishing returns)
        saturation_term = -0.1 * state * (state - 0.5)
        
        return diffusion_term + decay_term + control_term + saturation_term
    
    def payoff_function(self, state: np.ndarray, control: np.ndarray, 
                       centralities: Dict[str, np.ndarray]) -> float:
        """Payoff function for information diffusion game"""
        # Benefit from information spread (weighted by centrality)
        info_benefit = np.sum(centralities['degree'] * state)
        
        # Cost of control actions
        control_cost = self.params.control_cost * np.sum(control**2)
        
        # Bonus for uniform spread
        uniformity_bonus = -0.1 * np.var(state)
        
        return info_benefit - control_cost + uniformity_bonus
    
    def optimal_control_lqr(self, target_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Linear Quadratic Regulator for information diffusion control"""
        # Linearize around current state
        A = -self.params.diffusion_rate * self.laplacian - 0.01 * np.eye(self.n_nodes)
        B = np.eye(self.n_nodes)  # Direct control input
        
        # Cost matrices
        Q = np.eye(self.n_nodes)  # State cost
        R = self.params.control_cost * np.eye(self.n_nodes)  # Control cost
        
        # Solve continuous-time algebraic Riccati equation
        try:
            P = solve_continuous_are(A, B, Q, R)
            # Optimal control gain
            K = np.linalg.inv(R) @ B.T @ P
            
            # Optimal control policy
            optimal_control = -K @ (self.information_state - target_state)
            
            return optimal_control, K
        except:
            # Fallback to simple proportional control
            return -0.1 * (self.information_state - target_state), np.eye(self.n_nodes)
    
    def mpc_control(self, prediction_horizon: int = 5) -> np.ndarray:
        """Model Predictive Control for information diffusion"""
        def objective(u_sequence):
            u_sequence = u_sequence.reshape(prediction_horizon, self.n_nodes)
            state = self.information_state.copy()
            total_cost = 0.0
            dt = 0.1
            
            centralities = self.compute_centrality_measures()
            
            for k in range(prediction_horizon):
                control = u_sequence[k]
                
                # Simulate one step forward
                dstate_dt = self.diffusion_dynamics(k * dt, state, control)
                state = state + dt * dstate_dt
                state = np.clip(state, 0, 1)  # Keep in valid range
                
                # Accumulate cost (negative payoff)
                cost = -self.payoff_function(state, control, centralities)
                total_cost += cost * (self.params.discount_factor ** k)
            
            return total_cost
        
        # Constraints
        bounds = [(-1.0, 1.0)] * (prediction_horizon * self.n_nodes)
        
        # Initial guess
        u0 = np.zeros(prediction_horizon * self.n_nodes)
        
        # Optimize
        result = minimize(objective, u0, method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            optimal_sequence = result.x.reshape(prediction_horizon, self.n_nodes)
            return optimal_sequence[0]  # Return first control action
        else:
            return np.zeros(self.n_nodes)

class EpidemicControlGame(NetworkedGame):
    """Game model for optimal epidemic control strategies"""
    
    def __init__(self, network: nx.Graph, params: GameParameters):
        super().__init__(network, params)
        # SIR model states: [Susceptible, Infected, Recovered] for each node
        self.S = np.ones(self.n_nodes) * 0.9  # Susceptible
        self.I = np.ones(self.n_nodes) * 0.1  # Infected
        self.R = np.zeros(self.n_nodes)       # Recovered
        
        # Control actions: intervention strength at each node
        self.control_actions = np.zeros(self.n_nodes)
    
    def sir_dynamics(self, t: float, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """SIR epidemic dynamics with network effects and control"""
        n = self.n_nodes
        S, I, R = state[:n], state[n:2*n], state[2*n:]
        
        # Network transmission effects
        infection_pressure = np.zeros(n)
        for i in range(n):
            neighbors = list(self.network.neighbors(i))
            if neighbors:
                infection_pressure[i] = np.mean(I[neighbors])
        
        # Control reduces transmission (isolation, vaccination, etc.)
        effective_infection_rate = self.params.infection_rate * (1 - control)
        
        # SIR equations with network effects
        dS_dt = -effective_infection_rate * S * infection_pressure
        dI_dt = effective_infection_rate * S * infection_pressure - self.params.recovery_rate * I
        dR_dt = self.params.recovery_rate * I
        
        return np.concatenate([dS_dt, dI_dt, dR_dt])
    
    def epidemic_payoff(self, S: np.ndarray, I: np.ndarray, R: np.ndarray, 
                       control: np.ndarray, centralities: Dict[str, np.ndarray]) -> float:
        """Payoff function for epidemic control game"""
        # Cost of infections (weighted by centrality - important nodes matter more)
        infection_cost = np.sum(centralities['degree'] * I) * 100
        
        # Cost of control measures
        control_cost = self.params.control_cost * np.sum(control**2)
        
        # Bonus for protecting high-centrality nodes
        protection_bonus = np.sum(centralities['betweenness'] * S) * 10
        
        return -infection_cost - control_cost + protection_bonus
    
    def optimal_epidemic_control(self, method: str = "gradient") -> np.ndarray:
        """Find optimal epidemic control strategy"""
        centralities = self.compute_centrality_measures()
        
        if method == "gradient":
            return self._gradient_based_control(centralities)
        elif method == "genetic":
            return self._genetic_algorithm_control(centralities)
        else:
            return self._heuristic_control(centralities)
    
    def _gradient_based_control(self, centralities: Dict[str, np.ndarray]) -> np.ndarray:
        """Gradient-based optimization for epidemic control"""
        def objective(control):
            # Simulate epidemic with given control
            state0 = np.concatenate([self.S, self.I, self.R])
            
            def dynamics_wrapper(t, y):
                return self.sir_dynamics(t, y, control)
            
            t_span = (0, 5.0)  # Short prediction horizon
            t_eval = np.linspace(0, 5.0, 50)
            
            sol = solve_ivp(dynamics_wrapper, t_span, state0, t_eval=t_eval, 
                          method='RK45', rtol=1e-6)
            
            if sol.success:
                final_state = sol.y[:, -1]
                n = self.n_nodes
                final_S = final_state[:n]
                final_I = final_state[n:2*n]
                final_R = final_state[2*n:]
                
                return -self.epidemic_payoff(final_S, final_I, final_R, control, centralities)
            else:
                return 1e6  # Large penalty for failed simulation
        
        # Constraints: control between 0 and 1
        bounds = [(0, 1)] * self.n_nodes
        
        # Budget constraint
        def budget_constraint(control):
            return self.params.control_budget - np.sum(control)
        
        constraints = {'type': 'ineq', 'fun': budget_constraint}
        
        # Initial guess: proportional to centrality
        u0 = centralities['degree'] * 0.1
        
        result = minimize(objective, u0, method='SLSQP', bounds=bounds, 
                         constraints=constraints)
        
        return result.x if result.success else u0
    
    def _genetic_algorithm_control(self, centralities: Dict[str, np.ndarray]) -> np.ndarray:
        """Genetic algorithm for epidemic control optimization"""
        def objective(control):
            state0 = np.concatenate([self.S, self.I, self.R])
            
            def dynamics_wrapper(t, y):
                return self.sir_dynamics(t, y, control)
            
            t_span = (0, 3.0)
            sol = solve_ivp(dynamics_wrapper, t_span, state0, method='RK45')
            
            if sol.success:
                final_state = sol.y[:, -1]
                n = self.n_nodes
                final_S = final_state[:n]
                final_I = final_state[n:2*n]
                final_R = final_state[2*n:]
                
                payoff = self.epidemic_payoff(final_S, final_I, final_R, control, centralities)
                
                # Add budget penalty
                budget_violation = max(0, np.sum(control) - self.params.control_budget)
                payoff -= budget_violation * 1000
                
                return -payoff  # Minimize negative payoff
            else:
                return 1e6
        
        bounds = [(0, 1)] * self.n_nodes
        
        result = differential_evolution(objective, bounds, seed=42, maxiter=100)
        return result.x if result.success else np.zeros(self.n_nodes)
    
    def _heuristic_control(self, centralities: Dict[str, np.ndarray]) -> np.ndarray:
        """Heuristic control based on centrality and infection status"""
        # Prioritize high-centrality nodes with high infection risk
        risk_score = centralities['betweenness'] * self.I + centralities['degree'] * 0.1
        
        # Allocate control budget proportionally
        if np.sum(risk_score) > 0:
            control = self.params.control_budget * risk_score / np.sum(risk_score)
        else:
            control = np.ones(self.n_nodes) * self.params.control_budget / self.n_nodes
        
        return np.clip(control, 0, 1)

class GameEvolution:
    """Analyze evolutionary dynamics of strategies in networked games"""
    
    def __init__(self, game: NetworkedGame):
        self.game = game
        # Strategy space: each player chooses control allocation weights
        self.strategies = np.random.uniform(0, 1, (game.n_nodes, 3))  # 3D strategy space
        self.normalize_strategies()
    
    def normalize_strategies(self):
        """Ensure strategies are valid probability distributions"""
        for i in range(self.game.n_nodes):
            self.strategies[i] = self.strategies[i] / np.sum(self.strategies[i])
    
    def strategy_payoff(self, player: int, strategy: np.ndarray) -> float:
        """Calculate payoff for a player using a given strategy"""
        # Strategy determines how player allocates control effort
        centralities = self.game.compute_centrality_measures()
        
        # Control allocation based on strategy
        control_weights = strategy
        base_control = 0.1  # Base control level
        
        if isinstance(self.game, InformationDiffusionGame):
            control = base_control * control_weights[0] * centralities['degree'][player]
            state = self.game.information_state.copy()
            state[player] += control * 0.1  # Small perturbation to see effect
            return self.game.payoff_function(state, np.zeros(self.game.n_nodes), centralities)
        
        elif isinstance(self.game, EpidemicControlGame):
            control = base_control * control_weights[0] * centralities['degree'][player]
            # Simplified payoff calculation
            infection_reduction = control * self.game.I[player]
            control_cost = self.game.params.control_cost * control**2
            return infection_reduction * 10 - control_cost
        
        return 0.0
    
    def replicator_dynamics(self, dt: float = 0.01) -> np.ndarray:
        """Update strategies using replicator dynamics"""
        fitness = np.zeros((self.game.n_nodes, 3))
        
        # Calculate fitness for each strategy component
        for i in range(self.game.n_nodes):
            for j in range(3):
                test_strategy = np.zeros(3)
                test_strategy[j] = 1.0  # Pure strategy
                fitness[i, j] = self.strategy_payoff(i, test_strategy)
        
        # Average fitness for each player
        avg_fitness = np.sum(self.strategies * fitness, axis=1, keepdims=True)
        
        # Replicator equation
        dstrat_dt = self.strategies * (fitness - avg_fitness)
        
        # Update strategies
        self.strategies += dt * dstrat_dt
        self.strategies = np.clip(self.strategies, 0.001, 0.999)  # Prevent extinction
        self.normalize_strategies()
        
        return np.mean(np.abs(dstrat_dt))  # Return convergence measure
    
    def evolutionary_stable_strategy(self, iterations: int = 1000) -> Tuple[np.ndarray, List[float]]:
        """Find evolutionarily stable strategies"""
        convergence_history = []
        
        for _ in range(iterations):
            convergence_measure = self.replicator_dynamics()
            convergence_history.append(convergence_measure)
            
            if convergence_measure < 1e-6:
                break
        
        return self.strategies.copy(), convergence_history

class NetworkGameAnalyzer:
    """Comprehensive analyzer for networked games"""
    
    def __init__(self):
        self.results = {}
    
    def analyze_information_diffusion(self, params: GameParameters) -> Dict:
        """Comprehensive analysis of information diffusion game"""
        # Create network
        network_generator = NetworkedGame(nx.Graph(), params)
        network = network_generator.generate_network()
        
        # Initialize game
        game = InformationDiffusionGame(network, params)
        
        # Target: uniform high information level
        target_state = np.ones(game.n_nodes) * 0.8
        
        # Compare different control strategies
        strategies = {}
        
        # LQR Control
        optimal_control_lqr, gain_matrix = game.optimal_control_lqr(target_state)
        strategies['LQR'] = optimal_control_lqr
        
        # MPC Control
        optimal_control_mpc = game.mpc_control()
        strategies['MPC'] = optimal_control_mpc
        
        # Centrality-based heuristic
        centralities = game.compute_centrality_measures()
        heuristic_control = 0.1 * centralities['degree']
        strategies['Heuristic'] = heuristic_control
        
        # Simulate each strategy
        results = {}
        for name, control in strategies.items():
            game.control_actions = control
            
            # Simulate dynamics
            t_span = (0, params.time_horizon)
            t_eval = np.linspace(0, params.time_horizon, 100)
            
            def dynamics(t, y):
                return game.diffusion_dynamics(t, y, control)
            
            sol = solve_ivp(dynamics, t_span, game.information_state, 
                          t_eval=t_eval, method='RK45')
            
            if sol.success:
                final_state = sol.y[:, -1]
                total_payoff = game.payoff_function(final_state, control, centralities)
                
                results[name] = {
                    'trajectory': sol.y,
                    'time': sol.t,
                    'final_state': final_state,
                    'total_payoff': total_payoff,
                    'control': control,
                    'mse_to_target': mean_squared_error(final_state, target_state)
                }
        
        return {
            'network': network,
            'game': game,
            'strategies': results,
            'centralities': centralities,
            'target_state': target_state
        }
    
    def analyze_epidemic_control(self, params: GameParameters) -> Dict:
        """Comprehensive analysis of epidemic control game"""
        # Create network
        network_generator = NetworkedGame(nx.Graph(), params)
        network = network_generator.generate_network()
        
        # Initialize epidemic game
        game = EpidemicControlGame(network, params)
        
        # Set initial epidemic state
        # Start with few infected nodes (preferentially high-degree ones)
        centralities = game.compute_centrality_measures()
        high_degree_nodes = np.argsort(centralities['degree'])[-5:]  # Top 5 degree nodes
        game.I[:] = 0.01  # Low background infection
        game.I[high_degree_nodes] = 0.5  # High infection in central nodes
        game.S = 1 - game.I - game.R
        
        # Compare different control strategies
        control_methods = ['gradient', 'genetic', 'heuristic']
        results = {}
        
        for method in control_methods:
            game_copy = EpidemicControlGame(network, params)
            game_copy.S = game.S.copy()
            game_copy.I = game.I.copy()
            game_copy.R = game.R.copy()
            
            # Find optimal control
            optimal_control = game_copy.optimal_epidemic_control(method)
            
            # Simulate epidemic with control
            state0 = np.concatenate([game_copy.S, game_copy.I, game_copy.R])
            
            def dynamics(t, y):
                return game_copy.sir_dynamics(t, y, optimal_control)
            
            t_span = (0, params.time_horizon)
            t_eval = np.linspace(0, params.time_horizon, 100)
            
            sol = solve_ivp(dynamics, t_span, state0, t_eval=t_eval, method='RK45')
            
            if sol.success:
                n = game.n_nodes
                final_S = sol.y[:n, -1]
                final_I = sol.y[n:2*n, -1]
                final_R = sol.y[2*n:, -1]
                
                total_payoff = game_copy.epidemic_payoff(final_S, final_I, final_R, 
                                                       optimal_control, centralities)
                
                results[method] = {
                    'trajectory': sol.y,
                    'time': sol.t,
                    'final_S': final_S,
                    'final_I': final_I,
                    'final_R': final_R,
                    'total_payoff': total_payoff,
                    'control': optimal_control,
                    'peak_infection': np.max(np.sum(sol.y[n:2*n, :], axis=0)),
                    'total_infected': np.sum(final_R)  # Recovered = total infected
                }
        
        return {
            'network': network,
            'game': game,
            'strategies': results,
            'centralities': centralities
        }
    
    def evolutionary_analysis(self, game: NetworkedGame, iterations: int = 500) -> Dict:
        """Analyze evolutionary dynamics of strategies"""
        evolution = GameEvolution(game)
        final_strategies, convergence = evolution.evolutionary_stable_strategy(iterations)
        
        return {
            'final_strategies': final_strategies,
            'convergence_history': convergence,
            'strategy_diversity': np.mean(np.std(final_strategies, axis=0)),
            'convergence_rate': len([c for c in convergence if c > 1e-4])
        }

def run_comprehensive_analysis():
    """Run complete networked games analysis"""
    print("🎮 Optimal Control and Evolution in Networked Games")
    print("=" * 60)
    
    # Initialize parameters
    params = GameParameters(
        n_players=50,
        network_type="small_world",
        control_cost=0.05,
        control_budget=5.0,
        time_horizon=15.0
    )
    
    analyzer = NetworkGameAnalyzer()
    
    # 1. Information Diffusion Analysis
    print("\n📡 Analyzing Information Diffusion Game...")
    info_results = analyzer.analyze_information_diffusion(params)
    
    print("Information Diffusion Results:")
    for strategy, result in info_results['strategies'].items():
        print(f"  {strategy}:")
        print(f"    Total Payoff: {result['total_payoff']:.4f}")
        print(f"    MSE to Target: {result['mse_to_target']:.4f}")
        print(f"    Control Effort: {np.sum(result['control']**2):.4f}")
    
    # 2. Epidemic Control Analysis
    print("\n🦠 Analyzing Epidemic Control Game...")
    epidemic_results = analyzer.analyze_epidemic_control(params)
    
    print("Epidemic Control Results:")
    for strategy, result in epidemic_results['strategies'].items():
        print(f"  {strategy}:")
        print(f"    Total Payoff: {result['total_payoff']:.4f}")
        print(f"    Peak Infection: {result['peak_infection']:.4f}")
        print(f"    Total Infected: {result['total_infected']:.4f}")
        print(f"    Control Budget Used: {np.sum(result['control']):.4f}")
    
    # 3. Evolutionary Analysis
    print("\n🧬 Analyzing Strategy Evolution...")
    info_evolution = analyzer.evolutionary_analysis(info_results['game'])
    epidemic_evolution = analyzer.evolutionary_analysis(epidemic_results['game'])
    
    print("Evolutionary Dynamics:")
    print(f"  Information Game - Strategy Diversity: {info_evolution['strategy_diversity']:.4f}")
    print(f"  Epidemic Game - Strategy Diversity: {epidemic_evolution['strategy_diversity']:.4f}")
    
    # Visualization
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Information Diffusion Results
    ax1 = fig.add_subplot(gs[0, :2])
    for strategy, result in info_results['strategies'].items():
        mean_info = np.mean(result['trajectory'], axis=0)
        ax1.plot(result['time'], mean_info, label=f"{strategy}", linewidth=2)
    ax1.axhline(y=np.mean(info_results['target_state']), color='red', 
               linestyle='--', label='Target Level')
    ax1.set_title('Information Diffusion: Average Information Level')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Information Level')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Control strategies comparison
    ax2 = fig.add_subplot(gs[0, 2:])
    strategies = list(info_results['strategies'].keys())
    payoffs = [info_results['strategies'][s]['total_payoff'] for s in strategies]
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    bars = ax2.bar(strategies, payoffs, color=colors)
    ax2.set_title('Information Diffusion: Strategy Comparison')
    ax2.set_ylabel('Total Payoff')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Epidemic Control Results
    ax3 = fig.add_subplot(gs[1, :2])
    for strategy, result in epidemic_results['strategies'].items():
        n = epidemic_results['game'].n_nodes
        total_infected = np.sum(result['trajectory'][n:2*n, :], axis=0)
        ax3.plot(result['time'], total_infected, label=f"{strategy}", linewidth=2)
    ax3.set_title('Epidemic Control: Total Infected Population')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Number of Infected')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Epidemic control comparison
    ax4 = fig.add_subplot(gs[1, 2:])
    strategies = list(epidemic_results['strategies'].keys())
    peak_infections = [epidemic_results['strategies'][s]['peak_infection'] for s in strategies]
    bars = ax4.bar(strategies, peak_infections, color=['orange', 'purple', 'brown'])
    ax4.set_title('Epidemic Control: Peak Infection Comparison')
    ax4.set_ylabel('Peak Infections')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Network visualizations
    ax5 = fig.add_subplot(gs[2, 0])
    network = info_results['network']
    pos = nx.spring_layout(network, k=0.5, iterations=50)
    node_colors = info_results['centralities']['degree']
    nx.draw(network, pos, ax=ax5, node_color=node_colors, node_size=50, 
           cmap='viridis', alpha=0.8)
    ax5.set_title('Network: Degree Centrality')
    
    ax6 = fig.add_subplot(gs[2, 1])
    # Show final information state for best strategy
    best_info_strategy = max(info_results['strategies'].keys(), 
                           key=lambda x: info_results['strategies'][x]['total_payoff'])
    final_info = info_results['strategies'][best_info_strategy]['final_state']
    nx.draw(network, pos, ax=ax6, node_color=final_info, node_size=50, 
           cmap='coolwarm', alpha=0.8, vmin=0, vmax=1)
    ax6.set_title(f'Final Information State ({best_info_strategy})')
    
    # Epidemic network state
    ax7 = fig.add_subplot(gs[2, 2])
    best_epidemic_strategy = min(epidemic_results['strategies'].keys(), 
                               key=lambda x: epidemic_results['strategies'][x]['peak_infection'])
    final_infected = epidemic_results['strategies'][best_epidemic_strategy]['final_I']
    nx.draw(network, pos, ax=ax7, node_color=final_infected, node_size=50, 
           cmap='Reds', alpha=0.8, vmin=0, vmax=1)
    ax7.set_title(f'Final Infection State ({best_epidemic_strategy})')
    
    # Strategy evolution
    ax8 = fig.add_subplot(gs[2, 3])
    ax8.plot(info_evolution['convergence_history'], label='Info Game', linewidth=2)
    ax8.plot(epidemic_evolution['convergence_history'], label='Epidemic Game', linewidth=2)
    ax8.set_title('Strategy Evolution Convergence')
    ax8.set_xlabel('Iteration')
    ax8.set_ylabel('Convergence Measure')
    ax8.set_yscale('log')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    plt.suptitle('Optimal Control and Evolution in Networked Games', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Additional Analysis: Nash Equilibria
    print("\n⚖️ Nash Equilibrium Analysis...")
    
    # For information diffusion game
    info_nash = analyze_nash_equilibrium(info_results['game'], info_results['centralities'])
    print("Information Diffusion Nash Equilibrium:")
    print(f"  Equilibrium exists: {info_nash['exists']}")
    print(f"  Social welfare: {info_nash['social_welfare']:.4f}")
    print(f"  Price of anarchy: {info_nash['price_of_anarchy']:.4f}")
    
    # For epidemic control game
    epidemic_nash = analyze_nash_equilibrium(epidemic_results['game'], epidemic_results['centralities'])
    print("Epidemic Control Nash Equilibrium:")
    print(f"  Equilibrium exists: {epidemic_nash['exists']}")
    print(f"  Social welfare: {epidemic_nash['social_welfare']:.4f}")
    print(f"  Price of anarchy: {epidemic_nash['price_of_anarchy']:.4f}")
    
    # Network topology analysis
    print("\n🌐 Network Topology Impact...")
    topology_analysis = analyze_topology_impact(params)
    
    for topology, results in topology_analysis.items():
        print(f"  {topology.title()} Network:")
        print(f"    Info diffusion efficiency: {results['info_efficiency']:.4f}")
        print(f"    Epidemic control effectiveness: {results['epidemic_effectiveness']:.4f}")
        print(f"    Control cost ratio: {results['cost_ratio']:.4f}")
    
    print("\n📊 SUMMARY INSIGHTS")
    print("=" * 40)
    print(f"• Best information diffusion strategy: {best_info_strategy}")
    print(f"• Best epidemic control strategy: {best_epidemic_strategy}")
    print(f"• Information game converged in {info_evolution['convergence_rate']} iterations")
    print(f"• Epidemic game converged in {epidemic_evolution['convergence_rate']} iterations")
    print(f"• Network density: {nx.density(network):.4f}")
    print(f"• Average clustering: {nx.average_clustering(network):.4f}")
    
    return {
        'info_results': info_results,
        'epidemic_results': epidemic_results,
        'info_evolution': info_evolution,
        'epidemic_evolution': epidemic_evolution,
        'topology_analysis': topology_analysis
    }

def analyze_nash_equilibrium(game: NetworkedGame, centralities: Dict[str, np.ndarray]) -> Dict:
    """Analyze Nash equilibrium properties"""
    n_players = game.n_nodes
    
    # Simplified Nash equilibrium analysis
    # Each player optimizes their control given others' strategies
    
    def best_response(player_id: int, others_strategies: np.ndarray) -> float:
        """Find best response for a player given others' strategies"""
        def player_payoff(control):
            if isinstance(game, InformationDiffusionGame):
                # Simulate with player's control and others' fixed strategies
                total_control = others_strategies.copy()
                total_control[player_id] = control
                
                state = game.information_state.copy()
                # Simple forward simulation
                for _ in range(10):
                    dstate = game.diffusion_dynamics(0, state, total_control)
                    state += 0.1 * dstate
                    state = np.clip(state, 0, 1)
                
                return game.payoff_function(state, total_control, centralities)
            
            elif isinstance(game, EpidemicControlGame):
                # For epidemic game
                infection_reduction = control * game.I[player_id]
                control_cost = game.params.control_cost * control**2
                centrality_weight = centralities['degree'][player_id]
                return (infection_reduction * centrality_weight * 10 - control_cost)
        
        # Optimize player's control
        result = minimize(lambda x: -player_payoff(x[0]), [0.1], 
                         bounds=[(0, 1)], method='L-BFGS-B')
        return result.x[0] if result.success else 0.1
    
    # Iterative best response to find Nash equilibrium
    strategies = np.random.uniform(0.1, 0.3, n_players)  # Initial strategies
    
    for iteration in range(50):  # Max iterations
        new_strategies = strategies.copy()
        
        for player in range(n_players):
            new_strategies[player] = best_response(player, strategies)
        
        # Check convergence
        if np.allclose(strategies, new_strategies, rtol=1e-4):
            nash_equilibrium = new_strategies
            equilibrium_exists = True
            break
        
        strategies = 0.7 * strategies + 0.3 * new_strategies  # Damped update
    else:
        nash_equilibrium = strategies
        equilibrium_exists = False
    
    # Calculate social welfare at Nash equilibrium
    if isinstance(game, InformationDiffusionGame):
        state = game.information_state.copy()
        for _ in range(10):
            dstate = game.diffusion_dynamics(0, state, nash_equilibrium)
            state += 0.1 * dstate
            state = np.clip(state, 0, 1)
        
        nash_welfare = game.payoff_function(state, nash_equilibrium, centralities)
        
        # Optimal social welfare (centralized control)
        optimal_control, _ = game.optimal_control_lqr(np.ones(n_players) * 0.8)
        optimal_control = np.clip(optimal_control, 0, 1)
        
        state_opt = game.information_state.copy()
        for _ in range(10):
            dstate = game.diffusion_dynamics(0, state_opt, optimal_control)
            state_opt += 0.1 * dstate
            state_opt = np.clip(state_opt, 0, 1)
        
        optimal_welfare = game.payoff_function(state_opt, optimal_control, centralities)
        
    else:  # Epidemic game
        nash_welfare = np.sum([
            (nash_equilibrium[i] * game.I[i] * centralities['degree'][i] * 10 - 
             game.params.control_cost * nash_equilibrium[i]**2)
            for i in range(n_players)
        ])
        
        # Optimal welfare (simplified)
        optimal_control = game.optimal_epidemic_control('heuristic')
        optimal_welfare = np.sum([
            (optimal_control[i] * game.I[i] * centralities['degree'][i] * 10 - 
             game.params.control_cost * optimal_control[i]**2)
            for i in range(n_players)
        ])
    
    price_of_anarchy = optimal_welfare / nash_welfare if nash_welfare > 0 else float('inf')
    
    return {
        'exists': equilibrium_exists,
        'strategies': nash_equilibrium,
        'social_welfare': nash_welfare,
        'optimal_welfare': optimal_welfare,
        'price_of_anarchy': price_of_anarchy
    }

def analyze_topology_impact(base_params: GameParameters) -> Dict:
    """Analyze how network topology affects game outcomes"""
    topologies = ['small_world', 'scale_free', 'random']
    results = {}
    
    for topology in topologies:
        params = GameParameters(
            n_players=base_params.n_players,
            network_type=topology,
            control_cost=base_params.control_cost,
            control_budget=base_params.control_budget,
            time_horizon=10.0  # Shorter for efficiency
        )
        
        # Create network
        network_gen = NetworkedGame(nx.Graph(), params)
        network = network_gen.generate_network()
        
        # Information diffusion analysis
        info_game = InformationDiffusionGame(network, params)
        target_state = np.ones(info_game.n_nodes) * 0.8
        optimal_control, _ = info_game.optimal_control_lqr(target_state)
        
        # Simulate
        def dynamics(t, y):
            return info_game.diffusion_dynamics(t, y, optimal_control)
        
        sol = solve_ivp(dynamics, (0, 5), info_game.information_state, method='RK45')
        
        if sol.success:
            final_info = sol.y[:, -1]
            info_efficiency = np.mean(final_info) / np.mean(target_state)
        else:
            info_efficiency = 0.0
        
        # Epidemic control analysis
        epidemic_game = EpidemicControlGame(network, params)
        centralities = epidemic_game.compute_centrality_measures()
        
        # Set initial epidemic state
        high_degree_nodes = np.argsort(centralities['degree'])[-3:]
        epidemic_game.I[:] = 0.01
        epidemic_game.I[high_degree_nodes] = 0.3
        epidemic_game.S = 1 - epidemic_game.I - epidemic_game.R
        
        optimal_epidemic_control = epidemic_game.optimal_epidemic_control('heuristic')
        
        # Simulate epidemic
        state0 = np.concatenate([epidemic_game.S, epidemic_game.I, epidemic_game.R])
        
        def epidemic_dynamics(t, y):
            return epidemic_game.sir_dynamics(t, y, optimal_epidemic_control)
        
        sol_epidemic = solve_ivp(epidemic_dynamics, (0, 5), state0, method='RK45')
        
        if sol_epidemic.success:
            n = epidemic_game.n_nodes
            peak_infection = np.max(np.sum(sol_epidemic.y[n:2*n, :], axis=0))
            epidemic_effectiveness = 1.0 / (1.0 + peak_infection)  # Higher is better
        else:
            epidemic_effectiveness = 0.0
        
        # Control cost analysis
        info_cost = np.sum(optimal_control**2)
        epidemic_cost = np.sum(optimal_epidemic_control**2)
        total_cost = info_cost + epidemic_cost
        
        # Network properties
        clustering = nx.average_clustering(network)
        avg_path_length = nx.average_shortest_path_length(network)
        
        results[topology] = {
            'info_efficiency': info_efficiency,
            'epidemic_effectiveness': epidemic_effectiveness,
            'cost_ratio': total_cost / params.control_budget,
            'clustering': clustering,
            'avg_path_length': avg_path_length,
            'network_density': nx.density(network)
        }
    
    return results

class AdvancedControlStrategies:
    """Advanced control strategies for networked games"""
    
    def __init__(self, game: NetworkedGame):
        self.game = game
    
    def distributed_consensus_control(self, target_state: np.ndarray) -> np.ndarray:
        """Distributed consensus-based control"""
        # Each node tries to reach consensus while minimizing control effort
        laplacian = self.game.laplacian
        
        # Consensus control law: u = -k * L * (x - target)
        consensus_gain = 0.5
        
        if isinstance(self.game, InformationDiffusionGame):
            state_error = self.game.information_state - target_state
        else:
            state_error = self.game.I - target_state  # For epidemic game
        
        control = -consensus_gain * laplacian @ state_error
        return np.clip(control, -1, 1)
    
    def adaptive_learning_control(self, learning_rate: float = 0.01) -> np.ndarray:
        """Adaptive control that learns optimal strategies"""
        # Simple gradient-based learning
        centralities = self.game.compute_centrality_measures()
        
        # Initialize control based on centrality
        if not hasattr(self, 'learned_control'):
            self.learned_control = 0.1 * centralities['degree']
            self.performance_history = []
        
        # Evaluate current performance
        if isinstance(self.game, InformationDiffusionGame):
            current_performance = self.game.payoff_function(
                self.game.information_state, self.learned_control, centralities
            )
        else:
            current_performance = self.game.epidemic_payoff(
                self.game.S, self.game.I, self.game.R, self.learned_control, centralities
            )
        
        self.performance_history.append(current_performance)
        
        # Gradient estimation (finite difference)
        gradient_estimate = np.zeros_like(self.learned_control)
        epsilon = 0.01
        
        for i in range(len(self.learned_control)):
            # Perturb control
            perturbed_control = self.learned_control.copy()
            perturbed_control[i] += epsilon
            
            # Evaluate performance with perturbation
            if isinstance(self.game, InformationDiffusionGame):
                perturbed_performance = self.game.payoff_function(
                    self.game.information_state, perturbed_control, centralities
                )
            else:
                perturbed_performance = self.game.epidemic_payoff(
                    self.game.S, self.game.I, self.game.R, perturbed_control, centralities
                )
            
            gradient_estimate[i] = (perturbed_performance - current_performance) / epsilon
        
        # Update control using gradient ascent
        self.learned_control += learning_rate * gradient_estimate
        self.learned_control = np.clip(self.learned_control, 0, 1)
        
        return self.learned_control
    
    def hierarchical_control(self, hierarchy_levels: int = 3) -> np.ndarray:
        """Hierarchical control structure"""
        centralities = self.game.compute_centrality_measures()
        
        # Create hierarchy based on centrality
        sorted_indices = np.argsort(centralities['betweenness'])[::-1]
        
        control = np.zeros(self.game.n_nodes)
        budget_per_level = self.game.params.control_budget / hierarchy_levels
        
        for level in range(hierarchy_levels):
            start_idx = level * (self.game.n_nodes // hierarchy_levels)
            end_idx = (level + 1) * (self.game.n_nodes // hierarchy_levels)
            level_nodes = sorted_indices[start_idx:end_idx]
            
            # Higher level nodes get more control authority
            level_weight = hierarchy_levels - level
            
            for node in level_nodes:
                if isinstance(self.game, InformationDiffusionGame):
                    # Information nodes: more control for less informed nodes
                    info_deficit = 1.0 - self.game.information_state[node]
                    control[node] = (budget_per_level * level_weight * info_deficit / 
                                   len(level_nodes))
                else:
                    # Epidemic nodes: more control for highly infected nodes
                    infection_level = self.game.I[node]
                    control[node] = (budget_per_level * level_weight * infection_level / 
                                   len(level_nodes))
        
        return np.clip(control, 0, 1)

def run_advanced_analysis():
    """Run advanced control strategy analysis"""
    print("\n🚀 Advanced Control Strategies Analysis")
    print("=" * 50)
    
    params = GameParameters(n_players=40, time_horizon=10.0)
    
    # Create network and games
    network_gen = NetworkedGame(nx.Graph(), params)
    network = network_gen.generate_network()
    
    info_game = InformationDiffusionGame(network, params)
    epidemic_game = EpidemicControlGame(network, params)
    
    # Initialize epidemic state
    centralities = epidemic_game.compute_centrality_measures()
    high_degree = np.argsort(centralities['degree'])[-5:]
    epidemic_game.I[:] = 0.01
    epidemic_game.I[high_degree] = 0.4
    epidemic_game.S = 1 - epidemic_game.I - epidemic_game.R
    
    # Test advanced strategies
    advanced_strategies = AdvancedControlStrategies(info_game)
    
    # Consensus control
    target = np.ones(info_game.n_nodes) * 0.7
    consensus_control = advanced_strategies.distributed_consensus_control(target)
    
    # Adaptive learning control
    adaptive_control = advanced_strategies.adaptive_learning_control()
    
    # Hierarchical control
    hierarchical_control = advanced_strategies.hierarchical_control()
    
    print("Advanced Strategy Results:")
    print(f"  Consensus control effort: {np.sum(consensus_control**2):.4f}")
    print(f"  Adaptive control effort: {np.sum(adaptive_control**2):.4f}")
    print(f"  Hierarchical control effort: {np.sum(hierarchical_control**2):.4f}")
    
    return {
        'consensus': consensus_control,
        'adaptive': adaptive_control,
        'hierarchical': hierarchical_control
    }

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run main analysis
    main_results = run_comprehensive_analysis()
    
    # Run advanced analysis
    advanced_results = run_advanced_analysis()