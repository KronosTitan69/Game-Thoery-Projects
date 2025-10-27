"""
Three-Body Problem Computational Methods
========================================

This module provides comprehensive implementations of classical numerical integration,
Markov chain stochastic methods, and machine learning approaches for solving the
chaotic three-body problem.

Author: Three-Body Problem Research Team
Date: January 2025
Python Version: 3.8+

Dependencies:
- NumPy: Numerical computations
- SciPy: Scientific computing and integration
- Matplotlib: Visualization
- TensorFlow/PyTorch: Machine learning models
- scikit-learn: ML utilities
- pandas: Data manipulation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, ode
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional, Callable
import time
import warnings
warnings.filterwarnings('ignore')

# Try importing deep learning libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Some ML features will be disabled.")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Some ML features will be disabled.")


class ThreeBodySystem:
    """
    Defines the three-body gravitational system and equations of motion.
    
    The system consists of three bodies with masses m1, m2, m3 and gravitational
    parameter G. State vector is [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3].
    """
    
    def __init__(self, masses: List[float] = [1.0, 1.0, 1.0], G: float = 1.0):
        """
        Initialize three-body system.
        
        Args:
            masses: List of three body masses [m1, m2, m3]
            G: Gravitational constant
        """
        self.masses = np.array(masses)
        self.G = G
        self.n_bodies = 3
        self.n_dim = 2  # 2D problem
        self.state_size = 2 * self.n_dim * self.n_bodies  # positions + velocities
        
    def equations_of_motion(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Compute derivatives for the three-body problem.
        
        Args:
            t: Time (not used for autonomous system)
            state: State vector [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]
            
        Returns:
            State derivatives
        """
        # Extract positions and velocities
        pos = state[:6].reshape(3, 2)  # 3 bodies, 2 dimensions each
        vel = state[6:].reshape(3, 2)
        
        # Initialize acceleration array
        acc = np.zeros_like(pos)
        
        # Compute gravitational forces between all pairs
        for i in range(3):
            for j in range(3):
                if i != j:
                    # Distance vector from body i to body j
                    r_vec = pos[j] - pos[i]
                    r_mag = np.linalg.norm(r_vec)
                    
                    # Avoid singularities
                    if r_mag > 1e-10:
                        # Gravitational acceleration on body i due to body j
                        acc[i] += self.G * self.masses[j] * r_vec / r_mag**3
        
        # Return derivatives: [velocities, accelerations]
        return np.concatenate([vel.flatten(), acc.flatten()])
    
    def total_energy(self, state: np.ndarray) -> float:
        """Compute total energy (kinetic + potential) of the system."""
        pos = state[:6].reshape(3, 2)
        vel = state[6:].reshape(3, 2)
        
        # Kinetic energy
        kinetic = 0.5 * np.sum(self.masses[:, np.newaxis] * vel**2)
        
        # Potential energy
        potential = 0.0
        for i in range(3):
            for j in range(i+1, 3):
                r_mag = np.linalg.norm(pos[j] - pos[i])
                if r_mag > 1e-10:
                    potential -= self.G * self.masses[i] * self.masses[j] / r_mag
        
        return kinetic + potential
    
    def angular_momentum(self, state: np.ndarray) -> float:
        """Compute total angular momentum of the system."""
        pos = state[:6].reshape(3, 2)
        vel = state[6:].reshape(3, 2)
        
        total_L = 0.0
        for i in range(3):
            # Cross product in 2D: r × v = r_x * v_y - r_y * v_x
            L_z = pos[i, 0] * vel[i, 1] - pos[i, 1] * vel[i, 0]
            total_L += self.masses[i] * L_z
        
        return total_L


class ClassicalIntegrators:
    """
    Classical numerical integration methods for the three-body problem.
    """
    
    def __init__(self, system: ThreeBodySystem):
        self.system = system
        
    def runge_kutta_45(self, initial_state: np.ndarray, t_span: Tuple[float, float], 
                      rtol: float = 1e-8, atol: float = 1e-12, 
                      max_step: float = 0.1) -> Dict[str, Any]:
        """
        Solve using adaptive Runge-Kutta RK45 method.
        
        Args:
            initial_state: Initial conditions
            t_span: (t_start, t_end) integration time span
            rtol: Relative tolerance
            atol: Absolute tolerance
            max_step: Maximum step size
            
        Returns:
            Dictionary with solution and metadata
        """
        start_time = time.time()
        
        # Solve using scipy's RK45 implementation
        solution = solve_ivp(
            self.system.equations_of_motion,
            t_span,
            initial_state,
            method='RK45',
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            dense_output=True
        )
        
        integration_time = time.time() - start_time
        
        # Compute energy conservation
        energies = [self.system.total_energy(state) for state in solution.y.T]
        energy_error = np.std(energies) / abs(np.mean(energies))
        
        return {
            'solution': solution,
            'time': solution.t,
            'states': solution.y.T,
            'integration_time': integration_time,
            'n_evaluations': solution.nfev,
            'energy_conservation_error': energy_error,
            'success': solution.success,
            'message': solution.message
        }
    
    def bulirsch_stoer(self, initial_state: np.ndarray, t_span: Tuple[float, float],
                      rtol: float = 1e-10, atol: float = 1e-14) -> Dict[str, Any]:
        """
        Solve using Bulirsch-Stoer method for high precision.
        
        This method is particularly good for smooth problems requiring high accuracy.
        """
        start_time = time.time()
        
        # Use scipy's DOP853 which implements Dormand-Prince of order 8
        # This is a high-order method similar in spirit to Bulirsch-Stoer
        solution = solve_ivp(
            self.system.equations_of_motion,
            t_span,
            initial_state,
            method='DOP853',
            rtol=rtol,
            atol=atol,
            dense_output=True
        )
        
        integration_time = time.time() - start_time
        
        # Compute energy conservation
        energies = [self.system.total_energy(state) for state in solution.y.T]
        energy_error = np.std(energies) / abs(np.mean(energies))
        
        return {
            'solution': solution,
            'time': solution.t,
            'states': solution.y.T,
            'integration_time': integration_time,
            'n_evaluations': solution.nfev,
            'energy_conservation_error': energy_error,
            'success': solution.success,
            'message': solution.message
        }
    
    def hermite_integrator(self, initial_state: np.ndarray, t_span: Tuple[float, float],
                          h: float = 0.01, eta: float = 0.02) -> Dict[str, Any]:
        """
        Implement Hermite integrator specifically designed for N-body problems.
        
        The Hermite method uses both position and velocity information to achieve
        higher order accuracy with excellent energy conservation.
        
        Args:
            initial_state: Initial conditions
            t_span: (t_start, t_end) integration time span
            h: Initial time step
            eta: Time step control parameter
        """
        start_time = time.time()
        
        t_start, t_end = t_span
        t = t_start
        state = initial_state.copy()
        
        # Storage for solution
        times = [t]
        states = [state.copy()]
        n_evaluations = 0
        
        # Extract initial positions and velocities
        pos = state[:6].reshape(3, 2)
        vel = state[6:].reshape(3, 2)
        
        # Compute initial accelerations and jerks
        acc = self._compute_accelerations(pos)
        jerk = self._compute_jerks(pos, vel)
        n_evaluations += 2
        
        while t < t_end:
            # Adapt time step based on acceleration and jerk
            dt = min(h, eta * np.sqrt(np.max(np.linalg.norm(acc, axis=1)) / 
                                    np.max(np.linalg.norm(jerk, axis=1) + 1e-10)))
            dt = min(dt, t_end - t)
            
            # Predict positions and velocities
            pos_pred = pos + vel * dt + 0.5 * acc * dt**2 + (1/6) * jerk * dt**3
            vel_pred = vel + acc * dt + 0.5 * jerk * dt**2
            
            # Compute predicted accelerations
            acc_pred = self._compute_accelerations(pos_pred)
            n_evaluations += 1
            
            # Corrector step
            pos_corr = pos + vel * dt + 0.5 * (acc + acc_pred) * dt**2
            vel_corr = vel + 0.5 * (acc + acc_pred) * dt
            
            # Update state
            pos = pos_corr
            vel = vel_corr
            acc = acc_pred
            
            # Compute new jerk for next step
            jerk = self._compute_jerks(pos, vel)
            n_evaluations += 1
            
            t += dt
            times.append(t)
            states.append(np.concatenate([pos.flatten(), vel.flatten()]))
        
        integration_time = time.time() - start_time
        
        # Convert to arrays
        times = np.array(times)
        states = np.array(states)
        
        # Compute energy conservation
        energies = [self.system.total_energy(state) for state in states]
        energy_error = np.std(energies) / abs(np.mean(energies))
        
        return {
            'time': times,
            'states': states,
            'integration_time': integration_time,
            'n_evaluations': n_evaluations,
            'energy_conservation_error': energy_error,
            'success': True,
            'message': 'Hermite integration completed successfully'
        }
    
    def _compute_accelerations(self, pos: np.ndarray) -> np.ndarray:
        """Compute gravitational accelerations for all bodies."""
        acc = np.zeros_like(pos)
        
        for i in range(3):
            for j in range(3):
                if i != j:
                    r_vec = pos[j] - pos[i]
                    r_mag = np.linalg.norm(r_vec)
                    if r_mag > 1e-10:
                        acc[i] += self.system.G * self.system.masses[j] * r_vec / r_mag**3
        
        return acc
    
    def _compute_jerks(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Compute gravitational jerks (time derivatives of acceleration)."""
        jerk = np.zeros_like(pos)
        
        for i in range(3):
            for j in range(3):
                if i != j:
                    r_vec = pos[j] - pos[i]
                    v_vec = vel[j] - vel[i]
                    r_mag = np.linalg.norm(r_vec)
                    
                    if r_mag > 1e-10:
                        # Jerk = G*m_j * [v_vec/r^3 - 3*(r_vec·v_vec)*r_vec/r^5]
                        r_dot_v = np.dot(r_vec, v_vec)
                        jerk[i] += self.system.G * self.system.masses[j] * (
                            v_vec / r_mag**3 - 3 * r_dot_v * r_vec / r_mag**5
                        )
        
        return jerk


class MarkovChainMCMC:
    """
    Markov Chain Monte Carlo methods for stochastic three-body dynamics.
    
    Models close encounters, energy exchanges, and regime switching in chaotic dynamics.
    """
    
    def __init__(self, system: ThreeBodySystem):
        self.system = system
        self.states = ['stable', 'chaotic', 'close_encounter', 'escape']
        self.n_states = len(self.states)
        
    def build_transition_matrix(self, trajectory_data: np.ndarray, 
                               dt: float = 0.1) -> np.ndarray:
        """
        Build Markov transition matrix from trajectory data.
        
        Args:
            trajectory_data: Array of system states over time
            dt: Time step of the data
            
        Returns:
            Transition probability matrix
        """
        # Classify each state in the trajectory
        state_sequence = []
        for state in trajectory_data:
            state_class = self._classify_state(state)
            state_sequence.append(state_class)
        
        # Count transitions
        transition_counts = np.zeros((self.n_states, self.n_states))
        for i in range(len(state_sequence) - 1):
            current_state = state_sequence[i]
            next_state = state_sequence[i + 1]
            transition_counts[current_state, next_state] += 1
        
        # Normalize to get probabilities
        transition_matrix = np.zeros_like(transition_counts)
        for i in range(self.n_states):
            row_sum = np.sum(transition_counts[i, :])
            if row_sum > 0:
                transition_matrix[i, :] = transition_counts[i, :] / row_sum
            else:
                # If no transitions from this state, stay in same state
                transition_matrix[i, i] = 1.0
        
        return transition_matrix
    
    def _classify_state(self, state: np.ndarray) -> int:
        """
        Classify system state into one of the predefined categories.
        
        Args:
            state: System state vector
            
        Returns:
            State classification index
        """
        pos = state[:6].reshape(3, 2)
        vel = state[6:].reshape(3, 2)
        
        # Compute pairwise distances
        distances = []
        for i in range(3):
            for j in range(i+1, 3):
                dist = np.linalg.norm(pos[j] - pos[i])
                distances.append(dist)
        
        min_distance = min(distances)
        
        # Compute velocities
        speeds = [np.linalg.norm(vel[i]) for i in range(3)]
        max_speed = max(speeds)
        
        # Compute energy per unit mass
        energy = self.system.total_energy(state)
        kinetic = 0.5 * np.sum(self.system.masses[:, np.newaxis] * vel**2)
        
        # Classification logic
        if min_distance < 0.1:  # Very close encounter
            return 2  # close_encounter
        elif max_speed > 3.0 or energy > 0:  # High speed or positive energy (escape)
            return 3  # escape
        elif np.std(speeds) > 0.5:  # High velocity variance indicates chaos
            return 1  # chaotic
        else:
            return 0  # stable
    
    def monte_carlo_simulation(self, initial_state: np.ndarray, 
                             transition_matrix: np.ndarray,
                             n_steps: int = 1000, dt: float = 0.1) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation using Markov chain state transitions.
        
        Args:
            initial_state: Starting system state
            transition_matrix: State transition probabilities
            n_steps: Number of simulation steps
            dt: Time step size
            
        Returns:
            Simulation results and statistics
        """
        start_time = time.time()
        
        # Initialize
        current_state_class = self._classify_state(initial_state)
        current_system_state = initial_state.copy()
        
        # Storage
        state_history = []
        system_states = []
        times = []
        
        for step in range(n_steps):
            # Record current state
            state_history.append(current_state_class)
            system_states.append(current_system_state.copy())
            times.append(step * dt)
            
            # Sample next state based on transition probabilities
            next_state_class = np.random.choice(
                self.n_states, 
                p=transition_matrix[current_state_class, :]
            )
            
            # Evolve system state based on the regime
            current_system_state = self._evolve_state_by_regime(
                current_system_state, current_state_class, dt
            )
            
            current_state_class = next_state_class
        
        simulation_time = time.time() - start_time
        
        # Compute statistics
        state_probabilities = np.bincount(state_history, minlength=self.n_states) / n_steps
        
        return {
            'times': np.array(times),
            'state_history': state_history,
            'system_states': np.array(system_states),
            'state_probabilities': state_probabilities,
            'simulation_time': simulation_time,
            'transition_matrix': transition_matrix
        }
    
    def _evolve_state_by_regime(self, state: np.ndarray, regime: int, dt: float) -> np.ndarray:
        """
        Evolve system state based on current dynamical regime.
        
        Different regimes have different characteristic behaviors and noise levels.
        """
        pos = state[:6].reshape(3, 2)
        vel = state[6:].reshape(3, 2)
        
        # Add regime-specific perturbations
        if regime == 0:  # stable
            # Small random perturbations
            pos_noise = np.random.normal(0, 0.001, pos.shape)
            vel_noise = np.random.normal(0, 0.001, vel.shape)
        elif regime == 1:  # chaotic
            # Moderate perturbations with correlation
            pos_noise = np.random.normal(0, 0.01, pos.shape)
            vel_noise = np.random.normal(0, 0.01, vel.shape)
        elif regime == 2:  # close_encounter
            # Strong perturbations during close encounters
            pos_noise = np.random.normal(0, 0.05, pos.shape)
            vel_noise = np.random.normal(0, 0.1, vel.shape)
        else:  # escape
            # Escape regime - add energy
            pos_noise = np.random.normal(0, 0.02, pos.shape)
            vel_noise = np.random.normal(0, 0.2, vel.shape)
        
        # Apply perturbations
        new_pos = pos + pos_noise
        new_vel = vel + vel_noise
        
        return np.concatenate([new_pos.flatten(), new_vel.flatten()])


class MachineLearningPredictor:
    """
    Machine learning approaches for three-body problem prediction.
    
    Includes ANNs, LSTMs, MLPs, and hybrid methods combining ML with numerical integration.
    """
    
    def __init__(self, system: ThreeBodySystem):
        self.system = system
        self.scaler = StandardScaler()
        self.models = {}
        
    def generate_training_data(self, n_trajectories: int = 100, 
                             trajectory_length: int = 1000,
                             t_span: Tuple[float, float] = (0, 20)) -> Dict[str, np.ndarray]:
        """
        Generate training data using classical numerical integration.
        
        Args:
            n_trajectories: Number of different initial conditions
            trajectory_length: Length of each trajectory
            t_span: Time span for each trajectory
            
        Returns:
            Dictionary containing training data
        """
        print(f"Generating {n_trajectories} training trajectories...")
        
        X_data = []  # Input states
        y_data = []  # Next states
        trajectories = []
        
        integrator = ClassicalIntegrators(self.system)
        
        for i in range(n_trajectories):
            # Generate random initial conditions
            initial_state = self._generate_random_initial_condition()
            
            # Integrate trajectory
            try:
                result = integrator.runge_kutta_45(initial_state, t_span, max_step=0.02)
                
                if result['success'] and len(result['states']) > 10:
                    # Sample points from trajectory
                    states = result['states']
                    times = result['time']
                    
                    # Create input-output pairs for next-step prediction
                    for j in range(len(states) - 1):
                        X_data.append(states[j])
                        y_data.append(states[j + 1])
                    
                    trajectories.append({
                        'times': times,
                        'states': states,
                        'initial_state': initial_state
                    })
                    
            except Exception as e:
                print(f"Skipping trajectory {i} due to integration error: {e}")
                continue
                
            if (i + 1) % 20 == 0:
                print(f"Generated {i + 1}/{n_trajectories} trajectories")
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        # Normalize data
        X_scaled = self.scaler.fit_transform(X_data)
        y_scaled = self.scaler.transform(y_data)
        
        return {
            'X': X_scaled,
            'y': y_scaled,
            'X_raw': X_data,
            'y_raw': y_data,
            'trajectories': trajectories
        }
    
    def _generate_random_initial_condition(self) -> np.ndarray:
        """Generate random initial conditions for three-body system."""
        # Random positions in a reasonable range
        positions = np.random.uniform(-2, 2, (3, 2))
        
        # Ensure bodies are not too close
        for i in range(3):
            for j in range(i+1, 3):
                while np.linalg.norm(positions[j] - positions[i]) < 0.1:
                    positions[j] = np.random.uniform(-2, 2, 2)
        
        # Random velocities with some constraint for bound systems
        velocities = np.random.uniform(-1, 1, (3, 2))
        
        # Adjust center of mass to be at origin
        total_mass = np.sum(self.system.masses)
        com_pos = np.sum(self.system.masses[:, np.newaxis] * positions, axis=0) / total_mass
        com_vel = np.sum(self.system.masses[:, np.newaxis] * velocities, axis=0) / total_mass
        
        positions -= com_pos
        velocities -= com_vel
        
        return np.concatenate([positions.flatten(), velocities.flatten()])
    
    def build_ann_model(self, input_dim: int, hidden_layers: List[int] = [256, 128, 64]) -> Any:
        """
        Build Artificial Neural Network for trajectory prediction.
        
        Args:
            input_dim: Input dimension (should be 12 for 3-body problem)
            hidden_layers: List of hidden layer sizes
            
        Returns:
            Compiled neural network model
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available for ANN model")
        
        model = Sequential()
        model.add(Dense(hidden_layers[0], activation='relu', input_dim=input_dim))
        model.add(Dropout(0.2))
        
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(0.2))
        
        model.add(Dense(input_dim, activation='linear'))  # Output layer
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        
        return model
    
    def build_lstm_model(self, sequence_length: int, input_dim: int,
                        lstm_units: List[int] = [128, 64]) -> Any:
        """
        Build LSTM network for temporal sequence modeling.
        
        Args:
            sequence_length: Length of input sequences
            input_dim: Dimension of each time step
            lstm_units: List of LSTM layer sizes
            
        Returns:
            Compiled LSTM model
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available for LSTM model")
        
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(lstm_units[0], return_sequences=len(lstm_units) > 1,
                      input_shape=(sequence_length, input_dim)))
        model.add(Dropout(0.2))
        
        # Additional LSTM layers
        for i, units in enumerate(lstm_units[1:]):
            return_seq = i < len(lstm_units) - 2
            model.add(LSTM(units, return_sequences=return_seq))
            model.add(Dropout(0.2))
        
        # Output layer
        model.add(Dense(input_dim, activation='linear'))
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        
        return model
    
    def build_lyapunov_predictor(self, input_dim: int) -> Any:
        """
        Build MLP for predicting Lyapunov exponents (chaos indicators).
        
        Args:
            input_dim: Input dimension
            
        Returns:
            Compiled MLP model for Lyapunov prediction
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available for MLP model")
        
        model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')  # Single Lyapunov exponent output
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        
        return model
    
    def compute_lyapunov_exponents(self, trajectory: np.ndarray, dt: float = 0.1) -> float:
        """
        Compute largest Lyapunov exponent for a trajectory using finite differences.
        
        Args:
            trajectory: Array of system states over time
            dt: Time step
            
        Returns:
            Largest Lyapunov exponent
        """
        n_points = len(trajectory)
        if n_points < 10:
            return 0.0
        
        # Compute finite difference approximation
        separations = []
        initial_separation = 1e-8
        
        # Use numerical approximation
        divergence_rates = []
        for i in range(1, min(n_points, 100)):
            # Simple finite difference estimate
            state_diff = np.linalg.norm(trajectory[i] - trajectory[0])
            if state_diff > initial_separation:
                rate = np.log(state_diff / initial_separation) / (i * dt)
                divergence_rates.append(rate)
        
        if divergence_rates:
            return np.mean(divergence_rates)
        else:
            return 0.0
    
    def train_models(self, training_data: Dict[str, np.ndarray],
                    epochs: int = 50, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train all machine learning models.
        
        Args:
            training_data: Dictionary containing training data
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Dictionary containing trained models and training history
        """
        results = {}
        
        X = training_data['X']
        y = training_data['y']
        
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        if TF_AVAILABLE:
            # Train ANN model
            print("Training ANN model...")
            ann_model = self.build_ann_model(X.shape[1])
            ann_history = ann_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )
            
            self.models['ann'] = ann_model
            results['ann_history'] = ann_history.history
            
            # Train Lyapunov predictor
            print("Computing Lyapunov exponents for training...")
            lyapunov_targets = []
            for traj in training_data['trajectories'][:len(X_train)//100]:  # Sample subset
                lyap = self.compute_lyapunov_exponents(traj['states'])
                lyapunov_targets.extend([lyap] * 100)  # Repeat for each state in trajectory
            
            if len(lyapunov_targets) > 100:
                lyap_X = X_train[:len(lyapunov_targets)]
                lyap_y = np.array(lyapunov_targets)
                
                print("Training Lyapunov predictor...")
                lyap_model = self.build_lyapunov_predictor(X.shape[1])
                lyap_history = lyap_model.fit(
                    lyap_X, lyap_y,
                    validation_split=0.2,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0
                )
                
                self.models['lyapunov'] = lyap_model
                results['lyapunov_history'] = lyap_history.history
        
        # Train Gaussian Mixture Model for chaotic statistics
        print("Training Gaussian Mixture Model...")
        gmm = GaussianMixture(n_components=3, random_state=42)
        gmm.fit(X_train)
        
        self.models['gmm'] = gmm
        results['gmm_score'] = gmm.score(X_val)
        
        return results


class VisualizationSuite:
    """
    Comprehensive visualization suite for three-body problem analysis.
    """
    
    def __init__(self, system: ThreeBodySystem):
        self.system = system
        
    def plot_trajectory_comparison(self, classical_result: Dict[str, Any],
                                 ml_predictions: Optional[np.ndarray] = None,
                                 save_path: Optional[str] = None):
        """
        Plot trajectory predictions comparing classical and ML methods.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Three-Body Problem: Classical vs ML Predictions', fontsize=16)
        
        states = classical_result['states']
        times = classical_result['time']
        
        # Extract positions
        positions = states[:, :6].reshape(-1, 3, 2)
        
        # Plot trajectories in configuration space
        ax = axes[0, 0]
        colors = ['red', 'blue', 'green']
        for i in range(3):
            ax.plot(positions[:, i, 0], positions[:, i, 1], 
                   color=colors[i], label=f'Body {i+1}', alpha=0.7)
            ax.scatter(positions[0, i, 0], positions[0, i, 1], 
                      color=colors[i], marker='o', s=50, edgecolor='black')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Classical Integration Trajectories')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot energy conservation
        ax = axes[0, 1]
        energies = [self.system.total_energy(state) for state in states]
        ax.plot(times, energies, 'b-', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Total Energy')
        ax.set_title('Energy Conservation')
        ax.grid(True, alpha=0.3)
        
        # Plot phase space (velocity vs position) for body 1
        ax = axes[1, 0]
        ax.plot(positions[:, 0, 0], states[:, 6], 'r-', alpha=0.7, label='Body 1 x-v_x')
        ax.plot(positions[:, 0, 1], states[:, 7], 'r--', alpha=0.7, label='Body 1 y-v_y')
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_title('Phase Space (Body 1)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot pairwise distances
        ax = axes[1, 1]
        for i in range(3):
            for j in range(i+1, 3):
                distances = [np.linalg.norm(pos[j] - pos[i]) for pos in positions]
                ax.plot(times, distances, label=f'Distance {i+1}-{j+1}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Distance')
        ax.set_title('Pairwise Distances')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_poincare_map(self, trajectory_data: np.ndarray, save_path: Optional[str] = None):
        """
        Create Poincaré maps for phase space visualization.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Poincaré Maps - Phase Space Analysis', fontsize=16)
        
        states = trajectory_data
        positions = states[:, :6].reshape(-1, 3, 2)
        velocities = states[:, 6:].reshape(-1, 3, 2)
        
        # Poincaré section: x = 0 plane crossings for body 1
        crossings_pos = []
        crossings_vel = []
        
        for i in range(len(positions) - 1):
            # Check for zero crossing in x-coordinate of body 1
            if positions[i, 0, 0] * positions[i+1, 0, 0] < 0:
                # Linear interpolation to find exact crossing
                t_cross = -positions[i, 0, 0] / (positions[i+1, 0, 0] - positions[i, 0, 0])
                y_cross = positions[i, 0, 1] + t_cross * (positions[i+1, 0, 1] - positions[i, 0, 1])
                vy_cross = velocities[i, 0, 1] + t_cross * (velocities[i+1, 0, 1] - velocities[i, 0, 1])
                
                crossings_pos.append(y_cross)
                crossings_vel.append(vy_cross)
        
        if crossings_pos:
            axes[0].scatter(crossings_pos, crossings_vel, alpha=0.6, s=20)
            axes[0].set_xlabel('Y Position at X=0 Crossing')
            axes[0].set_ylabel('Y Velocity at X=0 Crossing')
            axes[0].set_title('Poincaré Map (Body 1, X=0 Section)')
            axes[0].grid(True, alpha=0.3)
        
        # Energy-based Poincaré map
        energies = [self.system.total_energy(state) for state in states]
        angular_momenta = [self.system.angular_momentum(state) for state in states]
        
        axes[1].scatter(angular_momenta, energies, alpha=0.6, s=20, c=range(len(energies)), cmap='viridis')
        axes[1].set_xlabel('Angular Momentum')
        axes[1].set_ylabel('Total Energy')
        axes[1].set_title('Energy-Angular Momentum Phase Space')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_computational_speedup(self, benchmark_results: Dict[str, Dict[str, float]],
                                 save_path: Optional[str] = None):
        """
        Visualize computational speed-ups achieved by different methods.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Computational Performance Comparison', fontsize=16)
        
        methods = list(benchmark_results.keys())
        times = [benchmark_results[method]['time'] for method in methods]
        accuracies = [benchmark_results[method].get('accuracy', 0) for method in methods]
        
        # Computation time comparison
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
        bars = ax1.bar(methods, times, color=colors)
        ax1.set_ylabel('Computation Time (seconds)')
        ax1.set_title('Integration Time Comparison')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add time labels on bars
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(times),
                    f'{time_val:.3f}s', ha='center', va='bottom')
        
        # Accuracy vs Speed scatter plot
        if any(acc > 0 for acc in accuracies):
            speedups = [max(times) / t for t in times]  # Relative speedup
            ax2.scatter(speedups, accuracies, s=100, c=colors[:len(methods)], alpha=0.7)
            
            for i, method in enumerate(methods):
                ax2.annotate(method, (speedups[i], accuracies[i]), 
                           xytext=(5, 5), textcoords='offset points')
            
            ax2.set_xlabel('Speedup Factor')
            ax2.set_ylabel('Accuracy (1 - relative error)')
            ax2.set_title('Speed vs Accuracy Trade-off')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def run_comprehensive_analysis():
    """
    Run complete three-body problem analysis with all methods.
    """
    print("🌌 Three-Body Problem: Comprehensive Computational Analysis")
    print("=" * 70)
    
    # Initialize system
    system = ThreeBodySystem(masses=[1.0, 1.0, 1.0])
    
    # Create stable initial conditions (approximate figure-8)
    initial_state = np.array([
        -1.0, 0.0,    # Body 1 position
         1.0, 0.0,    # Body 2 position
         0.0, 0.0,    # Body 3 position
         0.347, 0.532,    # Body 1 velocity
         0.347, 0.532,    # Body 2 velocity
        -0.694, -1.064    # Body 3 velocity
    ])
    
    print(f"Initial energy: {system.total_energy(initial_state):.6f}")
    print(f"Initial angular momentum: {system.angular_momentum(initial_state):.6f}")
    
    # Test classical integrators
    print("\n🔢 Testing Classical Numerical Integrators...")
    integrators = ClassicalIntegrators(system)
    t_span = (0, 20)
    
    benchmark_results = {}
    
    # RK45 method
    print("  Running RK45 integration...")
    rk45_result = integrators.runge_kutta_45(initial_state, t_span)
    benchmark_results['RK45'] = {
        'time': rk45_result['integration_time'],
        'accuracy': 1 - rk45_result['energy_conservation_error'],
        'result': rk45_result
    }
    
    # Bulirsch-Stoer method
    print("  Running Bulirsch-Stoer integration...")
    bs_result = integrators.bulirsch_stoer(initial_state, t_span)
    benchmark_results['Bulirsch-Stoer'] = {
        'time': bs_result['integration_time'],
        'accuracy': 1 - bs_result['energy_conservation_error'],
        'result': bs_result
    }
    
    # Hermite method
    print("  Running Hermite integration...")
    hermite_result = integrators.hermite_integrator(initial_state, t_span)
    benchmark_results['Hermite'] = {
        'time': hermite_result['integration_time'],
        'accuracy': 1 - hermite_result['energy_conservation_error'],
        'result': hermite_result
    }
    
    # Print results
    print("\n📊 Classical Integration Results:")
    for method, results in benchmark_results.items():
        print(f"  {method}:")
        print(f"    Time: {results['time']:.4f} seconds")
        print(f"    Energy conservation: {1-results['accuracy']:.2e}")
        print(f"    Function evaluations: {results['result'].get('n_evaluations', 'N/A')}")
    
    # Test Markov Chain methods
    print("\n🎲 Testing Markov Chain Monte Carlo Methods...")
    mcmc = MarkovChainMCMC(system)
    
    # Build transition matrix from RK45 trajectory
    transition_matrix = mcmc.build_transition_matrix(rk45_result['states'])
    print(f"Transition matrix shape: {transition_matrix.shape}")
    print("State transition probabilities:")
    states = ['stable', 'chaotic', 'close_encounter', 'escape']
    for i, from_state in enumerate(states):
        print(f"  From {from_state}:")
        for j, to_state in enumerate(states):
            print(f"    -> {to_state}: {transition_matrix[i,j]:.3f}")
    
    # Run Monte Carlo simulation
    mc_result = mcmc.monte_carlo_simulation(initial_state, transition_matrix, n_steps=500)
    print(f"\nMonte Carlo simulation completed in {mc_result['simulation_time']:.4f} seconds")
    print("State probabilities:")
    for i, (state, prob) in enumerate(zip(states, mc_result['state_probabilities'])):
        print(f"  {state}: {prob:.3f}")
    
    # Create visualizations
    print("\n📈 Generating Visualizations...")
    viz = VisualizationSuite(system)
    
    # Plot trajectory comparison
    viz.plot_trajectory_comparison(rk45_result)
    
    # Plot Poincaré maps
    viz.plot_poincare_map(rk45_result['states'])
    
    # Plot computational speedup
    viz.plot_computational_speedup(benchmark_results)
    
    print("\n✅ Comprehensive analysis completed successfully!")
    
    return {
        'system': system,
        'classical_results': benchmark_results,
        'mcmc_results': mc_result,
        'transition_matrix': transition_matrix
    }


if __name__ == "__main__":
    # Run comprehensive analysis
    results = run_comprehensive_analysis()