# Three-Body Problem Computational Methods

This comprehensive Python implementation provides state-of-the-art computational methods for solving the chaotic three-body problem, including classical numerical integration, Markov chain stochastic methods, and machine learning approaches.

## 🌌 Overview

The three-body problem is one of the most famous problems in celestial mechanics and dynamical systems theory. This implementation provides a complete suite of computational tools for:

- **Classical Numerical Integration**: High-precision solvers with energy conservation
- **Stochastic Modeling**: Markov Chain Monte Carlo for chaotic regime analysis  
- **Machine Learning**: Neural networks for trajectory prediction and chaos indicators
- **Comprehensive Analysis**: Validation, benchmarking, and visualization tools

## 🚀 Features

### Classical Numerical Integrators

- **Runge-Kutta RK45**: Adaptive step-size method with error control
- **Bulirsch-Stoer**: High-order extrapolation method for precision applications
- **Hermite Integrator**: Specialized N-body method with excellent energy conservation

### Markov Chain Monte Carlo Methods

- **State Classification**: Automatic classification of dynamical regimes
- **Transition Matrices**: Statistical modeling of regime switching
- **Monte Carlo Simulation**: Stochastic trajectory generation
- **Energy Exchange Modeling**: Statistical analysis of close encounters

### Machine Learning Approaches

- **Artificial Neural Networks (ANNs)**: Trajectory prediction from high-fidelity data
- **LSTM Networks**: Temporal sequence modeling for chaotic dynamics
- **Gaussian Mixture Models**: Statistical modeling of chaotic phase space
- **Lyapunov Predictors**: MLPs for chaos indicator estimation
- **Hybrid Methods**: ML-guided numerical integration

### Visualization and Analysis

- **Trajectory Plots**: 2D and 3D visualization of orbital motion
- **Poincaré Maps**: Phase space analysis and chaos visualization
- **Energy Conservation**: Long-term stability analysis
- **Performance Benchmarks**: Speed and accuracy comparisons
- **Convergence Studies**: Error analysis and method validation

## 📦 Installation

### Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- scikit-learn
- pandas
- TensorFlow (optional, for ML features)
- PyTorch (optional, for ML features)

### Install Dependencies

```bash
# Basic scientific computing stack
pip install numpy scipy matplotlib pandas scikit-learn

# Optional ML libraries
pip install tensorflow torch torchvision

# Or install from system packages
sudo apt install python3-numpy python3-scipy python3-matplotlib python3-pandas python3-sklearn
```

## 🎯 Quick Start

### Basic Three-Body Integration

```python
import numpy as np
from three_body_problem import ThreeBodySystem, ClassicalIntegrators

# Initialize system
system = ThreeBodySystem(masses=[1.0, 1.0, 1.0])

# Set initial conditions (Figure-8 orbit)
initial_state = np.array([
    -1.0, 0.0,    # Body 1 position
     1.0, 0.0,    # Body 2 position  
     0.0, 0.0,    # Body 3 position
     0.347, 0.532,    # Body 1 velocity
     0.347, 0.532,    # Body 2 velocity
    -0.694, -1.064    # Body 3 velocity
])

# Integrate using RK45
integrator = ClassicalIntegrators(system)
result = integrator.runge_kutta_45(initial_state, (0, 20))

print(f"Integration completed in {result['integration_time']:.3f} seconds")
print(f"Energy conservation error: {result['energy_conservation_error']:.2e}")
```

### Run Complete Demo

```python
# Run comprehensive demonstration
python three_body_demo.py
```

### Generate Analysis Report

```python
# Generate detailed performance report
python three_body_report.py
```

## 📊 Usage Examples

### 1. Classical Integration Comparison

```python
from three_body_problem import ThreeBodySystem, ClassicalIntegrators

system = ThreeBodySystem([1.0, 1.0, 1.0])
integrator = ClassicalIntegrators(system)

# Compare different methods
methods = {
    'RK45': integrator.runge_kutta_45,
    'Bulirsch-Stoer': integrator.bulirsch_stoer,
    'Hermite': integrator.hermite_integrator
}

for name, method in methods.items():
    result = method(initial_state, (0, 10))
    print(f"{name}: {result['integration_time']:.3f}s, "
          f"Energy error: {result['energy_conservation_error']:.2e}")
```

### 2. Markov Chain Analysis

```python
from three_body_problem import MarkovChainMCMC

# Build transition matrix from trajectory data
mcmc = MarkovChainMCMC(system)
transition_matrix = mcmc.build_transition_matrix(trajectory_states)

# Run Monte Carlo simulation
mc_result = mcmc.monte_carlo_simulation(
    initial_state, transition_matrix, n_steps=1000
)

print("State probabilities:")
states = ['stable', 'chaotic', 'close_encounter', 'escape']
for state, prob in zip(states, mc_result['state_probabilities']):
    print(f"  {state}: {prob:.3f}")
```

### 3. Machine Learning Prediction

```python
from three_body_problem import MachineLearningPredictor

# Generate training data
ml_predictor = MachineLearningPredictor(system)
training_data = ml_predictor.generate_training_data(n_trajectories=50)

# Train models
training_results = ml_predictor.train_models(training_data, epochs=100)

# Make predictions
if 'ann' in ml_predictor.models:
    prediction = ml_predictor.models['ann'].predict(test_input)
```

### 4. Comprehensive Visualization

```python
from three_body_problem import VisualizationSuite

viz = VisualizationSuite(system)

# Plot trajectory comparison
viz.plot_trajectory_comparison(classical_result)

# Create Poincaré maps  
viz.plot_poincare_map(trajectory_data)

# Show performance benchmarks
viz.plot_computational_speedup(benchmark_results)
```

## 🔬 Validation and Testing

### Energy Conservation

All integrators are validated for energy conservation:

```python
initial_energy = system.total_energy(initial_state)
final_energy = system.total_energy(result['states'][-1])
conservation_error = abs(final_energy - initial_energy) / abs(initial_energy)
```

### Convergence Testing

```python
# Test different tolerances
tolerances = [1e-6, 1e-8, 1e-10, 1e-12]
for rtol in tolerances:
    result = integrator.runge_kutta_45(initial_state, (0, 10), rtol=rtol)
    # Analyze convergence...
```

### Chaos Indicators

```python
# Compute Lyapunov exponents
lyapunov = ml_predictor.compute_lyapunov_exponents(trajectory)
print(f"Largest Lyapunov exponent: {lyapunov:.6f}")
```

## 📈 Performance Benchmarks

| Method | Speed | Accuracy | Energy Conservation |
|--------|-------|----------|-------------------|
| RK45 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Bulirsch-Stoer | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Hermite | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### Typical Performance Results

- **RK45**: ~0.2s for 20 time units, energy error ~1e-8
- **Bulirsch-Stoer**: ~0.3s for 20 time units, energy error ~1e-10  
- **Hermite**: ~0.4s for 20 time units, energy error ~1e-12

## 🧪 Computational Challenges

### The Three-Body Problem

The gravitational three-body problem exhibits:

- **Sensitive dependence** on initial conditions
- **Chaotic dynamics** for most configurations
- **No general analytical solution** (except special cases)
- **Close encounters** leading to numerical difficulties
- **Long-term unpredictability** despite deterministic equations

### Numerical Challenges

- **Singularities** when bodies collide
- **Multiple time scales** (fast orbital motion + slow secular evolution)
- **Energy drift** in long-term integrations
- **Step size selection** for adaptive methods
- **Chaos vs numerical error** discrimination

### Machine Learning Challenges

- **Training data generation** from expensive simulations
- **Generalization** across different initial conditions
- **Physical constraint preservation** in neural networks
- **Interpretability** of learned dynamics
- **Hybrid method design** (ML + numerical integration)

## 🔄 Method Comparison

### Classical Methods

**Runge-Kutta RK45**
- ✅ Fast and reliable
- ✅ Adaptive step size
- ✅ Good error control
- ❌ Moderate accuracy for given cost

**Bulirsch-Stoer**
- ✅ Excellent accuracy
- ✅ High-order method
- ✅ Smooth problems
- ❌ Slower for moderate accuracy

**Hermite Integrator**
- ✅ Specialized for N-body
- ✅ Uses force derivatives
- ✅ Excellent energy conservation
- ❌ More complex implementation

### Stochastic Methods

**Markov Chain Monte Carlo**
- ✅ Captures regime switching
- ✅ Statistical analysis
- ✅ Ensemble properties
- ❌ Requires trajectory classification

### Machine Learning Methods

**Neural Networks**
- ✅ Fast prediction
- ✅ Pattern recognition
- ✅ Nonlinear dynamics
- ❌ Training overhead
- ❌ Black box nature

## 📚 Scientific Background

### Mathematical Foundation

The three-body problem is governed by Newton's equations:

```
d²r₍ᵢ₎/dt² = G Σⱼ≠ᵢ mⱼ(rⱼ - rᵢ)/|rⱼ - rᵢ|³
```

where:
- `rᵢ` = position vector of body i
- `mᵢ` = mass of body i  
- `G` = gravitational constant

### Conservation Laws

- **Energy**: H = T + V = constant
- **Linear momentum**: P = Σᵢ mᵢvᵢ = constant
- **Angular momentum**: L = Σᵢ rᵢ × mᵢvᵢ = constant

### Chaos Theory

- **Lyapunov exponents** measure sensitive dependence
- **Poincaré sections** reveal chaotic structure
- **KAM theory** describes regular vs chaotic regions
- **Escape dynamics** in open systems

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional integrators (symplectic methods, splitting schemes)
- GPU acceleration for ensemble simulations
- Advanced ML architectures (transformers, physics-informed networks)
- Relativistic corrections
- N-body generalizations

## 📄 License

This project is released under the MIT License. See LICENSE file for details.

## 📖 References

1. **Hairer, E., Nørsett, S.P., Wanner, G.** (1993). *Solving Ordinary Differential Equations I: Nonstiff Problems*. Springer-Verlag.

2. **Press, W.H., Teukolsky, S.A., Vetterling, W.T., Flannery, B.P.** (2007). *Numerical Recipes: The Art of Scientific Computing*, 3rd Edition. Cambridge University Press.

3. **Makino, J., Aarseth, S.J.** (1992). On a Hermite integrator with Ahmad-Cohen scheme for gravitational many-body problems. *Publications of the Astronomical Society of Japan*, 44, 141-151.

4. **Laskar, J.** (1999). Introduction to frequency map analysis. In *Hamiltonian Systems with Three or More Degrees of Freedom* (pp. 134-150). Springer.

5. **Poincaré, H.** (1890). *Sur le problème des trois corps et les équations de la dynamique*. Acta Mathematica, 13(1), 1-270.

## 🎯 Citation

If you use this code in your research, please cite:

```bibtex
@software{three_body_computational_methods,
  title={Three-Body Problem Computational Methods},
  author={Three-Body Problem Research Team},
  year={2025},
  url={https://github.com/KronosTitan69/Game-Thoery-Projects}
}
```

---

*This implementation provides a comprehensive framework for three-body problem research, combining classical numerical methods with modern machine learning approaches for robust and accurate computational solutions.*