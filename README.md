# Game Theory & Computational Dynamics Projects

This repository contains comprehensive implementations of computational methods for complex dynamical systems, game theory, and chaotic dynamics. The projects combine classical mathematical approaches with modern machine learning techniques.

## 🌌 1. Three-Body Problem Computational Methods

**Complete implementation of classical numerical integration, Markov chain stochastic methods, and machine learning approaches for the chaotic three-body problem.**

### Key Features
- **Classical Integrators**: RK45, Bulirsch-Stoer, Hermite methods
- **Stochastic Modeling**: Markov Chain Monte Carlo for regime analysis
- **Machine Learning**: Neural networks for trajectory prediction
- **Comprehensive Analysis**: Validation, benchmarking, visualization

### Files
- `three_body_problem.py` - Core computational methods
- `three_body_demo.py` - Comprehensive demonstration
- `three_body_report.py` - Automated report generation
- `test_three_body.py` - Test suite
- `THREE_BODY_README.md` - Detailed documentation

### Quick Start
```python
from three_body_problem import ThreeBodySystem, ClassicalIntegrators

# Initialize system and integrate
system = ThreeBodySystem([1.0, 1.0, 1.0])
integrator = ClassicalIntegrators(system)
result = integrator.runge_kutta_45(initial_state, (0, 20))
```

**[📖 View Complete Three-Body Documentation](THREE_BODY_README.md)**

---

## 🧬 2. Evolutionary Social Dynamics

Simulation framework for modeling evolutionary game dynamics in large-scale, networked social systems with opinion dynamics, migration patterns, and social convention emergence.

### Key Features
- **Opinion Dynamics**: Replicator dynamics with network effects
- **Migration Models**: Adaptive population movement
- **Social Conventions**: Emergence and stabilization of norms
- **Stability Analysis**: Convergence and strategy variance analysis
- **Network Generation**: Realistic hybrid social networks

### Quick Start
```python
from evolutionary_social_dynamics import run_comprehensive_analysis
results = run_comprehensive_analysis()
```

---

## 🎮 3. Optimal Control and Evolution in Networked Games

Framework for simulating, optimizing, and analyzing strategic games on complex networks with control strategies and evolutionary dynamics.

### Key Features
- **Networked Games**: Information diffusion, epidemic spread
- **Control Strategies**: LQR, MPC, genetic algorithms
- **Network Analysis**: Scale-free, small-world, random topologies
- **Nash Equilibrium**: Game-theoretic solution concepts

### Quick Start
```python
from networked_games_control import run_networked_game_analysis
results = run_networked_game_analysis()
```

---

## 🗳️ 4. Election Forecasting Model

Agent-based election forecasting system combining Markov processes with demographic and socioeconomic modeling for Tamil Nadu elections.

### Key Features
- **Agent-Based Modeling**: Synthetic voter populations
- **Markov Processes**: Opinion state transitions
- **Data Integration**: Demographics, economics, candidate data
- **Forecasting Engine**: Deterministic and probabilistic predictions

### Quick Start
```python
from election_forecasting_demo import main
results = main()
```

---

## 🚀 Installation & Setup

### Requirements
- Python 3.8+
- NumPy, SciPy, Matplotlib
- scikit-learn, pandas
- NetworkX
- TensorFlow/PyTorch (optional for ML features)

### Install Dependencies
```bash
# Basic scientific stack
pip install numpy scipy matplotlib pandas scikit-learn networkx

# Optional ML libraries  
pip install tensorflow torch

# Or use system packages
sudo apt install python3-numpy python3-scipy python3-matplotlib python3-pandas python3-sklearn python3-networkx
```

### Quick Test
```bash
# Test three-body problem implementation
python test_three_body.py

# Run comprehensive demos
python three_body_demo.py
python election_forecasting_demo.py
python evolutionary_social_dynamics.py
```

## 📊 Performance Benchmarks

| Method | Speed | Accuracy | Energy Conservation |
|--------|-------|----------|-------------------|
| **Three-Body RK45** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Three-Body Hermite** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Social Dynamics** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Election Forecasting** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | N/A |

## 🔬 Scientific Applications

### Research Areas
- **Celestial Mechanics**: Orbital dynamics, asteroid trajectories
- **Chaos Theory**: Lyapunov exponents, phase space analysis  
- **Social Physics**: Opinion dynamics, voting behavior
- **Network Science**: Information spread, epidemic modeling
- **Computational Physics**: Numerical integration, ML-physics hybrid methods

### Academic Use
- Graduate research in dynamical systems
- Computational physics coursework
- Game theory and social choice studies
- Machine learning applications in physics

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- GPU acceleration for ensemble simulations
- Additional integrators (symplectic, splitting methods)
- Advanced ML architectures (physics-informed networks)
- Relativistic corrections for three-body dynamics
- Real-time election data integration

## 📚 References

1. **Hairer, E., et al.** *Solving Ordinary Differential Equations I*. Springer-Verlag.
2. **Strogatz, S.H.** *Nonlinear Dynamics and Chaos*. CRC Press.
3. **Newman, M.E.J.** *Networks*. Oxford University Press.
4. **Traulsen, A., Nowak, M.A.** Evolution of cooperation by multilevel selection. *PNAS*.

## 📄 License

MIT License - see LICENSE file for details.

---

*This repository provides a comprehensive computational framework for studying complex dynamical systems, combining rigorous mathematical methods with modern computational techniques for research and education in physics, mathematics, and social sciences.*
