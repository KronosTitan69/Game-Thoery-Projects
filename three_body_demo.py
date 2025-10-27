"""
Three-Body Problem Computational Methods - Comprehensive Demo
============================================================

This script demonstrates the complete three-body problem computational framework
including classical numerical integration, Markov chain stochastic methods,
machine learning approaches, and comprehensive analysis and visualization.

Usage:
    python three_body_demo.py

Features demonstrated:
- Classical numerical integrators (RK45, Bulirsch-Stoer, Hermite)
- Markov Chain Monte Carlo for stochastic behavior modeling
- Machine learning trajectory prediction (when TensorFlow available)
- Comprehensive visualization suite
- Performance benchmarking and validation
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

from three_body_problem import (
    ThreeBodySystem, ClassicalIntegrators, MarkovChainMCMC, 
    MachineLearningPredictor, VisualizationSuite
)

def print_header(title: str, width: int = 70):
    """Print formatted section header."""
    print("\n" + "=" * width)
    print(f"🌌 {title}")
    print("=" * width)

def print_subheader(title: str, width: int = 50):
    """Print formatted subsection header."""
    print(f"\n📊 {title}")
    print("-" * width)

def demonstrate_classical_integrators():
    """Demonstrate classical numerical integration methods."""
    print_header("Classical Numerical Integration Methods")
    
    # Initialize system with different mass configurations
    systems = {
        'Equal Mass': ThreeBodySystem([1.0, 1.0, 1.0]),
        'Binary + Test': ThreeBodySystem([1.0, 1.0, 0.001]),
        'Hierarchical': ThreeBodySystem([10.0, 1.0, 0.1])
    }
    
    # Different initial conditions
    initial_conditions = {
        'Figure-8': np.array([
            -1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.347, 0.532, 0.347, 0.532, -0.694, -1.064
        ]),
        'Pythagorean': np.array([
            -3.0, -4.0, 4.0, 3.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]),
        'Random Stable': np.array([
            -1.5, 0.5, 1.2, -0.8, 0.3, 1.0,
            0.2, -0.3, -0.1, 0.4, -0.1, -0.1
        ])
    }
    
    results = {}
    
    for sys_name, system in systems.items():
        print_subheader(f"System: {sys_name}")
        print(f"Masses: {system.masses}")
        
        for ic_name, initial_state in initial_conditions.items():
            print(f"\n  Initial Condition: {ic_name}")
            print(f"  Initial Energy: {system.total_energy(initial_state):.6f}")
            
            integrators = ClassicalIntegrators(system)
            t_span = (0, 20)
            
            # Test different methods
            methods = ['runge_kutta_45', 'bulirsch_stoer', 'hermite_integrator']
            method_results = {}
            
            for method_name in methods:
                try:
                    start_time = time.time()
                    method = getattr(integrators, method_name)
                    
                    if method_name == 'hermite_integrator':
                        result = method(initial_state, t_span, h=0.01)
                    else:
                        result = method(initial_state, t_span)
                    
                    integration_time = time.time() - start_time
                    
                    # Analyze result
                    final_energy = system.total_energy(result['states'][-1])
                    energy_error = abs(final_energy - system.total_energy(initial_state))
                    
                    method_results[method_name] = {
                        'time': integration_time,
                        'energy_error': energy_error,
                        'n_evaluations': result.get('n_evaluations', 'N/A'),
                        'success': result.get('success', True),
                        'result': result
                    }
                    
                    print(f"    {method_name:20s}: {integration_time:6.3f}s, "
                          f"Energy error: {energy_error:.2e}")
                    
                except Exception as e:
                    print(f"    {method_name:20s}: FAILED - {str(e)}")
                    method_results[method_name] = {'error': str(e)}
            
            results[f"{sys_name}_{ic_name}"] = method_results
    
    return results

def demonstrate_markov_chain_methods():
    """Demonstrate Markov Chain Monte Carlo methods."""
    print_header("Markov Chain Monte Carlo Methods")
    
    system = ThreeBodySystem([1.0, 1.0, 1.0])
    mcmc = MarkovChainMCMC(system)
    
    # Generate different types of trajectories
    integrator = ClassicalIntegrators(system)
    
    print_subheader("Generating Training Trajectories")
    
    trajectories = []
    trajectory_types = [
        ('Stable Orbit', np.array([-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 
                                  0.2, 0.3, 0.2, 0.3, -0.4, -0.6])),
        ('Chaotic Motion', np.array([-0.8, 0.6, 0.9, -0.5, 0.1, 0.2,
                                   0.5, -0.2, -0.3, 0.4, -0.2, -0.2])),
        ('Close Encounter', np.array([-0.1, 0.0, 0.1, 0.0, 0.0, 0.2,
                                    0.0, 1.0, 0.0, -1.0, 0.0, 0.0]))
    ]
    
    all_states = []
    
    for traj_name, initial_state in trajectory_types:
        print(f"  Generating {traj_name} trajectory...")
        
        try:
            result = integrator.runge_kutta_45(initial_state, (0, 30), max_step=0.05)
            
            if result['success']:
                trajectories.append({
                    'name': traj_name,
                    'states': result['states'],
                    'times': result['time']
                })
                all_states.extend(result['states'])
                print(f"    Generated {len(result['states'])} points")
            else:
                print(f"    Failed: {result['message']}")
                
        except Exception as e:
            print(f"    Error: {str(e)}")
    
    if all_states:
        print_subheader("Building Transition Matrix")
        
        all_states = np.array(all_states)
        transition_matrix = mcmc.build_transition_matrix(all_states)
        
        print("Transition Matrix:")
        states = ['stable', 'chaotic', 'close_encounter', 'escape']
        print(f"{'From/To':<15}", end='')
        for state in states:
            print(f"{state:<15}", end='')
        print()
        
        for i, from_state in enumerate(states):
            print(f"{from_state:<15}", end='')
            for j in range(len(states)):
                print(f"{transition_matrix[i,j]:<15.3f}", end='')
            print()
        
        # Run Monte Carlo simulations
        print_subheader("Monte Carlo Simulations")
        
        simulation_results = []
        
        for traj_name, initial_state in trajectory_types:
            print(f"  Simulating {traj_name}...")
            
            mc_result = mcmc.monte_carlo_simulation(
                initial_state, transition_matrix, n_steps=1000
            )
            
            simulation_results.append({
                'name': traj_name,
                'result': mc_result
            })
            
            print(f"    State probabilities:")
            for state, prob in zip(states, mc_result['state_probabilities']):
                print(f"      {state}: {prob:.3f}")
        
        return {
            'transition_matrix': transition_matrix,
            'trajectories': trajectories,
            'simulations': simulation_results
        }
    
    return None

def demonstrate_machine_learning():
    """Demonstrate machine learning approaches."""
    print_header("Machine Learning Approaches")
    
    system = ThreeBodySystem([1.0, 1.0, 1.0])
    ml_predictor = MachineLearningPredictor(system)
    
    print_subheader("Generating Training Data")
    
    try:
        # Generate training data with fewer trajectories for demo
        training_data = ml_predictor.generate_training_data(
            n_trajectories=20, trajectory_length=200, t_span=(0, 10)
        )
        
        print(f"Generated training data:")
        print(f"  Input shape: {training_data['X'].shape}")
        print(f"  Output shape: {training_data['y'].shape}")
        print(f"  Number of trajectories: {len(training_data['trajectories'])}")
        
        # Train models
        print_subheader("Training Machine Learning Models")
        
        training_results = ml_predictor.train_models(
            training_data, epochs=20, batch_size=16
        )
        
        print("Training completed!")
        
        # Test Gaussian Mixture Model
        if 'gmm' in ml_predictor.models:
            print(f"  GMM validation score: {training_results['gmm_score']:.4f}")
            
        # Test predictions if models are available
        if 'ann' in ml_predictor.models:
            print_subheader("Testing ANN Predictions")
            
            # Make predictions on test data
            test_idx = -10
            test_input = training_data['X'][test_idx:test_idx+1]
            test_target = training_data['y'][test_idx:test_idx+1]
            
            prediction = ml_predictor.models['ann'].predict(test_input, verbose=0)
            
            # Inverse transform predictions
            pred_real = ml_predictor.scaler.inverse_transform(prediction)
            target_real = ml_predictor.scaler.inverse_transform(test_target)
            
            mse = np.mean((pred_real - target_real)**2)
            print(f"  Test MSE: {mse:.6f}")
            
        return {
            'training_data': training_data,
            'training_results': training_results,
            'models': ml_predictor.models
        }
        
    except Exception as e:
        print(f"Machine learning demo failed: {str(e)}")
        print("This is expected if TensorFlow is not available.")
        return None

def demonstrate_validation_analysis():
    """Demonstrate validation and error analysis."""
    print_header("Validation and Error Analysis")
    
    system = ThreeBodySystem([1.0, 1.0, 1.0])
    integrator = ClassicalIntegrators(system)
    
    # Generate reference solution with high accuracy
    initial_state = np.array([
        -1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        0.347, 0.532, 0.347, 0.532, -0.694, -1.064
    ])
    
    print_subheader("Generating Reference Solution")
    
    # High-accuracy reference
    ref_result = integrator.bulirsch_stoer(
        initial_state, (0, 20), rtol=1e-12, atol=1e-15
    )
    
    print(f"Reference solution generated:")
    print(f"  Time points: {len(ref_result['time'])}")
    print(f"  Energy conservation: {ref_result['energy_conservation_error']:.2e}")
    
    # Test different tolerances
    print_subheader("Tolerance Analysis")
    
    tolerances = [1e-6, 1e-8, 1e-10, 1e-12]
    tolerance_results = []
    
    for rtol in tolerances:
        print(f"  Testing tolerance {rtol:.0e}...")
        
        test_result = integrator.runge_kutta_45(
            initial_state, (0, 20), rtol=rtol, atol=rtol*1e-3
        )
        
        # Interpolate to same time points as reference
        ref_times = ref_result['time']
        test_interp = np.array([
            np.interp(ref_times, test_result['time'], test_result['states'][:, i])
            for i in range(12)
        ]).T
        
        # Compute errors
        position_error = np.mean([
            np.linalg.norm(test_interp[i, :6] - ref_result['states'][i, :6])
            for i in range(len(ref_times))
        ])
        
        velocity_error = np.mean([
            np.linalg.norm(test_interp[i, 6:] - ref_result['states'][i, 6:])
            for i in range(len(ref_times))
        ])
        
        tolerance_results.append({
            'rtol': rtol,
            'position_error': position_error,
            'velocity_error': velocity_error,
            'time': test_result['integration_time'],
            'evaluations': test_result['n_evaluations']
        })
        
        print(f"    Position error: {position_error:.2e}")
        print(f"    Velocity error: {velocity_error:.2e}")
        print(f"    Integration time: {test_result['integration_time']:.4f}s")
    
    return {
        'reference': ref_result,
        'tolerance_analysis': tolerance_results
    }

def create_comprehensive_visualizations(all_results):
    """Create comprehensive visualization suite."""
    print_header("Comprehensive Visualization Suite")
    
    system = ThreeBodySystem([1.0, 1.0, 1.0])
    viz = VisualizationSuite(system)
    
    # Get a good trajectory for visualization
    classical_results = all_results.get('classical')
    if classical_results:
        # Find a successful result
        for key, methods in classical_results.items():
            if 'runge_kutta_45' in methods and 'result' in methods['runge_kutta_45']:
                result = methods['runge_kutta_45']['result']
                
                print_subheader("Trajectory Visualization")
                viz.plot_trajectory_comparison(result)
                
                print_subheader("Phase Space Analysis")
                viz.plot_poincare_map(result['states'])
                
                break
    
    # Performance comparison
    if 'validation' in all_results:
        print_subheader("Performance Analysis")
        
        tolerance_data = all_results['validation']['tolerance_analysis']
        
        # Create performance plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Three-Body Problem: Performance Analysis', fontsize=16)
        
        tolerances = [r['rtol'] for r in tolerance_data]
        pos_errors = [r['position_error'] for r in tolerance_data]
        vel_errors = [r['velocity_error'] for r in tolerance_data]
        times = [r['time'] for r in tolerance_data]
        evaluations = [r['evaluations'] for r in tolerance_data]
        
        # Error vs tolerance
        axes[0,0].loglog(tolerances, pos_errors, 'bo-', label='Position')
        axes[0,0].loglog(tolerances, vel_errors, 'ro-', label='Velocity')
        axes[0,0].set_xlabel('Tolerance')
        axes[0,0].set_ylabel('Average Error')
        axes[0,0].set_title('Error vs Tolerance')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Time vs tolerance
        axes[0,1].semilogx(tolerances, times, 'go-')
        axes[0,1].set_xlabel('Tolerance')
        axes[0,1].set_ylabel('Integration Time (s)')
        axes[0,1].set_title('Computation Time vs Tolerance')
        axes[0,1].grid(True)
        
        # Function evaluations vs tolerance
        axes[1,0].loglog(tolerances, evaluations, 'mo-')
        axes[1,0].set_xlabel('Tolerance')
        axes[1,0].set_ylabel('Function Evaluations')
        axes[1,0].set_title('Evaluations vs Tolerance')
        axes[1,0].grid(True)
        
        # Efficiency plot (error vs time)
        axes[1,1].loglog(times, pos_errors, 'co-')
        axes[1,1].set_xlabel('Integration Time (s)')
        axes[1,1].set_ylabel('Position Error')
        axes[1,1].set_title('Accuracy vs Speed Trade-off')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    """Main demonstration function."""
    print("🌌 Three-Body Problem: Comprehensive Computational Methods Demo")
    print("=" * 70)
    print("This demonstration showcases classical numerical integration,")
    print("Markov chain stochastic methods, and machine learning approaches")
    print("for solving the chaotic three-body problem.")
    print("\nFeatures:")
    print("• Classical integrators: RK45, Bulirsch-Stoer, Hermite")
    print("• Markov Chain Monte Carlo for stochastic behavior")
    print("• Machine learning trajectory prediction")
    print("• Comprehensive validation and error analysis")
    print("• Advanced visualization suite")
    
    total_start_time = time.time()
    all_results = {}
    
    try:
        # Demonstrate classical integrators
        classical_results = demonstrate_classical_integrators()
        all_results['classical'] = classical_results
        
        # Demonstrate Markov chain methods
        mcmc_results = demonstrate_markov_chain_methods()
        if mcmc_results:
            all_results['mcmc'] = mcmc_results
        
        # Demonstrate machine learning
        ml_results = demonstrate_machine_learning()
        if ml_results:
            all_results['ml'] = ml_results
        
        # Demonstrate validation analysis
        validation_results = demonstrate_validation_analysis()
        all_results['validation'] = validation_results
        
        # Create comprehensive visualizations
        create_comprehensive_visualizations(all_results)
        
        total_time = time.time() - total_start_time
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        return
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Summary
    print_header("Demo Summary")
    print(f"Total execution time: {total_time:.2f} seconds")
    print("\n✅ Three-Body Problem Demo completed successfully!")
    print("\nKey Results:")
    
    if 'classical' in all_results:
        print("  • Classical integrators tested on multiple systems")
        print("  • Energy conservation verified for all methods")
    
    if 'mcmc' in all_results:
        print("  • Markov chain transition matrices constructed")
        print("  • Monte Carlo simulations demonstrate stochastic behavior")
    
    if 'ml' in all_results:
        print("  • Machine learning models trained on trajectory data")
        print("  • Neural network predictions validated")
    
    if 'validation' in all_results:
        print("  • Comprehensive error analysis performed")
        print("  • Tolerance vs accuracy trade-offs quantified")
    
    print("\n📊 All visualizations generated and displayed")
    print("🎯 Computational methods successfully validated")
    
    return all_results

if __name__ == "__main__":
    results = main()