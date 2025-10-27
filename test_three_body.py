"""
Basic test suite for three-body problem computational methods.

This script runs basic functionality tests to ensure all components work correctly.
"""

import numpy as np
import time
import sys

def test_basic_integration():
    """Test basic three-body integration functionality."""
    print("Testing basic three-body integration...")
    
    try:
        from three_body_problem import ThreeBodySystem, ClassicalIntegrators
        
        # Initialize system
        system = ThreeBodySystem([1.0, 1.0, 1.0])
        integrator = ClassicalIntegrators(system)
        
        # Simple initial conditions
        initial_state = np.array([
            -1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.2, 0.3, 0.2, 0.3, -0.4, -0.6
        ])
        
        # Test RK45
        result = integrator.runge_kutta_45(initial_state, (0, 5))
        
        assert result['success'], "RK45 integration failed"
        assert len(result['states']) > 10, "Too few integration points"
        assert result['energy_conservation_error'] < 1e-6, "Poor energy conservation"
        
        print("  ✅ RK45 integration: PASSED")
        
        # Test energy conservation
        initial_energy = system.total_energy(initial_state)
        final_energy = system.total_energy(result['states'][-1])
        energy_error = abs(final_energy - initial_energy) / abs(initial_energy)
        
        assert energy_error < 1e-6, f"Energy conservation error too large: {energy_error}"
        print(f"  ✅ Energy conservation: {energy_error:.2e} - PASSED")
        
        # Test Bulirsch-Stoer
        bs_result = integrator.bulirsch_stoer(initial_state, (0, 5))
        assert bs_result['success'], "Bulirsch-Stoer integration failed"
        print("  ✅ Bulirsch-Stoer integration: PASSED")
        
        # Test Hermite
        hermite_result = integrator.hermite_integrator(initial_state, (0, 5))
        assert hermite_result['success'], "Hermite integration failed"
        print("  ✅ Hermite integration: PASSED")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic integration test FAILED: {e}")
        return False

def test_markov_chain():
    """Test Markov chain functionality."""
    print("Testing Markov chain methods...")
    
    try:
        from three_body_problem import ThreeBodySystem, ClassicalIntegrators, MarkovChainMCMC
        
        system = ThreeBodySystem([1.0, 1.0, 1.0])
        mcmc = MarkovChainMCMC(system)
        
        # Generate some trajectory data
        integrator = ClassicalIntegrators(system)
        initial_state = np.array([
            -1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.2, 0.3, 0.2, 0.3, -0.4, -0.6
        ])
        
        result = integrator.runge_kutta_45(initial_state, (0, 10))
        
        # Build transition matrix
        transition_matrix = mcmc.build_transition_matrix(result['states'])
        
        assert transition_matrix.shape == (4, 4), "Wrong transition matrix shape"
        assert np.allclose(np.sum(transition_matrix, axis=1), 1.0), "Transition matrix not normalized"
        
        print("  ✅ Transition matrix construction: PASSED")
        
        # Test Monte Carlo simulation
        mc_result = mcmc.monte_carlo_simulation(initial_state, transition_matrix, n_steps=100)
        
        assert len(mc_result['state_history']) == 100, "Wrong number of simulation steps"
        assert np.allclose(np.sum(mc_result['state_probabilities']), 1.0), "State probabilities not normalized"
        
        print("  ✅ Monte Carlo simulation: PASSED")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Markov chain test FAILED: {e}")
        return False

def test_machine_learning():
    """Test machine learning functionality."""
    print("Testing machine learning methods...")
    
    try:
        from three_body_problem import ThreeBodySystem, MachineLearningPredictor
        
        system = ThreeBodySystem([1.0, 1.0, 1.0])
        ml_predictor = MachineLearningPredictor(system)
        
        # Test training data generation
        training_data = ml_predictor.generate_training_data(n_trajectories=3, trajectory_length=50)
        
        assert 'X' in training_data, "Missing training input data"
        assert 'y' in training_data, "Missing training output data"
        assert training_data['X'].shape[1] == 12, "Wrong input dimension"
        assert training_data['y'].shape[1] == 12, "Wrong output dimension"
        
        print("  ✅ Training data generation: PASSED")
        
        # Test model training (basic components)
        training_results = ml_predictor.train_models(training_data, epochs=2)
        
        assert 'gmm' in ml_predictor.models, "GMM model not trained"
        
        print("  ✅ Basic model training: PASSED")
        
        # Test Lyapunov computation
        trajectory = training_data['trajectories'][0]['states']
        lyapunov = ml_predictor.compute_lyapunov_exponents(trajectory)
        
        assert isinstance(lyapunov, float), "Lyapunov exponent should be float"
        
        print("  ✅ Lyapunov computation: PASSED")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Machine learning test FAILED: {e}")
        return False

def test_visualization():
    """Test visualization functionality."""
    print("Testing visualization methods...")
    
    try:
        from three_body_problem import ThreeBodySystem, ClassicalIntegrators, VisualizationSuite
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for testing
        
        system = ThreeBodySystem([1.0, 1.0, 1.0])
        integrator = ClassicalIntegrators(system)
        viz = VisualizationSuite(system)
        
        # Generate test data
        initial_state = np.array([
            -1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.2, 0.3, 0.2, 0.3, -0.4, -0.6
        ])
        
        result = integrator.runge_kutta_45(initial_state, (0, 5))
        
        # Test trajectory visualization (should not crash)
        viz.plot_trajectory_comparison(result)
        
        print("  ✅ Trajectory visualization: PASSED")
        
        # Test Poincaré map
        viz.plot_poincare_map(result['states'])
        
        print("  ✅ Poincaré map visualization: PASSED")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Visualization test FAILED: {e}")
        return False

def run_all_tests():
    """Run all test functions."""
    print("🌌 Three-Body Problem - Running Basic Tests")
    print("=" * 50)
    
    start_time = time.time()
    
    tests = [
        ("Basic Integration", test_basic_integration),
        ("Markov Chain Methods", test_markov_chain),
        ("Machine Learning", test_machine_learning),
        ("Visualization", test_visualization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔄 {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: ALL TESTS PASSED")
            else:
                print(f"❌ {test_name}: SOME TESTS FAILED")
        except Exception as e:
            print(f"❌ {test_name}: CRITICAL ERROR - {e}")
    
    execution_time = time.time() - start_time
    
    print("\n" + "=" * 50)
    print("🎯 Test Summary")
    print(f"Passed: {passed}/{total} test suites")
    print(f"Execution time: {execution_time:.2f} seconds")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print("⚠️  SOME TESTS FAILED!")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)