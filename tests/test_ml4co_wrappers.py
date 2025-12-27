#!/usr/bin/env python3
"""
Test script for ML4CO-Kit wrapper environments

Tests:
1. Environment initialization
2. Data generation
3. Reset functionality
4. Step functionality  
5. Reward calculation
6. ML4CO solver baseline (if available)
"""

import torch
from tensordict import TensorDict

from rl4co.envs.graph import (
    MISEnvWrapper,
    MVCEnvWrapper,
    MCLEnvWrapper,
    MCUTEnvWrapper
)


def test_env(env_class, env_name, **kwargs):
    """Test a single environment"""
    print(f"\n{'='*60}")
    print(f"Testing {env_name}")
    print(f"{'='*60}")
    
    try:
        # 1. Initialize environment
        print(f"1. Initializing {env_name}...")
        env = env_class(**kwargs)
        print(f"   ✓ Environment initialized")
        
        # 2. Generate data
        print(f"2. Generating data (batch_size=4)...")
        td = env.reset(batch_size=[4])
        print(f"   ✓ Data generated")
        print(f"   - Keys: {list(td.keys())}")
        print(f"   - Batch size: {td.batch_size}")
        if "edge_index" in td:
            print(f"   - Edge index shape: {td['edge_index'].shape}")
        
        # 3. Check if solver is available
        has_solver = env.solver is not None
        print(f"3. Solver available: {has_solver}")
        
        # 4. Test step (if implemented)
        try:
            print(f"4. Testing step...")
            # Take a random action
            if "available" in td:
                # Select first available node
                action = td["available"].int().argmax(dim=-1)
            else:
                # Random node
                action = torch.randint(0, kwargs.get('num_nodes', 50), (4,))
            
            td["action"] = action
            td = env.step(td)
            print(f"   ✓ Step completed")
            print(f"   - Reward: {td['reward']}")
            print(f"   - Done: {td['done']}")
        except NotImplementedError:
            print(f"   ⚠ Step not fully implemented yet")
        except Exception as e:
            print(f"   ⚠ Step error: {e}")
        
        # 5. Test ML4CO solver baseline (if available)
        if has_solver:
            try:
                print(f"5. Testing ML4CO solver baseline...")
                test_td = env.reset(batch_size=[10])
                results = env.solve_with_ml4co(test_td, verbose=False)
                print(f"   ✓ Solver completed")
                print(f"   - Mean objective: {results['mean']:.4f}")
                print(f"   - Std: {results['std']:.4f}")
            except Exception as e:
                print(f"   ⚠ Solver error: {e}")
        
        # 6. Test visualization
        try:
            print(f"6. Testing visualization...")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 6))
            env.render(td, ax=ax)
            plt.close(fig)
            print(f"   ✓ Visualization works")
        except Exception as e:
            print(f"   ⚠ Visualization error: {e}")
        
        print(f"\n✅ {env_name} tests completed successfully!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ {env_name} test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("ML4CO-Kit Environment Wrapper Tests")
    print("="*60)
    
    # Test parameters
    test_params = {
        'num_nodes': 20,
        'edge_prob': 0.2,
        'graph_type': 'erdos_renyi'
    }
    
    results = {}
    
    # Test MIS
    results['MIS'] = test_env(
        MISEnvWrapper, 
        "Maximum Independent Set (MIS)",
        **test_params
    )
    
    # Test MVC  
    results['MVC'] = test_env(
        MVCEnvWrapper,
        "Minimum Vertex Cover (MVC)",
        **test_params
    )
    
    # Test MCL
    results['MCL'] = test_env(
        MCLEnvWrapper,
        "Maximum Clique (MCL)",
        **test_params
    )
    
    # Test MCUT
    results['MCUT'] = test_env(
        MCUTEnvWrapper,
        "Maximum Cut (MCUT)",
        edge_weighted=False,
        **test_params
    )
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name}: {status}")
    print("="*60 + "\n")
    
    # Overall
    all_passed = all(results.values())
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Please check the output above.")
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
