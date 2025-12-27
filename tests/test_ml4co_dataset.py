"""
Test script for ML4CO-Kit dataset reader.

This script tests loading and using ML4CO-Kit datasets with RL4CO.
"""

import torch
import numpy as np
from pathlib import Path
from rl4co.data.ml4co_dataset import load_ml4co_dataset, TSPDataset, CVRPDataset
from rl4co.envs import TSPEnv, CVRPEnv
from rl4co.envs.routing.tsp.solvers import solve_tsp


def test_tsp_dataset():
    """Test loading TSP dataset from ML4CO-Kit."""
    print("\n" + "="*80)
    print("TEST 1: Loading TSP Dataset")
    print("="*80)
    
    dataset_path = "/root/autodl-tmp/ML4CO-Kit/test_dataset/tsp/wrapper/tsp50_uniform_16ins.txt"
    
    if not Path(dataset_path).exists():
        print(f"⚠️  Dataset not found: {dataset_path}")
        return
    
    try:
        # Load dataset
        print(f"\n📂 Loading dataset from: {dataset_path}")
        dataset = load_ml4co_dataset('tsp', dataset_path)
        print(f"✅ Successfully loaded {len(dataset)} instances")
        
        # Examine first instance
        print("\n📊 First instance details:")
        instance = dataset[0]
        print(f"  - Locations shape: {instance['locs'].shape}")
        print(f"  - Tour shape: {instance['tour'].shape}")
        print(f"  - First 5 locations:\n{instance['locs'][:5]}")
        print(f"  - First 10 tour nodes: {instance['tour'][:10].tolist()}")
        
        # Validate tour
        num_nodes = len(instance['locs'])
        tour_set = set(instance['tour'].tolist())
        print(f"\n🔍 Tour validation:")
        print(f"  - Number of nodes: {num_nodes}")
        print(f"  - Tour length: {len(instance['tour'])}")
        print(f"  - Unique nodes in tour: {len(tour_set)}")
        print(f"  - Is valid tour: {len(tour_set) == num_nodes and tour_set == set(range(num_nodes))}")
        
        # Calculate tour cost
        locs = instance['locs'].numpy()
        tour = instance['tour'].numpy()
        tour_cost = 0.0
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            dist = np.linalg.norm(locs[tour[i]] - locs[tour[j]])
            tour_cost += dist
        print(f"  - Tour cost (ML4CO solution): {tour_cost:.4f}")
        
        # Test with RL4CO solver
        print(f"\n🔧 Testing with RL4CO solver:")
        try:
            # Use greedy as a baseline
            greedy_tour, greedy_cost, greedy_info = solve_tsp(
                locs, 
                method='greedy',
                verbose=False
            )
            print(f"  - RL4CO Greedy cost: {greedy_cost:.4f}")
            print(f"  - Gap to optimal: {((greedy_cost / tour_cost - 1) * 100):.2f}%")
        except Exception as e:
            print(f"  ⚠️  RL4CO solver error: {e}")
        
        # Test batch loading
        print(f"\n📦 Batch loading test:")
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch = next(iter(dataloader))
        print(f"  - Batch locs shape: {batch['locs'].shape}")
        print(f"  - Batch tour shape: {batch['tour'].shape}")
        
        print("\n✅ TSP dataset test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ TSP dataset test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cvrp_dataset():
    """Test loading CVRP dataset from ML4CO-Kit."""
    print("\n" + "="*80)
    print("TEST 2: Loading CVRP Dataset")
    print("="*80)
    
    dataset_path = "/root/autodl-tmp/ML4CO-Kit/test_dataset/cvrp/wrapper/cvrp50_uniform_16ins.txt"
    
    if not Path(dataset_path).exists():
        print(f"⚠️  Dataset not found: {dataset_path}")
        return
    
    try:
        # Load dataset
        print(f"\n📂 Loading dataset from: {dataset_path}")
        dataset = load_ml4co_dataset('cvrp', dataset_path)
        print(f"✅ Successfully loaded {len(dataset)} instances")
        
        # Examine first instance
        print("\n📊 First instance details:")
        instance = dataset[0]
        print(f"  - Depot shape: {instance['depot'].shape}")
        print(f"  - Depot location: {instance['depot'].tolist()}")
        print(f"  - Customer locations shape: {instance['locs'].shape}")
        print(f"  - Demands shape: {instance['demand'].shape}")
        print(f"  - Capacity: {instance['capacity'].item():.1f}")
        print(f"  - Tour length: {len(instance['tour'])}")
        print(f"  - First 5 customer locations:\n{instance['locs'][:5]}")
        print(f"  - First 10 demands: {instance['demand'][:10].tolist()}")
        print(f"  - First 20 tour nodes: {instance['tour'][:20].tolist()}")
        
        # Analyze tour
        tour = instance['tour'].numpy()
        demands = instance['demand'].numpy()
        capacity = instance['capacity'].item()
        
        print(f"\n🔍 Tour analysis:")
        depot_returns = (tour == 0).sum()
        print(f"  - Number of routes (depot returns): {depot_returns}")
        print(f"  - Total demand: {demands.sum():.1f}")
        print(f"  - Average demand per route: {demands.sum() / depot_returns:.1f}")
        print(f"  - Vehicle capacity: {capacity:.1f}")
        
        # Validate tour feasibility
        current_load = 0.0
        is_feasible = True
        for node in tour:
            if node == 0:  # Depot
                current_load = 0.0
            else:  # Customer
                current_load += demands[node - 1]  # tour may be 0-indexed with depot at 0
                if current_load > capacity:
                    is_feasible = False
                    break
        print(f"  - Tour feasibility: {'✅ Feasible' if is_feasible else '❌ Infeasible'}")
        
        # Test batch loading with custom collate function
        print(f"\n📦 Batch loading test:")
        from torch.utils.data import DataLoader
        from rl4co.data.ml4co_dataset import cvrp_collate_fn
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=cvrp_collate_fn)
        batch = next(iter(dataloader))
        print(f"  - Batch depot shape: {batch['depot'].shape}")
        print(f"  - Batch locs shape: {batch['locs'].shape}")
        print(f"  - Batch demand shape: {batch['demand'].shape}")
        print(f"  - Batch capacity shape: {batch['capacity'].shape}")
        print(f"  - Number of tours in batch: {len(batch['tours'])}")
        print(f"  - Tour lengths: {[len(t) for t in batch['tours']]}")
        
        print("\n✅ CVRP dataset test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ CVRP dataset test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_statistics():
    """Compute and display statistics for loaded datasets."""
    print("\n" + "="*80)
    print("TEST 3: Dataset Statistics")
    print("="*80)
    
    try:
        # TSP statistics
        tsp_path = "/root/autodl-tmp/ML4CO-Kit/test_dataset/tsp/wrapper/tsp50_uniform_16ins.txt"
        if Path(tsp_path).exists():
            print("\n📊 TSP Dataset Statistics:")
            dataset = load_ml4co_dataset('tsp', tsp_path)
            
            all_costs = []
            for instance in dataset:
                locs = instance['locs'].numpy()
                tour = instance['tour'].numpy()
                cost = 0.0
                for i in range(len(tour)):
                    j = (i + 1) % len(tour)
                    cost += np.linalg.norm(locs[tour[i]] - locs[tour[j]])
                all_costs.append(cost)
            
            print(f"  - Number of instances: {len(dataset)}")
            print(f"  - Problem size: {len(dataset[0]['locs'])} nodes")
            print(f"  - Optimal cost range: [{min(all_costs):.4f}, {max(all_costs):.4f}]")
            print(f"  - Mean optimal cost: {np.mean(all_costs):.4f}")
            print(f"  - Std optimal cost: {np.std(all_costs):.4f}")
        
        # CVRP statistics
        cvrp_path = "/root/autodl-tmp/ML4CO-Kit/test_dataset/cvrp/wrapper/cvrp50_uniform_16ins.txt"
        if Path(cvrp_path).exists():
            print("\n📊 CVRP Dataset Statistics:")
            dataset = load_ml4co_dataset('cvrp', cvrp_path)
            
            all_num_routes = []
            all_demands = []
            for instance in dataset:
                tour = instance['tour'].numpy()
                depot_returns = (tour == 0).sum()
                all_num_routes.append(depot_returns)
                all_demands.append(instance['demand'].sum().item())
            
            print(f"  - Number of instances: {len(dataset)}")
            print(f"  - Problem size: {len(dataset[0]['locs'])} customers")
            print(f"  - Capacity: {dataset[0]['capacity'].item():.1f}")
            print(f"  - Routes per instance: [{min(all_num_routes)}, {max(all_num_routes)}]")
            print(f"  - Mean routes: {np.mean(all_num_routes):.2f}")
            print(f"  - Total demand range: [{min(all_demands):.1f}, {max(all_demands):.1f}]")
            print(f"  - Mean total demand: {np.mean(all_demands):.1f}")
        
        print("\n✅ Dataset statistics test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Dataset statistics test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_env_integration():
    """Test integration with RL4CO environments."""
    print("\n" + "="*80)
    print("TEST 4: RL4CO Environment Integration")
    print("="*80)
    
    tsp_path = "/root/autodl-tmp/ML4CO-Kit/test_dataset/tsp/wrapper/tsp50_uniform_16ins.txt"
    
    if not Path(tsp_path).exists():
        print(f"⚠️  Dataset not found: {tsp_path}")
        return
    
    try:
        print("\n🔧 Testing TSP environment integration:")
        dataset = load_ml4co_dataset('tsp', tsp_path)
        instance = dataset[0]
        
        # Create TSP environment
        env = TSPEnv()
        
        # Create a batch with the instance
        locs = instance['locs'].unsqueeze(0)  # Add batch dimension
        print(f"  - Instance locs shape: {locs.shape}")
        
        # Reset environment with the loaded data
        td = env.reset(batch_size=[1])
        td['locs'] = locs
        
        print(f"  - Environment state initialized")
        print(f"  - State locs shape: {td['locs'].shape}")
        print(f"  - Action mask shape: {td['action_mask'].shape}")
        
        print("\n✅ Environment integration test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Environment integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("ML4CO-Kit Dataset Reader - Comprehensive Test Suite")
    print("="*80)
    
    results = {
        'TSP Dataset': test_tsp_dataset(),
        'CVRP Dataset': test_cvrp_dataset(),
        'Dataset Statistics': test_dataset_statistics(),
        'Environment Integration': test_env_integration()
    }
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<50} {status}")
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed successfully!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")


if __name__ == "__main__":
    main()
