"""
TSP Benchmark Test: Compare RL and Classical Solvers on ML4CO-Kit Dataset

This script evaluates all available TSP solvers (including RL policies) 
on ML4CO-Kit benchmark datasets and computes performance gaps.
"""

import torch
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple
from rl4co.data.ml4co_dataset import load_ml4co_dataset
from rl4co.envs.routing.tsp.solvers import solve_tsp
from rl4co.envs import TSPEnv


def compute_tour_cost(locs: np.ndarray, tour: np.ndarray) -> float:
    """Compute the total cost of a tour."""
    cost = 0.0
    for i in range(len(tour)):
        j = (i + 1) % len(tour)
        cost += np.linalg.norm(locs[tour[i]] - locs[tour[j]])
    return cost


def test_solver(
    solver_name: str,
    dataset,
    max_instances: int = None,
    verbose: bool = True
) -> Dict:
    """
    Test a single solver on the dataset.
    
    Args:
        solver_name: Name of the solver
        dataset: ML4CO dataset
        max_instances: Maximum number of instances to test (None = all)
        verbose: Print progress
    
    Returns:
        Dictionary with results
    """
    results = {
        'solver': solver_name,
        'gaps': [],
        'times': [],
        'costs': [],
        'optimal_costs': [],
        'num_success': 0,
        'num_failed': 0
    }
    
    num_instances = min(len(dataset), max_instances) if max_instances else len(dataset)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing {solver_name.upper()} on {num_instances} instances")
        print(f"{'='*60}")
    
    for i in range(num_instances):
        instance = dataset[i]
        locs = instance['locs'].numpy()
        optimal_tour = instance['tour'].numpy()
        
        # Compute optimal cost
        optimal_cost = compute_tour_cost(locs, optimal_tour)
        
        try:
            # Run solver
            start_time = time.time()
            tour, cost, info = solve_tsp(
                locs, 
                method=solver_name,
                verbose=False
            )
            solve_time = time.time() - start_time
            
            if tour is not None and cost is not None:
                # Compute gap
                gap = (cost / optimal_cost - 1) * 100
                
                results['gaps'].append(gap)
                results['times'].append(solve_time)
                results['costs'].append(cost)
                results['optimal_costs'].append(optimal_cost)
                results['num_success'] += 1
                
                if verbose and i < 3:  # Print first 3 instances
                    print(f"  Instance {i}: "
                          f"Optimal={optimal_cost:.4f}, "
                          f"Solver={cost:.4f}, "
                          f"Gap={gap:.2f}%, "
                          f"Time={solve_time:.3f}s")
            else:
                results['num_failed'] += 1
                if verbose:
                    print(f"  Instance {i}: FAILED (returned None)")
                    
        except Exception as e:
            results['num_failed'] += 1
            if verbose:
                print(f"  Instance {i}: ERROR - {e}")
    
    # Compute statistics
    if results['gaps']:
        results['mean_gap'] = np.mean(results['gaps'])
        results['std_gap'] = np.std(results['gaps'])
        results['min_gap'] = np.min(results['gaps'])
        results['max_gap'] = np.max(results['gaps'])
        results['mean_time'] = np.mean(results['times'])
        results['total_time'] = np.sum(results['times'])
    
    return results


def test_rl_policy(
    policy,
    dataset,
    max_instances: int = None,
    verbose: bool = True
) -> Dict:
    """
    Test an RL policy on the dataset.
    
    Args:
        policy: RL4CO policy
        dataset: ML4CO dataset
        max_instances: Maximum number of instances to test
        verbose: Print progress
    
    Returns:
        Dictionary with results
    """
    results = {
        'solver': 'RL Policy',
        'gaps': [],
        'times': [],
        'costs': [],
        'optimal_costs': [],
        'num_success': 0,
        'num_failed': 0
    }
    
    num_instances = min(len(dataset), max_instances) if max_instances else len(dataset)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing RL POLICY on {num_instances} instances")
        print(f"{'='*60}")
    
    # Get problem size from first instance
    first_instance = dataset[0]
    problem_size = len(first_instance['locs'])
    
    # Create environment with matching problem size
    env = TSPEnv(generator_params={'num_loc': problem_size})
    
    for i in range(num_instances):
        instance = dataset[i]
        locs = instance['locs']
        optimal_tour = instance['tour'].numpy()
        
        # Compute optimal cost
        optimal_cost = compute_tour_cost(locs.numpy(), optimal_tour)
        
        try:
            # Create environment state
            td = env.reset(batch_size=[1])
            td['locs'] = locs.unsqueeze(0)  # Add batch dimension
            
            # Run policy
            start_time = time.time()
            with torch.no_grad():
                out = policy(td, phase="test", decode_type="greedy", return_actions=True)
            solve_time = time.time() - start_time
            
            # Extract solution
            if isinstance(out, dict) and 'actions' in out:
                actions = out['actions'][0].cpu().numpy()
            else:
                actions = out[0][0].cpu().numpy()
            
            # Compute cost
            cost = compute_tour_cost(locs.numpy(), actions)
            
            # Compute gap
            gap = (cost / optimal_cost - 1) * 100
            
            results['gaps'].append(gap)
            results['times'].append(solve_time)
            results['costs'].append(cost)
            results['optimal_costs'].append(optimal_cost)
            results['num_success'] += 1
            
            if verbose and i < 3:
                print(f"  Instance {i}: "
                      f"Optimal={optimal_cost:.4f}, "
                      f"RL={cost:.4f}, "
                      f"Gap={gap:.2f}%, "
                      f"Time={solve_time:.3f}s")
                      
        except Exception as e:
            results['num_failed'] += 1
            if verbose:
                print(f"  Instance {i}: ERROR - {e}")
    
    # Compute statistics
    if results['gaps']:
        results['mean_gap'] = np.mean(results['gaps'])
        results['std_gap'] = np.std(results['gaps'])
        results['min_gap'] = np.min(results['gaps'])
        results['max_gap'] = np.max(results['gaps'])
        results['mean_time'] = np.mean(results['times'])
        results['total_time'] = np.sum(results['times'])
    
    return results


def print_summary(all_results: List[Dict], test_type: str = "Test"):
    """Print summary table of all results."""
    print(f"\n{'='*80}")
    print(f"{test_type} SUMMARY - TSP Solver Comparison")
    print(f"{'='*80}")
    
    # Header
    print(f"\n{'Solver':<20} {'Mean Gap':<12} {'Std Gap':<12} {'Min Gap':<12} "
          f"{'Max Gap':<12} {'Avg Time':<12} {'Success':<10}")
    print(f"{'-'*98}")
    
    # Sort by mean gap
    all_results_sorted = sorted(
        [r for r in all_results if 'mean_gap' in r],
        key=lambda x: x['mean_gap']
    )
    
    for result in all_results_sorted:
        print(f"{result['solver']:<20} "
              f"{result['mean_gap']:>10.2f}% "
              f"{result['std_gap']:>10.2f}% "
              f"{result['min_gap']:>10.2f}% "
              f"{result['max_gap']:>10.2f}% "
              f"{result['mean_time']:>10.4f}s "
              f"{result['num_success']:>3}/{result['num_success'] + result['num_failed']:<3}")
    
    # Print failed solvers
    failed_solvers = [r for r in all_results if 'mean_gap' not in r]
    if failed_solvers:
        print(f"\n{'Failed Solvers:':<20}")
        for result in failed_solvers:
            print(f"  {result['solver']:<18} - Failed on all instances")


def quick_test(dataset_path: str, test_rl: bool = True):
    """
    Quick test on a small subset of instances.
    
    Args:
        dataset_path: Path to ML4CO dataset file
        test_rl: Whether to test RL policy
    """
    print("\n" + "="*80)
    print("QUICK TEST - Testing 3 instances per solver")
    print("="*80)
    
    # Load dataset
    print(f"\nLoading dataset: {dataset_path}")
    dataset = load_ml4co_dataset('tsp', dataset_path)
    print(f"Dataset size: {len(dataset)} instances")
    
    # Test configuration
    max_instances = 3
    
    # Available solvers (fast ones for quick test)
    solvers = ['greedy']  # Only greedy for quick test
    
    all_results = []
    
    # Test each solver
    for solver in solvers:
        result = test_solver(
            solver,
            dataset,
            max_instances=max_instances,
            verbose=True
        )
        all_results.append(result)
    
    # Test RL policy if requested
    if test_rl:
        print("\n" + "="*60)
        print("Testing RL Policy (Greedy Decoding)")
        print("="*60)
        try:
            from rl4co.models import AttentionModel
            from rl4co.utils.trainer import RL4COTrainer
            
            # Create a baseline policy (untrained, greedy decoding)
            print("Loading baseline RL policy...")
            env = TSPEnv()
            policy = AttentionModel(env)
            policy.eval()
            
            rl_result = test_rl_policy(
                policy,
                dataset,
                max_instances=max_instances,
                verbose=True
            )
            all_results.append(rl_result)
        except Exception as e:
            print(f"⚠️  RL policy test skipped: {e}")
            print("Tip: RL policy requires a trained model or uses random baseline")
    
    # Print summary
    print_summary(all_results, "QUICK TEST")
    
    # Check if any solver worked
    working_solvers = [r for r in all_results if 'mean_gap' in r]
    
    if working_solvers:
        print(f"\n✅ Quick test PASSED - {len(working_solvers)}/{len(solvers)} solvers working")
        return True
    else:
        print(f"\n❌ Quick test FAILED - No solvers working")
        return False


def complete_test(dataset_path: str, output_file: str = None, test_rl: bool = True):
    """
    Complete test on all instances.
    
    Args:
        dataset_path: Path to ML4CO dataset file
        output_file: Optional file to save results
        test_rl: Whether to test RL policy
    """
    print("\n" + "="*80)
    print("COMPLETE TEST - Testing all instances")
    print("="*80)
    
    # Load dataset
    print(f"\nLoading dataset: {dataset_path}")
    dataset = load_ml4co_dataset('tsp', dataset_path)
    print(f"Dataset size: {len(dataset)} instances")
    
    # Available solvers
    solvers = ['greedy', 'ga', 'lkh', 'concorde']
    
    print(f"\nTesting classical solvers: {solvers}")
    
    all_results = []
    
    # Test each solver
    for solver in solvers:
        result = test_solver(
            solver,
            dataset,
            max_instances=None,  # Test all instances
            verbose=True
        )
        all_results.append(result)
    
    # Test RL policy
    if test_rl:
        print("\n" + "="*80)
        print("Testing RL Policy")
        print("="*80)
        try:
            from rl4co.models import AttentionModel
            
            print("\n⚠️  Note: Using UNTRAINED baseline policy")
            print("For better results, load a trained checkpoint:")
            print("  policy = AttentionModel.load_from_checkpoint('path/to/ckpt')")
            
            env = TSPEnv()
            policy = AttentionModel(env)
            policy.eval()
            
            rl_result = test_rl_policy(
                policy,
                dataset,
                max_instances=None,
                verbose=True
            )
            all_results.append(rl_result)
        except Exception as e:
            print(f"⚠️  RL policy test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print_summary(all_results, "COMPLETE TEST")
    
    # Save results if requested
    if output_file:
        save_results(all_results, dataset_path, output_file)
    
    return all_results


def save_results(results: List[Dict], dataset_path: str, output_file: str):
    """Save results to a markdown file."""
    with open(output_file, 'w') as f:
        f.write("# TSP Benchmark Results\n\n")
        f.write(f"**Dataset**: {dataset_path}\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary table
        f.write("## Summary\n\n")
        f.write("| Solver | Mean Gap | Std Gap | Min Gap | Max Gap | Avg Time | Success Rate |\n")
        f.write("|--------|----------|---------|---------|---------|----------|-------------|\n")
        
        results_sorted = sorted(
            [r for r in results if 'mean_gap' in r],
            key=lambda x: x['mean_gap']
        )
        
        for r in results_sorted:
            total = r['num_success'] + r['num_failed']
            success_rate = (r['num_success'] / total * 100) if total > 0 else 0
            f.write(f"| {r['solver']} | "
                   f"{r['mean_gap']:.2f}% | "
                   f"{r['std_gap']:.2f}% | "
                   f"{r['min_gap']:.2f}% | "
                   f"{r['max_gap']:.2f}% | "
                   f"{r['mean_time']:.4f}s | "
                   f"{success_rate:.1f}% |\n")
        
        # Best solver
        if results_sorted:
            best = results_sorted[0]
            f.write(f"\n**Best Solver**: {best['solver']} "
                   f"(Mean Gap: {best['mean_gap']:.2f}%)\n")
    
    print(f"\n✅ Results saved to: {output_file}")


def main():
    """Main test execution."""
    print("="*80)
    print("TSP BENCHMARK: RL vs Classical Solvers on ML4CO-Kit Dataset")
    print("="*80)
    
    # Dataset path
    dataset_path = "/root/autodl-tmp/ML4CO-Kit/test_dataset/tsp/wrapper/tsp50_uniform_16ins.txt"
    
    if not Path(dataset_path).exists():
        print(f"\n❌ Dataset not found: {dataset_path}")
        print("Please provide a valid dataset path.")
        return
    
    # Step 1: Quick test
    print("\n" + "="*80)
    print("STEP 1: QUICK TEST")
    print("="*80)
    
    quick_passed = quick_test(dataset_path)
    
    if not quick_passed:
        print("\n⚠️  Quick test failed. Please check solver availability.")
        print("Note: Some solvers (LKH, Concorde, Gurobi, OR-Tools) require installation.")
        return
    
    # Step 2: Complete test
    print("\n" + "="*80)
    print("STEP 2: COMPLETE TEST")
    print("="*80)
    
    proceed = input("\nProceed with complete test? (y/n): ").strip().lower()
    
    if proceed == 'y':
        output_file = "TSP_BENCHMARK_RESULTS.md"
        results = complete_test(dataset_path, output_file)
        
        print("\n" + "="*80)
        print("✅ BENCHMARK COMPLETE")
        print("="*80)
        print(f"\nResults saved to: {output_file}")
        
        # Print key insights
        working_results = [r for r in results if 'mean_gap' in r]
        if working_results:
            best = min(working_results, key=lambda x: x['mean_gap'])
            fastest = min(working_results, key=lambda x: x['mean_time'])
            
            print(f"\n📊 Key Insights:")
            print(f"  - Best quality: {best['solver']} ({best['mean_gap']:.2f}% gap)")
            print(f"  - Fastest: {fastest['solver']} ({fastest['mean_time']:.4f}s avg)")
    else:
        print("\n❌ Complete test cancelled.")


if __name__ == "__main__":
    main()
