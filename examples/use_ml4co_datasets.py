"""
Example: Using ML4CO-Kit Datasets with RL4CO

This script demonstrates how to load and use ML4CO-Kit test datasets
for evaluating TSP and CVRP solvers.
"""

import torch
import numpy as np
from pathlib import Path
from rl4co.data.ml4co_dataset import load_ml4co_dataset, cvrp_collate_fn
from rl4co.envs.routing.tsp.solvers import solve_tsp
from torch.utils.data import DataLoader


def example_1_basic_loading():
    """Example 1: Basic dataset loading"""
    print("\n" + "="*80)
    print("Example 1: Basic Dataset Loading")
    print("="*80)
    
    # Load TSP dataset
    tsp_dataset = load_ml4co_dataset(
        'tsp',
        '/root/autodl-tmp/ML4CO-Kit/test_dataset/tsp/wrapper/tsp50_uniform_16ins.txt'
    )
    
    print(f"\nLoaded {len(tsp_dataset)} TSP instances")
    
    # Access first instance
    instance = tsp_dataset[0]
    print(f"Instance 0:")
    print(f"  - Locations shape: {instance['locs'].shape}")
    print(f"  - Tour shape: {instance['tour'].shape}")
    print(f"  - First 3 locations:\n{instance['locs'][:3]}")


def example_2_evaluate_solver():
    """Example 2: Evaluate a solver against optimal solutions"""
    print("\n" + "="*80)
    print("Example 2: Evaluate Solver Performance")
    print("="*80)
    
    # Load dataset
    dataset = load_ml4co_dataset(
        'tsp',
        '/root/autodl-tmp/ML4CO-Kit/test_dataset/tsp/wrapper/tsp50_uniform_16ins.txt'
    )
    
    print(f"\nEvaluating greedy solver on {len(dataset)} instances...")
    
    gaps = []
    optimal_costs = []
    solver_costs = []
    
    for i, instance in enumerate(dataset):
        locs = instance['locs'].numpy()
        optimal_tour = instance['tour'].numpy()
        
        # Calculate optimal cost
        opt_cost = 0.0
        for j in range(len(optimal_tour)):
            k = (j + 1) % len(optimal_tour)
            opt_cost += np.linalg.norm(locs[optimal_tour[j]] - locs[optimal_tour[k]])
        
        # Run greedy solver
        tour, cost, info = solve_tsp(locs, method='greedy', verbose=False)
        
        # Calculate gap
        gap = (cost / opt_cost - 1) * 100
        gaps.append(gap)
        optimal_costs.append(opt_cost)
        solver_costs.append(cost)
        
        if i < 3:  # Print first 3 instances
            print(f"\nInstance {i}:")
            print(f"  Optimal cost: {opt_cost:.4f}")
            print(f"  Greedy cost:  {cost:.4f}")
            print(f"  Gap:          {gap:.2f}%")
    
    print(f"\n{'='*40}")
    print(f"Overall Statistics:")
    print(f"  Mean gap:     {np.mean(gaps):.2f}%")
    print(f"  Std gap:      {np.std(gaps):.2f}%")
    print(f"  Min gap:      {np.min(gaps):.2f}%")
    print(f"  Max gap:      {np.max(gaps):.2f}%")
    print(f"  Mean optimal: {np.mean(optimal_costs):.4f}")


def example_3_batch_processing():
    """Example 3: Batch processing with DataLoader"""
    print("\n" + "="*80)
    print("Example 3: Batch Processing with DataLoader")
    print("="*80)
    
    # Load TSP dataset
    dataset = load_ml4co_dataset(
        'tsp',
        '/root/autodl-tmp/ML4CO-Kit/test_dataset/tsp/wrapper/tsp50_uniform_16ins.txt'
    )
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    print(f"\nDataLoader created with batch_size=4")
    print(f"Number of batches: {len(dataloader)}")
    
    # Process first batch
    batch = next(iter(dataloader))
    print(f"\nFirst batch:")
    print(f"  Locations shape: {batch['locs'].shape}")
    print(f"  Tours shape:     {batch['tour'].shape}")
    
    # Process all batches
    print(f"\nProcessing all batches...")
    for i, batch in enumerate(dataloader):
        print(f"  Batch {i}: {batch['locs'].shape}")


def example_4_cvrp_dataset():
    """Example 4: Working with CVRP datasets"""
    print("\n" + "="*80)
    print("Example 4: CVRP Dataset with Custom Collate")
    print("="*80)
    
    # Load CVRP dataset
    dataset = load_ml4co_dataset(
        'cvrp',
        '/root/autodl-tmp/ML4CO-Kit/test_dataset/cvrp/wrapper/cvrp50_uniform_16ins.txt'
    )
    
    print(f"\nLoaded {len(dataset)} CVRP instances")
    
    # Access individual instance
    instance = dataset[0]
    print(f"\nInstance 0:")
    print(f"  Depot:    {instance['depot'].tolist()}")
    print(f"  Customers: {instance['locs'].shape}")
    print(f"  Demands:   {instance['demand'].shape}")
    print(f"  Capacity:  {instance['capacity'].item():.1f}")
    print(f"  Tour length: {len(instance['tour'])}")
    
    # Analyze tour
    tour = instance['tour'].numpy()
    depot_returns = (tour == 0).sum()
    print(f"  Number of routes: {depot_returns}")
    
    # Batch processing with custom collate
    print(f"\nBatch processing with custom collate function:")
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=False,
        collate_fn=cvrp_collate_fn
    )
    
    batch = next(iter(dataloader))
    print(f"  Depot shape:    {batch['depot'].shape}")
    print(f"  Locs shape:     {batch['locs'].shape}")
    print(f"  Demand shape:   {batch['demand'].shape}")
    print(f"  Capacity shape: {batch['capacity'].shape}")
    print(f"  Tours (list):   {len(batch['tours'])} tours")
    print(f"  Tour lengths:   {[len(t) for t in batch['tours']]}")


def example_5_compare_multiple_solvers():
    """Example 5: Compare multiple solvers on benchmark"""
    print("\n" + "="*80)
    print("Example 5: Compare Multiple TSP Solvers")
    print("="*80)
    
    # Load dataset
    dataset = load_ml4co_dataset(
        'tsp',
        '/root/autodl-tmp/ML4CO-Kit/test_dataset/tsp/wrapper/tsp50_uniform_16ins.txt'
    )
    
    # Test multiple solvers
    solvers = ['greedy', 'ga']  # Can add 'lkh', 'concorde', 'gurobi', 'ortools' if available
    
    results = {solver: {'gaps': [], 'times': []} for solver in solvers}
    
    print(f"\nTesting {len(solvers)} solvers on {len(dataset)} instances...")
    
    for instance in dataset:
        locs = instance['locs'].numpy()
        optimal_tour = instance['tour'].numpy()
        
        # Calculate optimal cost
        opt_cost = 0.0
        for i in range(len(optimal_tour)):
            j = (i + 1) % len(optimal_tour)
            opt_cost += np.linalg.norm(locs[optimal_tour[i]] - locs[optimal_tour[j]])
        
        # Test each solver
        for solver in solvers:
            try:
                tour, cost, info = solve_tsp(locs, method=solver, verbose=False)
                gap = (cost / opt_cost - 1) * 100
                results[solver]['gaps'].append(gap)
                results[solver]['times'].append(info.get('solve_time', 0))
            except Exception as e:
                print(f"  Warning: {solver} failed - {e}")
    
    # Print results
    print(f"\n{'Solver':<15} {'Mean Gap':<12} {'Std Gap':<12} {'Mean Time':<12}")
    print(f"{'-'*51}")
    for solver in solvers:
        if results[solver]['gaps']:
            mean_gap = np.mean(results[solver]['gaps'])
            std_gap = np.std(results[solver]['gaps'])
            mean_time = np.mean(results[solver]['times'])
            print(f"{solver:<15} {mean_gap:>10.2f}% {std_gap:>10.2f}% {mean_time:>10.3f}s")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("ML4CO-Kit Dataset Usage Examples for RL4CO")
    print("="*80)
    
    # Run examples
    example_1_basic_loading()
    example_2_evaluate_solver()
    example_3_batch_processing()
    example_4_cvrp_dataset()
    example_5_compare_multiple_solvers()
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)


if __name__ == "__main__":
    main()
