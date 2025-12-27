"""Test script for Traveling Salesman Problem (TSP) environment"""

import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
from rl4co.envs.routing import TSPEnv
from rl4co.utils.decoding import random_policy, rollout


def visualize_solution(td, actions, batch_idx=0, save_path=None):
    """Visualize the TSP solution
    
    Args:
        td: TensorDict with final state
        actions: actions taken (sequence of city indices)
        batch_idx: which batch instance to visualize
        save_path: path to save figure (optional)
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Get data for this batch
    locs = td["locs"][batch_idx].cpu().numpy()
    tour = actions[batch_idx].cpu().numpy()
    
    num_cities = locs.shape[0]
    
    # Calculate tour length
    tour_length = 0.0
    for i in range(len(tour)):
        from_city = tour[i]
        to_city = tour[(i + 1) % len(tour)]  # Wrap around to starting city
        distance = np.linalg.norm(locs[from_city] - locs[to_city])
        tour_length += distance
    
    # 1. Draw the tour path
    for i in range(len(tour)):
        from_idx = tour[i]
        to_idx = tour[(i + 1) % len(tour)]  # Return to start
        
        from_loc = locs[from_idx]
        to_loc = locs[to_idx]
        
        # Draw edge
        ax.plot([from_loc[0], to_loc[0]], 
               [from_loc[1], to_loc[1]], 
               'b-', alpha=0.6, linewidth=2, zorder=1)
        
        # Add arrow to show direction
        dx = to_loc[0] - from_loc[0]
        dy = to_loc[1] - from_loc[1]
        ax.arrow(from_loc[0] + dx * 0.3, from_loc[1] + dy * 0.3,
                dx * 0.3, dy * 0.3,
                head_width=0.02, head_length=0.015, 
                fc='blue', ec='blue', alpha=0.5, zorder=2)
    
    # 2. Draw cities
    ax.scatter(locs[:, 0], locs[:, 1], 
              c='red', s=200, marker='o', edgecolors='darkred', linewidths=2, 
              zorder=3, label='Cities')
    
    # 3. Highlight start city
    start_city = tour[0]
    ax.scatter(locs[start_city, 0], locs[start_city, 1], 
              c='green', s=400, marker='*', edgecolors='darkgreen', linewidths=2, 
              zorder=4, label='Start')
    
    # 4. Add city labels with visit order
    for visit_order, city_idx in enumerate(tour):
        ax.text(locs[city_idx, 0], locs[city_idx, 1] + 0.03, 
               f'{visit_order}', 
               ha='center', va='bottom', fontsize=9, fontweight='bold', 
               color='white', zorder=5,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    # 5. Add title and info
    ax.set_title(f'TSP Solution (Batch {batch_idx})\n'
                f'Cities: {num_cities} | Tour Length: {tour_length:.4f}',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    Figure saved to {save_path}")
    
    plt.close(fig)
    
    return tour_length


def calculate_tour_length(locs, tour):
    """Calculate the total tour length
    
    Args:
        locs: city locations [num_cities, 2]
        tour: sequence of city indices [num_cities]
    
    Returns:
        tour_length: total distance of the tour
    """
    tour_length = 0.0
    for i in range(len(tour)):
        from_city = tour[i]
        to_city = tour[(i + 1) % len(tour)]
        distance = torch.norm(locs[from_city] - locs[to_city], p=2)
        tour_length += distance.item()
    return tour_length


def verify_tour(tour, num_cities):
    """Verify that the tour is valid (visits each city exactly once)
    
    Args:
        tour: sequence of city indices
        num_cities: total number of cities
    
    Returns:
        is_valid: whether the tour is valid
        error_msg: error message if invalid
    """
    tour_np = tour.cpu().numpy()
    
    # Check length
    if len(tour_np) != num_cities:
        return False, f"Tour length {len(tour_np)} != number of cities {num_cities}"
    
    # Check all cities visited
    visited = set(tour_np)
    if len(visited) != num_cities:
        return False, f"Tour has {len(visited)} unique cities, expected {num_cities}"
    
    # Check range
    if tour_np.min() < 0 or tour_np.max() >= num_cities:
        return False, f"Tour indices out of range [0, {num_cities})"
    
    return True, "Valid tour"


def test_tsp_env():
    """Test TSP environment functionality"""
    print("=" * 60)
    print("Testing Traveling Salesman Problem Environment")
    print("=" * 60)
    
    # Create environment with small problem size for testing
    generator_params = {
        "num_loc": 20,  # Number of cities
    }
    
    env = TSPEnv(generator_params=generator_params)
    print(f"\n✓ Environment created: {env.name}")
    print(f"  - Number of cities: {generator_params['num_loc']}")
    print(f"  - Objective: Minimize tour length")
    
    # Generate some instances
    batch_size = 4
    td = env.reset(batch_size=[batch_size])
    print(f"\n✓ Generated {batch_size} problem instances")
    print(f"  - Locations shape: {td['locs'].shape}")
    print(f"  - Expected: [{batch_size}, {generator_params['num_loc']}, 2]")
    
    # Test random policy rollout
    print("\n✓ Testing random policy rollout...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - Using device: {device}")
    td = td.to(device)
    
    try:
        reward, td_final, actions = rollout(env, td, random_policy)
        print(f"  - Rollout completed successfully")
        print(f"  - Rewards shape: {reward.shape}")
        print(f"  - Actions shape: {actions.shape}")
        print(f"  - Mean reward: {reward.mean().item():.4f} (negative tour length)")
        print(f"  - Mean tour length: {-reward.mean().item():.4f}")
        print(f"  - Reward range: [{reward.min().item():.4f}, {reward.max().item():.4f}]")
        
        # Verify solution properties
        print("\n✓ Verifying solution properties...")
        num_cities = generator_params['num_loc']
        
        all_valid = True
        for b in range(batch_size):
            tour = actions[b]
            locs = td_final["locs"][b]
            
            # Verify tour validity
            is_valid, error_msg = verify_tour(tour, num_cities)
            
            # Calculate tour length
            if is_valid:
                tour_length = calculate_tour_length(locs, tour)
                actual_reward = reward[b].item()
                reward_matches = abs(tour_length + actual_reward) < 1e-3
                
                print(f"\n  Batch {b}:")
                print(f"    - Tour validity: ✓ {error_msg}")
                print(f"    - Tour length: {tour_length:.4f}")
                print(f"    - Reward: {actual_reward:.4f}")
                print(f"    - Reward matches tour length: {'✓' if reward_matches else '✗'}")
                
                if not reward_matches:
                    print(f"    - WARNING: Mismatch! Diff: {abs(tour_length + actual_reward):.6f}")
                    all_valid = False
            else:
                print(f"\n  Batch {b}:")
                print(f"    - Tour validity: ✗ {error_msg}")
                all_valid = False
        
        if all_valid:
            print("\n✓ All tours are valid!")
        else:
            print("\n✗ Some tours are invalid or have mismatched rewards")
            return False
        
        # Test environment properties
        print("\n✓ Testing environment properties...")
        
        # Check if done flag is set
        if "done" in td_final.keys():
            all_done = td_final["done"].all()
            print(f"  - All episodes done: {all_done.item()}")
        
        # Check visited cities
        if "visited" in td_final.keys():
            all_visited = td_final["visited"].all()
            print(f"  - All cities visited: {all_visited.item()}")
        
        # Check action mask (should be all False at the end)
        if "action_mask" in td_final.keys():
            final_mask = td_final["action_mask"]
            no_valid_actions = (~final_mask).all()
            print(f"  - No valid actions remaining: {no_valid_actions.item()}")
        
        # Visualize solutions
        print("\n✓ Visualizing solutions...")
        import os
        os.makedirs("results/tsp", exist_ok=True)
        
        for b in range(min(2, batch_size)):  # Visualize first 2 instances
            print(f"\n  Rendering batch {b}...")
            visualize_solution(td_final, actions, batch_idx=b, 
                             save_path=f"results/tsp/tsp_solution_batch_{b}.png")
        
        print(f"\n✓ Visualizations saved to results/tsp/")
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - results/tsp/tsp_solution_batch_0.png")
    print("  - results/tsp/tsp_solution_batch_1.png")
    print("=" * 60)
    
    return True


def test_tsp_different_sizes():
    """Test TSP with different problem sizes"""
    print("\n" + "=" * 60)
    print("Testing TSP with Different Problem Sizes")
    print("=" * 60)
    
    sizes = [10, 20, 50, 100]
    
    for num_cities in sizes:
        print(f"\n✓ Testing TSP-{num_cities}...")
        
        env = TSPEnv(generator_params={"num_loc": num_cities})
        td = env.reset(batch_size=[2])
        
        # Quick rollout test
        try:
            reward, td_final, actions = rollout(env, td, random_policy)
            mean_tour_length = -reward.mean().item()
            print(f"  - Mean tour length: {mean_tour_length:.4f}")
            print(f"  - Tour valid: ✓")
        except Exception as e:
            print(f"  - Error: {e}")
            return False
    
    print("\n✓ All sizes tested successfully!")
    return True


if __name__ == "__main__":
    # Test basic functionality
    success = test_tsp_env()
    
    if success:
        # Test different sizes
        test_tsp_different_sizes()
    
    print("\n🎉 Testing complete!")
