"""Test script for Facility Location Problem (FLP) environment"""

import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
from rl4co.envs.graph.flp import FLPEnv
from rl4co.utils.decoding import random_policy, rollout


def visualize_solution(td, actions, batch_idx=0, save_path=None):
    """Visualize the FLP solution
    
    Args:
        td: TensorDict with final state
        actions: actions taken (indices of chosen facilities)
        batch_idx: which batch instance to visualize
        save_path: path to save figure (optional)
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Get data for this batch
    locs = td["locs"][batch_idx].cpu().numpy()
    chosen = td["chosen"][batch_idx].cpu().numpy()
    distances = td["distances"][batch_idx].cpu().numpy()
    orig_distances = td["orig_distances"][batch_idx].cpu().numpy()
    
    num_loc = locs.shape[0]
    chosen_indices = np.where(chosen)[0]
    non_chosen_indices = np.where(~chosen)[0]
    
    # 1. Draw connections from each location to its nearest facility
    for i in range(num_loc):
        if not chosen[i]:  # Only draw connections for non-facility locations
            # Find nearest chosen facility
            min_dist = float('inf')
            nearest_facility = None
            for facility_idx in chosen_indices:
                dist = orig_distances[i, facility_idx]
                if dist < min_dist:
                    min_dist = dist
                    nearest_facility = facility_idx
            
            if nearest_facility is not None:
                ax.plot([locs[i, 0], locs[nearest_facility, 0]], 
                       [locs[i, 1], locs[nearest_facility, 1]], 
                       'gray', alpha=0.3, linewidth=0.8, linestyle='--', zorder=1)
    
    # 2. Draw non-chosen locations (customers) as small blue dots
    if len(non_chosen_indices) > 0:
        ax.scatter(locs[non_chosen_indices, 0], locs[non_chosen_indices, 1], 
                  c='lightblue', s=50, edgecolors='blue', linewidths=1, 
                  zorder=2, alpha=0.6, label='Customer locations')
    
    # 3. Draw chosen locations (facilities) as large red stars
    ax.scatter(locs[chosen_indices, 0], locs[chosen_indices, 1], 
              c='red', s=400, marker='*', edgecolors='darkred', linewidths=2, 
              zorder=4, label='Facilities')
    
    # Add facility indices
    for idx in chosen_indices:
        ax.text(locs[idx, 0], locs[idx, 1] + 0.03, f'F{idx}', 
               ha='center', va='bottom', fontsize=9, fontweight='bold', 
               color='darkred', zorder=5)
    
    # 4. Calculate and display statistics
    total_cost = distances.sum()
    avg_dist = distances.mean()
    max_dist = distances.max()
    
    # 5. Add title and info
    ax.set_title(f'Facility Location Problem Solution (Batch {batch_idx})\n'
                f'Facilities: {len(chosen_indices)}/{num_loc} | '
                f'Total Cost: {total_cost:.4f} | Avg Dist: {avg_dist:.4f} | Max Dist: {max_dist:.4f}',
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
    
    return total_cost


def test_flp_env():
    """Test FLP environment functionality"""
    print("=" * 50)
    print("Testing Facility Location Problem Environment")
    print("=" * 50)
    
    # Create environment with small problem size for testing
    generator_params = {
        "num_loc": 50,  # Total number of locations
        "num_keep": 10,  # Number of facilities to select
    }
    
    env = FLPEnv(generator_params=generator_params)
    print(f"\n✓ Environment created: {env.name}")
    print(f"  - Number of locations: {generator_params['num_loc']}")
    print(f"  - Number of facilities to choose: {generator_params['num_keep']}")
    
    # Generate some instances
    batch_size = 4
    td = env.reset(batch_size=[batch_size])
    print(f"\n✓ Generated {batch_size} problem instances")
    print(f"  - Locations shape: {td['locs'].shape}")
    print(f"  - Original distances shape: {td['orig_distances'].shape}")
    print(f"  - To choose: {td['to_choose'][0].item()}")
    
    # Test random policy rollout
    print("\n✓ Testing random policy rollout...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    td = td.to(device)
    
    try:
        reward, td_final, actions = rollout(env, td, random_policy)
        print(f"  - Rollout completed successfully")
        print(f"  - Rewards shape: {reward.shape}")
        print(f"  - Actions shape: {actions.shape}")
        print(f"  - Mean reward: {reward.mean().item():.4f}")
        print(f"  - Reward range: [{reward.min().item():.4f}, {reward.max().item():.4f}]")
        
        # Verify solution properties
        print("\n✓ Verifying solution properties...")
        for b in range(batch_size):
            num_chosen = td_final["chosen"][b].sum().item()
            expected_facilities = td_final["to_choose"][b].item()
            total_cost = td_final["distances"][b].sum().item()
            actual_reward = reward[b].item()
            
            print(f"  Batch {b}:")
            print(f"    - Facilities chosen: {num_chosen}/{expected_facilities}")
            print(f"    - Total cost: {total_cost:.4f}")
            print(f"    - Reward: {actual_reward:.4f}")
            print(f"    - Match: {abs(total_cost + actual_reward) < 1e-4}")
            
            # Check constraint satisfaction
            assert num_chosen == expected_facilities, \
                f"Batch {b}: Expected {expected_facilities} facilities, got {num_chosen}"
        
        print("\n✓ All constraints satisfied!")
        
        # Visualize solutions
        print("\n✓ Visualizing solutions...")
        for b in range(min(2, batch_size)):  # Visualize first 2 instances
            print(f"\n  Rendering batch {b}...")
            visualize_solution(td_final, actions, batch_idx=b, 
                             save_path=f"flp_solution_batch_{b}.png")
        
    except Exception as e:
        print(f"  ✗ Error during rollout: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
    return True


if __name__ == "__main__":
    test_flp_env()
