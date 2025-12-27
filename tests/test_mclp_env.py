"""Test script for Maximum Covering Location Problem (MCLP) environment"""

import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from rl4co.envs.graph import MCLPEnv
from rl4co.utils.decoding import random_policy, rollout


def visualize_solution(td, actions, batch_idx=0, save_path=None):
    """Visualize the MCLP solution
    
    Args:
        td: TensorDict with final state
        actions: actions taken (indices of chosen facilities)
        batch_idx: which batch instance to visualize
        save_path: path to save figure (optional)
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Get data for this batch
    demand_locs = td["demand_locs"][batch_idx].cpu().numpy()
    facility_locs = td["facility_locs"][batch_idx].cpu().numpy()
    demand_weights = td["demand_weights"][batch_idx].cpu().numpy()
    chosen = td["chosen"][batch_idx].cpu().numpy()
    is_covered = td["is_covered"][batch_idx].cpu().numpy()
    coverage_radius = td["coverage_radius"][batch_idx].item()
    
    chosen_facilities = np.where(chosen)[0]
    
    # 1. Draw coverage circles for selected facilities
    for facility_idx in chosen_facilities:
        circle = mpatches.Circle(
            facility_locs[facility_idx],
            coverage_radius,
            color='lightblue',
            alpha=0.2,
            zorder=1
        )
        ax.add_patch(circle)
        
        # Draw circle boundary
        circle_boundary = mpatches.Circle(
            facility_locs[facility_idx],
            coverage_radius,
            fill=False,
            edgecolor='blue',
            linewidth=1.5,
            linestyle='--',
            alpha=0.5,
            zorder=2
        )
        ax.add_patch(circle_boundary)
    
    # 2. Draw demand points
    # Covered demands (green)
    covered_indices = np.where(is_covered)[0]
    uncovered_indices = np.where(~is_covered)[0]
    
    if len(covered_indices) > 0:
        scatter_covered = ax.scatter(
            demand_locs[covered_indices, 0],
            demand_locs[covered_indices, 1],
            c=demand_weights[covered_indices],
            s=200,
            cmap='Greens',
            edgecolors='darkgreen',
            linewidths=2,
            marker='o',
            alpha=0.8,
            zorder=4,
            vmin=demand_weights.min(),
            vmax=demand_weights.max(),
            label='Covered demands'
        )
    
    # Uncovered demands (red)
    if len(uncovered_indices) > 0:
        scatter_uncovered = ax.scatter(
            demand_locs[uncovered_indices, 0],
            demand_locs[uncovered_indices, 1],
            c=demand_weights[uncovered_indices],
            s=200,
            cmap='Reds',
            edgecolors='darkred',
            linewidths=2,
            marker='x',
            alpha=0.6,
            zorder=4,
            vmin=demand_weights.min(),
            vmax=demand_weights.max(),
            label='Uncovered demands'
        )
    
    # 3. Draw facilities
    # Unchosen facilities (gray)
    unchosen_facilities = np.where(~chosen)[0]
    if len(unchosen_facilities) > 0:
        ax.scatter(
            facility_locs[unchosen_facilities, 0],
            facility_locs[unchosen_facilities, 1],
            c='lightgray',
            s=150,
            marker='s',
            edgecolors='gray',
            linewidths=1.5,
            alpha=0.5,
            zorder=3,
            label='Candidate facilities'
        )
    
    # Chosen facilities (blue star)
    ax.scatter(
        facility_locs[chosen_facilities, 0],
        facility_locs[chosen_facilities, 1],
        c='blue',
        s=500,
        marker='*',
        edgecolors='darkblue',
        linewidths=2.5,
        zorder=5,
        label='Selected facilities'
    )
    
    # Add facility labels
    for idx in chosen_facilities:
        ax.text(
            facility_locs[idx, 0],
            facility_locs[idx, 1] - 0.05,
            f'F{idx}',
            ha='center',
            va='top',
            fontsize=9,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7, edgecolor='darkblue')
        )
    
    # Calculate statistics
    total_demand = demand_weights.sum()
    covered_demand = demand_weights[is_covered].sum()
    coverage_pct = (covered_demand / total_demand) * 100 if total_demand > 0 else 0
    num_covered = is_covered.sum()
    num_demand = len(demand_weights)
    
    # Title and labels
    ax.set_title(
        f'Maximum Covering Location Problem Solution (Batch {batch_idx})\n'
        f'Selected Facilities: {len(chosen_facilities)} | Coverage Radius: {coverage_radius:.2f}\n'
        f'Covered Demands: {num_covered}/{num_demand} ({num_covered/num_demand*100:.1f}%) | '
        f'Total Covered Weight: {covered_demand:.1f}/{total_demand:.1f} ({coverage_pct:.1f}%)',
        fontsize=13,
        fontweight='bold'
    )
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Add colorbar
    if len(covered_indices) > 0 or len(uncovered_indices) > 0:
        sm = plt.cm.ScalarMappable(
            cmap='RdYlGn',
            norm=plt.Normalize(vmin=demand_weights.min(), vmax=demand_weights.max())
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Demand Weight')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    Figure saved to {save_path}")
    
    plt.close(fig)
    
    return covered_demand


def test_mclp_env():
    """Test MCLP environment functionality"""
    print("=" * 60)
    print("Testing Maximum Covering Location Problem Environment")
    print("=" * 60)
    
    # Create environment with moderate problem size
    generator_params = {
        "num_demand": 50,
        "num_facility": 30,
        "num_facilities_to_select": 8,
        "min_demand": 1.0,
        "max_demand": 10.0,
        "coverage_radius": 0.2,
    }
    
    env = MCLPEnv(generator_params=generator_params)
    print(f"\n✓ Environment created: {env.name}")
    print(f"  - Number of demand points: {generator_params['num_demand']}")
    print(f"  - Number of candidate facilities: {generator_params['num_facility']}")
    print(f"  - Facilities to select: {generator_params['num_facilities_to_select']}")
    print(f"  - Coverage radius: {generator_params['coverage_radius']}")
    print(f"  - Demand weight range: [{generator_params['min_demand']}, {generator_params['max_demand']}]")
    
    # Generate instances
    batch_size = 4
    td = env.reset(batch_size=[batch_size])
    print(f"\n✓ Generated {batch_size} problem instances")
    print(f"  - Demand locations shape: {td['demand_locs'].shape}")
    print(f"  - Facility locations shape: {td['facility_locs'].shape}")
    print(f"  - Demand weights shape: {td['demand_weights'].shape}")
    print(f"  - Distance matrix shape: {td['distance_matrix'].shape}")
    
    # Test random policy rollout
    print("\n✓ Testing random policy rollout...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    td = td.to(device)
    
    try:
        reward, td_final, actions = rollout(env, td, random_policy)
        print(f"  - Rollout completed successfully")
        print(f"  - Rewards shape: {reward.shape}")
        print(f"  - Actions shape: {actions.shape}")
        print(f"  - Mean reward: {reward.mean().item():.2f}")
        print(f"  - Reward range: [{reward.min().item():.2f}, {reward.max().item():.2f}]")
        
        # Verify solution properties
        print("\n✓ Verifying solution properties...")
        for b in range(batch_size):
            num_chosen = td_final["chosen"][b].sum().item()
            expected_facilities = td_final["num_facilities_to_select"][b].item()
            
            total_demand = td_final["demand_weights"][b].sum().item()
            covered_demand = td_final["covered_demand"][b].sum().item()
            coverage_pct = (covered_demand / total_demand) * 100 if total_demand > 0 else 0
            
            num_covered = td_final["is_covered"][b].sum().item()
            num_demand = td_final["demand_weights"][b].shape[0]
            
            actual_reward = reward[b].item()
            
            print(f"  Batch {b}:")
            print(f"    - Facilities chosen: {num_chosen}/{expected_facilities}")
            print(f"    - Demands covered: {num_covered}/{num_demand} ({num_covered/num_demand*100:.1f}%)")
            print(f"    - Total demand weight: {total_demand:.2f}")
            print(f"    - Covered demand weight: {covered_demand:.2f} ({coverage_pct:.1f}%)")
            print(f"    - Reward: {actual_reward:.2f}")
            print(f"    - Match: {abs(covered_demand - actual_reward) < 1e-4}")
            
            # Check constraint satisfaction
            assert num_chosen == expected_facilities, \
                f"Batch {b}: Expected {expected_facilities} facilities, got {num_chosen}"
        
        print("\n✓ All constraints satisfied!")
        
        # Visualize solutions
        print("\n✓ Visualizing solutions...")
        os.makedirs("results/figs", exist_ok=True)
        for b in range(min(2, batch_size)):
            print(f"\n  Rendering batch {b}...")
            visualize_solution(td_final, actions, batch_idx=b,
                             save_path=f"results/figs/mclp_solution_batch_{b}.png")
        
    except Exception as e:
        print(f"  ✗ Error during rollout: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_mclp_env()
