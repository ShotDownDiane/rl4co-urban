"""Test script for Maximum Coverage Problem (MCP) environment"""

import os
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
from rl4co.envs.graph import MCPEnv
from rl4co.utils.decoding import random_policy, rollout


def visualize_solution(td, actions, batch_idx=0, save_path=None):
    """Visualize the MCP solution
    
    Args:
        td: TensorDict with final state
        actions: actions taken (indices of chosen sets)
        batch_idx: which batch instance to visualize
        save_path: path to save figure (optional)
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Get data for this batch
    membership = td["orig_membership"][batch_idx].cpu().numpy()  # [num_sets, max_size]
    orig_weights = td["orig_weights"][batch_idx].cpu().numpy()  # [num_items]
    current_weights = td["weights"][batch_idx].cpu().numpy()  # [num_items] (0 if covered)
    chosen = td["chosen"][batch_idx].cpu().numpy()  # [num_sets]
    covered = (current_weights == 0) & (orig_weights > 0)  # Items that were covered
    
    num_sets = membership.shape[0]
    num_items = orig_weights.shape[0]
    max_size = membership.shape[1]
    
    chosen_sets = np.where(chosen)[0]
    
    # Create a heatmap-like visualization
    # Rows: sets (chosen first, then unchosen)
    # Columns: items
    
    # Reorder sets: chosen first
    unchosen_sets = np.where(~chosen)[0]
    ordered_sets = np.concatenate([chosen_sets, unchosen_sets])
    
    # Create coverage matrix
    coverage_matrix = np.zeros((num_sets, num_items))
    for set_idx in range(num_sets):
        items_in_set = membership[set_idx]
        items_in_set = items_in_set[items_in_set > 0]  # Remove padding (0s)
        for item_idx in items_in_set:
            if item_idx <= num_items:  # Valid item
                coverage_matrix[set_idx, int(item_idx) - 1] = 1  # -1 because items are 1-indexed
    
    # Reorder matrix
    coverage_matrix = coverage_matrix[ordered_sets, :]
    
    # Create visualization: use different colors for chosen/unchosen sets
    display_matrix = np.zeros((num_sets, num_items))
    for i, set_idx in enumerate(ordered_sets):
        for j in range(num_items):
            if coverage_matrix[i, j] > 0:
                if chosen[set_idx]:
                    # Chosen set: color by item weight
                    display_matrix[i, j] = orig_weights[j]
                else:
                    # Unchosen set: light gray
                    display_matrix[i, j] = -1
    
    # Plot
    # Mask for different coloring
    chosen_mask = np.zeros_like(display_matrix, dtype=bool)
    for i, set_idx in enumerate(ordered_sets):
        if chosen[set_idx]:
            chosen_mask[i, :] = coverage_matrix[i, :] > 0
    
    # Plot unchosen sets (light gray)
    unchosen_matrix = np.ma.masked_where(chosen_mask | (coverage_matrix == 0), coverage_matrix)
    ax.imshow(unchosen_matrix, cmap='Greys', alpha=0.3, aspect='auto', vmin=0, vmax=1)
    
    # Plot chosen sets (colored by weight)
    chosen_matrix = np.ma.masked_where(~chosen_mask, display_matrix)
    im = ax.imshow(chosen_matrix, cmap='YlOrRd', aspect='auto', vmin=orig_weights.min(), vmax=orig_weights.max())
    
    # Add colorbar for weights
    cbar = plt.colorbar(im, ax=ax, label='Item Weight')
    
    # Mark chosen sets
    num_chosen = len(chosen_sets)
    ax.axhline(y=num_chosen - 0.5, color='blue', linewidth=2, linestyle='--', label='Chosen/Unchosen boundary')
    
    # Highlight covered items
    covered_items = np.where(covered)[0]
    for item_idx in covered_items:
        ax.axvline(x=item_idx, color='green', alpha=0.1, linewidth=0.5)
    
    # Labels
    ax.set_xlabel('Items', fontsize=12)
    ax.set_ylabel('Sets (Chosen first)', fontsize=12)
    
    # Y-axis labels
    y_labels = [f'S{idx} {"✓" if chosen[idx] else ""}' for idx in ordered_sets]
    ax.set_yticks(range(num_sets))
    ax.set_yticklabels(y_labels, fontsize=6)
    
    # X-axis: show every 10th item
    x_ticks = range(0, num_items, max(1, num_items // 10))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'I{i}' for i in x_ticks], fontsize=8)
    
    # Calculate statistics
    total_weight = orig_weights[covered].sum()
    coverage_pct = (covered.sum() / num_items) * 100
    avg_set_size = np.mean([np.sum(membership[s] > 0) for s in chosen_sets]) if len(chosen_sets) > 0 else 0
    
    # Title
    ax.set_title(
        f'Maximum Coverage Problem Solution (Batch {batch_idx})\n'
        f'Chosen Sets: {num_chosen}/{num_sets} | '
        f'Covered Items: {covered.sum()}/{num_items} ({coverage_pct:.1f}%) | '
        f'Total Weight: {total_weight:.1f} | '
        f'Avg Set Size: {avg_set_size:.1f}',
        fontsize=12, fontweight='bold'
    )
    
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    Figure saved to {save_path}")
    
    plt.close(fig)
    
    return total_weight


def test_mcp_env():
    """Test MCP environment functionality"""
    print("=" * 50)
    print("Testing Maximum Coverage Problem Environment")
    print("=" * 50)
    
    # Create environment with smaller problem size for testing
    generator_params = {
        "num_items": 100,
        "num_sets": 50,
        "min_weight": 1,
        "max_weight": 10,
        "min_size": 5,
        "max_size": 15,
        "n_sets_to_choose": 10,
    }
    
    env = MCPEnv(generator_params=generator_params)
    print(f"\n✓ Environment created: {env.name}")
    print(f"  - Number of items: {generator_params['num_items']}")
    print(f"  - Number of sets: {generator_params['num_sets']}")
    print(f"  - Sets to choose: {generator_params['n_sets_to_choose']}")
    print(f"  - Set size range: [{generator_params['min_size']}, {generator_params['max_size']}]")
    print(f"  - Item weight range: [{generator_params['min_weight']}, {generator_params['max_weight']}]")
    
    # Generate some instances
    batch_size = 4
    td = env.reset(batch_size=[batch_size])
    print(f"\n✓ Generated {batch_size} problem instances")
    print(f"  - Membership shape: {td['membership'].shape}")
    print(f"  - Weights shape: {td['weights'].shape}")
    print(f"  - Sets to choose: {td['n_sets_to_choose'][0].item()}")
    
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
            expected_sets = td_final["n_sets_to_choose"][b].item()
            
            # Calculate covered items: items with weight 0 in final state but > 0 in original
            orig_weights = td_final["orig_weights"][b]
            current_weights = td_final["weights"][b]
            covered = (current_weights == 0) & (orig_weights > 0)
            
            num_covered = covered.sum().item()
            total_items = orig_weights.shape[0]
            total_weight = orig_weights[covered].sum().item()
            actual_reward = reward[b].item()
            
            print(f"  Batch {b}:")
            print(f"    - Sets chosen: {num_chosen}/{expected_sets}")
            print(f"    - Items covered: {num_covered}/{total_items} ({num_covered/total_items*100:.1f}%)")
            print(f"    - Total weight covered: {total_weight:.2f}")
            print(f"    - Reward: {actual_reward:.2f}")
            print(f"    - Match: {abs(total_weight - actual_reward) < 1e-4}")
            
            # Check constraint satisfaction
            assert num_chosen == expected_sets, \
                f"Batch {b}: Expected {expected_sets} sets, got {num_chosen}"
        
        print("\n✓ All constraints satisfied!")
        
        # Visualize solutions
        print("\n✓ Visualizing solutions...")
        os.makedirs("result/figs", exist_ok=True)  # Create directory if not exists
        for b in range(min(2, batch_size)):  # Visualize first 2 instances
            print(f"\n  Rendering batch {b}...")
            visualize_solution(td_final, actions, batch_idx=b, 
                             save_path=f"result/figs/mcp_solution_batch_{b}.png")
        
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
    test_mcp_env()
