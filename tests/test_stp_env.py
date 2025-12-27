"""Test script for Steiner Tree Problem environment"""

import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
from rl4co.envs.graph.stp import STPEnv
from rl4co.utils.decoding import random_policy, rollout


def visualize_solution(td, actions, batch_idx=0, save_path=None):
    """Visualize the Steiner Tree solution
    
    Args:
        td: TensorDict with final state
        actions: actions taken
        batch_idx: which batch instance to visualize
        save_path: path to save figure (optional)
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Get data for this batch
    locs = td["locs"][batch_idx].cpu().numpy()
    terminals = td["terminals"][batch_idx].cpu().numpy()
    adjacency = td["adjacency"][batch_idx].cpu().numpy()
    selected_edges = td["selected_edges"][batch_idx].cpu().numpy()
    edge_weights = td["edge_weights"][batch_idx].cpu().numpy()
    
    num_nodes = locs.shape[0]
    
    # 1. Draw all edges in the graph (light gray)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency[i, j]:
                ax.plot([locs[i, 0], locs[j, 0]], 
                       [locs[i, 1], locs[j, 1]], 
                       'gray', alpha=0.2, linewidth=0.5, zorder=1)
    
    # 2. Draw selected edges (solution tree) in blue
    total_cost = 0.0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if selected_edges[i, j]:
                weight = edge_weights[i, j]
                total_cost += weight
                ax.plot([locs[i, 0], locs[j, 0]], 
                       [locs[i, 1], locs[j, 1]], 
                       'blue', linewidth=2.5, alpha=0.8, zorder=2)
                # Optionally show edge weight
                mid_x = (locs[i, 0] + locs[j, 0]) / 2
                mid_y = (locs[i, 1] + locs[j, 1]) / 2
                ax.text(mid_x, mid_y, f'{weight:.2f}', 
                       fontsize=6, ha='center', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                       zorder=3)
    
    # 3. Draw nodes
    # Non-terminal nodes (white)
    non_terminals = [i for i in range(num_nodes) if i not in terminals]
    if non_terminals:
        ax.scatter(locs[non_terminals, 0], locs[non_terminals, 1], 
                  c='white', s=200, edgecolors='black', linewidths=2, zorder=4)
        for idx in non_terminals:
            ax.text(locs[idx, 0], locs[idx, 1], str(idx), 
                   ha='center', va='center', fontsize=8, zorder=5)
    
    # Terminal nodes (red)
    ax.scatter(locs[terminals, 0], locs[terminals, 1], 
              c='red', s=300, edgecolors='darkred', linewidths=2.5, 
              zorder=4, marker='s', label='Terminals')
    for idx in terminals:
        ax.text(locs[idx, 0], locs[idx, 1], str(idx), 
               ha='center', va='center', fontsize=9, fontweight='bold', 
               color='white', zorder=5)
    
    # 4. Add title and info
    num_edges_selected = selected_edges.sum() // 2
    ax.set_title(f'Steiner Tree Solution (Batch {batch_idx})\n'
                f'Terminals: {len(terminals)} | Edges: {num_edges_selected} | Total Cost: {total_cost:.4f}',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    Figure saved to {save_path}")
    
    plt.close(fig)  # Close to free memory
    
    return total_cost

# Test basic environment functionality
def test_stp_env():
    print("=" * 50)
    print("Testing Steiner Tree Problem Environment")
    print("=" * 50)
    
    # Create environment with small problem size for testing
    generator_params = {
        "num_nodes": 20,
        "num_terminals": 5,
        "graph_type": "delaunay",  # Use Delaunay triangulation (connected, planar)
    }
    
    env = STPEnv(generator_params=generator_params, project=True)
    print(f"\n✓ Environment created: {env.name}")
    print(f"  - Number of nodes: {generator_params['num_nodes']}")
    print(f"  - Number of terminals: {generator_params['num_terminals']}")
    print(f"  - Project invalid actions: {env.project}")
    
    # Generate some instances
    batch_size = 4
    td = env.reset(batch_size=[batch_size])
    print(f"\n✓ Generated {batch_size} problem instances")
    print(f"  - Locations shape: {td['locs'].shape}")
    print(f"  - Terminals shape: {td['terminals'].shape}")
    print(f"  - Edge weights shape: {td['edge_weights'].shape}")
    print(f"  - Adjacency shape: {td['adjacency'].shape}")
    print(f"  - Edge list shape: {td['edge_list'].shape}")
    print(f"  - Number of edges per instance: {td['num_edges'].tolist()}")
    
    # Show action space efficiency
    num_nodes = generator_params['num_nodes']
    old_action_space = num_nodes * num_nodes
    new_action_space = td['edge_list'].shape[1]
    avg_edges = td['num_edges'].float().mean().item()
    efficiency = avg_edges / old_action_space * 100
    print(f"\n✓ Action Space Optimization:")
    print(f"  - Old (node pairs): {old_action_space}")
    print(f"  - New (edge list): {new_action_space}")
    print(f"  - Avg actual edges: {avg_edges:.1f}")
    print(f"  - Efficiency: {efficiency:.1f}%")
    
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
        
        # Verify reward calculation
        print("\n✓ Verifying reward calculation...")
        for b in range(batch_size):
            # Calculate expected cost from actions
            expected_cost = 0.0
            edge_list = td_final["edge_list"][b]
            edge_weights = td_final["edge_weights"][b]
            num_edges_selected = actions.shape[1]
            
            for step in range(num_edges_selected):
                action_idx = actions[b, step].item()
                # Get edge endpoints from edge_list
                from_node = edge_list[action_idx, 0].item()
                to_node = edge_list[action_idx, 1].item()
                # Get weight
                weight = edge_weights[from_node, to_node].item()
                expected_cost += weight
            
            actual_reward = reward[b].item()
            expected_reward = -expected_cost
            
            print(f"  Batch {b}:")
            print(f"    - Edges selected: {num_edges_selected}")
            print(f"    - Expected cost: {expected_cost:.4f}")
            print(f"    - Expected reward: {expected_reward:.4f}")
            print(f"    - Actual reward: {actual_reward:.4f}")
            print(f"    - Match: {abs(expected_reward - actual_reward) < 1e-4}")
        
        # Check solution validity
        print("\n✓ Checking solution validity...")
        env.check_solution = True
        env.check_solution_validity(td_final, actions)
        print("  - All solutions are valid!")
        
        # Visualize solutions
        print("\n✓ Visualizing solutions...")
        for b in range(min(2, batch_size)):  # Visualize first 2 instances
            print(f"\n  Rendering batch {b}...")
            visualize_solution(td_final, actions, batch_idx=b, 
                             save_path=f"stp_solution_batch_{b}.png")
        
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
    test_stp_env()
