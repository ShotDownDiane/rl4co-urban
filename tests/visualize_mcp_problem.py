"""Visualize the original MCP problem structure"""

import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.collections import LineCollection
from rl4co.envs.graph import MCPEnv


def visualize_mcp_problem(td, batch_idx=0, save_path=None):
    """Visualize the MCP problem structure as a bipartite graph
    
    Args:
        td: TensorDict with problem data
        batch_idx: which batch instance to visualize
        save_path: path to save figure (optional)
    """
    # Get data for this batch
    membership = td["membership"][batch_idx].cpu().numpy()  # [num_sets, max_size]
    weights = td["weights"][batch_idx].cpu().numpy()  # [num_items]
    
    num_sets = membership.shape[0]
    num_items = weights.shape[0]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # ============================================
    # Subplot 1: Bipartite Graph (Sets <-> Items)
    # ============================================
    
    # Position sets on the left, items on the right
    set_positions = np.array([[0, i] for i in np.linspace(0, 1, num_sets)])
    item_positions = np.array([[1, i] for i in np.linspace(0, 1, num_items)])
    
    # Draw edges from sets to items
    edges = []
    for set_idx in range(num_sets):
        items_in_set = membership[set_idx]
        items_in_set = items_in_set[items_in_set > 0]  # Remove padding
        for item_idx in items_in_set:
            if item_idx <= num_items:
                # Draw edge from set to item
                edges.append([set_positions[set_idx], item_positions[int(item_idx) - 1]])
    
    # Draw edges
    lc = LineCollection(edges, colors='gray', alpha=0.2, linewidths=0.5)
    ax1.add_collection(lc)
    
    # Draw sets (left side) as blue circles
    ax1.scatter(set_positions[:, 0], set_positions[:, 1], 
               c='lightblue', s=200, edgecolors='blue', linewidths=2, 
               zorder=3, label='Sets')
    
    # Add set labels
    for i in range(num_sets):
        set_size = np.sum(membership[i] > 0)
        ax1.text(set_positions[i, 0] - 0.05, set_positions[i, 1], 
                f'S{i}\n({set_size})', 
                ha='right', va='center', fontsize=6, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    # Draw items (right side) as red circles, colored by weight
    scatter = ax1.scatter(item_positions[:, 0], item_positions[:, 1], 
                         c=weights, s=150, cmap='YlOrRd', 
                         edgecolors='darkred', linewidths=1.5, 
                         zorder=3, vmin=weights.min(), vmax=weights.max())
    
    # Add item labels
    for i in range(num_items):
        ax1.text(item_positions[i, 0] + 0.05, item_positions[i, 1], 
                f'I{i}\n({weights[i]:.0f})', 
                ha='left', va='center', fontsize=5)
    
    # Colorbar for weights
    cbar1 = plt.colorbar(scatter, ax=ax1, label='Item Weight')
    
    ax1.set_xlim(-0.3, 1.3)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Sets', 'Items'], fontsize=14, fontweight='bold')
    ax1.set_yticks([])
    ax1.set_title(f'MCP Problem Structure: Bipartite Graph\n'
                  f'{num_sets} Sets → {num_items} Items', 
                  fontsize=14, fontweight='bold')
    ax1.grid(False)
    
    # ============================================
    # Subplot 2: Heatmap (Set-Item Coverage Matrix)
    # ============================================
    
    # Create coverage matrix
    coverage_matrix = np.zeros((num_sets, num_items))
    for set_idx in range(num_sets):
        items_in_set = membership[set_idx]
        items_in_set = items_in_set[items_in_set > 0]
        for item_idx in items_in_set:
            if item_idx <= num_items:
                coverage_matrix[set_idx, int(item_idx) - 1] = weights[int(item_idx) - 1]
    
    # Plot heatmap
    im = ax2.imshow(coverage_matrix, cmap='YlOrRd', aspect='auto', 
                    vmin=0, vmax=weights.max())
    
    # Add colorbar
    cbar2 = plt.colorbar(im, ax=ax2, label='Item Weight (0 = not in set)')
    
    # Set axis labels
    ax2.set_xlabel('Items', fontsize=12)
    ax2.set_ylabel('Sets', fontsize=12)
    ax2.set_title(f'Set-Item Coverage Matrix\n'
                  f'Color intensity = Item weight', 
                  fontsize=14, fontweight='bold')
    
    # Y-axis: show set indices and sizes
    ax2.set_yticks(range(num_sets))
    set_labels = [f'S{i} ({int(np.sum(membership[i] > 0))})' for i in range(num_sets)]
    ax2.set_yticklabels(set_labels, fontsize=6)
    
    # X-axis: show item indices (every 5th)
    x_ticks = range(0, num_items, max(1, num_items // 20))
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([f'I{i}' for i in x_ticks], fontsize=7)
    
    # Add grid
    ax2.grid(True, which='major', color='white', linewidth=0.5, alpha=0.5)
    
    # Calculate statistics
    total_weight = weights.sum()
    avg_set_size = np.mean([np.sum(membership[s] > 0) for s in range(num_sets)])
    max_coverage = np.sum(coverage_matrix.any(axis=0))  # items that appear in at least one set
    
    # Add overall title with statistics
    fig.suptitle(
        f'Maximum Coverage Problem Visualization (Batch {batch_idx})\n'
        f'Total Items: {num_items} | Total Weight: {total_weight:.0f} | '
        f'Total Sets: {num_sets} | Avg Set Size: {avg_set_size:.1f} | '
        f'Max Coverable Items: {max_coverage}/{num_items}',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Figure saved to {save_path}")
    
    plt.close(fig)
    
    return coverage_matrix


def visualize_mcp_sets_circular(td, batch_idx=0, save_path=None):
    """Visualize MCP as circular layout with sets highlighted
    
    Args:
        td: TensorDict with problem data
        batch_idx: which batch instance to visualize
        save_path: path to save figure (optional)
    """
    # Get data
    membership = td["membership"][batch_idx].cpu().numpy()
    weights = td["weights"][batch_idx].cpu().numpy()
    
    num_sets = membership.shape[0]
    num_items = weights.shape[0]
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    
    # Arrange items in a circle
    angles = np.linspace(0, 2 * np.pi, num_items, endpoint=False)
    radius = 5
    item_positions = np.column_stack([radius * np.cos(angles), 
                                      radius * np.sin(angles)])
    
    # Color palette for sets
    colors = plt.cm.tab20(np.linspace(0, 1, min(num_sets, 20)))
    
    # Draw each set as a colored region
    for set_idx in range(min(num_sets, 20)):  # Limit to 20 sets for clarity
        items_in_set = membership[set_idx]
        items_in_set = items_in_set[items_in_set > 0]
        item_indices = [int(i) - 1 for i in items_in_set if i <= num_items]
        
        if len(item_indices) > 0:
            # Get positions of items in this set
            set_positions = item_positions[item_indices]
            
            # Draw polygon connecting items in this set
            from scipy.spatial import ConvexHull
            if len(item_indices) >= 3:
                try:
                    hull = ConvexHull(set_positions)
                    hull_points = set_positions[hull.vertices]
                    poly = mpatches.Polygon(hull_points, 
                                           color=colors[set_idx], 
                                           alpha=0.15, 
                                           linewidth=2,
                                           edgecolor=colors[set_idx],
                                           label=f'Set {set_idx}')
                    ax.add_patch(poly)
                except:
                    pass
            
            # Draw lines from set center to items
            set_center = set_positions.mean(axis=0)
            for pos in set_positions:
                ax.plot([set_center[0], pos[0]], [set_center[1], pos[1]], 
                       color=colors[set_idx], alpha=0.3, linewidth=1)
            
            # Draw set label at center
            ax.text(set_center[0], set_center[1], f'S{set_idx}', 
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='circle,pad=0.3', 
                           facecolor=colors[set_idx], alpha=0.5))
    
    # Draw items as colored circles
    scatter = ax.scatter(item_positions[:, 0], item_positions[:, 1], 
                        c=weights, s=300, cmap='YlOrRd', 
                        edgecolors='black', linewidths=2, 
                        zorder=10, vmin=weights.min(), vmax=weights.max())
    
    # Add item labels
    for i in range(num_items):
        ax.text(item_positions[i, 0] * 1.15, item_positions[i, 1] * 1.15, 
               f'I{i}\n({weights[i]:.0f})', 
               ha='center', va='center', fontsize=7,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Item Weight', shrink=0.8)
    
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'MCP Problem: Circular Layout (First {min(num_sets, 20)} Sets)\n'
                f'Each colored region represents a set covering its items',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Figure saved to {save_path}")
    
    plt.close(fig)


def main():
    """Generate and visualize MCP problem structure"""
    print("=" * 60)
    print("Generating MCP Problem Visualization")
    print("=" * 60)
    
    # Create environment with moderate problem size
    generator_params = {
        "num_items": 30,  # Smaller for clarity
        "num_sets": 15,
        "min_weight": 1,
        "max_weight": 10,
        "min_size": 3,
        "max_size": 8,
        "n_sets_to_choose": 5,
    }
    
    env = MCPEnv(generator_params=generator_params)
    print(f"\n✓ Environment created")
    print(f"  - Number of items: {generator_params['num_items']}")
    print(f"  - Number of sets: {generator_params['num_sets']}")
    print(f"  - Set size range: [{generator_params['min_size']}, {generator_params['max_size']}]")
    
    # Generate instances
    batch_size = 2
    td = env.reset(batch_size=[batch_size])
    print(f"\n✓ Generated {batch_size} problem instances")
    
    # Create output directory
    os.makedirs("result/figs", exist_ok=True)
    
    # Visualize
    print("\n✓ Generating visualizations...")
    for b in range(batch_size):
        print(f"\n  Batch {b}:")
        print(f"    - Bipartite graph + heatmap...")
        visualize_mcp_problem(td, batch_idx=b, 
                            save_path=f"results/figs/mcp_problem_structure_{b}.png")
        
        print(f"    - Circular layout...")
        visualize_mcp_sets_circular(td, batch_idx=b,
                                   save_path=f"results/figs/mcp_problem_circular_{b}.png")
    
    print("\n" + "=" * 60)
    print("Visualizations saved to results/figs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
