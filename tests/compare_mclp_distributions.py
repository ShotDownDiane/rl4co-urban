"""Compare different MCLP data distributions"""

import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from rl4co.envs.graph import MCLPEnv
from rl4co.utils.decoding import random_policy, rollout


def visualize_distribution(td, td_final, actions, name, save_path):
    """Visualize problem distribution and solution"""
    batch_idx = 0  # Visualize first instance
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    # Get data
    demand_locs = td["demand_locs"][batch_idx].cpu().numpy()
    facility_locs = td["facility_locs"][batch_idx].cpu().numpy()
    demand_weights = td["demand_weights"][batch_idx].cpu().numpy()
    coverage_radius = td["coverage_radius"][batch_idx].item()
    
    chosen = td_final["chosen"][batch_idx].cpu().numpy()
    is_covered = td_final["is_covered"][batch_idx].cpu().numpy()
    chosen_facilities = np.where(chosen)[0]
    
    # ============================================
    # Left: Problem Instance (without solution)
    # ============================================
    ax = axes[0]
    
    # Draw all demand points
    scatter = ax.scatter(
        demand_locs[:, 0],
        demand_locs[:, 1],
        c=demand_weights,
        s=150,
        cmap='YlOrRd',
        edgecolors='black',
        linewidths=1,
        alpha=0.7,
        zorder=3,
        vmin=demand_weights.min(),
        vmax=demand_weights.max()
    )
    
    # Draw all candidate facilities
    ax.scatter(
        facility_locs[:, 0],
        facility_locs[:, 1],
        c='lightblue',
        s=100,
        marker='s',
        edgecolors='blue',
        linewidths=1.5,
        alpha=0.6,
        zorder=2,
        label='Candidate facilities'
    )
    
    ax.set_title(f'{name}\nProblem Instance', fontsize=13, fontweight='bold')
    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Colorbar for demand weights
    cbar1 = plt.colorbar(scatter, ax=ax, label='Demand Weight')
    
    # ============================================
    # Right: Solution
    # ============================================
    ax = axes[1]
    
    # Draw coverage circles for selected facilities
    for facility_idx in chosen_facilities:
        circle = mpatches.Circle(
            facility_locs[facility_idx],
            coverage_radius,
            color='lightblue',
            alpha=0.15,
            zorder=1
        )
        ax.add_patch(circle)
        
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
    
    # Draw demand points (covered vs uncovered)
    covered_indices = np.where(is_covered)[0]
    uncovered_indices = np.where(~is_covered)[0]
    
    if len(covered_indices) > 0:
        ax.scatter(
            demand_locs[covered_indices, 0],
            demand_locs[covered_indices, 1],
            c=demand_weights[covered_indices],
            s=150,
            cmap='Greens',
            edgecolors='darkgreen',
            linewidths=2,
            marker='o',
            alpha=0.8,
            zorder=4,
            vmin=demand_weights.min(),
            vmax=demand_weights.max(),
            label=f'Covered ({len(covered_indices)})'
        )
    
    if len(uncovered_indices) > 0:
        ax.scatter(
            demand_locs[uncovered_indices, 0],
            demand_locs[uncovered_indices, 1],
            c=demand_weights[uncovered_indices],
            s=150,
            cmap='Reds',
            edgecolors='darkred',
            linewidths=2,
            marker='x',
            alpha=0.6,
            zorder=4,
            vmin=demand_weights.min(),
            vmax=demand_weights.max(),
            label=f'Uncovered ({len(uncovered_indices)})'
        )
    
    # Draw unchosen facilities
    unchosen_facilities = np.where(~chosen)[0]
    if len(unchosen_facilities) > 0:
        ax.scatter(
            facility_locs[unchosen_facilities, 0],
            facility_locs[unchosen_facilities, 1],
            c='lightgray',
            s=100,
            marker='s',
            edgecolors='gray',
            linewidths=1,
            alpha=0.4,
            zorder=3,
            label='Candidate facilities'
        )
    
    # Draw chosen facilities
    ax.scatter(
        facility_locs[chosen_facilities, 0],
        facility_locs[chosen_facilities, 1],
        c='blue',
        s=400,
        marker='*',
        edgecolors='darkblue',
        linewidths=2,
        zorder=5,
        label=f'Selected ({len(chosen_facilities)})'
    )
    
    # Calculate statistics
    total_demand = demand_weights.sum()
    covered_demand = demand_weights[is_covered].sum()
    coverage_pct = (len(covered_indices) / len(demand_weights)) * 100
    weight_coverage_pct = (covered_demand / total_demand) * 100
    
    ax.set_title(
        f'{name}\nSolution (Random Policy)\n'
        f'Coverage: {len(covered_indices)}/{len(demand_weights)} ({coverage_pct:.1f}%) | '
        f'Weight: {covered_demand:.1f}/{total_demand:.1f} ({weight_coverage_pct:.1f}%)\n'
        f'Radius: {coverage_radius:.3f}',
        fontsize=13,
        fontweight='bold'
    )
    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Visualization saved to {save_path}")
    plt.close(fig)


def test_distribution(distribution, dynamic_radius, name):
    """Test a specific distribution configuration"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    generator_params = {
        "num_demand": 50,
        "num_facility": 30,
        "num_facilities_to_select": 8,
        "min_demand": 1.0,
        "max_demand": 10.0,
        "coverage_radius": 0.2,
        "distribution": distribution,
        "dynamic_radius": dynamic_radius,
    }
    
    env = MCLPEnv(generator_params=generator_params)
    
    # Generate and test
    batch_size = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    td = env.reset(batch_size=[batch_size]).to(device)
    
    reward, td_final, actions = rollout(env, td, random_policy)
    
    # Statistics
    coverage_pct = (td_final["is_covered"].float().mean(dim=1) * 100).cpu()
    total_demand = td_final["demand_weights"].sum(dim=1).cpu()
    covered_demand = td_final["covered_demand"].sum(dim=1).cpu()
    coverage_weight_pct = (covered_demand / total_demand * 100)
    
    print(f"Distribution: {distribution}")
    print(f"Dynamic Radius: {dynamic_radius}")
    if dynamic_radius:
        radius_values = td_final["coverage_radius"].cpu().numpy()
        print(f"Radius range: [{radius_values.min():.4f}, {radius_values.max():.4f}]")
    else:
        print(f"Fixed Radius: {generator_params['coverage_radius']}")
    print(f"\nResults (10 instances):")
    print(f"  - Mean reward: {reward.mean().item():.2f} ± {reward.std().item():.2f}")
    print(f"  - Reward range: [{reward.min().item():.2f}, {reward.max().item():.2f}]")
    print(f"  - Mean coverage: {coverage_pct.mean().item():.1f}% ± {coverage_pct.std().item():.1f}%")
    print(f"  - Mean weight coverage: {coverage_weight_pct.mean().item():.1f}% ± {coverage_weight_pct.std().item():.1f}%")
    
    # Visualize the distribution
    safe_name = name.replace(" ", "_").replace("+", "").lower()
    vis_path = f"results/figs/mclp_dist_{safe_name}.png"
    visualize_distribution(td, td_final, actions, name, vis_path)
    
    return {
        "name": name,
        "distribution": distribution,
        "dynamic_radius": dynamic_radius,
        "mean_reward": reward.mean().item(),
        "std_reward": reward.std().item(),
        "mean_coverage": coverage_pct.mean().item(),
        "mean_weight_coverage": coverage_weight_pct.mean().item(),
        "td": td,
        "td_final": td_final,
        "actions": actions,
    }


def main():
    """Compare different distributions"""
    print("="*60)
    print("MCLP Distribution Comparison")
    print("="*60)
    
    # Ensure output directory exists
    os.makedirs("results/figs", exist_ok=True)
    
    configs = [
        ("uniform", False, "Uniform + Fixed Radius"),
        ("uniform", True, "Uniform + Dynamic Radius"),
        ("cluster", False, "Cluster + Fixed Radius"),
        ("cluster", True, "Cluster + Dynamic Radius"),
        ("explosion", False, "Explosion + Fixed Radius"),
        ("explosion", True, "Explosion + Dynamic Radius"),
    ]
    
    results = []
    for dist, dyn_r, name in configs:
        try:
            result = test_distribution(dist, dyn_r, name)
            results.append(result)
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("Summary Comparison")
    print("="*60)
    print(f"{'Configuration':<35} {'Reward':<15} {'Coverage %':<15}")
    print("-"*60)
    for r in results:
        print(f"{r['name']:<35} {r['mean_reward']:6.2f}±{r['std_reward']:5.2f}   {r['mean_coverage']:6.1f}%")
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    names = [r['name'] for r in results]
    rewards = [r['mean_reward'] for r in results]
    coverages = [r['mean_coverage'] for r in results]
    
    # Reward comparison
    axes[0].bar(range(len(names)), rewards, color='skyblue', edgecolor='navy', linewidth=1.5)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    axes[0].set_ylabel('Mean Reward', fontsize=12)
    axes[0].set_title('Reward by Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Coverage comparison
    axes[1].bar(range(len(names)), coverages, color='lightgreen', edgecolor='darkgreen', linewidth=1.5)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    axes[1].set_ylabel('Mean Coverage (%)', fontsize=12)
    axes[1].set_title('Coverage by Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs("results/figs", exist_ok=True)
    save_path = "results/figs/mclp_distribution_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison figure saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    main()
