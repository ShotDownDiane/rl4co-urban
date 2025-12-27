"""
Example: Using MCTS for Combinatorial Optimization in RL4CO

This example demonstrates how to use MCTS with and without neural network policies
for solving TSP problems.
"""

import torch
from rl4co.envs import TSPEnv
from rl4co.models.zoo.MCTS import MCTSModel
from rl4co.models.zoo.am import AttentionModelPolicy


def example_mcts_without_policy():
    """Example 1: Pure MCTS without neural network (random rollouts)."""
    print("=" * 60)
    print("Example 1: Pure MCTS (Random Rollouts)")
    print("=" * 60)
    
    # Create TSP environment with small problem size
    env = TSPEnv(generator_params={'num_loc': 10})
    
    # Create MCTS model without policy
    mcts_model = MCTSModel(
        env=env,
        policy=None,
        num_simulations=50,  # Fewer simulations for random rollouts
        c_puct=1.0,
        temperature=0.0,  # Greedy selection
        device='cpu',
    )
    
    # Generate a problem instance
    td = env.reset(batch_size=[1])
    print(f"\nProblem: TSP with {td['locs'].shape[1]} cities")
    
    # Solve using MCTS
    print("\nSolving with pure MCTS...")
    actions, reward, stats = mcts_model.solve(td)
    
    print(f"\nResults:")
    print(f"  Tour length: {-reward.item():.4f}")
    print(f"  Number of steps: {len(stats)}")
    print(f"  Actions: {actions.squeeze().tolist()[:10]}...")  # Show first 10 actions
    

def example_mcts_with_policy():
    """Example 2: MCTS with neural network policy (AlphaGo-style)."""
    print("\n" + "=" * 60)
    print("Example 2: MCTS with Neural Network Policy")
    print("=" * 60)
    
    # Create TSP environment
    env = TSPEnv(generator_params={'num_loc': 20})
    
    # Create and initialize Attention Model policy
    policy = AttentionModelPolicy(
        env_name='tsp',
        embed_dim=128,
        num_encoder_layers=3,
        num_heads=8,
    )
    
    print("\nNote: Using randomly initialized policy (no training)")
    print("In practice, you would load a pretrained policy checkpoint.")
    
    # Create MCTS model with policy
    mcts_model = MCTSModel(
        env=env,
        policy=policy,
        num_simulations=100,  # More simulations with policy guidance
        c_puct=1.0,
        temperature=0.0,
        device='cpu',
    )
    
    # Generate a problem instance
    td = env.reset(batch_size=[1])
    print(f"\nProblem: TSP with {td['locs'].shape[1]} cities")
    
    # Solve using MCTS with policy
    print("\nSolving with MCTS + Policy...")
    actions, reward, stats = mcts_model.solve(td)
    
    print(f"\nResults:")
    print(f"  Tour length: {-reward.item():.4f}")
    print(f"  Number of steps: {len(stats)}")
    
    # Show some search statistics
    if stats:
        print(f"\nSearch Statistics (first step):")
        print(f"  Root value: {stats[0]['root_value']:.4f}")
        print(f"  Visit counts: {dict(list(stats[0]['visit_counts'].items())[:5])}")


def example_compare_policies():
    """Example 3: Compare pure policy vs MCTS-enhanced policy."""
    print("\n" + "=" * 60)
    print("Example 3: Comparison of Pure Policy vs MCTS")
    print("=" * 60)
    
    # Create environment
    env = TSPEnv(generator_params={'num_loc': 15})
    
    # Create policy
    policy = AttentionModelPolicy(env_name='tsp')
    
    # Generate test instances
    td = env.reset(batch_size=[3])
    print(f"\nEvaluating on {td.batch_size[0]} TSP instances with {td['locs'].shape[1]} cities")
    
    # Method 1: Pure policy (greedy decoding)
    print("\n1. Pure Policy (Greedy):")
    with torch.no_grad():
        policy_out = policy(td, env, phase='test', decode_type='greedy')
    policy_reward = policy_out['reward'].mean().item()
    print(f"   Mean tour length: {-policy_reward:.4f}")
    
    # Method 2: MCTS with policy
    print("\n2. MCTS + Policy:")
    mcts_model = MCTSModel(
        env=env,
        policy=policy,
        num_simulations=50,
        c_puct=1.0,
        temperature=0.0,
    )
    mcts_results = mcts_model.evaluate(td, num_instances=3)
    
    print(f"\nComparison:")
    print(f"  Pure Policy: {-policy_reward:.4f}")
    print(f"  MCTS + Policy: {mcts_results['mean_tour_length']:.4f}")
    improvement = ((-policy_reward - mcts_results['mean_tour_length']) / -policy_reward) * 100
    print(f"  Improvement: {improvement:+.2f}%")


def example_parameter_tuning():
    """Example 4: Effect of MCTS parameters."""
    print("\n" + "=" * 60)
    print("Example 4: MCTS Parameter Tuning")
    print("=" * 60)
    
    # Create environment
    env = TSPEnv(generator_params={'num_loc': 10})
    td = env.reset(batch_size=[1])
    
    # Test different number of simulations
    sim_counts = [10, 50, 100]
    
    print("\nTesting different simulation counts:")
    for num_sims in sim_counts:
        mcts_model = MCTSModel(
            env=env,
            policy=None,
            num_simulations=num_sims,
            c_puct=1.0,
        )
        
        actions, reward, _ = mcts_model.solve(td.clone())
        print(f"  {num_sims:3d} simulations: tour length = {-reward.item():.4f}")
    
    # Test different exploration constants
    c_puct_values = [0.5, 1.0, 2.0]
    
    print("\nTesting different exploration constants (c_puct):")
    for c_puct in c_puct_values:
        mcts_model = MCTSModel(
            env=env,
            policy=None,
            num_simulations=50,
            c_puct=c_puct,
        )
        
        actions, reward, _ = mcts_model.solve(td.clone())
        print(f"  c_puct = {c_puct:.1f}: tour length = {-reward.item():.4f}")


def example_with_pretrained_policy():
    """Example 5: Using MCTS with a pretrained policy checkpoint."""
    print("\n" + "=" * 60)
    print("Example 5: MCTS with Pretrained Policy")
    print("=" * 60)
    
    print("\nTo use a pretrained policy:")
    print("1. Train or download a pretrained model:")
    print("   - Use rl4co training scripts")
    print("   - Or download from model zoo")
    print("\n2. Load the checkpoint:")
    print("   policy = AttentionModelPolicy.load_from_checkpoint('path/to/checkpoint.ckpt')")
    print("\n3. Create MCTS model:")
    print("   mcts_model = MCTSModel(env=env, policy=policy, num_simulations=100)")
    print("\n4. Solve problems:")
    print("   actions, reward, stats = mcts_model.solve(td)")
    
    print("\nNote: MCTS can significantly improve solution quality when combined with")
    print("      a good pretrained policy, especially on harder problem instances.")


if __name__ == "__main__":
    # Run examples
    print("\n" + "=" * 60)
    print("MCTS Examples for RL4CO")
    print("=" * 60)
    
    # Example 1: Pure MCTS
    example_mcts_without_policy()
    
    # Example 2: MCTS with policy
    example_mcts_with_policy()
    
    # Example 3: Comparison
    example_compare_policies()
    
    # Example 4: Parameter tuning
    example_parameter_tuning()
    
    # Example 5: Pretrained policy guide
    example_with_pretrained_policy()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
