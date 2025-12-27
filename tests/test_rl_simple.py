"""
Simple test for RL policy on TSP benchmark
"""

import torch
import numpy as np
from rl4co.data.ml4co_dataset import load_ml4co_dataset
from rl4co.envs import TSPEnv
from rl4co.models import AttentionModel


def compute_tour_cost(locs: np.ndarray, tour: np.ndarray) -> float:
    """Compute the total cost of a tour."""
    cost = 0.0
    for i in range(len(tour)):
        j = (i + 1) % len(tour)
        cost += np.linalg.norm(locs[tour[i]] - locs[tour[j]])
    return cost


def main():
    print("="*80)
    print("Simple RL Policy Test")
    print("="*80)
    
    # Load dataset
    dataset_path = "/root/autodl-tmp/ML4CO-Kit/test_dataset/tsp/wrapper/tsp50_uniform_16ins.txt"
    dataset = load_ml4co_dataset('tsp', dataset_path)
    print(f"\nLoaded {len(dataset)} instances")
    
    # Create environment and policy
    print("\nCreating RL policy (untrained baseline)...")
    # Get problem size from first instance
    first_instance = dataset[0]
    problem_size = len(first_instance['locs'])
    print(f"Problem size: {problem_size} nodes")
    
    env = TSPEnv(generator_params={'num_loc': problem_size})
    policy = AttentionModel(env)
    policy.eval()
    
    # Test on first 3 instances
    print("\nTesting on first 3 instances:")
    print("-" * 80)
    
    gaps = []
    times = []
    
    for i in range(min(3, len(dataset))):
        instance = dataset[i]
        locs = instance['locs']
        optimal_tour = instance['tour'].numpy()
        
        # Compute optimal cost
        optimal_cost = compute_tour_cost(locs.numpy(), optimal_tour)
        
        try:
            # Reset environment
            td = env.reset(batch_size=[1])
            td['locs'] = locs.unsqueeze(0)
            
            # Run policy
            import time
            start = time.time()
            with torch.no_grad():
                # Method 1: Direct call
                try:
                    out = policy(td, phase="test", decode_type="greedy", return_actions=True)
                    if isinstance(out, dict):
                        actions = out['actions'][0]
                    else:
                        actions = out[0][0]
                except Exception as e1:
                    print(f"  Method 1 failed: {e1}")
                    # Method 2: Using policy's decode
                    try:
                        out = policy.policy(td, env, decode_type="greedy")
                        actions = out['actions'][0]
                    except Exception as e2:
                        print(f"  Method 2 failed: {e2}")
                        # Method 3: Step-by-step
                        actions = []
                        current_td = td.clone()
                        while not current_td['done'].all():
                            logits = policy.policy(current_td)
                            action = logits.argmax(-1)
                            current_td = env.step(current_td, action)
                            actions.append(action.item())
                        actions = torch.tensor(actions)
            
            elapsed = time.time() - start
            
            # Compute cost
            actions_np = actions.cpu().numpy() if torch.is_tensor(actions) else np.array(actions)
            rl_cost = compute_tour_cost(locs.numpy(), actions_np)
            
            # Compute gap
            gap = (rl_cost / optimal_cost - 1) * 100
            
            gaps.append(gap)
            times.append(elapsed)
            
            print(f"\nInstance {i}:")
            print(f"  Optimal cost: {optimal_cost:.4f}")
            print(f"  RL cost:      {rl_cost:.4f}")
            print(f"  Gap:          {gap:.2f}%")
            print(f"  Time:         {elapsed:.4f}s")
            
        except Exception as e:
            print(f"\nInstance {i}: ERROR - {e}")
            import traceback
            traceback.print_exc()
    
    if gaps:
        print("\n" + "="*80)
        print("Summary:")
        print(f"  Mean gap:  {np.mean(gaps):.2f}%")
        print(f"  Mean time: {np.mean(times):.4f}s")
        print("="*80)


if __name__ == "__main__":
    main()
