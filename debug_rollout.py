"""调试rollout过程"""
import torch
from rl4co.envs import TSPEnv

# 创建环境
env = TSPEnv(generator_params={'num_loc': 5})

# 重置环境
td = env.reset(batch_size=[1])

print("=" * 60)
print("测试rollout过程")
print("=" * 60)
print(f"初始状态:")
print(f"  locs shape: {td['locs'].shape}")
print(f"  action_mask: {td['action_mask']}")
print(f"  done: {td['done']}")

# 手动rollout
td_rollout = td.clone()
actions = []
max_steps = 20
step = 0

while not td_rollout['done'].item() and step < max_steps:
    mask = td_rollout['action_mask']
    # Handle 2D mask properly
    if mask.dim() == 2:
        valid_actions = torch.where(mask[0])[0]
    else:
        valid_actions = torch.where(mask)[0]
    
    print(f"\nStep {step}:")
    print(f"  valid_actions: {valid_actions.tolist()}")
    print(f"  done: {td_rollout['done'].item()}")
    
    if len(valid_actions) == 0:
        print("  No valid actions! Breaking.")
        break
    
    # Random action
    action_idx = torch.randint(0, len(valid_actions), (1,)).item()
    action = valid_actions[action_idx].item()
    print(f"  selected action: {action}")
    
    # Step
    td_rollout['action'] = torch.tensor([action], device=td_rollout.device, dtype=torch.long)
    td_rollout = env.step(td_rollout)['next']
    actions.append(action)
    step += 1

print(f"\n" + "=" * 60)
print(f"Rollout完成")
print(f"=" * 60)
print(f"Total steps: {step}")
print(f"Actions: {actions}")
print(f"Final done: {td_rollout['done'].item()}")

# 测试get_reward
print(f"\n" + "=" * 60)
print(f"测试 get_reward")
print(f"=" * 60)
actions_tensor = torch.tensor(actions, device=td.device, dtype=torch.long).unsqueeze(0)
print(f"actions_tensor shape: {actions_tensor.shape}")
print(f"actions_tensor: {actions_tensor}")

try:
    reward = env.get_reward(td_rollout, actions_tensor)
    print(f"✓ Reward: {reward.item()}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
