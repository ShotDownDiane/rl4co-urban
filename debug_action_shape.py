"""调试action形状问题"""
import torch
from rl4co.envs import TSPEnv

# 创建环境
env = TSPEnv(generator_params={'num_loc': 5})

# 重置环境
td = env.reset(batch_size=[1])

print("=" * 60)
print("初始状态")
print("=" * 60)
print(f"batch_size: {td.batch_size}")
print(f"action_mask shape: {td['action_mask'].shape}")
print(f"current_node shape: {td['current_node'].shape}")
print(f"done: {td['done']}")

# 测试action
print("\n" + "=" * 60)
print("测试不同action形状")
print("=" * 60)

# 测试1: [1]
action1 = torch.tensor([2], dtype=torch.long)
print(f"\naction1 = torch.tensor([2])")
print(f"  shape: {action1.shape}")
print(f"  unsqueeze(-1): {action1.unsqueeze(-1).shape}")

# 测试2: scalar
action2 = torch.tensor(2, dtype=torch.long)
print(f"\naction2 = torch.tensor(2)")
print(f"  shape: {action2.shape}")
print(f"  需要先unsqueeze(0)才能用")

# 正确的做法
print("\n" + "=" * 60)
print("执行step")
print("=" * 60)

td['action'] = torch.tensor([2], dtype=torch.long)
print(f"设置 action shape: {td['action'].shape}")

try:
    td_next = env.step(td)['next']
    print("✓ Step成功!")
    print(f"  new current_node: {td_next['current_node']}")
    print(f"  new action_mask shape: {td_next['action_mask'].shape}")
    print(f"  done: {td_next['done']}")
except Exception as e:
    print(f"✗ Step失败: {e}")
