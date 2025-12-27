"""快速测试MCTS - 验证没有invalid tour警告"""
import torch
from rl4co.envs import TSPEnv
from rl4co.models.zoo.MCTS import MCTSModel

print("快速测试MCTS（检查警告）...")
print("=" * 60)

# 创建环境
env = TSPEnv(generator_params={'num_loc': 10})

# 创建MCTS模型
mcts = MCTSModel(
    env=env,
    policy=None,
    num_simulations=10,
)

# 生成问题
td = env.reset(batch_size=[1])

print(f"问题: TSP-{td['locs'].shape[1]}")
print("开始求解...")

# 求解（关闭verbose减少输出）
actions, reward, stats = mcts.solve(td, verbose=False)

print(f"\n✓ 求解完成")
print(f"  - 步数: {len(stats)}")
print(f"  - 路径长度: {-reward.item():.4f}")
print(f"  - Actions数量: {actions.shape[1]}")

# 验证
if actions.shape[1] == td['locs'].shape[1]:
    print(f"\n✓ Actions数量正确 ({actions.shape[1]} == {td['locs'].shape[1]})")
else:
    print(f"\n✗ Actions数量错误 ({actions.shape[1]} != {td['locs'].shape[1]})")

print("\n" + "=" * 60)
print("测试完成！")
print("如果没有看到'Invalid tour'警告，说明修复成功。")
