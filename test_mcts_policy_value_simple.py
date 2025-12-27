"""
简化版MCTS Policy+Value测试
快速验证架构，添加详细输出
"""

import torch
from rl4co.envs import TSPEnv
from rl4co.models.zoo.MCTS import MCTSModel
from rl4co.models.zoo.am import AttentionModelPolicy

print("="*80)
print("简化版MCTS架构测试")
print("="*80)

# 创建小规模问题以加速测试
problem_size = 10  # 减小问题规模
num_sims = 10      # 减少模拟次数
env = TSPEnv(generator_params={'num_loc': problem_size})

torch.manual_seed(123)
td = env.reset(batch_size=[1])

print(f"\n问题: TSP-{problem_size}")
print(f"模拟次数: {num_sims}次/步")

# ============================================================================
# 测试1: 纯MCTS
# ============================================================================
print("\n" + "="*80)
print("测试1: 纯MCTS（无神经网络）")
print("="*80)

print("创建MCTS模型...")
mcts_pure = MCTSModel(
    env=env,
    policy_net=None,
    value_net=None,
    num_simulations=num_sims,
    c_puct=1.5,
    temperature=0.0,
)

print("配置: policy_net=None, value_net=None")
print("开始求解...")

try:
    td_test = td.clone()
    actions, reward, stats = mcts_pure.solve(td_test, verbose=True)
    
    print(f"\n✓ 测试1完成")
    print(f"  路径长度: {-reward.item():.4f}")
    print(f"  决策步数: {len(stats)}")
    
    if stats:
        first_visits = list(stats[0]['visit_counts'].values())
        print(f"  第1步访问分布: {first_visits[:5]} (前5个)")
        
except Exception as e:
    print(f"\n✗ 测试1失败: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 测试2: Policy-guided MCTS
# ============================================================================
print("\n" + "="*80)
print("测试2: Policy-guided MCTS")
print("="*80)

print("创建policy网络...")
try:
    policy_net = AttentionModelPolicy(
        env_name='tsp',
        embed_dim=64,  # 更小的维度加速
        num_encoder_layers=2,
        num_heads=4,
    )
    policy_net.eval()
    print("✓ Policy网络创建成功")
    
    print("\n创建MCTS模型...")
    mcts_policy = MCTSModel(
        env=env,
        policy_net=policy_net,
        value_net=None,
        num_simulations=num_sims,
        c_puct=1.5,
        temperature=0.0,
    )
    
    print("配置: policy_net=AttentionModel, value_net=None")
    print("开始求解...")
    
    td_test = td.clone()
    actions, reward, stats = mcts_policy.solve(td_test, verbose=True)
    
    print(f"\n✓ 测试2完成")
    print(f"  路径长度: {-reward.item():.4f}")
    print(f"  决策步数: {len(stats)}")
    
    if stats:
        first_visits = list(stats[0]['visit_counts'].values())
        print(f"  第1步访问分布: {first_visits[:5]} (前5个)")
        
except Exception as e:
    print(f"\n✗ 测试2失败: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 测试3: 向后兼容API
# ============================================================================
print("\n" + "="*80)
print("测试3: 向后兼容API (policy参数)")
print("="*80)

try:
    policy = AttentionModelPolicy(env_name='tsp', embed_dim=64)
    policy.eval()
    
    mcts_compat = MCTSModel(
        env=env,
        policy=policy,  # 旧API
        num_simulations=num_sims,
        c_puct=1.5,
        temperature=0.0,
    )
    
    print("配置: policy=AttentionModel (自动转为policy_net和value_net)")
    print("开始求解...")
    
    td_test = td.clone()
    actions, reward, stats = mcts_compat.solve(td_test, verbose=True)
    
    print(f"\n✓ 测试3完成")
    print(f"  路径长度: {-reward.item():.4f}")
    
except Exception as e:
    print(f"\n✗ 测试3失败: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*80)
print("测试总结")
print("="*80)

print("\n支持的架构:")
print("  1. 纯MCTS: policy_net=None, value_net=None")
print("  2. Policy-guided: policy_net=model, value_net=None")
print("  3. 向后兼容: policy=model (自动转换)")

print("\n✓ 测试完成")
print("="*80)
