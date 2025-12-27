"""
展示MCTS结合神经网络Policy的效果
对比纯MCTS vs Policy-guided MCTS
"""

import torch
from rl4co.envs import TSPEnv
from rl4co.models.zoo.MCTS import MCTSModel
from rl4co.models.zoo.am import AttentionModelPolicy

def compare_mcts_methods():
    """对比不同MCTS方法的效果"""
    print("=" * 80)
    print("MCTS方法对比：纯MCTS vs Policy-guided MCTS")
    print("=" * 80)
    
    # 创建环境
    problem_size = 20
    env = TSPEnv(generator_params={'num_loc': problem_size})
    
    # 生成同一个问题实例用于对比
    torch.manual_seed(42)  # 固定随机种子
    td = env.reset(batch_size=[1])
    
    print(f"\n问题: TSP-{problem_size}")
    print(f"城市数量: {td['locs'].shape[1]}")
    
    # ========================================================================
    # 方法1: 纯MCTS（无Policy）
    # ========================================================================
    print("\n" + "=" * 80)
    print("方法1: 纯MCTS（随机Rollout）")
    print("=" * 80)
    
    mcts_pure = MCTSModel(
        env=env,
        policy=None,  # 不使用policy
        num_simulations=50,
        c_puct=1.5,
        temperature=0.0,
        device='cpu'
    )
    
    print(f"参数: num_simulations=50, c_puct=1.5, no policy")
    print("开始求解...")
    
    # 重置到相同的初始状态
    td_pure = td.clone()
    actions_pure, reward_pure, stats_pure = mcts_pure.solve(td_pure, verbose=False)
    
    print(f"\n结果:")
    print(f"  - 路径长度: {-reward_pure.item():.4f}")
    print(f"  - 决策步数: {len(stats_pure)}")
    print(f"  - 路径: {' → '.join(map(str, actions_pure[0].tolist()[:5]))} ... (前5步)")
    
    # 显示第一步的搜索分布
    if stats_pure:
        first_step = stats_pure[0]
        visit_counts = first_step['visit_counts']
        total_visits = sum(visit_counts.values())
        print(f"\n  第1步搜索分布 (总访问: {total_visits}):")
        sorted_visits = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (action, count) in enumerate(sorted_visits[:5]):
            pct = count / total_visits * 100
            print(f"    动作{action}: {count}次 ({pct:.1f}%)")
    
    # ========================================================================
    # 方法2: Policy-guided MCTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("方法2: Policy-guided MCTS")
    print("=" * 80)
    
    # 创建策略（使用随机初始化的网络）
    print("创建神经网络策略...")
    policy = AttentionModelPolicy(
        env_name='tsp',
        embed_dim=128,
        num_encoder_layers=3,
        num_heads=8,
    )
    policy.eval()  # 设置为评估模式
    
    print("✓ 策略创建成功")
    
    mcts_policy = MCTSModel(
        env=env,
        policy=policy,  # 使用policy
        num_simulations=50,
        c_puct=1.5,
        temperature=0.0,
        device='cpu'
    )
    
    print(f"参数: num_simulations=50, c_puct=1.5, with policy")
    print("开始求解...")
    
    # 重置到相同的初始状态
    td_policy = td.clone()
    actions_policy, reward_policy, stats_policy = mcts_policy.solve(td_policy, verbose=False)
    
    print(f"\n结果:")
    print(f"  - 路径长度: {-reward_policy.item():.4f}")
    print(f"  - 决策步数: {len(stats_policy)}")
    print(f"  - 路径: {' → '.join(map(str, actions_policy[0].tolist()[:5]))} ... (前5步)")
    
    # 显示第一步的搜索分布
    if stats_policy:
        first_step = stats_policy[0]
        visit_counts = first_step['visit_counts']
        total_visits = sum(visit_counts.values())
        print(f"\n  第1步搜索分布 (总访问: {total_visits}):")
        sorted_visits = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (action, count) in enumerate(sorted_visits[:5]):
            pct = count / total_visits * 100
            print(f"    动作{action}: {count}次 ({pct:.1f}%)")
    
    # ========================================================================
    # 对比总结
    # ========================================================================
    print("\n" + "=" * 80)
    print("对比总结")
    print("=" * 80)
    
    improvement = ((-reward_pure.item()) - (-reward_policy.item())) / (-reward_pure.item()) * 100
    
    print(f"\n路径长度对比:")
    print(f"  纯MCTS:         {-reward_pure.item():.4f}")
    print(f"  Policy-guided:  {-reward_policy.item():.4f}")
    
    if improvement > 0:
        print(f"  → Policy-guided 更好，改进: {improvement:.2f}%")
    elif improvement < 0:
        print(f"  → 纯MCTS 更好，差异: {abs(improvement):.2f}%")
    else:
        print(f"  → 结果相同")
    
    print(f"\n注意:")
    print(f"  - Policy使用的是随机初始化的网络（未训练）")
    print(f"  - 使用预训练的policy会有更显著的改进")
    print(f"  - Policy主要提供先验概率和值估计来指导搜索")
    
    # ========================================================================
    # 详细分析第一步
    # ========================================================================
    print("\n" + "=" * 80)
    print("第一步详细分析")
    print("=" * 80)
    
    if stats_pure and stats_policy:
        print("\n纯MCTS的搜索分布:")
        pure_visits = stats_pure[0]['visit_counts']
        pure_sorted = sorted(pure_visits.items(), key=lambda x: x[1], reverse=True)
        
        print(f"{'动作':<6} {'访问次数':<10} {'比例':<10}")
        print("-" * 30)
        total = sum(pure_visits.values())
        for action, count in pure_sorted[:8]:
            pct = count / total * 100
            bar = "█" * int(pct / 5)
            print(f"{action:<6} {count:<10} {pct:>6.1f}%  {bar}")
        
        print("\nPolicy-guided MCTS的搜索分布:")
        policy_visits = stats_policy[0]['visit_counts']
        policy_sorted = sorted(policy_visits.items(), key=lambda x: x[1], reverse=True)
        
        print(f"{'动作':<6} {'访问次数':<10} {'比例':<10}")
        print("-" * 30)
        total = sum(policy_visits.values())
        for action, count in policy_sorted[:8]:
            pct = count / total * 100
            bar = "█" * int(pct / 5)
            print(f"{action:<6} {count:<10} {pct:>6.1f}%  {bar}")
        
        print("\n观察:")
        print("  - 纯MCTS: 先验均匀，依赖随机rollout的结果")
        print("  - Policy-guided: 先验来自网络，可能更快聚焦到好的动作")
        print("  - 即使是未训练的网络，也能提供一定的启发")
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)

def visualize_policy_prior():
    """可视化Policy提供的先验概率"""
    print("\n" + "=" * 80)
    print("额外实验: Policy先验概率可视化")
    print("=" * 80)
    
    # 创建环境和策略
    env = TSPEnv(generator_params={'num_loc': 10})
    td = env.reset(batch_size=[1])
    
    policy = AttentionModelPolicy(env_name='tsp', embed_dim=64)
    policy.eval()
    
    print(f"\n问题: TSP-{td['locs'].shape[1]}")
    print(f"当前位置: 城市 {td['current_node'].item()}")
    print(f"可访问: {torch.where(td['action_mask'][0])[0].tolist()}")
    
    # 获取policy的输出
    with torch.no_grad():
        # Encode
        hidden = policy.encoder(td)
        
        # Decode一步
        logits, mask = policy.decoder(td, hidden, num_starts=0)
        logits = logits.squeeze(0)
        
        # 应用mask
        logits = logits.masked_fill(~mask[0].bool(), float('-inf'))
        probs = torch.softmax(logits, dim=-1)
    
    print(f"\nPolicy提供的先验概率:")
    print(f"{'动作':<6} {'概率':<10} {'可视化':<20}")
    print("-" * 40)
    
    valid_actions = torch.where(td['action_mask'][0])[0]
    for action in valid_actions:
        prob = probs[action].item()
        bar = "█" * int(prob * 50)
        print(f"{action.item():<6} {prob:<10.4f} {bar}")
    
    print("\n说明:")
    print("  - 这些概率会作为MCTS的先验P(s,a)")
    print("  - 在UCB公式中引导搜索方向")
    print("  - 预训练的policy会提供更有意义的先验")

if __name__ == "__main__":
    # 主要对比实验
    compare_mcts_methods()
    
    # 额外的先验可视化
    print("\n" * 2)
    visualize_policy_prior()
