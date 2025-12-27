"""
测试MCTS的Policy Network + Value Network架构
演示AlphaGo Zero风格的实现
"""

import torch
from rl4co.envs import TSPEnv
from rl4co.models.zoo.MCTS import MCTSModel
from rl4co.models.zoo.am import AttentionModelPolicy

def test_mcts_architectures():
    """测试不同的MCTS架构"""
    print("="*80)
    print("MCTS架构对比测试")
    print("="*80)
    
    # 创建环境
    problem_size = 20
    env = TSPEnv(generator_params={'num_loc': problem_size})
    
    # 生成同一个问题用于对比
    torch.manual_seed(123)
    td = env.reset(batch_size=[1])
    
    print(f"\n问题: TSP-{problem_size}")
    print(f"模拟次数: 50次/步")
    
    # ========================================================================
    # 架构1: 纯MCTS（无神经网络）
    # ========================================================================
    print("\n" + "="*80)
    print("架构1: 纯MCTS（无神经网络）")
    print("="*80)
    
    mcts_pure = MCTSModel(
        env=env,
        policy_net=None,
        value_net=None,
        num_simulations=50,
        c_puct=1.5,
        temperature=0.0,
    )
    
    print("配置: 无policy_net, 无value_net")
    print("- 先验概率: 均匀分布")
    print("- 值估计: 随机rollout")
    
    td_test = td.clone()
    actions, reward, stats = mcts_pure.solve(td_test, verbose=False)
    
    print(f"\n结果:")
    print(f"  路径长度: {-reward.item():.4f}")
    print(f"  第1步搜索分布: ", end="")
    if stats:
        visits = list(stats[0]['visit_counts'].values())
        print(f"{visits[:5]}... (前5个动作)")
    
    # ========================================================================
    # 架构2: Policy-guided MCTS
    # ========================================================================
    print("\n" + "="*80)
    print("架构2: Policy-guided MCTS")
    print("="*80)
    
    # 创建policy network
    policy_net = AttentionModelPolicy(
        env_name='tsp',
        embed_dim=128,
        num_encoder_layers=3,
        num_heads=8,
    )
    policy_net.eval()
    
    mcts_policy = MCTSModel(
        env=env,
        policy_net=policy_net,
        value_net=None,  # 不使用value net
        num_simulations=50,
        c_puct=1.5,
        temperature=0.0,
    )
    
    print("配置: 有policy_net, 无value_net")
    print("- 先验概率: 从policy network获取")
    print("- 值估计: 随机rollout")
    
    td_test = td.clone()
    actions, reward, stats = mcts_policy.solve(td_test, verbose=False)
    
    print(f"\n结果:")
    print(f"  路径长度: {-reward.item():.4f}")
    print(f"  第1步搜索分布: ", end="")
    if stats:
        visits = list(stats[0]['visit_counts'].values())
        print(f"{visits[:5]}... (前5个动作)")
    
    # ========================================================================
    # 架构3: AlphaGo Zero风格（Policy + Value）
    # ========================================================================
    print("\n" + "="*80)
    print("架构3: AlphaGo Zero风格（Policy + Value）")
    print("="*80)
    
    # 创建独立的policy和value网络
    # 在实际应用中，这两个网络可以共享encoder部分
    policy_net2 = AttentionModelPolicy(
        env_name='tsp',
        embed_dim=128,
        num_encoder_layers=3,
        num_heads=8,
    )
    policy_net2.eval()
    
    # 这里我们用同一个网络作为value_net
    # 在真实场景中，应该训练专门的value网络
    value_net = policy_net2
    
    mcts_alphago = MCTSModel(
        env=env,
        policy_net=policy_net2,
        value_net=value_net,
        num_simulations=50,
        c_puct=1.5,
        temperature=0.0,
    )
    
    print("配置: 有policy_net, 有value_net")
    print("- 先验概率: 从policy network获取")
    print("- 值估计: 从value network获取（当前使用rollout）")
    print("  注: value network集成还在开发中")
    
    td_test = td.clone()
    actions, reward, stats = mcts_alphago.solve(td_test, verbose=False)
    
    print(f"\n结果:")
    print(f"  路径长度: {-reward.item():.4f}")
    print(f"  第1步搜索分布: ", end="")
    if stats:
        visits = list(stats[0]['visit_counts'].values())
        print(f"{visits[:5]}... (前5个动作)")
    
    # ========================================================================
    # 架构4: 向后兼容的API（使用policy参数）
    # ========================================================================
    print("\n" + "="*80)
    print("架构4: 向后兼容API（policy参数）")
    print("="*80)
    
    policy = AttentionModelPolicy(env_name='tsp', embed_dim=128)
    policy.eval()
    
    mcts_compat = MCTSModel(
        env=env,
        policy=policy,  # 使用旧的API
        num_simulations=50,
        c_puct=1.5,
        temperature=0.0,
    )
    
    print("配置: policy参数（向后兼容）")
    print("- 自动设置: policy_net=policy, value_net=policy")
    
    td_test = td.clone()
    actions, reward, stats = mcts_compat.solve(td_test, verbose=False)
    
    print(f"\n结果:")
    print(f"  路径长度: {-reward.item():.4f}")
    
    # ========================================================================
    # 总结
    # ========================================================================
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    
    print("\n支持的架构:")
    print("  1. 纯MCTS: MCTSModel(env, policy_net=None, value_net=None)")
    print("  2. Policy-guided: MCTSModel(env, policy_net=policy, value_net=None)")
    print("  3. AlphaGo Zero: MCTSModel(env, policy_net=policy, value_net=value)")
    print("  4. 向后兼容: MCTSModel(env, policy=policy)")
    
    print("\n注意事项:")
    print("  - policy_net提供先验概率P(s,a)，引导搜索方向")
    print("  - value_net提供状态估值V(s)，替代随机rollout")
    print("  - 当前value_net功能还在完善中，仍使用rollout")
    print("  - 使用预训练的网络会有更好的效果")
    
    print("\n" + "="*80)
    print("✓ 测试完成")
    print("="*80)

def demonstrate_policy_prior():
    """演示policy network如何提供先验概率"""
    print("\n" + "="*80)
    print("演示: Policy Network的先验概率")
    print("="*80)
    
    env = TSPEnv(generator_params={'num_loc': 10})
    td = env.reset(batch_size=[1])
    
    # 创建policy
    policy_net = AttentionModelPolicy(env_name='tsp', embed_dim=64)
    policy_net.eval()
    
    print(f"\n问题: TSP-{td['locs'].shape[1]}")
    print(f"当前位置: 城市{td['current_node'].item()}")
    
    # 获取先验概率
    with torch.no_grad():
        try:
            hidden = policy_net.encoder(td)
            logits, mask = policy_net.decoder(td, hidden, num_starts=0)
            logits = logits.squeeze(0)
            
            action_mask = td['action_mask'][0]
            logits = logits.masked_fill(~action_mask.bool(), float('-inf'))
            probs = torch.softmax(logits, dim=-1)
            
            print(f"\nPolicy提供的先验概率:")
            print(f"{'动作':<6} {'概率':<10} {'可视化':<30}")
            print("-"*50)
            
            valid_actions = torch.where(action_mask)[0]
            for action in valid_actions[:8]:  # 只显示前8个
                prob = probs[action].item()
                bar = "█" * int(prob * 100)
                print(f"{action.item():<6} {prob:<10.4f} {bar}")
            
            print("\n这些概率在MCTS中作为先验P(s,a)，影响UCB公式:")
            print("  Score(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))")
            print("                                 ^^^^^^")
            print("                          来自policy network")
            
        except Exception as e:
            print(f"✗ 获取先验概率失败: {e}")
            print("  这在使用随机初始化的网络时是正常的")

if __name__ == "__main__":
    # 测试不同架构
    test_mcts_architectures()
    
    # 演示先验概率
    demonstrate_policy_prior()
