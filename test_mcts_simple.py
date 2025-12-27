"""
简单测试脚本：验证MCTS实现是否可以正常工作
"""

import torch
from rl4co.envs import TSPEnv
from rl4co.models.zoo.MCTS import MCTSModel
from rl4co.models.zoo.am import AttentionModelPolicy

def test_mcts_basic():
    """测试基础MCTS功能（无策略）"""
    print("=" * 60)
    print("测试1: 基础MCTS（无策略）")
    print("=" * 60)
    
    try:
        # 创建环境
        env = TSPEnv(generator_params={'num_loc': 10})
        print("✓ 环境创建成功")
        
        # 创建MCTS模型
        mcts = MCTSModel(
            env=env,
            policy=None,
            num_simulations=10,  # 减少模拟次数加快测试
            c_puct=1.0,
        )
        print("✓ MCTS模型创建成功")
        
        # 生成问题
        td = env.reset(batch_size=[1])
        print(f"✓ 生成问题实例: TSP-{td['locs'].shape[1]}")
        
        # 求解（显示详细输出）
        print("\n开始MCTS求解（显示详细过程）...")
        actions, reward, stats = mcts.solve(td, verbose=True)
        print(f"\n✓ 求解完成")
        print(f"  - 解的长度: {len(stats)} 步")
        print(f"  - 路径长度: {-reward.item():.4f}")
        print(f"  - 动作形状: {actions.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mcts_with_policy():
    """测试MCTS与策略结合"""
    print("\n" + "=" * 60)
    print("测试2: MCTS + 神经网络策略")
    print("=" * 60)
    
    try:
        # 创建环境
        env = TSPEnv(generator_params={'num_loc': 10})
        print("✓ 环境创建成功")
        
        # 创建策略
        policy = AttentionModelPolicy(
            env_name='tsp',
            embed_dim=64,  # 使用较小的维度加快测试
            num_encoder_layers=2,
        )
        print("✓ 策略创建成功")
        
        # 创建MCTS模型
        mcts = MCTSModel(
            env=env,
            policy=policy,
            num_simulations=10,  # 减少模拟次数加快测试
            c_puct=1.0,
        )
        print("✓ MCTS+策略模型创建成功")
        
        # 生成问题
        td = env.reset(batch_size=[1])
        print(f"✓ 生成问题实例: TSP-{td['locs'].shape[1]}")
        
        # 求解（关闭详细输出以加快速度）
        actions, reward, stats = mcts.solve(td, verbose=False)
        print(f"✓ 求解完成")
        print(f"  - 解的长度: {len(stats)} 步")
        print(f"  - 路径长度: {-reward.item():.4f}")
        print(f"  - 第一步搜索统计:")
        if stats and 'visit_counts' in stats[0]:
            visit_counts = stats[0]['visit_counts']
            print(f"    访问次数: {dict(list(visit_counts.items())[:3])}")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_evaluation():
    """测试批量评估"""
    print("\n" + "=" * 60)
    print("测试3: 批量评估")
    print("=" * 60)
    
    try:
        # 创建环境
        env = TSPEnv(generator_params={'num_loc': 10})
        
        # 创建MCTS模型
        mcts = MCTSModel(
            env=env,
            policy=None,
            num_simulations=8,  # 减少模拟次数加快测试
        )
        print("✓ MCTS模型创建成功")
        
        # 生成多个问题
        td = env.reset(batch_size=[2])  # 减少实例数量
        print(f"✓ 生成 {td.batch_size[0]} 个问题实例")
        
        # 评估
        results = mcts.evaluate(td, num_instances=2)
        print("✓ 评估完成")
        print(f"  - 平均路径长度: {results['mean_tour_length']:.4f}")
        print(f"  - 最短路径: {results['min_tour_length']:.4f}")
        print(f"  - 最长路径: {results['max_tour_length']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parameter_adjustment():
    """测试参数调整功能"""
    print("\n" + "=" * 60)
    print("测试4: 参数动态调整")
    print("=" * 60)
    
    try:
        env = TSPEnv(generator_params={'num_loc': 10})
        mcts = MCTSModel(env=env, policy=None, num_simulations=10)
        print("✓ MCTS模型创建成功")
        
        # 测试调整模拟次数
        mcts.set_num_simulations(50)
        print("✓ 模拟次数调整成功")
        
        # 测试调整温度
        mcts.set_temperature(1.0)
        print("✓ 温度参数调整成功")
        
        # 测试调整探索常数
        mcts.set_c_puct(2.0)
        print("✓ 探索常数调整成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MCTS 实现测试")
    print("=" * 60)
    
    results = []
    
    # 运行测试
    results.append(("基础MCTS", test_mcts_basic()))
    results.append(("MCTS+策略", test_mcts_with_policy()))
    results.append(("批量评估", test_batch_evaluation()))
    results.append(("参数调整", test_parameter_adjustment()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name:15s}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！MCTS实现正常工作。")
    else:
        print(f"\n⚠️  {total - passed} 个测试失败。")
