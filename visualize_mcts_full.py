"""
展示MCTS求解TSP的完整过程
"""

import torch
from rl4co.envs import TSPEnv
from rl4co.models.zoo.MCTS import MCTSModel

def visualize_full_solution():
    """展示完整的MCTS求解过程"""
    print("=" * 80)
    print("MCTS求解TSP完整过程展示")
    print("=" * 80)
    
    # 创建小规模问题
    problem_size = 20
    env = TSPEnv(generator_params={'num_loc': problem_size})
    td = env.reset(batch_size=[1])
    
    print(f"\n问题: TSP-{problem_size}")
    print(f"城市坐标:")
    for i, loc in enumerate(td['locs'][0]):
        print(f"  城市 {i}: ({loc[0].item():.3f}, {loc[1].item():.3f})")
    
    # 创建MCTS模型
    mcts = MCTSModel(
        env=env,
        policy=None,
        num_simulations=50,  # 每步10次模拟
        c_puct=2.0,
        temperature=1.0,
        device='cpu'
    )
    
    print(f"\nMCTS参数:")
    print(f"  - 每步模拟次数: {mcts.num_simulations}")
    print(f"  - 探索常数 c_puct: {mcts.c_puct}")
    print(f"  - 温度 temperature: {mcts.temperature}")
    
    print("\n" + "=" * 80)
    print("开始逐步求解")
    print("=" * 80)
    
    # 手动执行每一步
    from rl4co.models.zoo.MCTS.MCTS import MCTS as MCTSCore
    mcts_core = MCTSCore(
        env=env,
        policy=None,
        num_simulations=mcts.num_simulations,
        c_puct=mcts.c_puct,
        temperature=mcts.temperature,
        device='cpu'
    )
    
    td_current = td.clone()
    actions = []
    step = 0
    
    while not td_current['done'].item() and step < problem_size:
        print(f"\n{'='*80}")
        print(f"第 {step + 1} 步决策")
        print(f"{'='*80}")
        
        # 显示当前状态
        print(f"\n当前状态:")
        print(f"  - 当前位置: 城市 {td_current['current_node'].item()}")
        
        # 获取可访问城市
        mask = td_current['action_mask'][0]
        available_cities = torch.where(mask)[0].tolist()
        print(f"  - 可访问城市: {available_cities}")
        print(f"  - 已访问: {step} / {problem_size} 城市")
        
        # 运行MCTS搜索
        print(f"\n运行MCTS搜索 ({mcts.num_simulations} 次模拟)...")
        action, root = mcts_core.search(td_current)
        
        # 显示搜索结果
        print(f"\n搜索统计:")
        print(f"  根节点访问次数: {root.visit_count}")
        
        # 显示所有候选动作的统计
        print(f"\n  候选动作统计:")
        print(f"  {'动作':<8} {'访问次数':<12} {'访问率':<10} {'平均值':<12} {'先验':<10}")
        print(f"  {'-'*60}")
        
        visit_stats = []
        for action_id, child in sorted(root.children.items()):
            visit_rate = child.visit_count / root.visit_count * 100
            visit_stats.append({
                'action': action_id,
                'visits': child.visit_count,
                'rate': visit_rate,
                'value': child.value,
                'prior': child.prior
            })
        
        # 按访问次数排序
        visit_stats.sort(key=lambda x: x['visits'], reverse=True)
        
        for stat in visit_stats:
            marker = " ← 选择" if stat['action'] == action[0].item() else ""
            print(f"  {stat['action']:<8} {stat['visits']:<12} {stat['rate']:<10.1f}% "
                  f"{stat['value']:<12.4f} {stat['prior']:<10.3f}{marker}")
        
        # 执行动作
        action_value = action[0].item()
        print(f"\n→ 选择动作: {action_value} (从城市 {td_current['current_node'].item()} 到城市 {action_value})")
        
        # 计算距离
        current_loc = td_current['locs'][0, td_current['current_node'].item()]
        next_loc = td_current['locs'][0, action_value]
        distance = torch.sqrt(((current_loc - next_loc) ** 2).sum()).item()
        print(f"  距离: {distance:.4f}")
        
        # 更新状态
        actions.append(action_value)
        td_current['action'] = action
        td_current = env.step(td_current)['next']
        step += 1
    
    # 显示最终结果
    print("\n" + "=" * 80)
    print("求解完成")
    print("=" * 80)
    
    print(f"\n完整路径: {' → '.join(map(str, actions))}")
    
    # 计算总距离
    actions_tensor = torch.tensor(actions, device=td.device).unsqueeze(0)
    reward = env.get_reward(td_current, actions_tensor)
    total_distance = -reward.item()
    
    print(f"\n总路径长度: {total_distance:.4f}")
    
    # 显示详细路径
    print(f"\n详细路径:")
    print(f"  起点: 城市 0")
    cumulative_distance = 0.0
    
    for i, action in enumerate(actions):
        if i == 0:
            from_city = 0
        else:
            from_city = actions[i-1]
        to_city = action
        
        from_loc = td['locs'][0, from_city]
        to_loc = td['locs'][0, to_city]
        distance = torch.sqrt(((from_loc - to_loc) ** 2).sum()).item()
        cumulative_distance += distance
        
        print(f"  {i+1}. 城市 {from_city} → 城市 {to_city}: {distance:.4f} (累计: {cumulative_distance:.4f})")
    
    # 返回起点
    last_city = actions[-1]
    last_loc = td['locs'][0, last_city]
    start_loc = td['locs'][0, 0]
    return_distance = torch.sqrt(((last_loc - start_loc) ** 2).sum()).item()
    cumulative_distance += return_distance
    
    print(f"  {len(actions)+1}. 城市 {last_city} → 城市 0 (返回): {return_distance:.4f}")
    print(f"\n总距离: {cumulative_distance:.4f}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    visualize_full_solution()
