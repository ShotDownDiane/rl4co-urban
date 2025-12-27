"""
可视化MCTS在TSP问题上的选择步骤
详细展示每一步的搜索过程、UCB计算和决策
"""

import torch
import math
from rl4co.envs import TSPEnv
from rl4co.models.zoo.MCTS import MCTS

def print_separator(char="=", length=80):
    """打印分隔线"""
    print(char * length)

def print_node_info(node, indent=0):
    """打印节点信息"""
    prefix = "  " * indent
    print(f"{prefix}Node Info:")
    print(f"{prefix}  - Visit count: {node.visit_count}")
    print(f"{prefix}  - Value sum: {node.value_sum:.4f}")
    print(f"{prefix}  - Average value: {node.value:.4f}")
    print(f"{prefix}  - Children: {len(node.children)}")
    print(f"{prefix}  - Expanded: {node.is_expanded}")

def print_ucb_calculation(parent, child, action, c_puct):
    """详细打印UCB计算过程"""
    q_value = child.value
    u_value = c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
    score = q_value + u_value
    
    print(f"    Action {action}:")
    print(f"      Q(s,a) = {child.value_sum}/{child.visit_count} = {q_value:.4f}")
    print(f"      U(s,a) = {c_puct:.1f} * {child.prior:.3f} * sqrt({parent.visit_count}) / (1+{child.visit_count})")
    print(f"             = {c_puct:.1f} * {child.prior:.3f} * {math.sqrt(parent.visit_count):.3f} / {1+child.visit_count}")
    print(f"             = {u_value:.4f}")
    print(f"      Score  = {q_value:.4f} + {u_value:.4f} = {score:.4f}")
    print(f"      (visits: {child.visit_count}, prior: {child.prior:.3f})")
    
    return score

def visualize_mcts_step(env, num_simulations=5, problem_size=5):
    """可视化MCTS的一步决策过程"""
    print_separator("=")
    print("MCTS 决策过程可视化")
    print_separator("=")
    
    # 创建环境和MCTS
    td = env.reset(batch_size=[1])
    print(f"\n问题: TSP-{problem_size}")
    print(f"城市坐标:\n{td['locs']}")
    
    # 创建MCTS实例
    mcts = MCTS(
        env=env,
        policy=None,
        num_simulations=num_simulations,
        c_puct=1.0,
        temperature=0.0,
        device='cpu'
    )
    
    print(f"\n参数:")
    print(f"  - 模拟次数: {num_simulations}")
    print(f"  - c_puct: {mcts.c_puct}")
    print(f"  - temperature: {mcts.temperature}")
    
    # 运行第一步的MCTS搜索
    print_separator("-")
    print("开始MCTS搜索（第1步决策）")
    print_separator("-")
    
    # 创建根节点
    from rl4co.models.zoo.MCTS.MCTS import MCTSNode
    root = MCTSNode(state=td.clone())
    
    print("\n根节点初始状态:")
    print(f"  - 当前位置: {td['current_node'].item()}")
    print(f"  - 可访问城市: {torch.where(td['action_mask'][0])[0].tolist()}")
    print(f"  - done: {td['done'].item()}")
    
    # 手动执行几次模拟来展示过程
    print_separator("-")
    print(f"执行 {num_simulations} 次模拟")
    print_separator("-")
    
    for sim_idx in range(num_simulations):
        print(f"\n{'='*60}")
        print(f"模拟 {sim_idx + 1}/{num_simulations}")
        print(f"{'='*60}")
        
        # 模拟过程
        node = root
        path = []
        depth = 0
        
        # Selection阶段
        while node.is_expanded and not node.state['done'].item():
            print(f"\n[深度 {depth}] Selection: 选择最佳子节点")
            print_node_info(node, indent=0)
            
            if len(node.children) == 0:
                print("  没有子节点，停止selection")
                break
            
            print(f"\n  计算所有子节点的UCB分数:")
            scores = {}
            for action, child in node.children.items():
                score = print_ucb_calculation(node, child, action, mcts.c_puct)
                scores[action] = score
            
            # 选择最大分数
            best_action = max(scores, key=scores.get)
            best_score = scores[best_action]
            print(f"\n  → 选择动作 {best_action} (Score={best_score:.4f})")
            
            node = node.children[best_action]
            path.append(best_action)
            depth += 1
        
        # Expansion阶段
        if not node.is_expanded and not node.state['done'].item():
            print(f"\n[深度 {depth}] Expansion: 扩展节点")
            
            # 评估节点
            action_probs, value = mcts._evaluate(node.state)
            action_mask = node.state['action_mask']
            
            print(f"  评估结果:")
            print(f"    - 估计值: {value:.4f}")
            
            # 获取有效动作
            if action_mask.dim() == 2:
                valid_actions = torch.where(action_mask[0])[0]
            else:
                valid_actions = torch.where(action_mask)[0]
            
            print(f"    - 有效动作: {valid_actions.tolist()}")
            print(f"    - 动作概率:")
            for action_idx in valid_actions:
                action = action_idx.item()
                prob = action_probs[action_idx].item()
                print(f"      动作 {action}: {prob:.3f}")
            
            # 扩展
            node.expand(action_probs, action_mask, env)
            print(f"  → 创建了 {len(node.children)} 个子节点")
            
            # Backpropagation
            print(f"\n[Backpropagation] 回传值: {value:.4f}")
            node.backpropagate(value)
            print(f"  路径: root" + " → ".join([f" → {a}" for a in path]))
        
        # 如果已经到达终止状态
        if node.state['done'].item():
            print(f"\n[深度 {depth}] 到达终止状态")
            value = mcts._get_value(node)
            print(f"  终止节点值: {value:.4f}")
    
    # 显示最终的根节点统计
    print_separator("=")
    print("搜索完成 - 根节点统计")
    print_separator("=")
    
    print(f"\n根节点被访问 {root.visit_count} 次")
    print(f"所有子节点的访问统计:\n")
    
    # 按访问次数排序
    children_stats = []
    for action, child in root.children.items():
        children_stats.append({
            'action': action,
            'visits': child.visit_count,
            'value': child.value,
            'prior': child.prior
        })
    
    children_stats.sort(key=lambda x: x['visits'], reverse=True)
    
    print(f"{'动作':<8} {'访问次数':<12} {'平均值':<12} {'先验概率':<12}")
    print("-" * 50)
    for stat in children_stats:
        print(f"{stat['action']:<8} {stat['visits']:<12} {stat['value']:<12.4f} {stat['prior']:<12.3f}")
    
    # 选择最终动作
    print_separator("-")
    print("最终决策")
    print_separator("-")
    
    best_action = max(root.children.items(), key=lambda x: x[1].visit_count)
    print(f"\n根据访问次数选择动作: {best_action[0]}")
    print(f"  - 访问次数: {best_action[1].visit_count}")
    print(f"  - 平均值: {best_action[1].value:.4f}")
    
    # 显示最终的UCB分数（用于对比）
    print(f"\n当前所有动作的UCB分数:")
    for action, child in sorted(root.children.items()):
        score = child.value + mcts.c_puct * child.prior * math.sqrt(root.visit_count) / (1 + child.visit_count)
        print(f"  动作 {action}: {score:.4f} (Q={child.value:.4f}, visits={child.visit_count})")
    
    print_separator("=")

def main():
    """主函数"""
    # 创建小规模TSP问题便于观察
    problem_size = 5
    env = TSPEnv(generator_params={'num_loc': problem_size})
    
    # 可视化一步决策
    visualize_mcts_step(env, num_simulations=5, problem_size=problem_size)
    
    print("\n" + "="*80)
    print("提示: 可以调整参数来观察不同的行为:")
    print("  - 增加 num_simulations 查看更多模拟")
    print("  - 修改 c_puct 观察探索vs利用的权衡")
    print("  - 改变 problem_size 测试不同规模的问题")
    print("="*80)

if __name__ == "__main__":
    main()
