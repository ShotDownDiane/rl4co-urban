#!/usr/bin/env python
"""
FLP (Facility Location Problem) Gurobi Solver
演示如何为没有专门 Solver 的问题创建 Gurobi 集成
"""
import numpy as np
import torch
from rl4co.envs.graph.flp import FLPEnv

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

def solve_flp_with_gurobi(locations, to_choose, time_limit=60):
    """
    使用 Gurobi 求解 FLP
    
    Args:
        locations: [n, 2] numpy array of locations
        to_choose: number of facilities to choose
        time_limit: time limit in seconds
    
    Returns:
        selected_indices: indices of selected facilities
        obj_value: objective value (total distance)
    """
    if not GUROBI_AVAILABLE:
        raise ImportError("Gurobi is not available")
    
    n = len(locations)
    
    # 计算距离矩阵
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(locations[i] - locations[j])
    
    # 创建 Gurobi 模型
    model = gp.Model("FLP")
    model.setParam('OutputFlag', 0)  # 静音
    model.setParam('TimeLimit', time_limit)
    
    # 决策变量
    # x[i] = 1 if facility i is selected
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    
    # y[i,j] = 1 if location i is assigned to facility j
    y = model.addVars(n, n, vtype=GRB.BINARY, name="y")
    
    # 目标函数: 最小化总距离
    model.setObjective(
        gp.quicksum(dist_matrix[i, j] * y[i, j] for i in range(n) for j in range(n)),
        GRB.MINIMIZE
    )
    
    # 约束1: 恰好选择 to_choose 个设施
    model.addConstr(gp.quicksum(x[i] for i in range(n)) == to_choose, "choose")
    
    # 约束2: 每个位置必须分配到恰好一个设施
    for i in range(n):
        model.addConstr(gp.quicksum(y[i, j] for j in range(n)) == 1, f"assign_{i}")
    
    # 约束3: 只能分配到被选中的设施
    for i in range(n):
        for j in range(n):
            model.addConstr(y[i, j] <= x[j], f"open_{i}_{j}")
    
    # 求解
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        selected = [i for i in range(n) if x[i].X > 0.5]
        obj_value = model.objVal
        return selected, obj_value
    else:
        return None, None


def test_flp_gurobi():
    """测试 FLP Gurobi Solver"""
    print("="*70)
    print("🏭 FLP (Facility Location Problem) - Gurobi Solver")
    print("="*70)
    
    if not GUROBI_AVAILABLE:
        print("\n❌ Gurobi 不可用")
        print("   安装: pip install gurobipy")
        print("   License: https://www.gurobi.com/downloads/")
        return
    
    # 创建 FLP 环境
    env = FLPEnv(generator_params={'num_loc': 20, 'min_loc': 0, 'max_loc': 1})
    
    # 生成测试实例
    num_instances = 10
    td = env.reset(batch_size=[num_instances])
    
    print(f"\n测试配置:")
    print(f"  - 位置数: 20")
    print(f"  - 选择数: {td['to_choose'][0].item()}")
    print(f"  - 实例数: {num_instances}")
    print(f"  - Solver: Gurobi")
    
    # 对每个实例求解
    results = []
    solve_times = []
    
    print(f"\n求解中...")
    import time
    for i in range(num_instances):
        locs = td['locs'][i].cpu().numpy()
        to_choose = td['to_choose'][i].item()
        
        start_time = time.time()
        selected, obj_val = solve_flp_with_gurobi(locs, to_choose, time_limit=10)
        solve_time = time.time() - start_time
        
        if selected is not None:
            results.append(obj_val)
            solve_times.append(solve_time)
            if (i + 1) % max(1, num_instances // 5) == 0:
                print(f"  Progress: {i+1}/{num_instances} | Obj: {obj_val:.4f} | Time: {solve_time:.3f}s")
    
    # 统计结果
    print(f"\n" + "="*70)
    print(f"结果:")
    print(f"="*70)
    print(f"  ✅ 成功率: {len(results)/num_instances*100:.0f}%")
    print(f"  📊 平均总距离: {np.mean(results):.4f} ± {np.std(results):.4f}")
    print(f"  📈 最小: {np.min(results):.4f}, 最大: {np.max(results):.4f}")
    print(f"  ⏱️  平均求解时间: {np.mean(solve_times):.3f}s")
    
    # 与随机策略对比
    print(f"\n对比随机策略:")
    random_results = []
    for i in range(num_instances):
        locs = td['locs'][i].cpu().numpy()
        to_choose = td['to_choose'][i].item()
        n = len(locs)
        
        # 随机选择设施
        selected = np.random.choice(n, to_choose, replace=False)
        
        # 计算目标值
        dist_matrix = np.linalg.norm(locs[:, None, :] - locs[None, :, :], axis=2)
        min_dists = dist_matrix[:, selected].min(axis=1)
        obj_val = min_dists.sum()
        random_results.append(obj_val)
    
    random_mean = np.mean(random_results)
    solver_mean = np.mean(results)
    
    print(f"  🤖 Gurobi Solver: {solver_mean:.4f}")
    print(f"  🎲 随机选择: {random_mean:.4f}")
    if solver_mean > 0:
        improvement = (random_mean - solver_mean) / random_mean * 100
        print(f"  📈 Solver 优化: {improvement:.1f}%")
    
    print("="*70)


if __name__ == '__main__':
    test_flp_gurobi()
