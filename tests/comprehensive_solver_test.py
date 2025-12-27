#!/usr/bin/env python
"""
综合测试: 所有图问题的 Solver 性能
"""
import torch
import numpy as np
import time
from rl4co.envs.graph import MISEnvWrapper
from rl4co.envs.graph.flp import FLPEnv

# 导入 Gurobi (for FLP)
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

def solve_flp_gurobi(locations, to_choose, time_limit=10):
    """使用 Gurobi 求解 FLP"""
    n = len(locations)
    dist_matrix = np.linalg.norm(locations[:, None, :] - locations[None, :, :], axis=2)
    
    model = gp.Model("FLP")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    y = model.addVars(n, n, vtype=GRB.BINARY, name="y")
    
    model.setObjective(
        gp.quicksum(dist_matrix[i, j] * y[i, j] for i in range(n) for j in range(n)),
        GRB.MINIMIZE
    )
    
    model.addConstr(gp.quicksum(x[i] for i in range(n)) == to_choose)
    for i in range(n):
        model.addConstr(gp.quicksum(y[i, j] for j in range(n)) == 1)
        for j in range(n):
            model.addConstr(y[i, j] <= x[j])
    
    model.optimize()
    return model.objVal if model.status == GRB.OPTIMAL else None

def greedy_baseline(env, td):
    """贪心策略作为 baseline"""
    results = []
    for i in range(td.batch_size[0]):
        env_single = env.__class__(num_nodes=env.num_nodes if hasattr(env, 'num_nodes') else 20)
        td_single = env_single.reset(batch_size=[1])
        count = 0
        while not td_single["done"].item() and count < 100:
            avail = td_single["available"][0].nonzero(as_tuple=True)[0] if "available" in td_single else torch.tensor([0])
            if len(avail) == 0:
                break
            td_single["action"] = torch.tensor([avail[0].item()])
            td_single = env_single._step(td_single)
            count += 1
        results.append(count)
    return np.mean(results)

print("="*70)
print("🔬 综合 Solver 性能测试")
print("="*70)

# ============================================================================
# 1. MIS - KaMIS Solver
# ============================================================================
print("\n" + "="*70)
print("1️⃣  MIS (Maximum Independent Set)")
print("="*70)

env_mis = MISEnvWrapper(num_nodes=30, edge_prob=0.2, node_weighted=True)
td_mis = env_mis.generate_data(batch_size=20)

print(f"\n配置:")
print(f"  - 问题: MIS (Maximum Independent Set)")
print(f"  - 节点数: 30")
print(f"  - 边概率: 0.2")
print(f"  - 带权重: Yes")
print(f"  - Solver: {type(env_mis.ml4co_solver).__name__}")

start = time.time()
results_mis = env_mis.solve_with_ml4co(td_mis, verbose=False, return_solutions=True)
total_time = time.time() - start

print(f"\n结果:")
print(f"  ✅ 成功率: {results_mis['success_rate']:.0f}%")
print(f"  📊 平均权重: {results_mis['statistics']['mean']:.4f}")

# 检查解
if results_mis['solutions'][0] is not None:
    sol = results_mis['solutions'][0]
    num_selected = sol.sum()
    print(f"  💡 示例解: 选择了 {num_selected} 个节点")
else:
    print(f"  ⚠️  解返回为 None")

print(f"  ⏱️  总时间: {total_time:.2f}s ({total_time/20:.3f}s per instance)")

# ============================================================================
# 2. FLP - Gurobi Solver
# ============================================================================
print("\n" + "="*70)
print("2️⃣  FLP (Facility Location Problem)")
print("="*70)

if GUROBI_AVAILABLE:
    env_flp = FLPEnv(generator_params={'num_loc': 20, 'min_loc': 0, 'max_loc': 1})
    td_flp = env_flp.reset(batch_size=[20])
    
    print(f"\n配置:")
    print(f"  - 问题: FLP (Facility Location Problem)")
    print(f"  - 位置数: 20")
    print(f"  - 选择数: {td_flp['to_choose'][0].item()}")
    print(f"  - Solver: Gurobi (MIP)")
    
    flp_results = []
    flp_times = []
    
    for i in range(20):
        locs = td_flp['locs'][i].cpu().numpy()
        to_choose = td_flp['to_choose'][i].item()
        
        start = time.time()
        obj_val = solve_flp_gurobi(locs, to_choose)
        solve_time = time.time() - start
        
        if obj_val is not None:
            flp_results.append(obj_val)
            flp_times.append(solve_time)
    
    print(f"\n结果:")
    print(f"  ✅ 成功率: {len(flp_results)/20*100:.0f}%")
    print(f"  📊 平均距离: {np.mean(flp_results):.4f} ± {np.std(flp_results):.4f}")
    print(f"  ⏱️  平均时间: {np.mean(flp_times):.3f}s per instance")
    
    # 随机 baseline
    random_results = []
    for i in range(20):
        locs = td_flp['locs'][i].cpu().numpy()
        to_choose = td_flp['to_choose'][i].item()
        selected = np.random.choice(len(locs), to_choose, replace=False)
        dist_matrix = np.linalg.norm(locs[:, None, :] - locs[None, :, :], axis=2)
        obj_val = dist_matrix[:, selected].min(axis=1).sum()
        random_results.append(obj_val)
    
    print(f"\n对比:")
    print(f"  🤖 Gurobi: {np.mean(flp_results):.4f}")
    print(f"  🎲 Random: {np.mean(random_results):.4f}")
    print(f"  📈 提升: {(np.mean(random_results) - np.mean(flp_results)) / np.mean(random_results) * 100:.1f}%")
else:
    print("\n⚠️  Gurobi 不可用，跳过 FLP 测试")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*70)
print("📊 总结")
print("="*70)

print(f"\n✅ 测试的问题和 Solver:")
print(f"  1. MIS: KaMIS Solver (专用, 开源)")
print(f"  2. FLP: Gurobi Solver (通用 MIP, 商业)")

print(f"\n💡 Solver 选择建议:")
print(f"  - 图问题 (MIS/MVC/MCL):")
print(f"    • KaMIS (MIS) - 最快，专门优化")
print(f"    • Gurobi/SCIP - 通用 MIP solver")
print(f"  - 组合优化 (FLP/TSP/CVRP):")
print(f"    • Gurobi - 商业，高性能")
print(f"    • SCIP - 开源，免费")
print(f"    • LKH/HGS - 专用启发式")

print(f"\n🚀 RL4CO + ML4CO-Kit 集成优势:")
print(f"  ✅ 统一的接口: env.solve_with_ml4co()")
print(f"  ✅ 多种 Solver: 根据问题选择最优 Solver")
print(f"  ✅ Baseline 对比: Solver 作为 RL 训练的 baseline")
print(f"  ✅ Expert data: Solver 解用于模仿学习")

print("="*70)
