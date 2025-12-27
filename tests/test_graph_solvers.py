#!/usr/bin/env python
"""
测试不同 Solver 在图问题上的表现
"""
import torch
import numpy as np
from rl4co.envs.graph import MISEnvWrapper, MVCEnvWrapper, MCLEnvWrapper

print("="*70)
print("🔬 测试图问题 Solver 性能对比")
print("="*70)

# 测试配置
num_nodes = 30
edge_prob = 0.2
num_instances = 20

# ============================================================================
# 1. MIS (Maximum Independent Set) - 使用 KaMIS
# ============================================================================
print("\n" + "="*70)
print("1️⃣  MIS (Maximum Independent Set) - KaMIS Solver")
print("="*70)

env_mis = MISEnvWrapper(num_nodes=num_nodes, edge_prob=edge_prob, node_weighted=False)
td_mis = env_mis.generate_data(batch_size=num_instances)

print(f"\n测试配置:")
print(f"  - 节点数: {num_nodes}")
print(f"  - 边概率: {edge_prob}")
print(f"  - 实例数: {num_instances}")
print(f"  - Solver: {type(env_mis.ml4co_solver).__name__}")

results_mis = env_mis.solve_with_ml4co(td_mis, verbose=False, return_solutions=True)

print(f"\n结果:")
print(f"  ✅ 成功率: {results_mis['success_rate']:.0f}%")
print(f"  📊 平均选择节点数: {results_mis['statistics']['mean']:.2f} ± {results_mis['statistics']['std']:.2f}")
print(f"  📈 最大: {results_mis['statistics']['max']:.0f}, 最小: {results_mis['statistics']['min']:.0f}")
print(f"  ⏱️  平均求解时间: {results_mis['timing']['mean_per_instance']:.3f}s")

# 与贪心策略对比
print(f"\n对比贪心策略:")
greedy_results = []
for i in range(min(10, num_instances)):
    env_single = MISEnvWrapper(num_nodes=num_nodes, edge_prob=edge_prob)
    td_single = env_single.reset(batch_size=[1])
    count = 0
    while not td_single["done"].item():
        avail = td_single["available"][0].nonzero(as_tuple=True)[0]
        if len(avail) == 0:
            break
        td_single["action"] = torch.tensor([avail[0].item()])
        td_single = env_single._step(td_single)
        count += 1
    greedy_results.append(count)

greedy_mean = np.mean(greedy_results)
print(f"  🤖 KaMIS Solver: {results_mis['statistics']['mean']:.2f} 节点")
print(f"  🎲 贪心策略: {greedy_mean:.2f} 节点")
if results_mis['statistics']['mean'] > 0 and greedy_mean > 0:
    improvement = (results_mis['statistics']['mean'] - greedy_mean) / greedy_mean * 100
    print(f"  📈 Solver 提升: {improvement:+.1f}%")

# ============================================================================
# 2. MVC (Minimum Vertex Cover) - 使用 Gurobi
# ============================================================================
print("\n" + "="*70)
print("2️⃣  MVC (Minimum Vertex Cover) - Gurobi Solver")
print("="*70)

try:
    from ml4co_kit.solver.gurobi import GurobiSolver
    
    env_mvc = MVCEnvWrapper(num_nodes=num_nodes, edge_prob=edge_prob, node_weighted=False)
    
    # 替换为 Gurobi Solver
    env_mvc.ml4co_solver = GurobiSolver()
    
    td_mvc = env_mvc.generate_data(batch_size=10)  # 少一点，因为 Gurobi 可能较慢
    
    print(f"\n测试配置:")
    print(f"  - 节点数: {num_nodes}")
    print(f"  - 边概率: {edge_prob}")
    print(f"  - 实例数: 10")
    print(f"  - Solver: {type(env_mvc.ml4co_solver).__name__}")
    
    results_mvc = env_mvc.solve_with_ml4co(td_mvc, verbose=False)
    
    print(f"\n结果:")
    print(f"  ✅ 成功率: {results_mvc['success_rate']:.0f}%")
    print(f"  📊 平均覆盖节点数: {results_mvc['statistics']['mean']:.2f} ± {results_mvc['statistics']['std']:.2f}")
    print(f"  ⏱️  平均求解时间: {results_mvc['timing']['mean_per_instance']:.3f}s")
    
    # MIS 和 MVC 的关系: MVC(G) = n - MIS(G)
    print(f"\n✨ 理论验证 (MVC + MIS = n):")
    print(f"  - MIS: ~{results_mis['statistics']['mean']:.0f} 节点")
    print(f"  - MVC: ~{results_mvc['statistics']['mean']:.0f} 节点") 
    print(f"  - 总计: ~{results_mis['statistics']['mean'] + results_mvc['statistics']['mean']:.0f} (期望: {num_nodes})")
    
except ImportError:
    print("⚠️  Gurobi 未安装或无 license")
except Exception as e:
    print(f"⚠️  MVC + Gurobi 测试失败: {e}")

# ============================================================================
# 3. MCL (Maximum Clique) - 使用 SCIP
# ============================================================================
print("\n" + "="*70)
print("3️⃣  MCL (Maximum Clique) - SCIP Solver")
print("="*70)

try:
    from ml4co_kit.solver.scip import SCIPSolver
    
    env_mcl = MCLEnvWrapper(num_nodes=num_nodes, edge_prob=edge_prob, node_weighted=False)
    
    # 替换为 SCIP Solver
    env_mcl.ml4co_solver = SCIPSolver()
    
    td_mcl = env_mcl.generate_data(batch_size=10)
    
    print(f"\n测试配置:")
    print(f"  - 节点数: {num_nodes}")
    print(f"  - 边概率: {edge_prob}")
    print(f"  - 实例数: 10")
    print(f"  - Solver: {type(env_mcl.ml4co_solver).__name__}")
    
    results_mcl = env_mcl.solve_with_ml4co(td_mcl, verbose=False)
    
    print(f"\n结果:")
    print(f"  ✅ 成功率: {results_mcl['success_rate']:.0f}%")
    print(f"  📊 平均团大小: {results_mcl['statistics']['mean']:.2f} ± {results_mcl['statistics']['std']:.2f}")
    print(f"  ⏱️  平均求解时间: {results_mcl['timing']['mean_per_instance']:.3f}s")
    
except ImportError:
    print("⚠️  SCIP 未安装")
except Exception as e:
    print(f"⚠️  MCL + SCIP 测试失败: {e}")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*70)
print("📊 Solver 性能总结")
print("="*70)

print(f"\n✅ 已测试的 Solver:")
print(f"  1. KaMIS (MIS专用) - 高性能，专门优化")
print(f"  2. Gurobi (通用MIP) - 商业求解器，需要 license")
print(f"  3. SCIP (开源MIP) - 开源替代方案")

print(f"\n💡 建议:")
print(f"  - MIS: 使用 KaMIS (最快，专门优化)")
print(f"  - MVC/MCL: 使用 Gurobi 或 SCIP")
print(f"  - FLP: 可以使用 Gurobi/SCIP 建模")

print("="*70)
