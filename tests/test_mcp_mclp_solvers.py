#!/usr/bin/env python
"""
MCP 和 MCLP Solvers 综合测试
"""
import numpy as np
import torch
from rl4co.envs.graph.mcp import MCPEnv
from rl4co.envs.graph.mclp import MCLPEnv
from rl4co.envs.graph.mcp.solvers import solve_mcp
from rl4co.envs.graph.mclp.solvers import solve_mclp

print("="*70)
print("🔬 MCP & MCLP Solvers 综合测试")
print("="*70)

# ============================================================================
# 1. MCP (Maximum Coverage Problem) 测试
# ============================================================================
print("\n" + "="*70)
print("1️⃣  MCP (Maximum Coverage Problem)")
print("="*70)

# 创建 MCP 环境
env_mcp = MCPEnv(generator_params={
    'num_items': 20,
    'num_sets': 15,
    'min_weight': 1.0,
    'max_weight': 10.0,
    'min_size': 2,
    'max_size': 8,
})

# 生成测试实例
print(f"\n生成 MCP 实例...")
td_mcp = env_mcp.reset(batch_size=[10])

print(f"  - Items: {td_mcp['weights'].shape[1]}")
print(f"  - Sets: {td_mcp['membership'].shape[1]}")
print(f"  - To choose: {td_mcp['n_sets_to_choose'][0].item()}")

# 测试实例 (第一个)
membership = td_mcp['membership'][0].cpu().numpy()
weights = td_mcp['weights'][0].cpu().numpy()
n_to_choose = td_mcp['n_sets_to_choose'][0].item()

# Gurobi
print(f"\n1️⃣  Gurobi:")
try:
    selected, obj, info = solve_mcp(membership, weights, n_to_choose, method='gurobi', verbose=False)
    print(f"  ✅ Selected sets: {selected}")
    print(f"  📊 Coverage: {obj:.2f}")
    print(f"  📈 Covered items: {info['covered_items']}/{len(weights)}")
    print(f"  ⏱️  Time: {info['solve_time']:.3f}s")
    gurobi_obj = obj
except Exception as e:
    print(f"  ❌ Failed: {e}")
    gurobi_obj = None

# SCIP  
print(f"\n2️⃣  SCIP:")
try:
    selected, obj, info = solve_mcp(membership, weights, n_to_choose, method='scip', verbose=False)
    print(f"  ✅ Selected sets: {selected}")
    print(f"  📊 Coverage: {obj:.2f}")
    print(f"  📈 Covered items: {info['covered_items']}/{len(weights)}")
    print(f"  ⏱️  Time: {info['solve_time']:.3f}s")
except Exception as e:
    print(f"  ❌ Failed: {e}")

# GA
print(f"\n3️⃣  GA:")
try:
    selected, obj, info = solve_mcp(membership, weights, n_to_choose, method='ga', 
                                     population_size=100, generations=100, verbose=False)
    print(f"  ✅ Selected sets: {selected}")
    print(f"  📊 Coverage: {obj:.2f}")
    print(f"  📈 Covered items: {info['covered_items']}/{len(weights)}")
    print(f"  ⏱️  Time: {info['solve_time']:.3f}s")
    if gurobi_obj is not None:
        gap = (gurobi_obj - obj) / gurobi_obj * 100 if gurobi_obj > 0 else 0
        print(f"  📉 Gap to Gurobi: {gap:.2f}%")
except Exception as e:
    print(f"  ❌ Failed: {e}")

# ============================================================================
# 2. MCLP (Maximum Covering Location Problem) 测试
# ============================================================================
print("\n" + "="*70)
print("2️⃣  MCLP (Maximum Covering Location Problem)")
print("="*70)

# 创建 MCLP 环境
env_mclp = MCLPEnv(generator_params={
    'num_demand': 30,
    'num_facility': 20,
    'num_facilities_to_select': 5,
    'min_demand': 1.0,
    'max_demand': 10.0,
    'coverage_radius': 0.25,
})

# 生成测试实例
print(f"\n生成 MCLP 实例...")
td_mclp = env_mclp.reset(batch_size=[10])

print(f"  - Demand points: {td_mclp['demand_locs'].shape[1]}")
print(f"  - Facilities: {td_mclp['facility_locs'].shape[1]}")
print(f"  - To select: {td_mclp['num_facilities_to_select'][0].item()}")
print(f"  - Coverage radius: {td_mclp['coverage_radius'][0].item():.2f}")

# 测试实例
demand_locs = td_mclp['demand_locs'][0].cpu().numpy()
demand_weights = td_mclp['demand_weights'][0].cpu().numpy()
facility_locs = td_mclp['facility_locs'][0].cpu().numpy()
coverage_radius = td_mclp['coverage_radius'][0].item()
num_to_select = td_mclp['num_facilities_to_select'][0].item()

# Gurobi
print(f"\n1️⃣  Gurobi:")
try:
    selected, obj, info = solve_mclp(demand_locs, demand_weights, facility_locs,
                                      coverage_radius, num_to_select, method='gurobi', verbose=False)
    print(f"  ✅ Selected facilities: {selected}")
    print(f"  📊 Coverage: {obj:.2f}")
    print(f"  📈 Covered demands: {info['covered_demands']}/{len(demand_weights)}")
    print(f"  📍 Coverage rate: {info['coverage_rate']*100:.1f}%")
    print(f"  ⏱️  Time: {info['solve_time']:.3f}s")
    gurobi_obj_mclp = obj
except Exception as e:
    print(f"  ❌ Failed: {e}")
    gurobi_obj_mclp = None

# SCIP
print(f"\n2️⃣  SCIP:")
try:
    selected, obj, info = solve_mclp(demand_locs, demand_weights, facility_locs,
                                      coverage_radius, num_to_select, method='scip', verbose=False)
    print(f"  ✅ Selected facilities: {selected}")
    print(f"  📊 Coverage: {obj:.2f}")
    print(f"  📈 Covered demands: {info['covered_demands']}/{len(demand_weights)}")
    print(f"  📍 Coverage rate: {info['coverage_rate']*100:.1f}%")
    print(f"  ⏱️  Time: {info['solve_time']:.3f}s")
except Exception as e:
    print(f"  ❌ Failed: {e}")

# GA
print(f"\n3️⃣  GA:")
try:
    selected, obj, info = solve_mclp(demand_locs, demand_weights, facility_locs,
                                      coverage_radius, num_to_select, method='ga',
                                      population_size=100, generations=100, verbose=False)
    print(f"  ✅ Selected facilities: {selected}")
    print(f"  📊 Coverage: {obj:.2f}")
    print(f"  📈 Covered demands: {info['covered_demands']}/{len(demand_weights)}")
    print(f"  �� Coverage rate: {info['coverage_rate']*100:.1f}%")
    print(f"  ⏱️  Time: {info['solve_time']:.3f}s")
    if gurobi_obj_mclp is not None:
        gap = (gurobi_obj_mclp - obj) / gurobi_obj_mclp * 100 if gurobi_obj_mclp > 0 else 0
        print(f"  📉 Gap to Gurobi: {gap:.2f}%")
except Exception as e:
    print(f"  ❌ Failed: {e}")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*70)
print("📊 总结")
print("="*70)

print(f"\n✅ 已实现的 Solver:")
print(f"  1. MCP (Maximum Coverage Problem)")
print(f"     - Gurobi (精确MIP)")
print(f"     - SCIP (开源MIP)")
print(f"     - GA (遗传算法)")
print(f"\n  2. MCLP (Maximum Covering Location Problem)")
print(f"     - Gurobi (精确MIP)")
print(f"     - SCIP (开源MIP)")
print(f"     - GA (遗传算法)")

print(f"\n💡 使用建议:")
print(f"  - 精确求解: 使用 Gurobi 或 SCIP")
print(f"  - 快速求解: 使用 GA (质量接近)")
print(f"  - RL Baseline: 使用 Gurobi 作为最优基准")

print("="*70)
