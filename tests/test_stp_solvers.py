#!/usr/bin/env python
"""
STP Solvers 综合测试
"""
import numpy as np
import torch
from rl4co.envs.graph.stp import STPEnv
from rl4co.envs.graph.stp.solvers import solve_stp

print("="*70)
print("🔬 STP (Steiner Tree Problem) Solvers 测试")
print("="*70)

# 创建 STP 环境
env = STPEnv(generator_params={
    'num_nodes': 20,
    'num_terminals': 5,
    'graph_type': 'delaunay',  # 使用 Delaunay 图
})

# 生成测试实例
print(f"\n生成 STP 实例...")
td = env.reset(batch_size=[5])

print(f"  - Nodes: {td['locs'].shape[1]}")
print(f"  - Terminals: {td['terminals'].shape[1]}")
print(f"  - Edges: {td['num_edges'][0].item()}")
print(f"  - Graph type: delaunay")

# 提取第一个实例的数据
locs = td['locs'][0].cpu().numpy()
terminals = td['terminals'][0].cpu().numpy()
edge_list_full = td['edge_list'][0].cpu().numpy()
edge_weights = td['edge_weights'][0].cpu().numpy()
num_edges = int(td['num_edges'][0].item())

# 只保留有效的边
edge_list = edge_list_full[:num_edges]

print(f"\n实例信息:")
print(f"  - Nodes: {len(locs)}")
print(f"  - Terminals: {terminals}")
print(f"  - Edges: {len(edge_list)}")

# ============================================================================
# 1. MST-based Approximation
# ============================================================================
print(f"\n" + "="*70)
print(f"1️⃣  MST-based Approximation (2-近似)")
print("="*70)

try:
    selected, obj, info = solve_stp(
        locs, terminals, edge_list, edge_weights,
        method='mst',
        verbose=False
    )
    
    print(f"  ✅ Selected edges: {len(selected)}")
    print(f"  📊 Total weight: {obj:.4f}")
    print(f"  ⏱️  Time: {info['solve_time']:.3f}s")
    print(f"  📈 Algorithm: {info['algorithm']}")
    mst_obj = obj
except Exception as e:
    print(f"  ❌ Failed: {e}")
    mst_obj = None

# ============================================================================
# 2. Gurobi Solver
# ============================================================================
print(f"\n" + "="*70)
print(f"2️⃣  Gurobi Solver (精确求解)")
print("="*70)

try:
    selected, obj, info = solve_stp(
        locs, terminals, edge_list, edge_weights,
        method='gurobi',
        time_limit=30.0,
        verbose=False
    )
    
    print(f"  ✅ Selected edges: {len(selected)}")
    print(f"  📊 Total weight: {obj:.4f}")
    if 'num_nodes' in info:
        print(f"  📈 Nodes in tree: {info['num_nodes']}")
    print(f"  ⏱️  Time: {info['solve_time']:.3f}s")
    if 'method' in info:
        print(f"  🎯 Method: {info['method']}")
    else:
        print(f"  🎯 Status: {info['status']}")
    
    if mst_obj is not None:
        gap = (mst_obj - obj) / obj * 100 if obj > 0 else 0
        print(f"  📉 MST approximation ratio: {gap:.2f}%")
    
    gurobi_obj = obj
except ImportError as e:
    print(f"  ⚠️  Gurobi not available: {e}")
    gurobi_obj = None
except Exception as e:
    print(f"  ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    gurobi_obj = None

# ============================================================================
# 3. Genetic Algorithm
# ============================================================================
print(f"\n" + "="*70)
print(f"3️⃣  Genetic Algorithm")
print("="*70)

try:
    selected, obj, info = solve_stp(
        locs, terminals, edge_list, edge_weights,
        method='ga',
        time_limit=10.0,
        population_size=50,
        generations=50,
        verbose=False
    )
    
    print(f"  ✅ Selected edges: {len(selected)}")
    print(f"  📊 Total weight: {obj:.4f}")
    print(f"  ⏱️  Time: {info['solve_time']:.3f}s")
    print(f"  🔄 Generations: {info['generations']}")
    
    if gurobi_obj is not None:
        gap = (obj - gurobi_obj) / gurobi_obj * 100 if gurobi_obj > 0 else 0
        print(f"  📉 Gap to Gurobi: {gap:.2f}%")
    elif mst_obj is not None:
        gap = (obj - mst_obj) / mst_obj * 100 if mst_obj > 0 else 0
        print(f"  📉 Gap to MST: {gap:.2f}%")
    
except Exception as e:
    print(f"  ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 总结
# ============================================================================
print(f"\n" + "="*70)
print(f"📊 总结")
print("="*70)

print(f"\n✅ 已实现的 Solver:")
print(f"  1. MST-based Approximation - 2-近似算法")
print(f"  2. Gurobi - 精确MIP求解")
print(f"  3. Genetic Algorithm - 元启发式")

print(f"\n💡 使用建议:")
print(f"  - 快速求解: MST-based (2-近似)")
print(f"  - 精确求解: Gurobi (需要license)")
print(f"  - 大规模问题: GA (可扩展)")

print("="*70)
