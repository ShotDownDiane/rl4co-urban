#!/usr/bin/env python
"""
TSP Solvers 综合测试
"""
import numpy as np
import torch
from rl4co.envs import TSPEnv
from rl4co.envs.routing.tsp.solvers import solve_tsp

print("="*70)
print("🔬 TSP (Traveling Salesman Problem) Solvers 测试")
print("="*70)

# 创建 TSP 环境
env = TSPEnv(generator_params={'num_loc': 20})

# 生成测试实例
print(f"\n生成 TSP 实例...")
td = env.reset(batch_size=[5])

print(f"  - Nodes: {td['locs'].shape[1]}")

# 提取第一个实例
locs = td['locs'][0].cpu().numpy()

print(f"\n实例信息:")
print(f"  - Nodes: {len(locs)}")

results = {}

# ============================================================================
# 1. Greedy (Nearest Neighbor)
# ============================================================================
print(f"\n" + "="*70)
print(f"1️⃣  Nearest Neighbor (Greedy)")
print("="*70)

try:
    tour, obj, info = solve_tsp(locs, method='greedy')
    
    print(f"  ✅ Tour length: {len(tour)}")
    print(f"  📊 Distance: {obj:.4f}")
    print(f"  ⏱️  Time: {info['solve_time']:.3f}s")
    
    results['greedy'] = obj
except Exception as e:
    print(f"  ❌ Failed: {e}")

# ============================================================================
# 2. Genetic Algorithm
# ============================================================================
print(f"\n" + "="*70)
print(f"2️⃣  Genetic Algorithm")
print("="*70)

try:
    tour, obj, info = solve_tsp(
        locs, 
        method='ga',
        time_limit=5.0,
        population_size=50,
        generations=100,
        verbose=False
    )
    
    print(f"  ✅ Tour length: {len(tour)}")
    print(f"  📊 Distance: {obj:.4f}")
    print(f"  ⏱️  Time: {info['solve_time']:.3f}s")
    print(f"  🔄 Generations: {info['generations']}")
    
    if 'greedy' in results:
        improvement = (results['greedy'] - obj) / results['greedy'] * 100
        print(f"  📈 vs Greedy: {improvement:+.2f}%")
    
    results['ga'] = obj
except Exception as e:
    print(f"  ❌ Failed: {e}")

# ============================================================================
# 3. Gurobi
# ============================================================================
print(f"\n" + "="*70)
print(f"3️⃣  Gurobi MIP Solver")
print("="*70)

try:
    tour, obj, info = solve_tsp(locs, method='gurobi', time_limit=30.0, verbose=False)
    
    print(f"  ✅ Tour length: {len(tour)}")
    print(f"  📊 Distance: {obj:.4f}")
    print(f"  ⏱️  Time: {info['solve_time']:.3f}s")
    print(f"  🎯 Status: {info['status']}")
    
    if 'greedy' in results:
        improvement = (results['greedy'] - obj) / results['greedy'] * 100
        print(f"  📈 vs Greedy: {improvement:+.2f}%")
    
    results['gurobi'] = obj
except Exception as e:
    print(f"  ⚠️  Failed: {e}")

# ============================================================================
# 4. OR-Tools
# ============================================================================
print(f"\n" + "="*70)
print(f"4️⃣  Google OR-Tools")
print("="*70)

try:
    tour, obj, info = solve_tsp(locs, method='ortools', time_limit=10.0, verbose=False)
    
    print(f"  ✅ Tour length: {len(tour)}")
    print(f"  📊 Distance: {obj:.4f}")
    print(f"  ⏱️  Time: {info['solve_time']:.3f}s")
    
    if 'greedy' in results:
        improvement = (results['greedy'] - obj) / results['greedy'] * 100
        print(f"  📈 vs Greedy: {improvement:+.2f}%")
    
    results['ortools'] = obj
except Exception as e:
    print(f"  ⚠️  Failed: {e}")

# ============================================================================
# 5. LKH
# ============================================================================
print(f"\n" + "="*70)
print(f"5️⃣  LKH Solver (最强启发式)")
print("="*70)

try:
    tour, obj, info = solve_tsp(locs, method='lkh', time_limit=10.0, verbose=False)
    
    if tour is not None and obj is not None:
        print(f"  ✅ Tour length: {len(tour)}")
        print(f"  📊 Distance: {obj:.4f}")
        print(f"  ⏱️  Time: {info['solve_time']:.3f}s")
        
        if 'greedy' in results:
            improvement = (results['greedy'] - obj) / results['greedy'] * 100
            print(f"  📈 vs Greedy: {improvement:+.2f}%")
        
        results['lkh'] = obj
    else:
        print(f"  ⚠️  LKH返回空解")
        print(f"  💡 提示: LKH可能安装成功但求解失败")
except Exception as e:
    print(f"  ⚠️  Failed: {e}")
    import traceback
    if "ML4CO-Kit" in str(e) or "LKH" in str(e):
        print(f"  💡 提示: LKH 需要 ML4CO-Kit 和编译")

# ============================================================================
# 6. Concorde
# ============================================================================
print(f"\n" + "="*70)
print(f"6️⃣  Concorde (精确求解)")
print("="*70)

try:
    tour, obj, info = solve_tsp(locs, method='concorde', time_limit=30.0, verbose=False)
    
    if tour is not None and obj is not None:
        print(f"  ✅ Tour length: {len(tour)}")
        print(f"  📊 Distance: {obj:.4f}")
        print(f"  ⏱️  Time: {info['solve_time']:.3f}s")
        print(f"  🎯 Status: {info['status']}")
        
        if 'greedy' in results:
            improvement = (results['greedy'] - obj) / results['greedy'] * 100
            print(f"  📈 vs Greedy: {improvement:+.2f}%")
        
        results['concorde'] = obj
    else:
        print(f"  ⚠️  Concorde返回空解")
        print(f"  💡 提示: Concorde可能未正确安装")
except Exception as e:
    print(f"  ⚠️  Failed: {e}")

# ============================================================================
# 总结
# ============================================================================
print(f"\n" + "="*70)
print(f"📊 性能对比")
print("="*70)

if results:
    print(f"\n{'Solver':<15} {'Distance':<12} {'vs Best':<12} {'Type':<15}")
    print("-" * 70)
    
    best_obj = min(results.values())
    
    solver_info = {
        'greedy': ('Greedy', 'Heuristic'),
        'ga': ('GA', 'Metaheuristic'),
        'gurobi': ('Gurobi', 'MIP Exact'),
        'ortools': ('OR-Tools', 'Heuristic'),
        'lkh': ('LKH', 'Best Heuristic'),
        'concorde': ('Concorde', 'Exact'),
    }
    
    for solver, obj in sorted(results.items(), key=lambda x: x[1]):
        name, solver_type = solver_info.get(solver, (solver, 'Unknown'))
        gap = (obj - best_obj) / best_obj * 100 if best_obj > 0 else 0
        print(f"{name:<15} {obj:<12.4f} {gap:>+10.2f}% {solver_type:<15}")

print(f"\n" + "="*70)
print(f"💡 总结")
print("="*70)

print(f"\n✅ 已测试的 Solver:")
tested = []
if 'greedy' in results:
    tested.append("Greedy (Nearest Neighbor)")
if 'ga' in results:
    tested.append("Genetic Algorithm")
if 'gurobi' in results:
    tested.append("Gurobi")
if 'ortools' in results:
    tested.append("OR-Tools")
if 'lkh' in results:
    tested.append("LKH")
if 'concorde' in results:
    tested.append("Concorde")

for i, solver in enumerate(tested, 1):
    print(f"  {i}. {solver}")

print(f"\n💡 推荐使用:")
print(f"  - 最优解: LKH 或 Concorde")
print(f"  - 快速求解: OR-Tools 或 Greedy")
print(f"  - 无依赖: GA")
print(f"  - RL Baseline: LKH")

print("="*70)
