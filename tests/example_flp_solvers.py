#!/usr/bin/env python
"""
FLP Solvers 快速使用示例
展示如何在 RL4CO 中使用 Gurobi, SCIP, GA 求解 FLP
"""
import numpy as np
from rl4co.envs.graph.flp import FLPEnv
from rl4co.envs.graph.flp.solvers import solve_flp

print("="*70)
print("🚀 FLP Solvers 快速使用示例")
print("="*70)

# ============================================================================
# 示例 1: 基本使用
# ============================================================================
print("\n" + "="*70)
print("示例 1: 基本使用")
print("="*70)

# 生成随机位置
np.random.seed(42)
locations = np.random.rand(20, 2)
to_choose = 5

print(f"\n问题: 从 {len(locations)} 个位置中选择 {to_choose} 个设施")

# 使用 Gurobi
print(f"\n1️⃣  Gurobi Solver:")
selected, obj_val, info = solve_flp(locations, to_choose, method='gurobi')
print(f"  - Selected: {selected}")
print(f"  - Objective: {obj_val:.4f}")
print(f"  - Time: {info['solve_time']:.3f}s")

# 使用 SCIP
print(f"\n2️⃣  SCIP Solver:")
selected, obj_val, info = solve_flp(locations, to_choose, method='scip')
print(f"  - Selected: {selected}")
print(f"  - Objective: {obj_val:.4f}")
print(f"  - Time: {info['solve_time']:.3f}s")

# 使用 GA
print(f"\n3️⃣  GA Solver:")
selected, obj_val, info = solve_flp(
    locations, to_choose, 
    method='ga',
    population_size=50,
    generations=100
)
print(f"  - Selected: {selected}")
print(f"  - Objective: {obj_val:.4f}")
print(f"  - Time: {info['solve_time']:.3f}s")

# ============================================================================
# 示例 2: 与 RL4CO 环境集成
# ============================================================================
print("\n" + "="*70)
print("示例 2: 与 RL4CO 环境集成")
print("="*70)

# 创建环境
env = FLPEnv(generator_params={'num_loc': 20, 'min_loc': 0, 'max_loc': 1})
td = env.reset(batch_size=[5])

print(f"\n生成了 {td.batch_size[0]} 个 FLP 实例")

# 批量求解
print(f"\n使用 Gurobi 批量求解:")
for i in range(5):
    locs = td['locs'][i].cpu().numpy()
    k = td['to_choose'][i].item()
    
    selected, obj, info = solve_flp(locs, k, method='gurobi', verbose=False)
    print(f"  Instance {i+1}: obj={obj:.4f}, time={info['solve_time']:.3f}s")

# ============================================================================
# 示例 3: 对比不同 Solver
# ============================================================================
print("\n" + "="*70)
print("示例 3: Solver 性能对比")
print("="*70)

# 测试实例
test_locs = np.random.rand(30, 2)
test_k = 10

results = {}
for method in ['gurobi', 'scip', 'ga']:
    try:
        selected, obj, info = solve_flp(
            test_locs, test_k, 
            method=method,
            verbose=False
        )
        results[method] = {
            'obj': obj,
            'time': info['solve_time']
        }
    except Exception as e:
        results[method] = {'obj': None, 'error': str(e)}

print(f"\n{'Method':<10} {'Objective':<12} {'Time (s)':<10} {'vs Best':<10}")
print("-" * 50)

best_obj = min([r['obj'] for r in results.values() if r.get('obj') is not None])

for method, res in results.items():
    if res.get('obj') is not None:
        obj = res['obj']
        time_val = res['time']
        gap = (obj - best_obj) / best_obj * 100
        print(f"{method:<10} {obj:<12.4f} {time_val:<10.3f} {gap:>+8.2f}%")
    else:
        print(f"{method:<10} {'Failed':<12} {'-':<10} {'-':<10}")

# ============================================================================
# 示例 4: GA 参数调优
# ============================================================================
print("\n" + "="*70)
print("示例 4: GA 参数调优")
print("="*70)

ga_configs = [
    {'name': 'Fast', 'pop': 50, 'gen': 100},
    {'name': 'Balanced', 'pop': 100, 'gen': 200},
    {'name': 'Quality', 'pop': 200, 'gen': 300},
]

print(f"\n测试不同 GA 配置:")
for config in ga_configs:
    selected, obj, info = solve_flp(
        test_locs, test_k,
        method='ga',
        population_size=config['pop'],
        generations=config['gen'],
        verbose=False
    )
    
    print(f"  {config['name']:<10}: obj={obj:.4f}, time={info['solve_time']:.3f}s")

print("\n" + "="*70)
print("✅ 示例完成！")
print("="*70)

print(f"\n📚 更多信息:")
print(f"  - 查看 FLP_SOLVERS_GUIDE.md 了解详细用法")
print(f"  - 查看 test_flp_solvers.py 了解完整测试")
print(f"  - 查看 rl4co/envs/graph/flp/solvers.py 了解实现")

print("="*70)
