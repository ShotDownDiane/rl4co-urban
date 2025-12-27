#!/usr/bin/env python
"""
FLP Solvers 综合测试和对比
测试 Gurobi, SCIP, GA 三种求解器
"""
import numpy as np
import torch
import time
from rl4co.envs.graph.flp import FLPEnv
from rl4co.envs.graph.flp.solvers import solve_flp

print("="*70)
print("🔬 FLP Solvers 综合测试")
print("="*70)

# 测试配置
num_locations = 30
num_facilities = 10
num_instances = 20

# 创建 FLP 环境
env = FLPEnv(generator_params={
    'num_loc': num_locations,
    'min_loc': 0,
    'max_loc': 1
})

# 生成测试实例
print(f"\n📊 测试配置:")
print(f"  - 位置数: {num_locations}")
print(f"  - 设施数: {num_facilities}")
print(f"  - 实例数: {num_instances}")

td = env.reset(batch_size=[num_instances])

# ============================================================================
# 1. Gurobi Solver
# ============================================================================
print("\n" + "="*70)
print("1️⃣  Gurobi Solver (精确求解)")
print("="*70)

try:
    gurobi_results = []
    gurobi_times = []
    gurobi_gaps = []
    
    print("\n求解中...")
    for i in range(num_instances):
        locs = td['locs'][i].cpu().numpy()
        
        selected, obj_val, info = solve_flp(
            locs, 
            num_facilities,
            method='gurobi',
            time_limit=10.0,
            verbose=False
        )
        
        if selected is not None:
            gurobi_results.append(obj_val)
            gurobi_times.append(info['solve_time'])
            gurobi_gaps.append(info.get('gap', 0.0))
            
            if (i + 1) % 5 == 0:
                print(f"  Progress: {i+1}/{num_instances} | Obj: {obj_val:.4f} | Time: {info['solve_time']:.3f}s")
    
    print(f"\n✅ 结果:")
    print(f"  - 成功率: {len(gurobi_results)/num_instances*100:.0f}%")
    print(f"  - 平均目标值: {np.mean(gurobi_results):.4f} ± {np.std(gurobi_results):.4f}")
    print(f"  - 最优性gap: {np.mean(gurobi_gaps)*100:.2f}%")
    print(f"  - 平均时间: {np.mean(gurobi_times):.3f}s")
    
    gurobi_available = True
    
except ImportError as e:
    print(f"\n⚠️  Gurobi 不可用: {e}")
    gurobi_available = False
except Exception as e:
    print(f"\n❌ Gurobi 测试失败: {e}")
    gurobi_available = False

# ============================================================================
# 2. SCIP Solver
# ============================================================================
print("\n" + "="*70)
print("2️⃣  SCIP Solver (开源精确求解)")
print("="*70)

try:
    scip_results = []
    scip_times = []
    scip_gaps = []
    
    print("\n求解中...")
    for i in range(num_instances):
        locs = td['locs'][i].cpu().numpy()
        
        selected, obj_val, info = solve_flp(
            locs,
            num_facilities,
            method='scip',
            time_limit=10.0,
            verbose=False
        )
        
        if selected is not None:
            scip_results.append(obj_val)
            scip_times.append(info['solve_time'])
            scip_gaps.append(info.get('gap', 0.0))
            
            if (i + 1) % 5 == 0:
                print(f"  Progress: {i+1}/{num_instances} | Obj: {obj_val:.4f} | Time: {info['solve_time']:.3f}s")
    
    print(f"\n✅ 结果:")
    print(f"  - 成功率: {len(scip_results)/num_instances*100:.0f}%")
    print(f"  - 平均目标值: {np.mean(scip_results):.4f} ± {np.std(scip_results):.4f}")
    print(f"  - 最优性gap: {np.mean(scip_gaps)*100:.2f}%")
    print(f"  - 平均时间: {np.mean(scip_times):.3f}s")
    
    scip_available = True
    
except ImportError as e:
    print(f"\n⚠️  SCIP 不可用: {e}")
    scip_available = False
except Exception as e:
    print(f"\n❌ SCIP 测试失败: {e}")
    scip_available = False

# ============================================================================
# 3. Genetic Algorithm
# ============================================================================
print("\n" + "="*70)
print("3️⃣  Genetic Algorithm (启发式)")
print("="*70)

try:
    ga_results = []
    ga_times = []
    
    print("\n求解中...")
    for i in range(num_instances):
        locs = td['locs'][i].cpu().numpy()
        
        selected, obj_val, info = solve_flp(
            locs,
            num_facilities,
            method='ga',
            time_limit=10.0,
            verbose=False,
            population_size=100,
            generations=200,
            crossover_rate=0.8,
            mutation_rate=0.1
        )
        
        if selected is not None:
            ga_results.append(obj_val)
            ga_times.append(info['solve_time'])
            
            if (i + 1) % 5 == 0:
                print(f"  Progress: {i+1}/{num_instances} | Obj: {obj_val:.4f} | Time: {info['solve_time']:.3f}s | Gens: {info['generations']}")
    
    print(f"\n✅ 结果:")
    print(f"  - 成功率: {len(ga_results)/num_instances*100:.0f}%")
    print(f"  - 平均目标值: {np.mean(ga_results):.4f} ± {np.std(ga_results):.4f}")
    print(f"  - 平均时间: {np.mean(ga_times):.3f}s")
    
    ga_available = True
    
except Exception as e:
    print(f"\n❌ GA 测试失败: {e}")
    ga_available = False

# ============================================================================
# 4. 随机 Baseline
# ============================================================================
print("\n" + "="*70)
print("4️⃣  Random Baseline")
print("="*70)

random_results = []

for i in range(num_instances):
    locs = td['locs'][i].cpu().numpy()
    selected = np.random.choice(num_locations, num_facilities, replace=False)
    
    # 计算目标值
    dist_matrix = np.linalg.norm(
        locs[:, None, :] - locs[None, :, :],
        axis=2
    )
    min_dists = dist_matrix[:, selected].min(axis=1)
    obj_val = min_dists.sum()
    random_results.append(obj_val)

print(f"\n✅ 结果:")
print(f"  - 平均目标值: {np.mean(random_results):.4f} ± {np.std(random_results):.4f}")

# ============================================================================
# 性能对比
# ============================================================================
print("\n" + "="*70)
print("📊 性能对比")
print("="*70)

results_table = []

if gurobi_available:
    gurobi_mean = np.mean(gurobi_results)
    gurobi_time = np.mean(gurobi_times)
    gurobi_vs_random = (np.mean(random_results) - gurobi_mean) / np.mean(random_results) * 100
    results_table.append(('Gurobi', gurobi_mean, gurobi_time, gurobi_vs_random, '精确'))

if scip_available:
    scip_mean = np.mean(scip_results)
    scip_time = np.mean(scip_times)
    scip_vs_random = (np.mean(random_results) - scip_mean) / np.mean(random_results) * 100
    results_table.append(('SCIP', scip_mean, scip_time, scip_vs_random, '精确'))

if ga_available:
    ga_mean = np.mean(ga_results)
    ga_time = np.mean(ga_times)
    ga_vs_random = (np.mean(random_results) - ga_mean) / np.mean(random_results) * 100
    results_table.append(('GA', ga_mean, ga_time, ga_vs_random, '启发式'))

random_mean = np.mean(random_results)
results_table.append(('Random', random_mean, 0.0, 0.0, 'Baseline'))

print(f"\n{'Solver':<12} {'Obj Value':<12} {'Time (s)':<10} {'vs Random':<12} {'Type':<10}")
print("-" * 70)
for name, obj, time_val, improvement, solver_type in results_table:
    print(f"{name:<12} {obj:<12.4f} {time_val:<10.3f} {improvement:>+10.1f}% {solver_type:<10}")

# 质量对比
if gurobi_available and ga_available:
    print(f"\n💡 质量对比 (以 Gurobi 为基准):")
    if scip_available:
        scip_gap = (scip_mean - gurobi_mean) / gurobi_mean * 100
        print(f"  - SCIP vs Gurobi: {scip_gap:+.2f}%")
    ga_gap = (ga_mean - gurobi_mean) / gurobi_mean * 100
    print(f"  - GA vs Gurobi: {ga_gap:+.2f}%")

# 速度对比
if gurobi_available and ga_available:
    print(f"\n⏱️  速度对比:")
    print(f"  - Gurobi: {gurobi_time:.3f}s")
    if scip_available:
        print(f"  - SCIP: {scip_time:.3f}s ({scip_time/gurobi_time:.1f}x)")
    print(f"  - GA: {ga_time:.3f}s ({ga_time/gurobi_time:.1f}x)")

# ============================================================================
# 总结和建议
# ============================================================================
print("\n" + "="*70)
print("💡 总结和建议")
print("="*70)

print(f"\n✅ 可用的 Solver:")
if gurobi_available:
    print(f"  ✓ Gurobi - 商业MIP solver, 高精度高速度")
if scip_available:
    print(f"  ✓ SCIP - 开源MIP solver, 精确求解")
if ga_available:
    print(f"  ✓ GA - 遗传算法, 快速启发式")

print(f"\n📊 推荐使用场景:")
print(f"  - 需要最优解: Gurobi (商业) 或 SCIP (开源)")
print(f"  - 快速求解/大规模: GA (启发式)")
print(f"  - 研究/对比: 使用多种 Solver 对比")

print(f"\n🚀 集成到 RL 训练:")
print(f"  1. Baseline: 使用 Gurobi/SCIP 作为最优 baseline")
print(f"  2. Expert data: 用 Solver 解训练 RL 模型")
print(f"  3. Warm start: GA 快速初始化 + RL 精细优化")

print("="*70)
