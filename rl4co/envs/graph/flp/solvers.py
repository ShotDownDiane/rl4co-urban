"""
FLP (Facility Location Problem) Solvers
支持多种求解算法：Gurobi, SCIP, GA
"""
import numpy as np
import time
from typing import Tuple, Optional

# ============================================================================
# Gurobi Solver
# ============================================================================

def solve_flp_gurobi(
    locations: np.ndarray, 
    to_choose: int, 
    time_limit: float = 60.0,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
    """
    使用 Gurobi 求解 FLP
    
    Args:
        locations: [n, 2] 位置坐标
        to_choose: 要选择的设施数量
        time_limit: 时间限制（秒）
        verbose: 是否打印详细信息
        
    Returns:
        selected: 选中的设施索引
        obj_value: 目标值（总距离）
        info: 额外信息（求解时间等）
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        raise ImportError("Gurobi is not available. Install: pip install gurobipy")
    
    start_time = time.time()
    n = len(locations)
    
    # 计算距离矩阵
    dist_matrix = np.linalg.norm(
        locations[:, None, :] - locations[None, :, :], 
        axis=2
    )
    
    # 创建模型
    model = gp.Model("FLP")
    model.setParam('OutputFlag', 1 if verbose else 0)
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
    model.addConstr(
        gp.quicksum(x[i] for i in range(n)) == to_choose, 
        "num_facilities"
    )
    
    # 约束2: 每个位置必须分配到恰好一个设施
    for i in range(n):
        model.addConstr(
            gp.quicksum(y[i, j] for j in range(n)) == 1, 
            f"assign_{i}"
        )
    
    # 约束3: 只能分配到被选中的设施
    for i in range(n):
        for j in range(n):
            model.addConstr(y[i, j] <= x[j], f"open_{i}_{j}")
    
    # 求解
    model.optimize()
    
    solve_time = time.time() - start_time
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        selected = np.array([i for i in range(n) if x[i].X > 0.5])
        obj_value = model.objVal
        
        info = {
            'solve_time': solve_time,
            'status': 'optimal' if model.status == GRB.OPTIMAL else 'time_limit',
            'gap': model.MIPGap if hasattr(model, 'MIPGap') else 0.0,
        }
        
        return selected, obj_value, info
    else:
        return None, None, {'solve_time': solve_time, 'status': 'failed'}


# ============================================================================
# SCIP Solver
# ============================================================================

def solve_flp_scip(
    locations: np.ndarray,
    to_choose: int,
    time_limit: float = 60.0,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
    """
    使用 SCIP 求解 FLP
    
    Args:
        locations: [n, 2] 位置坐标
        to_choose: 要选择的设施数量
        time_limit: 时间限制（秒）
        verbose: 是否打印详细信息
        
    Returns:
        selected: 选中的设施索引
        obj_value: 目标值（总距离）
        info: 额外信息
    """
    try:
        from pyscipopt import Model, quicksum
    except ImportError:
        raise ImportError("SCIP is not available. Install: pip install pyscipopt")
    
    start_time = time.time()
    n = len(locations)
    
    # 计算距离矩阵
    dist_matrix = np.linalg.norm(
        locations[:, None, :] - locations[None, :, :],
        axis=2
    )
    
    # 创建模型
    model = Model("FLP")
    if not verbose:
        model.hideOutput()
    model.setRealParam('limits/time', time_limit)
    
    # 决策变量
    x = {}  # x[i] = 1 if facility i is selected
    y = {}  # y[i,j] = 1 if location i assigned to facility j
    
    for i in range(n):
        x[i] = model.addVar(vtype="B", name=f"x_{i}")
        for j in range(n):
            y[i, j] = model.addVar(vtype="B", name=f"y_{i}_{j}")
    
    # 目标函数
    model.setObjective(
        quicksum(dist_matrix[i, j] * y[i, j] for i in range(n) for j in range(n)),
        "minimize"
    )
    
    # 约束1: 选择固定数量的设施
    model.addCons(quicksum(x[i] for i in range(n)) == to_choose, "num_facilities")
    
    # 约束2: 每个位置分配到一个设施
    for i in range(n):
        model.addCons(
            quicksum(y[i, j] for j in range(n)) == 1,
            f"assign_{i}"
        )
    
    # 约束3: 只能分配到被选中的设施
    for i in range(n):
        for j in range(n):
            model.addCons(y[i, j] <= x[j], f"open_{i}_{j}")
    
    # 求解
    model.optimize()
    
    solve_time = time.time() - start_time
    
    if model.getStatus() == 'optimal' or model.getStatus() == 'timelimit':
        selected = np.array([i for i in range(n) if model.getVal(x[i]) > 0.5])
        obj_value = model.getObjVal()
        
        info = {
            'solve_time': solve_time,
            'status': model.getStatus(),
            'gap': model.getGap(),
        }
        
        return selected, obj_value, info
    else:
        return None, None, {'solve_time': solve_time, 'status': model.getStatus()}


# ============================================================================
# Genetic Algorithm (GA) Solver
# ============================================================================

class FLPGeneticSolver:
    """遗传算法求解 FLP"""
    
    def __init__(
        self,
        population_size: int = 100,
        generations: int = 200,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elite_size: int = 10,
        verbose: bool = False
    ):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.verbose = verbose
    
    def _evaluate(self, individual: np.ndarray, locations: np.ndarray, dist_matrix: np.ndarray) -> float:
        """评估个体的适应度（总距离）"""
        selected = np.where(individual == 1)[0]
        if len(selected) == 0:
            return float('inf')
        
        # 计算每个位置到最近设施的距离
        min_dists = dist_matrix[:, selected].min(axis=1)
        return min_dists.sum()
    
    def _create_individual(self, n: int, to_choose: int) -> np.ndarray:
        """创建一个个体（随机选择 to_choose 个设施）"""
        individual = np.zeros(n, dtype=int)
        selected = np.random.choice(n, to_choose, replace=False)
        individual[selected] = 1
        return individual
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray, to_choose: int) -> np.ndarray:
        """交叉操作"""
        if np.random.rand() > self.crossover_rate:
            return parent1.copy()
        
        # 单点交叉
        child = np.zeros_like(parent1)
        point = np.random.randint(1, len(parent1))
        
        # 从parent1继承前半部分
        child[:point] = parent1[:point]
        
        # 从parent2继承剩余需要的设施
        selected_count = child.sum()
        if selected_count < to_choose:
            # 找到parent2中未被选中的设施
            available = np.where((parent2 == 1) & (child == 0))[0]
            need = to_choose - selected_count
            if len(available) >= need:
                add = np.random.choice(available, need, replace=False)
                child[add] = 1
            else:
                # 随机补充
                available_all = np.where(child == 0)[0]
                need_more = need - len(available)
                child[available] = 1
                if need_more > 0 and len(available_all) > len(available):
                    others = np.setdiff1d(available_all, available)
                    add = np.random.choice(others, min(need_more, len(others)), replace=False)
                    child[add] = 1
        elif selected_count > to_choose:
            # 移除多余的
            selected = np.where(child == 1)[0]
            remove = np.random.choice(selected, selected_count - to_choose, replace=False)
            child[remove] = 0
        
        return child
    
    def _mutate(self, individual: np.ndarray, to_choose: int) -> np.ndarray:
        """变异操作"""
        if np.random.rand() > self.mutation_rate:
            return individual
        
        # 随机交换一个选中的和一个未选中的
        selected = np.where(individual == 1)[0]
        unselected = np.where(individual == 0)[0]
        
        if len(selected) > 0 and len(unselected) > 0:
            remove = np.random.choice(selected)
            add = np.random.choice(unselected)
            individual[remove] = 0
            individual[add] = 1
        
        return individual
    
    def solve(
        self,
        locations: np.ndarray,
        to_choose: int,
        time_limit: float = 60.0
    ) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
        """
        使用遗传算法求解 FLP
        
        Args:
            locations: [n, 2] 位置坐标
            to_choose: 要选择的设施数量
            time_limit: 时间限制（秒）
            
        Returns:
            selected: 选中的设施索引
            obj_value: 目标值
            info: 额外信息
        """
        start_time = time.time()
        n = len(locations)
        
        # 计算距离矩阵
        dist_matrix = np.linalg.norm(
            locations[:, None, :] - locations[None, :, :],
            axis=2
        )
        
        # 初始化种群
        population = [self._create_individual(n, to_choose) for _ in range(self.population_size)]
        
        best_individual = None
        best_fitness = float('inf')
        history = []
        
        for gen in range(self.generations):
            # 检查时间限制
            if time.time() - start_time > time_limit:
                break
            
            # 评估适应度
            fitness = [self._evaluate(ind, locations, dist_matrix) for ind in population]
            
            # 记录最佳
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fitness:
                best_fitness = fitness[min_idx]
                best_individual = population[min_idx].copy()
            
            history.append(best_fitness)
            
            if self.verbose and (gen + 1) % 20 == 0:
                print(f"  Gen {gen+1}/{self.generations}: Best = {best_fitness:.4f}")
            
            # 选择（锦标赛选择）
            new_population = []
            
            # 精英保留
            elite_indices = np.argsort(fitness)[:self.elite_size]
            new_population.extend([population[i].copy() for i in elite_indices])
            
            # 生成新个体
            while len(new_population) < self.population_size:
                # 锦标赛选择
                tournament_size = 3
                tournament = np.random.choice(self.population_size, tournament_size)
                tournament_fitness = [fitness[i] for i in tournament]
                parent1 = population[tournament[np.argmin(tournament_fitness)]]
                
                tournament = np.random.choice(self.population_size, tournament_size)
                tournament_fitness = [fitness[i] for i in tournament]
                parent2 = population[tournament[np.argmin(tournament_fitness)]]
                
                # 交叉
                child = self._crossover(parent1, parent2, to_choose)
                
                # 变异
                child = self._mutate(child, to_choose)
                
                new_population.append(child)
            
            population = new_population
        
        solve_time = time.time() - start_time
        
        if best_individual is not None:
            selected = np.where(best_individual == 1)[0]
            
            info = {
                'solve_time': solve_time,
                'status': 'completed',
                'generations': len(history),
                'best_fitness_history': history,
            }
            
            return selected, best_fitness, info
        else:
            return None, None, {'solve_time': solve_time, 'status': 'failed'}


def solve_flp_ga(
    locations: np.ndarray,
    to_choose: int,
    time_limit: float = 60.0,
    verbose: bool = False,
    **kwargs
) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
    """
    使用遗传算法求解 FLP（函数接口）
    
    Args:
        locations: [n, 2] 位置坐标
        to_choose: 要选择的设施数量
        time_limit: 时间限制（秒）
        verbose: 是否打印详细信息
        **kwargs: 传递给 FLPGeneticSolver 的参数
        
    Returns:
        selected: 选中的设施索引
        obj_value: 目标值
        info: 额外信息
    """
    solver = FLPGeneticSolver(verbose=verbose, **kwargs)
    return solver.solve(locations, to_choose, time_limit)


# ============================================================================
# 统一求解接口
# ============================================================================

def solve_flp(
    locations: np.ndarray,
    to_choose: int,
    method: str = 'gurobi',
    time_limit: float = 60.0,
    verbose: bool = False,
    **kwargs
) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
    """
    统一的 FLP 求解接口
    
    Args:
        locations: [n, 2] 位置坐标
        to_choose: 要选择的设施数量
        method: 求解方法 ('gurobi', 'scip', 'ga')
        time_limit: 时间限制（秒）
        verbose: 是否打印详细信息
        **kwargs: 额外参数
        
    Returns:
        selected: 选中的设施索引
        obj_value: 目标值
        info: 额外信息
    """
    if method.lower() == 'gurobi':
        return solve_flp_gurobi(locations, to_choose, time_limit, verbose)
    elif method.lower() == 'scip':
        return solve_flp_scip(locations, to_choose, time_limit, verbose)
    elif method.lower() == 'ga':
        return solve_flp_ga(locations, to_choose, time_limit, verbose, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from ['gurobi', 'scip', 'ga']")
