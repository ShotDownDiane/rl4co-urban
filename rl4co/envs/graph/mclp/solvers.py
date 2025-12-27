"""
MCLP (Maximum Covering Location Problem) Solvers
支持多种求解算法：Gurobi, SCIP, GA
"""
import numpy as np
import time
from typing import Tuple, Optional

# ============================================================================
# Gurobi Solver
# ============================================================================

def solve_mclp_gurobi(
    demand_locs: np.ndarray,
    demand_weights: np.ndarray,
    facility_locs: np.ndarray,
    coverage_radius: float,
    num_facilities_to_select: int,
    time_limit: float = 60.0,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
    """
    使用 Gurobi 求解 MCLP
    
    Args:
        demand_locs: [n_demand, 2] 需求点位置
        demand_weights: [n_demand] 需求点权重
        facility_locs: [n_facility, 2] 候选设施位置
        coverage_radius: 覆盖半径
        num_facilities_to_select: 要选择的设施数量
        time_limit: 时间限制（秒）
        verbose: 是否打印详细信息
        
    Returns:
        selected: 选中的设施索引
        obj_value: 目标值（覆盖的需求总量）
        info: 额外信息
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        raise ImportError("Gurobi is not available. Install: pip install gurobipy")
    
    start_time = time.time()
    n_demand = len(demand_locs)
    n_facility = len(facility_locs)
    
    # 计算距离矩阵并确定覆盖关系
    dist_matrix = np.linalg.norm(
        demand_locs[:, None, :] - facility_locs[None, :, :],
        axis=2
    )
    coverage_matrix = (dist_matrix <= coverage_radius).astype(int)
    
    # 创建模型
    model = gp.Model("MCLP")
    model.setParam('OutputFlag', 1 if verbose else 0)
    model.setParam('TimeLimit', time_limit)
    
    # 决策变量
    # x[j] = 1 if facility j is selected
    x = model.addVars(n_facility, vtype=GRB.BINARY, name="x")
    
    # y[i] = 1 if demand point i is covered
    y = model.addVars(n_demand, vtype=GRB.BINARY, name="y")
    
    # 目标函数: 最大化覆盖的需求总量
    model.setObjective(
        gp.quicksum(demand_weights[i] * y[i] for i in range(n_demand)),
        GRB.MAXIMIZE
    )
    
    # 约束1: 恰好选择 num_facilities_to_select 个设施
    model.addConstr(
        gp.quicksum(x[j] for j in range(n_facility)) == num_facilities_to_select,
        "num_facilities"
    )
    
    # 约束2: 需求点只有在被至少一个已选设施覆盖时才被计入
    for i in range(n_demand):
        covering_facilities = [j for j in range(n_facility) if coverage_matrix[i, j] == 1]
        if covering_facilities:
            model.addConstr(
                y[i] <= gp.quicksum(x[j] for j in covering_facilities),
                f"cover_{i}"
            )
        else:
            # 需求点不在任何设施的覆盖范围内
            model.addConstr(y[i] == 0, f"no_cover_{i}")
    
    # 求解
    model.optimize()
    
    solve_time = time.time() - start_time
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        selected = np.array([j for j in range(n_facility) if x[j].X > 0.5])
        obj_value = model.objVal
        
        info = {
            'solve_time': solve_time,
            'status': 'optimal' if model.status == GRB.OPTIMAL else 'time_limit',
            'gap': model.MIPGap if hasattr(model, 'MIPGap') else 0.0,
            'covered_demands': sum(1 for i in range(n_demand) if y[i].X > 0.5),
            'coverage_rate': sum(1 for i in range(n_demand) if y[i].X > 0.5) / n_demand,
        }
        
        return selected, obj_value, info
    else:
        return None, None, {'solve_time': solve_time, 'status': 'failed'}


# ============================================================================
# SCIP Solver
# ============================================================================

def solve_mclp_scip(
    demand_locs: np.ndarray,
    demand_weights: np.ndarray,
    facility_locs: np.ndarray,
    coverage_radius: float,
    num_facilities_to_select: int,
    time_limit: float = 60.0,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
    """
    使用 SCIP 求解 MCLP
    
    Args:
        demand_locs: [n_demand, 2] 需求点位置
        demand_weights: [n_demand] 需求点权重
        facility_locs: [n_facility, 2] 候选设施位置
        coverage_radius: 覆盖半径
        num_facilities_to_select: 要选择的设施数量
        time_limit: 时间限制（秒）
        verbose: 是否打印详细信息
        
    Returns:
        selected: 选中的设施索引
        obj_value: 目标值
        info: 额外信息
    """
    try:
        from pyscipopt import Model, quicksum
    except ImportError:
        raise ImportError("SCIP is not available. Install: pip install pyscipopt")
    
    start_time = time.time()
    n_demand = len(demand_locs)
    n_facility = len(facility_locs)
    
    # 计算覆盖关系
    dist_matrix = np.linalg.norm(
        demand_locs[:, None, :] - facility_locs[None, :, :],
        axis=2
    )
    coverage_matrix = (dist_matrix <= coverage_radius).astype(int)
    
    # 创建模型
    model = Model("MCLP")
    if not verbose:
        model.hideOutput()
    model.setRealParam('limits/time', time_limit)
    
    # 决策变量
    x = {}
    y = {}
    for j in range(n_facility):
        x[j] = model.addVar(vtype="B", name=f"x_{j}")
    for i in range(n_demand):
        y[i] = model.addVar(vtype="B", name=f"y_{i}")
    
    # 目标函数
    model.setObjective(
        quicksum(demand_weights[i] * y[i] for i in range(n_demand)),
        "maximize"
    )
    
    # 约束
    model.addCons(
        quicksum(x[j] for j in range(n_facility)) == num_facilities_to_select,
        "num_facilities"
    )
    
    for i in range(n_demand):
        covering_facilities = [j for j in range(n_facility) if coverage_matrix[i, j] == 1]
        if covering_facilities:
            model.addCons(
                y[i] <= quicksum(x[j] for j in covering_facilities),
                f"cover_{i}"
            )
        else:
            model.addCons(y[i] == 0, f"no_cover_{i}")
    
    # 求解
    model.optimize()
    
    solve_time = time.time() - start_time
    
    if model.getStatus() == 'optimal' or model.getStatus() == 'timelimit':
        selected = np.array([j for j in range(n_facility) if model.getVal(x[j]) > 0.5])
        obj_value = model.getObjVal()
        
        info = {
            'solve_time': solve_time,
            'status': model.getStatus(),
            'gap': model.getGap(),
            'covered_demands': sum(1 for i in range(n_demand) if model.getVal(y[i]) > 0.5),
            'coverage_rate': sum(1 for i in range(n_demand) if model.getVal(y[i]) > 0.5) / n_demand,
        }
        
        return selected, obj_value, info
    else:
        return None, None, {'solve_time': solve_time, 'status': model.getStatus()}


# ============================================================================
# Genetic Algorithm (GA) Solver
# ============================================================================

class MCLPGeneticSolver:
    """遗传算法求解 MCLP"""
    
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
    
    def _evaluate(
        self,
        individual: np.ndarray,
        demand_weights: np.ndarray,
        coverage_matrix: np.ndarray
    ) -> float:
        """评估个体的适应度（覆盖的需求总量）"""
        selected_facilities = np.where(individual == 1)[0]
        if len(selected_facilities) == 0:
            return 0.0
        
        # 计算被覆盖的需求点
        covered = coverage_matrix[:, selected_facilities].any(axis=1)
        total_weight = demand_weights[covered].sum()
        
        return float(total_weight)
    
    def _create_individual(self, n_facility: int, n_to_select: int) -> np.ndarray:
        """创建一个个体（随机选择设施）"""
        individual = np.zeros(n_facility, dtype=int)
        selected = np.random.choice(n_facility, n_to_select, replace=False)
        individual[selected] = 1
        return individual
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray, n_to_select: int) -> np.ndarray:
        """交叉操作"""
        if np.random.rand() > self.crossover_rate:
            return parent1.copy()
        
        child = np.zeros_like(parent1)
        point = np.random.randint(1, len(parent1))
        
        child[:point] = parent1[:point]
        
        selected_count = child.sum()
        if selected_count < n_to_select:
            available = np.where((parent2 == 1) & (child == 0))[0]
            need = n_to_select - selected_count
            if len(available) >= need:
                add = np.random.choice(available, need, replace=False)
                child[add] = 1
            else:
                child[available] = 1
                available_all = np.where(child == 0)[0]
                if len(available_all) > 0:
                    need_more = need - len(available)
                    add = np.random.choice(available_all, min(need_more, len(available_all)), replace=False)
                    child[add] = 1
        elif selected_count > n_to_select:
            selected = np.where(child == 1)[0]
            remove = np.random.choice(selected, selected_count - n_to_select, replace=False)
            child[remove] = 0
        
        return child
    
    def _mutate(self, individual: np.ndarray, n_to_select: int) -> np.ndarray:
        """变异操作"""
        if np.random.rand() > self.mutation_rate:
            return individual
        
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
        demand_locs: np.ndarray,
        demand_weights: np.ndarray,
        facility_locs: np.ndarray,
        coverage_radius: float,
        num_facilities_to_select: int,
        time_limit: float = 60.0
    ) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
        """
        使用遗传算法求解 MCLP
        
        Args:
            demand_locs: [n_demand, 2] 需求点位置
            demand_weights: [n_demand] 需求点权重
            facility_locs: [n_facility, 2] 候选设施位置
            coverage_radius: 覆盖半径
            num_facilities_to_select: 要选择的设施数量
            time_limit: 时间限制（秒）
            
        Returns:
            selected: 选中的设施索引
            obj_value: 目标值
            info: 额外信息
        """
        start_time = time.time()
        n_facility = len(facility_locs)
        
        # 计算覆盖关系
        dist_matrix = np.linalg.norm(
            demand_locs[:, None, :] - facility_locs[None, :, :],
            axis=2
        )
        coverage_matrix = (dist_matrix <= coverage_radius).astype(int)
        
        # 初始化种群
        population = [self._create_individual(n_facility, num_facilities_to_select)
                     for _ in range(self.population_size)]
        
        best_individual = None
        best_fitness = 0.0
        history = []
        
        for gen in range(self.generations):
            if time.time() - start_time > time_limit:
                break
            
            # 评估适应度
            fitness = [self._evaluate(ind, demand_weights, coverage_matrix) 
                      for ind in population]
            
            # 记录最佳
            max_idx = np.argmax(fitness)
            if fitness[max_idx] > best_fitness:
                best_fitness = fitness[max_idx]
                best_individual = population[max_idx].copy()
            
            history.append(best_fitness)
            
            if self.verbose and (gen + 1) % 20 == 0:
                print(f"  Gen {gen+1}/{self.generations}: Best = {best_fitness:.4f}")
            
            # 选择
            new_population = []
            
            # 精英保留
            elite_indices = np.argsort(fitness)[-self.elite_size:]
            new_population.extend([population[i].copy() for i in elite_indices])
            
            # 生成新个体
            while len(new_population) < self.population_size:
                tournament_size = 3
                tournament = np.random.choice(self.population_size, tournament_size)
                tournament_fitness = [fitness[i] for i in tournament]
                parent1 = population[tournament[np.argmax(tournament_fitness)]]
                
                tournament = np.random.choice(self.population_size, tournament_size)
                tournament_fitness = [fitness[i] for i in tournament]
                parent2 = population[tournament[np.argmax(tournament_fitness)]]
                
                child = self._crossover(parent1, parent2, num_facilities_to_select)
                child = self._mutate(child, num_facilities_to_select)
                
                new_population.append(child)
            
            population = new_population
        
        solve_time = time.time() - start_time
        
        if best_individual is not None:
            selected = np.where(best_individual == 1)[0]
            
            # 计算覆盖信息
            covered = coverage_matrix[:, selected].any(axis=1)
            n_demand = len(demand_locs)
            
            info = {
                'solve_time': solve_time,
                'status': 'completed',
                'generations': len(history),
                'best_fitness_history': history,
                'covered_demands': int(covered.sum()),
                'coverage_rate': float(covered.sum() / n_demand),
            }
            
            return selected, best_fitness, info
        else:
            return None, None, {'solve_time': solve_time, 'status': 'failed'}


def solve_mclp_ga(
    demand_locs: np.ndarray,
    demand_weights: np.ndarray,
    facility_locs: np.ndarray,
    coverage_radius: float,
    num_facilities_to_select: int,
    time_limit: float = 60.0,
    verbose: bool = False,
    **kwargs
) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
    """
    使用遗传算法求解 MCLP（函数接口）
    
    Args:
        demand_locs: [n_demand, 2] 需求点位置
        demand_weights: [n_demand] 需求点权重
        facility_locs: [n_facility, 2] 候选设施位置
        coverage_radius: 覆盖半径
        num_facilities_to_select: 要选择的设施数量
        time_limit: 时间限制（秒）
        verbose: 是否打印详细信息
        **kwargs: 传递给 MCLPGeneticSolver 的参数
        
    Returns:
        selected: 选中的设施索引
        obj_value: 目标值
        info: 额外信息
    """
    solver = MCLPGeneticSolver(verbose=verbose, **kwargs)
    return solver.solve(demand_locs, demand_weights, facility_locs, 
                        coverage_radius, num_facilities_to_select, time_limit)


# ============================================================================
# 统一求解接口
# ============================================================================

def solve_mclp(
    demand_locs: np.ndarray,
    demand_weights: np.ndarray,
    facility_locs: np.ndarray,
    coverage_radius: float,
    num_facilities_to_select: int,
    method: str = 'gurobi',
    time_limit: float = 60.0,
    verbose: bool = False,
    **kwargs
) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
    """
    统一的 MCLP 求解接口
    
    Args:
        demand_locs: [n_demand, 2] 需求点位置
        demand_weights: [n_demand] 需求点权重
        facility_locs: [n_facility, 2] 候选设施位置
        coverage_radius: 覆盖半径
        num_facilities_to_select: 要选择的设施数量
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
        return solve_mclp_gurobi(demand_locs, demand_weights, facility_locs,
                                  coverage_radius, num_facilities_to_select,
                                  time_limit, verbose)
    elif method.lower() == 'scip':
        return solve_mclp_scip(demand_locs, demand_weights, facility_locs,
                                coverage_radius, num_facilities_to_select,
                                time_limit, verbose)
    elif method.lower() == 'ga':
        return solve_mclp_ga(demand_locs, demand_weights, facility_locs,
                             coverage_radius, num_facilities_to_select,
                             time_limit, verbose, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from ['gurobi', 'scip', 'ga']")
