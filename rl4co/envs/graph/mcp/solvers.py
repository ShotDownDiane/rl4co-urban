"""
MCP (Maximum Coverage Problem) Solvers
支持多种求解算法：Gurobi, SCIP, GA
"""
import numpy as np
import time
from typing import Tuple, Optional

# ============================================================================
# Gurobi Solver
# ============================================================================

def solve_mcp_gurobi(
    membership: np.ndarray,
    weights: np.ndarray,
    n_sets_to_choose: int,
    time_limit: float = 60.0,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
    """
    使用 Gurobi 求解 MCP
    
    Args:
        membership: [n_sets, max_size] 集合成员关系（-1表示padding）
        weights: [n_items] 物品权重
        n_sets_to_choose: 要选择的集合数量
        time_limit: 时间限制（秒）
        verbose: 是否打印详细信息
        
    Returns:
        selected: 选中的集合索引
        obj_value: 目标值（覆盖物品的总权重）
        info: 额外信息
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        raise ImportError("Gurobi is not available. Install: pip install gurobipy")
    
    start_time = time.time()
    n_sets = len(membership)
    n_items = len(weights)
    
    # 构建集合-物品关系
    set_items = []
    for s in range(n_sets):
        items = membership[s]
        items = items[items >= 0]  # 去除padding
        # 只保留有效的物品索引（防止越界）
        valid_items = [int(i) for i in items.tolist() if int(i) < n_items]
        set_items.append(set(valid_items))
    
    # 创建模型
    model = gp.Model("MCP")
    model.setParam('OutputFlag', 1 if verbose else 0)
    model.setParam('TimeLimit', time_limit)
    
    # 决策变量
    # x[s] = 1 if set s is selected
    x = model.addVars(n_sets, vtype=GRB.BINARY, name="x")
    
    # y[i] = 1 if item i is covered
    y = model.addVars(n_items, vtype=GRB.BINARY, name="y")
    
    # 目标函数: 最大化覆盖物品的总权重
    model.setObjective(
        gp.quicksum(weights[i] * y[i] for i in range(n_items)),
        GRB.MAXIMIZE
    )
    
    # 约束1: 恰好选择 n_sets_to_choose 个集合
    model.addConstr(
        gp.quicksum(x[s] for s in range(n_sets)) == n_sets_to_choose,
        "num_sets"
    )
    
    # 约束2: 如果物品被覆盖，至少有一个包含它的集合被选中
    for i in range(n_items):
        sets_containing_i = [s for s in range(n_sets) if i in set_items[s]]
        if sets_containing_i:
            model.addConstr(
                y[i] <= gp.quicksum(x[s] for s in sets_containing_i),
                f"cover_{i}"
            )
        else:
            # 物品不在任何集合中，不能被覆盖
            model.addConstr(y[i] == 0, f"no_cover_{i}")
    
    # 求解
    model.optimize()
    
    solve_time = time.time() - start_time
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        selected = np.array([s for s in range(n_sets) if x[s].X > 0.5])
        obj_value = model.objVal
        
        info = {
            'solve_time': solve_time,
            'status': 'optimal' if model.status == GRB.OPTIMAL else 'time_limit',
            'gap': model.MIPGap if hasattr(model, 'MIPGap') else 0.0,
            'covered_items': sum(1 for i in range(n_items) if y[i].X > 0.5),
        }
        
        return selected, obj_value, info
    else:
        return None, None, {'solve_time': solve_time, 'status': 'failed'}


# ============================================================================
# SCIP Solver
# ============================================================================

def solve_mcp_scip(
    membership: np.ndarray,
    weights: np.ndarray,
    n_sets_to_choose: int,
    time_limit: float = 60.0,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
    """
    使用 SCIP 求解 MCP
    
    Args:
        membership: [n_sets, max_size] 集合成员关系
        weights: [n_items] 物品权重
        n_sets_to_choose: 要选择的集合数量
        time_limit: 时间限制（秒）
        verbose: 是否打印详细信息
        
    Returns:
        selected: 选中的集合索引
        obj_value: 目标值
        info: 额外信息
    """
    try:
        from pyscipopt import Model, quicksum
    except ImportError:
        raise ImportError("SCIP is not available. Install: pip install pyscipopt")
    
    start_time = time.time()
    n_sets = len(membership)
    n_items = len(weights)
    
    # 构建集合-物品关系
    set_items = []
    for s in range(n_sets):
        items = membership[s]
        items = items[items >= 0]
        # 只保留有效的物品索引（防止越界）
        valid_items = [int(i) for i in items.tolist() if int(i) < n_items]
        set_items.append(set(valid_items))
    
    # 创建模型
    model = Model("MCP")
    if not verbose:
        model.hideOutput()
    model.setRealParam('limits/time', time_limit)
    
    # 决策变量
    x = {}
    y = {}
    for s in range(n_sets):
        x[s] = model.addVar(vtype="B", name=f"x_{s}")
    for i in range(n_items):
        y[i] = model.addVar(vtype="B", name=f"y_{i}")
    
    # 目标函数
    model.setObjective(
        quicksum(weights[i] * y[i] for i in range(n_items)),
        "maximize"
    )
    
    # 约束
    model.addCons(
        quicksum(x[s] for s in range(n_sets)) == n_sets_to_choose,
        "num_sets"
    )
    
    for i in range(n_items):
        sets_containing_i = [s for s in range(n_sets) if i in set_items[s]]
        if sets_containing_i:
            model.addCons(
                y[i] <= quicksum(x[s] for s in sets_containing_i),
                f"cover_{i}"
            )
        else:
            model.addCons(y[i] == 0, f"no_cover_{i}")
    
    # 求解
    model.optimize()
    
    solve_time = time.time() - start_time
    
    if model.getStatus() == 'optimal' or model.getStatus() == 'timelimit':
        selected = np.array([s for s in range(n_sets) if model.getVal(x[s]) > 0.5])
        obj_value = model.getObjVal()
        
        info = {
            'solve_time': solve_time,
            'status': model.getStatus(),
            'gap': model.getGap(),
            'covered_items': sum(1 for i in range(n_items) if model.getVal(y[i]) > 0.5),
        }
        
        return selected, obj_value, info
    else:
        return None, None, {'solve_time': solve_time, 'status': model.getStatus()}


# ============================================================================
# Genetic Algorithm (GA) Solver
# ============================================================================

class MCPGeneticSolver:
    """遗传算法求解 MCP"""
    
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
        membership: np.ndarray, 
        weights: np.ndarray
    ) -> float:
        """评估个体的适应度（覆盖物品的总权重）"""
        selected_sets = np.where(individual == 1)[0]
        if len(selected_sets) == 0:
            return 0.0
        
        # 收集所有被选中集合覆盖的物品
        covered_items = set()
        n_items = len(weights)
        for s in selected_sets:
            items = membership[s]
            items = items[items >= 0]  # 去除padding
            # 只添加有效的物品索引（防止越界）
            covered_items.update(int(i) for i in items.tolist() if int(i) < n_items)
        
        # 计算总权重
        total_weight = sum(weights[int(i)] for i in covered_items)
        return float(total_weight)
    
    def _create_individual(self, n_sets: int, n_to_choose: int) -> np.ndarray:
        """创建一个个体（随机选择 n_to_choose 个集合）"""
        individual = np.zeros(n_sets, dtype=int)
        n_to_choose = int(n_to_choose)  # 确保是整数
        selected = np.random.choice(n_sets, n_to_choose, replace=False)
        individual[selected] = 1
        return individual
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray, n_to_choose: int) -> np.ndarray:
        """交叉操作"""
        if np.random.rand() > self.crossover_rate:
            return parent1.copy()
        
        child = np.zeros_like(parent1)
        point = np.random.randint(1, len(parent1))
        
        child[:point] = parent1[:point]
        
        selected_count = child.sum()
        if selected_count < n_to_choose:
            available = np.where((parent2 == 1) & (child == 0))[0]
            need = n_to_choose - selected_count
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
        elif selected_count > n_to_choose:
            selected = np.where(child == 1)[0]
            remove = np.random.choice(selected, selected_count - n_to_choose, replace=False)
            child[remove] = 0
        
        return child
    
    def _mutate(self, individual: np.ndarray, n_to_choose: int) -> np.ndarray:
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
        membership: np.ndarray,
        weights: np.ndarray,
        n_sets_to_choose: int,
        time_limit: float = 60.0
    ) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
        """
        使用遗传算法求解 MCP
        
        Args:
            membership: [n_sets, max_size] 集合成员关系
            weights: [n_items] 物品权重
            n_sets_to_choose: 要选择的集合数量
            time_limit: 时间限制（秒）
            
        Returns:
            selected: 选中的集合索引
            obj_value: 目标值
            info: 额外信息
        """
        start_time = time.time()
        n_sets = len(membership)
        n_sets_to_choose = int(n_sets_to_choose)  # 确保是整数
        
        # 初始化种群
        population = [self._create_individual(n_sets, n_sets_to_choose) 
                     for _ in range(self.population_size)]
        
        best_individual = None
        best_fitness = 0.0
        history = []
        
        for gen in range(self.generations):
            if time.time() - start_time > time_limit:
                break
            
            # 评估适应度
            fitness = [self._evaluate(ind, membership, weights) for ind in population]
            
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
                # 锦标赛选择
                tournament_size = 3
                tournament = np.random.choice(self.population_size, tournament_size)
                tournament_fitness = [fitness[i] for i in tournament]
                parent1 = population[tournament[np.argmax(tournament_fitness)]]
                
                tournament = np.random.choice(self.population_size, tournament_size)
                tournament_fitness = [fitness[i] for i in tournament]
                parent2 = population[tournament[np.argmax(tournament_fitness)]]
                
                child = self._crossover(parent1, parent2, n_sets_to_choose)
                child = self._mutate(child, n_sets_to_choose)
                
                new_population.append(child)
            
            population = new_population
        
        solve_time = time.time() - start_time
        
        if best_individual is not None:
            selected = np.where(best_individual == 1)[0]
            
            # 计算覆盖的物品数
            covered_items = set()
            n_items = len(weights)
            for s in selected:
                items = membership[s]
                items = items[items >= 0]
                # 只添加有效的物品索引
                covered_items.update(int(i) for i in items.tolist() if int(i) < n_items)
            
            info = {
                'solve_time': solve_time,
                'status': 'completed',
                'generations': len(history),
                'best_fitness_history': history,
                'covered_items': len(covered_items),
            }
            
            return selected, best_fitness, info
        else:
            return None, None, {'solve_time': solve_time, 'status': 'failed'}


def solve_mcp_ga(
    membership: np.ndarray,
    weights: np.ndarray,
    n_sets_to_choose: int,
    time_limit: float = 60.0,
    verbose: bool = False,
    **kwargs
) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
    """
    使用遗传算法求解 MCP（函数接口）
    
    Args:
        membership: [n_sets, max_size] 集合成员关系
        weights: [n_items] 物品权重
        n_sets_to_choose: 要选择的集合数量
        time_limit: 时间限制（秒）
        verbose: 是否打印详细信息
        **kwargs: 传递给 MCPGeneticSolver 的参数
        
    Returns:
        selected: 选中的集合索引
        obj_value: 目标值
        info: 额外信息
    """
    solver = MCPGeneticSolver(verbose=verbose, **kwargs)
    return solver.solve(membership, weights, n_sets_to_choose, time_limit)


# ============================================================================
# 统一求解接口
# ============================================================================

def solve_mcp(
    membership: np.ndarray,
    weights: np.ndarray,
    n_sets_to_choose: int,
    method: str = 'gurobi',
    time_limit: float = 60.0,
    verbose: bool = False,
    **kwargs
) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
    """
    统一的 MCP 求解接口
    
    Args:
        membership: [n_sets, max_size] 集合成员关系
        weights: [n_items] 物品权重
        n_sets_to_choose: 要选择的集合数量
        method: 求解方法 ('gurobi', 'scip', 'ga')
        time_limit: 时间限制（秒）
        verbose: 是否打印详细信息
        **kwargs: 额外参数
        
    Returns:
        selected: 选中的集合索引
        obj_value: 目标值
        info: 额外信息
    """
    if method.lower() == 'gurobi':
        return solve_mcp_gurobi(membership, weights, n_sets_to_choose, time_limit, verbose)
    elif method.lower() == 'scip':
        return solve_mcp_scip(membership, weights, n_sets_to_choose, time_limit, verbose)
    elif method.lower() == 'ga':
        return solve_mcp_ga(membership, weights, n_sets_to_choose, time_limit, verbose, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from ['gurobi', 'scip', 'ga']")
