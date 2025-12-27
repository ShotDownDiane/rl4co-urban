"""
TSP (Traveling Salesman Problem) Solvers
支持多种求解算法：LKH, Concorde, Gurobi, OR-Tools, GA
"""
import numpy as np
import time
from typing import Tuple, Optional

# ============================================================================
# LKH Solver (使用 ML4CO-Kit)
# ============================================================================

def solve_tsp_lkh(
    locs: np.ndarray,
    time_limit: float = 60.0,
    verbose: bool = False,
    **lkh_params
) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
    """
    使用 LKH 求解 TSP
    
    LKH (Lin-Kernighan-Helsgaun) 是最强大的TSP启发式算法之一
    
    Args:
        locs: [n_nodes, 2] 节点坐标
        time_limit: 时间限制（秒）
        verbose: 是否打印详细信息
        **lkh_params: LKH参数
            - lkh_max_trials: 最大尝试次数 (默认: 500)
            - lkh_runs: 运行次数 (默认: 1)
            - lkh_seed: 随机种子 (默认: 1234)
        
    Returns:
        tour: 访问顺序（节点索引）
        obj_value: tour长度
        info: 额外信息
    """
    try:
        from ml4co_kit.solver.lkh import LKHSolver
        from ml4co_kit.task.routing.tsp import TSPTask
    except ImportError:
        raise ImportError("ML4CO-Kit is not available. Install: pip install ml4co-kit")
    
    start_time = time.time()
    n_nodes = len(locs)
    
    # 创建 TSP Task
    task = TSPTask()
    task.from_data(points=locs.astype(np.float32))
    
    # 创建 LKH Solver
    solver_params = {
        'lkh_max_trials': lkh_params.get('lkh_max_trials', 500),
        'lkh_runs': lkh_params.get('lkh_runs', 1),
        'lkh_seed': lkh_params.get('lkh_seed', 1234),
    }
    solver = LKHSolver(**solver_params)
    
    # 求解
    try:
        result_task = solver.solve(task)
        tour = result_task.sol
        
        # 计算目标值
        if tour is not None:
            obj_value = result_task.evaluate(tour)
        else:
            obj_value = None
        
        solve_time = time.time() - start_time
        
        if tour is not None and obj_value is not None:
            info = {
                'solve_time': solve_time,
                'status': 'success',
                'solver': 'LKH',
                'params': solver_params,
            }
            return tour, float(obj_value), info
        else:
            return None, None, {'solve_time': solve_time, 'status': 'failed', 'error': 'No solution found'}
            
    except Exception as e:
        solve_time = time.time() - start_time
        if verbose:
            print(f"LKH solver failed: {e}")
        return None, None, {'solve_time': solve_time, 'status': 'failed', 'error': str(e)}


# ============================================================================
# Concorde Solver (使用 ML4CO-Kit)
# ============================================================================

def solve_tsp_concorde(
    locs: np.ndarray,
    time_limit: float = 60.0,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
    """
    使用 Concorde 求解 TSP
    
    Concorde 是最好的TSP精确求解器，可以求解数万节点的问题
    
    Args:
        locs: [n_nodes, 2] 节点坐标
        time_limit: 时间限制（秒）
        verbose: 是否打印详细信息
        
    Returns:
        tour: 访问顺序
        obj_value: tour长度
        info: 额外信息
    """
    try:
        from ml4co_kit.solver.concorde import ConcordeSolver
        from ml4co_kit.task.routing.tsp import TSPTask
    except ImportError:
        raise ImportError("ML4CO-Kit is not available. Install: pip install ml4co-kit")
    
    start_time = time.time()
    
    # 创建 TSP Task
    task = TSPTask()
    task.from_data(points=locs.astype(np.float32))
    
    # 创建 Concorde Solver
    solver = ConcordeSolver()
    
    # 求解
    try:
        result_task = solver.solve(task)
        tour = result_task.sol
        
        # 计算目标值
        if tour is not None:
            obj_value = result_task.evaluate(tour)
        else:
            obj_value = None
        
        solve_time = time.time() - start_time
        
        if tour is not None and obj_value is not None:
            info = {
                'solve_time': solve_time,
                'status': 'optimal',
                'solver': 'Concorde',
            }
            return tour, float(obj_value), info
        else:
            return None, None, {'solve_time': solve_time, 'status': 'failed', 'error': 'No solution found'}
            
    except Exception as e:
        solve_time = time.time() - start_time
        if verbose:
            print(f"Concorde solver failed: {e}")
        return None, None, {'solve_time': solve_time, 'status': 'failed', 'error': str(e)}


# ============================================================================
# Gurobi Solver
# ============================================================================

def solve_tsp_gurobi(
    locs: np.ndarray,
    time_limit: float = 60.0,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
    """
    使用 Gurobi 求解 TSP (Miller-Tucker-Zemlin formulation)
    
    Args:
        locs: [n_nodes, 2] 节点坐标
        time_limit: 时间限制（秒）
        verbose: 是否打印详细信息
        
    Returns:
        tour: 访问顺序
        obj_value: tour长度
        info: 额外信息
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        raise ImportError("Gurobi is not available. Install: pip install gurobipy")
    
    start_time = time.time()
    n = len(locs)
    
    # 计算距离矩阵
    dist = np.linalg.norm(locs[:, None, :] - locs[None, :, :], axis=2)
    
    # 创建模型
    model = gp.Model("TSP")
    model.setParam('OutputFlag', 1 if verbose else 0)
    model.setParam('TimeLimit', time_limit)
    
    # 决策变量
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    u = model.addVars(n, vtype=GRB.CONTINUOUS, name="u")  # MTZ subtour elimination
    
    # 目标函数
    model.setObjective(
        gp.quicksum(dist[i, j] * x[i, j] for i in range(n) for j in range(n) if i != j),
        GRB.MINIMIZE
    )
    
    # 约束1: 每个城市恰好离开一次
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) == 1)
    
    # 约束2: 每个城市恰好进入一次
    for j in range(n):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n) if i != j) == 1)
    
    # 约束3: MTZ subtour elimination
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1)
    
    # 求解
    model.optimize()
    
    solve_time = time.time() - start_time
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        # 提取 tour
        tour = [0]
        current = 0
        visited = {0}
        
        while len(tour) < n:
            for j in range(n):
                if j not in visited and x[current, j].X > 0.5:
                    tour.append(j)
                    visited.add(j)
                    current = j
                    break
        
        tour = np.array(tour)
        obj_value = model.objVal
        
        info = {
            'solve_time': solve_time,
            'status': 'optimal' if model.status == GRB.OPTIMAL else 'time_limit',
            'gap': model.MIPGap if hasattr(model, 'MIPGap') else 0.0,
            'solver': 'Gurobi',
        }
        
        return tour, obj_value, info
    else:
        return None, None, {'solve_time': solve_time, 'status': 'failed'}


# ============================================================================
# OR-Tools Solver
# ============================================================================

def solve_tsp_ortools(
    locs: np.ndarray,
    time_limit: float = 60.0,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
    """
    使用 Google OR-Tools 求解 TSP
    
    Args:
        locs: [n_nodes, 2] 节点坐标
        time_limit: 时间限制（秒）
        verbose: 是否打印详细信息
        
    Returns:
        tour: 访问顺序
        obj_value: tour长度
        info: 额外信息
    """
    try:
        from ortools.constraint_solver import routing_enums_pb2
        from ortools.constraint_solver import pywrapcp
    except ImportError:
        raise ImportError("OR-Tools is not available. Install: pip install ortools")
    
    start_time = time.time()
    n = len(locs)
    
    # 计算距离矩阵（整数化）
    dist = np.linalg.norm(locs[:, None, :] - locs[None, :, :], axis=2)
    dist_int = (dist * 1e6).astype(int)  # 放大并转换为整数
    
    # 创建距离回调函数
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_int[from_node, to_node]
    
    # 创建 routing model
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # 1个车辆，从节点0开始
    routing = pywrapcp.RoutingModel(manager)
    
    # 注册距离回调
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # 设置搜索参数
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = int(time_limit)
    search_parameters.log_search = verbose
    
    # 求解
    solution = routing.SolveWithParameters(search_parameters)
    
    solve_time = time.time() - start_time
    
    if solution:
        # 提取 tour
        tour = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            tour.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        
        tour = np.array(tour)
        obj_value = solution.ObjectiveValue() / 1e6  # 还原缩放
        
        info = {
            'solve_time': solve_time,
            'status': 'success',
            'solver': 'OR-Tools',
        }
        
        return tour, float(obj_value), info
    else:
        return None, None, {'solve_time': solve_time, 'status': 'failed'}


# ============================================================================
# Nearest Neighbor Heuristic
# ============================================================================

def solve_tsp_nearest_neighbor(
    locs: np.ndarray,
    start: int = 0
) -> Tuple[np.ndarray, float, dict]:
    """
    最近邻启发式算法求解 TSP
    
    Args:
        locs: [n_nodes, 2] 节点坐标
        start: 起始节点
        
    Returns:
        tour: 访问顺序
        obj_value: tour长度
        info: 额外信息
    """
    start_time = time.time()
    n = len(locs)
    
    # 计算距离矩阵
    dist = np.linalg.norm(locs[:, None, :] - locs[None, :, :], axis=2)
    
    # 最近邻算法
    tour = [start]
    unvisited = set(range(n)) - {start}
    current = start
    
    while unvisited:
        # 找到最近的未访问节点
        nearest = min(unvisited, key=lambda x: dist[current, x])
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    tour = np.array(tour)
    
    # 计算总距离
    obj_value = sum(dist[tour[i], tour[i+1]] for i in range(n-1))
    obj_value += dist[tour[-1], tour[0]]  # 回到起点
    
    solve_time = time.time() - start_time
    
    info = {
        'solve_time': solve_time,
        'status': 'success',
        'solver': 'Nearest Neighbor',
        'algorithm': 'greedy heuristic',
    }
    
    return tour, float(obj_value), info


# ============================================================================
# Genetic Algorithm Solver
# ============================================================================

class TSPGeneticSolver:
    """遗传算法求解 TSP"""
    
    def __init__(
        self,
        population_size: int = 100,
        generations: int = 500,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        elite_size: int = 10,
        verbose: bool = False
    ):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.verbose = verbose
    
    def _evaluate(self, tour: np.ndarray, dist: np.ndarray) -> float:
        """计算tour的长度"""
        n = len(tour)
        length = sum(dist[tour[i], tour[i+1]] for i in range(n-1))
        length += dist[tour[-1], tour[0]]  # 回到起点
        return length
    
    def _create_individual(self, n: int) -> np.ndarray:
        """创建一个随机tour"""
        tour = np.arange(n)
        np.random.shuffle(tour)
        return tour
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Order Crossover (OX)"""
        if np.random.rand() > self.crossover_rate:
            return parent1.copy()
        
        n = len(parent1)
        start, end = sorted(np.random.choice(n, 2, replace=False))
        
        # 从parent1继承中间部分
        child = np.full(n, -1)
        child[start:end] = parent1[start:end]
        
        # 从parent2按顺序填充剩余部分
        p2_filtered = [x for x in parent2 if x not in child]
        idx = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = p2_filtered[idx]
                idx += 1
        
        return child
    
    def _mutate(self, tour: np.ndarray) -> np.ndarray:
        """2-opt mutation"""
        if np.random.rand() > self.mutation_rate:
            return tour
        
        n = len(tour)
        i, j = sorted(np.random.choice(n, 2, replace=False))
        
        # 反转i到j之间的部分
        tour = tour.copy()
        tour[i:j+1] = tour[i:j+1][::-1]
        
        return tour
    
    def solve(
        self,
        locs: np.ndarray,
        time_limit: float = 60.0
    ) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
        """
        使用遗传算法求解 TSP
        """
        start_time = time.time()
        n = len(locs)
        
        # 计算距离矩阵
        dist = np.linalg.norm(locs[:, None, :] - locs[None, :, :], axis=2)
        
        # 初始化种群
        population = [self._create_individual(n) for _ in range(self.population_size)]
        
        best_tour = None
        best_length = float('inf')
        history = []
        
        for gen in range(self.generations):
            if time.time() - start_time > time_limit:
                break
            
            # 评估
            fitness = [self._evaluate(tour, dist) for tour in population]
            
            # 记录最佳
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_length:
                best_length = fitness[min_idx]
                best_tour = population[min_idx].copy()
            
            history.append(best_length)
            
            if self.verbose and (gen + 1) % 50 == 0:
                print(f"  Gen {gen+1}/{self.generations}: Best = {best_length:.4f}")
            
            # 选择
            new_population = []
            
            # 精英保留
            elite_indices = np.argsort(fitness)[:self.elite_size]
            new_population.extend([population[i].copy() for i in elite_indices])
            
            # 生成新个体
            while len(new_population) < self.population_size:
                # 锦标赛选择
                tournament_size = 5
                tournament = np.random.choice(self.population_size, tournament_size)
                tournament_fitness = [fitness[i] for i in tournament]
                parent1 = population[tournament[np.argmin(tournament_fitness)]]
                
                tournament = np.random.choice(self.population_size, tournament_size)
                tournament_fitness = [fitness[i] for i in tournament]
                parent2 = population[tournament[np.argmin(tournament_fitness)]]
                
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population
        
        solve_time = time.time() - start_time
        
        if best_tour is not None:
            info = {
                'solve_time': solve_time,
                'status': 'completed',
                'generations': len(history),
                'best_fitness_history': history,
                'solver': 'Genetic Algorithm',
            }
            
            return best_tour, best_length, info
        else:
            return None, None, {'solve_time': solve_time, 'status': 'failed'}


def solve_tsp_ga(
    locs: np.ndarray,
    time_limit: float = 60.0,
    verbose: bool = False,
    **kwargs
) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
    """
    使用遗传算法求解 TSP（函数接口）
    """
    solver = TSPGeneticSolver(verbose=verbose, **kwargs)
    return solver.solve(locs, time_limit)


# ============================================================================
# 统一求解接口
# ============================================================================

def solve_tsp(
    locs: np.ndarray,
    method: str = 'lkh',
    time_limit: float = 60.0,
    verbose: bool = False,
    **kwargs
) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
    """
    统一的 TSP 求解接口
    
    Args:
        locs: [n_nodes, 2] 节点坐标
        method: 求解方法
            - 'lkh': LKH solver (推荐，最强启发式)
            - 'concorde': Concorde (精确求解)
            - 'gurobi': Gurobi MIP
            - 'ortools': Google OR-Tools
            - 'ga': Genetic Algorithm
            - 'greedy': Nearest Neighbor
        time_limit: 时间限制（秒）
        verbose: 是否打印详细信息
        **kwargs: 额外参数
        
    Returns:
        tour: 访问顺序
        obj_value: tour长度
        info: 额外信息
    """
    method = method.lower()
    
    if method == 'lkh':
        return solve_tsp_lkh(locs, time_limit, verbose, **kwargs)
    elif method == 'concorde':
        return solve_tsp_concorde(locs, time_limit, verbose)
    elif method == 'gurobi':
        return solve_tsp_gurobi(locs, time_limit, verbose)
    elif method == 'ortools':
        return solve_tsp_ortools(locs, time_limit, verbose)
    elif method == 'ga':
        return solve_tsp_ga(locs, time_limit, verbose, **kwargs)
    elif method == 'greedy' or method == 'nn':
        return solve_tsp_nearest_neighbor(locs)
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Choose from ['lkh', 'concorde', 'gurobi', 'ortools', 'ga', 'greedy']"
        )
