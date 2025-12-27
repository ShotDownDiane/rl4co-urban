"""
Generic Wrapper for ML4CO-Kit Graph Tasks as RL4CO Environments

This module provides wrapper classes that integrate ML4CO-Kit graph problem instances
into RL4CO's training framework without reimplementing the problem logic.

Supported Problems:
- MIS (Maximum Independent Set)
- MVC (Minimum Vertex Cover)
- MCL (Maximum Clique)
- MCUT (Maximum Cut)
"""

import torch
import numpy as np
from typing import Optional, Type, Dict, Any, List, Tuple, Callable
from tensordict import TensorDict

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index


class ML4COGraphWrapper(RL4COEnvBase):
    """
    ML4CO-Kit Graph Problems 的真正 Wrapper 基类
    
    设计原则：
    1. ✅ 重用 ML4CO-Kit 的 Generator（不重新实现图生成）
    2. ✅ 重用 ML4CO-Kit 的 evaluate/check_constraints（不重新实现评估）
    3. ✅ 重用 ML4CO-Kit 的 Solver（baseline 对比）
    4. ✅ 重用 ML4CO-Kit 的 render（可视化）
    5. ✅ 只做格式转换：TensorDict ↔ ML4CO Task
    6. ✅ 只实现 RL 特有逻辑（reset/step/reward）
    """
    
    def __init__(
        self,
        ml4co_generator_class,    # ML4CO-Kit Generator 类
        ml4co_task_class,         # ML4CO-Kit Task 类
        ml4co_solver_class=None,  # ML4CO-Kit Solver 类（可选）
        generator_kwargs: dict = None,  # Generator 的参数
        solver_kwargs: dict = None,     # Solver 的参数
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # 1. 创建 ML4CO-Kit Generator（重用！）
        generator_kwargs = generator_kwargs or {}
        self.ml4co_generator = ml4co_generator_class(**generator_kwargs)
        
        # 2. 保存 Task 类（用于创建实例）
        self.ml4co_task_class = ml4co_task_class
        
        # 3. 创建 ML4CO-Kit Solver（重用！）
        if ml4co_solver_class is not None:
            solver_kwargs = solver_kwargs or {}
            try:
                self.ml4co_solver = ml4co_solver_class(**solver_kwargs)
            except Exception as e:
                print(f"Warning: Failed to initialize solver: {e}")
                self.ml4co_solver = None
        else:
            self.ml4co_solver = None
        
        # 4. 从 Generator 中提取环境参数
        self.num_nodes = getattr(self.ml4co_generator, 'nodes_num', 50)
        self.node_weighted = getattr(self.ml4co_generator, 'node_weighted', True)
        self.edge_weighted = getattr(self.ml4co_generator, 'edge_weighted', True)
        
        # 5. RL4CO 兼容性（不使用 generator）
        self.generator = None
    
    def generate_data(self, batch_size) -> TensorDict:
        """
        生成数据 - 使用 ML4CO-Kit Generator（重用！）
        """
        # 处理各种 batch_size 格式
        if isinstance(batch_size, (tuple, list)):
            batch_size = batch_size[0]
        if isinstance(batch_size, torch.Tensor):
            batch_size = batch_size.item()
        batch_size = int(batch_size)
        
        # 🔑 关键改动：使用 ML4CO-Kit Generator 生成 Tasks（重用！）
        tasks = [self.ml4co_generator.generate() for _ in range(batch_size)]
        
        # 格式转换：ML4CO Task → TensorDict
        return self._tasks_to_tensordict(tasks)
    
    def _tasks_to_tensordict(self, tasks: List) -> TensorDict:
        """
        格式转换：ML4CO Task → TensorDict
        这是 Wrapper 的核心工作：格式适配
        """
        batch_size = len(tasks)
        
        edge_indices = []
        nodes_weights = []
        edge_nums = []
        
        for task in tasks:
            # 从 ML4CO-Kit Task 中提取数据
            edge_index = task.edge_index  # [2, num_edges]
            edge_indices.append(torch.from_numpy(edge_index).long())
            edge_nums.append(edge_index.shape[1])
            
            # 节点权重
            if self.node_weighted and task.nodes_weight is not None:
                nodes_weights.append(torch.from_numpy(task.nodes_weight).float())
            else:
                # 默认权重为 1
                nodes_weights.append(torch.ones(task.nodes_num, dtype=torch.float32))
        
        # Pad edge_index to the same length (max_edges)
        max_edges = max(edge_nums)
        padded_edge_indices = []
        for edge_idx, num_edges in zip(edge_indices, edge_nums):
            if num_edges < max_edges:
                # Pad with -1 (invalid edge marker)
                padding = torch.full((2, max_edges - num_edges), -1, dtype=torch.long)
                edge_idx = torch.cat([edge_idx, padding], dim=1)
            padded_edge_indices.append(edge_idx)
        
        td = TensorDict({
            "edge_index": torch.stack(padded_edge_indices).to(self.device),
            "nodes_weight": torch.stack(nodes_weights).to(self.device),
            "nodes_num": torch.full((batch_size,), self.num_nodes, 
                                   dtype=torch.long, device=self.device),
            "edge_nums": torch.tensor(edge_nums, dtype=torch.long, device=self.device)
        }, batch_size=[batch_size])
        
        return td
    
    def reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        """重置环境（重写以避免调用 generator）"""
        if batch_size is None:
            batch_size = self.batch_size if td is None else td.batch_size
        
        # 标准化 batch_size 格式
        if isinstance(batch_size, (tuple, list)):
            batch_size_int = batch_size[0] if isinstance(batch_size[0], int) else batch_size[0].item()
        elif isinstance(batch_size, torch.Tensor):
            batch_size_int = batch_size.item()
        else:
            batch_size_int = int(batch_size)
        
        if td is None or "edge_index" not in td:
            td = self.generate_data(batch_size=batch_size)
        
        # 初始化状态（传入标准化的整数 batch_size）
        td = self._init_state(td, batch_size_int)
        
        return td
    
    def _reset(self, td: TensorDict = None, batch_size=None) -> TensorDict:
        """内部 reset（兼容父类）"""
        return self.reset(td, batch_size)
    
    def _init_state(self, td: TensorDict, batch_size) -> TensorDict:
        """初始化 RL 状态（由子类实现具体逻辑）"""
        raise NotImplementedError
    
    def _step(self, td: TensorDict) -> TensorDict:
        """执行一步动作（由子类实现）"""
        raise NotImplementedError
    
    def _get_reward(self, td: TensorDict, actions=None) -> torch.Tensor:
        """
        计算奖励 - 使用 ML4CO-Kit 的 evaluate 方法（重用！）
        """
        # 格式转换：TensorDict → ML4CO Tasks
        tasks = self._tensordict_to_tasks(td)
        
        # 🔑 关键改动：使用 ML4CO-Kit 的 evaluate 方法（重用！）
        rewards = []
        for task in tasks:
            if task.sol is not None:
                try:
                    obj_val = task.evaluate(task.sol)
                    # 根据最大化/最小化调整符号（RL4CO 统一为最大化）
                    reward = obj_val if not task.minimize else -obj_val
                    rewards.append(float(reward))
                except:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)
    
    def check_solution_validity(self, td: TensorDict, actions: torch.Tensor) -> None:
        """
        检查解的有效性 - 使用 ML4CO-Kit 的 check_constraints（重用！）
        """
        tasks = self._tensordict_to_tasks(td)
        
        # 🔑 关键改动：使用 ML4CO-Kit 的 check_constraints 方法（重用！）
        for i, task in enumerate(tasks):
            if task.sol is not None:
                is_valid = task.check_constraints(task.sol)
                if not is_valid:
                    # 🔍 调试信息：打印详细错误
                    selected_nodes = np.where(task.sol == 1)[0]
                    adj_matrix = task.to_adj_matrix()
                    np.fill_diagonal(adj_matrix, 0)
                    conflicts = adj_matrix[selected_nodes][:, selected_nodes]
                    conflict_pairs = np.argwhere(conflicts)
                    
                    print(f"⚠️  Warning: Invalid solution for instance {i}")
                    print(f"   - Selected nodes: {selected_nodes.tolist()}")
                    print(f"   - Num selected: {len(selected_nodes)}")
                    print(f"   - Conflict pairs (within selected): {conflict_pairs.tolist() if len(conflict_pairs) > 0 else 'None'}")
                    if len(conflict_pairs) > 0:
                        for p1, p2 in conflict_pairs[:3]:  # 只显示前3个冲突
                            node1, node2 = selected_nodes[p1], selected_nodes[p2]
                            print(f"   - Conflict: node {node1} and node {node2} are adjacent but both selected!")
    
    def _tensordict_to_tasks(self, td: TensorDict) -> List:
        """
        格式转换：TensorDict → ML4CO Tasks
        这是 Wrapper 的核心工作：格式适配
        """
        batch_size = td.batch_size[0]
        tasks = []
        
        for i in range(batch_size):
            # 创建 ML4CO-Kit Task 实例
            task = self.ml4co_task_class(
                node_weighted=self.node_weighted,
                precision=np.float32
            )
            
            # 提取数据（移除 padding）
            edge_index = td["edge_index"][i].cpu().numpy()
            valid_mask = edge_index[0] >= 0
            valid_edge_index = edge_index[:, valid_mask]
            
            nodes_weight = None
            if self.node_weighted:
                nodes_weight = td["nodes_weight"][i].cpu().numpy()
            
            # 🔑 关键改动：使用 ML4CO-Kit 的 from_data 方法（重用！）
            task.from_data(
                edge_index=valid_edge_index,
                nodes_weight=nodes_weight
            )
            
            # 如果有解，设置解
            if "selected" in td:
                solution = td["selected"][i].cpu().numpy().astype(np.int32)
                task.sol = solution
                
                # 使用 ML4CO-Kit 的 evaluate 方法
                try:
                    task.obj_val = task.evaluate(solution)
                except:
                    pass
            
            tasks.append(task)
        
        return tasks
    
    def solve_with_ml4co(
        self, 
        td: TensorDict, 
        verbose: bool = False,
        return_solutions: bool = False,
        time_limit: float = None
    ) -> dict:
        """
        使用 ML4CO-Kit Solver 求解（重用！）
        
        Args:
            td: TensorDict with problem instances
            verbose: Print detailed solving information
            return_solutions: Return the actual solutions (not just obj values)
            time_limit: Time limit per instance (if solver supports)
        
        Returns:
            dict: {
                'obj_vals': List of objective values,
                'solutions': List of solutions (if return_solutions=True),
                'solve_times': List of solving times per instance,
                'success_rate': Percentage of successfully solved instances,
                'statistics': {mean, std, min, max},
            }
        """
        if self.ml4co_solver is None:
            raise ValueError(
                f"ML4CO Solver is not initialized for {self.name}. "
                f"Please install the required solver."
            )
        
        import time
        
        # 格式转换：TensorDict → ML4CO Tasks
        tasks = self._tensordict_to_tasks(td)
        batch_size = len(tasks)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"ML4CO-Kit Solver: {type(self.ml4co_solver).__name__}")
            print(f"Problem: {self.name.upper()}")
            print(f"Instances: {batch_size}")
            if time_limit:
                print(f"Time limit: {time_limit}s per instance")
            print(f"{'='*70}\n")
        
        # 求解所有实例
        obj_vals = []
        solutions = []
        solve_times = []
        failed_count = 0
        
        for i, task in enumerate(tasks):
            start_time = time.time()
            try:
                # 使用 ML4CO-Kit Solver 求解
                solved_task = self.ml4co_solver.solve(task)
                solve_time = time.time() - start_time
                
                # 提取结果
                obj_val = solved_task.obj_val if hasattr(solved_task, 'obj_val') else 0.0
                obj_vals.append(float(obj_val))
                solve_times.append(solve_time)
                
                if return_solutions:
                    solutions.append(solved_task.sol if hasattr(solved_task, 'sol') else None)
                
                if verbose and (i + 1) % max(1, batch_size // 10) == 0:
                    print(f"  Progress: {i+1}/{batch_size} | "
                          f"Obj: {obj_val:.4f} | Time: {solve_time:.3f}s")
                          
            except Exception as e:
                solve_time = time.time() - start_time
                solve_times.append(solve_time)
                obj_vals.append(0.0)
                solutions.append(None)
                failed_count += 1
                
                if verbose:
                    print(f"  ⚠️  Instance {i} failed: {e}")
        
        # 统计结果
        success_rate = (batch_size - failed_count) / batch_size * 100
        
        results = {
            'obj_vals': obj_vals,
            'solve_times': solve_times,
            'success_rate': success_rate,
            'statistics': {
                'mean': float(np.mean(obj_vals)) if obj_vals else 0.0,
                'std': float(np.std(obj_vals)) if obj_vals else 0.0,
                'min': float(np.min(obj_vals)) if obj_vals else 0.0,
                'max': float(np.max(obj_vals)) if obj_vals else 0.0,
            },
            'timing': {
                'mean_per_instance': float(np.mean(solve_times)) if solve_times else 0.0,
                'total': float(np.sum(solve_times)) if solve_times else 0.0,
            },
            'failed_count': failed_count,
        }
        
        if return_solutions:
            results['solutions'] = solutions
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Results Summary")
            print(f"{'='*70}")
            print(f"Success rate: {success_rate:.1f}% ({batch_size-failed_count}/{batch_size})")
            print(f"\nObjective values:")
            print(f"  Mean: {results['statistics']['mean']:.4f}")
            print(f"  Std:  {results['statistics']['std']:.4f}")
            print(f"  Min:  {results['statistics']['min']:.4f}")
            print(f"  Max:  {results['statistics']['max']:.4f}")
            print(f"\nSolving time:")
            print(f"  Mean per instance: {results['timing']['mean_per_instance']:.3f}s")
            print(f"  Total: {results['timing']['total']:.2f}s")
            print(f"{'='*70}\n")
        
        return results
    
    def render(self, td: TensorDict, idx: int = 0, save_path: str = None):
        """
        可视化 - 使用 ML4CO-Kit 的 render 方法（重用！）
        """
        # 格式转换：TensorDict → ML4CO Task
        tasks = self._tensordict_to_tasks(td)
        task = tasks[idx]
        
        # 🔑 关键改动：使用 ML4CO-Kit 的 render 方法（重用！）
        import pathlib
        if save_path is None:
            save_path = f"{self.name}_{idx}.png"
        
        task.render(save_path=pathlib.Path(save_path))
        print(f"Rendered to {save_path}")


class MISEnvWrapper(ML4COGraphWrapper):
    """
    Maximum Independent Set (MIS) Environment Wrapper
    
    目标：选择最大的独立集（节点集合中任意两个节点不相邻）
    
    观察空间:
        - edge_index: 边索引 [batch, 2, num_edges]
        - nodes_weight: 节点权重 [batch, num_nodes]
        - selected: 已选择的节点 [batch, num_nodes] (binary)
        - available: 可选节点 [batch, num_nodes] (binary)
    
    动作空间:
        - 选择一个可用节点加入独立集
    
    奖励:
        - 最大化选中节点的权重之和
    """
    
    name = "mis"
    
    def __init__(
        self,
        num_nodes: int = 50,
        graph_type: str = 'erdos_renyi',
        edge_prob: float = 0.15,
        node_weighted: bool = False,
        **kwargs
    ):
        # 导入 ML4CO-Kit 的类
        from ml4co_kit.generator.graph.mis import MISGenerator, GRAPH_TYPE
        from ml4co_kit.task.graph.mis import MISTask
        
        # Solver（如果可用）
        try:
            from ml4co_kit.solver.kamis import KaMISSolver
            solver_class = KaMISSolver
        except ImportError:
            print("Warning: KaMIS solver not available")
            solver_class = None
        
        # 映射图类型
        graph_type_map = {
            'erdos_renyi': GRAPH_TYPE.ER,
            'barabasi_albert': GRAPH_TYPE.BA,
            'watts_strogatz': GRAPH_TYPE.WS,
        }
        
        # Generator 参数（使用 ML4CO-Kit 的 API）
        generator_kwargs = {
            'distribution_type': graph_type_map.get(graph_type, GRAPH_TYPE.ER),
            'nodes_num_scale': (num_nodes, num_nodes),  # 固定节点数
            'er_prob': edge_prob,
            'node_weighted': node_weighted,
        }
        
        # 调用父类（使用 ML4CO-Kit Generator！）
        super().__init__(
            ml4co_generator_class=MISGenerator,
            ml4co_task_class=MISTask,
            ml4co_solver_class=solver_class,
            generator_kwargs=generator_kwargs,
            **kwargs
        )
    
    def _init_state(self, td: TensorDict, batch_size: int) -> TensorDict:
        """初始化状态（batch_size 已经是整数）"""
        # 初始化：需要覆盖所有边
        num_edges = td["edge_index"].shape[2] // 2  # 无向图，除以2
        
        td.update({
            "selected": torch.zeros(
                batch_size, self.num_nodes, 
                dtype=torch.bool, 
                device=self.device
            ),
            "available": torch.ones(
                batch_size, self.num_nodes, 
                dtype=torch.bool, 
                device=self.device
            ),
            "i": torch.zeros(batch_size, dtype=torch.int64, device=self.device),
            "done": torch.zeros(batch_size, dtype=torch.bool, device=self.device),
            "current_node": torch.zeros(batch_size, dtype=torch.long, device=self.device),
            "action_mask": torch.ones(
                batch_size, self.num_nodes,
                dtype=torch.bool,
                device=self.device
            ),
        })
        return td
    
    def _step(self, td: TensorDict) -> TensorDict:
        """执行一步：选择一个节点"""
        selected_node = td["action"]
        
        # 🔧 关键修复：只更新未完成的实例
        prev_done = td["done"]  # 之前的 done 状态
        
        # 更新 selected（只更新未完成的实例）
        selected_mask = td["selected"].clone()
        # 只有未完成的实例才真正选择节点
        selected_mask[~prev_done] = selected_mask[~prev_done].scatter(
            -1, 
            selected_node[~prev_done].unsqueeze(-1), 
            1
        )
        
        # 更新 available：移除选中节点及其所有邻居（只对未完成的实例）
        available_mask = td["available"].clone()
        if (~prev_done).any():
            # 只更新未完成的实例的 available
            updated_available = self._update_available(
                td["edge_index"][~prev_done], 
                selected_node[~prev_done],
                td["available"][~prev_done]
            )
            available_mask[~prev_done] = updated_available
        
        # 计算即时奖励（只对未完成的实例）
        reward = torch.zeros_like(prev_done, dtype=torch.float32)
        if (~prev_done).any():
            reward[~prev_done] = gather_by_index(
                td["nodes_weight"][~prev_done], 
                selected_node[~prev_done]
            )
        
        # 检查完成（合并之前的 done 状态）
        done = prev_done | (~available_mask.any(-1))
        
        # 当 done 时，为了避免 decoder 错误，我们需要至少保留一个可用动作
        action_mask = available_mask.clone()
        action_mask[done] = True  # 当 done 时，设置第一个节点为可用（虚拟动作）
        action_mask[done, 1:] = False  # 只保留第一个节点可用
        
        td.update({
            "selected": selected_mask,
            "available": available_mask,
            "reward": reward,
            "done": done,
            "i": td["i"] + 1,
            "current_node": selected_node,
            "action_mask": action_mask,
        })
        
        return td
    
    def _update_available(self, edge_index, selected_nodes, available):
        """更新可用节点：移除选中节点及其邻居"""
        batch_size = edge_index.shape[0]
        available = available.clone()
        
        for b in range(batch_size):
            node_idx = selected_nodes[b].item()
            # 找到该节点的所有邻居
            edges = edge_index[b]  # [2, num_edges]
            
            # 过滤掉 padding 的边（-1）
            valid_mask = edges[0] >= 0
            valid_edges = edges[:, valid_mask]
            
            # 🔧 修复：对于无向图，需要查找两个方向的边
            # 方向1: edge_index[0] == node_idx 的 edge_index[1]
            neighbors_1 = valid_edges[1, valid_edges[0] == node_idx]
            # 方向2: edge_index[1] == node_idx 的 edge_index[0]
            neighbors_2 = valid_edges[0, valid_edges[1] == node_idx]
            
            # 合并两个方向的邻居
            neighbors = torch.cat([neighbors_1, neighbors_2]).unique()
            
            # 移除节点本身和所有邻居
            available[b, node_idx] = False
            if len(neighbors) > 0:
                available[b, neighbors] = False
        
        return available
    
    def _set_task_data(self, task, edge_index, nodes_weight, td, idx):
        """设置 MISTask 数据"""
        # 移除 padding 的边（-1）
        valid_mask = edge_index[0] >= 0
        valid_edge_index = edge_index[:, valid_mask]
        
        # 使用 from_data 方法
        task.from_data(
            nodes_num=self.num_nodes,
            edge_index=valid_edge_index,
            nodes_weight=nodes_weight if self.node_weighted else None
        )
        
        # 如果有解，设置解
        if "selected" in td:
            solution = td["selected"][idx].cpu().numpy().astype(np.int32)
            task.sol = solution
            try:
                task.obj_val = task.evaluate(solution)
            except:
                pass
    
    @staticmethod
    def render(td: TensorDict, actions=None, ax=None):
        """可视化"""
        import matplotlib.pyplot as plt
        import networkx as nx
        
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))
        
        # 取第一个样本
        edge_index = td["edge_index"][0].cpu().numpy()
        selected = td.get("selected", None)
        if selected is not None:
            selected = selected[0].cpu().numpy()
        
        # 构建 NetworkX 图
        G = nx.Graph()
        G.add_nodes_from(range(td["nodes_num"][0].item()))
        edges = [(edge_index[0, i], edge_index[1, i]) 
                 for i in range(edge_index.shape[1]) if edge_index[0, i] < edge_index[1, i]]
        G.add_edges_from(edges)
        
        # 布局
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # 节点颜色
        if selected is not None:
            node_colors = ['orange' if selected[i] else 'lightblue' 
                          for i in range(len(G.nodes()))]
            title = f"MIS Solution (Size: {selected.sum()})"
        else:
            node_colors = 'lightblue'
            title = "MIS Problem Instance"
        
        # 绘制
        nx.draw(G, pos, node_color=node_colors, with_labels=True, 
               node_size=500, font_size=10, font_weight='bold', ax=ax)
        ax.set_title(title)


class MVCEnvWrapper(ML4COGraphWrapper):
    """
    Minimum Vertex Cover (MVC) Environment Wrapper
    
    目标：选择最小的顶点覆盖（所有边至少有一个端点被选中）
    
    MVC 是 MIS 的对偶问题：V \ MVC(G) = MIS(G)
    """
    
    name = "mvc"
    
    def __init__(
        self,
        num_nodes: int = 50,
        graph_type: str = 'erdos_renyi',
        edge_prob: float = 0.15,
        node_weighted: bool = False,
        **kwargs
    ):
        # 导入 ML4CO-Kit 的类
        from ml4co_kit.generator.graph.mvc import MVCGenerator, GRAPH_TYPE
        from ml4co_kit.task.graph.mvc import MVCTask
        
        # 映射图类型
        graph_type_map = {
            'erdos_renyi': GRAPH_TYPE.ER,
            'barabasi_albert': GRAPH_TYPE.BA,
            'watts_strogatz': GRAPH_TYPE.WS,
        }
        
        # Generator 参数
        generator_kwargs = {
            'distribution_type': graph_type_map.get(graph_type, GRAPH_TYPE.ER),
            'nodes_num_scale': (num_nodes, num_nodes),
            'er_prob': edge_prob,
            'node_weighted': node_weighted,
        }
        
        # 调用父类（使用 ML4CO-Kit Generator！）
        super().__init__(
            ml4co_generator_class=MVCGenerator,
            ml4co_task_class=MVCTask,
            ml4co_solver_class=None,
            generator_kwargs=generator_kwargs,
            **kwargs
        )
    
    def _init_state(self, td: TensorDict, batch_size) -> TensorDict:
        """初始化状态"""
        # 处理各种 batch_size 格式
        if isinstance(batch_size, (tuple, list)):
            batch_size = batch_size[0]
        if isinstance(batch_size, torch.Tensor):
            batch_size = batch_size.item()
        batch_size = int(batch_size)
        
        # 初始化：需要覆盖所有边
        num_edges = td["edge_index"].shape[2] // 2  # 无向图，除以2
        
        td.update({
            "selected": torch.zeros(
                batch_size, self.num_nodes, 
                dtype=torch.bool, 
                device=self.device
            ),
            "covered_edges": torch.zeros(
                batch_size, num_edges, 
                dtype=torch.bool, 
                device=self.device
            ),
            "i": torch.zeros(batch_size, dtype=torch.int64, device=self.device),
        })
        return td
    
    def _step(self, td: TensorDict) -> TensorDict:
        """执行一步：选择一个节点加入顶点覆盖"""
        selected_node = td["action"]
        
        # 🔧 关键修复：只更新未完成的实例
        prev_done = td["done"]
        
        # 更新 selected（只更新未完成的实例）
        selected_mask = td["selected"].clone()
        if (~prev_done).any():
            selected_mask[~prev_done] = selected_mask[~prev_done].scatter(
                -1, 
                selected_node[~prev_done].unsqueeze(-1), 
                1
            )
        
        # 更新 covered_edges：标记该节点覆盖的边（只对未完成的实例）
        covered_edges = td["covered_edges"].clone()
        if (~prev_done).any():
            updated_covered = self._update_covered_edges(
                td["edge_index"][~prev_done],
                selected_node[~prev_done],
                td["covered_edges"][~prev_done]
            )
            covered_edges[~prev_done] = updated_covered
        
        # 计算惩罚（最小化问题，每选一个节点都是代价）
        reward = torch.zeros_like(prev_done, dtype=torch.float32)
        if (~prev_done).any():
            reward[~prev_done] = -gather_by_index(
                td["nodes_weight"][~prev_done], 
                selected_node[~prev_done]
            )
        
        # 检查完成：所有边都被覆盖（合并之前的 done 状态）
        done = prev_done | covered_edges.all(-1)
        
        td.update({
            "selected": selected_mask,
            "covered_edges": covered_edges,
            "reward": reward,
            "done": done,
            "i": td["i"] + 1,
        })
        
        return td
    
    def _update_covered_edges(self, edge_index, selected_nodes, covered_edges):
        """更新已覆盖的边"""
        batch_size = edge_index.shape[0]
        covered_edges = covered_edges.clone()
        
        for b in range(batch_size):
            node_idx = selected_nodes[b].item()
            edges = edge_index[b]  # [2, num_edges]
            
            # 🔧 修复：过滤掉 padding 的边
            valid_mask = edges[0] >= 0
            
            # 找到包含该节点的所有边（只计算一半，因为是无向图）
            edge_mask = valid_mask & ((edges[0] == node_idx) | (edges[1] == node_idx)) & (edges[0] < edges[1])
            edge_indices = edge_mask.nonzero(as_tuple=True)[0]
            
            # 标记这些边为已覆盖
            if len(edge_indices) > 0:
                covered_edges[b, edge_indices] = True
        
        return covered_edges
    
    def _set_task_data(self, task, edge_index, nodes_weight, td, idx):
        """设置 MVCTask 数据"""
        task.from_data(
            nodes_num=self.num_nodes,
            edge_index=edge_index,
            nodes_weight=nodes_weight if self.node_weighted else None
        )
        
        if "selected" in td:
            solution = td["selected"][idx].cpu().numpy().astype(np.int32)
            task.sol = solution
            try:
                task.obj_val = task.evaluate(solution)
            except:
                pass
    
    @staticmethod
    def render(td: TensorDict, actions=None, ax=None):
        """可视化"""
        import matplotlib.pyplot as plt
        import networkx as nx
        
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))
        
        edge_index = td["edge_index"][0].cpu().numpy()
        selected = td.get("selected", None)
        if selected is not None:
            selected = selected[0].cpu().numpy()
        
        G = nx.Graph()
        G.add_nodes_from(range(td["nodes_num"][0].item()))
        edges = [(edge_index[0, i], edge_index[1, i]) 
                 for i in range(edge_index.shape[1]) if edge_index[0, i] < edge_index[1, i]]
        G.add_edges_from(edges)
        
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        if selected is not None:
            node_colors = ['red' if selected[i] else 'lightblue' 
                          for i in range(len(G.nodes()))]
            title = f"MVC Solution (Size: {selected.sum()})"
        else:
            node_colors = 'lightblue'
            title = "MVC Problem Instance"
        
        nx.draw(G, pos, node_color=node_colors, with_labels=True, 
               node_size=500, font_size=10, font_weight='bold', ax=ax)
        ax.set_title(title)


class MCLEnvWrapper(ML4COGraphWrapper):
    """
    Maximum Clique (MCL) Environment Wrapper
    
    目标：找到最大的团（完全子图，任意两个节点都相邻）
    """
    
    name = "mcl"
    
    def __init__(
        self,
        num_nodes: int = 50,
        graph_type: str = 'erdos_renyi',
        edge_prob: float = 0.15,
        node_weighted: bool = False,
        **kwargs
    ):
        from ml4co_kit.task.graph.mcl import MClTask  # 注意：是 MClTask 不是 MCLTask
        
        super().__init__(
            task_class=MClTask,
            task_params={'node_weighted': node_weighted},
            num_nodes=num_nodes,
            graph_type=graph_type,
            graph_params={'edge_prob': edge_prob},
            solver_class=None,
            **kwargs
        )
        
        self.node_weighted = node_weighted
    
    def _init_state(self, td: TensorDict, batch_size: int) -> TensorDict:
        """初始化状态（batch_size 已经是整数）"""
        # batch_size 已经在 reset() 中标准化为整数了
        
        td.update({
            "selected": torch.zeros(
                batch_size, self.num_nodes, 
                dtype=torch.bool, 
                device=self.device
            ),
            "available": torch.ones(
                batch_size, self.num_nodes, 
                dtype=torch.bool, 
                device=self.device
            ),
            "i": torch.zeros(batch_size, dtype=torch.int64, device=self.device),
            "done": torch.zeros(batch_size, dtype=torch.bool, device=self.device),
            "current_node": torch.zeros(batch_size, dtype=torch.long, device=self.device),
            "action_mask": torch.ones(
                batch_size, self.num_nodes,
                dtype=torch.bool,
                device=self.device
            ),
        })
        return td
    
    def _step(self, td: TensorDict) -> TensorDict:
        """执行一步：选择一个节点加入团"""
        selected_node = td["action"]
        
        # 🔧 关键修复：只更新未完成的实例
        prev_done = td["done"]
        
        # 更新 selected（只更新未完成的实例）
        selected_mask = td["selected"].clone()
        if (~prev_done).any():
            selected_mask[~prev_done] = selected_mask[~prev_done].scatter(
                -1, 
                selected_node[~prev_done].unsqueeze(-1), 
                1
            )
        
        # 更新 available：只保留与当前团中所有节点都相邻的节点（只对未完成的实例）
        available_mask = td["available"].clone()
        if (~prev_done).any():
            updated_available = self._update_available_for_clique(
                td["edge_index"][~prev_done],
                selected_mask[~prev_done],
                td["available"][~prev_done]
            )
            available_mask[~prev_done] = updated_available
        
        # 计算奖励（只对未完成的实例）
        reward = torch.zeros_like(prev_done, dtype=torch.float32)
        if (~prev_done).any():
            reward[~prev_done] = gather_by_index(
                td["nodes_weight"][~prev_done], 
                selected_node[~prev_done]
            )
        
        # 检查完成（合并之前的 done 状态）
        done = prev_done | (~available_mask.any(-1))
        
        # 当 done 时，为了避免 decoder 错误，我们需要至少保留一个可用动作
        action_mask = available_mask.clone()
        action_mask[done] = True  # 当 done 时，设置第一个节点为可用（虚拟动作）
        action_mask[done, 1:] = False  # 只保留第一个节点可用
        
        td.update({
            "selected": selected_mask,
            "available": available_mask,
            "reward": reward,
            "done": done,
            "i": td["i"] + 1,
            "current_node": selected_node,
            "action_mask": action_mask,
        })
        
        return td
    
    def _update_available_for_clique(self, edge_index, selected, available):
        """更新可用节点：只保留与所有已选节点都相邻的节点"""
        batch_size = edge_index.shape[0]
        available = available.clone()
        
        for b in range(batch_size):
            selected_nodes = selected[b].nonzero(as_tuple=True)[0]
            if len(selected_nodes) == 0:
                continue
            
            edges = edge_index[b]
            # 过滤掉 padding 的边
            valid_mask = edges[0] >= 0
            valid_edges = edges[:, valid_mask]
            
            # 对每个候选节点，检查是否与所有已选节点相邻
            for node in range(self.num_nodes):
                if not available[b, node]:
                    continue
                
                # 检查是否与所有已选节点相邻
                is_connected_to_all = True
                for selected_node in selected_nodes:
                    # 🔧 修复：对于无向图，需要检查两个方向
                    # 方向1: (node, selected_node)
                    mask1 = (valid_edges[0] == node) & (valid_edges[1] == selected_node.item())
                    # 方向2: (selected_node, node)
                    mask2 = (valid_edges[0] == selected_node.item()) & (valid_edges[1] == node)
                    
                    if not (mask1.any() or mask2.any()):
                        is_connected_to_all = False
                        break
                
                if not is_connected_to_all:
                    available[b, node] = False
        
        return available
    
    def _set_task_data(self, task, edge_index, nodes_weight, td, idx):
        """设置 MCLTask 数据"""
        task.from_data(
            nodes_num=self.num_nodes,
            edge_index=edge_index,
            nodes_weight=nodes_weight if self.node_weighted else None
        )
        
        if "selected" in td:
            solution = td["selected"][idx].cpu().numpy().astype(np.int32)
            task.sol = solution
            try:
                task.obj_val = task.evaluate(solution)
            except:
                pass
    
    @staticmethod
    def render(td: TensorDict, actions=None, ax=None):
        """可视化"""
        import matplotlib.pyplot as plt
        import networkx as nx
        
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))
        
        edge_index = td["edge_index"][0].cpu().numpy()
        selected = td.get("selected", None)
        if selected is not None:
            selected = selected[0].cpu().numpy()
        
        G = nx.Graph()
        G.add_nodes_from(range(td["nodes_num"][0].item()))
        edges = [(edge_index[0, i], edge_index[1, i]) 
                 for i in range(edge_index.shape[1]) if edge_index[0, i] < edge_index[1, i]]
        G.add_edges_from(edges)
        
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        if selected is not None:
            node_colors = ['green' if selected[i] else 'lightblue' 
                          for i in range(len(G.nodes()))]
            title = f"Maximum Clique (Size: {selected.sum()})"
        else:
            node_colors = 'lightblue'
            title = "MCL Problem Instance"
        
        nx.draw(G, pos, node_color=node_colors, with_labels=True, 
               node_size=500, font_size=10, font_weight='bold', ax=ax)
        ax.set_title(title)


class MCUTEnvWrapper(ML4COGraphWrapper):
    """
    Maximum Cut (MCUT) Environment Wrapper
    
    目标：将图的节点分成两个集合，最大化两集合之间的边数
    """
    
    name = "mcut"
    
    def __init__(
        self,
        num_nodes: int = 50,
        graph_type: str = 'erdos_renyi',
        edge_prob: float = 0.15,
        edge_weighted: bool = False,
        **kwargs
    ):
        from ml4co_kit.task.graph.mcut import MCutTask  # 注意：是 MCutTask 不是 MCUTTask
        
        super().__init__(
            task_class=MCutTask,
            task_params={'edge_weighted': edge_weighted},
            num_nodes=num_nodes,
            graph_type=graph_type,
            graph_params={'edge_prob': edge_prob},
            solver_class=None,
            **kwargs
        )
        
        self.edge_weighted = edge_weighted
    
    def _generate_graph(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """生成图（覆盖父类方法以支持边权重）"""
        edge_index, _ = super()._generate_graph()
        
        # 生成边权重
        if self.edge_weighted:
            num_edges = edge_index.shape[1]
            edges_weight = np.random.rand(num_edges).astype(np.float32)
        else:
            edges_weight = None
        
        return edge_index, edges_weight
    
    def generate_data(self, batch_size) -> TensorDict:
        """生成数据（覆盖以支持边权重）"""
        # 处理各种 batch_size 格式
        if isinstance(batch_size, (tuple, list)):
            batch_size = batch_size[0]
        if isinstance(batch_size, torch.Tensor):
            batch_size = batch_size.item()
        batch_size = int(batch_size)
        
        edge_indices = []
        edges_weights = []
        edge_nums = []
        
        for _ in range(batch_size):
            edge_index, edges_weight = self._generate_graph()
            edge_indices.append(torch.from_numpy(edge_index).long())
            edge_nums.append(edge_index.shape[1])
            
            if edges_weight is not None:
                edges_weights.append(torch.from_numpy(edges_weight).float())
            else:
                num_edges = edge_index.shape[1]
                edges_weights.append(torch.ones(num_edges, dtype=torch.float32))
        
        # Pad to same length
        max_edges = max(edge_nums)
        padded_edge_indices = []
        padded_edges_weights = []
        
        for edge_idx, edge_weight, num_edges in zip(edge_indices, edges_weights, edge_nums):
            if num_edges < max_edges:
                # Pad edge_index with -1
                padding = torch.full((2, max_edges - num_edges), -1, dtype=torch.long)
                edge_idx = torch.cat([edge_idx, padding], dim=1)
                # Pad edge_weight with 0
                weight_padding = torch.zeros(max_edges - num_edges, dtype=torch.float32)
                edge_weight = torch.cat([edge_weight, weight_padding])
            padded_edge_indices.append(edge_idx)
            padded_edges_weights.append(edge_weight)
        
        td = TensorDict({
            "edge_index": torch.stack(padded_edge_indices).to(self.device),
            "edges_weight": torch.stack(padded_edges_weights).to(self.device),
            "nodes_num": torch.full((batch_size,), self.num_nodes, 
                                   dtype=torch.long, device=self.device),
            "edge_nums": torch.tensor(edge_nums, dtype=torch.long, device=self.device)
        }, batch_size=[batch_size])
        
        return td
    
    def _init_state(self, td: TensorDict, batch_size: int) -> TensorDict:
        """初始化状态（batch_size 已经是整数）"""
        # batch_size 已经在 reset() 中标准化为整数了
        
        # partition: 0 或 1，表示节点属于哪个分区
        td.update({
            "partition": torch.zeros(
                batch_size, self.num_nodes, 
                dtype=torch.long, 
                device=self.device
            ),
            "i": torch.zeros(batch_size, dtype=torch.int64, device=self.device),
        })
        return td
    
    def _step(self, td: TensorDict) -> TensorDict:
        """执行一步：将一个节点分配到某个分区"""
        # action 可以是节点索引，或者 (节点索引, 分区编号)
        node_idx = td["action"]
        
        # 简化：将节点分配到分区1，其余在分区0
        partition = td["partition"].clone()
        partition.scatter_(-1, node_idx.unsqueeze(-1), 1)
        
        # 计算当前的 cut 值（跨分区的边权重和）
        cut_value = self._calculate_cut(td["edge_index"], td["edges_weight"], partition)
        reward = cut_value - td.get("prev_cut", torch.zeros_like(cut_value))
        
        # 检查完成（所有节点都已分配）
        done = td["i"] >= self.num_nodes - 1
        
        td.update({
            "partition": partition,
            "reward": reward,
            "prev_cut": cut_value,
            "done": done,
            "i": td["i"] + 1,
        })
        
        return td
    
    def _calculate_cut(self, edge_index, edges_weight, partition):
        """计算 cut 值"""
        batch_size = edge_index.shape[0]
        cut_values = torch.zeros(batch_size, device=self.device)
        
        for b in range(batch_size):
            edges = edge_index[b]  # [2, num_edges]
            weights = edges_weight[b]
            part = partition[b]
            
            # 只计算一半的边（无向图）
            for e in range(edges.shape[1]):
                if edges[0, e] < edges[1, e]:  # 避免重复计数
                    u, v = edges[0, e].item(), edges[1, e].item()
                    if part[u] != part[v]:  # 跨分区的边
                        cut_values[b] += weights[e]
        
        return cut_values
    
    def _set_task_data(self, task, edge_index, nodes_weight, td, idx):
        """设置 MCUTTask 数据"""
        edges_weight = td["edges_weight"][idx].cpu().numpy() if self.edge_weighted else None
        
        task.from_data(
            nodes_num=self.num_nodes,
            edge_index=edge_index,
            edges_weight=edges_weight
        )
        
        if "partition" in td:
            solution = td["partition"][idx].cpu().numpy().astype(np.int32)
            task.sol = solution
            try:
                task.obj_val = task.evaluate(solution)
            except:
                pass
    
    @staticmethod
    def render(td: TensorDict, actions=None, ax=None):
        """可视化"""
        import matplotlib.pyplot as plt
        import networkx as nx
        
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))
        
        edge_index = td["edge_index"][0].cpu().numpy()
        partition = td.get("partition", None)
        if partition is not None:
            partition = partition[0].cpu().numpy()
        
        G = nx.Graph()
        G.add_nodes_from(range(td["nodes_num"][0].item()))
        edges = [(edge_index[0, i], edge_index[1, i]) 
                 for i in range(edge_index.shape[1]) if edge_index[0, i] < edge_index[1, i]]
        G.add_edges_from(edges)
        
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        if partition is not None:
            node_colors = ['salmon' if partition[i] == 0 else 'lightgreen' 
                          for i in range(len(G.nodes()))]
            title = "Maximum Cut Solution"
        else:
            node_colors = 'lightblue'
            title = "MCUT Problem Instance"
        
        nx.draw(G, pos, node_color=node_colors, with_labels=True, 
               node_size=500, font_size=10, font_weight='bold', ax=ax)
        ax.set_title(title)
