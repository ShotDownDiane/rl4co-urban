import torch
import torch.nn as nn

from tensordict import TensorDict

from rl4co.utils.ops import batched_scatter_sum, gather_by_index


def env_context_embedding(env_name: str, config: dict) -> nn.Module:
    """Get environment context embedding. The context embedding is used to modify the
    query embedding of the problem node of the current partial solution.
    Usually consists of a projection of gathered node embeddings and features to the embedding space.

    Args:
        env: Environment or its name.
        config: A dictionary of configuration options for the environment.
    """
    embedding_registry = {
        "tsp": TSPContext,
        "atsp": TSPContext,
        "cvrp": VRPContext,
        "cvrptw": VRPTWContext,
        "cvrpmvc": VRPContext,
        "ffsp": FFSPContext,
        "svrp": SVRPContext,
        "sdvrp": VRPContext,
        "pctsp": PCTSPContext,
        "spctsp": PCTSPContext,
        "op": OPContext,
        "dpp": DPPContext,
        "mdpp": DPPContext,
        "pdp": PDPContext,
        "mdcpdp": MDCPDPContext,
        "mtsp": MTSPContext,
        "smtwtp": SMTWTPContext,
        "mtvrp": MTVRPContext,
        "shpp": TSPContext,
        "flp": FLPContext,
        "mcp": MCPContext,
        "mclp": MCLPContext,
        "stp": STPContext,
        # Graph problems (ML4CO wrappers)
        "mis": GraphContext,
        "mvc": GraphContext,
        "mcl": GraphContext,
        "mcut": GraphContext,
    }

    if env_name not in embedding_registry:
        raise ValueError(
            f"Unknown environment name '{env_name}'. Available context embeddings: {embedding_registry.keys()}"
        )

    return embedding_registry[env_name](**config)


class EnvContext(nn.Module):
    """Base class for environment context embeddings. The context embedding is used to modify the
    query embedding of the problem node of the current partial solution.
    Consists of a linear layer that projects the node features to the embedding space."""

    def __init__(self, embed_dim, step_context_dim=None, linear_bias=False):
        super(EnvContext, self).__init__()
        self.embed_dim = embed_dim
        step_context_dim = step_context_dim if step_context_dim is not None else embed_dim
        self.project_context = nn.Linear(step_context_dim, embed_dim, bias=linear_bias)

    def _cur_node_embedding(self, embeddings, td):
        """Get embedding of current node"""
        cur_node_embedding = gather_by_index(embeddings, td["current_node"])
        return cur_node_embedding

    def _state_embedding(self, embeddings, td):
        """Get state embedding"""
        raise NotImplementedError("Implement for each environment")

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        state_embedding = self._state_embedding(embeddings, td)
        context_embedding = torch.cat([cur_node_embedding, state_embedding], -1)
        return self.project_context(context_embedding)


class FFSPContext(EnvContext):
    def __init__(self, embed_dim, stage_cnt=None):
        self.has_stage_emb = stage_cnt is not None
        step_context_dim = (1 + int(self.has_stage_emb)) * embed_dim
        super().__init__(embed_dim=embed_dim, step_context_dim=step_context_dim)
        if self.has_stage_emb:
            self.stage_emb = nn.Parameter(torch.rand(stage_cnt, embed_dim))

    def _cur_node_embedding(self, embeddings: TensorDict, td):
        cur_node_embedding = gather_by_index(
            embeddings["machine_embeddings"], td["stage_machine_idx"]
        )
        return cur_node_embedding

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        if self.has_stage_emb:
            state_embedding = self._state_embedding(embeddings, td)
            context_embedding = torch.cat([cur_node_embedding, state_embedding], -1)
            return self.project_context(context_embedding)
        else:
            return self.project_context(cur_node_embedding)

    def _state_embedding(self, _, td):
        cur_stage_emb = self.stage_emb[td["stage_idx"]]
        return cur_stage_emb


class TSPContext(EnvContext):
    """Context embedding for the Traveling Salesman Problem (TSP).
    Project the following to the embedding space:
        - first node embedding
        - current node embedding
    """

    def __init__(self, embed_dim):
        super(TSPContext, self).__init__(embed_dim, 2 * embed_dim)
        self.W_placeholder = nn.Parameter(
            torch.Tensor(2 * self.embed_dim).uniform_(-1, 1)
        )

    def forward(self, embeddings, td):
        batch_size = embeddings.size(0)
        # By default, node_dim = -1 (we only have one node embedding per node)
        node_dim = (
            (-1,) if td["first_node"].dim() == 1 else (td["first_node"].size(-1), -1)
        )
        if td["i"][(0,) * td["i"].dim()].item() < 1:  # get first item fast
            if len(td.batch_size) < 2:
                context_embedding = self.W_placeholder[None, :].expand(
                    batch_size, self.W_placeholder.size(-1)
                )
            else:
                context_embedding = self.W_placeholder[None, None, :].expand(
                    batch_size, td.batch_size[1], self.W_placeholder.size(-1)
                )
        else:
            context_embedding = gather_by_index(
                embeddings,
                torch.stack([td["first_node"], td["current_node"]], -1).view(
                    batch_size, -1
                ),
            ).view(batch_size, *node_dim)
        return self.project_context(context_embedding)


class VRPContext(EnvContext):
    """Context embedding for the Capacitated Vehicle Routing Problem (CVRP).
    Project the following to the embedding space:
        - current node embedding
        - remaining capacity (vehicle_capacity - used_capacity)
    """

    def __init__(self, embed_dim):
        super(VRPContext, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim + 1
        )

    def _state_embedding(self, embeddings, td):
        state_embedding = td["vehicle_capacity"] - td["used_capacity"]
        return state_embedding


class VRPTWContext(VRPContext):
    """Context embedding for the Capacitated Vehicle Routing Problem (CVRP).
    Project the following to the embedding space:
        - current node embedding
        - remaining capacity (vehicle_capacity - used_capacity)
        - current time
    """

    def __init__(self, embed_dim):
        super(VRPContext, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim + 2
        )

    def _state_embedding(self, embeddings, td):
        capacity = super()._state_embedding(embeddings, td)
        current_time = td["current_time"]
        return torch.cat([capacity, current_time], -1)


class SVRPContext(EnvContext):
    """Context embedding for the Skill Vehicle Routing Problem (SVRP).
    Project the following to the embedding space:
        - current node embedding
        - current technician
    """

    def __init__(self, embed_dim):
        super(SVRPContext, self).__init__(embed_dim=embed_dim, step_context_dim=embed_dim)

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td).squeeze()
        return self.project_context(cur_node_embedding)


class PCTSPContext(EnvContext):
    """Context embedding for the Prize Collecting TSP (PCTSP).
    Project the following to the embedding space:
        - current node embedding
        - remaining prize (prize_required - cur_total_prize)
    """

    def __init__(self, embed_dim):
        super(PCTSPContext, self).__init__(embed_dim, embed_dim + 1)

    def _state_embedding(self, embeddings, td):
        state_embedding = torch.clamp(
            td["prize_required"] - td["cur_total_prize"], min=0
        )[..., None]
        return state_embedding


class OPContext(EnvContext):
    """Context embedding for the Orienteering Problem (OP).
    Project the following to the embedding space:
        - current node embedding
        - remaining distance (max_length - tour_length)
    """

    def __init__(self, embed_dim):
        super(OPContext, self).__init__(embed_dim, embed_dim + 1)

    def _state_embedding(self, embeddings, td):
        state_embedding = td["max_length"][..., 0] - td["tour_length"]
        return state_embedding[..., None]


class DPPContext(EnvContext):
    """Context embedding for the Decap Placement Problem (DPP), EDA (electronic design automation).
    Project the following to the embedding space:
        - current cell embedding
    """

    def __init__(self, embed_dim):
        super(DPPContext, self).__init__(embed_dim)

    def forward(self, embeddings, td):
        """Context cannot be defined by a single node embedding for DPP, hence 0.
        We modify the dynamic embedding instead to capture placed items
        """
        return embeddings.new_zeros(embeddings.size(0), self.embed_dim)


class PDPContext(EnvContext):
    """Context embedding for the Pickup and Delivery Problem (PDP).
    Project the following to the embedding space:
        - current node embedding
    """

    def __init__(self, embed_dim):
        super(PDPContext, self).__init__(embed_dim, embed_dim)

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td).squeeze()
        return self.project_context(cur_node_embedding)


class MTSPContext(EnvContext):
    """Context embedding for the Multiple Traveling Salesman Problem (mTSP).
    Project the following to the embedding space:
        - current node embedding
        - remaining_agents
        - current_length
        - max_subtour_length
        - distance_from_depot
    """

    def __init__(self, embed_dim, linear_bias=False):
        super(MTSPContext, self).__init__(embed_dim, 2 * embed_dim)
        proj_in_dim = (
            4  # remaining_agents, current_length, max_subtour_length, distance_from_depot
        )
        self.proj_dynamic_feats = nn.Linear(proj_in_dim, embed_dim, bias=linear_bias)

    def _cur_node_embedding(self, embeddings, td):
        cur_node_embedding = gather_by_index(embeddings, td["current_node"])
        return cur_node_embedding.squeeze()

    def _state_embedding(self, embeddings, td):
        dynamic_feats = torch.stack(
            [
                (td["num_agents"] - td["agent_idx"]).float(),
                td["current_length"],
                td["max_subtour_length"],
                self._distance_from_depot(td),
            ],
            dim=-1,
        )
        return self.proj_dynamic_feats(dynamic_feats)

    def _distance_from_depot(self, td):
        # Euclidean distance from the depot (loc[..., 0, :])
        cur_loc = gather_by_index(td["locs"], td["current_node"])
        return torch.norm(cur_loc - td["locs"][..., 0, :], dim=-1)


class SMTWTPContext(EnvContext):
    """Context embedding for the Single Machine Total Weighted Tardiness Problem (SMTWTP).
    Project the following to the embedding space:
        - current node embedding
        - current time
    """

    def __init__(self, embed_dim):
        super(SMTWTPContext, self).__init__(embed_dim, embed_dim + 1)

    def _cur_node_embedding(self, embeddings, td):
        cur_node_embedding = gather_by_index(embeddings, td["current_job"])
        return cur_node_embedding

    def _state_embedding(self, embeddings, td):
        state_embedding = td["current_time"]
        return state_embedding


class MDCPDPContext(EnvContext):
    """Context embedding for the MDCPDP.
    Project the following to the embedding space:
        - current node embedding
    """

    def __init__(self, embed_dim):
        super(MDCPDPContext, self).__init__(embed_dim, embed_dim * 2 + 5)

    def _state_embedding(self, embeddings, td):
        # get number of visited cities over total
        num_agents = td["capacity"].shape[-1]
        num_cities = td["locs"].shape[-2] - num_agents
        unvisited_number = td["available"][..., num_agents:].sum(-1)
        agent_capacity = td["capacity"].gather(-1, td["current_depot"])
        current_to_deliver = td["to_deliver"][..., num_agents + num_cities // 2 :]

        context_feats = torch.cat(
            [
                agent_capacity - td["current_carry"],  # current available capacity
                td["current_length"].gather(-1, td["current_depot"]),
                unvisited_number[..., None] / num_cities,
                current_to_deliver.sum(-1)[..., None],  # current to deliver number
                td["current_length"].max(-1)[0][..., None],  # max length
            ],
            -1,
        )
        return context_feats

    def _cur_agent_embedding(self, embeddings, td):
        """Get embedding of current agent"""
        cur_agent_embedding = gather_by_index(embeddings, td["current_depot"])
        return cur_agent_embedding

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        cur_agent_embedding = self._cur_agent_embedding(embeddings, td)
        state_embedding = self._state_embedding(embeddings, td)
        context_embedding = torch.cat(
            [cur_node_embedding, cur_agent_embedding, state_embedding], -1
        )
        return self.project_context(context_embedding)


class SchedulingContext(nn.Module):
    def __init__(self, embed_dim: int, scaling_factor: int = 1000):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.proj_busy = nn.Linear(1, embed_dim, bias=False)

    def forward(self, h, td):
        busy_for = (td["busy_until"] - td["time"].unsqueeze(1)) / self.scaling_factor
        busy_proj = self.proj_busy(busy_for.unsqueeze(-1))
        # (b m e)
        return h + busy_proj


class MTVRPContext(VRPContext):
    """Context embedding for Multi-Task VRPEnv.
    Project the following to the embedding space:
        - current node embedding
        - remaining_linehaul_capacity (vehicle_capacity - used_capacity_linehaul)
        - remaining_backhaul_capacity (vehicle_capacity - used_capacity_backhaul)
        - current time
        - current_route_length
        - open route indicator
    """

    def __init__(self, embed_dim):
        super(VRPContext, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim + 5
        )

    def _state_embedding(self, embeddings, td):
        remaining_linehaul_capacity = (
            td["vehicle_capacity"] - td["used_capacity_linehaul"]
        )
        remaining_backhaul_capacity = (
            td["vehicle_capacity"] - td["used_capacity_backhaul"]
        )
        current_time = td["current_time"]
        current_route_length = td["current_route_length"]
        open_route = td["open_route"]
        return torch.cat(
            [
                remaining_linehaul_capacity,
                remaining_backhaul_capacity,
                current_time,
                current_route_length,
                open_route,
            ],
            -1,
        )


class FLPContext(EnvContext):
    """Context embedding for the Facility Location Problem (FLP)."""

    def __init__(self, embed_dim: int):
        super(FLPContext, self).__init__(embed_dim=embed_dim)
        self.embed_dim = embed_dim
        self.project_context = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, embeddings, td):
        cur_dist = td["distances"].unsqueeze(-2)  # (batch_size, 1, n_points)
        dist_improve = cur_dist - td["orig_distances"]  # (batch_size, n_points, n_points)
        dist_improve = torch.clamp(dist_improve, min=0).sum(-1)  # (batch_size, n_points)

        # softmax
        loc_best_soft = torch.softmax(dist_improve, dim=-1)  # (batch_size, n_points)
        context_embedding = (embeddings * loc_best_soft[..., None]).sum(-2)
        return self.project_context(context_embedding)


class MCPContext(EnvContext):
    """Context embedding for the Maximum Coverage Problem (MCP)."""

    def __init__(self, embed_dim: int):
        super(MCPContext, self).__init__(embed_dim=embed_dim)
        self.embed_dim = embed_dim
        self.project_context = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, embeddings, td):
        membership_weighted = batched_scatter_sum(
            td["weights"].unsqueeze(-1), td["membership"].long()
        )
        membership_weighted.squeeze_(-1)
        # membership_weighted: [batch_size, n_sets]

        # softmax; higher weights for better sets
        membership_weighted = torch.softmax(
            membership_weighted, dim=-1
        )  # (batch_size, n_sets)
        context_embedding = (membership_weighted.unsqueeze(-1) * embeddings).sum(1)
        return self.project_context(context_embedding)

class MCLPContext(EnvContext):
    def __init__(self, embed_dim: int):
        super(MCLPContext, self).__init__(embed_dim=embed_dim)
        self.embed_dim = embed_dim

        # 输入维度解释:
        # 1. weighted_embedding [embed_dim]: 通过 Softmax 加权后的设施特征
        # 2. step_progress [1]: 进度条
        # 3. covered_fraction [1]: 覆盖率
        # 4. max_gain_ratio [1]: 当前盘面上最大的潜在收益 (强度特征)
        self.project_context = nn.Linear(embed_dim + 3, embed_dim, bias=True)

    def forward(self, embeddings, td):
        # --- 1. 懒加载静态覆盖 Mask (Lazy Initialization) ---
        if self.coverage_mask is None:
            dist_matrix = td["distance_matrix"]
            radius = td["coverage_radius"]
            
            # 维度对齐
            if radius.dim() == 1:
                radius = radius.view(-1, 1, 1)
            elif radius.dim() == 2:
                radius = radius.unsqueeze(-1)
            
            # 计算并 detach (不需要梯度)
            mask = (dist_matrix <= radius).float().detach()
            td["coverage_mask"] = mask

        # --- 2. 计算潜在增益 (Potential Gain) ---
        # 找出尚未被覆盖的需求
        coverage_mask = td["coverage_mask"]
        uncovered_mask = ~td["is_covered"]
        valuable_demand = td["demand_weights"] * uncovered_mask.float()
        
        # 矩阵乘法计算每个 Facility 能覆盖的剩余权重和
        # [B, N_demand, 1] * [B, N_demand, N_facility] -> sum -> [B, N_facility]
        potential_gain = (valuable_demand.unsqueeze(-1) * coverage_mask).sum(dim=1)

        # --- 3. 提取最大收益特征 (保留此特征以增强判断力) ---
        max_gain, _ = potential_gain.max(dim=1)
        total_demand = td["demand_weights"].sum(dim=-1)
        max_gain_ratio = max_gain / (total_demand + 1e-8)

        # --- 4. Softmax 加权聚合 (你要求的核心部分) ---
        # 使用 Softmax 将 potential_gain 转化为概率分布
        # 收益越高的节点，其 Embedding 在最终 Context 中的占比越大
        potential_gain_norm = torch.softmax(potential_gain + 1e-8, dim=-1)
        
        # [B, N_fac, D] * [B, N_fac, 1] -> sum -> [B, D]
        weighted_embedding = (embeddings * potential_gain_norm.unsqueeze(-1)).sum(dim=1)

        # --- 5. 全局状态特征 ---
        num_select = td["num_facilities_to_select"]
        if num_select.dim() == 2: 
            num_select = num_select.squeeze(-1)
        
        step_progress = td["i"].float() / num_select.float()
        covered_fraction = td["covered_demand"].sum(dim=-1) / (total_demand + 1e-8)

        # --- 6. 拼接并投影 ---
        context_input = torch.cat([
            weighted_embedding,
            step_progress.unsqueeze(-1),
            covered_fraction.unsqueeze(-1),
            max_gain_ratio.unsqueeze(-1)
        ], dim=-1)

        return self.project_context(context_input)


class STPContext(EnvContext):
    """Context embedding for Edge-Based Steiner Tree Problem.
    
    Derives 'visited' status from 'selected_edges' to identify:
    1. Nodes already in the tree (Source of growth).
    2. Terminals NOT yet in the tree (Targets).
    """

    def __init__(self, embed_dim: int):
        super(STPContext, self).__init__(embed_dim=embed_dim)
        self.embed_dim = embed_dim
        
        # 输入维度: 
        # weighted_embedding [D] + step_progress [1] + unvisited_terminal_ratio [1]
        self.project_context = nn.Linear(embed_dim + 2, embed_dim, bias=True)

    def forward(self, embeddings, td):
        """
        embeddings: [batch_size, num_nodes, embed_dim]
        td keys utilized:
            - terminals: [batch_size, num_terminals]
            - selected_edges: [batch_size, num_nodes, num_nodes] (Adjacency of partial solution)
            - i: Current step
        """
        batch_size, num_nodes, _ = embeddings.shape
        device = embeddings.device

        # --- 1. Lazy Init: Terminal Mask ---
        # 这一步不变，先把哪些点是 Terminal 标记出来
        if "is_terminal" not in td.keys():
            is_terminal = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=device)
            is_terminal.scatter_(1, td["terminals"].long(), True)
            td["is_terminal"] = is_terminal # 缓存到 td
            
        is_terminal = td["is_terminal"]

        # --- 2. 关键修改: 从 selected_edges 推导 Visited Mask ---
        # 逻辑：如果一个节点连接了至少一条被选中的边，它就是"visited" (在树中)
        # selected_edges 是 [B, N, N] 的布尔矩阵
        selected_adj = td["selected_edges"]
        
        # 假设矩阵是无向的(对称的)，或者只需检查出度/入度之一即可
        # 只要有一条边连着它，它就是树的一部分
        # shape: [batch_size, num_nodes]
        visited = selected_adj.any(dim=-1) 
        
        # 特殊处理 Step 0:
        # 如果 i=0，没有任何边被选，visited 全为 False。
        # 在这种情况下，我们可能希望模型关注所有 Terminal。
        
        # --- 3. 识别当前目标 (Unvisited Terminals) ---
        # 既是 Terminal，又还没有连入树中的点
        unvisited_terminals = is_terminal & (~visited)
        
        # --- 4. 构建动态权重 (Dynamic Weighting) ---
        node_weights = torch.ones(batch_size, num_nodes, device=device)
        
        # 策略：
        # - 没连上的 Terminal: 权重极高 (3.0) -> "快来连我"
        # - 已经在树里的点 (Visited): 权重较低 (0.5) -> "我是树的根基，可以从我这里延伸"
        # - 没连上的普通点: 权重中等 (1.0) -> "我可以做桥梁"
        
        node_weights[visited] = 0.5 
        node_weights[~is_terminal] = 1.0
        node_weights[unvisited_terminals] = 3.0 

        # --- 5. Softmax 加权聚合 ---
        node_weights_norm = torch.softmax(node_weights, dim=-1) # [B, N]
        weighted_embeddings = (embeddings * node_weights_norm.unsqueeze(-1)).sum(dim=1)

        # --- 6. 全局标量特征 ---
        # 进度：当前选了多少条边 (相对于节点数)
        # 假设完全连通大约需要 N-1 条边 (最小生成树)
        step_progress = td["i"].float() / num_nodes 
        
        # 剩余任务比例：还有多少 terminal 没连上
        num_terminals = td["terminals"].size(1)
        num_unvisited_term = unvisited_terminals.sum(dim=1).float()
        unvisited_ratio = num_unvisited_term / (num_terminals + 1e-8)

        # --- 7. 拼接 ---
        context_input = torch.cat([
            weighted_embeddings,
            step_progress.unsqueeze(-1),
            unvisited_ratio.unsqueeze(-1)
        ], dim=-1)
        
        return self.project_context(context_input)


class GraphContext(EnvContext):
    """Context embedding for graph problems (MIS, MVC, MCL, MCUT).
    
    Simple context that tracks the number of selected nodes and available nodes.
    """
    
    def __init__(self, embed_dim: int):
        # step_context_dim = embed_dim (current node) + 2 (state features)
        super(GraphContext, self).__init__(embed_dim, step_context_dim=embed_dim + 2)
        # We use 2D state features: [num_selected, num_available]
        
    def _state_embedding(self, embeddings: torch.Tensor, td: TensorDict) -> torch.Tensor:
        # Selected nodes count
        num_selected = td["selected"].float().sum(dim=-1)
        
        # Available nodes count
        num_available = td["available"].float().sum(dim=-1)
        
        # Concatenate: [batch, 2]
        context_features = torch.stack([num_selected, num_available], dim=-1)
        
        return context_features
