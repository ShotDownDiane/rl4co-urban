import torch
import torch.nn as nn

from tensordict.tensordict import TensorDict

from rl4co.models.nn.ops import PositionalEncoding
from rl4co.utils.ops import batched_scatter_sum, cartesian_to_polar


def env_init_embedding(env_name: str, config: dict) -> nn.Module:
    """Get environment initial embedding. The init embedding is used to initialize the
    general embedding of the problem nodes without any solution information.
    Consists of a linear layer that projects the node features to the embedding space.

    Args:
        env: Environment or its name.
        config: A dictionary of configuration options for the environment.
    """
    embedding_registry = {
        "tsp": TSPInitEmbedding,
        "atsp": TSPInitEmbedding,
        "matnet": MatNetInitEmbedding,
        "cvrp": VRPInitEmbedding,
        "cvrptw": VRPTWInitEmbedding,
        "cvrpmvc": VRPInitEmbedding,
        "svrp": SVRPInitEmbedding,
        "sdvrp": VRPInitEmbedding,
        "pctsp": PCTSPInitEmbedding,
        "spctsp": PCTSPInitEmbedding,
        "op": OPInitEmbedding,
        "dpp": DPPInitEmbedding,
        "mdpp": MDPPInitEmbedding,
        "pdp": PDPInitEmbedding,
        "pdp_ruin_repair": TSPInitEmbedding,
        "tsp_kopt": TSPInitEmbedding,
        "mtsp": MTSPInitEmbedding,
        "smtwtp": SMTWTPInitEmbedding,
        "mdcpdp": MDCPDPInitEmbedding,
        "fjsp": FJSPInitEmbedding,
        "jssp": FJSPInitEmbedding,
        "mtvrp": MTVRPInitEmbedding,
        "shpp": TSPInitEmbedding,
        "flp": FLPInitEmbedding,
        "mcp": MCPInitEmbedding,
        "mclp": MCLPInitEmbedding,
        "stp": STPInitEmbedding,
        # Graph problems (ML4CO wrappers)
        "mis": GraphInitEmbedding,
        "mvc": GraphInitEmbedding,
        "mcl": GraphInitEmbedding,
        "mcut": GraphInitEmbedding,
    }

    if env_name not in embedding_registry:
        raise ValueError(
            f"Unknown environment name '{env_name}'. Available init embeddings: {embedding_registry.keys()}"
        )

    return embedding_registry[env_name](**config)


class TSPInitEmbedding(nn.Module):
    """Initial embedding for the Traveling Salesman Problems (TSP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the cities
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(TSPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)

    def forward(self, td):
        out = self.init_embed(td["locs"])
        return out


class MatNetInitEmbedding(nn.Module):
    """
    Preparing the initial row and column embeddings for MatNet.

    Reference:
    https://github.com/yd-kwon/MatNet/blob/782698b60979effe2e7b61283cca155b7cdb727f/ATSP/ATSP_MatNet/ATSPModel.py#L51


    """

    def __init__(self, embed_dim: int, mode: str = "RandomOneHot") -> None:
        super().__init__()

        self.embed_dim = embed_dim
        assert mode in {
            "RandomOneHot",
            "Random",
        }, "mode must be one of ['RandomOneHot', 'Random']"
        self.mode = mode

    def forward(self, td: TensorDict):
        dmat = td["cost_matrix"]
        b, r, c = dmat.shape

        row_emb = torch.zeros(b, r, self.embed_dim, device=dmat.device)

        if self.mode == "RandomOneHot":
            # MatNet uses one-hot encoding for column embeddings
            # https://github.com/yd-kwon/MatNet/blob/782698b60979effe2e7b61283cca155b7cdb727f/ATSP/ATSP_MatNet/ATSPModel.py#L60
            col_emb = torch.zeros(b, c, self.embed_dim, device=dmat.device)
            rand = torch.rand(b, c)
            rand_idx = rand.argsort(dim=1)
            b_idx = torch.arange(b)[:, None].expand(b, c)
            n_idx = torch.arange(c)[None, :].expand(b, c)
            col_emb[b_idx, n_idx, rand_idx] = 1.0

        elif self.mode == "Random":
            col_emb = torch.rand(b, c, self.embed_dim, device=dmat.device)
        else:
            raise NotImplementedError

        return row_emb, col_emb, dmat


class VRPInitEmbedding(nn.Module):
    """Initial embedding for the Vehicle Routing Problems (VRP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot and customers separately)
        - demand: demand of the customers
    """

    def __init__(self, embed_dim, linear_bias=True, node_dim: int = 3):
        super(VRPInitEmbedding, self).__init__()
        node_dim = node_dim  # 3: x, y, demand
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)  # depot embedding

    def forward(self, td):
        # [batch, 1, 2]-> [batch, 1, embed_dim]
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        depot_embedding = self.init_embed_depot(depot)
        # [batch, n_city, 2, batch, n_city, 1]  -> [batch, n_city, embed_dim]
        node_embeddings = self.init_embed(
            torch.cat((cities, td["demand"][..., None]), -1)
        )
        # [batch, n_city+1, embed_dim]
        out = torch.cat((depot_embedding, node_embeddings), -2)
        return out


class VRPTWInitEmbedding(VRPInitEmbedding):
    def __init__(self, embed_dim, linear_bias=True, node_dim: int = 6):
        # node_dim = 6: x, y, demand, tw start, tw end, service time
        super(VRPTWInitEmbedding, self).__init__(embed_dim, linear_bias, node_dim)

    def forward(self, td):
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        durations = td["durations"][..., 1:]
        time_windows = td["time_windows"][..., 1:, :]
        # embeddings
        depot_embedding = self.init_embed_depot(depot)
        node_embeddings = self.init_embed(
            torch.cat(
                (cities, td["demand"][..., None], time_windows, durations[..., None]), -1
            )
        )
        return torch.cat((depot_embedding, node_embeddings), -2)


class VRPPolarInitEmbedding(nn.Module):
    """Initial embedding for the Vehicle Routing Problems (VRP).
    Embed the following node features to the embedding space, based on polar coordinates:
        - locs: r, theta coordinates of the nodes, with the depot as the origin
        - demand: demand of the customers
    """

    def __init__(
        self,
        embed_dim,
        linear_bias=True,
        node_dim: int = 3,
        attach_cartesian_coords=False,
    ):
        super(VRPPolarInitEmbedding, self).__init__()
        self.node_dim = node_dim + (
            2 if attach_cartesian_coords else 0
        )  # 3: r, theta, demand; 5: r, theta, demand, x, y;
        self.attach_cartesian_coords = attach_cartesian_coords
        self.init_embed = nn.Linear(self.node_dim, embed_dim, linear_bias)
        self.init_embed_depot = nn.Linear(
            self.node_dim, embed_dim, linear_bias
        )  # depot embedding

    def forward(self, td):
        with torch.no_grad():
            locs = td["locs"]
            polar_locs = cartesian_to_polar(locs, locs[..., 0:1, :])
            td["polar_locs"] = polar_locs

            demand = td["demand"]
            demand_with_depot = torch.concat(
                (torch.zeros(demand.shape[0], 1, device=demand.device), demand),
                dim=-1,
            ).unsqueeze(-1)

            if self.attach_cartesian_coords:
                x = torch.concat((polar_locs, demand_with_depot, locs), dim=-1)
            else:
                x = torch.concat((polar_locs, demand_with_depot), dim=-1)

        depot, cities = x[:, :1, :], x[:, 1:, :]
        depot_embedding = self.init_embed_depot(depot)
        node_embeddings = self.init_embed(cities)

        out = torch.cat((depot_embedding, node_embeddings), -2)
        return out


class SVRPInitEmbedding(nn.Module):
    def __init__(self, embed_dim, linear_bias=True, node_dim: int = 3):
        super(SVRPInitEmbedding, self).__init__()
        node_dim = node_dim  # 3: x, y, skill
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)  # depot embedding

    def forward(self, td):
        # [batch, 1, 2]-> [batch, 1, embed_dim]
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        depot_embedding = self.init_embed_depot(depot)
        # [batch, n_city, 2, batch, n_city, 1]  -> [batch, n_city, embed_dim]
        node_embeddings = self.init_embed(torch.cat((cities, td["skills"]), -1))
        # [batch, n_city+1, embed_dim]
        out = torch.cat((depot_embedding, node_embeddings), -2)
        return out


class PCTSPInitEmbedding(nn.Module):
    """Initial embedding for the Prize Collecting Traveling Salesman Problems (PCTSP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot and customers separately)
        - expected_prize: expected prize for visiting the customers.
            In PCTSP, this is the actual prize. In SPCTSP, this is the expected prize.
        - penalty: penalty for not visiting the customers
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(PCTSPInitEmbedding, self).__init__()
        node_dim = 4  # x, y, prize, penalty
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)

    def forward(self, td):
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        depot_embedding = self.init_embed_depot(depot)
        node_embeddings = self.init_embed(
            torch.cat(
                (
                    cities,
                    td["expected_prize"][..., None],
                    td["penalty"][..., 1:, None],
                ),
                -1,
            )
        )
        # batch, n_city+1, embed_dim
        out = torch.cat((depot_embedding, node_embeddings), -2)
        return out


class OPInitEmbedding(nn.Module):
    """Initial embedding for the Orienteering Problems (OP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot and customers separately)
        - prize: prize for visiting the customers
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(OPInitEmbedding, self).__init__()
        node_dim = 3  # x, y, prize
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)  # depot embedding

    def forward(self, td):
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        depot_embedding = self.init_embed_depot(depot)
        node_embeddings = self.init_embed(
            torch.cat(
                (
                    cities,
                    td["prize"][..., 1:, None],  # exclude depot
                ),
                -1,
            )
        )
        out = torch.cat((depot_embedding, node_embeddings), -2)
        return out


class DPPInitEmbedding(nn.Module):
    """Initial embedding for the Decap Placement Problem (DPP), EDA (electronic design automation).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (cells)
        - probe: index of the (single) probe cell. We embed the euclidean distance from the probe to all cells.
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(DPPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed = nn.Linear(node_dim, embed_dim // 2, linear_bias)  # locs
        self.init_embed_probe = nn.Linear(1, embed_dim // 2, linear_bias)  # probe

    def forward(self, td):
        node_embeddings = self.init_embed(td["locs"])
        probe_embedding = self.init_embed_probe(
            self._distance_probe(td["locs"], td["probe"])
        )
        return torch.cat([node_embeddings, probe_embedding], -1)

    def _distance_probe(self, locs, probe):
        # Euclidean distance from probe to all locations
        probe_loc = torch.gather(locs, 1, probe.unsqueeze(-1).expand(-1, -1, 2))
        return torch.norm(locs - probe_loc, dim=-1).unsqueeze(-1)


class MDPPInitEmbedding(nn.Module):
    """Initial embedding for the Multi-port Placement Problem (MDPP), EDA (electronic design automation).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (cells)
        - probe: indexes of the probe cells (multiple). We embed the euclidean distance of each cell to the closest probe.
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(MDPPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)  # locs
        self.init_embed_probe_distance = nn.Linear(
            1, embed_dim, linear_bias
        )  # probe_distance
        self.project_out = nn.Linear(embed_dim * 2, embed_dim, linear_bias)

    def forward(self, td):
        probes = td["probe"]
        locs = td["locs"]
        node_embeddings = self.init_embed(locs)

        # Get the shortest distance from any probe
        dist = torch.cdist(locs, locs, p=2)
        dist[~probes] = float("inf")
        min_dist, _ = torch.min(dist, dim=1)
        min_probe_dist_embedding = self.init_embed_probe_distance(min_dist[..., None])

        return self.project_out(
            torch.cat([node_embeddings, min_probe_dist_embedding], -1)
        )


class PDPInitEmbedding(nn.Module):
    """Initial embedding for the Pickup and Delivery Problem (PDP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot, pickups and deliveries separately)
           Note that pickups and deliveries are interleaved in the input.
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(PDPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)
        self.init_embed_pick = nn.Linear(node_dim * 2, embed_dim, linear_bias)
        self.init_embed_delivery = nn.Linear(node_dim, embed_dim, linear_bias)

    def forward(self, td):
        depot, locs = td["locs"][..., 0:1, :], td["locs"][..., 1:, :]
        num_locs = locs.size(-2)
        pick_feats = torch.cat(
            [locs[:, : num_locs // 2, :], locs[:, num_locs // 2 :, :]], -1
        )  # [batch_size, graph_size//2, 4]
        delivery_feats = locs[:, num_locs // 2 :, :]  # [batch_size, graph_size//2, 2]
        depot_embeddings = self.init_embed_depot(depot)
        pick_embeddings = self.init_embed_pick(pick_feats)
        delivery_embeddings = self.init_embed_delivery(delivery_feats)
        # concatenate on graph size dimension
        return torch.cat([depot_embeddings, pick_embeddings, delivery_embeddings], -2)


class MTSPInitEmbedding(nn.Module):
    """Initial embedding for the Multiple Traveling Salesman Problem (mTSP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot, cities)
    """

    def __init__(self, embed_dim, linear_bias=True):
        """NOTE: new made by Fede. May need to be checked"""
        super(MTSPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)  # depot embedding

    def forward(self, td):
        depot_embedding = self.init_embed_depot(td["locs"][..., 0:1, :])
        node_embedding = self.init_embed(td["locs"][..., 1:, :])
        return torch.cat([depot_embedding, node_embedding], -2)


class SMTWTPInitEmbedding(nn.Module):
    """Initial embedding for the Single Machine Total Weighted Tardiness Problem (SMTWTP).
    Embed the following node features to the embedding space:
        - job_due_time: due time of the jobs
        - job_weight: weights of the jobs
        - job_process_time: the processing time of jobs
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(SMTWTPInitEmbedding, self).__init__()
        node_dim = 3  # job_due_time, job_weight, job_process_time
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)

    def forward(self, td):
        job_due_time = td["job_due_time"]
        job_weight = td["job_weight"]
        job_process_time = td["job_process_time"]
        feat = torch.stack((job_due_time, job_weight, job_process_time), dim=-1)
        out = self.init_embed(feat)
        return out


class MDCPDPInitEmbedding(nn.Module):
    """Initial embedding for the MDCPDP environment
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot, pickups and deliveries separately)
           Note that pickups and deliveries are interleaved in the input.
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(MDCPDPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)
        self.init_embed_pick = nn.Linear(node_dim * 2, embed_dim, linear_bias)
        self.init_embed_delivery = nn.Linear(node_dim, embed_dim, linear_bias)

    def forward(self, td):
        num_agents = td["capacity"].size(-1)
        depot, locs = td["locs"][..., 0:num_agents, :], td["locs"][..., num_agents:, :]
        num_locs = locs.size(-2)
        pick_feats = torch.cat(
            [locs[:, : num_locs // 2, :], locs[:, num_locs // 2 :, :]], -1
        )  # [batch_size, graph_size//2, 4]
        delivery_feats = locs[:, num_locs // 2 :, :]  # [batch_size, graph_size//2, 2]
        depot_embeddings = self.init_embed_depot(depot)
        pick_embeddings = self.init_embed_pick(pick_feats)
        delivery_embeddings = self.init_embed_delivery(delivery_feats)
        # concatenate on graph size dimension
        return torch.cat([depot_embeddings, pick_embeddings, delivery_embeddings], -2)


class JSSPInitEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim,
        linear_bias: bool = True,
        scaling_factor: int = 1000,
        num_op_feats=5,
    ):
        super(JSSPInitEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.scaling_factor = scaling_factor
        self.init_ops_embed = nn.Linear(num_op_feats, embed_dim, linear_bias)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=0.0)

    def _op_features(self, td):
        proc_times = td["proc_times"]
        mean_durations = proc_times.sum(1) / (proc_times.gt(0).sum(1) + 1e-9)
        feats = [
            mean_durations / self.scaling_factor,
            # td["lbs"] / self.scaling_factor,
            td["is_ready"],
            td["num_eligible"],
            td["ops_job_map"],
            td["op_scheduled"],
        ]
        return torch.stack(feats, dim=-1)

    def _init_ops_embed(self, td: TensorDict):
        ops_feat = self._op_features(td)
        ops_emb = self.init_ops_embed(ops_feat)
        ops_emb = self.pos_encoder(ops_emb, td["ops_sequence_order"])

        # zero out padded and finished ops
        mask = td["pad_mask"]  # NOTE dont mask scheduled - leads to instable training
        ops_emb[mask.unsqueeze(-1).expand_as(ops_emb)] = 0
        return ops_emb

    def forward(self, td):
        return self._init_ops_embed(td)


class FJSPInitEmbedding(JSSPInitEmbedding):
    def __init__(self, embed_dim, linear_bias=False, scaling_factor: int = 100):
        super().__init__(embed_dim, linear_bias, scaling_factor)
        self.init_ma_embed = nn.Linear(1, self.embed_dim, bias=linear_bias)
        self.edge_embed = nn.Linear(1, embed_dim, bias=linear_bias)

    def forward(self, td: TensorDict):
        ops_emb = self._init_ops_embed(td)
        ma_emb = self._init_machine_embed(td)
        edge_emb = self._init_edge_embed(td)
        # get edges between operations and machines
        # (bs, ops, ma)
        edges = td["ops_ma_adj"].transpose(1, 2)
        return ops_emb, ma_emb, edge_emb, edges

    def _init_edge_embed(self, td: TensorDict):
        proc_times = td["proc_times"].transpose(1, 2) / self.scaling_factor
        edge_embed = self.edge_embed(proc_times.unsqueeze(-1))
        return edge_embed

    def _init_machine_embed(self, td: TensorDict):
        busy_for = (td["busy_until"] - td["time"].unsqueeze(1)) / self.scaling_factor
        ma_embeddings = self.init_ma_embed(busy_for.unsqueeze(2))
        return ma_embeddings


class FJSPMatNetInitEmbedding(JSSPInitEmbedding):
    def __init__(
        self,
        embed_dim,
        linear_bias: bool = False,
        scaling_factor: int = 1000,
    ):
        super().__init__(embed_dim, linear_bias, scaling_factor)
        self.init_ma_embed = nn.Linear(1, self.embed_dim, bias=linear_bias)

    def _init_machine_embed(self, td: TensorDict):
        busy_for = (td["busy_until"] - td["time"].unsqueeze(1)) / self.scaling_factor
        ma_embeddings = self.init_ma_embed(busy_for.unsqueeze(2))
        return ma_embeddings

    def forward(self, td: TensorDict):
        proc_times = td["proc_times"]
        ops_emb = self._init_ops_embed(td)
        # encoding machines
        ma_emb = self._init_machine_embed(td)
        # edgeweights for matnet
        matnet_edge_weights = proc_times.transpose(1, 2) / self.scaling_factor
        return ops_emb, ma_emb, matnet_edge_weights


class MTVRPInitEmbedding(VRPInitEmbedding):
    def __init__(self, embed_dim, linear_bias=True, node_dim: int = 7):
        # node_dim = 7: x, y, demand_linehaul, demand_backhaul, tw start, tw end, service time
        super(MTVRPInitEmbedding, self).__init__(embed_dim, linear_bias, node_dim)

    def forward(self, td):
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        demand_linehaul, demand_backhaul = (
            td["demand_linehaul"][..., 1:],
            td["demand_backhaul"][..., 1:],
        )
        service_time = td["service_time"][..., 1:]
        time_windows = td["time_windows"][..., 1:, :]
        # [!] convert [0, inf] -> [0, 0] if a problem does not include the time window constraint, do not modify in-place
        time_windows = torch.nan_to_num(time_windows, posinf=0.0)
        # embeddings
        depot_embedding = self.init_embed_depot(depot)
        node_embeddings = self.init_embed(
            torch.cat(
                (
                    cities,
                    demand_linehaul[..., None],
                    demand_backhaul[..., None],
                    time_windows,
                    service_time[..., None],
                ),
                -1,
            )
        )
        return torch.cat((depot_embedding, node_embeddings), -2)


class FLPInitEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.projection = nn.Linear(2, embed_dim, bias=True)

    def forward(self, td: TensorDict):
        hdim = self.projection(td["locs"])
        return hdim


class MCPInitEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.projection_items = nn.Linear(1, embed_dim, bias=True)

    def forward(self, td: TensorDict):
        items_embed = self.projection_items(td["weights"].unsqueeze(-1))
        # sum pooling
        membership_emb = batched_scatter_sum(items_embed, td["membership"].long())
        return membership_emb

class MCLPInitEmbedding(nn.Module):
    """Initial embedding for MCLP based on Demand Aggregation.
    
    Philosophy (MCP-like):
    A facility is defined by the demand nodes it covers.
    Embedding = (Projected Facility Coords) + (Sum of Covered Demand Embeddings)
    """

    def __init__(self, embed_dim: int, linear_bias=True):
        super(MCLPInitEmbedding, self).__init__()
        self.embed_dim = embed_dim
        
        # 1. 需求点特征映射 (Demand Projection)
        # 将需求点的权重 (scalar) 映射为向量
        self.project_demand = nn.Linear(1, embed_dim, bias=linear_bias)
        
        # 2. 设施自身位置映射 (Facility Projection)
        # 保留这个是为了区分空间位置不同但覆盖相同的极端情况
        self.project_facility = nn.Linear(2, embed_dim, bias=linear_bias)

    def forward(self, td: TensorDict):
        # --- A. 准备数据 ---
        # 需求权重: [batch_size, num_demand]
        demand_weights = td["demand_weights"]
        
        # 设施坐标: [batch_size, num_facility, 2]
        facility_locs = td["facility_locs"]
        
        # 覆盖掩码 (Lazy Init 逻辑，同 Context)
        # 如果前面 Context 算过了，这里直接拿；没算过这里算。
        # [batch_size, num_demand, num_facility]
        if "coverage_mask" not in td.keys():
            dist = td["distance_matrix"]
            rad = td["coverage_radius"]
            if rad.dim() == 1: rad = rad.view(-1, 1, 1)
            elif rad.dim() == 2: rad = rad.unsqueeze(-1)
            td["coverage_mask"] = (dist <= rad).float().detach()
        
        coverage_mask = td["coverage_mask"]

        # --- B. 需求侧 Embedding (Item Embedding) ---
        # [B, N_d] -> [B, N_d, 1] -> [B, N_d, embed_dim]
        demand_emb = self.project_demand(demand_weights.unsqueeze(-1))

        # --- C. 聚合逻辑 (Aggregation / Pooling) ---
        # 类似于 MCP 的 batched_scatter_sum，但这里是多对多关系，用矩阵乘法更高效
        # 公式: Facility_Emb = Sum_{d \in Covered} (Demand_Emb_d)
        
        # Transpose mask: [B, N_facility, N_demand]
        mask_t = coverage_mask.transpose(1, 2)
        
        # MatMul: [B, N_f, N_d] @ [B, N_d, Dim] -> [B, N_f, Dim]
        # 这一步就把"该设施覆盖的所有需求点的特征"加总到了设施上
        aggregated_demand_emb = torch.matmul(mask_t, demand_emb)

        # --- D. 设施侧 Embedding (Location Embedding) ---
        # [B, N_f, 2] -> [B, N_f, Dim]
        facility_loc_emb = self.project_facility(facility_locs)

        # --- E. 融合 (Fusion) ---
        # 简单的相加 (ResNet style)，融合"我在哪"和"我覆盖谁"的信息
        final_embedding = facility_loc_emb + aggregated_demand_emb
        
        return final_embedding


class STPInitEmbedding(nn.Module):
    """Memory-Efficient Graph-Aware Embedding for STP.
    
    Architecture:
    1. Node Stream: Loc + Terminal ID -> [B, N, D]
    2. Edge Stream: Edge Cost -> Learned Scalar Weight -> [B, N, N]
    3. Aggregation: Weighted GCN (Matrix Multiplication)
    
    Memory Complexity: O(N^2) instead of O(N^2 * D)
    """

    def __init__(self, embed_dim: int, linear_bias=True):
        super(STPInitEmbedding, self).__init__()
        self.embed_dim = embed_dim
        
        # --- 1. 节点特征处理 ---
        self.project_loc = nn.Linear(2, embed_dim, bias=linear_bias)
        self.embed_type = nn.Embedding(2, embed_dim) # 0: Steiner, 1: Terminal
        
        # --- 2. 边权重变换 (关键优化) ---
        # 这是一个微型 MLP，用于将"成本(Cost)"转化为"亲和度(Affinity)"
        # 输入 1 维 (权重)，输出 1 维 (聚合系数)
        self.edge_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1), 
            nn.Sigmoid() # 输出归一化到 (0, 1)，类似于门控
        )
        
        # --- 3. 融合层 ---
        # 输入是: 自身特征 [D] + 邻居聚合特征 [D]
        self.proj_fusion = nn.Linear(2 * embed_dim, embed_dim, bias=linear_bias)

    def forward(self, td: TensorDict):
        # td["locs"]: [B, N, 2]
        batch_size, num_nodes, _ = td["locs"].shape
        device = td["locs"].device
        
        # ==========================================
        # Part A: 构建基础节点特征 (Node Features)
        # ==========================================
        
        # 1. 生成 Terminal Mask (Lazy Init + Scatter)
        if "is_terminal" not in td.keys():
            is_terminal = torch.zeros(batch_size, num_nodes, dtype=torch.long, device=device)
            is_terminal.scatter_(1, td["terminals"].long(), 1) # Index -> 1
            td["is_terminal"] = is_terminal
        
        # 2. 融合几何与语义 (Add)
        # [B, N, D]
        h_geo = self.project_loc(td["locs"])
        h_type = self.embed_type(td["is_terminal"])
        h_nodes = h_geo + h_type 
        
        # ==========================================
        # Part B: 学习型图聚合 (Learned Graph Aggregation)
        # ==========================================
        
        # 1. 准备边数据
        # edge_weights: [B, N, N] (Costs/Distances)
        # adjacency: [B, N, N] (0/1 Mask, indicating edge existence)
        edge_costs = td["edge_weights"]
        adj_mask = td["adjacency"]
        
        # 2. 计算动态聚合系数 (Learnable Coefficients)
        # 为什么要学？因为 STP 中 EdgeWeight 是 Cost。
        # 直接用 Cost 做聚合权重是错的（越远权重应该越小）。
        # 让网络学习一个函数 f(cost) -> affinity
        
        # [B, N, N, 1]
        cost_input = edge_costs.unsqueeze(-1) 
        
        # [B, N, N, 1] -> [B, N, N]
        # 使用 detach() 仅仅是为了安全，如果 edge_weights 需要梯度则去掉
        # 但通常 STP 的 edge_weights 是固定输入
        affinity_scores = self.edge_gate(cost_input).squeeze(-1)
        
        # 3. 应用邻接掩码
        # 只保留真实存在的边。
        # 这一步产生稀疏权重的 Dense 表示
        weighted_adj = affinity_scores * adj_mask.float()
        
        # 4. 图卷积聚合 (Matrix Multiplication)
        # [B, N, N] @ [B, N, D] -> [B, N, D]
        # 含义：每个节点收集所有邻居的特征，且根据边的"好坏"加权
        h_struct = torch.matmul(weighted_adj, h_nodes)
        
        # ==========================================
        # Part C: 最终融合 (Fusion)
        # ==========================================
        
        # 拼接 自身信息 [D] 和 结构环境信息 [D]
        # [B, N, 2D]
        h_combined = torch.cat([h_nodes, h_struct], dim=-1)
        
        # 投影回 [B, N, D]
        h_final = self.proj_fusion(h_combined)
        
        return h_final


class GraphInitEmbedding(nn.Module):
    """Initial embedding for graph problems (MIS, MVC, MCL, MCUT).
    
    Embeds node weights to the embedding space. For unweighted graphs,
    uses uniform weights.
    """
    
    def __init__(self, embed_dim: int):
        super().__init__()
        # Project node weight (1D) to embedding space
        self.projection = nn.Linear(1, embed_dim, bias=True)
    
    def forward(self, td: TensorDict):
        # td["nodes_weight"]: [batch, num_nodes]
        # Unsqueeze to [batch, num_nodes, 1] for projection
        node_weights = td["nodes_weight"].unsqueeze(-1)
        
        # Project to embedding space: [batch, num_nodes, embed_dim]
        h = self.projection(node_weights)
        
        return h
