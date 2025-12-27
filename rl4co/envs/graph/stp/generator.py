from typing import Callable, Union

import torch
import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class STPGenerator(Generator):
    """Efficient Data generator for the Steiner Tree Problem (STP).
    
    Optimized to avoid NetworkX overhead and use vectorized PyTorch/Scipy operations where possible.
    
    Args:
        num_nodes: total number of nodes in the graph
        num_terminals: number of terminal nodes that must be connected
        min_loc: minimum value for the node coordinates
        max_loc: maximum value for the node coordinates
        loc_distribution: distribution for the node coordinates
        graph_type: type of graph generation method
            - "delaunay": Delaunay triangulation (planar and connected)
            - "knn": k-nearest neighbors + MST backbone
            - "radius": radius graph + MST backbone
            - "complete": complete graph (all edges exist)
        knn_k: number of nearest neighbors (for knn graph type)
        radius: connection radius (for radius graph type)
    """
    
    def __init__(
        self,
        num_nodes: int = 50,
        num_terminals: int = 10,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[int, float, str, type, Callable] = Uniform,
        graph_type: str = "delaunay",
        knn_k: int = 3,
        radius: float = 0.2,
        **kwargs,
    ):
        self.num_nodes = num_nodes
        self.num_terminals = num_terminals
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.graph_type = graph_type
        self.knn_k = knn_k
        self.radius = radius
        
        assert num_terminals <= num_nodes, "Number of terminals must be <= total nodes"
        assert graph_type in ["delaunay", "knn", "radius", "complete"], \
            f"graph_type must be one of ['delaunay', 'knn', 'radius', 'complete'], got {graph_type}"
        
        if kwargs.get("loc_sampler", None) is not None:
            self.loc_sampler = kwargs["loc_sampler"]
        else:
            self.loc_sampler = get_sampler(
                "loc", loc_distribution, min_loc, max_loc, **kwargs
            )

    def _generate(self, batch_size) -> TensorDict:
        # 1. Generate Locations (Batch)
        locs = self.loc_sampler.sample((*batch_size, self.num_nodes, 2))
        batch_size_val = locs.shape[0]
        device = locs.device

        # 2. Compute Distance Matrix (Batch)
        # dists: [B, N, N]
        dists = torch.cdist(locs, locs, p=2)
        
        # Initialize Adjacency Mask [B, N, N]
        adj_mask = torch.zeros((batch_size_val, self.num_nodes, self.num_nodes), dtype=torch.bool, device=device)

        # 3. Generate Graph Connectivity based on type
        if self.graph_type == "complete":
            # Set all to True (excluding diagonal naturally, or explicitly set diagonal to False)
            adj_mask = torch.ones_like(adj_mask, dtype=torch.bool)
            adj_mask.diagonal(dim1=-2, dim2=-1).fill_(False)

        elif self.graph_type == "radius":
            # Vectorized Radius Graph
            adj_mask = (dists <= self.radius)
            # Remove self-loops
            adj_mask.diagonal(dim1=-2, dim2=-1).fill_(False)
            # Add MST backbone to ensure connectivity
            self._add_mst_backbone(adj_mask, dists)

        elif self.graph_type == "knn":
            # Vectorized KNN
            # topk returns smallest distances. k+1 because the node itself (dist 0) is included
            _, indices = dists.topk(self.knn_k + 1, largest=False)
            # Create scatter indices
            batch_indices = torch.arange(batch_size_val, device=device).view(-1, 1, 1).expand(-1, self.num_nodes, self.knn_k + 1)
            source_indices = torch.arange(self.num_nodes, device=device).view(1, -1, 1).expand(batch_size_val, -1, self.knn_k + 1)
            
            # Fill mask
            # Note: KNN is directed by default, we make it undirected (symmetric)
            adj_mask[batch_indices, source_indices, indices] = True
            adj_mask[batch_indices, indices, source_indices] = True
            
            # Remove self-loops (dist 0)
            adj_mask.diagonal(dim1=-2, dim2=-1).fill_(False)
            
            # Add MST backbone
            self._add_mst_backbone(adj_mask, dists)

        elif self.graph_type == "delaunay":
            # Delaunay is hard to vectorize fully due to variable number of edges per instance
            # But we can optimize the extraction without NetworkX
            locs_np = locs.cpu().numpy()
            for b in range(batch_size_val):
                tri = Delaunay(locs_np[b])
                # simplices: [N_tri, 3] -> indices of vertices
                # Extract edges: (0,1), (1,2), (2,0)
                simplices = tri.simplices
                u = np.concatenate([simplices[:, 0], simplices[:, 1], simplices[:, 2]])
                v = np.concatenate([simplices[:, 1], simplices[:, 2], simplices[:, 0]])
                
                # Assign symmetric connections
                adj_mask[b, u, v] = True
                adj_mask[b, v, u] = True

        # 4. Prepare Outputs
        # Edge Weights: just use Euclidean distances
        # No need to set inf for non-existent edges, adjacency mask handles that
        edge_weights = dists.clone()
        
        # Edge List: [B, Max_Edges, 2]
        # We need to extract indices. To do this efficiently for padding:
        # Find max number of edges in the batch
        num_edges_per_instance = adj_mask.sum(dim=(1, 2)) // 2 # Divided by 2 for undirected
        max_edges = num_edges_per_instance.max().item()
        
        # Create padded edge list
        # We use upper triangular to avoid duplicates in undirected graph
        upper_tri_mask = torch.triu(adj_mask, diagonal=1)
        nonzero_indices = upper_tri_mask.nonzero(as_tuple=False) # [Total_Edges, 3] -> (b, i, j)
        
        edge_list_padded = torch.zeros((batch_size_val, max_edges, 2), dtype=torch.long, device=device)
        
        # This part is a bit tricky to vectorize fully for filling padded tensor without loops
        # But loop over batch is fine here if max_edges is not huge, 
        # or we can use advanced indexing if we compute split points
        # For simplicity and readability, we loop to fill the padded structure
        # (This is still faster than creating NetworkX graphs)
        for b in range(batch_size_val):
            mask_b = upper_tri_mask[b]
            edges = mask_b.nonzero(as_tuple=False)
            n = edges.shape[0]
            edge_list_padded[b, :n] = edges

        # 5. Terminals
        # Vectorized random selection without replacement
        # randperm is not batchable directly, so we use argsort of rand
        rand_vals = torch.rand((batch_size_val, self.num_nodes), device=device)
        terminals = rand_vals.argsort(dim=1)[:, :self.num_terminals]

        return TensorDict(
            {
                "locs": locs,
                "terminals": terminals,
                "edge_weights": edge_weights,
                "adjacency": adj_mask,
                "edge_list": edge_list_padded,
                "num_edges": num_edges_per_instance,
            },
            batch_size=batch_size,
        )

    def _add_mst_backbone(self, adj_mask: torch.Tensor, dists: torch.Tensor):
        """Calculates MST for each instance in batch using Scipy and adds edges to adj_mask.
        This ensures the graph is connected.
        """
        # Move to CPU for Scipy
        dists_np = dists.cpu().numpy()
        batch_size = dists.shape[0]
        
        for b in range(batch_size):
            # Compute MST using Scipy (very fast C implementation)
            # dists_np[b] is a dense matrix, csr_matrix conversion is cheap for N=50-100
            csr = csr_matrix(dists_np[b])
            mst = minimum_spanning_tree(csr)
            
            # mst is a sparse matrix (coo/csr). Get row/col indices
            # These are the edges in the MST
            rows, cols = mst.nonzero()
            
            # Add to PyTorch mask (symmetric)
            adj_mask[b, rows, cols] = True
            adj_mask[b, cols, rows] = True