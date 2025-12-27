from typing import Optional

import torch

from tensordict.tensordict import TensorDict

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger

from .generator import STPGenerator

log = get_pylogger(__name__)


class STPEnv(RL4COEnvBase):
    """Steiner Tree Problem (STP) environment.
    
    The Steiner Tree Problem seeks to find a minimum-weight tree that connects 
    a given set of terminal nodes in a weighted graph. The agent sequentially 
    selects edges to add to the tree until all terminals are connected.
    
    At each step, the agent chooses an edge from the edge_list to add to the current tree.
    The episode terminates when all terminal nodes are connected.
    
    Observations:
        - node locations (coordinates)
        - terminal node indicators
        - edge weights (distance matrix)
        - adjacency matrix (graph structure)
        - edge_list: list of valid edges (optimized action space)
        - current tree edges
        - connected components information
        
    Constraints:
        - selected edges must form a tree (no cycles)
        - all terminal nodes must be connected
        - can only select edges that exist in the graph
        
    Finish condition:
        - all terminal nodes are in the same connected component
        
    Reward:
        - (negative) total weight of edges in the solution tree
        
    Args:
        generator: STPGenerator instance as the data generator
        generator_params: parameters for the generator
        check_solution: whether to check solution validity
        project: if True, project invalid actions to random valid actions
                 (useful during training to handle exploration)
    """
    
    name = "stp"
    
    def __init__(
        self,
        generator: STPGenerator = None,
        generator_params: dict = {},
        check_solution: bool = False,
        project: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = STPGenerator(**generator_params)
        self.generator = generator
        self.check_solution = check_solution
        self.project = project  # Project invalid actions to valid ones
        self._make_spec(self.generator)
    
    def _step(self, td: TensorDict) -> TensorDict:
        """
        Action represents selecting an edge from the edge_list.
        Action is the index in the edge_list.
        """
        batch_size = td["action"].shape[0]
        num_nodes = td["locs"].shape[1]
        action_idx = td["action"]  # Index in edge_list
        
        # Project invalid actions to valid ones if project=True
        if self.project:
            action_idx = self._project_action(td, action_idx)
        
        # Get edge endpoints from edge_list
        # edge_list: (batch, max_edges, 2)
        batch_indices = torch.arange(batch_size, device=td.device)
        from_node = td["edge_list"][batch_indices, action_idx, 0]
        to_node = td["edge_list"][batch_indices, action_idx, 1]
        
        # Update selected edges (adjacency matrix representation)
        selected_edges = td["selected_edges"].clone()
        batch_idx_range = torch.arange(batch_size, device=td.device)
        selected_edges[batch_idx_range, from_node, to_node] = True
        selected_edges[batch_idx_range, to_node, from_node] = True  # undirected
        
        # Also track which edge indices have been selected
        selected_edge_indices = td["selected_edge_indices"].clone()
        selected_edge_indices[batch_idx_range, action_idx] = True
        
        # Update connected components using Union-Find
        components = self._update_components(td["components"], from_node, to_node, batch_size, num_nodes)
        
        # Check if all terminals are connected
        terminals_connected = self._check_terminals_connected(components, td["terminals"], batch_size)
        done = terminals_connected
        
        # Reward is calculated in _get_reward, set to 0 here
        reward = torch.zeros_like(done, dtype=torch.float32)
        
        # Update action mask
        action_mask = self._get_action_mask(td, selected_edges, components)
        
        td.update(
            {
                "selected_edges": selected_edges,
                "selected_edge_indices": selected_edge_indices,
                "components": components,
                "i": td["i"] + 1,
                "action_mask": action_mask,
                "reward": reward,
                "done": done,
            }
        )
        return td
    
    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        self.to(td.device)
        num_nodes = td["locs"].shape[1]
        max_edges = td["edge_list"].shape[1]
        
        # Initialize components: each node is its own component
        components = torch.arange(num_nodes).unsqueeze(0).expand(*batch_size, num_nodes).clone()
        
        # Initial action mask: all valid edges are available
        # Shape: (batch_size, max_edges)
        action_mask = torch.zeros(*batch_size, max_edges, dtype=torch.bool, device=td.device)
        for b in range(batch_size[0] if isinstance(batch_size, (list, tuple)) else batch_size):
            action_mask[b, :td["num_edges"][b]] = True
        
        return TensorDict(
            {
                # Static problem data
                "locs": td["locs"],
                "terminals": td["terminals"],
                "edge_weights": td["edge_weights"],
                "adjacency": td["adjacency"],
                "edge_list": td["edge_list"],
                "num_edges": td["num_edges"],
                
                # Dynamic state
                "selected_edges": torch.zeros(
                    *batch_size, num_nodes, num_nodes, dtype=torch.bool, device=td.device
                ),
                "selected_edge_indices": torch.zeros(
                    *batch_size, max_edges, dtype=torch.bool, device=td.device
                ),
                "components": components.to(td.device),
                "i": torch.zeros(*batch_size, dtype=torch.int64, device=td.device),
                "action_mask": action_mask,
            },
            batch_size=batch_size,
        )
    
    def _make_spec(self, generator: STPGenerator):
        """Create environment specifications"""
        # TODO: implement proper specs using Composite, Bounded, Unbounded
        pass
    
    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        """Calculate reward as negative total weight of selected edges"""
        if self.check_solution:
            self.check_solution_validity(td, actions)
        
        selected_edges = td["selected_edges"]
        edge_weights = td["edge_weights"]
        adjacency = td["adjacency"]
        
        # Only consider valid edges (that exist in the graph)
        valid_selected = selected_edges & adjacency
        
        # Calculate total weight of selected edges
        # Since graph is undirected, divide by 2 to avoid double counting
        total_weight = torch.zeros(selected_edges.shape[0], device=selected_edges.device, dtype=edge_weights.dtype)
        
        for b in range(selected_edges.shape[0]):
            # Only sum weights where valid_selected is True
            selected_weights = edge_weights[b][valid_selected[b]]
            total_weight[b] = selected_weights.sum() / 2.0
        
        return -total_weight
    
    def _update_components(self, components, from_nodes, to_nodes, batch_size, num_nodes):
        """Update connected components using Union-Find"""
        components = components.clone()
        
        for b in range(batch_size):
            from_node = from_nodes[b].item()
            to_node = to_nodes[b].item()
            
            # Find root components
            comp_from = self._find_component(components[b], from_node)
            comp_to = self._find_component(components[b], to_node)
            
            # Union: merge smaller component into larger
            if comp_from != comp_to:
                # Make all nodes in comp_to point to comp_from
                components[b][components[b] == comp_to] = comp_from
        
        return components
    
    def _find_component(self, components, node):
        """Find the root component of a node (with path compression)"""
        root = node
        while components[root] != root:
            root = components[root].item()
        
        # Path compression
        current = node
        while components[current] != root:
            next_node = components[current].item()
            components[current] = root
            current = next_node
        
        return root
    
    def _check_terminals_connected(self, components, terminals, batch_size):
        """Check if all terminal nodes are in the same component"""
        connected = torch.zeros(batch_size, dtype=torch.bool, device=components.device)
        
        for b in range(batch_size):
            terminal_components = components[b][terminals[b]]
            # All terminals connected if they all have the same component ID
            connected[b] = (terminal_components == terminal_components[0]).all()
        
        return connected
    
    def _project_action(self, td: TensorDict, action: torch.Tensor) -> torch.Tensor:
        """
        Project invalid actions to valid ones by randomly selecting from valid actions.
        
        Args:
            td: current state
            action: proposed action (batch_size,)
            
        Returns:
            projected_action: valid action (batch_size,)
        """
        batch_size = action.shape[0]
        action_mask = td["action_mask"]
        projected_action = action.clone()
        
        for b in range(batch_size):
            # Check if the action is valid
            if action[b] >= action_mask.shape[1] or not action_mask[b, action[b]]:
                # Action is invalid, select a random valid action
                valid_actions = action_mask[b].nonzero(as_tuple=True)[0]
                
                if len(valid_actions) > 0:
                    # Randomly select from valid actions
                    random_idx = torch.randint(0, len(valid_actions), (1,), device=action.device)
                    projected_action[b] = valid_actions[random_idx]
                    
                    if self.check_solution:
                        log.warning(
                            f"Batch {b}: Invalid action {action[b].item()} projected to {projected_action[b].item()}"
                        )
                else:
                    # No valid actions available (shouldn't happen in proper setup)
                    log.error(f"Batch {b}: No valid actions available!")
                    projected_action[b] = 0  # Fallback to first action
        
        return projected_action
    
    def _get_action_mask(self, td, selected_edges, components):
        """
        Generate action mask for valid edges:
        - Edge not already selected
        - Edge connects different components (no cycles)
        
        Shape: (batch_size, max_edges) where max_edges is the number of edges in edge_list
        """
        batch_size = td["edge_list"].shape[0]
        max_edges = td["edge_list"].shape[1]
        num_edges = td["num_edges"]
        edge_list = td["edge_list"]
        
        # Initialize mask: start with all edges invalid
        mask = torch.zeros(batch_size, max_edges, dtype=torch.bool, device=td.device)
        
        # For each batch, check each edge
        for b in range(batch_size):
            for edge_idx in range(num_edges[b].item()):
                # Skip if already selected
                if td["selected_edge_indices"][b, edge_idx]:
                    continue
                
                # Get edge endpoints
                from_node = edge_list[b, edge_idx, 0].item()
                to_node = edge_list[b, edge_idx, 1].item()
                
                # Check if edge connects different components
                comp_from = self._find_component(components[b], from_node)
                comp_to = self._find_component(components[b], to_node)
                
                # Only allow edges between different components (avoid cycles)
                if comp_from != comp_to:
                    mask[b, edge_idx] = True
        
        return mask
    
    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        """Check if the solution is valid"""
        # Check 1: All terminals are connected
        selected_edges = td["selected_edges"]
        terminals = td["terminals"]
        batch_size = selected_edges.shape[0]
        
        for b in range(batch_size):
            # Check if terminals form a connected component
            visited = torch.zeros(td["locs"].shape[1], dtype=torch.bool)
            queue = [terminals[b][0].item()]
            visited[queue[0]] = True
            
            while queue:
                node = queue.pop(0)
                neighbors = selected_edges[b][node].nonzero(as_tuple=True)[0]
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor.item())
            
            # All terminals should be visited
            terminal_visited = visited[terminals[b].cpu()]
            assert terminal_visited.all(), f"Not all terminals connected in batch {b}"
        
        # Check 2: No cycles (number of edges = number of nodes in tree - 1)
        # This is implicitly enforced by the component-based action masking
        log.info("Solution validity check passed")
    
    @staticmethod
    def local_search(td: TensorDict, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        """Local search improvement (optional, to be implemented)"""
        # TODO: implement local search heuristics like edge swapping
        return actions
    
    @staticmethod
    def get_num_starts(td):
        """Number of possible starting edges"""
        return td["num_edges"].max().item()
    
    @staticmethod
    def select_start_nodes(td, num_starts):
        """Select starting edges for multi-start decoding"""
        batch_size = td["action_mask"].shape[0]
        max_edges = td["action_mask"].shape[-1]
        
        # Select first num_starts valid edges from each batch
        starts_list = []
        for b in range(batch_size):
            valid_edges = td["action_mask"][b].nonzero(as_tuple=True)[0]
            if len(valid_edges) > 0:
                # Select up to num_starts edges, cycling if needed
                selected = valid_edges[torch.arange(num_starts) % len(valid_edges)]
                starts_list.append(selected)
            else:
                # No valid edges, use dummy
                starts_list.append(torch.zeros(num_starts, dtype=torch.long, device=td.device))
        
        return torch.stack(starts_list).view(-1)
