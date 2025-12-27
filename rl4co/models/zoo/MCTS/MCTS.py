"""
Monte Carlo Tree Search (MCTS) implementation for RL4CO.
This module provides a simple MCTS that can be combined with neural network policies.
"""

import math
from typing import Dict, Optional, Tuple

import torch
from tensordict import TensorDict

from rl4co.envs import RL4COEnvBase
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class MCTSNode:
    """Node in the MCTS tree.
    
    Args:
        state: TensorDict representing the environment state
        parent: Parent node
        action: Action taken from parent to reach this node
        prior: Prior probability from neural network
    """
    
    def __init__(
        self,
        state: TensorDict,
        parent: Optional['MCTSNode'] = None,
        action: Optional[torch.Tensor] = None,
        prior: float = 1.0,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        
        # MCTS statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, 'MCTSNode'] = {}
        self.is_expanded = False
        
    @property
    def value(self) -> float:
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def expand(
        self,
        action_probs: torch.Tensor,
        action_mask: torch.Tensor,
        env: RL4COEnvBase,
    ) -> None:
        """Expand node by creating children for all valid actions.
        
        Args:
            action_probs: Action probabilities from policy [num_actions]
            action_mask: Valid action mask [num_actions]
            env: Environment for stepping
        """
        # Handle 2D mask [batch_size, num_actions] - get indices from first batch
        if action_mask.dim() == 2:
            valid_actions = torch.where(action_mask[0])[0]
        else:
            valid_actions = torch.where(action_mask)[0]
        
        for action_idx in valid_actions:
            action = action_idx.item()
            
            # Create child state by stepping environment
            child_td = self.state.clone()
            # Action should match batch dimension of state
            child_td['action'] = torch.tensor([action], device=self.state.device, dtype=torch.long)
            child_td = env.step(child_td)['next']
            
            # Create child node with prior probability
            prior = action_probs[action_idx].item()
            self.children[action] = MCTSNode(
                state=child_td,
                parent=self,
                action=action,
                prior=prior,
            )
        
        self.is_expanded = True
    
    def select_child(self, c_puct: float = 1.0) -> Tuple[int, 'MCTSNode']:
        """Select child with highest UCB score.
        
        Args:
            c_puct: Exploration constant
            
        Returns:
            Tuple of (action, child_node)
        """
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in self.children.items():
            # UCB formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            q_value = child.value
            u_value = (
                c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
            )
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def update(self, value: float) -> None:
        """Update node statistics with backpropagated value.
        
        Args:
            value: Value to backpropagate
        """
        self.visit_count += 1
        self.value_sum += value
    
    def backpropagate(self, value: float) -> None:
        """Backpropagate value up the tree.
        
        Args:
            value: Value to backpropagate
        """
        node = self
        while node is not None:
            node.update(value)
            node = node.parent


class MCTS:
    """Monte Carlo Tree Search.
    
    This implementation supports two architectures:
    1. Pure MCTS: No neural networks (random rollout)
    2. AlphaGo Zero style: Policy network + Value network
    
    Args:
        env: RL4CO environment
        policy_net: Optional policy network for prior probabilities P(s,a)
        value_net: Optional value network for state evaluation V(s)
        policy: Optional combined policy (backward compatibility, deprecated)
        num_simulations: Number of MCTS simulations per action
        c_puct: Exploration constant for UCB
        temperature: Temperature for action selection
        device: Device to run on
    """
    
    def __init__(
        self,
        env: RL4COEnvBase,
        policy_net=None,
        value_net=None,
        policy=None,  # Backward compatibility
        num_simulations: int = 100,
        c_puct: float = 1.0,
        temperature: float = 1.0,
        device: str = "cpu",
    ):
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net
        
        # Backward compatibility: if policy is provided, use it as both
        if policy is not None and policy_net is None:
            self.policy_net = policy
        if policy is not None and value_net is None:
            self.value_net = policy
            
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.device = device
        
    def search(self, td: TensorDict, verbose: bool = False) -> Tuple[torch.Tensor, MCTSNode]:
        """Run MCTS from given state.
        
        Args:
            td: Initial state TensorDict
            verbose: Whether to print simulation progress
            
        Returns:
            Tuple of (best_action, root_node)
        """
        # Create root node
        root = MCTSNode(state=td.clone())
        
        # Run simulations
        for i in range(self.num_simulations):
            self._simulate(root)
            # Print progress every 25% for first step only
            if verbose and i > 0 and i % max(1, self.num_simulations // 4) == 0:
                progress = (i / self.num_simulations) * 100
                log.info(f"      Simulation progress: {progress:.0f}% ({i}/{self.num_simulations})")
                
        # Select best action based on visit counts
        action = self._select_action(root)
        
        return action, root
    
    def _simulate(self, node: MCTSNode) -> float:
        """Run one MCTS simulation from node.
        
        Args:
            node: Node to simulate from
            
        Returns:
            Value of the simulation
        """
        # Check if terminal
        if node.state['done'].item():
            # Return reward for terminal state
            return self._get_value(node)
        
        # Expand node if not expanded
        if not node.is_expanded:
            action_probs, value = self._evaluate(node.state)
            action_mask = node.state['action_mask']
            node.expand(action_probs, action_mask, self.env)
            # CRITICAL: Don't backpropagate immediately!
            # Let the parent node's update() do it, so the tree can grow deeper
            return value
        
        # Select and recurse
        action, child = node.select_child(self.c_puct)
        value = self._simulate(child)
        node.update(value)
        
        return value
    
    def _evaluate(self, td: TensorDict) -> Tuple[torch.Tensor, float]:
        """Evaluate state to get action probabilities and value.
        
        Args:
            td: State to evaluate
            
        Returns:
            Tuple of (action_probabilities, value)
        """
        mask = td['action_mask']
        if mask.dim() == 2:
            mask_1d = mask[0]
        else:
            mask_1d = mask
        
        # Get prior probabilities from policy_net
        # Note: Full policy network integration requires proper state caching
        # For now, we use uniform priors even with policy_net
        # TODO: Implement efficient policy network forward pass with caching
        probs = mask_1d.float() / mask_1d.float().sum()
        
        if self.policy_net is not None:
            # Policy network is available but we use uniform prior for simplicity
            # In a full implementation, you would:
            # 1. Cache encoder output for each state
            # 2. Use decoder to get one-step action probabilities
            # 3. Handle the state embedding properly
            pass
        
        # Get value estimate from value_net
        if self.value_net is not None:
            with torch.no_grad():
                try:
                    # Use value network for quick evaluation
                    value = self._get_value_from_network(td)
                except Exception as e:
                    log.warning(f"Value network failed, using rollout: {e}")
                    value = self._rollout_random(td)
        else:
            # No value network: use random rollout
            value = self._rollout_random(td)
        
        return probs, value
    
    def _get_value_from_network(self, td: TensorDict) -> float:
        """Get value estimate from value network.
        
        Args:
            td: State to evaluate
            
        Returns:
            Value estimate
        """
        # Greedy rollout with value network as heuristic
        # For simplicity, use random rollout for now
        # TODO: Implement proper value network forward pass
        return self._rollout_random(td)
    
    def _rollout_random(self, td: TensorDict) -> float:
        """Rollout with random policy to estimate value.
        
        Args:
            td: State to rollout from
            
        Returns:
            Estimated value (heuristic based on remaining nodes)
        """
        td_rollout = td.clone()
        actions = []
        step = 0
        max_steps = 100
        
        while not td_rollout['done'].item() and step < max_steps:
            # Sample random valid action
            mask = td_rollout['action_mask']
            # Handle 2D mask [batch_size, num_actions] - get indices of True values in first batch
            if mask.dim() == 2:
                valid_actions = torch.where(mask[0])[0]
            else:
                valid_actions = torch.where(mask)[0]
            
            if len(valid_actions) == 0:
                break  # No valid actions, must be done
            
            # Random sample from valid actions
            action_idx = torch.randint(0, len(valid_actions), (1,)).item()
            action = valid_actions[action_idx].item()
            
            # Set action with correct batch dimension
            td_rollout['action'] = torch.tensor([action], device=td_rollout.device, dtype=torch.long)
            td_rollout = self.env.step(td_rollout)['next']
            actions.append(action)
            step += 1
        
        # Get reward from final state
        # Use heuristic: negative of remaining nodes (simple approximation)
        if td_rollout['done'].item():
            mask = td_rollout['action_mask']
            if mask.dim() == 2:
                remaining = mask[0].sum().item()
            else:
                remaining = mask.sum().item()
            
            # Simple heuristic: penalize by remaining nodes
            if remaining == 0:
                # Tour complete - approximate reward
                return -len(actions)
            else:
                # Tour incomplete - large penalty
                return -1000.0
        else:
            # Didn't finish in max steps
            return -1000.0
    
    def _get_value(self, node: MCTSNode) -> float:
        """Get value for terminal node.
        
        Args:
            node: Terminal node
            
        Returns:
            Reward value
        """
        # Get reward from state (negative tour length for TSP-like problems)
        if 'reward' in node.state:
            return node.state['reward'].item()
        return 0.0
    
    def _select_action(self, root: MCTSNode) -> torch.Tensor:
        """Select action from root based on visit counts.
        
        Args:
            root: Root node
            
        Returns:
            Selected action
        """
        # Handle case with only one child
        if len(root.children) == 1:
            action = list(root.children.keys())[0]
        elif self.temperature == 0:
            # Greedy: select most visited
            visit_counts = {a: c.visit_count for a, c in root.children.items()}
            action = max(visit_counts, key=visit_counts.get)
        else:
            # Sample proportional to visit_count^(1/temperature)
            actions = list(root.children.keys())
            visits = torch.tensor(
                [root.children[a].visit_count for a in actions],
                dtype=torch.float32,
            )
            # Add small epsilon to avoid 0^x issues
            visits = visits + 1e-8
            probs = torch.pow(visits, 1.0 / self.temperature)
            probs = probs / probs.sum()
            
            action = actions[torch.multinomial(probs, 1).item()]
        
        # Return action with batch dimension
        return torch.tensor([action], device=root.state.device, dtype=torch.long)
    
    def solve(
        self,
        td: TensorDict,
        max_steps: int = 1000,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """Solve problem using MCTS.
        
        Args:
            td: Initial problem state
            max_steps: Maximum number of steps
            verbose: Whether to print progress
            
        Returns:
            Tuple of (actions, reward, search_stats)
        """
        actions = []
        search_stats = []
        td_current = td.clone()
        
        if verbose:
            log.info(f"Starting MCTS with {self.num_simulations} simulations per step")
            print(f"Starting MCTS with {self.num_simulations} simulations per step")
        step = 0
        while not td_current['done'].item() and step < max_steps:
            # Run MCTS search
            if verbose and step % 5 == 0:
                log.info(f"  Step {step}: Running MCTS search...")
            
            action, root = self.search(td_current)
            
            # Store stats
            visit_counts = {a: c.visit_count for a, c in root.children.items()}
            best_action = max(visit_counts, key=visit_counts.get)
            
            # Get action value for logging and storage
            action_value = action[0].item() if action.dim() > 0 else action.item()
            
            if verbose and step < 3:  # Show details for first 3 steps
                log.info(f"    Selected action {action_value}, visits={visit_counts.get(action_value, 0)}, value={root.value:.4f}")
            
            search_stats.append({
                'step': step,
                'visit_counts': visit_counts,
                'root_value': root.value,
            })
            
            # Take action
            actions.append(action_value)
            td_current['action'] = action  # action shape is [1]
            td_current = self.env.step(td_current)['next']
            
            step += 1
        
        # Get final reward
        actions_tensor = torch.tensor(actions, device=td.device).unsqueeze(0)
        reward = self.env.get_reward(td_current, actions_tensor)
        
        if verbose:
            log.info(f"✓ MCTS completed in {step} steps, tour length: {-reward.item():.4f}")
        
        return actions_tensor, reward, search_stats
