"""
MCTS Model wrapper for RL4CO framework.
This module provides a model wrapper that integrates MCTS with the RL4CO framework.
"""

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.utils.pylogger import get_pylogger

from .MCTS import MCTS

log = get_pylogger(__name__)


class MCTSModel(nn.Module):
    """MCTS Model wrapper for RL4CO.
    
    This model combines MCTS with neural networks for combinatorial optimization.
    Supports three modes:
    1. Pure MCTS: No neural networks (random rollout)
    2. Policy-guided MCTS: Policy network provides priors P(s,a)
    3. AlphaGo Zero style: Policy network + Value network
    
    Args:
        env: RL4CO environment or environment name
        policy_net: Optional policy network for prior probabilities
        value_net: Optional value network for state evaluation
        policy: Optional combined policy (backward compatibility)
        num_simulations: Number of MCTS simulations per action
        c_puct: Exploration constant for UCB formula
        temperature: Temperature for action selection (0 = greedy)
        device: Device to run on
        
    Example:
        >>> from rl4co.envs import TSPEnv
        >>> from rl4co.models.zoo import AttentionModelPolicy
        >>> 
        >>> # Create environment and policy
        >>> env = TSPEnv(generator_params={'num_loc': 20})
        >>> policy = AttentionModelPolicy(env_name='tsp')
        >>> 
        >>> # Create MCTS model
        >>> mcts_model = MCTSModel(
        >>>     env=env,
        >>>     policy=policy,
        >>>     num_simulations=50,
        >>>     c_puct=1.0,
        >>> )
        >>> 
        >>> # Generate problem and solve
        >>> td = env.reset(batch_size=[1])
        >>> actions, reward, stats = mcts_model.solve(td)
    """
    
    def __init__(
        self,
        env: RL4COEnvBase | str,
        policy_net: nn.Module = None,
        value_net: nn.Module = None,
        policy: nn.Module = None,  # Backward compatibility
        num_simulations: int = 100,
        c_puct: float = 1.0,
        temperature: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__()
        
        # Setup environment
        if isinstance(env, str):
            self.env = get_env(env)
        else:
            self.env = env
        
        # Setup networks
        self.policy_net = policy_net
        self.value_net = value_net
        
        # Backward compatibility: if policy is provided, use it
        if policy is not None:
            self.policy = policy
            if policy_net is None:
                self.policy_net = policy
            if value_net is None:
                self.value_net = policy
        else:
            self.policy = None
        
        # Set networks to eval mode
        if self.policy_net is not None:
            self.policy_net.eval()
        if self.value_net is not None and self.value_net is not self.policy_net:
            self.value_net.eval()
        
        # Log configuration
        if self.policy_net is not None and self.value_net is not None:
            if self.policy_net is self.value_net:
                log.info("MCTS initialized with combined policy+value network")
            else:
                log.info("MCTS initialized with separate policy and value networks")
        elif self.policy_net is not None:
            log.info("MCTS initialized with policy network only")
        elif self.value_net is not None:
            log.info("MCTS initialized with value network only")
        else:
            log.info("MCTS initialized without neural networks (pure MCTS)")
        
        # MCTS parameters
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.device = device
        
    def forward(
        self,
        td: TensorDict,
        phase: str = "test",
        return_actions: bool = True,
        return_stats: bool = False,
        **kwargs,
    ) -> dict:
        """Forward pass using MCTS.
        
        Args:
            td: Input TensorDict with problem instances
            phase: Phase (train/val/test) - currently only supports test
            return_actions: Whether to return actions
            return_stats: Whether to return MCTS search statistics
            
        Returns:
            Dictionary with 'reward', 'actions' (optional), and 'stats' (optional)
        """
        batch_size = td.batch_size[0] if len(td.batch_size) > 0 else 1
        
        if batch_size > 1:
            log.warning(
                f"MCTS currently processes one instance at a time. "
                f"Batch size {batch_size} will be processed sequentially."
            )
        
        all_actions = []
        all_rewards = []
        all_stats = []
        
        # Process each instance
        for i in range(batch_size):
            # Get single instance, keep batch dimension
            if batch_size > 1:
                td_single = td[i:i+1]
            else:
                td_single = td
            
            # Create MCTS solver for this instance
            mcts = MCTS(
                env=self.env,
                policy_net=self.policy_net,
                value_net=self.value_net,
                num_simulations=self.num_simulations,
                c_puct=self.c_puct,
                temperature=self.temperature,
                device=self.device,
            )
            
            # Solve
            actions, reward, stats = mcts.solve(td_single)
            
            all_actions.append(actions)
            all_rewards.append(reward)
            all_stats.append(stats)
        
        # Stack results
        actions_stacked = torch.cat(all_actions, dim=0) if batch_size > 1 else all_actions[0]
        rewards_stacked = torch.cat(all_rewards, dim=0) if batch_size > 1 else all_rewards[0]
        
        # Build output dictionary
        out = {"reward": rewards_stacked}
        
        if return_actions:
            out["actions"] = actions_stacked
        
        if return_stats:
            out["stats"] = all_stats
        
        return out
    
    def solve(
        self,
        td: TensorDict,
        return_stats: bool = True,
        verbose: bool = True,
    ) -> tuple:
        """Solve problem using MCTS.
        
        Args:
            td: Input TensorDict with problem instance
            return_stats: Whether to return search statistics
            verbose: Whether to print progress information
            
        Returns:
            Tuple of (actions, reward, stats) if return_stats=True
            Tuple of (actions, reward) otherwise
        """
        mcts = MCTS(
            env=self.env,
            policy_net=self.policy_net,
            value_net=self.value_net,
            num_simulations=self.num_simulations,
            c_puct=self.c_puct,
            temperature=self.temperature,
            device=self.device,
        )
        
        actions, reward, stats = mcts.solve(td, verbose=verbose)
        
        if return_stats:
            return actions, reward, stats
        else:
            return actions, reward
    
    @torch.no_grad()
    def evaluate(
        self,
        td: TensorDict,
        num_instances: int = None,
    ) -> dict:
        """Evaluate MCTS on multiple problem instances.
        
        Args:
            td: Input TensorDict with problem instances
            num_instances: Number of instances to evaluate (None = all)
            
        Returns:
            Dictionary with evaluation results
        """
        self.eval()
        
        batch_size = td.batch_size[0] if len(td.batch_size) > 0 else 1
        if num_instances is not None:
            batch_size = min(batch_size, num_instances)
        
        rewards = []
        tour_lengths = []
        
        log.info(f"Evaluating MCTS on {batch_size} instances...")
        
        for i in range(batch_size):
            # Keep batch dimension when indexing
            if td.batch_size:
                td_single = td[i:i+1]
            else:
                td_single = td
            
            out = self.forward(td_single, return_actions=True, return_stats=False)
            
            rewards.append(out['reward'].item())
            # For TSP-like problems, reward is negative tour length
            tour_lengths.append(-out['reward'].item())
        
        results = {
            'mean_reward': sum(rewards) / len(rewards),
            'mean_tour_length': sum(tour_lengths) / len(tour_lengths),
            'min_tour_length': min(tour_lengths),
            'max_tour_length': max(tour_lengths),
            'num_instances': batch_size,
        }
        
        log.info(
            f"MCTS Evaluation Results:\n"
            f"  Mean Reward: {results['mean_reward']:.4f}\n"
            f"  Mean Tour Length: {results['mean_tour_length']:.4f}\n"
            f"  Min/Max Tour: {results['min_tour_length']:.4f}/{results['max_tour_length']:.4f}"
        )
        
        return results
    
    def set_num_simulations(self, num_simulations: int):
        """Update number of MCTS simulations."""
        self.num_simulations = num_simulations
        log.info(f"Updated MCTS simulations to {num_simulations}")
    
    def set_temperature(self, temperature: float):
        """Update temperature for action selection."""
        self.temperature = temperature
        log.info(f"Updated MCTS temperature to {temperature}")
    
    def set_c_puct(self, c_puct: float):
        """Update exploration constant."""
        self.c_puct = c_puct
        log.info(f"Updated MCTS c_puct to {c_puct}")
