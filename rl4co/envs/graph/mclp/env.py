from typing import Optional

import torch
from tensordict.tensordict import TensorDict

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.pylogger import get_pylogger

from .generator import MCLPGenerator

log = get_pylogger(__name__)


class MCLPEnv(RL4COEnvBase):
    """Maximum Covering Location Problem (MCLP) environment.
    
    At each step, the agent chooses a facility location. The reward is 0 unless 
    enough number of facilities are chosen. The reward is the total demand covered
    by the selected facilities within the coverage radius.
    
    Observations:
        - demand point locations and weights
        - candidate facility locations
        - coverage radius
        - distance matrix between demands and facilities
        - current covered demand
        - facility selection status
    
    Constraints:
        - the given number of facilities must be chosen
        - a facility can only be chosen once
    
    Finish condition:
        - the given number of facilities are chosen
    
    Reward:
        - the total weight of demand points covered by selected facilities
        - a demand point is covered if it is within the coverage radius of at least one selected facility
    
    Args:
        generator: MCLPGenerator instance as the data generator
        generator_params: parameters for the generator
        check_solution: whether to check solution validity
    """
    
    name = "mclp"
    
    def __init__(
        self,
        generator: MCLPGenerator = None,
        generator_params: dict = {},
        check_solution: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = MCLPGenerator(**generator_params)
        self.generator = generator
        self.check_solution = check_solution
    
    def _step(self, td: TensorDict) -> TensorDict:
        """Execute one step: select a facility and update coverage"""
        batch_size = td["action"].shape[0]
        selected = td["action"]  # [batch_size]
        
        # Update facility selection status
        chosen = td["chosen"].clone()
        chosen[torch.arange(batch_size, device=td.device), selected] = True
        
        # Update covered demand
        # For each demand point, check if it's within coverage radius of any selected facility
        distance_matrix = td["distance_matrix"]  # [batch_size, num_demand, num_facility]
        coverage_radius = td["coverage_radius"]  # [batch_size, 1]
        demand_weights = td["demand_weights"]  # [batch_size, num_demand]
        
        # Check which demands are within coverage radius of selected facilities
        # [batch_size, num_demand, num_facility] <= [batch_size, 1, 1]
        within_radius = distance_matrix <= coverage_radius.unsqueeze(-1)
        
        # A demand is covered if it's within radius of ANY selected facility
        # chosen: [batch_size, num_facility] -> [batch_size, 1, num_facility]
        # within_radius: [batch_size, num_demand, num_facility]
        is_covered = (within_radius & chosen.unsqueeze(1)).any(dim=-1)  # [batch_size, num_demand]
        
        # Calculate covered demand (weight of covered demand points)
        covered_demand = demand_weights * is_covered.float()
        
        # Check if done
        done = td["i"] >= (td["num_facilities_to_select"] - 1)
        
        # Reward is calculated in _get_reward for efficiency
        reward = torch.zeros(batch_size, dtype=torch.float32, device=td.device)
        
        # Update action mask: cannot choose already selected facilities
        action_mask = ~chosen
        
        td.update(
            {
                "chosen": chosen,
                "covered_demand": covered_demand,
                "is_covered": is_covered,  # Keep track of which demands are covered
                "i": td["i"] + 1,
                "action_mask": action_mask,
                "reward": reward,
                "done": done,
            }
        )
        return td
    
    def _reset(
        self,
        td: Optional[TensorDict] = None,
        batch_size: Optional[list] = None,
    ) -> TensorDict:
        """Reset the environment"""
        if batch_size is None:
            batch_size = self.batch_size if td is None else td.batch_size
        if td is None or td.is_empty():
            td = self.generator(batch_size=batch_size)
        
        device = td.device
        
        return TensorDict(
            {
                # Problem data (constant)
                "demand_locs": td["demand_locs"],
                "facility_locs": td["facility_locs"],
                "demand_weights": td["demand_weights"],
                "coverage_radius": td["coverage_radius"],
                "distance_matrix": td["distance_matrix"],
                "num_facilities_to_select": td["num_facilities_to_select"],
                # Dynamic state
                "chosen": torch.zeros(
                    *batch_size,
                    self.generator.num_facility,
                    dtype=torch.bool,
                    device=device
                ),
                "covered_demand": torch.zeros(
                    *batch_size,
                    self.generator.num_demand,
                    dtype=torch.float32,
                    device=device
                ),
                "is_covered": torch.zeros(
                    *batch_size,
                    self.generator.num_demand,
                    dtype=torch.bool,
                    device=device
                ),
                "i": torch.zeros(*batch_size, dtype=torch.int64, device=device),
                "action_mask": torch.ones(
                    *batch_size,
                    self.generator.num_facility,
                    dtype=torch.bool,
                    device=device
                ),
            },
            batch_size=batch_size,
        )
    
    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        """Calculate reward as total covered demand weight"""
        if self.check_solution:
            self.check_solution_validity(td, actions)
        
        # The reward is the sum of covered demand weights
        # covered_demand already contains weights of covered demands
        total_covered = td["covered_demand"].sum(dim=-1)
        
        return total_covered
    
    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        """Check if the solution is valid"""
        batch_size = actions.shape[0]
        
        for b in range(batch_size):
            # Check 1: correct number of facilities selected
            num_selected = td["chosen"][b].sum().item()
            expected = td["num_facilities_to_select"][b].item()
            assert num_selected == expected, \
                f"Batch {b}: Expected {expected} facilities, got {num_selected}"
            
            # Check 2: no duplicate selections
            unique_actions = torch.unique(actions[b])
            assert len(unique_actions) == len(actions[b]), \
                f"Batch {b}: Duplicate facility selections detected"
            
        log.info("Solution validity check passed")
