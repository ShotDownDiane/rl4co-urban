from typing import Callable, Union

import torch
import numpy as np
from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class MCLPGenerator(Generator):
    """Data generator for the Maximum Covering Location Problem (MCLP).
    
    Integrated with robust generation logic for OOD generalization (Clusters, Explosion).
    Implements constraints to ensure realistic facility placement:
    1. Facilities are generated near demand points (sampled from demand locations).
    2. Facilities enforce a minimum distance separation to avoid redundancy.
    3. (Optional) Heuristic Seeding: Uses a greedy algorithm to ensure good candidates exist.
    
    Args:
        num_demand: number of demand points
        num_facility: number of candidate facility locations
        num_facilities_to_select: number of facilities to select
        min_demand: minimum value for demand weights
        max_demand: maximum value for demand weights
        coverage_radius: base radius (used if dynamic_radius is False)
        min_loc: minimum value for location coordinates
        max_loc: maximum value for location coordinates
        distribution: data distribution type ('uniform', 'cluster', 'explosion')
        dynamic_radius: whether to calculate radius dynamically based on density
        min_facility_dist: minimum distance between any two candidate facilities (constraint)
        heuristic_seeding: if True, first N candidates are chosen via Greedy MCLP to ensure solvability.
    """
    
    def __init__(
        self,
        num_demand: int = 100,
        num_facility: int = 50,
        num_facilities_to_select: int = 10,
        min_demand: float = 1.0,
        max_demand: float = 10.0,
        coverage_radius: float = 0.2,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        distribution: str = "uniform", # 'uniform', 'cluster', 'explosion'
        dynamic_radius: bool = False,
        min_facility_dist: float = 0.05, # Minimum distance between facilities
        heuristic_seeding: bool = True,  # Enable smart seeding
        loc_distribution: Union[int, float, str, type, Callable] = Uniform,
        **kwargs,
    ):
        self.num_demand = num_demand
        self.num_facility = num_facility
        self.num_facilities_to_select = num_facilities_to_select
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.coverage_radius = coverage_radius
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.distribution = distribution
        self.dynamic_radius = dynamic_radius
        self.min_facility_dist = min_facility_dist
        self.heuristic_seeding = heuristic_seeding
        
        # Samplers for uniform distribution fallback
        self.loc_sampler = get_sampler(
            "loc", loc_distribution, min_loc, max_loc, **kwargs
        )
        
    def _generate(self, batch_size) -> TensorDict:
        # Get batch size value if it's a list/tuple
        batch_size_val = batch_size[0] if isinstance(batch_size, (list, tuple)) else batch_size

        # 1. Generate Demand Locations based on Distribution
        demand_locs = self.loc_sampler.sample((*batch_size, self.num_demand, 2))

        # 2. Generate Demand Weights (Need these early for Greedy Seeding)
        demand_weights = self._generate_weights(demand_locs, batch_size_val)

        # 3. Calculate Coverage Radius (Need this early for Greedy Seeding)
        if self.dynamic_radius:
            # Calculate dynamic radius based on KNN distance of demand points
            coverage_radius = self._calculate_dynamic_radius(demand_locs)
        else:
            # Fixed radius
            coverage_radius = torch.full((batch_size_val, 1), self.coverage_radius)

        # 4. Generate Facility Locations (Constrained & Seeded)
        # Now we pass weights and radius to perform "Heuristic Seeding"
        facility_locs = self._generate_constrained_facilities(
            demand_locs, batch_size_val, demand_weights, coverage_radius
        )

        # 5. Calculate Distance Matrix
        # [batch_size, num_demand, num_facility]
        distance_matrix = torch.cdist(demand_locs, facility_locs, p=2)
        
        # Initialize state variables
        covered_demand = torch.zeros(batch_size_val, self.num_demand, dtype=torch.float32)
        chosen = torch.zeros(batch_size_val, self.num_facility, dtype=torch.bool)
        
        # Number of facilities to select tensor
        num_facilities_to_select = torch.full((batch_size_val, 1), self.num_facilities_to_select)
        
        return TensorDict(
            {
                "demand_locs": demand_locs,
                "facility_locs": facility_locs,
                "demand_weights": demand_weights,
                "coverage_radius": coverage_radius,
                "distance_matrix": distance_matrix,
                "covered_demand": covered_demand,
                "chosen": chosen,
                "num_facilities_to_select": num_facilities_to_select,
            },
            batch_size=batch_size,
        )

    def _generate_constrained_facilities(self, demand_locs, batch_size, weights, radius):
        """
        Generates facility candidates by sampling from demand locations.
        Enforces a minimum distance between selected facilities.
        Can use Greedy MCLP to seed the first few candidates with high-quality locations.
        """
        device = demand_locs.device
        num_demand = demand_locs.shape[1]
        
        # Tensor to store selected facilities [Batch, Num_Facility, 2]
        facility_locs = torch.zeros(batch_size, self.num_facility, 2, device=device)
        
        # Mask to keep track of available demand points that can be turned into facilities
        available_mask = torch.ones(batch_size, num_demand, dtype=torch.bool, device=device)
        
        # Pre-compute distances between all demand pairs
        # This is O(N^2), but usually fine for N <= 2000.
        # For N > 5000, consider chunking or approximate nearest neighbor.
        dists_all = torch.cdist(demand_locs, demand_locs, p=2) # [B, N, N]
        
        # State for Greedy Seeding
        if self.heuristic_seeding:
            # Track which demand points are already covered by the greedy selected facilities
            currently_covered = torch.zeros(batch_size, num_demand, dtype=torch.bool, device=device)
            # Coverage matrix: [B, Demand_i, Candidate_j] -> True if j covers i
            coverage_matrix = dists_all < radius.unsqueeze(2)
            # Convert to float for matmul
            coverage_float = coverage_matrix.float()

        for i in range(self.num_facility):
            # Decide selection strategy: Greedy or Random?
            # We seed the first `num_facilities_to_select` spots with Greedy, rest Random
            use_greedy = self.heuristic_seeding and (i < self.num_facilities_to_select)
            
            if use_greedy:
                # --- Greedy Step Optimized ---
                # Effective weights: weights * (~currently_covered) -> [B, N]
                # We want to find candidate j that maximizes sum(weight_i * coverage_ij)
                # This is a matrix multiplication: Weights x Coverage
                
                eff_weights = weights * (~currently_covered).float()
                
                # Use MatMul instead of broadcast multiply to save memory
                # [B, 1, N] x [B, N, N] -> [B, 1, N]
                gains = torch.matmul(eff_weights.unsqueeze(1), coverage_float).squeeze(1)
                
                # Apply mask (set invalid candidates to -inf)
                gains[~available_mask] = -1.0
                
                # Select best candidate
                selected_indices = torch.argmax(gains, dim=1) # [B]
                
                # Update coverage status
                # new_coverage_col: [B, N_demand]
                new_coverage_col = torch.gather(coverage_matrix, 2, selected_indices.unsqueeze(1).unsqueeze(2).expand(-1, num_demand, 1)).squeeze(2)
                currently_covered = currently_covered | new_coverage_col

            else:
                # --- Random Step (Constrained) ---
                probs = available_mask.float()
                
                # Fallback if no valid points
                invalid_batches = probs.sum(dim=1) == 0
                if invalid_batches.any():
                    probs[invalid_batches] = 1.0 
                
                selected_indices = torch.multinomial(probs, 1).squeeze(1) # [B]

            # Store coordinates
            selected_coords = torch.gather(demand_locs, 1, selected_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, 2)).squeeze(1)
            facility_locs[:, i, :] = selected_coords
            
            # Update available mask (Distance Constraint)
            # Find distances from all points to the newly selected point
            dists_to_new = torch.gather(dists_all, 1, selected_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, num_demand)).squeeze(1)
            
            too_close = dists_to_new < self.min_facility_dist
            available_mask = available_mask & (~too_close)
            
        # --- Apply Jitter & Noise ---
        # 1. For Greedy Seeds: Larger disturbance (0.05) to ensure they are not perfect
        # 2. For Random Candidates: Small jitter (0.005) just to move off the demand point
        
        if self.heuristic_seeding:
            n_seeded = min(self.num_facilities_to_select, self.num_facility)
            
            # Perturb greedy seeds
            jitter_seeded = torch.randn(batch_size, n_seeded, 2, device=device) * 0.05
            facility_locs[:, :n_seeded] += jitter_seeded
            
            # Perturb remaining random candidates
            if self.num_facility > n_seeded:
                jitter_random = torch.randn(batch_size, self.num_facility - n_seeded, 2, device=device) * 0.005
                facility_locs[:, n_seeded:] += jitter_random
        else:
            # If no seeding, apply small jitter to all
            jitter = torch.randn_like(facility_locs) * 0.005
            facility_locs += jitter

        # Clamp to bounds [0, 1]
        facility_locs = torch.clamp(facility_locs, 0.0, 1.0)
        
        # --- Shuffle Candidates ---
        # We must shuffle to prevent the agent from cheating (learning that first N are best)
        shuffle_noise = torch.rand(batch_size, self.num_facility, device=device)
        perm = torch.argsort(shuffle_noise, dim=1)
        
        # Apply permutation to facility_locs: [B, N_fac, 2]
        facility_locs = torch.gather(facility_locs, 1, perm.unsqueeze(2).expand(-1, -1, 2))
        
        return facility_locs

    def _generate_clusters(self, batch_size, num_points):
        """Generates Gaussian Mixture Clusters."""
        locs = []
        for _ in range(batch_size):
            num_clusters = np.random.randint(3, 8) 
            points_per_cluster = num_points // num_clusters
            
            cluster_centers = torch.rand(num_clusters, 2)
            cluster_std = torch.rand(num_clusters) * 0.05 + 0.02 
            
            batch_locs = []
            points_generated = 0
            for i in range(num_clusters):
                # Calculate remaining points correctly
                n_p = points_per_cluster if i < num_clusters - 1 else num_points - points_generated
                center = cluster_centers[i]
                std = cluster_std[i]
                points = torch.randn(n_p, 2) * std + center
                batch_locs.append(points)
                points_generated += n_p
            
            batch_locs = torch.cat(batch_locs, dim=0)
            assert batch_locs.shape[0] == num_points, f"Generated {batch_locs.shape[0]} points, expected {num_points}"
            batch_locs = torch.clamp(batch_locs, 0.0, 1.0)
            locs.append(batch_locs)
            
        return torch.stack(locs)

    def _generate_explosion(self, batch_size, num_points):
        """Explosion distribution: Dense center, sparse outskirts."""
        locs = torch.randn(batch_size, num_points, 2) * 0.15 + 0.5
        return torch.clamp(locs, 0.0, 1.0)

    def _generate_weights(self, locs, batch_size):
        """Generate weights. Can be density-aware or random."""
        # Simple random weights [1, 10]
        base_weights = torch.rand(batch_size, self.num_demand) * (self.max_demand - self.min_demand) + self.min_demand
        return base_weights

    def _calculate_dynamic_radius(self, locs, k=5):
        """Calculates dynamic radius based on average k-NN distance."""
        dists = torch.cdist(locs, locs)
        # Get distance to k-th nearest neighbor (k+1 because self is 0)
        vals, _ = torch.topk(dists, k=k+1, dim=-1, largest=False)
        kth_dists = vals[:, :, -1] 
        # Average over all nodes and scale
        avg_k_dist = kth_dists.mean(dim=1, keepdim=True)
        return avg_k_dist * 1.5