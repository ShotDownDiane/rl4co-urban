import os
import numpy as np
from typing import Optional, Tuple, Union

def solve_cvrp(
    depot: np.ndarray,
    locs: np.ndarray,
    demand: np.ndarray,
    capacity: float,
    method: str = "pyvrp",
    time_limit: float = 10.0,
    verbose: bool = False,
    **kwargs
) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
    """
    Solve CVRP using various methods.
    
    Args:
        depot: Depot location [2]
        locs: Customer locations [N, 2]
        demand: Customer demands [N]
        capacity: Vehicle capacity
        method: Solver method ("pyvrp")
        time_limit: Time limit in seconds
        verbose: Whether to print logging info
        
    Returns:
        tuple: (tour, cost, info)
            tour: List of visited nodes (including depot 0) or similar structure
            cost: Total distance/cost
            info: Dictionary with additional info
    """
    method = method.lower()
    
    if method == "pyvrp":
        return solve_cvrp_pyvrp(depot, locs, demand, capacity, time_limit, verbose, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

def solve_cvrp_pyvrp(
    depot: np.ndarray,
    locs: np.ndarray,
    demand: np.ndarray,
    capacity: float,
    time_limit: float = 10.0,
    verbose: bool = False,
    **kwargs
) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
    try:
        from pyvrp import Model, ProblemData, Client, Depot, VehicleType
        from pyvrp.stop import MaxRuntime
        from pyvrp import solve
    except ImportError:
        return None, None, {"status": "failed", "error": "PyVRP not installed"}

    try:
        # Scale to integers (PyVRP typically works with integers)
        scaling = 10000
        depot_int = (depot * scaling).astype(int)
        locs_int = (locs * scaling).astype(int)
        demand_int = (demand * scaling).astype(int)
        capacity_int = int(capacity * scaling)
        
        # Prepare clients and depot
        clients = []
        for i in range(len(locs)):
            clients.append(Client(
                x=locs_int[i][0], 
                y=locs_int[i][1], 
                delivery=[int(demand_int[i])]
            ))
            
        depot_obj = Depot(x=depot_int[0], y=depot_int[1])
        
        # Prepare vehicle type
        # Using a large number of vehicles to ensure feasibility
        num_vehicles = len(locs)
        # capacity must be a list for VehicleType constructor
        vehicle_type = VehicleType(num_vehicles, capacity=[capacity_int])
        
        # Compute distance matrix
        # Combine depot and clients: index 0 is depot, 1..N are clients
        all_locs = np.vstack((depot_int.reshape(1, 2), locs_int))
        diff = all_locs[:, np.newaxis, :] - all_locs[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1)).astype(int)
        
        # Create ProblemData
        # distances and durations must be lists of matrices (one per profile)
        data = ProblemData(
            clients=clients,
            depots=[depot_obj],
            vehicle_types=[vehicle_type],
            distance_matrices=[dist_matrix],
            duration_matrices=[np.zeros_like(dist_matrix)],
            groups=[]
        )

        # Solve
        res = solve(data, stop=MaxRuntime(time_limit), display=verbose)
        
        cost = res.cost() / scaling
        
        # Check feasibility implicitly via cost (PyVRP returns huge cost if infeasible? No, we check is_feasible)
        if not res.is_feasible():
             return None, None, {"status": "infeasible", "solve_time": res.runtime}

        return None, cost, {"status": "solved", "solve_time": res.runtime}
        
    except Exception as e:
        if verbose:
            print(f"PyVRP failed: {e}")
        return None, None, {"status": "failed", "error": str(e)}
