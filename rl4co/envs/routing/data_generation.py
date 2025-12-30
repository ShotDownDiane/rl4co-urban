import os
import sys
import torch
import numpy as np
import random
import pickle
from typing import Optional, Dict, Any, List

# Add project root to path to ensure imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rl4co.envs.routing.tsp.generator import TSPGenerator
from rl4co.envs.routing.cvrp.generator import CVRPGenerator
from rl4co.envs.routing.tsp.solvers import solve_tsp
from rl4co.envs.routing.cvrp.solvers import solve_cvrp

def solve_instance(env_type: str, data: Dict[str, Any]) -> float:
    """
    Solve a single instance using the appropriate solver.
    """
    if env_type == "tsp":
        locs = data["locs"]
        # Try LKH first, then ORTools, then Greedy
        try:
            # LKH is the strongest heuristic
            tour, obj, _ = solve_tsp(locs, method="lkh", time_limit=10.0)
            if obj is None:
                raise Exception("LKH returned None")
        except Exception:
            try:
                # OR-Tools is also very good
                tour, obj, _ = solve_tsp(locs, method="ortools", time_limit=10.0)
                if obj is None:
                    raise Exception("ORTools returned None")
            except Exception:
                # Fallback to greedy
                tour, obj, _ = solve_tsp(locs, method="greedy")
        return obj

    elif env_type == "cvrp":
        depot = data["depot"]
        locs = data["locs"]
        demand = data["demand"]
        capacity = data["capacity"]

        # Use the robust PyVRP solver wrapper
        _, obj, info = solve_cvrp(depot, locs, demand, capacity, method="pyvrp", time_limit=10.0)
        
        if obj is None:
             print(f"CVRP Solver failed: {info.get('error', 'Unknown error')}")
             return None
        return obj
    
    else:
        raise ValueError(f"Unknown env_type: {env_type}")

def generate_and_solve(
    env_type: str,
    batch_size: int = 10,
    seed: int = 1234,
    **kwargs
):
    """
    Generate data and solve it.
    """
    print(f"\n[{env_type.upper()}] Generating {batch_size} samples...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if env_type == "tsp":
        num_loc = kwargs.pop("num_loc", 50)
        distribution = kwargs.pop("distribution", "uniform")
        
        print(f"Config: num_loc={num_loc}, dist={distribution}")
        
        generator = TSPGenerator(
            num_loc=num_loc,
            loc_distribution=distribution,
            **kwargs
        )
        
    elif env_type == "cvrp":
        num_loc = kwargs.pop("num_loc", 50)
        distribution = kwargs.pop("distribution", "uniform")
        
        print(f"Config: num_loc={num_loc}, dist={distribution}")
        
        generator = CVRPGenerator(
            num_loc=num_loc,
            loc_distribution=distribution,
            **kwargs
        )
        
    else:
        raise ValueError("Invalid env_type")

    # Generate Batch
    td = generator(batch_size=batch_size)
    
    return td, generator

def solve_batch(env_type, td):
    batch_size = td.batch_size[0]
    results = []
    
    print(f"Solving {batch_size} instances...")
    
    for i in range(batch_size):
        sample_data = {}
        
        if env_type == "tsp":
            sample_data["locs"] = td["locs"][i].cpu().numpy()
            
        elif env_type == "cvrp":
            sample_data["locs"] = td["locs"][i].cpu().numpy()
            sample_data["depot"] = td["depot"][i].cpu().numpy()
            
            # CVRPGenerator normalizes demand by capacity
            # We need to restore it or pass capacity properly
            cap = td["capacity"][i].item()
            sample_data["capacity"] = cap
            # Restore demand magnitude relative to capacity
            sample_data["demand"] = td["demand"][i].cpu().numpy() * cap
            
        try:
            obj = solve_instance(env_type, sample_data)
            results.append(obj)
            if obj is not None:
                print(f"  Sample {i}: Objective = {obj:.4f}")
            else:
                print(f"  Sample {i}: Failed to solve")
        except Exception as e:
            print(f"  Sample {i}: Failed to solve - {e}")
            results.append(None)
            
    return results

def main():
    results_storage = {
        "tsp": [],
        "cvrp": []
    }
    distributions = ['uniform', 'cluster', "gaussian_mixture", "mix_multi_distributions"]
    
    # Capacities for CVRP (from Kool et al. 2019)
    CAPACITIES = {
        10: 20.0, 15: 25.0, 20: 30.0, 30: 33.0, 40: 37.0, 50: 40.0, 60: 43.0, 
        75: 45.0, 100: 50.0, 125: 55.0, 150: 60.0, 200: 70.0, 500: 100.0, 1000: 150.0
    }

    def get_capacity(num_loc):
        # Find the closest key
        closest_loc = min(CAPACITIES.keys(), key=lambda x: abs(x - num_loc))
        return CAPACITIES[closest_loc]

    # Generate 10 samples (iterations) as requested
    for i in range(10):
        # 1. TSP
        tsp_num_loc = random.randint(20, 100)
        # Note: TSP doesn't have a 'choice' parameter like FLP/MCLP, so we skip that part of the logic
        tsp_dist = random.choice(distributions)
        kwargs = {}
        if tsp_dist == "cluster":
            kwargs["n_cluster"] = random.randint(2,3)
        if tsp_dist == "gaussian_mixture":
            kwargs["num_modes"] = random.randint(1,3)
            kwargs["cdist"] = random.random()*2
    
        print(f"\n--- TSP Configuration ---")
        print(f"Num Loc: {tsp_num_loc}, Dist: {tsp_dist}")
    
        td_tsp, gen_tsp = generate_and_solve(
            "tsp", 
            batch_size=1, 
            num_loc=tsp_num_loc, 
            distribution=tsp_dist,
            **kwargs
        )
        tsp_objs = solve_batch("tsp", td_tsp)
        results_storage["tsp"].append({
            "td": td_tsp.cpu(), 
            "objs": tsp_objs, 
            "config": {"num_loc": tsp_num_loc, "dist": tsp_dist}
        })

        # 2. CVRP
        cvrp_num_loc = random.randint(20, 100)
        cvrp_dist = random.choice(distributions)
        kwargs = {}
        # Explicitly set capacity
        kwargs["capacity"] = get_capacity(cvrp_num_loc)
        
        if cvrp_dist == "cluster":
            kwargs["n_cluster"] = random.randint(2,3)
        if cvrp_dist == "gaussian_mixture":
            kwargs["num_modes"] = random.randint(1,3)
            kwargs["cdist"] = random.random()*2
        
        print(f"\n--- CVRP Configuration ---")
        print(f"Num Loc: {cvrp_num_loc}, Dist: {cvrp_dist}, Capacity: {kwargs['capacity']}")
        
        td_cvrp, gen_cvrp = generate_and_solve(
            "cvrp", 
            batch_size=1, 
            num_loc=cvrp_num_loc,
            distribution=cvrp_dist,
            **kwargs
        )
            
        cvrp_objs = solve_batch("cvrp", td_cvrp)
        results_storage["cvrp"].append({
            "td": td_cvrp.cpu(), 
            "objs": cvrp_objs, 
            "config": {"num_loc": cvrp_num_loc, "dist": cvrp_dist}
        })  

    # Save to pickle
    save_path = "routing_results.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(results_storage, f)
    print(f"\nSaved results to {save_path}")

if __name__ == "__main__":
    main()
