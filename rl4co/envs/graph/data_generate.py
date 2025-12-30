
import os
import sys
import torch
import numpy as np
import time
import random
import pickle
from typing import Optional, Dict, Any, List

# Add project root to path to ensure imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from rl4co.envs.graph.flp.generator import FLPGenerator
from rl4co.envs.graph.flp.solvers import solve_flp_scip
from rl4co.envs.graph.mclp.generator import MCLPGenerator
from rl4co.envs.graph.mclp.solvers import solve_mclp_scip
from rl4co.envs.graph.stp.generator import STPGenerator
from rl4co.envs.graph.stp.solvers import solve_stp_scip

def solve_instance(env_type: str, data: Dict[str, Any]) -> float:
    """
    Solve a single instance using the appropriate solver.
    """
    if env_type == "flp":
        # FLP Data: locs (N, 2), to_choose (int)
        locs = data["locs"]
        to_choose = data["to_choose"]
        
        # solve_flp_gurobi expects numpy arrays
        selected, obj_val, info = solve_flp_scip(
            locations=locs,
            to_choose=to_choose,
            time_limit=10.0,
            verbose=False
        )
        return obj_val

    elif env_type == "mclp":
        # MCLP Data: demand_locs (N, 2), facility_locs (M, 2), demand_weights (N), radius, to_select
        demand_locs = data["demand_locs"]
        facility_locs = data["facility_locs"]
        demand_weights = data["demand_weights"]
        coverage_radius = data["coverage_radius"]
        num_facilities_to_select = data["num_facilities_to_select"]
        
        selected, obj_val, info = solve_mclp_scip(
            demand_locs=demand_locs,
            demand_weights=demand_weights,
            facility_locs=facility_locs,
            coverage_radius=coverage_radius,
            num_facilities_to_select=num_facilities_to_select,
            time_limit=10.0,
            verbose=False
        )
        return obj_val

    elif env_type == "stp":
        # STP Data: locs, terminals, edge_index/weights (if available) or raw graph
        # STPGenerator provides: locs, graph connectivity (adj matrix or similar)
        # solve_stp_gurobi expects: locs, terminals, edge_list, edge_weights
        
        locs = data["locs"]
        terminals = data["terminals"] # indices
        
        # We need to reconstruct edge list and weights from the generator output
        # STPGenerator returns 'adj_mask' and 'dists' in TensorDict
        # But here 'data' is a single sample extracted and converted to numpy
        
        dists = data["dists"] # (N, N)
        adj_mask = data["adj_mask"] # (N, N) bool
        
        # Convert adj_mask to edge_list
        # indices of true values
        rows, cols = np.where(adj_mask)
        # Filter strictly upper triangle to avoid duplicates if undirected, 
        # but solver might expect directed or undirected. 
        # Usually edge_list is (E, 2).
        
        edge_list = np.stack([rows, cols], axis=1)
        edge_weights = dists # Pass full matrix
        
        selected, obj_val, info = solve_stp_scip(
            locs=locs,
            terminals=terminals,
            edge_list=edge_list,
            edge_weights=edge_weights,
            time_limit=10.0,
            verbose=False
        )
        return obj_val
        
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
    
    # 1. Instantiate Generator
    if env_type == "flp":
        # Kwargs can control num_loc, distribution etc.
        num_loc = kwargs.pop("num_loc", 50)
        to_choose = kwargs.pop("to_choose", 10)
        distribution = kwargs.pop("distribution", "uniform")
        
        print(f"Config: num_loc={num_loc}, to_choose={to_choose}, dist={distribution}")
        
        generator = FLPGenerator(
            num_loc=num_loc,
            to_choose=to_choose,
            loc_distribution=distribution,
            **kwargs
        )
        
    elif env_type == "mclp":
        num_demand = kwargs.pop("num_demand", 50)
        num_facility = kwargs.pop("num_facility", 20)
        to_select = kwargs.pop("to_select", 5)
        distribution = kwargs.pop("distribution", "uniform")
        
        print(f"Config: n_demand={num_demand}, n_facility={num_facility}, select={to_select}, dist={distribution}")
        
        generator = MCLPGenerator(
            num_demand=num_demand,
            num_facility=num_facility,
            num_facilities_to_select=to_select,
            distribution=distribution,
            **kwargs
        )
        
    elif env_type == "stp":
        num_nodes = kwargs.pop("num_nodes", 50)
        num_terminals = kwargs.pop("num_terminals", 10)
        distribution = kwargs.pop("distribution", "uniform")
        graph_type = kwargs.pop("graph_type", "delaunay")
        
        print(f"Config: n_nodes={num_nodes}, n_terminals={num_terminals}, dist={distribution}, graph={graph_type}")
        
        generator = STPGenerator(
            num_nodes=num_nodes,
            num_terminals=num_terminals,
            loc_distribution=distribution,
            graph_type=graph_type,
            **kwargs
        )
    else:
        raise ValueError("Invalid env_type")

    # 2. Generate Batch
    td = generator(batch_size=batch_size)
    
    # 3. Solve each instance
    results = []
    
    return td, generator

def solve_batch(env_type, td):
    batch_size = td.batch_size[0]
    results = []
    
    print(f"Solving {batch_size} instances...")
    
    for i in range(batch_size):
        sample_data = {}
        
        if env_type == "flp":
            sample_data["locs"] = td["locs"][i].cpu().numpy()
            sample_data["to_choose"] = td["to_choose"][i].item()
            
        elif env_type == "mclp":
            sample_data["demand_locs"] = td["demand_locs"][i].cpu().numpy()
            sample_data["facility_locs"] = td["facility_locs"][i].cpu().numpy()
            sample_data["demand_weights"] = td["demand_weights"][i].cpu().numpy()
            
            cr = td["coverage_radius"][i]
            sample_data["coverage_radius"] = cr.item() if cr.numel() == 1 else cr.cpu().numpy()
            
            sample_data["num_facilities_to_select"] = td["num_facilities_to_select"][i].item()
                
        elif env_type == "stp":
            sample_data["locs"] = td["locs"][i].cpu().numpy()
            sample_data["terminals"] = td["terminals"][i].cpu().numpy()
            sample_data["dists"] = td["edge_weights"][i].cpu().numpy()
            sample_data["adj_mask"] = td["adjacency"][i].cpu().numpy()

        try:
            obj = solve_instance(env_type, sample_data)
            results.append(obj)
            if obj is not None:
                print(f"  Sample {i}: Objective = {obj:.4f}")
            else:
                print(f"  Sample {i}: Failed to solve - Solver returned None")
        except Exception as e:
            print(f"  Sample {i}: Failed to solve - {e}")
            results.append(None)
            
    return results


def main():
    results_storage = {}
    distributions = ['uniform', 'cluster', "gaussian_mixture", "mix_multi_distributions"]
    results_storage={
        "flp": [],
        "mclp": [],
        "stp": []
    }
    for i in range(10):
        # 1. FLP
        flp_num_loc = random.randint(100, 500)
        # Linear mapping: 100 -> 10, 500 -> 30
        flp_choice = int(10 + (flp_num_loc - 100) / 400 * 20)
        flp_dist = random.choice(distributions)
        kwargs = {}
        if flp_dist == "cluster":
            n_cluster = random.randint(3,5)
            kwargs["n_cluster"] = n_cluster
        if flp_dist == "gaussian_mixture":
            kwargs["num_modes"] = random.randint(1,3)
            kwargs["cdist"] = random.random()*2
    
        print(f"\n--- FLP Configuration ---")
        print(f"Num Loc: {flp_num_loc}, Choice: {flp_choice}, Dist: {flp_dist}")
    
        td_flp, gen_flp = generate_and_solve(
            "flp", 
            batch_size=1, 
            num_loc=flp_num_loc, 
            to_choose=flp_choice,
            distribution=flp_dist,
            **kwargs
        )
        flp_objs = solve_batch("flp", td_flp)
        results_storage["flp"].append({
            "td": td_flp.cpu(), 
            "objs": flp_objs, 
            "config": {"num_loc": flp_num_loc, "to_choose": flp_choice, "dist": flp_dist}
        })

        # 2. MCLP
        mclp_num_demand = random.randint(100, 500)
        mclp_select = int(10 + (mclp_num_demand - 100) / 400 * 20)
        # Ensure sufficient facilities. Proportional to demand, e.g., 20% but at least select + 10
        mclp_num_facility = max(mclp_select + 10, int(mclp_num_demand * 0.2))
        mclp_dist = random.choice(distributions)
        kwargs = {}
        if mclp_dist == "cluster":
            n_cluster = random.randint(3,5)
            kwargs["n_cluster"] = n_cluster
        if mclp_dist == "gaussian_mixture":
            kwargs["num_modes"] = random.randint(1,3)
            kwargs["cdist"] = random.random()*2
        
        print(f"\n--- MCLP Configuration ---")
        print(f"Num Demand: {mclp_num_demand}, Num Facility: {mclp_num_facility}, Select: {mclp_select}, Dist: {mclp_dist}")
        
        td_mclp, gen_mclp = generate_and_solve(
            "mclp", 
            batch_size=1, 
            num_demand=mclp_num_demand,
            num_facility=mclp_num_facility,
            to_select=mclp_select,
            distribution=mclp_dist,
            **kwargs
        )
        # Inject num_facilities_to_select if missing
        if "num_facilities_to_select" not in td_mclp.keys():
            td_mclp["num_facilities_to_select"] = torch.full((1,), mclp_select) 
            
        mclp_objs = solve_batch("mclp", td_mclp)
        results_storage["mclp"].append({
            "td": td_mclp.cpu(), 
            "objs": mclp_objs, 
            "config": {"num_demand": mclp_num_demand, "num_facility": mclp_num_facility, "to_select": mclp_select, "dist": mclp_dist}
        })  

        # 3. STP
        stp_nodes = random.randint(100, 500)
        stp_terminals = int(10 + (stp_nodes - 100) / 400 * 20)
        stp_dist = random.choice(distributions)
        kwargs = {}
        if stp_dist == "cluster":
            n_cluster = random.randint(3,5)
            kwargs["n_cluster"] = n_cluster
        if stp_dist == "gaussian_mixture":
            kwargs["num_modes"] = random.randint(1,3)
            kwargs["cdist"] = random.random()*2
        
        print(f"\n--- STP Configuration ---")
        print(f"Num Nodes: {stp_nodes}, Num Terminals: {stp_terminals}, Dist: {stp_dist}")
        
        td_stp, gen_stp = generate_and_solve(
            "stp",
            batch_size=1,
            num_nodes=stp_nodes,
            num_terminals=stp_terminals,
            distribution=stp_dist,
            graph_type="delaunay",
            **kwargs
        )
        stp_objs = solve_batch("stp", td_stp)
        results_storage["stp"].append({
            "td": td_stp.cpu(), 
            "objs": stp_objs, 
            "config": {"num_nodes": stp_nodes, "num_terminals": stp_terminals, "dist": stp_dist}
        })
    
    # Save to pickle
    save_path = "results.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(results_storage, f)
    print(f"\nSaved results to {save_path}")

if __name__ == "__main__":
    main()
