"""Tool for pre-generating datasets for any RL4CO environment"""

import os
import argparse
import time
import torch
from rl4co.envs.graph import FLPEnv, MCLPEnv, MCPEnv, STPEnv
from rl4co.data.utils import save_tensordict_to_npz


def get_env(env_name, **env_params):
    """Get environment by name"""
    env_map = {
        "flp": FLPEnv,
        "mclp": MCLPEnv,
        "mcp": MCPEnv,
        "stp": STPEnv,
    }
    
    if env_name.lower() not in env_map:
        raise ValueError(f"Unknown environment: {env_name}. Available: {list(env_map.keys())}")
    
    return env_map[env_name.lower()](generator_params=env_params)


def pregenerate_dataset(
    env_name: str,
    output_dir: str,
    num_train: int = 10000,
    num_val: int = 1000,
    num_test: int = 1000,
    batch_size: int = 1000,
    compress: bool = False,
    **env_params
):
    """Pre-generate datasets for training, validation, and testing
    
    Args:
        env_name: Name of the environment (flp, mclp, mcp, stp)
        output_dir: Directory to save the datasets
        num_train: Number of training instances
        num_val: Number of validation instances
        num_test: Number of test instances
        batch_size: Batch size for generation (larger = faster but more memory)
        compress: Whether to compress the npz files
        **env_params: Environment-specific parameters
    """
    print("="*80)
    print(f"Pre-generating {env_name.upper()} Dataset")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Display configuration
    print(f"\n✓ Configuration:")
    print(f"  - Environment: {env_name}")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Training instances: {num_train}")
    print(f"  - Validation instances: {num_val}")
    print(f"  - Test instances: {num_test}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Compress: {compress}")
    print(f"  - Environment parameters:")
    for k, v in env_params.items():
        print(f"    - {k}: {v}")
    
    # Create environment
    print(f"\n✓ Creating environment...")
    env = get_env(env_name, **env_params)
    print(f"  - Environment: {env.name}")
    
    # Generate datasets
    splits = {
        "train": num_train,
        "val": num_val,
        "test": num_test,
    }
    
    total_time = 0
    total_instances = 0
    
    for split_name, num_instances in splits.items():
        if num_instances == 0:
            continue
            
        print(f"\n{'='*80}")
        print(f"Generating {split_name} split ({num_instances} instances)")
        print(f"{'='*80}")
        
        # Generate in batches
        all_data = []
        num_batches = (num_instances + batch_size - 1) // batch_size
        
        start_time = time.time()
        
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_instances - i * batch_size)
            
            # Generate batch
            td_batch = env.reset(batch_size=[current_batch_size])
            all_data.append(td_batch.cpu())
            
            print(f"  Batch {i+1}/{num_batches}: Generated {current_batch_size} instances", end="\r")
        
        print()  # New line after progress
        
        # Concatenate all batches
        print(f"  Concatenating batches...")
        from tensordict import cat
        td_full = cat(all_data, dim=0)
        
        # Save to file
        save_path = os.path.join(output_dir, f"{env_name}_{split_name}.npz")
        print(f"  Saving to {save_path}...")
        
        save_tensordict_to_npz(td_full, save_path, compress=compress)
        
        elapsed = time.time() - start_time
        file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
        
        print(f"✓ Saved {num_instances} instances")
        print(f"  - File: {save_path}")
        print(f"  - Size: {file_size:.2f} MB")
        print(f"  - Time: {elapsed:.2f}s ({num_instances/elapsed:.0f} instances/s)")
        
        total_time += elapsed
        total_instances += num_instances
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Summary")
    print(f"{'='*80}")
    print(f"✓ Successfully generated {total_instances} instances")
    print(f"  - Total time: {total_time:.2f}s ({total_instances/total_time:.0f} instances/s)")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Files:")
    for split_name, num_instances in splits.items():
        if num_instances > 0:
            file_path = os.path.join(output_dir, f"{env_name}_{split_name}.npz")
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"    - {env_name}_{split_name}.npz: {num_instances} instances ({file_size:.2f} MB)")
    
    print(f"\n✓ Dataset ready for training!")


def main():
    parser = argparse.ArgumentParser(description="Pre-generate datasets for RL4CO environments")
    
    # Basic arguments
    parser.add_argument("--env", type=str, required=True, 
                       choices=["flp", "mclp", "mcp", "stp"],
                       help="Environment name")
    parser.add_argument("--output-dir", type=str, default="data/pregenerated",
                       help="Output directory")
    parser.add_argument("--num-train", type=int, default=10000,
                       help="Number of training instances")
    parser.add_argument("--num-val", type=int, default=1000,
                       help="Number of validation instances")
    parser.add_argument("--num-test", type=int, default=1000,
                       help="Number of test instances")
    parser.add_argument("--batch-size", type=int, default=1000,
                       help="Batch size for generation")
    parser.add_argument("--compress", action="store_true",
                       help="Compress the npz files")
    
    # FLP-specific arguments
    parser.add_argument("--num-loc", type=int, default=100,
                       help="FLP: Number of locations")
    parser.add_argument("--to-choose", type=int, default=10,
                       help="FLP: Number of facilities to choose")
    
    # MCLP-specific arguments
    parser.add_argument("--num-demand", type=int, default=100,
                       help="MCLP: Number of demand points")
    parser.add_argument("--num-facility", type=int, default=50,
                       help="MCLP: Number of candidate facilities")
    parser.add_argument("--num-facilities-to-select", type=int, default=10,
                       help="MCLP: Number of facilities to select")
    parser.add_argument("--coverage-radius", type=float, default=0.2,
                       help="MCLP: Coverage radius")
    parser.add_argument("--distribution", type=str, default="uniform",
                       choices=["uniform", "cluster", "explosion"],
                       help="MCLP: Data distribution")
    parser.add_argument("--dynamic-radius", action="store_true",
                       help="MCLP: Use dynamic radius")
    
    # MCP-specific arguments
    parser.add_argument("--num-items", type=int, default=100,
                       help="MCP: Number of items")
    parser.add_argument("--num-sets", type=int, default=50,
                       help="MCP: Number of sets")
    parser.add_argument("--n-sets-to-choose", type=int, default=10,
                       help="MCP: Number of sets to choose")
    
    # STP-specific arguments
    parser.add_argument("--num-nodes", type=int, default=50,
                       help="STP: Number of nodes")
    parser.add_argument("--num-terminals", type=int, default=10,
                       help="STP: Number of terminal nodes")
    
    args = parser.parse_args()
    
    # Build environment parameters based on environment type
    env_params = {}
    
    if args.env == "flp":
        env_params = {
            "num_loc": args.num_loc,
            "to_choose": args.to_choose,
        }
        output_dir = os.path.join(args.output_dir, f"flp_n{args.num_loc}_k{args.to_choose}")
    
    elif args.env == "mclp":
        env_params = {
            "num_demand": args.num_demand,
            "num_facility": args.num_facility,
            "num_facilities_to_select": args.num_facilities_to_select,
            "coverage_radius": args.coverage_radius,
            "distribution": args.distribution,
            "dynamic_radius": args.dynamic_radius,
        }
        output_dir = os.path.join(args.output_dir, 
                                 f"mclp_d{args.num_demand}_f{args.num_facility}_k{args.num_facilities_to_select}_{args.distribution}")
    
    elif args.env == "mcp":
        env_params = {
            "num_items": args.num_items,
            "num_sets": args.num_sets,
            "n_sets_to_choose": args.n_sets_to_choose,
        }
        output_dir = os.path.join(args.output_dir, 
                                 f"mcp_i{args.num_items}_s{args.num_sets}_k{args.n_sets_to_choose}")
    
    elif args.env == "stp":
        env_params = {
            "num_nodes": args.num_nodes,
            "num_terminals": args.num_terminals,
        }
        output_dir = os.path.join(args.output_dir, 
                                 f"stp_n{args.num_nodes}_t{args.num_terminals}")
    
    # Generate dataset
    pregenerate_dataset(
        env_name=args.env,
        output_dir=output_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        batch_size=args.batch_size,
        compress=args.compress,
        **env_params
    )


if __name__ == "__main__":
    main()
