"""Test script for pre-generating and loading instances"""

import os
import torch
import numpy as np
from tensordict.tensordict import TensorDict
from rl4co.envs.graph import FLPEnv
from rl4co.data.utils import save_tensordict_to_npz, load_npz_to_tensordict
from rl4co.utils.decoding import random_policy, rollout


def compare_tensordicts(td1, td2, name1="TD1", name2="TD2", tolerance=1e-6):
    """Compare two TensorDicts for equality"""
    print(f"\n{'='*60}")
    print(f"Comparing {name1} vs {name2}")
    print(f"{'='*60}")
    
    # Check if they have the same keys
    keys1 = set(td1.keys())
    keys2 = set(td2.keys())
    
    if keys1 != keys2:
        print(f"✗ Different keys!")
        print(f"  Only in {name1}: {keys1 - keys2}")
        print(f"  Only in {name2}: {keys2 - keys1}")
        return False
    
    all_match = True
    for key in keys1:
        val1 = td1[key]
        val2 = td2[key]
        
        # Check shape
        if val1.shape != val2.shape:
            print(f"✗ Key '{key}': Different shapes {val1.shape} vs {val2.shape}")
            all_match = False
            continue
        
        # Check dtype
        if val1.dtype != val2.dtype:
            print(f"✗ Key '{key}': Different dtypes {val1.dtype} vs {val2.dtype}")
            all_match = False
            continue
        
        # Check values
        if torch.is_floating_point(val1):
            max_diff = (val1 - val2).abs().max().item()
            if max_diff > tolerance:
                print(f"✗ Key '{key}': Max difference {max_diff:.2e} > tolerance {tolerance:.2e}")
                all_match = False
            else:
                print(f"✓ Key '{key}': Match (max diff: {max_diff:.2e})")
        else:
            if not torch.equal(val1, val2):
                diff_count = (val1 != val2).sum().item()
                print(f"✗ Key '{key}': {diff_count} elements differ")
                all_match = False
            else:
                print(f"✓ Key '{key}': Exact match")
    
    if all_match:
        print(f"\n✓ All keys match!")
    else:
        print(f"\n✗ Some keys don't match!")
    
    return all_match


def test_pregenerate_and_load():
    """Test pre-generating, saving, and loading FLP instances"""
    print("="*60)
    print("Testing Pre-generation and Loading of FLP Instances")
    print("="*60)
    
    # Setup
    num_instances = 10
    data_dir = "data/pregenerated"
    os.makedirs(data_dir, exist_ok=True)
    
    # Configuration
    generator_params = {
        "num_loc": 20,
        "to_choose": 5,
    }
    
    print(f"\n✓ Configuration:")
    print(f"  - Number of instances: {num_instances}")
    print(f"  - Number of locations: {generator_params['num_loc']}")
    print(f"  - Facilities to choose: {generator_params['to_choose']}")
    
    # ============================================
    # Step 1: Generate instances
    # ============================================
    print(f"\n{'='*60}")
    print("Step 1: Generating instances")
    print(f"{'='*60}")
    
    env = FLPEnv(generator_params=generator_params)
    td_original = env.reset(batch_size=[num_instances])
    
    print(f"✓ Generated {num_instances} instances")
    print(f"  - Batch size: {td_original.batch_size}")
    print(f"  - Keys: {list(td_original.keys())}")
    print(f"  - Shapes:")
    for key, val in td_original.items():
        print(f"    - {key}: {val.shape} ({val.dtype})")
    
    # ============================================
    # Step 2: Save instances
    # ============================================
    print(f"\n{'='*60}")
    print("Step 2: Saving instances to disk")
    print(f"{'='*60}")
    
    save_path = os.path.join(data_dir, "flp_instances_test.npz")
    
    # Move to CPU before saving
    td_to_save = td_original.cpu()
    save_tensordict_to_npz(td_to_save, save_path)
    
    file_size = os.path.getsize(save_path) / 1024  # KB
    print(f"✓ Saved to: {save_path}")
    print(f"  - File size: {file_size:.2f} KB")
    
    # ============================================
    # Step 3: Load instances
    # ============================================
    print(f"\n{'='*60}")
    print("Step 3: Loading instances from disk")
    print(f"{'='*60}")
    
    td_loaded = load_npz_to_tensordict(save_path)
    
    print(f"✓ Loaded from: {save_path}")
    print(f"  - Batch size: {td_loaded.batch_size}")
    print(f"  - Keys: {list(td_loaded.keys())}")
    
    # ============================================
    # Step 4: Compare original vs loaded
    # ============================================
    print(f"\n{'='*60}")
    print("Step 4: Comparing original vs loaded instances")
    print(f"{'='*60}")
    
    # Convert to same device for comparison
    td_original_cpu = td_original.cpu()
    
    # Convert loaded numpy arrays to tensors
    td_loaded_torch = TensorDict({
        k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v 
        for k, v in td_loaded.items()
    }, batch_size=td_loaded.batch_size)
    
    match = compare_tensordicts(td_original_cpu, td_loaded_torch, 
                                "Original", "Loaded")
    
    if not match:
        print("\n✗ Test FAILED: Instances don't match!")
        return False
    
    # ============================================
    # Step 5: Test using loaded instances in env
    # ============================================
    print(f"\n{'='*60}")
    print("Step 5: Testing loaded instances in environment")
    print(f"{'='*60}")
    
    # Test with original instances
    print("\nRolling out with original instances...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    td_original_device = td_original.to(device)
    
    # Reset with pre-generated data
    td_reset_original = env.reset(td=td_original_device)
    reward_original, td_final_original, actions_original = rollout(
        env, td_reset_original, random_policy
    )
    
    print(f"✓ Original instances rollout complete")
    print(f"  - Mean reward: {reward_original.mean().item():.2f}")
    print(f"  - Reward range: [{reward_original.min().item():.2f}, {reward_original.max().item():.2f}]")
    
    # Test with loaded instances
    print("\nRolling out with loaded instances...")
    td_loaded_device = td_loaded_torch.to(device)
    td_reset_loaded = env.reset(td=td_loaded_device)
    reward_loaded, td_final_loaded, actions_loaded = rollout(
        env, td_reset_loaded, random_policy
    )
    
    print(f"✓ Loaded instances rollout complete")
    print(f"  - Mean reward: {reward_loaded.mean().item():.2f}")
    print(f"  - Reward range: [{reward_loaded.min().item():.2f}, {reward_loaded.max().item():.2f}]")
    
    # ============================================
    # Step 6: Compare rollout results (with same random seed)
    # ============================================
    print(f"\n{'='*60}")
    print("Step 6: Comparing rollout consistency")
    print(f"{'='*60}")
    
    # Set same random seed for deterministic comparison
    torch.manual_seed(42)
    td_reset_original_det = env.reset(td=td_original_device)
    reward_orig_det, _, actions_orig_det = rollout(env, td_reset_original_det, random_policy)
    
    torch.manual_seed(42)
    td_reset_loaded_det = env.reset(td=td_loaded_device)
    reward_load_det, _, actions_load_det = rollout(env, td_reset_loaded_det, random_policy)
    
    # Compare rewards
    reward_match = torch.allclose(reward_orig_det, reward_load_det, rtol=1e-5)
    print(f"\n✓ Rewards match: {reward_match}")
    if reward_match:
        print(f"  - Rewards are identical (with same random seed)")
    else:
        print(f"  - Max reward difference: {(reward_orig_det - reward_load_det).abs().max().item():.2e}")
    
    # Compare actions
    actions_match = torch.equal(actions_orig_det, actions_load_det)
    print(f"✓ Actions match: {actions_match}")
    if actions_match:
        print(f"  - Actions are identical (with same random seed)")
    else:
        diff_count = (actions_orig_det != actions_load_det).sum().item()
        print(f"  - {diff_count} actions differ")
    
    # ============================================
    # Summary
    # ============================================
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    success = match and reward_match and actions_match
    
    if success:
        print("✓ All tests PASSED!")
        print("\nKey findings:")
        print("  ✓ Instances can be saved and loaded correctly")
        print("  ✓ Loaded instances produce identical results")
        print("  ✓ Pre-generated instances work seamlessly with env.reset()")
        print(f"\nData saved to: {save_path}")
    else:
        print("✗ Some tests FAILED!")
        if not match:
            print("  ✗ Instance data doesn't match after loading")
        if not reward_match:
            print("  ✗ Rewards don't match with loaded instances")
        if not actions_match:
            print("  ✗ Actions don't match with loaded instances")
    
    return success


def test_multiple_files():
    """Test generating multiple separate files"""
    print(f"\n{'='*60}")
    print("Bonus: Testing multiple file generation")
    print(f"{'='*60}")
    
    data_dir = "data/pregenerated/flp"
    os.makedirs(data_dir, exist_ok=True)
    
    env = FLPEnv(generator_params={"num_loc": 20, "to_choose": 5})
    
    # Generate train, val, test splits
    splits = {
        "train": 1000,
        "val": 100,
        "test": 100,
    }
    
    for split_name, num_instances in splits.items():
        print(f"\nGenerating {split_name} split ({num_instances} instances)...")
        td = env.reset(batch_size=[num_instances])
        
        save_path = os.path.join(data_dir, f"flp_{split_name}.npz")
        save_tensordict_to_npz(td.cpu(), save_path)
        
        file_size = os.path.getsize(save_path) / 1024  # KB
        print(f"  ✓ Saved to {save_path} ({file_size:.2f} KB)")
    
    print(f"\n✓ Generated {sum(splits.values())} total instances across 3 splits")
    print(f"✓ Files saved to: {data_dir}/")


if __name__ == "__main__":
    # Main test
    success = test_pregenerate_and_load()
    
    # Bonus: multiple file generation
    if success:
        test_multiple_files()
    
    print("\n" + "="*60)
    if success:
        print("All tests completed successfully! ✓")
    else:
        print("Some tests failed! ✗")
    print("="*60)
