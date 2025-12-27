"""
Simple training script for MCLP (Maximum Covering Location Problem)
Based on RL4CO's AttentionModel
"""

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from rl4co.envs.graph import MCLPEnv
from rl4co.models.zoo import AttentionModel
from rl4co.models.zoo.am.policy import AttentionModelPolicy


def train_mclp_simple():
    """Simple training example for MCLP"""
    
    print("="*80)
    print("Training AttentionModel on MCLP")
    print("="*80)
    
    # ============================================
    # 1. Environment Setup
    # ============================================
    print("\n1. Setting up environment...")
    
    env = MCLPEnv(generator_params={
        "num_demand": 50,               # Number of demand points
        "num_facility": 30,             # Number of candidate facilities
        "num_facilities_to_select": 8,  # Number of facilities to select
        "coverage_radius": 0.2,         # Coverage radius
        "distribution": "uniform",      # Distribution type
        "dynamic_radius": False,        # Fixed radius
    })
    
    print(f"✓ Environment: {env.name}")
    print(f"  - Demand points: 50")
    print(f"  - Candidate facilities: 30")
    print(f"  - Facilities to select: 8")
    print(f"  - Coverage radius: 0.2 (fixed)")
    print(f"  - Distribution: uniform")
    
    # ============================================
    # 2. Policy Setup
    # ============================================
    print("\n2. Setting up policy...")
    
    policy = AttentionModelPolicy(
        env_name=env.name,
        embed_dim=128,
        num_encoder_layers=3,
        num_heads=8,
    )
    
    print(f"✓ Policy: AttentionModelPolicy")
    
    # ============================================
    # 3. Model Setup
    # ============================================
    print("\n3. Setting up model...")
    
    model = AttentionModel(
        env,
        policy=policy,
        baseline="rollout",
        batch_size=256,              # MCLP needs smaller batch due to complexity
        val_batch_size=64,
        test_batch_size=64,
        train_data_size=10_000,
        val_data_size=1_000,
        test_data_size=1_000,
        optimizer_kwargs={"lr": 1e-4},
        policy_kwargs={
            "train_decode_type": "sampling",
            "val_decode_type": "greedy",
            "test_decode_type": "greedy",
        }
    )
    
    print(f"✓ Model: AttentionModel (REINFORCE)")
    
    # ============================================
    # 4. Training Setup
    # ============================================
    print("\n4. Setting up trainer...")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/mclp_uniform",
        filename="mclp-{epoch:02d}-{val_reward:.2f}",
        monitor="val/reward",
        mode="max",  # MCLP: maximize covered demand
        save_top_k=3,
        verbose=True,
    )
    
    logger = TensorBoardLogger("logs", name="mclp_uniform")
    
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        val_check_interval=0.5,
    )
    
    print(f"✓ Trainer configured")
    print(f"  - Max epochs: 20")
    print(f"  - Checkpoints: checkpoints/mclp_uniform/")
    
    # ============================================
    # 5. Training
    # ============================================
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    trainer.fit(model)
    
    # ============================================
    # 6. Testing
    # ============================================
    print("\n" + "="*80)
    print("Running test...")
    print("="*80 + "\n")
    
    trainer.test(model)
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"✓ Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"\nTo view TensorBoard:")
    print(f"  tensorboard --logdir logs/mclp_uniform/")


def train_mclp_cluster():
    """Training on cluster distribution (more challenging)"""
    
    print("="*80)
    print("Training MCLP on Cluster Distribution")
    print("="*80)
    
    env = MCLPEnv(generator_params={
        "num_demand": 100,
        "num_facility": 50,
        "num_facilities_to_select": 10,
        "distribution": "cluster",      # Cluster distribution
        "dynamic_radius": True,         # Dynamic radius based on density
        "coverage_radius": 0.2,         # Default radius (overridden by dynamic)
    })
    
    print(f"\n✓ Environment: MCLP Cluster Distribution")
    print(f"  - Demand points: 100")
    print(f"  - Candidate facilities: 50")
    print(f"  - Facilities to select: 10")
    print(f"  - Distribution: cluster (challenging!)")
    print(f"  - Dynamic radius: True")
    
    policy = AttentionModelPolicy(
        env_name=env.name,
        embed_dim=128,
        num_encoder_layers=3,
        num_heads=8,
    )
    
    model = AttentionModel(
        env,
        policy=policy,
        baseline="rollout",
        batch_size=128,              # Smaller batch for larger problem
        val_batch_size=32,
        test_batch_size=32,
        train_data_size=50_000,      # More data for harder problem
        val_data_size=1_000,
        test_data_size=1_000,
        optimizer_kwargs={"lr": 1e-4},
        policy_kwargs={
            "train_decode_type": "sampling",
            "val_decode_type": "greedy",
            "test_decode_type": "greedy",
        }
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/mclp_cluster",
        filename="mclp-{epoch:02d}-{val_reward:.2f}",
        monitor="val/reward",
        mode="max",
        save_top_k=3,
    )
    
    logger = TensorBoardLogger("logs", name="mclp_cluster")
    
    trainer = pl.Trainer(
        max_epochs=50,              # More epochs for challenging distribution
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,
        val_check_interval=0.5,
    )
    
    print(f"\n✓ Training for 50 epochs (challenging problem)...")
    
    trainer.fit(model)
    trainer.test(model)
    
    print("\n✓ Training complete!")


def train_mclp_with_pregenerated():
    """Example of training with pre-generated dataset"""
    
    print("="*80)
    print("Training MCLP with Pre-generated Dataset")
    print("="*80)
    
    from rl4co.data.utils import load_npz_to_tensordict
    from torch.utils.data import DataLoader, TensorDataset
    
    # Check if pre-generated data exists
    import os
    train_path = "data/pregenerated/mclp_d50_f30_k8_uniform/mclp_train.npz"
    val_path = "data/pregenerated/mclp_d50_f30_k8_uniform/mclp_val.npz"
    
    if not os.path.exists(train_path):
        print(f"\n✗ Pre-generated data not found: {train_path}")
        print("  Please run the following command first:")
        print("  python tools/pregenerate_dataset.py --env mclp \\")
        print("    --num-demand 50 --num-facility 30 --num-facilities-to-select 8 \\")
        print("    --distribution uniform --num-train 10000 --num-val 1000")
        return
    
    print(f"\n✓ Loading pre-generated data...")
    print(f"  - Train: {train_path}")
    print(f"  - Val: {val_path}")
    
    train_data = load_npz_to_tensordict(train_path)
    val_data = load_npz_to_tensordict(val_path)
    
    print(f"✓ Loaded {len(train_data)} training instances")
    print(f"✓ Loaded {len(val_data)} validation instances")
    
    # Create environment
    env = MCLPEnv(generator_params={
        "num_demand": 50,
        "num_facility": 30,
        "num_facilities_to_select": 8,
        "coverage_radius": 0.2,
        "distribution": "uniform",
        "dynamic_radius": False,
    })
    
    # Create policy and model
    policy = AttentionModelPolicy(env_name=env.name, embed_dim=128, 
                                  num_encoder_layers=3, num_heads=8)
    
    model = AttentionModel(
        env,
        policy=policy,
        baseline="rollout",
        batch_size=256,
        optimizer_kwargs={"lr": 1e-4},
    )
    
    # Note: For using pre-generated data in training loop,
    # you would need to customize the training step or use a custom DataModule
    # This is an advanced topic - for now we'll use the standard approach
    
    print("\n✓ Model setup complete")
    print("  Note: Full integration of pre-generated data requires custom DataModule")
    print("  For now, the model will generate data on-the-fly during training")
    
    # Standard training
    trainer = pl.Trainer(max_epochs=10, accelerator="auto", devices=1)
    trainer.fit(model)
    
    print("\n✓ Training complete!")


def evaluate_mclp():
    """Evaluate trained MCLP model on different distributions"""
    
    print("="*80)
    print("Evaluating MCLP Model on Different Distributions")
    print("="*80)
    
    checkpoint_path = "checkpoints/mclp_uniform/best.ckpt"
    
    try:
        model = AttentionModel.load_from_checkpoint(checkpoint_path)
        print(f"✓ Model loaded from: {checkpoint_path}")
    except FileNotFoundError:
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print("  Please train a model first using train_mclp_simple()")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Test on different distributions
    distributions = ["uniform", "cluster", "explosion"]
    
    print(f"\n{'='*80}")
    print("Testing on Different Distributions (OOD Generalization)")
    print(f"{'='*80}")
    
    from rl4co.utils.decoding import rollout
    
    results = []
    
    for dist in distributions:
        print(f"\n{dist.capitalize()} Distribution:")
        
        # Create environment with this distribution
        env = MCLPEnv(generator_params={
            "num_demand": 50,
            "num_facility": 30,
            "num_facilities_to_select": 8,
            "coverage_radius": 0.2,
            "distribution": dist,
            "dynamic_radius": False,
        })
        
        # Generate test instances
        td = env.reset(batch_size=[100]).to(device)
        
        # Evaluate
        model.policy.decode_type = "greedy"
        reward, td_final, actions = rollout(env, td, model.policy)
        
        # Calculate coverage statistics
        coverage_mask = td_final["coverage_mask"]
        coverage_pct = (coverage_mask.sum(dim=1).float() / coverage_mask.shape[1] * 100)
        
        mean_reward = reward.mean().item()
        mean_coverage = coverage_pct.mean().item()
        
        print(f"  Mean reward: {mean_reward:.2f}")
        print(f"  Mean coverage: {mean_coverage:.1f}%")
        print(f"  Reward range: [{reward.min().item():.2f}, {reward.max().item():.2f}]")
        
        results.append({
            "distribution": dist,
            "reward": mean_reward,
            "coverage": mean_coverage,
        })
    
    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"{'Distribution':<15} {'Reward':<10} {'Coverage':<10}")
    print("-"*40)
    for r in results:
        print(f"{r['distribution']:<15} {r['reward']:<10.2f} {r['coverage']:<10.1f}%")
    
    # Check generalization gap
    uniform_reward = results[0]["reward"]
    print(f"\nGeneralization Analysis:")
    for r in results[1:]:
        gap = (uniform_reward - r["reward"]) / abs(uniform_reward) * 100
        print(f"  {r['distribution'].capitalize()}: {gap:+.1f}% gap from uniform")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train AttentionModel on MCLP")
    parser.add_argument("--mode", type=str, default="simple",
                       choices=["simple", "cluster", "pregenerated", "evaluate"],
                       help="Training mode")
    
    args = parser.parse_args()
    
    if args.mode == "simple":
        train_mclp_simple()
    elif args.mode == "cluster":
        train_mclp_cluster()
    elif args.mode == "pregenerated":
        train_mclp_with_pregenerated()
    elif args.mode == "evaluate":
        evaluate_mclp()
