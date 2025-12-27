"""
Simple training script for TSP (Traveling Salesman Problem)
Based on RL4CO's AttentionModel with different decoding strategies

TSP is the classic problem that AttentionModel was originally designed for!
"""

import torch
import argparse
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from rl4co.envs.routing import TSPEnv
from rl4co.models.zoo import AttentionModel
from rl4co.models.zoo.am.policy import AttentionModelPolicy


def train_tsp_simple():
    """Simple training example for TSP - 20 cities"""
    
    print("="*80)
    print("Training AttentionModel on TSP (20 cities)")
    print("="*80)
    
    # ============================================
    # 1. Environment Setup
    # ============================================
    print("\n1. Setting up environment...")
    
    env = TSPEnv(generator_params={
        "num_loc": 20,      # Number of cities
    })
    
    print(f"✓ Environment: {env.name}")
    print(f"  - Number of cities: 20")
    print(f"  - Objective: Minimize tour length")
    
    # ============================================
    # 2. Policy Setup (Neural Network)
    # ============================================
    print("\n2. Setting up policy (encoder-decoder architecture)...")
    
    policy = AttentionModelPolicy(
        env_name=env.name,
        embed_dim=128,           # Embedding dimension
        num_encoder_layers=3,    # Number of encoder layers
        num_heads=8,             # Number of attention heads
    )
    
    print(f"✓ Policy: AttentionModelPolicy")
    print(f"  - Embedding dimension: 128")
    print(f"  - Encoder layers: 3")
    print(f"  - Attention heads: 8")
    print(f"  - Parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # ============================================
    # 3. Model Setup (RL Algorithm)
    # ============================================
    print("\n3. Setting up model (REINFORCE with rollout baseline)...")
    
    model = AttentionModel(
        env,
        policy=policy,
        baseline="rollout",          # Baseline: rollout (greedy rollout)
        batch_size=512,              # Training batch size
        val_batch_size=128,          # Validation batch size
        test_batch_size=128,         # Test batch size
        train_data_size=100_000,     # Number of training instances
        val_data_size=10_000,        # Number of validation instances
        test_data_size=10_000,       # Number of test instances
        optimizer_kwargs={"lr": 1e-4},  # Learning rate
        policy_kwargs={
            "train_decode_type": "sampling",    # Training: sample from distribution
            "val_decode_type": "greedy",        # Validation: greedy decoding
            "test_decode_type": "greedy",       # Test: greedy decoding
        }
    )
    
    print(f"✓ Model: AttentionModel")
    print(f"  - Algorithm: REINFORCE")
    print(f"  - Baseline: rollout")
    print(f"  - Batch size: 512")
    print(f"  - Learning rate: 1e-4")
    print(f"  - Training instances: 100,000")
    print(f"  - Validation instances: 10,000")
    
    # ============================================
    # 4. Training Setup
    # ============================================
    print("\n4. Setting up trainer...")
    
    # Checkpoint callback - saves best model
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/tsp",
        filename="tsp20-{epoch:02d}-{val_reward:.2f}",
        monitor="val/reward",
        mode="max",  # TSP: maximize reward (minimize negative tour length)
        save_top_k=3,
        verbose=True,
    )
    
    # TensorBoard logger
    logger = TensorBoardLogger("logs", name="tsp")
    
    # PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=100,              # Train for 100 epochs
        callbacks=[checkpoint_callback],
        logger=logger,
        gradient_clip_val=1.0,      # Gradient clipping
        accelerator="auto",         # Automatically select GPU/CPU
        devices=1,
        precision="16-mixed",       # Mixed precision training for speed
    )
    
    print(f"✓ Trainer configured")
    print(f"  - Max epochs: 100")
    print(f"  - Gradient clipping: 1.0")
    print(f"  - Precision: 16-bit mixed")
    
    # ============================================
    # 5. Training
    # ============================================
    print("\n5. Starting training...")
    print("-" * 80)
    
    trainer.fit(model)
    
    # ============================================
    # 6. Testing
    # ============================================
    print("\n6. Testing best model...")
    print("-" * 80)
    
    trainer.test(model)
    
    print("\n" + "="*80)
    print("✓ Training completed!")
    print(f"  - Best model saved in: checkpoints/tsp/")
    print(f"  - Logs saved in: logs/tsp/")
    print("="*80)


def train_tsp_larger():
    """Training example for larger TSP - 50 cities"""
    
    print("="*80)
    print("Training AttentionModel on TSP (50 cities)")
    print("="*80)
    
    print("\n1. Setting up environment...")
    env = TSPEnv(generator_params={"num_loc": 50})
    
    print(f"✓ Environment: {env.name}")
    print(f"  - Number of cities: 50")
    
    print("\n2. Setting up policy...")
    policy = AttentionModelPolicy(
        env_name=env.name,
        embed_dim=128,
        num_encoder_layers=3,
        num_heads=8,
    )
    
    print(f"✓ Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    print("\n3. Setting up model...")
    model = AttentionModel(
        env,
        policy=policy,
        baseline="rollout",
        batch_size=256,              # Smaller batch for larger problems
        val_batch_size=64,
        test_batch_size=64,
        train_data_size=100_000,
        val_data_size=10_000,
        test_data_size=10_000,
        optimizer_kwargs={"lr": 1e-4},
        policy_kwargs={
            "train_decode_type": "sampling",
            "val_decode_type": "greedy",
            "test_decode_type": "greedy",
        }
    )
    
    print("\n4. Setting up trainer...")
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/tsp",
        filename="tsp50-{epoch:02d}-{val_reward:.2f}",
        monitor="val/reward",
        mode="max",
        save_top_k=3,
        verbose=True,
    )
    
    logger = TensorBoardLogger("logs", name="tsp_50")
    
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[checkpoint_callback],
        logger=logger,
        gradient_clip_val=1.0,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
    )
    
    print("\n5. Starting training...")
    print("-" * 80)
    trainer.fit(model)
    
    print("\n6. Testing...")
    print("-" * 80)
    trainer.test(model)
    
    print("\n" + "="*80)
    print("✓ Training completed!")
    print("="*80)


def train_tsp_beam_search():
    """Training with beam search decoding"""
    
    print("="*80)
    print("Training TSP with Beam Search (20 cities)")
    print("="*80)
    
    print("\n1. Setting up environment...")
    env = TSPEnv(generator_params={"num_loc": 20})
    
    print("\n2. Setting up policy...")
    policy = AttentionModelPolicy(
        env_name=env.name,
        embed_dim=128,
        num_encoder_layers=3,
        num_heads=8,
    )
    
    print("\n3. Setting up model with beam search...")
    model = AttentionModel(
        env,
        policy=policy,
        baseline="rollout",
        batch_size=256,
        val_batch_size=64,
        test_batch_size=64,
        train_data_size=100_000,
        val_data_size=10_000,
        test_data_size=10_000,
        optimizer_kwargs={"lr": 1e-4},
        policy_kwargs={
            "train_decode_type": "sampling",
            "val_decode_type": "greedy",
            "test_decode_type": "greedy",  # Will use beam search after loading
        }
    )
    
    print(f"✓ Model configured")
    print(f"  - Training: sampling")
    print(f"  - Validation: greedy")
    print(f"  - Test: beam search (width=5)")
    
    print("\n4. Setting up trainer...")
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/tsp",
        filename="tsp20_beam-{epoch:02d}-{val_reward:.2f}",
        monitor="val/reward",
        mode="max",
        save_top_k=3,
        verbose=True,
    )
    
    logger = TensorBoardLogger("logs", name="tsp_beam")
    
    trainer = pl.Trainer(
        max_epochs=50,
        callbacks=[checkpoint_callback],
        logger=logger,
        gradient_clip_val=1.0,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
    )
    
    print("\n5. Training...")
    print("-" * 80)
    trainer.fit(model)
    
    # Change to beam search for testing
    print("\n6. Testing with beam search...")
    print("-" * 80)
    model.policy.test_decode_type = "beam_search"
    model.policy.beam_width = 5
    trainer.test(model)
    
    print("\n" + "="*80)
    print("✓ Training completed!")
    print("="*80)


def evaluate_tsp():
    """Evaluate a trained TSP model with different decoding strategies"""
    
    print("="*80)
    print("Evaluating TSP Model with Different Decoding Strategies")
    print("="*80)
    
    # Load checkpoint
    checkpoint_path = "checkpoints/tsp/best_model.ckpt"
    
    print("\n1. Loading checkpoint...")
    try:
        model = AttentionModel.load_from_checkpoint(checkpoint_path)
        print(f"✓ Model loaded from: {checkpoint_path}")
    except:
        print("✗ Checkpoint not found. Please train a model first.")
        print("  Run: python test_tsp.py --mode simple")
        return
    
    # Setup environment for testing
    print("\n2. Setting up test environment...")
    env = TSPEnv(generator_params={"num_loc": 20})
    td = env.reset(batch_size=[100])  # Generate 100 test instances
    
    print(f"✓ Generated 100 test instances (20 cities each)")
    
    # Test different decoding strategies
    strategies = [
        ("greedy", {}),
        ("sampling", {}),
        ("beam_search", {"beam_width": 5}),
        ("beam_search", {"beam_width": 10}),
    ]
    
    print("\n3. Testing different decoding strategies...")
    print("-" * 80)
    
    results = []
    
    for decode_type, kwargs in strategies:
        model.policy.test_decode_type = decode_type
        for key, value in kwargs.items():
            setattr(model.policy, key, value)
        
        # Run inference
        with torch.no_grad():
            output = model.policy(td, env, decode_type=decode_type, **kwargs)
        
        # Calculate rewards
        rewards = env.get_reward(td, output["actions"])
        mean_reward = rewards.mean().item()
        std_reward = rewards.std().item()
        
        # For TSP, reward is negative tour length
        mean_length = -mean_reward
        
        strategy_name = decode_type
        if kwargs:
            strategy_name += f" ({', '.join(f'{k}={v}' for k, v in kwargs.items())})"
        
        print(f"\n{strategy_name}:")
        print(f"  - Mean tour length: {mean_length:.4f}")
        print(f"  - Std: {std_reward:.4f}")
        
        results.append({
            "strategy": strategy_name,
            "mean_length": mean_length,
            "std": std_reward,
        })
    
    # Print summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    # Sort by performance (lower tour length is better)
    results.sort(key=lambda x: x["mean_length"])
    
    print(f"\n{'Strategy':<30} {'Mean Length':>15} {'Std':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['strategy']:<30} {r['mean_length']:>15.4f} {r['std']:>10.4f}")
    
    print("\n✓ Evaluation completed!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Train AttentionModel on TSP")
    parser.add_argument(
        "--mode",
        type=str,
        default="simple",
        choices=["simple", "larger", "beam", "evaluate"],
        help="Training mode: simple (20 cities), larger (50 cities), beam (beam search), evaluate"
    )
    
    args = parser.parse_args()
    
    if args.mode == "simple":
        train_tsp_simple()
    elif args.mode == "larger":
        train_tsp_larger()
    elif args.mode == "beam":
        train_tsp_beam_search()
    elif args.mode == "evaluate":
        evaluate_tsp()
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
