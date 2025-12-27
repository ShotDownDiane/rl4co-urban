"""
Simple training script for FLP (Facility Location Problem)
Based on RL4CO's AttentionModel with different decoding strategies
"""

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from rl4co.envs.graph import FLPEnv
from rl4co.models.zoo import AttentionModel
from rl4co.models.zoo.am.policy import AttentionModelPolicy


def train_flp_simple():
    """Simple training example for FLP"""
    
    print("="*80)
    print("Training AttentionModel on FLP")
    print("="*80)
    
    # ============================================
    # 1. Environment Setup
    # ============================================
    print("\n1. Setting up environment...")
    
    env = FLPEnv(generator_params={
        "num_loc": 50,      # Number of candidate locations
        "to_choose": 5,      # Number of facilities to select
    })
    
    print(f"✓ Environment: {env.name}")
    print(f"  - Candidate locations: 50")
    print(f"  - Facilities to select: 5")
    
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
    
    # ============================================
    # 3. Model Setup (RL Algorithm)
    # ============================================
    print("\n3. Setting up model (REINFORCE with rollout baseline)...")
    
    model = AttentionModel(
        env,
        policy=policy,
        baseline="rollout",          # Baseline: rollout (greedy rollout)
        batch_size=512,              # Training batch size
        val_batch_size=64,           # Validation batch size
        test_batch_size=64,          # Test batch size
        train_data_size=10_000,      # Number of training instances (smaller for demo)
        val_data_size=1_000,         # Number of validation instances
        test_data_size=1_000,        # Number of test instances
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
    print(f"  - Training instances: 10,000")
    print(f"  - Validation instances: 1,000")
    
    # ============================================
    # 4. Training Setup
    # ============================================
    print("\n4. Setting up trainer...")
    
    # Checkpoint callback - saves best model
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/flp",
        filename="flp-{epoch:02d}-{val_reward:.2f}",
        monitor="val/reward",
        mode="max",  # FLP: maximize reward (minimize negative distance)
        save_top_k=3,
        verbose=True,
    )
    
    # TensorBoard logger
    logger = TensorBoardLogger(
        save_dir="logs",
        name="flp",
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=10,               # Number of epochs (increase for real training)
        accelerator="auto",          # Automatically select GPU/CPU
        devices=1,                   # Number of devices
        logger=logger,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,       # Gradient clipping
        log_every_n_steps=10,        # Log frequency
        val_check_interval=0.5,      # Validate twice per epoch
    )
    
    print(f"✓ Trainer configured")
    print(f"  - Max epochs: 10")
    print(f"  - Device: {trainer.accelerator}")
    print(f"  - Checkpoints: checkpoints/flp/")
    print(f"  - Logs: logs/flp/")
    
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
    print(f"✓ TensorBoard logs: logs/flp/")
    print(f"\nTo view TensorBoard:")
    print(f"  tensorboard --logdir logs/flp/")


def train_flp_advanced():
    """Advanced training example with multiple decoding strategies"""
    
    print("="*80)
    print("Advanced Training: Multiple Decoding Strategies")
    print("="*80)
    
    # Environment
    env = FLPEnv(generator_params={
        "num_loc": 100,     # Larger problem
        "to_choose": 10,
    })
    
    # Policy
    policy = AttentionModelPolicy(
        env_name=env.name,
        embed_dim=128,
        num_encoder_layers=3,
        num_heads=8,
    )
    
    # Model with different decoding strategies
    model = AttentionModel(
        env,
        policy=policy,
        baseline="rollout",
        batch_size=256,              # Larger problems = smaller batch
        val_batch_size=32,
        test_batch_size=32,
        train_data_size=50_000,      # More training data
        val_data_size=1_000,
        test_data_size=1_000,
        optimizer_kwargs={"lr": 1e-4},
        policy_kwargs={
            "train_decode_type": "sampling",     # Training: sampling
            "val_decode_type": "greedy",         # Validation: greedy
            "test_decode_type": "beam_search",   # Test: beam search (better quality)
            "beam_width": 5,                     # Beam width for beam search
        }
    )
    
    print(f"\n✓ Configuration:")
    print(f"  - Problem size: 100 locations, select 10")
    print(f"  - Training: sampling")
    print(f"  - Validation: greedy")
    print(f"  - Test: beam search (width=5)")
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/flp_large",
        filename="flp-{epoch:02d}-{val_reward:.2f}",
        monitor="val/reward",
        mode="max",
        save_top_k=3,
    )
    
    # Logger
    logger = TensorBoardLogger("logs", name="flp_large")
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=50,              # More epochs for larger problem
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,
        val_check_interval=0.5,
    )
    
    print(f"\n✓ Training for 50 epochs...")
    print(f"✓ Checkpoints: checkpoints/flp_large/")
    print(f"✓ Logs: logs/flp_large/")
    
    # Train
    trainer.fit(model)
    
    # Test with beam search
    print("\n" + "="*80)
    print("Testing with beam search decoding...")
    print("="*80)
    trainer.test(model)
    
    print("\n✓ Training complete!")


def train_flp_from_pretrained():
    """Example of loading a checkpoint and continuing training"""
    
    print("="*80)
    print("Training from Checkpoint")
    print("="*80)
    
    # Load checkpoint path (replace with actual path)
    checkpoint_path = "checkpoints/flp/flp-epoch=09-val_reward=-3.45.ckpt"
    
    print(f"\nLoading from: {checkpoint_path}")
    
    # Load model from checkpoint
    model = AttentionModel.load_from_checkpoint(checkpoint_path)
    
    print(f"✓ Model loaded successfully")
    
    # Continue training
    trainer = pl.Trainer(
        max_epochs=20,  # Train for additional epochs
        accelerator="auto",
        devices=1,
    )
    
    print(f"\nContinuing training for 10 more epochs...")
    trainer.fit(model)
    
    print("\n✓ Fine-tuning complete!")


def evaluate_flp():
    """Evaluate a trained model"""
    
    print("="*80)
    print("Evaluating Trained Model")
    print("="*80)
    
    # Load checkpoint
    checkpoint_path = "checkpoints/flp/best.ckpt"
    
    try:
        model = AttentionModel.load_from_checkpoint(checkpoint_path)
        print(f"✓ Model loaded from: {checkpoint_path}")
    except FileNotFoundError:
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print("  Please train a model first using train_flp_simple()")
        return
    
    # Create environment
    env = FLPEnv(generator_params={"num_loc": 50, "to_choose": 5})
    
    # Generate test instances
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print("\nGenerating test instances...")
    td = env.reset(batch_size=[100]).to(device)
    
    # Evaluate with different decoding strategies
    print("\n" + "="*80)
    print("Comparing Decoding Strategies")
    print("="*80)
    
    from rl4co.utils.decoding import rollout
    
    # Greedy decoding
    print("\n1. Greedy decoding...")
    model.policy.decode_type = "greedy"
    reward_greedy, _, _ = rollout(env, td, model.policy)
    print(f"   Mean reward: {reward_greedy.mean().item():.4f}")
    
    # Sampling (multiple samples)
    print("\n2. Sampling (10 samples)...")
    model.policy.decode_type = "sampling"
    rewards_sampling = []
    for _ in range(10):
        reward, _, _ = rollout(env, td, model.policy)
        rewards_sampling.append(reward)
    reward_sampling = torch.stack(rewards_sampling).max(dim=0)[0]  # Best of 10
    print(f"   Mean reward (best of 10): {reward_sampling.mean().item():.4f}")
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Greedy:       {reward_greedy.mean().item():.4f}")
    print(f"Sampling x10: {reward_sampling.mean().item():.4f}")
    print(f"Improvement:  {((reward_sampling.mean() - reward_greedy.mean()) / reward_greedy.mean().abs() * 100).item():.2f}%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train AttentionModel on FLP")
    parser.add_argument("--mode", type=str, default="simple",
                       choices=["simple", "advanced", "from_ckpt", "evaluate"],
                       help="Training mode")
    
    args = parser.parse_args()
    
    if args.mode == "simple":
        train_flp_simple()
    elif args.mode == "advanced":
        train_flp_advanced()
    elif args.mode == "from_ckpt":
        train_flp_from_pretrained()
    elif args.mode == "evaluate":
        evaluate_flp()
