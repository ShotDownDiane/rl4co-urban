#!/usr/bin/env python3
"""
Training script for FLP, MCLP, STP, and TSP problems
Modular pipeline with data generation/caching, model training, and evaluation
"""

import os
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import json

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
import matplotlib.pyplot as plt

from rl4co.envs import FLPEnv, MCLPEnv, STPEnv, TSPEnv
from rl4co.envs.graph import MISEnvWrapper, MVCEnvWrapper, MCLEnvWrapper, MCUTEnvWrapper
from rl4co.models.zoo import AttentionModel, DeepACO
from rl4co.utils.trainer import RL4COTrainer
from rl4co.data.utils import save_tensordict_to_npz
from tensordict import TensorDict

# 导入自定义进度回调
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from callbacks.training_progress import DetailedProgressCallback, DeepACOProgressCallback


# ===========================
# Data Module
# ===========================

class DataModule:
    """Data module for generating, saving, and loading datasets"""
    
    def __init__(
        self,
        problem: str,
        data_dir: str = "data",
        num_samples_train: int = 100_000,
        num_samples_val: int = 10_000,
        num_samples_test: int = 10_000,
        **generator_params
    ):
        self.problem = problem.upper()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_samples_train = num_samples_train
        self.num_samples_val = num_samples_val
        self.num_samples_test = num_samples_test
        self.generator_params = generator_params
        self.env_class = self._get_env_class()
        
    def _get_env_class(self):
        """Get environment class based on problem type"""
        env_map = {
            # Original RL4CO environments
            'FLP': FLPEnv,
            'MCLP': MCLPEnv,
            'STP': STPEnv,
            'TSP': TSPEnv,
            # ML4CO-Kit Wrappers (Graph problems)
            'MIS': MISEnvWrapper,
            'MVC': MVCEnvWrapper,
            'MCL': MCLEnvWrapper,
            'MCUT': MCUTEnvWrapper,
        }
        
        if self.problem not in env_map:
            raise ValueError(f"Unknown problem: {self.problem}. Choose from {list(env_map.keys())}")
        
        return env_map[self.problem]
    
    def _get_cache_path(self, split: str) -> Path:
        """Get cache file path for a specific split"""
        # Create unique identifier based on generator params
        params_str = "_".join([f"{k}_{v}" for k, v in sorted(self.generator_params.items())])
        filename = f"{self.problem}_{split}_{params_str}.npz"  # Use .npz format
        return self.data_dir / filename
    
    def _save_dataset(self, dataset, split: str):
        """Save dataset to disk in npz format"""
        cache_path = self._get_cache_path(split)
        print(f"Saving {split} dataset to {cache_path}")
        
        # Handle different dataset types
        if isinstance(dataset, TensorDict):
            # Already a TensorDict (from ML4CO wrappers)
            td = dataset
        elif hasattr(dataset, 'data'):
            # TensorDictDataset (standard RL4CO)
            # dataset.data is a list of dicts, we need to stack them into a TensorDict
            data_list = dataset.data
            
            # Get all keys from the first sample
            keys = list(data_list[0].keys())
            
            # Stack tensors for each key across all samples
            stacked = {}
            for key in keys:
                tensors = [sample[key] for sample in data_list]
                stacked[key] = torch.stack(tensors, dim=0)
            
            # Create a single TensorDict
            td = TensorDict(stacked, batch_size=len(data_list))
        else:
            raise TypeError(f"Unsupported dataset type: {type(dataset)}")
        
        # Save using RL4CO's npz format
        save_tensordict_to_npz(td, str(cache_path), compress=False)
    
    # No need for _load_dataset() - env.load_data() will handle it
    
    def prepare_data(self, force_regenerate: bool = False) -> Dict[str, str]:
        """Prepare train/val/test datasets with caching and return file paths"""
        file_paths = {}
        
        # Create temporary env for data generation
        temp_env = self.env_class(generator_params=self.generator_params)
        
        for split, num_samples in [
            ('train', self.num_samples_train),
            ('val', self.num_samples_val),
            ('test', self.num_samples_test)
        ]:
            cache_path = self._get_cache_path(split)
            file_paths[split] = str(cache_path)
            
            if not cache_path.exists() or force_regenerate:
                # Generate new dataset
                print(f"Generating {split} dataset with {num_samples} samples...")
                
                # Use appropriate method based on env type
                if hasattr(temp_env, 'generate_data'):
                    # For ML4CO wrappers
                    dataset = temp_env.generate_data(batch_size=num_samples)
                else:
                    # For standard RL4CO envs
                    dataset = temp_env.dataset(num_samples)
                
                # Save to cache
                self._save_dataset(dataset, split)
            else:
                print(f"Using cached {split} dataset from {cache_path}")
        
        # Save metadata
        metadata = {
            'problem': self.problem,
            'generator_params': self.generator_params,
            'num_samples_train': self.num_samples_train,
            'num_samples_val': self.num_samples_val,
            'num_samples_test': self.num_samples_test,
        }
        
        metadata_path = self.data_dir / f"{self.problem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Data preparation complete!")
        print(f"  Problem: {self.problem}")
        print(f"  Train samples: {self.num_samples_train}")
        print(f"  Val samples: {self.num_samples_val}")
        print(f"  Test samples: {self.num_samples_test}")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Train file: {file_paths['train']}")
        print(f"  Val file: {file_paths['val']}")
        print(f"  Test file: {file_paths['test']}")
        print(f"{'='*60}\n")
        
        return file_paths
    
    def create_env(self, train_file: str = None, val_file: str = None, test_file: str = None):
        """Create environment with data file paths"""
        # Extract just the filename (not the full path) since env will add data_dir
        import os
        train_name = os.path.basename(train_file) if train_file else None
        val_name = os.path.basename(val_file) if val_file else None
        test_name = os.path.basename(test_file) if test_file else None
        
        return self.env_class(
            generator_params=self.generator_params,
            data_dir=str(self.data_dir),
            train_file=train_name,
            val_file=val_name,
            test_file=test_name
        )


# ===========================
# Model Module
# ===========================

class ModelModule:
    """Model module for building and configuring the RL model"""
    
    def __init__(
        self,
        env,
        model_type: str = "AttentionModel",
        baseline: str = "rollout",
        batch_size: int = 512,
        learning_rate: float = 1e-4,
        **model_kwargs
    ):
        self.env = env
        self.model_type = model_type
        self.baseline = baseline
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_kwargs = model_kwargs
        
    def build_model(self, train_data_size: int, val_data_size: int):
        """Build the RL model"""
        
        # Currently only AttentionModel is supported, but can be extended
        if self.model_type == "AttentionModel":
            model = AttentionModel(
                self.env,
                baseline=self.baseline,
                batch_size=self.batch_size,
                train_data_size=train_data_size,
                val_data_size=val_data_size
            )
        elif self.model_type == "DeepACO":
            model = DeepACO(
                env=self.env,
                train_with_local_search=True,
                policy_kwargs={
                    "aco_kwargs":{
                        "use_local_search": True
                    },
                    "temperature": 1.0,
                    "top_p": 1,
                    "n_ants": 20,
                    "top_k": 50,
                    "embed_dim": 128,
                    "num_layers_graph_encoder": 3,
                },
                batch_size=self.batch_size,
                train_data_size=train_data_size,
                val_data_size=val_data_size
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"\n{'='*60}")
        print(f"Model built: {self.model_type}")
        print(f"  Baseline: {self.baseline}")
        print(f"  Batch size: {self.batch_size}")
        print(f"{'='*60}\n")
        
        return model


# ===========================
# Evaluation Module
# ===========================

class EvaluationModule:
    """Evaluation module with visualization support"""
    
    def __init__(self, log_dir: str = "logs/evaluation"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate(
        self,
        model,
        env,
        test_dataset,
        batch_size: int = 100,
        decode_type: str = "greedy",
        device: str = "cuda"
    ) -> Dict:
        """Evaluate model on test dataset"""
        
        model.to(device)
        model.eval()
        
        # Create dataloader
        dataloader = model._dataloader(test_dataset, batch_size=batch_size, shuffle=False)
        
        all_rewards = []
        all_actions = []
        
        print(f"\n{'='*60}")
        print(f"Evaluating model on test set...")
        print(f"  Test samples: {len(test_dataset)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Decode type: {decode_type}")
        print(f"{'='*60}\n")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                td = env.reset(batch).to(device)
                out = model.policy(td, env, phase="test", decode_type=decode_type)
                
                all_rewards.append(out['reward'].cpu())
                all_actions.append(out['actions'].cpu())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {(batch_idx + 1) * batch_size}/{len(test_dataset)} samples")
        
        # Aggregate results
        all_rewards = torch.cat(all_rewards)
        all_actions = torch.cat(all_actions)
        
        results = {
            'rewards': all_rewards,
            'actions': all_actions,
            'mean_reward': -all_rewards.mean().item(),
            'std_reward': all_rewards.std().item(),
            'min_reward': -all_rewards.max().item(),
            'max_reward': -all_rewards.min().item(),
        }
        
        print(f"\n{'='*60}")
        print(f"Evaluation Results:")
        print(f"  Mean cost: {results['mean_reward']:.4f}")
        print(f"  Std: {results['std_reward']:.4f}")
        print(f"  Min cost: {results['min_reward']:.4f}")
        print(f"  Max cost: {results['max_reward']:.4f}")
        print(f"{'='*60}\n")
        
        return results
    
    def visualize_solutions(
        self,
        env,
        test_dataset,
        actions,
        num_samples: int = 5,
        save_path: Optional[str] = None
    ):
        """Visualize solution examples"""
        
        num_samples = min(num_samples, len(test_dataset))
        
        fig, axes = plt.subplots(1, num_samples, figsize=(5 * num_samples, 5))
        if num_samples == 1:
            axes = [axes]
        
        for i in range(num_samples):
            # Get sample and convert dict to TensorDict if needed
            sample = test_dataset[i]
            if isinstance(sample, dict):
                sample = TensorDict(sample, batch_size=[])
            td = env.reset(sample)
            env.render(td, actions[i], ax=axes[i])
            axes[i].set_title(f"Sample {i+1}")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.close()
    
    def save_results(self, results: Dict, filename: str = "test_results.json"):
        """Save evaluation results to file"""
        
        # Convert tensors to lists for JSON serialization
        results_serializable = {
            'mean_reward': results['mean_reward'],
            'std_reward': results['std_reward'],
            'min_reward': results['min_reward'],
            'max_reward': results['max_reward'],
        }
        
        save_path = self.log_dir / filename
        with open(save_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Results saved to {save_path}")


# ===========================
# Training Pipeline
# ===========================

def train_pipeline(args):
    """Main training pipeline"""
    
    print(f"\n{'#'*60}")
    print(f"# Starting Training Pipeline")
    print(f"# Problem: {args.problem}")
    print(f"{'#'*60}\n")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # ===========================
    # 1. Data Module
    # ===========================
    
    # Prepare generator params based on problem
    generator_params = {}
    if args.problem.upper() in ['TSP']:
        generator_params['num_loc'] = args.num_loc
    elif args.problem.upper() in ['FLP', 'MCLP', 'STP']:
        generator_params['num_loc'] = args.num_loc
        if hasattr(args, 'num_facilities') and args.num_facilities:
            generator_params['num_facilities'] = args.num_facilities
    elif args.problem.upper() in ['MIS', 'MVC', 'MCL', 'MCUT']:
        # ML4CO-Kit graph problems
        generator_params['num_nodes'] = getattr(args, 'num_loc', 50)
        generator_params['graph_type'] = getattr(args, 'graph_type', 'erdos_renyi')
        generator_params['edge_prob'] = getattr(args, 'edge_prob', 0.15)
    
    data_module = DataModule(
        problem=args.problem,
        data_dir=args.data_dir,
        num_samples_train=args.train_size,
        num_samples_val=args.val_size,
        num_samples_test=args.test_size,
        **generator_params
    )
    
    # Prepare datasets (generate or load from cache) - returns file paths
    file_paths = data_module.prepare_data(force_regenerate=args.force_regenerate)
    
    # Create env with file paths
    env = data_module.create_env(
        train_file=file_paths['train'],
        val_file=file_paths['val'],
        test_file=file_paths['test']
    )
    
    # ===========================
    # 2. Model Module
    # ===========================
    
    model_module = ModelModule(
        env=env,
        model_type=args.model_type,
        baseline=args.baseline,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    
    model = model_module.build_model(
        train_data_size=args.train_size,
        val_data_size=args.val_size
    )
    
    # ===========================
    # 3. Training Setup
    # ===========================
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.checkpoint_dir}/{args.problem}",
        filename="epoch_{epoch:03d}",
        save_top_k=3,
        save_last=True,
        monitor="val/reward",
        mode="max",
        verbose=True,  # 输出checkpoint保存信息
    )
    
    # 进度条回调
    progress_bar = RichProgressBar(
        leave=True,  # 保留进度条
    )
    
    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    
    rich_model_summary = RichModelSummary(max_depth=3)
    
    # 组装callbacks
    callbacks = [checkpoint_callback, rich_model_summary, progress_bar, lr_monitor]
    
    # Logger - Using TensorBoard and CSV
    tb_logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=f"{args.problem}_{args.model_type}",
        default_hp_metric=False
    )
    
    # 使用多个logger
    logger = [tb_logger]
    
    # Trainer
    trainer = RL4COTrainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        log_every_n_steps=10,  # 每10步记录一次
        enable_progress_bar=True,  # 启用进度条
        enable_model_summary=True,  # 启用模型摘要
    )
    
    # ===========================
    # 4. Training
    # ===========================
    
    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs...")
    print(f"Train file: {file_paths['train']}")
    print(f"Val file: {file_paths['val']}")
    print(f"{'='*60}\n")
    
    # Let the model handle dataloader creation internally
    trainer.fit(model)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}\n")
    
    # ===========================
    # 5. Evaluation Module
    # ===========================
    
    if not args.skip_evaluation:
        eval_module = EvaluationModule(
            log_dir=f"{args.log_dir}/{args.problem}_evaluation"
        )
        
        # Load test dataset for evaluation
        test_dataset = env.dataset(phase='test')
        
        # Evaluate on test set
        results = eval_module.evaluate(
            model=model,
            env=env,
            test_dataset=test_dataset,
            batch_size=args.eval_batch_size,
            device=device
        )
        
        # Save results
        eval_module.save_results(
            results,
            filename=f"{args.problem}_test_results.json"
        )
        
        # Visualize some solutions
        if args.visualize:
            vis_path = eval_module.log_dir / f"{args.problem}_solutions.png"
            eval_module.visualize_solutions(
                env=env,
                test_dataset=test_dataset,
                actions=results['actions'],
                num_samples=min(5, args.test_size),
                save_path=str(vis_path)
            )
    
    print(f"\n{'#'*60}")
    print(f"# Pipeline Complete!")
    print(f"# Checkpoints saved to: {args.checkpoint_dir}/{args.problem}")
    print(f"# Logs saved to: {args.log_dir}")
    print(f"{'#'*60}\n")


# ===========================
# Main Entry Point
# ===========================

def main():
    parser = argparse.ArgumentParser(
        description="Train RL models on combinatorial optimization problems"
    )
    
    # Problem settings
    parser.add_argument(
        '--problem',
        type=str,
        required=True,
        choices=['FLP', 'MCLP', 'STP', 'TSP', 'MIS', 'MVC', 'MCL', 'MCUT'],
        help='Problem type to solve'
    )
    parser.add_argument(
        '--num-loc',
        type=int,
        default=20,
        help='Number of locations/nodes'
    )
    parser.add_argument(
        '--num-facilities',
        type=int,
        default=None,
        help='Number of facilities (for FLP, MCLP)'
    )
    parser.add_argument(
        '--graph-type',
        type=str,
        default='erdos_renyi',
        choices=['erdos_renyi', 'barabasi_albert', 'watts_strogatz'],
        help='Graph type for graph problems (MIS, MVC, MCL, MCUT)'
    )
    parser.add_argument(
        '--edge-prob',
        type=float,
        default=0.15,
        help='Edge probability for Erdős-Rényi graph (for graph problems)'
    )
    
    # Data settings
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory for data storage'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=512,
        help='Batch size for training'
    )
    parser.add_argument(
        '--train-size',
        type=int,
        default=100_000,
        help='Training dataset size'
    )
    parser.add_argument(
        '--val-size',
        type=int,
        default=10_000,
        help='Validation dataset size'
    )
    parser.add_argument(
        '--test-size',
        type=int,
        default=10_000,
        help='Test dataset size'
    )
    parser.add_argument(
        '--force-regenerate',
        action='store_true',
        help='Force regenerate datasets even if cached'
    )
    
    # Model settings
    parser.add_argument(
        '--model-type',
        type=str,
        default='DeepACO',
        help='Model architecture'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        default='rollout',
        choices=['rollout', 'exponential', 'critic','no'],
        help='Baseline type for REINFORCE'
    )
    parser.add_argument(
        '--embed-dim',
        type=int,
        default=128,
        help='Embedding dimension'
    )
    parser.add_argument(
        '--num-encoder-layers',
        type=int,
        default=3,
        help='Number of encoder layers'
    )
    parser.add_argument(
        '--num-heads',
        type=int,
        default=8,
        help='Number of attention heads'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    
    # Training settings
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory for model checkpoints'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Directory for TensorBoard logs'
    )
    
    # Evaluation settings
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip evaluation after training'
    )
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=100,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualizations of solutions'
    )
    
    args = parser.parse_args()
    
    # Run training pipeline
    train_pipeline(args)


if __name__ == "__main__":
    main()
