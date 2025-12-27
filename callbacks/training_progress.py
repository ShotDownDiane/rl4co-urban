"""
Custom callback for detailed training progress output
"""
import time
from lightning.pytorch.callbacks import Callback
import torch


class DetailedProgressCallback(Callback):
    """显示详细的训练进度信息"""
    
    def __init__(self, print_every_n_batches=1, print_every_n_epochs=1):
        super().__init__()
        self.print_every_n_batches = print_every_n_batches
        self.print_every_n_epochs = print_every_n_epochs
        self.epoch_start_time = None
        self.batch_start_time = None
        self.batch_times = []
        self.total_batches = 0
        
    def on_train_start(self, trainer, pl_module):
        print("\n" + "="*80)
        print("🚀 TRAINING STARTED")
        print("="*80)
        print(f"📊 Total Epochs: {trainer.max_epochs}")
        print(f"🎯 Model: {pl_module.__class__.__name__}")
        if hasattr(pl_module, 'policy'):
            if hasattr(pl_module.policy, 'n_ants'):
                print(f"🐜 Ants (train): {pl_module.policy.n_ants.get('train', 'N/A')}")
                print(f"🐜 Ants (val): {pl_module.policy.n_ants.get('val', 'N/A')}")
            if hasattr(pl_module.policy, 'n_iterations'):
                print(f"🔄 Iterations (train): {pl_module.policy.n_iterations.get('train', 'N/A')}")
        print("="*80 + "\n")
    
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        self.batch_times = []
        
        if trainer.current_epoch % self.print_every_n_epochs == 0:
            print(f"\n{'='*80}")
            print(f"📅 Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} Started")
            print(f"{'='*80}")
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.batch_start_time = time.time()
        self.total_batches = trainer.num_training_batches
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        batch_time = time.time() - self.batch_start_time
        self.batch_times.append(batch_time)
        
        if batch_idx % self.print_every_n_batches == 0:
            # 计算平均batch时间
            avg_batch_time = sum(self.batch_times[-10:]) / min(len(self.batch_times[-10:]), 10)
            
            # 计算剩余时间
            remaining_batches = self.total_batches - batch_idx - 1
            eta_seconds = remaining_batches * avg_batch_time
            eta_mins = eta_seconds / 60
            
            # 获取metrics
            loss = outputs.get('loss', torch.tensor(float('nan')))
            if isinstance(loss, torch.Tensor):
                loss_val = loss.item()
            else:
                loss_val = float(loss)
            
            # 进度百分比
            progress = (batch_idx + 1) / self.total_batches * 100
            
            # 打印信息
            print(f"⚡ Epoch {trainer.current_epoch + 1} | "
                  f"Batch {batch_idx + 1}/{self.total_batches} ({progress:.1f}%) | "
                  f"Loss: {loss_val:.4f} | "
                  f"Time: {batch_time:.2f}s | "
                  f"Avg: {avg_batch_time:.2f}s/batch | "
                  f"ETA: {eta_mins:.1f}min")
    
    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.epoch_start_time
        epoch_mins = epoch_time / 60
        
        if trainer.current_epoch % self.print_every_n_epochs == 0:
            # 获取训练metrics
            metrics = trainer.callback_metrics
            train_reward = metrics.get('train/reward', torch.tensor(float('nan')))
            train_loss = metrics.get('train/loss', torch.tensor(float('nan')))
            
            if isinstance(train_reward, torch.Tensor):
                train_reward = train_reward.item()
            if isinstance(train_loss, torch.Tensor):
                train_loss = train_loss.item()
            
            print(f"\n{'─'*80}")
            print(f"✅ Epoch {trainer.current_epoch + 1} Completed")
            print(f"   ⏱️  Time: {epoch_mins:.2f} minutes")
            print(f"   📉 Train Loss: {train_loss:.4f}")
            print(f"   🎯 Train Reward: {train_reward:.4f}")
            
            # 计算平均batch时间
            avg_time = sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0
            print(f"   ⚡ Avg Batch Time: {avg_time:.2f}s")
            print(f"{'─'*80}\n")
    
    def on_validation_start(self, trainer, pl_module):
        print(f"\n{'─'*80}")
        print(f"🔍 Validation Started (Epoch {trainer.current_epoch + 1})")
        print(f"{'─'*80}")
        self.val_start_time = time.time()
    
    def on_validation_epoch_end(self, trainer, pl_module):
        val_time = time.time() - self.val_start_time
        
        # 获取验证metrics
        metrics = trainer.callback_metrics
        val_reward = metrics.get('val/reward', torch.tensor(float('nan')))
        
        if isinstance(val_reward, torch.Tensor):
            val_reward = val_reward.item()
        
        print(f"\n{'─'*80}")
        print(f"✅ Validation Completed")
        print(f"   ⏱️  Time: {val_time:.2f}s")
        print(f"   🎯 Val Reward: {val_reward:.4f}")
        print(f"{'─'*80}\n")
    
    def on_train_end(self, trainer, pl_module):
        print("\n" + "="*80)
        print("🎉 TRAINING COMPLETED")
        print("="*80)
        print(f"✅ Total Epochs: {trainer.current_epoch + 1}")
        print(f"✅ Total Batches: {trainer.global_step}")
        
        # 最终metrics
        metrics = trainer.callback_metrics
        final_train_reward = metrics.get('train/reward', torch.tensor(float('nan')))
        final_val_reward = metrics.get('val/reward', torch.tensor(float('nan')))
        
        if isinstance(final_train_reward, torch.Tensor):
            final_train_reward = final_train_reward.item()
        if isinstance(final_val_reward, torch.Tensor):
            final_val_reward = final_val_reward.item()
        
        print(f"📊 Final Train Reward: {final_train_reward:.4f}")
        print(f"📊 Final Val Reward: {final_val_reward:.4f}")
        print("="*80 + "\n")


class DeepACOProgressCallback(Callback):
    """专门为DeepACO设计的进度回调，显示ACO特定信息"""
    
    def __init__(self, print_every_n_batches=5):
        super().__init__()
        self.print_every_n_batches = print_every_n_batches
        self.batch_start_time = None
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx % self.print_every_n_batches == 0:
            self.batch_start_time = time.time()
            print(f"\n{'─'*80}")
            print(f"🐜 DeepACO Batch {batch_idx + 1} Processing...")
            
            # 显示ACO参数
            if hasattr(pl_module, 'policy'):
                policy = pl_module.policy
                if hasattr(policy, 'n_ants'):
                    print(f"   🐜 Number of Ants: {policy.n_ants.get('train', 'N/A')}")
                if hasattr(policy, 'n_iterations'):
                    print(f"   🔄 ACO Iterations: {policy.n_iterations.get('train', 'N/A')}")
                if hasattr(pl_module, 'train_with_local_search'):
                    print(f"   🔧 Local Search: {'Enabled' if pl_module.train_with_local_search else 'Disabled'}")
            
            print(f"   ⏰ Start Time: {time.strftime('%H:%M:%S')}")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.print_every_n_batches == 0:
            elapsed = time.time() - self.batch_start_time
            
            # 获取详细metrics
            loss = outputs.get('loss', torch.tensor(float('nan')))
            if isinstance(loss, torch.Tensor):
                loss_val = loss.item()
            else:
                loss_val = float(loss)
            
            print(f"   ✅ Batch Completed in {elapsed:.2f}s")
            print(f"   📉 Loss: {loss_val:.4f}")
            
            # 如果有reward信息
            if 'train/reward' in trainer.callback_metrics:
                reward = trainer.callback_metrics['train/reward']
                if isinstance(reward, torch.Tensor):
                    reward = reward.item()
                print(f"   🎯 Reward: {reward:.4f}")
            
            print(f"{'─'*80}\n")
