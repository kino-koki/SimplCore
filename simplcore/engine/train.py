# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore Training Toolchain - 青小团自研训练工具链
包含DataLoader、ModelCheckpoint、Logger、Trainer等核心组件
"""
from __future__ import annotations

import numpy as np
import os
import json
import time
import pickle
import math
from typing import Optional, List, Dict, Any, Callable, Tuple
from abc import ABC, abstractmethod


class DataLoader:
    """数据加载器"""
    
    def __init__(
        self,
        dataset: Any,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self._indices = list(range(len(dataset)))
        self._current_idx = 0
    
    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)
    
    def __iter__(self):
        self._current_idx = 0
        if self.shuffle:
            np.random.shuffle(self._indices)
        return self
    
    def __next__(self):
        if self._current_idx >= len(self.dataset):
            raise StopIteration
        
        end_idx = min(self._current_idx + self.batch_size, len(self.dataset))
        
        if self.drop_last and end_idx - self._current_idx < self.batch_size:
            raise StopIteration
        
        batch_indices = self._indices[self._current_idx:end_idx]
        batch_data = [self.dataset[i] for i in batch_indices]
        
        if isinstance(batch_data[0], tuple):
            batch = tuple(np.array([item[i] for item in batch_data]) for i in range(len(batch_data[0])))
        else:
            batch = np.array(batch_data)
        
        self._current_idx = end_idx
        return batch


class Dataset(ABC):
    """数据集抽象基类"""
    
    @abstractmethod
    def __len__(self) -> int:
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        pass


class TensorDataset(Dataset):
    """张量数据集"""
    
    def __init__(self, *tensors: np.ndarray):
        self.tensors = tensors
        assert all(len(t) == len(tensors[0]) for t in tensors)
    
    def __len__(self) -> int:
        return len(self.tensors[0])
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, ...]:
        return tuple(t[idx] for t in self.tensors)


class ModelCheckpoint:
    """模型检查点管理器"""
    
    def __init__(
        self,
        save_dir: str = "./checkpoints",
        save_best_only: bool = True,
        monitor: str = "val_loss",
        mode: str = "min",
        save_freq: int = 1,
        max_checkpoints: int = 5,
    ):
        self.save_dir = save_dir
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.save_freq = save_freq
        self.max_checkpoints = max_checkpoints
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.checkpoints = []
        
        os.makedirs(save_dir, exist_ok=True)
    
    def save(self, model: Any, epoch: int, metrics: Dict[str, float]) -> None:
        """保存检查点"""
        if epoch % self.save_freq != 0:
            return
        
        score = metrics.get(self.monitor, float("inf"))
        
        if self.save_best_only:
            if (self.mode == "min" and score < self.best_score) or \
               (self.mode == "max" and score > self.best_score):
                self.best_score = score
                self._save_checkpoint(model, epoch, metrics, is_best=True)
        else:
            self._save_checkpoint(model, epoch, metrics, is_best=False)
    
    def _save_checkpoint(self, model: Any, epoch: int, metrics: Dict[str, float], is_best: bool) -> None:
        """实际保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "metrics": metrics,
            "best_score": self.best_score,
            "timestamp": time.time(),
        }
        
        filename = f"model_epoch_{epoch}.pth"
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, "wb") as f:
            pickle.dump(checkpoint, f)
        
        self.checkpoints.append((epoch, filepath, metrics))
        
        if len(self.checkpoints) > self.max_checkpoints:
            oldest_epoch, oldest_path, _ = self.checkpoints.pop(0)
            if os.path.exists(oldest_path):
                os.remove(oldest_path)
        
        if is_best:
            best_path = os.path.join(self.save_dir, "model_best.pth")
            with open(best_path, "wb") as f:
                pickle.dump(checkpoint, f)
    
    def load(self, model: Any, checkpoint_path: Optional[str] = None) -> Dict[str, float]:
        """加载检查点"""
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.save_dir, "model_best.pth")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        
        model.load_state_dict(checkpoint["state_dict"])
        self.best_score = checkpoint["best_score"]
        
        return checkpoint["metrics"]
    
    def get_best_checkpoint(self) -> Optional[str]:
        """获取最佳检查点路径"""
        best_path = os.path.join(self.save_dir, "model_best.pth")
        return best_path if os.path.exists(best_path) else None


class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: str = "./logs", name: str = "training"):
        self.log_dir = log_dir
        self.name = name
        self.logs = []
        self.metrics_history = {}
        
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(log_dir, f"{name}.log")
        self.metrics_file = os.path.join(log_dir, f"{name}_metrics.json")
    
    def log(self, message: str, level: str = "INFO") -> None:
        """记录日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
        
        self.logs.append(log_entry)
        print(log_entry)
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float], phase: str = "train") -> None:
        """记录指标"""
        if phase not in self.metrics_history:
            self.metrics_history[phase] = []
        
        entry = {
            "epoch": epoch,
            "phase": phase,
            "timestamp": time.time(),
            **metrics,
        }
        
        self.metrics_history[phase].append(entry)
        
        metrics_str = ", ".join(f"{k}={v:.6f}" for k, v in metrics.items())
        self.log(f"Epoch {epoch} [{phase}] {metrics_str}")
    
    def save_metrics(self) -> None:
        """保存指标到文件"""
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
    
    def load_metrics(self) -> Dict[str, List[Dict[str, float]]]:
        """加载指标"""
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, "r", encoding="utf-8") as f:
                self.metrics_history = json.load(f)
        return self.metrics_history
    
    def plot_metrics(self, metrics: List[str] = None) -> None:
        """绘制指标曲线"""
        try:
            import matplotlib.pyplot as plt
            
            if metrics is None:
                metrics = set()
                for phase in self.metrics_history:
                    for entry in self.metrics_history[phase]:
                        metrics.update(entry.keys())
                metrics = sorted([m for m in metrics if m not in ["epoch", "phase", "timestamp"]])
            
            fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 5 * len(metrics)))
            
            if len(metrics) == 1:
                axes = [axes]
            
            for i, metric in enumerate(metrics):
                ax = axes[i]
                for phase in self.metrics_history:
                    epochs = [entry["epoch"] for entry in self.metrics_history[phase]]
                    values = [entry[metric] for entry in self.metrics_history[phase] if metric in entry]
                    ax.plot(epochs, values, label=phase)
                
                ax.set_xlabel("Epoch")
                ax.set_ylabel(metric)
                ax.set_title(f"{metric} vs Epoch")
                ax.legend()
                ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, f"{self.name}_metrics.png"))
            plt.close()
            
        except ImportError:
            self.log("matplotlib not available, skipping plot", level="WARNING")


class EarlyStopping:
    """早停机制"""
    
    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 5,
        min_delta: float = 0.0,
        verbose: bool = False,
    ):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.early_stop = False
    
    def step(self, metrics: Dict[str, float]) -> bool:
        """执行一步早停检查"""
        score = metrics.get(self.monitor, float("inf"))
        
        if (self.mode == "min" and score < self.best_score - self.min_delta) or \
           (self.mode == "max" and score > self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class Trainer:
    """训练器"""
    
    def __init__(
        self,
        model: Any,
        optimizer: Any,
        criterion: Callable,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        logger: Optional[TrainingLogger] = None,
        checkpoint: Optional[ModelCheckpoint] = None,
        early_stopping: Optional[EarlyStopping] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger or TrainingLogger()
        self.checkpoint = checkpoint
        self.early_stopping = early_stopping
        self.device = device
        
        self.train_loss_history = []
        self.val_loss_history = []
    
    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            if isinstance(batch, tuple):
                inputs, targets = batch
            else:
                inputs, targets = batch, None
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            
            if targets is not None:
                loss = self.criterion(outputs, targets)
            else:
                loss = outputs
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss._data.item() * inputs.shape[0]
            total_samples += inputs.shape[0]
            
            if batch_idx % 100 == 0:
                self.logger.log(f"Epoch {epoch} Batch {batch_idx}/{len(self.train_loader)} "
                               f"Loss: {loss._data.item():.6f}")
        
        avg_loss = total_loss / total_samples
        self.train_loss_history.append(avg_loss)
        
        return {"loss": avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        for batch in self.val_loader:
            if isinstance(batch, tuple):
                inputs, targets = batch
            else:
                inputs, targets = batch, None
            
            outputs = self.model(inputs)
            
            if targets is not None:
                loss = self.criterion(outputs, targets)
                total_loss += loss._data.item() * inputs.shape[0]
                total_samples += inputs.shape[0]
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
        self.val_loss_history.append(avg_loss)
        
        return {"loss": avg_loss}
    
    def fit(self, epochs: int, start_epoch: int = 0) -> None:
        """训练主循环"""
        self.logger.log(f"Starting training for {epochs} epochs")
        
        for epoch in range(start_epoch, start_epoch + epochs):
            self.logger.log(f"===== Epoch {epoch + 1}/{start_epoch + epochs} =====")
            
            train_metrics = self.train_one_epoch(epoch)
            self.logger.log_metrics(epoch, train_metrics, phase="train")
            
            val_metrics = self.validate()
            if val_metrics:
                self.logger.log_metrics(epoch, val_metrics, phase="val")
            
            combined_metrics = {**train_metrics, **val_metrics}
            
            if self.checkpoint:
                self.checkpoint.save(self.model, epoch, combined_metrics)
            
            if self.early_stopping and self.early_stopping.step(combined_metrics):
                self.logger.log("Early stopping triggered", level="INFO")
                break
        
        self.logger.log("Training completed")
        self.logger.save_metrics()


class LearningRateFinder:
    """学习率查找器"""
    
    def __init__(
        self,
        model: Any,
        optimizer: Any,
        criterion: Callable,
        train_loader: DataLoader,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
    
    def find(
        self,
        start_lr: float = 1e-7,
        end_lr: float = 10,
        num_iter: int = 100,
        step_mode: str = "exp",
    ) -> Tuple[List[float], List[float]]:
        """查找最佳学习率"""
        lrs = []
        losses = []
        
        if step_mode == "exp":
            lr_step = (end_lr / start_lr) ** (1 / (num_iter - 1))
        else:
            lr_step = (end_lr - start_lr) / (num_iter - 1)
        
        current_lr = start_lr
        
        for i in range(num_iter):
            self.optimizer.param_groups[0]["lr"] = current_lr
            
            for batch in self.train_loader:
                if isinstance(batch, tuple):
                    inputs, targets = batch
                else:
                    inputs, targets = batch, None
                
                self.optimizer.zero_grad()
                
                outputs = self.model(inputs)
                
                if targets is not None:
                    loss = self.criterion(outputs, targets)
                else:
                    loss = outputs
                
                loss.backward()
                self.optimizer.step()
                
                lrs.append(current_lr)
                losses.append(loss._data.item())
                
                break
            
            if step_mode == "exp":
                current_lr *= lr_step
            else:
                current_lr += lr_step
        
        return lrs, losses
    
    def plot(self, lrs: List[float], losses: List[float]) -> None:
        """绘制学习率曲线"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.semilogx(lrs, losses)
            plt.xlabel("Learning Rate")
            plt.ylabel("Loss")
            plt.title("Learning Rate Finder")
            plt.grid(True)
            plt.savefig("lr_finder_plot.png")
            plt.close()
            
        except ImportError:
            pass


if __name__ == "__main__":
    print("=" * 60)
    print("SimplCore Training Toolchain 自检")
    print("=" * 60)
    
    print("\n1. Testing DataLoader...")
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 10, 100)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    for batch in loader:
        inputs, targets = batch
        assert inputs.shape == (10, 10)
        assert targets.shape == (10,)
        break
    
    assert len(loader) == 10
    print("   DataLoader: PASSED")
    
    print("\n2. Testing ModelCheckpoint...")
    checkpoint = ModelCheckpoint(save_dir="./test_checkpoints", save_best_only=False)
    
    class DummyModel:
        def state_dict(self):
            return {"weight": np.array([1.0, 2.0])}
        
        def load_state_dict(self, state_dict):
            pass
    
    model = DummyModel()
    checkpoint.save(model, 0, {"loss": 0.5})
    checkpoint.save(model, 1, {"loss": 0.3})
    
    metrics = checkpoint.load(model, "./test_checkpoints/model_epoch_1.pth")
    assert "loss" in metrics
    print("   ModelCheckpoint: PASSED")
    
    import shutil
    if os.path.exists("./test_checkpoints"):
        shutil.rmtree("./test_checkpoints")
    
    print("\n3. Testing TrainingLogger...")
    logger = TrainingLogger(log_dir="./test_logs")
    logger.log("Test message")
    logger.log_metrics(0, {"loss": 0.5, "accuracy": 0.8})
    logger.save_metrics()
    loaded = logger.load_metrics()
    assert "train" in loaded
    print("   TrainingLogger: PASSED")
    
    if os.path.exists("./test_logs"):
        shutil.rmtree("./test_logs")
    
    print("\n4. Testing EarlyStopping...")
    early_stopping = EarlyStopping(patience=2, mode="min")
    early_stopping.step({"val_loss": 1.0})
    early_stopping.step({"val_loss": 0.9})
    early_stopping.step({"val_loss": 0.91})
    early_stopping.step({"val_loss": 0.92})
    assert early_stopping.early_stop
    print("   EarlyStopping: PASSED")
    
    print("\n" + "=" * 60)
    print("SimplCore Training Toolchain 自检 PASSED")
    print("=" * 60)

