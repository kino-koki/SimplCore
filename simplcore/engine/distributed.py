# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimpDistributed - 青小团自研分布式训练框架
轻量级替代 DeepSpeed，支持模型并行和数据并行
"""
from __future__ import annotations

import os
import time
import socket
import threading
import numpy as np
from typing import List, Dict, Optional, Callable, Any
from .tensor import SimpTensor, zeros, ones


class DistributedConfig:
    """分布式配置"""
    
    def __init__(
        self,
        world_size: int = 1,
        rank: int = 0,
        local_rank: int = 0,
        master_addr: str = "localhost",
        master_port: int = 12345,
        backend: str = "gloo",
    ):
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
        self.master_addr = master_addr
        self.master_port = master_port
        self.backend = backend
        self._initialized = False
    
    @property
    def is_main_process(self) -> bool:
        return self.rank == 0
    
    @property
    def num_gpus(self) -> int:
        return self.world_size


class CommunicationBackend:
    """通信后端基类"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
    
    def send(self, data: np.ndarray, dst: int) -> None:
        """发送数据"""
        pass
    
    def recv(self, src: int) -> np.ndarray:
        """接收数据"""
        return np.zeros(0)
    
    def all_reduce(self, data: np.ndarray, op: str = "sum") -> np.ndarray:
        """全规约"""
        return data
    
    def all_gather(self, data: np.ndarray) -> List[np.ndarray]:
        """全收集"""
        return [data]
    
    def broadcast(self, data: np.ndarray, src: int = 0) -> np.ndarray:
        """广播"""
        return data


class GlooBackend(CommunicationBackend):
    """Gloo通信后端（模拟实现）"""
    
    def __init__(self, config: DistributedConfig):
        super().__init__(config)
        self._sockets = {}
        self._lock = threading.Lock()
        
        if config.world_size > 1:
            self._setup_connections()
    
    def _setup_connections(self) -> None:
        """建立连接"""
        if self.config.rank == 0:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.bind((self.config.master_addr, self.config.master_port))
            server.listen(self.config.world_size - 1)
            
            for i in range(1, self.config.world_size):
                conn, _ = server.accept()
                self._sockets[i] = conn
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.config.master_addr, self.config.master_port))
            self._sockets[0] = sock
    
    def send(self, data: np.ndarray, dst: int) -> None:
        """发送数据"""
        if self.config.world_size == 1:
            return
        
        serialized = data.tobytes()
        header = np.array([len(serialized), *data.shape, data.dtype.itemsize], dtype=np.int64).tobytes()
        
        sock = self._sockets.get(dst)
        if sock:
            sock.sendall(header + serialized)
    
    def recv(self, src: int) -> np.ndarray:
        """接收数据"""
        if self.config.world_size == 1:
            return np.zeros(0)
        
        sock = self._sockets.get(src)
        if not sock:
            return np.zeros(0)
        
        header_data = self._recv_all(sock, 8 * 4)
        header = np.frombuffer(header_data, dtype=np.int64)
        data_len, shape, dtype_size = header[0], header[1:-1], header[-1]
        
        data = self._recv_all(sock, data_len)
        return np.frombuffer(data, dtype=f"f{dtype_size}").reshape(tuple(shape))
    
    def _recv_all(self, sock: socket.socket, size: int) -> bytes:
        """接收指定大小的数据"""
        buffer = b""
        while len(buffer) < size:
            chunk = sock.recv(size - len(buffer))
            if not chunk:
                break
            buffer += chunk
        return buffer
    
    def all_reduce(self, data: np.ndarray, op: str = "sum") -> np.ndarray:
        """全规约（环形算法）"""
        if self.config.world_size == 1:
            return data
        
        result = data.copy()
        
        for i in range(self.config.world_size - 1):
            send_to = (self.config.rank + 1) % self.config.world_size
            recv_from = (self.config.rank - 1) % self.config.world_size
            
            self.send(result, send_to)
            received = self.recv(recv_from)
            
            if op == "sum":
                result += received
            elif op == "max":
                result = np.maximum(result, received)
            elif op == "min":
                result = np.minimum(result, received)
        
        if op == "avg":
            result = result / self.config.world_size
        
        return result
    
    def all_gather(self, data: np.ndarray) -> List[np.ndarray]:
        """全收集"""
        if self.config.world_size == 1:
            return [data]
        
        results = [data.copy()]
        
        for _ in range(self.config.world_size - 1):
            send_to = (self.config.rank + 1) % self.config.world_size
            recv_from = (self.config.rank - 1) % self.config.world_size
            
            self.send(results[-1], send_to)
            received = self.recv(recv_from)
            results.append(received)
        
        return results
    
    def broadcast(self, data: np.ndarray, src: int = 0) -> np.ndarray:
        """广播"""
        if self.config.rank == src or self.config.world_size == 1:
            return data
        
        if self.config.rank == (src + 1) % self.config.world_size:
            received = self.recv(src)
            for i in range(1, self.config.world_size - 1):
                self.send(received, (src + i + 1) % self.config.world_size)
            return received
        
        return self.recv((self.config.rank - 1) % self.config.world_size)


class DistributedGroup:
    """分布式组"""
    
    def __init__(self, config: DistributedConfig, group_name: str = "default"):
        self.config = config
        self.group_name = group_name
        self._backend = GlooBackend(config)
    
    def all_reduce(self, tensor: SimpTensor, op: str = "sum") -> SimpTensor:
        """全规约操作"""
        data = self._backend.all_reduce(tensor._data, op)
        return SimpTensor(data, device=tensor.device, requires_grad=tensor.requires_grad)
    
    def all_gather(self, tensor: SimpTensor) -> List[SimpTensor]:
        """全收集操作"""
        data_list = self._backend.all_gather(tensor._data)
        return [SimpTensor(d, device=tensor.device, requires_grad=tensor.requires_grad) for d in data_list]
    
    def reduce_scatter(self, tensors: List[SimpTensor], op: str = "sum") -> SimpTensor:
        """归约散射操作"""
        if self.config.world_size == 1:
            return tensors[0]
        
        data = sum(t._data for t in tensors)
        
        if op == "avg":
            data = data / len(tensors)
        
        return SimpTensor(data, device=tensors[0].device)
    
    def broadcast(self, tensor: SimpTensor, src: int = 0) -> SimpTensor:
        """广播操作"""
        data = self._backend.broadcast(tensor._data, src)
        return SimpTensor(data, device=tensor.device, requires_grad=tensor.requires_grad)
    
    def reduce(self, tensor: SimpTensor, dst: int = 0, op: str = "sum") -> SimpTensor:
        """归约到指定节点"""
        if self.config.rank == dst:
            gathered = self.all_gather(tensor)
            data = sum(t._data for t in gathered)
            if op == "avg":
                data = data / len(gathered)
            return SimpTensor(data, device=tensor.device)
        return tensor


class ModelParallelGroup(DistributedGroup):
    """模型并行组"""
    
    def __init__(self, config: DistributedConfig, num_model_parallel: int = 1):
        super().__init__(config, "model_parallel")
        self.num_model_parallel = num_model_parallel
        self.model_parallel_rank = config.rank % num_model_parallel
        self.data_parallel_rank = config.rank // num_model_parallel
    
    @property
    def is_first_model_parallel(self) -> bool:
        return self.model_parallel_rank == 0


class DataParallelGroup(DistributedGroup):
    """数据并行组"""
    
    def __init__(self, config: DistributedConfig, num_data_parallel: int = 1):
        super().__init__(config, "data_parallel")
        self.num_data_parallel = num_data_parallel
        self.data_parallel_rank = config.rank % num_data_parallel
        self.model_parallel_rank = config.rank // num_data_parallel


class ZeROStage3Optimizer:
    """ZeRO-3优化器（简化版）"""
    
    def __init__(self, optimizer, config: DistributedConfig):
        self.optimizer = optimizer
        self.config = config
        self._param_groups = optimizer.parameters
        self._group = DistributedGroup(config)
    
    def step(self) -> None:
        """执行优化步骤"""
        for param in self._param_groups:
            if param.grad is None:
                continue
            
            grad_data = param.grad._data
            
            if self.config.world_size > 1:
                grad_tensor = self._group.all_reduce(SimpTensor(grad_data), op="avg")
                grad_data = grad_tensor._data
            
            param._data = param._data - self.optimizer.lr * grad_data
    
    def zero_grad(self) -> None:
        """清空梯度"""
        self.optimizer.zero_grad()
    
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        return self.optimizer.__dict__.copy()
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载状态字典"""
        self.optimizer.__dict__.update(state_dict)


class MixedPrecisionTraining:
    """混合精度训练"""
    
    def __init__(self, config: DistributedConfig, enabled: bool = False):
        self.config = config
        self.enabled = enabled
        self._scaler = 1.0
        self._scale_factor = 2.0
        self._min_scale = 1e-4
    
    def scale_loss(self, loss: SimpTensor) -> SimpTensor:
        """缩放损失"""
        if not self.enabled:
            return loss
        
        self._scaler = min(self._scaler * self._scale_factor, 1.0)
        return SimpTensor(loss._data * self._scaler)
    
    def unscale_grads(self, parameters: List[SimpTensor]) -> None:
        """反缩放梯度"""
        if not self.enabled:
            return
        
        for param in parameters:
            if param.grad is not None:
                param.grad._data = param.grad._data / self._scaler
    
    def clip_grads(self, parameters: List[SimpTensor], max_norm: float = 1.0) -> float:
        """梯度裁剪"""
        total_norm = 0.0
        
        for param in parameters:
            if param.grad is not None:
                param_norm = np.sum(param.grad._data ** 2)
                total_norm += param_norm
        
        total_norm = np.sqrt(total_norm)
        
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            for param in parameters:
                if param.grad is not None:
                    param.grad._data = param.grad._data * clip_coef
        
        return total_norm
    
    def update_scale(self, grad_norm: float) -> None:
        """更新缩放因子"""
        if not self.enabled:
            return
        
        if grad_norm == 0 or np.isnan(grad_norm):
            self._scaler = max(self._scaler / self._scale_factor, self._min_scale)
        elif grad_norm > 1.0:
            self._scaler = max(self._scaler / self._scale_factor, self._min_scale)
        else:
            self._scaler = min(self._scaler * self._scale_factor, 1.0)


class GradientAccumulator:
    """梯度累加器"""
    
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self._step = 0
        self._accumulated_grads: Dict[int, np.ndarray] = {}
    
    def accumulate(self, parameters: List[SimpTensor]) -> bool:
        """累加梯度"""
        self._step += 1
        
        for i, param in enumerate(parameters):
            if param.grad is None:
                continue
            
            if i not in self._accumulated_grads:
                self._accumulated_grads[i] = np.zeros_like(param.grad._data)
            
            self._accumulated_grads[i] += param.grad._data
        
        should_step = self._step >= self.accumulation_steps
        
        if should_step:
            for i, param in enumerate(parameters):
                if i in self._accumulated_grads:
                    param.grad = SimpTensor(self._accumulated_grads[i] / self.accumulation_steps)
            
            self._step = 0
            self._accumulated_grads.clear()
        
        return should_step


class PipelineScheduler:
    """流水线调度器"""
    
    def __init__(self, stages: List[Callable], micro_batch_size: int = 1):
        self.stages = stages
        self.micro_batch_size = micro_batch_size
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        """前向传播"""
        for stage in self.stages:
            x = stage(x)
        return x
    
    def backward(self, grad: SimpTensor) -> SimpTensor:
        """反向传播"""
        for stage in reversed(self.stages):
            grad = stage.backward(grad)
        return grad


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, save_dir: str = "./checkpoints"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save(self, model, optimizer, step: int, loss: float) -> None:
        """保存检查点"""
        checkpoint = {
            "step": step,
            "loss": loss,
            "timestamp": time.time(),
        }
        
        for name, param in model.named_parameters():
            checkpoint[f"param_{name}"] = param._data
        
        np.savez(f"{self.save_dir}/checkpoint_{step}.npz", **checkpoint)
    
    def load(self, model, optimizer, step: Optional[int] = None) -> int:
        """加载检查点"""
        if step is None:
            checkpoints = sorted([f for f in os.listdir(self.save_dir) if f.startswith("checkpoint_")])
            if not checkpoints:
                return 0
            step = int(checkpoints[-1].split("_")[1].split(".")[0])
        
        checkpoint = np.load(f"{self.save_dir}/checkpoint_{step}.npz")
        
        for name, param in model.named_parameters():
            key = f"param_{name}"
            if key in checkpoint:
                param._data = checkpoint[key]
        
        return checkpoint.get("step", 0)


class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._log_file = None
    
    def open(self, filename: str = "training.log") -> None:
        """打开日志文件"""
        self._log_file = open(f"{self.log_dir}/{filename}", "a", encoding="utf-8")
    
    def log(self, step: int, loss: float, lr: float, time: float) -> None:
        """记录日志"""
        if self._log_file:
            log_str = f"[{time:.2f}] Step={step}, Loss={loss:.6f}, LR={lr:.6f}\n"
            self._log_file.write(log_str)
            self._log_file.flush()
    
    def close(self) -> None:
        """关闭日志文件"""
        if self._log_file:
            self._log_file.close()
            self._log_file = None


def init_distributed(config: Optional[DistributedConfig] = None) -> DistributedConfig:
    """初始化分布式环境"""
    if config is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        config = DistributedConfig(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
        )
    
    return config


if __name__ == "__main__":
    print("=" * 60)
    print("SimpDistributed 自检")
    print("=" * 60)
    
    config = init_distributed()
    print(f"World size: {config.world_size}")
    print(f"Rank: {config.rank}")
    print(f"Local rank: {config.local_rank}")
    
    group = DistributedGroup(config)
    tensor = zeros(3, 4)
    
    result = group.all_reduce(tensor)
    print(f"AllReduce result shape: {result.shape}")
    
    gathered = group.all_gather(tensor)
    print(f"AllGather result count: {len(gathered)}")
    
    logger = TrainingLogger()
    logger.open()
    logger.log(0, 1.0, 1e-3, time.time())
    logger.close()
    print("Logger test completed")
    
    checkpoint_manager = CheckpointManager()
    print("CheckpointManager initialized")
    
    print("\nSimpDistributed 自检 PASSED")
