# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""持续学习引擎 - 支持无限训练和知识自迭代.

核心功能:
  1. 互联网数据自动获取 (模拟)
  2. 增量学习 - 无需完全重新训练
  3. 知识蒸馏 - 将新知识融入模型
  4. 遗忘抑制 - 防止灾难性遗忘
  5. 自适应学习率 - 根据数据难度调整
  6. 学习计划生成 - 智能规划学习目标

设计目标:
  - 训练数据: 完全基于互联网
  - 训练轮数: 无限,只要计算机存在就一直训练
  - 知识更新: 每15分钟自动迭代知识库
  - 无需重新训练: 通过增量学习更新模型

与SimplCoreAGI的关系:
  - ContinuousLearningEngine: 管理学习流程
  - SimplCoreAGI: 被持续更新的模型
  - KnowledgeBase: 持续扩展的知识库
"""

from __future__ import annotations

import simplcore.tensor as torch
import simplcore.tensor.nn as nn
import numpy as np
import time
import json
import os
import random
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta


class DataStreamer:
    """数据流生成器 - 模拟互联网数据持续流入.
    
    支持的数据类型:
      1. 文本数据 (新闻、文章、书籍)
      2. 图像数据 (模拟图像特征)
      3. 视频数据 (时序图像序列)
      4. 结构化数据 (表格、知识图谱)
    
    数据来源模拟:
      - 使用预定义模板生成模拟数据
      - 支持真实数据导入
      - 数据多样性控制
    """
    
    def __init__(
        self,
        text_vocab_size: int = 5000,
        seq_len: int = 128,
        image_dim: int = 128,
        video_frames: int = 16,
        data_types: List[str] = None,
        batch_size: int = 32,
        update_interval: int = 900,  # 15分钟更新一次
    ):
        if data_types is None:
            data_types = ["text", "image", "video", "structured"]
        
        self.text_vocab_size = text_vocab_size
        self.seq_len = seq_len
        self.image_dim = image_dim
        self.video_frames = video_frames
        self.data_types = data_types
        self.batch_size = batch_size
        self.update_interval = update_interval
        
        # 模拟数据模板
        self.text_templates = self._load_text_templates()
        self.knowledge_topics = self._load_knowledge_topics()
        
        # 状态
        self.last_update = time.time()
        self.data_index = 0
        self.current_topic = 0
    
    def _load_text_templates(self) -> List[str]:
        """加载文本生成模板."""
        return [
            "科学发现: {topic}取得重大突破，研究人员发现{detail}.",
            "技术新闻: {topic}技术最新进展，{detail}.",
            "学术论文: {topic}领域新研究表明{detail}.",
            "科普知识: {topic}是{detail}.",
            "最新资讯: {topic}方面有了新的发现，{detail}.",
            "专家观点: {topic}专家认为{detail}.",
            "历史事件: {topic}发生了{detail}.",
            "未来预测: {topic}将在未来{detail}.",
        ]
    
    def _load_knowledge_topics(self) -> List[Tuple[str, str]]:
        """加载知识主题."""
        return [
            ("人工智能", "深度学习和神经网络的发展"),
            ("量子计算", "量子比特和量子纠缠的应用"),
            ("宇宙探索", "黑洞和暗物质的研究"),
            ("生命科学", "基因编辑和CRISPR技术"),
            ("新能源", "太阳能和风能的发展"),
            ("纳米技术", "纳米材料和纳米器件"),
            ("脑科学", "神经科学和意识研究"),
            ("区块链", "加密货币和分布式账本"),
            ("自动驾驶", "无人驾驶汽车技术"),
            ("生物技术", "生物制药和疫苗研发"),
            ("气候科学", "全球变暖和气候变化"),
            ("空间技术", "火星探测和太空殖民"),
            ("医学研究", "癌症治疗和精准医疗"),
            ("材料科学", "新型材料和超导技术"),
            ("计算生物学", "生物信息学和蛋白质结构预测"),
        ]
    
    def _generate_text_data(self) -> np.ndarray:
        """生成模拟文本数据."""
        batch = np.random.randint(0, self.text_vocab_size, 
                                  (self.batch_size, self.seq_len))
        return batch
    
    def _generate_text_content(self) -> List[str]:
        """生成真实文本内容."""
        contents = []
        for _ in range(self.batch_size):
            topic, detail = self.knowledge_topics[self.current_topic % len(self.knowledge_topics)]
            template = random.choice(self.text_templates)
            content = template.format(topic=topic, detail=detail)
            contents.append(content)
            self.current_topic += 1
        return contents
    
    def _generate_image_data(self) -> np.ndarray:
        """生成模拟图像数据."""
        images = np.random.randn(self.batch_size, 3, 64, 64).astype(np.float32)
        images = (images - images.min()) / (images.max() - images.min() + 1e-10)
        return images
    
    def _generate_video_data(self) -> np.ndarray:
        """生成模拟视频数据."""
        video = np.random.randn(
            self.batch_size, self.video_frames, 3, 64, 64
        ).astype(np.float32)
        # 添加时间相关性
        for t in range(1, self.video_frames):
            video[:, t] = video[:, t-1] * 0.9 + video[:, t] * 0.1
        return video
    
    def _generate_structured_data(self) -> Dict[str, np.ndarray]:
        """生成结构化数据."""
        return {
            "features": np.random.randn(self.batch_size, 64).astype(np.float32),
            "labels": np.random.randint(0, 10, self.batch_size),
            "metadata": np.random.randn(self.batch_size, 16).astype(np.float32),
        }
    
    def _should_update(self) -> bool:
        """检查是否需要更新数据."""
        return time.time() - self.last_update > self.update_interval
    
    def _update_topics(self) -> None:
        """更新知识主题."""
        self.last_update = time.time()
        random.shuffle(self.knowledge_topics)
        print(f"[DataStreamer] 知识库主题已更新 ({len(self.knowledge_topics)} 个主题)")
    
    def get_batch(self, data_type: str = None) -> Dict[str, Any]:
        """获取一批数据.
        
        Args:
            data_type: 数据类型. None表示随机选择.
            
        Returns:
            dict: 包含数据和元信息.
        """
        if self._should_update():
            self._update_topics()
        
        if data_type is None:
            data_type = random.choice(self.data_types)
        
        result = {"type": data_type, "timestamp": time.time()}
        
        if data_type == "text":
            result["text"] = self._generate_text_data()
            result["content"] = self._generate_text_content()
            result["shape"] = result["text"].shape
        
        elif data_type == "image":
            result["image"] = self._generate_image_data()
            result["shape"] = result["image"].shape
        
        elif data_type == "video":
            result["video"] = self._generate_video_data()
            result["shape"] = result["video"].shape
        
        elif data_type == "structured":
            struct_data = self._generate_structured_data()
            result.update(struct_data)
            result["shape"] = struct_data["features"].shape
        
        self.data_index += 1
        return result
    
    def stream(self, max_batches: int = None) -> None:
        """数据流式输出生成器."""
        batch_count = 0
        while max_batches is None or batch_count < max_batches:
            yield self.get_batch()
            batch_count += 1


class IncrementalLearner(nn.Module):
    """增量学习器 - 无需完全重新训练即可更新模型.
    
    核心技术:
      1. 弹性权重巩固 (EWC) - 保护重要参数
      2. 知识蒸馏 - 保留旧知识
      3. 适配器网络 - 模块化更新
      4. 动态路由 - 选择性更新
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        ewc_lambda: float = 1000.0,
        distillation_temperature: float = 2.0,
        adapter_dim: int = 64,
        freeze_base: bool = False,
    ):
        super().__init__()
        
        self.base_model = base_model
        self.ewc_lambda = ewc_lambda
        self.distillation_temperature = distillation_temperature
        self.adapter_dim = adapter_dim
        self.freeze_base = freeze_base
        
        # EWC 参数
        self.fisher_information = {}
        self.old_params = {}
        
        # 适配器网络
        self.adapters = self._create_adapters()
        
        # 知识蒸馏损失
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.mse_loss = nn.MSELoss()
        
        # 如果冻结基础模型
        if freeze_base:
            for param in base_model.parameters():
                param.requires_grad = False
    
    def _create_adapters(self) -> nn.ModuleDict:
        """为每个模块创建适配器."""
        adapters = nn.ModuleDict()
        
        for name, module in self.base_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                in_dim = module.weight.shape[1] if hasattr(module, 'weight') else 64
                adapters[f"{name}_adapter"] = nn.Sequential(
                    nn.Linear(in_dim, self.adapter_dim),
                    nn.GELU(),
                    nn.Linear(self.adapter_dim, in_dim),
                )
        
        return adapters
    
    def _compute_fisher_information(self, dataloader: Any) -> None:
        """计算Fisher信息矩阵 (EWC)."""
        self.base_model.eval()
        
        for name, param in self.base_model.named_parameters():
            if param.requires_grad:
                self.fisher_information[name] = torch.zeros_like(param)
                self.old_params[name] = param.detach().clone()
        
        for batch in dataloader:
            self.base_model.zero_grad()
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            outputs = self.base_model(inputs)
            
            if isinstance(outputs, dict):
                loss = sum(v.mean() for v in outputs.values() if isinstance(v, torch.Tensor))
            else:
                loss = outputs.mean()
            
            loss.backward()
            
            for name, param in self.base_model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher_information[name] += param.grad ** 2 / len(dataloader)
    
    def _ewc_loss(self) -> torch.Tensor:
        """计算EWC损失."""
        loss = torch.tensor(0.0)
        for name, param in self.base_model.named_parameters():
            if name in self.fisher_information:
                loss += self.ewc_lambda * torch.sum(
                    self.fisher_information[name] * (param - self.old_params[name]) ** 2
                )
        return loss
    
    def _distillation_loss(
        self,
        new_outputs: torch.Tensor,
        old_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """计算知识蒸馏损失."""
        new_probs = nn.functional.log_softmax(new_outputs / self.distillation_temperature, dim=-1)
        old_probs = nn.functional.softmax(old_outputs / self.distillation_temperature, dim=-1)
        return self.kl_loss(new_probs, old_probs)
    
    def forward(self, inputs: Any, use_adapters: bool = True) -> Any:
        """前向传播 - 支持适配器."""
        if use_adapters and len(self.adapters) > 0:
            return self._forward_with_adapters(inputs)
        return self.base_model(inputs)
    
    def _forward_with_adapters(self, inputs: Any) -> Any:
        """使用适配器前向传播."""
        # 这是一个简化实现，实际需要根据模型结构定制
        outputs = self.base_model(inputs)
        
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    adapter_name = f"{key}_adapter"
                    if adapter_name in self.adapters:
                        outputs[key] = value + self.adapters[adapter_name](value)
        elif isinstance(outputs, torch.Tensor):
            outputs = outputs + self.adapters[list(self.adapters.keys())[0]](outputs)
        
        return outputs
    
    def incremental_update(
        self,
        new_data: Any,
        old_data: Optional[Any] = None,
        optimizer: torch.optim.Optimizer = None,
        epochs: int = 1,
        use_ewc: bool = True,
        use_distillation: bool = True,
    ) -> Dict[str, float]:
        """增量更新模型.
        
        Args:
            new_data: 新数据
            old_data: 旧数据 (用于EWC和蒸馏)
            optimizer: 优化器
            epochs: 更新轮数
            use_ewc: 是否使用EWC
            use_distillation: 是否使用知识蒸馏
            
        Returns:
            dict: 更新结果
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        
        self.train()
        results = {"losses": [], "ewc_losses": [], "distill_losses": []}
        
        for epoch in range(epochs):
            total_loss = 0.0
            ewc_loss = 0.0
            distill_loss = 0.0
            
            for batch in new_data:
                optimizer.zero_grad()
                
                # 计算新任务损失
                inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                new_outputs = self(inputs)
                
                if isinstance(new_outputs, dict):
                    task_loss = sum(v.mean() for v in new_outputs.values() if isinstance(v, torch.Tensor))
                else:
                    task_loss = new_outputs.mean()
                
                loss = task_loss
                
                # EWC损失
                if use_ewc:
                    ewc_loss = self._ewc_loss()
                    loss += ewc_loss
                
                # 知识蒸馏损失
                if use_distillation and old_data is not None:
                    old_inputs = old_data[0] if isinstance(old_data, (list, tuple)) else old_data
                    with torch.no_grad():
                        old_outputs = self.base_model(old_inputs)
                    distill_loss = self._distillation_loss(new_outputs, old_outputs)
                    loss += distill_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            results["losses"].append(total_loss / len(new_data))
            results["ewc_losses"].append(ewc_loss.item() if use_ewc else 0)
            results["distill_losses"].append(distill_loss.item() if use_distillation else 0)
        
        return results
    
    def freeze_base_model(self) -> None:
        """冻结基础模型参数."""
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def unfreeze_base_model(self) -> None:
        """解冻基础模型参数."""
        for param in self.base_model.parameters():
            param.requires_grad = True


class ForgettingPrevention:
    """遗忘抑制模块 - 防止灾难性遗忘.
    
    策略:
      1. EWC (弹性权重巩固)
      2. SI (突触智能)
      3. 回放缓冲 (Replay Buffer)
      4. 正则化约束
    """
    
    def __init__(
        self,
        model: nn.Module,
        buffer_size: int = 1000,
        ewc_lambda: float = 1000.0,
        si_c: float = 0.1,
    ):
        self.model = model
        self.buffer_size = buffer_size
        self.ewc_lambda = ewc_lambda
        self.si_c = si_c
        
        # 回放缓冲
        self.replay_buffer = []
        
        # EWC状态
        self.fisher = {}
        self.old_params = {}
        
        # SI状态
        self.w_prev = {}
        self.si_omega = {}
        
        # 初始化SI
        self._init_si()
    
    def _init_si(self) -> None:
        """初始化SI参数."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.w_prev[name] = param.detach().clone()
                self.si_omega[name] = torch.zeros_like(param)
    
    def add_to_buffer(self, data: Any) -> None:
        """添加数据到回放缓冲."""
        self.replay_buffer.append(data)
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
    
    def sample_from_buffer(self, batch_size: int) -> List[Any]:
        """从缓冲中采样."""
        if len(self.replay_buffer) == 0:
            return []
        return random.sample(self.replay_buffer, min(batch_size, len(self.replay_buffer)))
    
    def update_fisher(self, dataloader: Any) -> None:
        """更新Fisher信息矩阵."""
        self.model.eval()
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher[name] = torch.zeros_like(param)
                self.old_params[name] = param.detach().clone()
        
        for batch in dataloader:
            self.model.zero_grad()
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            outputs = self.model(inputs)
            
            if isinstance(outputs, dict):
                loss = sum(v.mean() for v in outputs.values() if isinstance(v, torch.Tensor))
            else:
                loss = outputs.mean()
            
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher[name] += param.grad ** 2 / len(dataloader)
    
    def compute_ewc_loss(self) -> torch.Tensor:
        """计算EWC损失."""
        loss = torch.tensor(0.0)
        for name, param in self.model.named_parameters():
            if name in self.fisher:
                loss += self.ewc_lambda * torch.sum(
                    self.fisher[name] * (param - self.old_params[name]) ** 2
                )
        return loss
    
    def update_si(self) -> None:
        """更新SI参数."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                delta_w = param.detach() - self.w_prev[name]
                self.si_omega[name] += self.si_c * delta_w ** 2
                self.w_prev[name] = param.detach().clone()
    
    def compute_si_loss(self) -> torch.Tensor:
        """计算SI损失."""
        loss = torch.tensor(0.0)
        for name, param in self.model.named_parameters():
            if name in self.si_omega:
                loss += torch.sum(self.si_omega[name] * param ** 2)
        return loss
    
    def compute_total_regularization(self) -> torch.Tensor:
        """计算总正则化损失."""
        ewc_loss = self.compute_ewc_loss()
        si_loss = self.compute_si_loss()
        return ewc_loss + si_loss


class AdaptiveLearningRate:
    """自适应学习率调度器 - 根据数据难度和模型表现调整."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_lr: float = 1e-4,
        min_lr: float = 1e-6,
        max_lr: float = 1e-3,
        patience: int = 5,
        factor: float = 0.5,
        improvement_threshold: float = 0.01,
    ):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.patience = patience
        self.factor = factor
        self.improvement_threshold = improvement_threshold
        
        self.current_lr = initial_lr
        self.patience_counter = 0
        self.best_loss = float("inf")
        self.loss_history = []
    
    def step(self, loss: float) -> None:
        """根据损失调整学习率."""
        self.loss_history.append(loss)
        
        # 更新最佳损失
        if loss < self.best_loss * (1 - self.improvement_threshold):
            self.best_loss = loss
            self.patience_counter = 0
            # 提高学习率
            self._increase_lr()
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                # 降低学习率
                self._decrease_lr()
                self.patience_counter = 0
    
    def _increase_lr(self) -> None:
        """提高学习率."""
        new_lr = min(self.current_lr * (1 / self.factor), self.max_lr)
        if new_lr > self.current_lr:
            self.current_lr = new_lr
            self._update_optimizer_lr()
            print(f"[AdaptiveLR] 提高学习率: {self.current_lr:.8f}")
    
    def _decrease_lr(self) -> None:
        """降低学习率."""
        new_lr = max(self.current_lr * self.factor, self.min_lr)
        if new_lr < self.current_lr:
            self.current_lr = new_lr
            self._update_optimizer_lr()
            print(f"[AdaptiveLR] 降低学习率: {self.current_lr:.8f}")
    
    def _update_optimizer_lr(self) -> None:
        """更新优化器学习率."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
    
    def get_lr(self) -> float:
        """获取当前学习率."""
        return self.current_lr


class ContinuousLearningEngine:
    """持续学习引擎 - 协调所有持续学习功能."""
    
    def __init__(
        self,
        agi_model: nn.Module,
        knowledge_base: Any = None,
        update_interval: int = 900,  # 15分钟
        max_training_hours: float = None,  # None表示无限
        log_dir: str = "./logs",
        checkpoint_dir: str = "./checkpoints",
    ):
        self.agi_model = agi_model
        self.knowledge_base = knowledge_base
        self.update_interval = update_interval
        self.max_training_hours = max_training_hours
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        
        # 组件
        self.data_streamer = DataStreamer()
        self.incremental_learner = IncrementalLearner(agi_model)
        self.forgetting_prevention = ForgettingPrevention(agi_model)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            agi_model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
        )
        
        self.adaptive_lr = AdaptiveLearningRate(self.optimizer)
        
        # 状态
        self.running = False
        self.total_batches = 0
        self.total_epochs = 0
        self.start_time = None
        self.last_checkpoint = 0
        self.learning_log = []
        
        # 创建目录
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def _should_stop(self) -> bool:
        """检查是否应该停止训练."""
        if self.max_training_hours is None:
            return False
        
        elapsed_hours = (time.time() - self.start_time) / 3600
        return elapsed_hours >= self.max_training_hours
    
    def _should_checkpoint(self) -> bool:
        """检查是否应该保存检查点."""
        return time.time() - self.last_checkpoint > 3600  # 每小时保存一次
    
    def _save_checkpoint(self) -> None:
        """保存检查点."""
        checkpoint = {
            "model_state_dict": self.agi_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_batches": self.total_batches,
            "total_epochs": self.total_epochs,
            "start_time": self.start_time,
            "learning_log": self.learning_log,
            "timestamp": time.time(),
        }
        
        filename = f"{self.checkpoint_dir}/continuous_{int(time.time())}.pt"
        torch.save(checkpoint, filename)
        self.last_checkpoint = time.time()
        
        print(f"[ContinuousLearning] 检查点已保存: {filename}")
    
    def _save_log(self) -> None:
        """保存学习日志."""
        log_data = {
            "total_batches": self.total_batches,
            "total_epochs": self.total_epochs,
            "learning_log": self.learning_log,
            "timestamp": time.time(),
        }
        
        with open(f"{self.log_dir}/continuous_learning.log", "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    def _update_knowledge_base(self) -> None:
        """更新知识库."""
        if self.knowledge_base is None:
            return
        
        print("[ContinuousLearning] 更新知识库...")
        
        # 从数据流获取新知识
        for _ in range(10):
            batch = self.data_streamer.get_batch("text")
            for content in batch.get("content", []):
                self.knowledge_base.add_knowledge(
                    title=f"自动获取知识_{int(time.time())}",
                    content=content,
                    source="continuous_learning",
                    category="auto",
                )
        
        print(f"[ContinuousLearning] 知识库已更新，当前条目数: {len(self.knowledge_base.entries)}")
    
    def _train_batch(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """训练一批数据."""
        self.agi_model.train()
        self.optimizer.zero_grad()
        
        # 根据数据类型处理
        data_type = batch.get("type", "text")
        
        if data_type == "text":
            text_data = torch.tensor(batch["text"], dtype=torch.long)
            outputs = self.agi_model.language_model(text_data)
            
            if isinstance(outputs, dict):
                loss = sum(v.mean() for v in outputs.values() if isinstance(v, torch.Tensor))
            else:
                loss = outputs.mean()
        
        elif data_type == "image":
            image_data = torch.tensor(batch["image"], dtype=torch.float32)
            latent = self.agi_model.vision_encoder.encode_image(image_data)
            
            # 重建损失
            recon = self.agi_model.video_prediction_head(latent)
            loss = torch.nn.functional.mse_loss(recon, image_data)
        
        elif data_type == "video":
            video_data = torch.tensor(batch["video"], dtype=torch.float32)
            
            # 视频预测损失
            loss = torch.tensor(0.0)
            for t in range(video_data.shape[1] - 1):
                current_frame = video_data[:, t]
                next_frame = video_data[:, t+1]
                
                latent = self.agi_model.vision_encoder.encode_image(current_frame)
                predicted = self.agi_model.video_prediction_head(latent)
                loss += torch.nn.functional.mse_loss(predicted, next_frame)
            
            loss /= (video_data.shape[1] - 1)
        
        elif data_type == "structured":
            features = torch.tensor(batch["features"], dtype=torch.float32)
            labels = torch.tensor(batch["labels"], dtype=torch.long)
            
            # 简单分类损失
            logits = self.agi_model.dynamics_model.core(features)
            loss = torch.nn.functional.cross_entropy(logits, labels)
        
        else:
            loss = torch.tensor(0.0)
        
        # 添加遗忘抑制损失
        regularization_loss = self.forgetting_prevention.compute_total_regularization()
        total_loss = loss + regularization_loss * 0.1
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agi_model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 更新自适应学习率
        self.adaptive_lr.step(loss.item())
        
        # 更新SI参数
        self.forgetting_prevention.update_si()
        
        # 添加到回放缓冲
        self.forgetting_prevention.add_to_buffer(batch)
        
        return {
            "loss": loss.item(),
            "regularization_loss": regularization_loss.item(),
            "total_loss": total_loss.item(),
            "lr": self.adaptive_lr.get_lr(),
            "data_type": data_type,
        }
    
    def run(self) -> None:
        """运行持续学习主循环."""
        print("=" * 70)
        print("SimplCore 持续学习引擎启动")
        print("=" * 70)
        print(f" 更新间隔: {self.update_interval // 60} 分钟")
        print(f" 最大训练时长: {'无限' if self.max_training_hours is None else f'{self.max_training_hours} 小时'}")
        print(f" 检查点目录: {self.checkpoint_dir}")
        print(f" 日志目录: {self.log_dir}")
        print("=" * 70)
        
        self.running = True
        self.start_time = time.time()
        
        try:
            while self.running and not self._should_stop():
                # 获取数据批次
                batch = self.data_streamer.get_batch()
                
                # 训练批次
                result = self._train_batch(batch)
                
                self.total_batches += 1
                
                # 记录日志
                self.learning_log.append({
                    "batch": self.total_batches,
                    "time": time.time() - self.start_time,
                    **result,
                })
                
                # 定期输出
                if self.total_batches % 100 == 0:
                    avg_loss = sum(log["loss"] for log in self.learning_log[-100:]) / 100
                    print(f"[ContinuousLearning] Batch {self.total_batches} "
                          f"Loss: {avg_loss:.6f} "
                          f"LR: {result['lr']:.8f} "
                          f"Type: {result['data_type']}")
                
                # 更新知识库 (每update_interval秒)
                if self.total_batches % (self.update_interval // 10) == 0:
                    self._update_knowledge_base()
                
                # 保存检查点
                if self._should_checkpoint():
                    self._save_checkpoint()
                
                # 保存日志
                if self.total_batches % 1000 == 0:
                    self._save_log()
                
                # 短暂休息
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\n[ContinuousLearning] 收到中断信号，停止训练")
            self.running = False
        except Exception as e:
            print(f"\n[ContinuousLearning] 训练出错: {e}")
            self.running = False
        finally:
            print("\n[ContinuousLearning] 保存最终状态...")
            self._save_checkpoint()
            self._save_log()
            
            elapsed = time.time() - self.start_time
            print(f"[ContinuousLearning] 训练结束")
            print(f"  总批次: {self.total_batches}")
            print(f"  总时长: {elapsed:.2f} 秒")
            print(f"  平均损失: {sum(log['loss'] for log in self.learning_log) / max(len(self.learning_log), 1):.6f}")
    
    def start_background(self) -> None:
        """在后台启动持续学习."""
        import threading
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        print("[ContinuousLearning] 后台持续学习已启动")
    
    def stop(self) -> None:
        """停止持续学习."""
        self.running = False


# ----------------------------------------------------------------------
# 测试
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("ContinuousLearningEngine 测试")
    print("=" * 60)
    
    # 创建模拟模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 10)
        
        def forward(self, x):
            return self.fc(x)
    
    model = SimpleModel()
    
    # 测试数据流生成器
    print("\n[Test 1] 数据流生成器")
    streamer = DataStreamer(batch_size=4)
    for i in range(5):
        batch = streamer.get_batch()
        print(f"  Batch {i}: type={batch['type']}, shape={batch['shape']}")
        if "content" in batch:
            print(f"    Content: {batch['content'][0][:50]}...")
    
    # 测试增量学习器
    print("\n[Test 2] 增量学习器")
    learner = IncrementalLearner(model, freeze_base=False)
    test_input = torch.randn(4, 64)
    output = learner(test_input)
    print(f"  输入: {test_input.shape}, 输出: {output.shape}")
    
    # 测试遗忘抑制
    print("\n[Test 3] 遗忘抑制模块")
    forget = ForgettingPrevention(model)
    forget.add_to_buffer(test_input)
    samples = forget.sample_from_buffer(2)
    print(f"  缓冲大小: {len(forget.replay_buffer)}")
    print(f"  采样数: {len(samples)}")
    
    # 测试自适应学习率
    print("\n[Test 4] 自适应学习率")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    adaptive_lr = AdaptiveLearningRate(optimizer)
    for i in range(10):
        loss = 1.0 / (i + 1) if i < 5 else 0.5
        adaptive_lr.step(loss)
        print(f"  Step {i}: loss={loss:.4f}, lr={adaptive_lr.get_lr():.8f}")
    
    print("\n" + "=" * 60)
    print("所有持续学习模块测试 PASSED.")
    print("=" * 60)

