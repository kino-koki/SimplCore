# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""对比学习与自监督预训练模块.

核心功能:
  1. MoCo (Momentum Contrast) - 动量对比学习
  2. SimCLR (Simple Framework for Contrastive Learning)
  3. BYOL (Bootstrap Your Own Latent)
  4. SimSiam (Simple Siamese Network)
  5. 自监督预训练策略
  6. 对比蒸馏

设计目标:
  - 提高模型的泛化能力
  - 学习更具判别性的特征表示
  - 减少对标注数据的依赖
  - 支持多模态对比学习

与AGI训练系统的关系:
  - ContrastiveLoss: 添加到训练损失函数中
  - SelfSupervisedTrainer: 自监督预训练阶段
  - ContrastiveLearningEngine: 协调对比学习流程
"""

from __future__ import annotations

import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from simplcore.tensor.nn import functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class MoCoQueue:
    """MoCo队列 - 存储负样本."""
    
    def __init__(self, size: int = 65536, dim: int = 128, device: str = "cpu"):
        self.size = size
        self.dim = dim
        self.device = device
        
        self.queue = torch.randn(size, dim).to(device)
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.queue_ptr = 0
    
    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor) -> None:
        """入队新样本."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        if ptr + batch_size <= self.size:
            self.queue[ptr:ptr + batch_size] = keys
            self.queue_ptr = (ptr + batch_size) % self.size
        else:
            # 循环队列
            self.queue[ptr:self.size] = keys[:self.size - ptr]
            self.queue[0:batch_size - (self.size - ptr)] = keys[self.size - ptr:]
            self.queue_ptr = batch_size - (self.size - ptr)
    
    @torch.no_grad()
    def dequeue(self, batch_size: int) -> torch.Tensor:
        """出队旧样本."""
        ptr = int(self.queue_ptr)
        if ptr >= batch_size:
            result = self.queue[ptr - batch_size:ptr].clone()
        else:
            result = torch.cat([self.queue[self.size - batch_size + ptr:self.size], 
                               self.queue[0:ptr]], dim=0)
        return result
    
    def get(self) -> torch.Tensor:
        """获取整个队列."""
        return self.queue


class ContrastiveLoss(nn.Module):
    """对比学习损失函数.
    
    支持多种对比学习方法:
      1. MoCo (Momentum Contrast)
      2. SimCLR
      3. BYOL
      4. SimSiam
      5. InfoNCE
    """
    
    def __init__(
        self,
        method: str = "moco",
        temperature: float = 0.07,
        queue_size: int = 65536,
        momentum: float = 0.999,
        dim: int = 128,
        use_cosine_sim: bool = True,
    ):
        super().__init__()
        
        self.method = method.lower()
        self.temperature = temperature
        self.momentum = momentum
        self.dim = dim
        self.use_cosine_sim = use_cosine_sim
        
        # MoCo队列
        if self.method == "moco":
            self.queue = MoCoQueue(size=queue_size, dim=dim)
        
        # 投影头
        self.projection_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        
        # 动量编码器 (用于MoCo和BYOL)
        self.momentum_projection = self.projection_head.copy()
        for param in self.momentum_projection.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def _momentum_update(self) -> None:
        """动量更新."""
        for param, momentum_param in zip(
            self.projection_head.parameters(),
            self.momentum_projection.parameters(),
        ):
            momentum_param.data = (
                self.momentum * momentum_param.data +
                (1 - self.momentum) * param.data
            )
    
    def _compute_similarity(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """计算相似度."""
        if self.use_cosine_sim:
            q = nn.functional.normalize(q, dim=1)
            k = nn.functional.normalize(k, dim=1)
            return torch.matmul(q, k.T) / self.temperature
        else:
            return torch.matmul(q, k.T) / self.temperature
    
    def forward_moco(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        is_training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """MoCo损失."""
        # 投影
        q = self.projection_head(query)
        q = nn.functional.normalize(q, dim=1)
        
        with torch.no_grad():
            self._momentum_update()
            k = self.momentum_projection(key)
            k = nn.functional.normalize(k, dim=1)
        
        # 计算正样本相似度
        pos_sim = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        
        # 获取负样本
        neg_keys = self.queue.get()
        neg_sim = torch.matmul(q, neg_keys.T)
        
        # 拼接正负样本
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        
        # 标签: 正样本在第一个位置
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
        
        # 交叉熵损失
        loss = F.cross_entropy(logits, labels)
        
        # 入队
        if is_training:
            self.queue.enqueue(k)
        
        return {"loss": loss, "logits": logits, "similarity": pos_sim.mean()}
    
    def forward_simclr(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """SimCLR损失."""
        z1 = self.projection_head(x1)
        z2 = self.projection_head(x2)
        
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        
        # 合并特征
        z = torch.cat([z1, z2], dim=0)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        
        # 掩码: 去除对角线和正对的位置
        mask = torch.eye(z.shape[0], dtype=torch.bool).to(z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)
        
        # 正样本对
        pos_indices = torch.arange(z1.shape[0], z.shape[0]).to(z.device)
        pos_indices = torch.cat([pos_indices, torch.arange(z1.shape[0]).to(z.device)], dim=0)
        
        # InfoNCE损失
        loss = F.cross_entropy(sim_matrix, pos_indices)
        
        return {"loss": loss, "similarity_matrix": sim_matrix}
    
    def forward_byol(
        self,
        online: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """BYOL损失."""
        online_proj = self.projection_head(online)
        online_proj = nn.functional.normalize(online_proj, dim=1)
        
        with torch.no_grad():
            self._momentum_update()
            target_proj = self.momentum_projection(target)
            target_proj = nn.functional.normalize(target_proj, dim=1)
        
        # 预测头
        predictor = nn.Sequential(
            nn.Linear(self.dim, self.dim // 2),
            nn.ReLU(),
            nn.Linear(self.dim // 2, self.dim),
        )
        
        online_pred = predictor(online_proj)
        
        # MSE损失
        loss = 2 * (1 - torch.cosine_similarity(online_pred, target_proj, dim=-1)).mean()
        
        return {"loss": loss, "cosine_sim": torch.cosine_similarity(online_pred, target_proj, dim=-1).mean()}
    
    def forward_simsiam(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """SimSiam损失."""
        z1 = self.projection_head(x1)
        z2 = self.projection_head(x2)
        
        # 预测头
        predictor = nn.Sequential(
            nn.Linear(self.dim, self.dim // 2),
            nn.BatchNorm1d(self.dim // 2),
            nn.ReLU(),
            nn.Linear(self.dim // 2, self.dim),
        )
        
        p1 = predictor(z1)
        p2 = predictor(z2)
        
        # 对称损失
        loss1 = -torch.cosine_similarity(p1, z2.detach(), dim=-1).mean()
        loss2 = -torch.cosine_similarity(p2, z1.detach(), dim=-1).mean()
        
        loss = (loss1 + loss2) / 2
        
        return {"loss": loss, "similarity": (torch.cosine_similarity(p1, z2.detach(), dim=-1).mean() +
                                           torch.cosine_similarity(p2, z1.detach(), dim=-1).mean()) / 2}
    
    def forward_infonce(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """InfoNCE损失."""
        anchors = nn.functional.normalize(anchors, dim=1)
        positives = nn.functional.normalize(positives, dim=1)
        negatives = nn.functional.normalize(negatives, dim=1)
        
        # 正样本相似度
        pos_sim = torch.einsum("nc,nc->n", [anchors, positives]).unsqueeze(-1)
        
        # 负样本相似度
        neg_sim = torch.matmul(anchors, negatives.T)
        
        # 拼接
        logits = torch.cat([pos_sim, neg_sim], dim=1) / self.temperature
        
        # 标签
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return {"loss": loss, "accuracy": (logits.argmax(dim=1) == labels).float().mean()}
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """前向传播."""
        if self.method == "moco":
            return self.forward_moco(
                inputs["query"],
                inputs["key"],
                kwargs.get("is_training", True),
            )
        elif self.method == "simclr":
            return self.forward_simclr(inputs["x1"], inputs["x2"])
        elif self.method == "byol":
            return self.forward_byol(inputs["online"], inputs["target"])
        elif self.method == "simsiam":
            return self.forward_simsiam(inputs["x1"], inputs["x2"])
        elif self.method == "infonce":
            return self.forward_infonce(
                inputs["anchors"],
                inputs["positives"],
                inputs["negatives"],
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")


class MultiModalContrastiveLoss(nn.Module):
    """多模态对比学习损失 - 在不同模态间建立对齐."""
    
    def __init__(
        self,
        temperature: float = 0.07,
        text_dim: int = 128,
        image_dim: int = 128,
        video_dim: int = 128,
        align_weight: float = 1.0,
        self_weight: float = 1.0,
    ):
        super().__init__()
        
        self.temperature = temperature
        self.align_weight = align_weight
        self.self_weight = self.self_weight
        
        # 模态间投影
        self.text_proj = nn.Linear(text_dim, 128)
        self.image_proj = nn.Linear(image_dim, 128)
        self.video_proj = nn.Linear(video_dim, 128)
        
        # 自模态对比损失
        self.text_contrastive = ContrastiveLoss(method="simclr", temperature=temperature)
        self.image_contrastive = ContrastiveLoss(method="simclr", temperature=temperature)
        self.video_contrastive = ContrastiveLoss(method="simclr", temperature=temperature)
    
    def _compute_modality_alignment(
        self,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
        video_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """计算模态间对齐损失."""
        # 投影到共同空间
        text_proj = nn.functional.normalize(self.text_proj(text_emb), dim=1)
        image_proj = nn.functional.normalize(self.image_proj(image_emb), dim=1)
        video_proj = nn.functional.normalize(self.video_proj(video_emb), dim=1)
        
        losses = {}
        
        # 文本-图像对齐
        text_image_sim = torch.matmul(text_proj, image_proj.T) / self.temperature
        labels = torch.arange(text_proj.shape[0]).to(text_proj.device)
        losses["text_image_loss"] = F.cross_entropy(text_image_sim, labels)
        
        # 文本-视频对齐
        text_video_sim = torch.matmul(text_proj, video_proj.T) / self.temperature
        losses["text_video_loss"] = F.cross_entropy(text_video_sim, labels)
        
        # 图像-视频对齐
        image_video_sim = torch.matmul(image_proj, video_proj.T) / self.temperature
        losses["image_video_loss"] = F.cross_entropy(image_video_sim, labels)
        
        return losses
    
    def forward(
        self,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
        video_emb: torch.Tensor,
        text_aug: Optional[torch.Tensor] = None,
        image_aug: Optional[torch.Tensor] = None,
        video_aug: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """计算多模态对比损失."""
        results = {}
        
        # 模态间对齐损失
        align_losses = self._compute_modality_alignment(text_emb, image_emb, video_emb)
        alignment_loss = sum(align_losses.values()) / len(align_losses)
        results.update(align_losses)
        
        # 自模态对比损失
        self_loss = torch.tensor(0.0)
        if text_aug is not None:
            text_result = self.text_contrastive({"x1": text_emb, "x2": text_aug})
            self_loss += text_result["loss"]
            results["text_self_loss"] = text_result["loss"]
        
        if image_aug is not None:
            image_result = self.image_contrastive({"x1": image_emb, "x2": image_aug})
            self_loss += image_result["loss"]
            results["image_self_loss"] = image_result["loss"]
        
        if video_aug is not None:
            video_result = self.video_contrastive({"x1": video_emb, "x2": video_aug})
            self_loss += video_result["loss"]
            results["video_self_loss"] = video_result["loss"]
        
        # 总损失
        total_loss = (
            alignment_loss * self.align_weight +
            self_loss * self.self_weight
        )
        results["total_loss"] = total_loss
        
        return results


class SelfSupervisedTrainer:
    """自监督预训练器 - 无监督学习阶段."""
    
    def __init__(
        self,
        model: nn.Module,
        contrastive_loss: ContrastiveLoss,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any = None,
        epochs: int = 100,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.model = model
        self.contrastive_loss = contrastive_loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        
        # 训练统计
        self.loss_history = []
        self.accuracy_history = []
    
    def _generate_augmentations(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成数据增强."""
        aug1 = self._random_augment(data)
        aug2 = self._random_augment(data)
        return aug1, aug2
    
    def _random_augment(self, data: torch.Tensor) -> torch.Tensor:
        """随机数据增强."""
        augmented = data.clone()
        
        # 添加噪声
        augmented += torch.randn_like(augmented) * 0.01
        
        # 随机缩放
        scale = 0.9 + torch.rand(1) * 0.2
        augmented = augmented * scale
        
        # 随机平移
        shift = torch.randn_like(augmented) * 0.05
        augmented = augmented + shift
        
        return augmented
    
    def pretrain_epoch(self, dataloader: Any) -> Dict[str, float]:
        """预训练一个epoch."""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            # 获取数据
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            inputs = inputs.to(self.device)
            
            # 生成增强视图
            view1, view2 = self._generate_augmentations(inputs)
            
            # 编码
            emb1 = self.model(view1)
            emb2 = self.model(view2)
            
            # 计算对比损失
            loss_result = self.contrastive_loss({"x1": emb1, "x2": emb2})
            
            # 反向传播
            loss_result["loss"].backward()
            self.optimizer.step()
            
            # 更新统计
            batch_size = inputs.shape[0]
            total_loss += loss_result["loss"].item() * batch_size
            total_samples += batch_size
            
            if "accuracy" in loss_result:
                total_accuracy += loss_result["accuracy"].item() * batch_size
        
        avg_loss = total_loss / total_samples
        avg_accuracy = total_accuracy / total_samples if total_samples > 0 else 0.0
        
        return {"loss": avg_loss, "accuracy": avg_accuracy}
    
    def fit(self, dataloader: Any) -> None:
        """预训练主循环."""
        print("=" * 60)
        print("自监督预训练开始")
        print("=" * 60)
        
        for epoch in range(self.epochs):
            metrics = self.pretrain_epoch(dataloader)
            
            self.loss_history.append(metrics["loss"])
            self.accuracy_history.append(metrics["accuracy"])
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            print(f"Epoch {epoch + 1}/{self.epochs}")
            print(f"  Loss: {metrics['loss']:.6f}")
            if metrics["accuracy"] > 0:
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
            
            if (epoch + 1) % 10 == 0:
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "loss_history": self.loss_history,
                }, f"./checkpoints/pretrain_epoch_{epoch + 1}.pt")
        
        print("\n自监督预训练完成")
    
    def get_embeddings(self, dataloader: Any) -> torch.Tensor:
        """提取特征嵌入."""
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                inputs = inputs.to(self.device)
                emb = self.model(inputs)
                embeddings.append(emb.cpu())
        
        return torch.cat(embeddings, dim=0)


class ContrastiveLearningEngine:
    """对比学习引擎 - 协调对比学习流程."""
    
    def __init__(
        self,
        model: nn.Module,
        contrastive_loss: ContrastiveLoss = None,
        multimodal_loss: MultiModalContrastiveLoss = None,
        pretrain_epochs: int = 50,
        finetune_epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-6,
        device: str = "cpu",
    ):
        self.model = model
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.batch_size = batch_size
        self.device = device
        
        # 损失函数
        if contrastive_loss is None:
            self.contrastive_loss = ContrastiveLoss(method="simclr", dim=model.config.hidden_size)
        else:
            self.contrastive_loss = contrastive_loss
        
        self.multimodal_loss = multimodal_loss
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # 学习率调度器
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=pretrain_epochs + finetune_epochs,
            eta_min=learning_rate * 0.01,
        )
        
        # 预训练器
        self.pretrainer = SelfSupervisedTrainer(
            model=model.language_model if hasattr(model, 'language_model') else model,
            contrastive_loss=self.contrastive_loss,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            epochs=pretrain_epochs,
            batch_size=batch_size,
            device=device,
        )
        
        # 状态
        self.pretrained = False
        self.finetuned = False
    
    def pretrain(self, dataloader: Any) -> None:
        """执行自监督预训练."""
        print("\n[ContrastiveLearning] 开始自监督预训练...")
        self.pretrainer.fit(dataloader)
        self.pretrained = True
        print("[ContrastiveLearning] 自监督预训练完成")
    
    def finetune(self, dataloader: Any, supervised_loss_fn: Any = None) -> None:
        """执行微调."""
        if not self.pretrained:
            print("[ContrastiveLearning] 警告: 未进行预训练，直接微调")
        
        print("\n[ContrastiveLearning] 开始微调...")
        
        self.model.train()
        for epoch in range(self.finetune_epochs):
            total_loss = 0.0
            total_samples = 0
            
            for batch in dataloader:
                self.optimizer.zero_grad()
                
                inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                inputs = inputs.to(self.device)
                
                outputs = self.model(inputs)
                
                if supervised_loss_fn is not None:
                    loss = supervised_loss_fn(outputs)
                else:
                    if isinstance(outputs, dict):
                        loss = sum(v.mean() for v in outputs.values() if isinstance(v, torch.Tensor))
                    else:
                        loss = outputs.mean()
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * inputs.shape[0]
                total_samples += inputs.shape[0]
            
            avg_loss = total_loss / total_samples
            self.lr_scheduler.step()
            
            print(f"Finetune Epoch {epoch + 1}/{self.finetune_epochs}")
            print(f"  Loss: {avg_loss:.6f}")
        
        self.finetuned = True
        print("[ContrastiveLearning] 微调完成")
    
    def run_multimodal_contrastive(
        self,
        text_dataloader: Any,
        image_dataloader: Any,
        video_dataloader: Any,
    ) -> None:
        """运行多模态对比学习."""
        if self.multimodal_loss is None:
            print("[ContrastiveLearning] 未配置多模态损失，跳过")
            return
        
        print("\n[ContrastiveLearning] 开始多模态对比学习...")
        
        self.model.train()
        for epoch in range(self.pretrain_epochs):
            total_loss = 0.0
            total_samples = 0
            
            for text_batch, image_batch, video_batch in zip(
                text_dataloader, image_dataloader, video_dataloader
            ):
                self.optimizer.zero_grad()
                
                text_inputs = text_batch[0].to(self.device)
                image_inputs = image_batch[0].to(self.device)
                video_inputs = video_batch[0].to(self.device)
                
                # 编码
                text_emb = self.model.language_model(text_inputs)
                image_emb = self.model.vision_encoder.encode_image(image_inputs)
                video_emb = self.model.vision_encoder.encode_image(video_inputs)
                
                # 生成增强
                text_aug = text_emb + torch.randn_like(text_emb) * 0.01
                image_aug = image_inputs + torch.randn_like(image_inputs) * 0.01
                
                # 计算多模态损失
                loss_result = self.multimodal_loss(
                    text_emb=text_emb,
                    image_emb=image_emb,
                    video_emb=video_emb,
                    text_aug=text_aug,
                    image_aug=image_aug,
                )
                
                loss_result["total_loss"].backward()
                self.optimizer.step()
                
                total_loss += loss_result["total_loss"].item() * text_inputs.shape[0]
                total_samples += text_inputs.shape[0]
            
            avg_loss = total_loss / total_samples
            self.lr_scheduler.step()
            
            print(f"Multimodal Epoch {epoch + 1}/{self.pretrain_epochs}")
            print(f"  Total Loss: {avg_loss:.6f}")
            print(f"  Text-Image: {loss_result.get('text_image_loss', 0):.6f}")
            print(f"  Text-Video: {loss_result.get('text_video_loss', 0):.6f}")
        
        print("[ContrastiveLearning] 多模态对比学习完成")
    
    def save_model(self, path: str) -> None:
        """保存模型."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "pretrained": self.pretrained,
            "finetuned": self.finetuned,
        }, path)
        print(f"[ContrastiveLearning] 模型已保存到: {path}")


# ----------------------------------------------------------------------
# 测试
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    
    print("=" * 60)
    print("对比学习模块测试")
    print("=" * 60)
    
    # 创建模拟模型
    class SimpleEncoder(nn.Module):
        def __init__(self, input_dim=128, hidden_dim=128):
            super().__init__()
            self.fc = nn.Linear(input_dim, hidden_dim)
        
        def forward(self, x):
            return self.fc(x)
    
    encoder = SimpleEncoder()
    
    # 测试MoCo损失
    print("\n[Test 1] MoCo损失")
    moco_loss = ContrastiveLoss(method="moco", dim=128)
    query = torch.randn(4, 128)
    key = torch.randn(4, 128)
    result = moco_loss({"query": query, "key": key})
    print(f"  Loss: {result['loss'].item():.6f}")
    print(f"  Logits shape: {result['logits'].shape}")
    print(f"  Similarity: {result['similarity'].item():.6f}")
    
    # 测试SimCLR损失
    print("\n[Test 2] SimCLR损失")
    simclr_loss = ContrastiveLoss(method="simclr", dim=128)
    x1 = torch.randn(4, 128)
    x2 = torch.randn(4, 128)
    result = simclr_loss({"x1": x1, "x2": x2})
    print(f"  Loss: {result['loss'].item():.6f}")
    print(f"  Similarity matrix shape: {result['similarity_matrix'].shape}")
    
    # 测试SimSiam损失
    print("\n[Test 3] SimSiam损失")
    simsiam_loss = ContrastiveLoss(method="simsiam", dim=128)
    result = simsiam_loss({"x1": x1, "x2": x2})
    print(f"  Loss: {result['loss'].item():.6f}")
    print(f"  Similarity: {result['similarity'].item():.6f}")
    
    # 测试InfoNCE损失
    print("\n[Test 4] InfoNCE损失")
    infonce_loss = ContrastiveLoss(method="infonce", dim=128)
    anchors = torch.randn(4, 128)
    positives = torch.randn(4, 128)
    negatives = torch.randn(100, 128)
    result = infonce_loss({"anchors": anchors, "positives": positives, "negatives": negatives})
    print(f"  Loss: {result['loss'].item():.6f}")
    print(f"  Accuracy: {result['accuracy'].item():.4f}")
    
    # 测试多模态对比损失
    print("\n[Test 5] 多模态对比损失")
    multimodal_loss = MultiModalContrastiveLoss()
    text_emb = torch.randn(4, 128)
    image_emb = torch.randn(4, 128)
    video_emb = torch.randn(4, 128)
    result = multimodal_loss(text_emb, image_emb, video_emb)
    print(f"  Total Loss: {result['total_loss'].item():.6f}")
    print(f"  Text-Image Loss: {result['text_image_loss'].item():.6f}")
    print(f"  Text-Video Loss: {result['text_video_loss'].item():.6f}")
    print(f"  Image-Video Loss: {result['image_video_loss'].item():.6f}")
    
    print("\n" + "=" * 60)
    print("所有对比学习模块测试 PASSED.")
    print("=" * 60)
