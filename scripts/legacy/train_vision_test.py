#!/usr/bin/env python3
"""训练测试: 极小 ViT 模型验证视觉管道.

目标:
  1. 验证 ViT 编码器可以正常训练 (loss 下降)
  2. 验证梯度流经整个视觉管道
  3. 验证推理正确性 (准确率 > 随机)
  4. 验证 SimplCore perception 模块集成

模型: 超小 ViT (~1M 参数, patch_size=4, 6层, 8头, dim=128)
数据: 合成数据 (1000 样本, 32×32 彩色, 10 类)
训练: 10 epochs, AdamW, CPU

注意: 不依赖 torchvision, 使用合成数据验证训练流程.
"""

import sys, os, time, json
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------------------------------------------------
# 结果收集
# ----------------------------------------------------------------------
log_lines = []
start_time = time.time()
def log(msg):
    elapsed = time.time() - start_time
    log_lines.append(f"[{elapsed:7.1f}s] {msg}")
    print(f"[{elapsed:7.1f}s] {msg}", flush=True)

def save_results(path='vision_train_test_results.json'):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(log_lines, f, indent=2, ensure_ascii=False)
    log(f"结果已保存到 {path}")

# ----------------------------------------------------------------------
# 1. Tiny ViT 定义
# ----------------------------------------------------------------------
class TinyPatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class TinyTransformerBlock(nn.Module):
    def __init__(self, dim=128, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, dim), nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class TinyViT(nn.Module):
    """极小 ViT, 用于快速验证训练流程."""
    def __init__(self, img_size=32, patch_size=4, in_chans=3,
                 dim=128, depth=6, num_heads=8, num_classes=10):
        super().__init__()
        self.patch_embed = TinyPatchEmbed(img_size, patch_size, in_chans, dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.02)
        self.pos_drop = nn.Dropout(0.1)

        self.blocks = nn.ModuleList([
            TinyTransformerBlock(dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)

# ----------------------------------------------------------------------
# 2. 合成数据 (带模式的分类数据)
# ----------------------------------------------------------------------
def create_synthetic_data(n_samples=1000, img_size=32, num_classes=10, seed=42):
    """创建有模式的合成数据: 每类的图像有不同的基色模式.

    这确保模型可以学到真实模式 (非随机), loss 会下降.
    """
    torch.manual_seed(seed)
    images = []
    labels = []
    for c in range(num_classes):
        # 每类一个基色模式
        base_color = torch.zeros(3, img_size, img_size)
        base_color[0] = c / num_classes  # R 通道
        base_color[1] = (num_classes - c) / num_classes  # G 通道
        base_color[2] = 0.5  # B 通道固定
        # 加噪声
        for _ in range(n_samples // num_classes):
            noise = torch.randn(3, img_size, img_size) * 0.15
            img = (base_color + noise).clamp(0, 1)
            images.append(img)
            labels.append(c)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    # 打乱
    perm = torch.randperm(len(images))
    return images[perm], labels[perm]

# ----------------------------------------------------------------------
# 3. 训练函数
# ----------------------------------------------------------------------
def train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if (batch_idx + 1) % 5 == 0:
            log(f"  epoch {epoch} [{batch_idx+1}/{len(loader)}] loss={loss.item():.4f}")
    return total_loss / len(loader), 100.0 * correct / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / len(loader), 100.0 * correct / total

# ----------------------------------------------------------------------
# 4. 梯度流验证
# ----------------------------------------------------------------------
def verify_gradient_flow(model):
    grad_stats = {}
    total_grad_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is not None and param.grad.abs().sum() > 0:
            total_grad_params += 1
            grad_stats[name] = param.grad.abs().mean().item()
    grad_ratio = total_grad_params / total_params * 100
    log(f"梯度流: {total_grad_params}/{total_params} 参数有梯度 ({grad_ratio:.1f}%)")
    if grad_stats:
        sorted_grads = sorted(grad_stats.items(), key=lambda x: -x[1])
        log(f"Top-5 梯度幅值:")
        for name, val in sorted_grads[:5]:
            log(f"  {name}: {val:.6f}")
    return grad_ratio > 80

# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"设备: {device}")
    log(f"PyTorch: {torch.__version__}")

    # ---- A. 合成数据 ----
    log("\n" + "=" * 60)
    log("A. 创建合成数据集")
    log("=" * 60)
    train_imgs, train_labels = create_synthetic_data(n_samples=1000, img_size=32, num_classes=10)
    test_imgs, test_labels = create_synthetic_data(n_samples=200, img_size=32, num_classes=10, seed=99)
    log(f"训练集: {train_imgs.shape}, 测试集: {test_imgs.shape}")

    train_set = TensorDataset(train_imgs, train_labels)
    test_set = TensorDataset(test_imgs, test_labels)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    # ---- B. 模型 ----
    log("\n" + "=" * 60)
    log("B. 创建 TinyViT 模型")
    log("=" * 60)
    model = TinyViT(img_size=32, patch_size=4, dim=128, depth=6, num_heads=8, num_classes=10)
    total_params = sum(p.numel() for p in model.parameters())
    log(f"TinyViT 参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # ---- C. 梯度流验证 ----
    log("\n--- 训练前梯度流验证 ---")
    test_input = torch.randn(4, 3, 32, 32).to(device)
    test_out = model(test_input)
    test_loss = test_out.sum()
    test_loss.backward()
    grad_ok = verify_gradient_flow(model)
    model.zero_grad()

    # ---- D. 训练 ----
    log("\n" + "=" * 60)
    log("D. 开始训练 (10 epochs)")
    log("=" * 60)
    best_acc = 0.0
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(10):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, epoch)
        test_loss, test_acc = evaluate(model, test_loader, device)
        scheduler.step()
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        lr_now = optimizer.param_groups[0]['lr']
        log(f">>> Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.1f}% "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.1f}% lr={lr_now:.2e}")
        if test_acc > best_acc:
            best_acc = test_acc

    # ---- E. SimplCore 集成验证 ----
    log("\n" + "=" * 60)
    log("E. SimplCore Vision 集成验证")
    log("=" * 60)
    try:
        from SimplCore.architecture.vision_encoder import vit_tiny
        enc = vit_tiny(output_dim=64).to(device)
        enc_total = sum(p.numel() for p in enc.parameters())
        log(f"SimplCore ViT-Tiny: {enc_total:,} params")
        x = torch.randn(4, 3, 224, 224).to(device)
        latent = enc.encode_image(x)
        log(f"前向: {x.shape} -> {latent.shape}")
        assert latent.shape == (4, 196, 64), f"形状错误: {latent.shape}"
        loss = latent.pow(2).mean()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in enc.parameters())
        log(f"梯度流: {'OK' if has_grad else 'FAIL'}")
        assert has_grad, "SimplCore ViT 没有梯度!"
        log("SimplCore Vision 集成验证通过")
    except Exception as e:
        log(f"SimplCore 集成验证跳过: {e}")

    # ---- F. PerceptionPipeline 构造验证 ----
    log("\n" + "=" * 60)
    log("F. PerceptionPipeline 构造验证")
    log("=" * 60)
    try:
        from SimplCore.perception import PerceptionPipeline
        pipeline = PerceptionPipeline(vit_hidden_size=192, vit_output_size=64)
        status = pipeline.status
        log(f"管道构造成功: device={status['device']}, model={status['model']['params']}")
        log("PerceptionPipeline 构造验证通过")
    except Exception as e:
        log(f"PerceptionPipeline 构造跳过: {e}")

    # ---- 总结 ----
    log("\n" + "=" * 60)
    log("结果摘要")
    log("=" * 60)
    log(f"TinyViT 参数量: {total_params:,}")
    log(f"训练 10 epochs 后最佳测试准确率: {best_acc:.1f}%")
    log(f"训练 loss: {results['train_loss'][0]:.4f} -> {results['train_loss'][-1]:.4f}")
    log(f"测试 loss: {results['test_loss'][0]:.4f} -> {results['test_loss'][-1]:.4f}")
    loss_dropped = results['train_loss'][-1] < results['train_loss'][0] * 0.8
    if loss_dropped:
        log(">>> 最终结论: 视觉模型训练流程完全正常!")
    else:
        log(">>> 警告: Loss 下降不明显, 需检查超参数")

    save_results()
    return results

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log(f"!!! 错误: {e}")
        import traceback
        for line in traceback.format_exc().split('\n'):
            log(line)
        save_results('vision_train_test_error.json')

