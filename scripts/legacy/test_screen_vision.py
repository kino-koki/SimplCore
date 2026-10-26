#!/usr/bin/env python3
"""屏幕视觉测试: 让1M模型"看到"屏幕画面.

由于 Python 3.13 没有 opencv-python wheel, 使用 mss + PIL 替代.
流程:
  1. 使用 mss 捕获屏幕帧
  2. 使用 PIL 显示画面 + 推理结果
  3. 使用 TinyViT 模型进行颜色分类推理
"""

import sys, os, time
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mss
import mss.tools

# ----------------------------------------------------------------------
# 加载1M TinyViT模型
# ----------------------------------------------------------------------
print("[INFO] 加载1M TinyViT模型...")

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyViT(img_size=32, patch_size=4, dim=128, depth=6, num_heads=8, num_classes=10)
model = model.to(device).eval()
total_params = sum(p.numel() for p in model.parameters())
print(f"[OK] TinyViT模型: {total_params:,} 参数 ({total_params/1e6:.2f}M)")

CLASS_NAMES = [
    "红色系", "橙红色", "橙黄色", "黄绿色", 
    "绿色系", "青绿色", "青蓝色", "蓝紫色", 
    "紫色系", "紫红色"
]

# ----------------------------------------------------------------------
# 屏幕捕获设置
# ----------------------------------------------------------------------
print("\n[INFO] 初始化屏幕捕获...")
sct = mss.mss()
monitor = sct.monitors[1]
print(f"[OK] 屏幕分辨率: {monitor['width']}×{monitor['height']}")

# ----------------------------------------------------------------------
# 主循环
# ----------------------------------------------------------------------
print("\n[INFO] 开始屏幕视觉推理...")
print("按 Ctrl+C 退出")

frame_count = 0
start_time = time.time()
fps_display = 0.0
display_size = (640, 480)

try:
    while True:
        # 捕获屏幕
        sct_img = sct.grab(monitor)
        img = Image.frombytes('RGB', (sct_img.width, sct_img.height), sct_img.rgb)
        
        # 预处理: 缩小到 32x32, 归一化
        img_small = img.resize((32, 32))
        img_np = np.array(img_small)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # 推理
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_conf = probs[0, pred_idx].item()
        
        pred_name = CLASS_NAMES[pred_idx]
        
        # 计算帧率
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            start_time = time.time()
        
        # 准备显示画面
        display_img = img.resize(display_size)
        draw = ImageDraw.Draw(display_img)
        
        # 显示标题
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 10), "SimplCore Vision Test", fill=(0, 255, 0), font=font)
        
        # 显示推理结果
        result_text = f"分类: {pred_name} ({pred_conf:.2f})"
        draw.text((10, 40), result_text, fill=(255, 255, 0), font=font)
        
        # 显示帧率
        fps_text = f"FPS: {fps_display:.1f}"
        draw.text((10, display_size[1] - 30), fps_text, fill=(0, 255, 255), font=font)
        
        # 显示小图预览
        small_preview = img_small.resize((128, 128))
        display_img.paste(small_preview, (display_size[0] - 140, display_size[1] - 140))
        
        # 显示
        display_img.show(title="SimplCore Vision", command=None)
        
except KeyboardInterrupt:
    print("\n[INFO] 用户退出")
    sct.close()
    print("[INFO] 测试结束")

