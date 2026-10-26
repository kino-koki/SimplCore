#!/usr/bin/env python3
"""摄像头视觉测试: 让1M模型真正"看到"真实世界.

流程:
  1. 检查/安装 opencv-python
  2. 启动摄像头捕获实时帧
  3. 使用 TinyViT 模型对每帧进行分类推理
  4. 显示摄像头画面 + 推理结果

注意: 需要在独立终端运行, 因为涉及 GUI 窗口显示.
"""

import sys, os, time, subprocess, traceback

sys.path.insert(0, '.')

# ----------------------------------------------------------------------
# 检查依赖
# ----------------------------------------------------------------------
try:
    import cv2
    print("[OK] opencv-python 已安装")
except ImportError:
    print("[INFO] 正在安装 opencv-python...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "opencv-python",
            "--quiet"
        ])
        import cv2
        print("[OK] opencv-python 安装成功")
    except Exception as e:
        print(f"[ERROR] 安装失败: {e}")
        sys.exit(1)

import torch
import torch.nn as nn
import numpy as np

# ----------------------------------------------------------------------
# 加载1M TinyViT模型
# ----------------------------------------------------------------------
print("\n[INFO] 加载1M TinyViT模型...")

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
print(f"[OK] TinyViT模型加载成功: {total_params:,} 参数 ({total_params/1e6:.2f}M)")

# 合成数据的类别名称 (训练时使用的模式)
CLASS_NAMES = [
    "红色系", "橙红色", "橙黄色", "黄绿色", 
    "绿色系", "青绿色", "青蓝色", "蓝紫色", 
    "紫色系", "紫红色"
]

# ----------------------------------------------------------------------
# 摄像头捕获
# ----------------------------------------------------------------------
print("\n[INFO] 启动摄像头...")

camera_id = 0
cap = cv2.VideoCapture(camera_id)

if not cap.isOpened():
    print(f"[ERROR] 无法打开摄像头 #{camera_id}")
    print("请检查:")
    print("  1. 摄像头是否已连接")
    print("  2. 设备ID是否正确 (0=内置, 1=外接)")
    print("  3. 是否有其他程序占用摄像头")
    sys.exit(1)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"[OK] 摄像头 #{camera_id} 已启动: {actual_w}×{actual_h} @ {actual_fps:.1f}fps")

# ----------------------------------------------------------------------
# 主循环
# ----------------------------------------------------------------------
print("\n[INFO] 开始实时视觉推理...")
print("按 'q' 退出")

frame_count = 0
start_time = time.time()
fps_display = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] 无法读取帧")
        break

    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed > 1.0:
        fps_display = frame_count / elapsed
        frame_count = 0
        start_time = time.time()

    # 预处理: 缩小到 32x32 (模型输入尺寸), BGR→RGB, 归一化
    img_small = cv2.resize(frame, (32, 32))
    img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_conf = probs[0, pred_idx].item()

    pred_name = CLASS_NAMES[pred_idx]

    # 在画面上显示结果
    display_frame = frame.copy()
    display_w, display_h = display_frame.shape[1], display_frame.shape[0]
    
    # 显示标题
    cv2.putText(display_frame, "SimplCore Vision Test", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # 显示推理结果
    result_text = f"分类: {pred_name} (置信度: {pred_conf:.2f})"
    cv2.putText(display_frame, result_text, 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    # 显示帧率
    cv2.putText(display_frame, f"FPS: {fps_display:.1f}", 
                (10, display_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 显示小图预览
    small_display = cv2.resize(img_small, (128, 128))
    display_frame[-140:-12, -140:-12] = small_display

    cv2.imshow("SimplCore Vision - 1M TinyViT", display_frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------------------------------------------------
# 清理
# ----------------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
print("\n[INFO] 测试结束")

