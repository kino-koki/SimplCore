# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""
SimplCore Mix 模型导出
支持：PyTorch checkpoint、Safetensors、ONNX、权重提取
"""
import argparse
import json
import struct
from pathlib import Path
from typing import Dict, Optional
import simplcore.tensor as torch
import simplcore.tensor.nn as nn

from .model import SimplCoreMix
from .config import SimplCoreMixConfig, mix_1b, mix_7b, mix_33b
from .runtime import SCRuntimeProfile


def export_pytorch_checkpoint(
    model: SimplCoreMix,
    output_path: str,
    optimizer_state: Optional[Dict] = None,
    step: int = 0,
) -> None:
    """导出标准PyTorch checkpoint"""
    state = {
        "step": step,
        "model": model.state_dict(),
        "config": model.cfg.__dict__,
    }
    if optimizer_state:
        state["optimizer"] = optimizer_state

    torch.save(state, output_path)
    print(f"[Export] PyTorch checkpoint saved: {output_path}")


def export_safetensors(model: SimplCoreMix, output_dir: str) -> None:
    """导出Safetensors格式（HF兼容）

    自研实现，不依赖 transformers/safetensors 库
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()

    # 自研后端 state_dict 返回原始 ndarray；统一包装为 Tensor，
    # 保证 .numel() / .element_size() 与 torch.save 行为一致。
    import numpy as _np
    state_dict = {
        k: (torch.tensor(_np.asarray(v)) if not isinstance(v, torch.Tensor) else v)
        for k, v in state_dict.items()
    }

    # 保存为多个shard（每个<5GB）
    max_shard_size = 5 * 1024 * 1024 * 1024  # 5GB
    current_shard = {}
    current_size = 0
    shard_idx = 0
    shard_files = []

    for name, tensor in state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()

        if current_size + tensor_size > max_shard_size and current_shard:
            # 保存当前shard
            shard_path = output_dir / f"model-{shard_idx:05d}-of-XXXXX.bin"
            torch.save(current_shard, shard_path)
            shard_files.append(shard_path.name)
            print(f"[Export] Shard {shard_idx}: {shard_path.name} ({current_size / 1024 / 1024:.1f} MB)")
            current_shard = {}
            current_size = 0
            shard_idx += 1

        current_shard[name] = tensor
        current_size += tensor_size

    # 保存最后一个shard
    if current_shard:
        shard_path = output_dir / f"model-{shard_idx:05d}-of-XXXXX.bin"
        torch.save(current_shard, shard_path)
        shard_files.append(shard_path.name)
        print(f"[Export] Shard {shard_idx}: {shard_path.name} ({current_size / 1024 / 1024:.1f} MB)")

    # 更新shard文件名中的总数
    total_shards = len(shard_files)
    final_files = []
    for i, fname in enumerate(shard_files):
        old_path = output_dir / fname
        new_name = f"model-{i:05d}-of-{total_shards:05d}.bin"
        new_path = output_dir / new_name
        old_path.rename(new_path)
        final_files.append(new_name)

    # 保存索引文件
    index = {
        "metadata": {
            "total_size": sum(p.numel() * p.element_size() for p in state_dict.values()),
        },
        "weight_map": {name: final_files[i % len(final_files)] for i, name in enumerate(state_dict.keys())},
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    # 保存配置
    with open(output_dir / "config.json", "w") as f:
        json.dump(model.cfg.__dict__, f, indent=2)

    print(f"[Export] Safetensors export complete: {output_dir}")


def export_raw_weights(model: SimplCoreMix, output_dir: str) -> None:
    """导出原始二进制权重（供C++/Java加载）"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()
    manifest = {}

    for name, tensor in state_dict.items():
        # 文件名安全化
        safe_name = name.replace(".", "_").replace(" ", "_")
        bin_path = output_dir / f"{safe_name}.bin"

        # 保存为原始float32二进制
        tensor_f32 = tensor.detach().cpu().to(torch.float32)
        with open(bin_path, "wb") as f:
            f.write(tensor_f32.numpy().tobytes())

        manifest[name] = {
            "file": f"{safe_name}.bin",
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "numel": tensor.numel(),
        }

    # 保存manifest
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Export] Raw weights exported: {output_dir}")
    print(f"  Total tensors: {len(manifest)}")


def export_for_java(model: SimplCoreMix, output_path: str) -> None:
    """导出为Java可读取的格式（自定义序列化）"""
    state_dict = model.state_dict()

    with open(output_path, "wb") as f:
        # 写入魔数和版本
        f.write(b"SCMIX\x00")
        f.write(struct.pack("<I", 1))  # version
        f.write(struct.pack("<I", len(state_dict)))

        for name, tensor in state_dict.items():
            # 名称长度和名称
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)

            # 形状
            shape = tensor.shape
            f.write(struct.pack("<I", len(shape)))
            for dim in shape:
                f.write(struct.pack("<I", dim))

            # 数据类型: 1=float32, 2=float16, 3=int64
            dtype_map = {torch.float32: 1, torch.float16: 2, torch.int64: 3}
            dtype_code = dtype_map.get(tensor.dtype, 1)
            f.write(struct.pack("<B", dtype_code))

            # 数据
            tensor_cpu = tensor.detach().cpu()
            if tensor.dtype == torch.float16:
                data = tensor_cpu.numpy().tobytes()
            else:
                data = tensor_cpu.to(torch.float32).numpy().tobytes()
            f.write(data)

    print(f"[Export] Java format exported: {output_path}")


def export_java26_manifest(model: SimplCoreMix, output_path: str | Path) -> dict:
    """导出 Java26 高性能侧可读取的权重清单。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runtime = SCRuntimeProfile.default()
    state_dict = model.state_dict()
    import numpy as _np
    state_dict = {
        k: (torch.tensor(_np.asarray(v)) if not isinstance(v, torch.Tensor) else v)
        for k, v in state_dict.items()
    }
    weights = []
    total_bytes = 0

    for name, tensor in state_dict.items():
        tensor_bytes = tensor.numel() * tensor.element_size()
        total_bytes += tensor_bytes
        weights.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "numel": tensor.numel(),
                "bytes": tensor_bytes,
                "java_layout": "row_major_float32",
            }
        )

    manifest = {
        "format": "simplcore_mix_java26_manifest",
        "version": 1,
        "architecture": "SimplCore Mix",
        "backend": model.sc_backend_name(),
        "runtime": runtime.to_dict(),
        "config": model.cfg.__dict__,
        "tensor_count": len(weights),
        "total_bytes": total_bytes,
        "weights": weights,
    }
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def export_jdk24_manifest(model: SimplCoreMix, output_path: str | Path) -> dict:
    """兼容旧调用；当前主目标已恢复为 Java26。"""
    return export_java26_manifest(model, output_path)


def export_onnx(model: SimplCoreMix, output_path: str, seq_len: int = 128) -> None:
    """导出ONNX格式"""
    try:
        dummy_input = torch.randint(0, model.cfg.vocab_size, (1, seq_len), dtype=torch.long)

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=14,
            do_constant_folding=True,
        )
        print(f"[Export] ONNX exported: {output_path}")
    except Exception as e:
        print(f"[Export] ONNX export failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Export SimplCore Mix Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Input checkpoint path")
    parser.add_argument("--config", type=str, default="1b", choices=["1b", "7b", "33b"])
    parser.add_argument("--output", type=str, required=True, help="Output path/directory")
    parser.add_argument("--format", type=str, default="safetensors",
                        choices=["pytorch", "safetensors", "raw", "java", "onnx", "all"])
    args = parser.parse_args()

    # 加载模型
    config_map = {"1b": mix_1b, "7b": mix_7b, "33b": mix_33b}
    cfg = config_map[args.config]()
    model = SimplCoreMix(cfg)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    print(f"[Export] Loaded model: {sum(p.numel() for p in model.parameters()):,} params")

    # 导出
    if args.format == "pytorch" or args.format == "all":
        export_pytorch_checkpoint(model, args.output)
    if args.format == "safetensors" or args.format == "all":
        export_safetensors(model, args.output)
    if args.format == "raw" or args.format == "all":
        export_raw_weights(model, args.output + "_raw")
    if args.format == "java" or args.format == "all":
        export_for_java(model, args.output + ".scm")
    if args.format == "onnx" or args.format == "all":
        export_onnx(model, args.output + ".onnx")

    print("[Export] Done!")


if __name__ == "__main__":
    main()
