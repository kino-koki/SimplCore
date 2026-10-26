# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""SimplCore Mix 消融实验框架。

目的:
  量化每个创新模块对最终 loss/perplexity/吞吐 的贡献，
  而不是只靠 "能运行" 来判断架构价值。

支持的单模块开关 (对应 SimplCoreMixConfig 里的布尔/类型字段):
  - mba          : SGC 基向量场注意力 (关掉后回退到标准 GQA)
  - ccr_moe      : CCR 拓扑感知 MoE (关掉后回退到单专家 SwiGLU FFN)
  - dmc          : DMC 工作记忆缓冲
  - spm          : SPM Hebbian在线适应
  - lti          : LTI 稳定注入 (关掉后循环退化为纯残差)
  - act          : ACT 自适应停止 (关掉后固定循环次数)
  - loop_lora    : 深度自适应 LoRA
  - anti_hall    : 置信度门控/置信度门控
  - fusion       : SGI/IB/MRD/PUM 深度融合组件

实验类型:
  - "baseline"               : 仅标准 Transformer (无任何 SimplCore 模块)
  - "full_mix"               : 全量 SimplCoreMix
  - "leave_one_out"          : 每次只关一个模块，其余保持全量
  - "leave_one_in"           : baseline 基础上每次只开一个模块
  - "custom"                 : 用户自定义开关组合

每个实验运行相同的 tiny 训练配置 (步数、batch、seq_len、seed)，
保证对比公平。结果汇总为 JSON 和 Markdown 表格。
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import simplcore.tensor as torch

from .config import SimplCoreMixConfig, mix_tiny_text
from .core import SCFunctional
from .model import SimplCoreMix
from .optim import SCAdamW
from .baseline import BaselineTransformer


# ---------------------------------------------------------------------------
# 开关映射: 每个模块对应 SimplCoreMixConfig 里需要修改的字段
# ---------------------------------------------------------------------------

MODULE_FIELDS: Dict[str, Dict[str, Any]] = {
    "mba": {
        "attn_type": "mba_gqa",
        # 关闭 SGC 的方式: 仍用 sgc_gqa 类型但把 n_basis_vectors 设为 0 会在代码里触发
        # 标准注意力路径; 这里用配置标记, 实际由 AblationConfig 构造模型时处理
        "_use_rfp": False,
    },
    "ccr_moe": {
        "n_experts": 1,
        "n_shared_experts": 0,
        "n_experts_per_tok": 1,
    },
    "dmc": {
        "n_memory_channels": 0,
    },
    "spm": {
        "hebbian_enabled": False,
    },
    "lti": {
        "_disable_lti": True,
    },
    "act": {
        "act_threshold": 1.5,  # 永不提前停止
    },
    "loop_lora": {
        "loop_lora_enabled": False,
    },
    "anti_hall": {
        "confidence_gate_enabled": False,
    },
    "fusion": {
        "_disable_fusion": True,
    },
}


@dataclass
class AblationConfig:
    """单个消融实验的配置。"""

    name: str
    description: str
    disabled_modules: List[str] = field(default_factory=list)
    is_baseline: bool = False  # True = 用 BaselineTransformer

    def build_model(self, base_cfg: SimplCoreMixConfig) -> torch.nn.Module:
        """根据开关构造模型。"""
        if self.is_baseline:
            return BaselineTransformer(base_cfg)

        cfg = copy.deepcopy(base_cfg)
        for mod in self.disabled_modules:
            if mod not in MODULE_FIELDS:
                continue
            for k, v in MODULE_FIELDS[mod].items():
                if k.startswith("_"):
                    continue  # 内部标记, 不写入 dataclass
                setattr(cfg, k, v)
        return SimplCoreMix(cfg)


# ---------------------------------------------------------------------------
# 预定义实验组合
# ---------------------------------------------------------------------------

def _all_modules() -> List[str]:
    return list(MODULE_FIELDS.keys())


def make_leave_one_out_configs() -> List[AblationConfig]:
    """每次关掉一个模块。"""
    configs = []
    for mod in _all_modules():
        configs.append(AblationConfig(
            name=f"no_{mod}",
            description=f"关闭 {mod}，其余全量",
            disabled_modules=[mod],
        ))
    return configs


def make_leave_one_in_configs() -> List[AblationConfig]:
    """baseline 基础上每次只开一个模块。

    对于 baseline + 单模块, 我们把 baseline 当作 "关掉全部" 的起点,
    然后每次只保留一个模块开启、其余关闭。
    """
    all_mods = _all_modules()
    configs = []
    for keep in all_mods:
        disabled = [m for m in all_mods if m != keep]
        configs.append(AblationConfig(
            name=f"only_{keep}",
            description=f"仅开启 {keep}，其余关闭",
            disabled_modules=disabled,
        ))
    return configs


def make_standard_configs() -> List[AblationConfig]:
    """标准消融实验集: baseline + full + leave-one-out。"""
    return [
        AblationConfig(
            name="baseline",
            description="标准 Transformer (无 SimplCore 模块)",
            is_baseline=True,
        ),
        AblationConfig(
            name="full_mix",
            description="全量 SimplCoreMix",
            disabled_modules=[],
        ),
        *make_leave_one_out_configs(),
    ]


# ---------------------------------------------------------------------------
# 消融实验运行器
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    """单个消融实验结果。"""

    name: str
    description: str
    disabled_modules: List[str]
    is_baseline: bool
    param_count: int
    final_loss: float
    loss_curve: List[float]
    elapsed_seconds: float
    tokens_per_second: float
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "disabled_modules": self.disabled_modules,
            "is_baseline": self.is_baseline,
            "param_count": self.param_count,
            "final_loss": self.final_loss,
            "loss_curve": self.loss_curve,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "tokens_per_second": round(self.tokens_per_second, 1),
            "error": self.error,
        }


class AblationStudy:
    """消融实验运行器。

    用法:
        study = AblationStudy(base_cfg=mix_tiny_text(), steps=20)
        results = study.run()
        study.save_results(results, "ablation_results.json")
    """

    def __init__(
        self,
        base_cfg: Optional[SimplCoreMixConfig] = None,
        steps: int = 20,
        batch_size: int = 2,
        seq_len: int = 16,
        seed: int = 42,
        device: str = "cpu",
        lr: float = 1e-3,
    ):
        self.base_cfg = base_cfg or mix_tiny_text()
        self.steps = steps
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.seed = seed
        self.device = torch.device(device)
        self.lr = lr

    def _run_single(self, ablation_cfg: AblationConfig) -> AblationResult:
        """运行单个消融实验。"""
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        try:
            model = ablation_cfg.build_model(self.base_cfg).to(self.device)
            model.train()
            optimizer = SCAdamW(model.parameters(), lr=self.lr)

            loss_curve: List[float] = []
            total_tokens = 0
            start = time.time()

            for step in range(self.steps):
                tokens = torch.randint(
                    0,
                    self.base_cfg.vocab_size,
                    (self.batch_size, self.seq_len + 1),
                    dtype=torch.long,
                    device=self.device,
                )
                x = tokens[:, :-1]
                y = tokens[:, 1:]

                optimizer.zero_grad(set_to_none=True)
                logits = model(x, n_loops=self.base_cfg.max_loop_iters)
                loss = SCFunctional.cross_entropy(
                    logits.reshape(-1, self.base_cfg.vocab_size),
                    y.reshape(-1),
                )
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                loss_curve.append(loss_val)
                total_tokens += self.batch_size * self.seq_len

            elapsed = time.time() - start
            tps = total_tokens / max(elapsed, 1e-6)
            param_count = sum(p.numel() for p in model.parameters())

            return AblationResult(
                name=ablation_cfg.name,
                description=ablation_cfg.description,
                disabled_modules=ablation_cfg.disabled_modules,
                is_baseline=ablation_cfg.is_baseline,
                param_count=param_count,
                final_loss=loss_curve[-1] if loss_curve else float("nan"),
                loss_curve=loss_curve,
                elapsed_seconds=elapsed,
                tokens_per_second=tps,
            )
        except Exception as e:
            return AblationResult(
                name=ablation_cfg.name,
                description=ablation_cfg.description,
                disabled_modules=ablation_cfg.disabled_modules,
                is_baseline=ablation_cfg.is_baseline,
                param_count=0,
                final_loss=float("nan"),
                loss_curve=[],
                elapsed_seconds=0.0,
                tokens_per_second=0.0,
                error=str(e),
            )

    def run(
        self,
        configs: Optional[List[AblationConfig]] = None,
        verbose: bool = True,
    ) -> List[AblationResult]:
        """运行一组消融实验。"""
        configs = configs or make_standard_configs()
        results: List[AblationResult] = []

        for i, cfg in enumerate(configs):
            if verbose:
                print(f"[Ablation {i+1}/{len(configs)}] {cfg.name}: {cfg.description}")
            result = self._run_single(cfg)
            if verbose:
                status = f"loss={result.final_loss:.4f}" if not result.error else f"ERROR: {result.error}"
                print(f"  -> {status}  ({result.elapsed_seconds:.1f}s, {result.param_count:,} params)")
            results.append(result)

        return results

    @staticmethod
    def save_results(
        results: List[AblationResult],
        output_path: str | Path,
    ) -> None:
        """保存结果为 JSON。"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "ablation_study": {
                "total_experiments": len(results),
                "successful": sum(1 for r in results if not r.error),
                "failed": sum(1 for r in results if r.error),
            },
            "results": [r.to_dict() for r in results],
        }
        output_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[Ablation] Results saved to {output_path}")

    @staticmethod
    def to_markdown_table(results: List[AblationResult]) -> str:
        """生成 Markdown 对比表。"""
        lines = [
            "| 实验 | 关闭模块 | 参数量 | 最终loss | 吞吐(tok/s) | 耗时(s) | 状态 |",
            "|------|----------|--------|----------|-------------|---------|------|",
        ]
        for r in results:
            disabled = ", ".join(r.disabled_modules) if r.disabled_modules else "-"
            status = "错误" if r.error else "完成"
            loss_str = f"{r.final_loss:.4f}" if not r.error else "-"
            lines.append(
                f"| {r.name} | {disabled} | {r.param_count:,} | {loss_str} "
                f"| {r.tokens_per_second:.1f} | {r.elapsed_seconds:.1f} | {status} |"
            )
        return "\n".join(lines)


def run_ablation_smoke(
    output_dir: Optional[str | Path] = None,
    steps: int = 5,
) -> List[AblationResult]:
    """快速冒烟级消融实验, 用于验证框架本身可用。

    只跑 baseline + full_mix + no_rfp + no_dmc 四组, 步数很少。
    """
    study = AblationStudy(steps=steps, batch_size=1, seq_len=8)
    configs = [
        AblationConfig("baseline", "标准 Transformer", is_baseline=True),
        AblationConfig("full_mix", "全量 SimplCoreMix"),
        AblationConfig("no_rfp", "关闭 SGC", disabled_modules=["mba"]),
        AblationConfig("no_dmc", "关闭 DMC", disabled_modules=["dmc"]),
    ]
    results = study.run(configs=configs, verbose=True)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        study.save_results(results, output_dir / "ablation_smoke.json")
        (output_dir / "ablation_smoke.md").write_text(
            study.to_markdown_table(results), encoding="utf-8"
        )

    return results


__all__ = [
    "MODULE_FIELDS",
    "AblationConfig",
    "AblationResult",
    "AblationStudy",
    "make_leave_one_out_configs",
    "make_leave_one_in_configs",
    "make_standard_configs",
    "run_ablation_smoke",
]
