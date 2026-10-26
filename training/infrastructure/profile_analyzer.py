"""
青小团 70B MoE 分布式训练基础设施
青奈科技 (c) 2026

profile_analyzer.py - 性能分析模块
训练性能分析、瓶颈检测、优化建议、可视化报告
"""

import os
import sys
import json
import time
import math
import logging
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import traceback

import torch
import numpy as np

logger = logging.getLogger("青小团.ProfileAnalyzer")


@dataclass
class StepTiming:
    """单步时间分解"""
    step: int = 0
    total_time_ms: float = 0.0
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    optimizer_time_ms: float = 0.0
    data_load_time_ms: float = 0.0
    communication_time_ms: float = 0.0
    checkpoint_time_ms: float = 0.0
    other_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    samples_per_second: float = 0.0
    gpu_utilization: float = 0.0
    memory_used_gb: float = 0.0
    loss: float = 0.0
    gradient_norm: float = 0.0


@dataclass
class ProfileSummary:
    """性能分析摘要"""
    total_steps_analyzed: int = 0
    avg_step_time_ms: float = 0.0
    avg_forward_time_ms: float = 0.0
    avg_backward_time_ms: float = 0.0
    avg_optimizer_time_ms: float = 0.0
    avg_data_load_time_ms: float = 0.0
    avg_communication_time_ms: float = 0.0
    avg_checkpoint_time_ms: float = 0.0
    avg_tokens_per_second: float = 0.0
    avg_gpu_utilization: float = 0.0
    peak_memory_gb: float = 0.0
    compute_to_comm_ratio: float = 0.0
    data_to_compute_ratio: float = 0.0
    bottleneck: str = ""
    recommendations: List[str] = field(default_factory=list)


class ProfileAnalyzer:
    """
    青小团性能分析器

    分析训练性能、检测瓶颈、生成优化建议和可视化报告
    """

    def __init__(
        self,
        model_name: str = "青小团",
        output_dir: str = "./profile_results",
        window_size: int = 100,
        comm_overhead_threshold: float = 0.3,
        data_overhead_threshold: float = 0.2,
        gpu_idle_threshold: float = 0.3,
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.window_size = window_size
        self.comm_overhead_threshold = comm_overhead_threshold
        self.data_overhead_threshold = data_overhead_threshold
        self.gpu_idle_threshold = gpu_idle_threshold

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._timings: List[StepTiming] = []
        self._profile_history: List[ProfileSummary] = []
        self._current_profile: Optional[ProfileSummary] = None

        self._start_time = time.time()
        self._step_start_time = 0.0
        self._current_timing: Optional[StepTiming] = None
        self._timers: Dict[str, float] = {}
        self._lock = threading.Lock()

        logger.info(f"青小团性能分析器初始化完成")
        logger.info(f"青奈科技 - 输出目录={output_dir}")

    def start_step(self, step: int):
        """开始记录一步"""
        self._step_start_time = time.time()
        self._current_timing = StepTiming(step=step)
        self._timers = {
            "step_start": time.time(),
            "forward_start": 0.0,
            "backward_start": 0.0,
            "optimizer_start": 0.0,
            "data_load_start": 0.0,
        }

    def mark_forward_start(self):
        """标记前向传播开始"""
        if self._current_timing:
            now = time.time()
            self._current_timing.data_load_time_ms = (now - self._step_start_time) * 1000
            self._timers["forward_start"] = now

    def mark_backward_start(self):
        """标记反向传播开始"""
        if self._current_timing:
            now = time.time()
            self._current_timing.forward_time_ms = (now - self._timers["forward_start"]) * 1000
            self._timers["backward_start"] = now

    def mark_optimizer_start(self):
        """标记优化器开始"""
        if self._current_timing:
            now = time.time()
            self._current_timing.backward_time_ms = (now - self._timers["backward_start"]) * 1000
            self._timers["optimizer_start"] = now

    def mark_communication(self, time_ms: float):
        """标记通信耗时"""
        if self._current_timing:
            self._current_timing.communication_time_ms += time_ms

    def mark_checkpoint(self, time_ms: float):
        """标记检查点保存耗时"""
        if self._current_timing:
            self._current_timing.checkpoint_time_ms += time_ms

    def end_step(
        self,
        loss: float = 0.0,
        gradient_norm: float = 0.0,
        tokens_per_second: float = 0.0,
        samples_per_second: float = 0.0,
        gpu_utilization: float = 0.0,
        memory_used_gb: float = 0.0,
    ):
        """
        结束一步记录

        Args:
            loss: 当前loss
            gradient_norm: 梯度范数
            tokens_per_second: 每秒处理token数
            samples_per_second: 每秒处理样本数
            gpu_utilization: GPU利用率
            memory_used_gb: 显存使用
        """
        if self._current_timing is None:
            return

        now = time.time()
        total_time = (now - self._step_start_time) * 1000
        optimizer_time = (now - self._timers.get("optimizer_start", now)) * 1000

        self._current_timing.total_time_ms = total_time
        self._current_timing.optimizer_time_ms = max(0, optimizer_time)
        self._current_timing.other_time_ms = max(0, total_time - (
            self._current_timing.forward_time_ms +
            self._current_timing.backward_time_ms +
            self._current_timing.optimizer_time_ms +
            self._current_timing.data_load_time_ms +
            self._current_timing.communication_time_ms +
            self._current_timing.checkpoint_time_ms
        ))
        self._current_timing.loss = loss
        self._current_timing.gradient_norm = gradient_norm
        self._current_timing.tokens_per_second = tokens_per_second
        self._current_timing.samples_per_second = samples_per_second
        self._current_timing.gpu_utilization = gpu_utilization
        self._current_timing.memory_used_gb = memory_used_gb

        with self._lock:
            self._timings.append(self._current_timing)
            if len(self._timings) > 10000:
                self._timings = self._timings[-5000:]

        self._current_timing = None

    def _get_recent_timings(self, n: Optional[int] = None) -> List[StepTiming]:
        """获取最近的timing记录"""
        n = n or self.window_size
        return self._timings[-min(n, len(self._timings)):]

    def analyze(self, num_steps: Optional[int] = None) -> ProfileSummary:
        """
        分析性能

        Args:
            num_steps: 分析步数，None使用所有

        Returns:
            性能分析摘要
        """
        timings = self._timings[-num_steps:] if num_steps else self._timings
        if not timings:
            return ProfileSummary()

        avg = lambda vals: float(np.mean(vals)) if vals else 0.0
        max_val = lambda vals: float(np.max(vals)) if vals else 0.0

        summary = ProfileSummary(
            total_steps_analyzed=len(timings),
            avg_step_time_ms=avg([t.total_time_ms for t in timings]),
            avg_forward_time_ms=avg([t.forward_time_ms for t in timings]),
            avg_backward_time_ms=avg([t.backward_time_ms for t in timings]),
            avg_optimizer_time_ms=avg([t.optimizer_time_ms for t in timings]),
            avg_data_load_time_ms=avg([t.data_load_time_ms for t in timings]),
            avg_communication_time_ms=avg([t.communication_time_ms for t in timings]),
            avg_checkpoint_time_ms=avg([t.checkpoint_time_ms for t in timings]),
            avg_tokens_per_second=avg([t.tokens_per_second for t in timings]),
            avg_gpu_utilization=avg([t.gpu_utilization for t in timings]),
            peak_memory_gb=max_val([t.memory_used_gb for t in timings]),
        )

        compute_time = summary.avg_forward_time_ms + summary.avg_backward_time_ms
        summary.compute_to_comm_ratio = (
            compute_time / (summary.avg_communication_time_ms + 1e-8)
        )
        summary.data_to_compute_ratio = (
            summary.avg_data_load_time_ms / (compute_time + 1e-8)
        )

        bottleneck, recommendations = self._detect_bottlenecks(summary)
        summary.bottleneck = bottleneck
        summary.recommendations = recommendations

        self._current_profile = summary
        self._profile_history.append(summary)

        self._save_profile_summary(summary)

        return summary

    def _detect_bottlenecks(
        self,
        summary: ProfileSummary,
    ) -> Tuple[str, List[str]]:
        """检测性能瓶颈并生成优化建议"""
        recommendations = []

        comm_ratio = summary.avg_communication_time_ms / (summary.avg_step_time_ms + 1e-8)
        data_ratio = summary.avg_data_load_time_ms / (summary.avg_step_time_ms + 1e-8)
        compute_ratio = (summary.avg_forward_time_ms + summary.avg_backward_time_ms) / (
            summary.avg_step_time_ms + 1e-8
        )
        gpu_idle_ratio = 1.0 - summary.avg_gpu_utilization / 100.0

        ratios = {
            "通信瓶颈": comm_ratio,
            "数据加载瓶颈": data_ratio,
            "计算瓶颈": compute_ratio,
            "GPU空闲": gpu_idle_ratio,
        }

        bottleneck = max(ratios, key=ratios.get)

        if comm_ratio > self.comm_overhead_threshold:
            recommendations.extend([
                "考虑使用梯度压缩减少通信量",
                "启用梯度累积减少通信频率",
                "使用更高效的通信拓扑（如环形All-Reduce）",
                "增加计算通信重叠",
                "考虑使用混合精度训练减少通信量",
            ])

        if data_ratio > self.data_overhead_threshold:
            recommendations.extend([
                "增加DataLoader的num_workers",
                "启用预取和缓存机制",
                "使用更高速的存储（如NVMe SSD）",
                "考虑使用内存映射文件",
                "增加批次大小减少数据加载频率",
            ])

        if compute_ratio < 0.3:
            recommendations.extend([
                "增加模型规模或批次大小",
                "启用Tensor Core加速",
                "使用Flash Attention优化",
                "考虑使用编译器优化（如torch.compile）",
            ])

        if gpu_idle_ratio > self.gpu_idle_threshold:
            recommendations.extend([
                "检查CPU是否成为瓶颈",
                "优化数据预处理流水线",
                "增加数据加载并行度",
                "使用异步数据加载",
                "考虑使用CUDA Graph减少内核启动开销",
            ])

        if summary.avg_gpu_utilization < 50.0:
            recommendations.append("GPU利用率不足50%，建议增大global batch size")

        if summary.avg_tokens_per_second < 1000:
            recommendations.append("吞吐量较低，建议检查模型规模和硬件匹配度")

        return bottleneck, recommendations

    def generate_profile_report(self, output_path: Optional[str] = None) -> str:
        """
        生成性能分析报告

        Args:
            output_path: 输出路径

        Returns:
            报告内容
        """
        summary = self._current_profile or self.analyze()

        report = f"""
{'='*60}
青小团性能分析报告 - 青奈科技
{'='*60}

分析概览:
  分析步数: {summary.total_steps_analyzed}
  模型: {self.model_name}
  分析时间: {datetime.now().isoformat()}

时间分解:
  平均每步耗时: {summary.avg_step_time_ms:.2f} ms
  ├── 前向传播: {summary.avg_forward_time_ms:.2f} ms ({summary.avg_forward_time_ms/summary.avg_step_time_ms*100 if summary.avg_step_time_ms else 0:.1f}%)
  ├── 反向传播: {summary.avg_backward_time_ms:.2f} ms ({summary.avg_backward_time_ms/summary.avg_step_time_ms*100 if summary.avg_step_time_ms else 0:.1f}%)
  ├── 优化器: {summary.avg_optimizer_time_ms:.2f} ms ({summary.avg_optimizer_time_ms/summary.avg_step_time_ms*100 if summary.avg_step_time_ms else 0:.1f}%)
  ├── 数据加载: {summary.avg_data_load_time_ms:.2f} ms ({summary.avg_data_load_time_ms/summary.avg_step_time_ms*100 if summary.avg_step_time_ms else 0:.1f}%)
  ├── 通信: {summary.avg_communication_time_ms:.2f} ms ({summary.avg_communication_time_ms/summary.avg_step_time_ms*100 if summary.avg_step_time_ms else 0:.1f}%)
  ├── 检查点: {summary.avg_checkpoint_time_ms:.2f} ms ({summary.avg_checkpoint_time_ms/summary.avg_step_time_ms*100 if summary.avg_step_time_ms else 0:.1f}%)
  └── 其他: {(summary.avg_step_time_ms - summary.avg_forward_time_ms - summary.avg_backward_time_ms - summary.avg_optimizer_time_ms - summary.avg_data_load_time_ms - summary.avg_communication_time_ms - summary.avg_checkpoint_time_ms):.2f} ms

性能指标:
  平均吞吐量: {summary.avg_tokens_per_second:.2f} tokens/sec
  平均GPU利用率: {summary.avg_gpu_utilization:.1f}%
  峰值显存: {summary.peak_memory_gb:.1f} GB
  计算通信比: {summary.compute_to_comm_ratio:.2f}
  数据计算比: {summary.data_to_compute_ratio:.2f}

瓶颈分析:
  主要瓶颈: {summary.bottleneck}

优化建议:
"""
        for i, rec in enumerate(summary.recommendations, 1):
            report += f"  {i}. {rec}\n"

        report += f"""
{'='*60}
青奈科技 - 青小团分布式训练系统
{'='*60}
"""
        if output_path:
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path_obj, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"青小团 - 性能报告已保存: {output_path}")
        else:
            report_path = self.output_dir / "profile_report.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)

        return report

    def _save_profile_summary(self, summary: ProfileSummary):
        """保存性能分析摘要"""
        summary_file = self.output_dir / f"profile_summary_{int(time.time())}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump({
                "company": "青奈科技",
                "model": self.model_name,
                "timestamp": datetime.now().isoformat(),
                "summary": asdict(summary),
            }, f, ensure_ascii=False, indent=2)

    def compare_profiles(self, profile_paths: List[str]) -> Dict[str, Any]:
        """
        比较多次性能分析结果

        Args:
            profile_paths: 性能分析文件路径列表

        Returns:
            比较结果
        """
        comparisons = []
        for path in profile_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                comparisons.append({
                    "file": Path(path).name,
                    "timestamp": data.get("timestamp", ""),
                    "summary": data.get("summary", {}),
                })
            except Exception as e:
                logger.warning(f"青小团 - 加载性能文件失败 {path}: {e}")

        result = {
            "company": "青奈科技",
            "model": self.model_name,
            "comparison_time": datetime.now().isoformat(),
            "num_profiles": len(comparisons),
            "profiles": comparisons,
        }

        if len(comparisons) >= 2:
            first_throughput = comparisons[0].get("summary", {}).get("avg_tokens_per_second", 0)
            last_throughput = comparisons[-1].get("summary", {}).get("avg_tokens_per_second", 0)
            if first_throughput > 0:
                improvement = (last_throughput - first_throughput) / first_throughput * 100
                result["throughput_improvement"] = f"{improvement:.1f}%"

        comparison_file = self.output_dir / f"profile_comparison_{int(time.time())}.json"
        with open(comparison_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result

    def export_timeline(self, output_path: str):
        """导出时间线数据"""
        timeline = []
        for t in self._timings:
            timeline.append({
                "step": t.step,
                "total_ms": t.total_time_ms,
                "forward_ms": t.forward_time_ms,
                "backward_ms": t.backward_time_ms,
                "optimizer_ms": t.optimizer_time_ms,
                "data_load_ms": t.data_load_time_ms,
                "communication_ms": t.communication_time_ms,
                "checkpoint_ms": t.checkpoint_time_ms,
                "loss": t.loss,
                "tokens_per_sec": t.tokens_per_second,
                "gpu_util": t.gpu_utilization,
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(timeline, f, ensure_ascii=False, indent=2)

        logger.info(f"青小团 - 时间线数据已导出: {output_path}")

    def visualize_flame_graph(self, output_path: str):
        """
        生成火焰图数据

        Args:
            output_path: 输出路径
        """
        recent = self._get_recent_timings(100)
        if not recent:
            return

        flame_data = {
            "name": f"青小团训练 ({self.model_name})",
            "value": sum(t.total_time_ms for t in recent),
            "children": [
                {
                    "name": "前向传播",
                    "value": sum(t.forward_time_ms for t in recent),
                    "children": [],
                },
                {
                    "name": "反向传播",
                    "value": sum(t.backward_time_ms for t in recent),
                    "children": [],
                },
                {
                    "name": "优化器",
                    "value": sum(t.optimizer_time_ms for t in recent),
                    "children": [],
                },
                {
                    "name": "数据加载",
                    "value": sum(t.data_load_time_ms for t in recent),
                    "children": [],
                },
                {
                    "name": "通信",
                    "value": sum(t.communication_time_ms for t in recent),
                    "children": [],
                },
                {
                    "name": "检查点",
                    "value": sum(t.checkpoint_time_ms for t in recent),
                    "children": [],
                },
                {
                    "name": "其他",
                    "value": sum(t.other_time_ms for t in recent),
                    "children": [],
                },
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(flame_data, f, ensure_ascii=False, indent=2)

    def get_performance_trend(self) -> Dict[str, Any]:
        """获取性能趋势"""
        if len(self._timings) < 10:
            return {}

        chunk_size = max(1, len(self._timings) // 20)
        chunks = [
            self._timings[i:i + chunk_size]
            for i in range(0, len(self._timings), chunk_size)
        ]

        trend = []
        for chunk in chunks:
            if chunk:
                trend.append({
                    "step": chunk[-1].step,
                    "avg_step_time_ms": float(np.mean([t.total_time_ms for t in chunk])),
                    "avg_tokens_per_sec": float(np.mean([t.tokens_per_second for t in chunk])),
                    "avg_gpu_util": float(np.mean([t.gpu_utilization for t in chunk])),
                    "avg_loss": float(np.mean([t.loss for t in chunk])),
                })

        first = trend[0] if trend else {}
        last = trend[-1] if trend else {}

        return {
            "trend": trend,
            "throughput_change": (
                (last.get("avg_tokens_per_sec", 0) - first.get("avg_tokens_per_sec", 0))
                / (first.get("avg_tokens_per_sec", 1) + 1e-8) * 100
            ),
            "step_time_change": (
                (last.get("avg_step_time_ms", 0) - first.get("avg_step_time_ms", 0))
                / (first.get("avg_step_time_ms", 1) + 1e-8) * 100
            ),
        }

    def visualize(self, output_dir: Optional[str] = None):
        """
        生成可视化图表

        Args:
            output_dir: 输出目录
        """
        out_dir = Path(output_dir or self.output_dir / "visualizations")
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            timings = self._timings[-500:]
            if not timings:
                return

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f"青小团训练性能分析 - {self.model_name}", fontsize=14)

            steps = [t.step for t in timings]

            axes[0, 0].plot(steps, [t.total_time_ms for t in timings], alpha=0.7)
            axes[0, 0].set_xlabel("步数")
            axes[0, 0].set_ylabel("耗时(ms)")
            axes[0, 0].set_title("每步耗时")
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].plot(steps, [t.tokens_per_second for t in timings], color="green", alpha=0.7)
            axes[0, 1].set_xlabel("步数")
            axes[0, 1].set_ylabel("tokens/sec")
            axes[0, 1].set_title("吞吐量")
            axes[0, 1].grid(True, alpha=0.3)

            axes[0, 2].plot(steps, [t.gpu_utilization for t in timings], color="purple", alpha=0.7)
            axes[0, 2].set_xlabel("步数")
            axes[0, 2].set_ylabel("利用率(%)")
            axes[0, 2].set_title("GPU利用率")
            axes[0, 2].grid(True, alpha=0.3)

            recent = timings[-100:]
            if recent:
                avg_times = {
                    "前向": float(np.mean([t.forward_time_ms for t in recent])),
                    "反向": float(np.mean([t.backward_time_ms for t in recent])),
                    "优化器": float(np.mean([t.optimizer_time_ms for t in recent])),
                    "数据加载": float(np.mean([t.data_load_time_ms for t in recent])),
                    "通信": float(np.mean([t.communication_time_ms for t in recent])),
                }
                labels = list(avg_times.keys())
                values = list(avg_times.values())
                colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
                axes[1, 0].pie(values, labels=labels, colors=colors[:len(labels)],
                               autopct="%1.1f%%", startangle=90)
                axes[1, 0].set_title("时间分解(最近100步)")

            axes[1, 1].plot(steps, [t.loss for t in timings], color="red", alpha=0.5)
            if len(timings) > 10:
                smoothed = np.convolve([t.loss for t in timings],
                                       np.ones(10) / 10, mode="valid")
                axes[1, 1].plot(steps[9:], smoothed, color="darkred", linewidth=2)
            axes[1, 1].set_xlabel("步数")
            axes[1, 1].set_ylabel("Loss")
            axes[1, 1].set_title("Loss曲线")
            axes[1, 1].grid(True, alpha=0.3)

            memory_vals = [t.memory_used_gb for t in timings]
            axes[1, 2].hist(memory_vals, bins=20, alpha=0.7, color="orange")
            axes[1, 2].set_xlabel("显存(GB)")
            axes[1, 2].set_ylabel("频次")
            axes[1, 2].set_title("显存使用分布")
            axes[1, 2].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = out_dir / "performance_analysis.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(f"青小团 - 性能可视化已保存: {plot_path}")

            self._generate_time_breakdown_chart(timings, out_dir)

        except ImportError:
            logger.warning("青小团 - matplotlib不可用，跳过可视化")
        except Exception as e:
            logger.warning(f"青小团 - 可视化失败: {e}")

    def _generate_time_breakdown_chart(
        self,
        timings: List[StepTiming],
        out_dir: Path,
    ):
        """生成时间分解堆叠图"""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            steps = [t.step for t in timings]
            fig, ax = plt.subplots(figsize=(14, 6))

            ax.stackplot(
                steps,
                [t.forward_time_ms for t in timings],
                [t.backward_time_ms for t in timings],
                [t.optimizer_time_ms for t in timings],
                [t.data_load_time_ms for t in timings],
                [t.communication_time_ms for t in timings],
                labels=["前向", "反向", "优化器", "数据加载", "通信"],
                alpha=0.8,
                colors=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"],
            )

            ax.set_xlabel("步数")
            ax.set_ylabel("耗时(ms)")
            ax.set_title("青小团训练 - 时间分解堆叠图")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

            plot_path = out_dir / "time_breakdown.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

        except Exception:
            pass

    def reset(self):
        """重置性能分析器"""
        self._timings.clear()
        self._profile_history.clear()
        self._current_profile = None
        logger.info("青小团 - 性能分析器已重置")

    def close(self):
        """关闭性能分析器"""
        if self._timings:
            final_report = self.generate_profile_report(
                str(self.output_dir / "final_profile_report.txt")
            )
            self.export_timeline(
                str(self.output_dir / "timeline.json")
            )
            self.visualize()
            logger.info(f"青小团 - 性能分析器已关闭")
