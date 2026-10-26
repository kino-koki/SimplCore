# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""
SimplCore Mix 评估系统
支持：Perplexity、生成质量评估、下游任务benchmark
"""
import json
import math
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from simplcore.tensor.utils.data import DataLoader

from .model import SimplCoreMix
from .config import SimplCoreMixConfig


class Evaluator:
    """模型评估器"""

    def __init__(self, model: SimplCoreMix, tokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def compute_perplexity(self, dataloader: DataLoader, max_batches: Optional[int] = None) -> float:
        """计算困惑度(Perplexity)"""
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        for batch_idx, (x, y) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                reduction="sum",
                ignore_index=-100,
            )

            total_loss += loss.item()
            total_tokens += (y != -100).sum().item()
            num_batches += 1

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(avg_loss)
        return perplexity

    @torch.no_grad()
    def evaluate_generation(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        n_loops: int = 8,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> List[Dict]:
        """评估生成质量"""
        results = []

        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt)
            input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)

            start_time = time.time()
            generated = self.model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                n_loops=n_loops,
                temperature=temperature,
                top_k=top_k,
            )
            elapsed = time.time() - start_time

            output_tokens = generated[0, len(tokens):].tolist()
            output_text = self.tokenizer.decode(output_tokens)

            results.append({
                "prompt": prompt,
                "output": output_text,
                "input_tokens": len(tokens),
                "output_tokens": len(output_tokens),
                "time_seconds": elapsed,
                "tokens_per_second": len(output_tokens) / max(elapsed, 1e-6),
            })

        return results

    @torch.no_grad()
    def benchmark_throughput(
        self,
        batch_sizes: List[int] = [1, 2, 4, 8],
        seq_len: int = 512,
        n_loops: int = 8,
        num_runs: int = 10,
    ) -> Dict:
        """基准测试吞吐量"""
        results = {}

        for bs in batch_sizes:
            tokens = torch.randint(0, self.model.cfg.vocab_size, (bs, seq_len), device=self.device)

            # Warmup
            for _ in range(3):
                _ = self.model(tokens, n_loops=n_loops)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Benchmark
            start = time.time()
            for _ in range(num_runs):
                _ = self.model(tokens, n_loops=n_loops)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - start

            total_tokens = bs * seq_len * num_runs
            throughput = total_tokens / elapsed
            latency = elapsed / num_runs * 1000  # ms

            results[f"batch_{bs}"] = {
                "throughput_tokens_per_sec": throughput,
                "latency_ms": latency,
            }

        return results

    @torch.no_grad()
    def evaluate_diversity(self, prompts: List[str], n_samples: int = 5, max_new_tokens: int = 50) -> Dict:
        """评估生成多样性（self-BLEU近似）"""
        from collections import Counter

        all_outputs = []
        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt)
            input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)

            samples = []
            for _ in range(n_samples):
                generated = self.model.generate(
                    input_tensor,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                    top_k=50,
                )
                output_tokens = generated[0, len(tokens):].tolist()
                output_text = self.tokenizer.decode(output_tokens)
                samples.append(output_text)
            all_outputs.append(samples)

        # 计算多样性：不同输出之间的token重叠度
        diversity_scores = []
        for samples in all_outputs:
            if len(samples) < 2:
                continue
            total_distinct = 0
            total_pairs = 0
            for i in range(len(samples)):
                for j in range(i + 1, len(samples)):
                    tokens_i = set(self.tokenizer.encode(samples[i]))
                    tokens_j = set(self.tokenizer.encode(samples[j]))
                    jaccard = len(tokens_i & tokens_j) / max(len(tokens_i | tokens_j), 1)
                    total_distinct += 1 - jaccard
                    total_pairs += 1
            if total_pairs > 0:
                diversity_scores.append(total_distinct / total_pairs)

        avg_diversity = sum(diversity_scores) / max(len(diversity_scores), 1)
        return {
            "diversity_score": avg_diversity,
            "num_prompts": len(prompts),
            "samples_per_prompt": n_samples,
        }

    def run_full_eval(self, eval_config: Dict, output_path: Optional[str] = None) -> Dict:
        """运行完整评估"""
        print("=" * 60)
        print("SimplCore Mix Evaluation")
        print("=" * 60)

        results = {
            "model_params": self.model.get_param_count(),
            "config": self.model.cfg.__dict__,
        }

        # Perplexity
        if "perplexity" in eval_config:
            print("\n[Eval] Computing Perplexity...")
            from .data_loader import build_data_loader
            dataloader = build_data_loader(
                eval_config["perplexity"]["data_path"],
                self.tokenizer,
                batch_size=eval_config["perplexity"].get("batch_size", 4),
                max_seq_len=self.model.cfg.max_seq_len,
            )
            ppl = self.compute_perplexity(dataloader, eval_config["perplexity"].get("max_batches"))
            results["perplexity"] = ppl
            print(f"  Perplexity: {ppl:.4f}")

        # Generation
        if "generation" in eval_config:
            print("\n[Eval] Evaluating Generation...")
            gen_results = self.evaluate_generation(
                eval_config["generation"]["prompts"],
                max_new_tokens=eval_config["generation"].get("max_new_tokens", 100),
                n_loops=eval_config["generation"].get("n_loops", 8),
            )
            results["generation"] = gen_results
            for r in gen_results:
                print(f"  Prompt: {r['prompt'][:50]}...")
                print(f"  Output: {r['output'][:100]}...")
                print(f"  Speed: {r['tokens_per_second']:.1f} tok/s")

        # Throughput
        if "throughput" in eval_config:
            print("\n[Eval] Benchmarking Throughput...")
            throughput = self.benchmark_throughput()
            results["throughput"] = throughput
            for k, v in throughput.items():
                print(f"  {k}: {v['throughput_tokens_per_sec']:.1f} tok/s, {v['latency_ms']:.1f} ms/batch")

        # Diversity
        if "diversity" in eval_config:
            print("\n[Eval] Evaluating Diversity...")
            diversity = self.evaluate_diversity(
                eval_config["diversity"]["prompts"],
                n_samples=eval_config["diversity"].get("n_samples", 5),
            )
            results["diversity"] = diversity
            print(f"  Diversity Score: {diversity['diversity_score']:.4f}")

        # Factual accuracy
        if "factual" in eval_config:
            print("\n[Eval] Evaluating Factual Accuracy...")
            factual_eval = FactualEvaluator(self.model, self.tokenizer, self.device)
            factual_results = factual_eval.evaluate(
                eval_config["factual"]["qa_pairs"],
                max_new_tokens=eval_config["factual"].get("max_new_tokens", 100),
                n_loops=eval_config["factual"].get("n_loops", 8),
            )
            results["factual"] = factual_results
            print(f"  Accuracy: {factual_results['accuracy']:.2%}")
            print(f"  Refusal Rate: {factual_results['refusal_rate']:.2%}")

        # Save results
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n[Eval] Results saved to {output_path}")

        return results


class FactualEvaluator:
    """事实性评测器。

    目的:
      "置信度门控" (原置信度门控) 只是一个 hidden-state gate,
      不能直接证明模型减少了事实性幻觉。真正的事实性评测需要
      用 QA 对检查模型输出是否与已知答案一致。

    评测方式:
      1. 给模型一个事实性问题。
      2. 收集模型生成结果。
      3. 检查生成结果是否包含正确答案 (substring match)。
      4. 统计准确率、拒答率、平均生成长度。

    拒答判定:
      如果输出包含 "不知道"、"无法确定"、"不确定" 等短语,
      或输出过短 (< 5 字符), 视为拒答。

    注意:
      substring match 是最简单的事实性评测, 不等同于完整的事实性验证。
      真正的生产级事实性评测需要人工标注、检索增强、引用验证等。
      这里提供一个可自动运行的基线, 让消融实验有可比指标。
    """

    REFUSAL_PATTERNS = [
        "不知道", "无法确定", "不确定", "不清楚", "不了解",
        "无法回答", "没有相关信息", "我不确定",
        "don't know", "not sure", "cannot determine",
    ]

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def _is_refusal(self, text: str) -> bool:
        """判断是否为拒答。"""
        text_lower = text.lower().strip()
        if len(text_lower) < 5:
            return True
        for pattern in self.REFUSAL_PATTERNS:
            if pattern in text_lower:
                return True
        return False

    def _check_answer(self, generated: str, expected: str) -> bool:
        """检查生成结果是否包含正确答案 (substring match)。"""
        return expected.lower() in generated.lower()

    @torch.no_grad()
    def evaluate(
        self,
        qa_pairs: List[Dict[str, str]],
        max_new_tokens: int = 100,
        n_loops: int = 8,
        temperature: float = 0.1,
    ) -> Dict:
        """
        Args:
            qa_pairs: [{"question": "...", "answer": "..."}, ...]
        Returns:
            {"accuracy": float, "refusal_rate": float, "total": int, "details": [...]}
        """
        results = []
        correct = 0
        refusals = 0

        for qa in qa_pairs:
            question = qa["question"]
            expected = qa["answer"]

            tokens = self.tokenizer.encode(question)
            input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)

            generated = self.model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                n_loops=n_loops,
                temperature=temperature,
                top_k=50,
            )

            output_tokens = generated[0, len(tokens):].tolist()
            output_text = self.tokenizer.decode(output_tokens).strip()

            is_refusal = self._is_refusal(output_text)
            is_correct = self._check_answer(output_text, expected) if not is_refusal else False

            if is_refusal:
                refusals += 1
            if is_correct:
                correct += 1

            results.append({
                "question": question,
                "expected": expected,
                "generated": output_text,
                "correct": is_correct,
                "refusal": is_refusal,
            })

        total = len(qa_pairs)
        return {
            "accuracy": correct / max(total, 1),
            "refusal_rate": refusals / max(total, 1),
            "correct_count": correct,
            "refusal_count": refusals,
            "total": total,
            "details": results,
        }
