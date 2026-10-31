# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Phase 1 训练执行器 —— packed 语料 → 课程窗口流 → StreamingTrainer 的端到端。

上一层的 ``trainer.py`` 是**驱动**（课程推进/分块 BPTT/回滚闸门），本模块是
**执行器**：把真实（或合成演示）的 packed 语料按课程档位切窗喂进去，跑完
课程、按预算评估、落报告。二者分离的理由与 trainer 相同——换数据源、换
后端、换课程都不该互相改代码。

数据侧的诚实约定（对应战役闸门 P0-budget-frozen 的数据部分）
------------------------------------------------------------
* **语料指纹**：报告携带 ``index.json`` 的 SHA-256 与各 split 的 token 数；
  语料一旦冻结，指纹变了就是换了数据，账对不上。
* **无泄漏是结构性的**：train/val 各自从 manifest 声明的分片读（分片文件
  集合必须不相交，加载时强校验）；评估只碰 val 分片，train 窗口在结构上
  进不了评估路径。
* **token 流连续消费**：课程从 L=16 涨到 L=64 时，游标按 token 推进、
  窗口重切不回头——同一条 token 流不重读不漏读；读完即换 epoch（计数
  入报告），不存在"课程变长就偷偷重采样"的空间。
* **评估口径固定**：no-regression gate 的评估一律用固定 ``eval_seq_len``
  （默认=最短档）的 val 批，跨步比较才有意义；报告记录该口径。
* **宣称许可**：合成语料跑出的结果只证明管线语义正确——``claim_permitted``
  必须为 False 且报告标注 ``synthetic: true``；外部宣称只能出自冻结的
  真实语料 + 评估改善同时成立。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .data import MANIFEST_NAME, compute_sha256, read_manifest
from .trainer import ContextCurriculum, CurriculumStage, StreamingTrainer

__all__ = [
    "CorpusBatches",
    "SyntheticCorpusSpec",
    "make_synthetic_corpus",
    "run_phase1",
    "tiny_curriculum",
    "tiny_model",
    "main",
]


# ---------------------------------------------------------------------------
# 合成语料（走 data.py 同一套 manifest 格式，mmap/哈希/复算全部真实生效）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SyntheticCorpusSpec:
    vocab_size: int
    seq_len: int                 # 语料打包窗口长（与课程档无关，token 流才是本源）
    train_tokens: int
    val_tokens: int
    seed: int


def _synthetic_document(rng: np.random.Generator, vocab_size: int) -> List[int]:
    """一篇可学习的文档：若干短周期节律拼接，模型数得出 → loss 可观测下降。"""
    doc: List[int] = []
    for _ in range(int(rng.integers(3, 7))):
        period = int(rng.integers(2, 9))
        start = int(rng.integers(0, vocab_size))
        n = int(rng.integers(period, period * 8))
        doc.extend((start + period * np.arange(n)) % vocab_size)
    return doc


def make_synthetic_corpus(out_dir: Any, spec: SyntheticCorpusSpec) -> Dict[str, Any]:
    """生成合成 packed 语料：``<split>.bin``（uint32 LE）+ ``index.json``。

    格式与 :func:`data.pack_documents` 完全一致（同一 manifest 读取器可读），
    区别仅在文档来源是确定性节律生成器——这足以让"管线语义正确"可验证，
    且报告会被标注 synthetic。
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    eos_id = spec.vocab_size - 1          # 词表最后一个作 EOS，正文不会用到
    body_vocab = eos_id                    # 正文 token ∈ [0, eos_id)

    splits: Dict[str, Dict[str, Any]] = {}
    for split, n_tokens in (("train", spec.train_tokens), ("val", spec.val_tokens)):
        rng = np.random.default_rng(spec.seed + (0 if split == "train" else 1))
        stream: List[int] = []
        while len(stream) < n_tokens + 1:
            doc = _synthetic_document(rng, body_vocab)
            stream.extend(doc)
            stream.append(eos_id)
        arr = np.asarray(stream[:n_tokens], dtype=np.uint32)
        fname = f"{split}.bin"
        (out_path / fname).write_bytes(arr.tobytes())
        n_windows = n_tokens // spec.seq_len
        splits[split] = {
            "n_docs": int(np.count_nonzero(arr == eos_id)),
            "n_tokens": int(n_tokens),
            "n_tokens_encoded": int(n_tokens),
            "n_windows": int(n_windows),
            "remainder_tokens": int(n_tokens % spec.seq_len),
            "bytes": int(arr.nbytes),
            "elapsed_seconds": 0.0,
            "files": [{"path": fname, "n_tokens": int(n_tokens),
                       "sha256": compute_sha256(out_path / fname)}],
        }
        if n_windows < 1:
            raise ValueError(f"{split} token 预算不足以切出一窗（需 ≥ {spec.seq_len}）")

    manifest = {
        "format": "simplcore-token-pack-v1",
        "synthetic": True,
        "generator_seed": int(spec.seed),
        "seq_len": int(spec.seq_len),
        "dtype": "uint32",
        "vocab_size": int(spec.vocab_size),
        "eos_id": int(eos_id),
        "total_tokens": int(spec.train_tokens + spec.val_tokens),
        "splits": splits,
    }
    mpath = out_path / MANIFEST_NAME
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# 课程窗口流（连续 token 游标；换档重切不回头）
# ---------------------------------------------------------------------------
class CorpusBatches:
    """对 packed 语料的定长窗口供给器：游标按 token 推进，跨课程档连续。"""

    def __init__(self, corpus_root: Any, split: str = "train") -> None:
        self.root = Path(corpus_root)
        self.manifest = read_manifest(self.root)
        entry = self.manifest.get("splits", {}).get(split)
        if not entry:
            raise KeyError(f"split {split!r} not in manifest {self.root / MANIFEST_NAME}")
        self.split = split
        self.vocab_size = int(self.manifest.get("vocab_size", 0) or 0)
        self.eos_id = int(self.manifest.get("eos_id", -1))
        self._maps: List[np.memmap] = []
        self._bounds: List[int] = []
        total = 0
        for f in entry["files"]:
            path = self.root / f["path"]
            n = int(f.get("n_tokens", 0))
            self._maps.append(np.memmap(path, dtype=np.uint32, mode="r", shape=(n,)))
            total += n
            self._bounds.append(total)
        self.total_tokens = total
        self.cursor = 0
        self.epochs = 0

    def read_tokens(self, start: int, count: int) -> np.ndarray:
        """跨分片连续读 token（mmap 切片拼接，零拷贝读盘）。"""
        if count <= 0:
            return np.empty(0, dtype=np.uint32)
        if start < 0 or start + count > self.total_tokens:
            raise IndexError(f"[{start}, {start + count}) 超出 {self.split} "
                             f"共 {self.total_tokens} token")
        out = np.empty(count, dtype=np.uint32)
        got = 0
        while got < count:
            pos = start + got
            i = int(np.searchsorted(self._bounds, pos, side="right"))
            base = self._bounds[i - 1] if i else 0
            take = min(count - got, int(self._bounds[i]) - pos)
            out[got:got + take] = self._maps[i][pos - base:pos - base + take]
            got += take
        return out

    def _next(self, n_tokens: int) -> np.ndarray:
        """游标取 n_tokens；读到尾即回卷并计 epoch（报告可见）。"""
        if n_tokens > self.total_tokens:
            raise ValueError(f"单次请求 {n_tokens} token 超过 {self.split} 总量 "
                             f"{self.total_tokens}")
        if self.cursor + n_tokens > self.total_tokens:
            self.cursor = 0
            self.epochs += 1
        out = self.read_tokens(self.cursor, n_tokens)
        self.cursor += n_tokens
        return out

    def next_batch(self, seq_len: int, batch: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        """取一个课程批：窗口 L+1 → (x=L, y=L) 错位对齐（trainer 语义）。"""
        if seq_len < 1 or batch < 1:
            raise ValueError("seq_len/batch 必须 ≥ 1")
        flat = self._next(batch * (seq_len + 1))
        win = torch.from_numpy(flat.astype(np.int64)).reshape(batch, seq_len + 1)
        return win[:, :-1].contiguous(), win[:, 1:].contiguous()


# ---------------------------------------------------------------------------
# Phase 1 执行器
# ---------------------------------------------------------------------------
def tiny_model(vocab_size: int):
    """演示用 tiny 配置 + 原生权重（与 trainer CLI 同款，口径不变）。"""
    from .adapter_model import AdapterModel
    from .config import SimplCoreConfig
    from .conformance_runner import TorchReferenceAdapter
    from .model import SimplCoreModel

    cfg = SimplCoreConfig(vocab_size=vocab_size, dim=32, num_layers=2, num_bases=8,
                          local_kernel=4, ffn_ratio=2.0, norm="rmsnorm",
                          decay_max=256.0, tie_embeddings=False)
    model = AdapterModel(cfg, TorchReferenceAdapter())
    model.load_native(SimplCoreModel(cfg).state_dict())
    return model


def tiny_curriculum(seq_lens: Sequence[int] = (16, 32, 64),
                    tokens_per_pos: int = 40) -> ContextCurriculum:
    """演示课程：各档 token 预算 = seq_len × batch × tokens_per_pos（口径同战役）。"""
    return ContextCurriculum([CurriculumStage(L, L * 4 * tokens_per_pos)
                              for L in seq_lens])


def _check_splits_disjoint(manifest: dict, root: Path) -> None:
    """train/val 分片文件必须不相交——泄漏检查是结构性的，不是承诺。"""
    def _files(split: str) -> set:
        return {f["path"] for f in manifest["splits"].get(split, {}).get("files", [])}
    inter = _files("train") & _files("val")
    if inter:
        raise ValueError(f"train/val 共享分片文件，数据泄漏：{sorted(inter)}")


def run_phase1(corpus_root: Any, *, model=None, lr: float = 3e-3,
               batch: int = 4, max_steps: Optional[int] = None,
               curriculum: Optional[ContextCurriculum] = None,
               eval_every: int = 10, eval_batch: int = 4,
               eval_seq_len: Optional[int] = None, chunk: Optional[int] = None,
               seed: int = 2026, checkpoint_every: int = 0,
               checkpoint_path: Any = None,
               resume_from: Any = None) -> Dict[str, Any]:
    """端到端：语料 → 课程 → 分块 BPTT → 评估/gate → 报告（dict）。

    checkpoint 语义（``checkpoint_every``>0 时启用，默认存
    ``checkpoint_path`` 或 ``<corpus>/../phase1_ckpt.pt``）：
    * 周期保存参数 + 优化器 + 账本 + 最优快照 + **数据游标**；
    * ``resume_from`` 指向 checkpoint 时从该状态续训（课程/超参/adapter
      不一致即拒绝——续训不是换戏本的借口）；
    * 续训的轨迹与不间断运行**逐位一致**（守护测试钉死），因为恢复的是
      全部决定性状态，不是"大概齐"。
    """
    torch.manual_seed(seed)
    root = Path(corpus_root)
    manifest = read_manifest(root)
    _check_splits_disjoint(manifest, root)
    synthetic = bool(manifest.get("synthetic", False))
    vocab_size = int(manifest.get("vocab_size", 0) or 0)

    train_src = CorpusBatches(root, "train")
    val_src = CorpusBatches(root, "val")
    curriculum = curriculum or tiny_curriculum()
    eval_len = int(eval_seq_len if eval_seq_len is not None
                    else min(s.seq_len for s in curriculum.stages))

    # 评估批固定：同一批 val 窗口贯穿全程，跨步比较才同口径
    eval_x, eval_y = val_src.next_batch(eval_len, eval_batch)
    if model is None:
        model = tiny_model(vocab_size)

    resumed = False
    if resume_from is not None:
        trainer, ckpt_meta = StreamingTrainer.load_checkpoint(
            resume_from, model, model.adapter, curriculum)
        ds = ckpt_meta.get("data_state") or {}
        if ds:
            # 数据游标恢复：校验语料没换（指纹 + 总 token 数都得上对）
            if ds.get("corpus_sha256") and ds["corpus_sha256"] != \
                    compute_sha256(root / MANIFEST_NAME):
                raise ValueError("checkpoint 的语料指纹与当前语料不符——换了数据不算续训")
            train_src.cursor = int(ds.get("train_cursor", 0))
            train_src.epochs = int(ds.get("train_epochs", 0))
            val_src.cursor = int(ds.get("val_cursor", 0))
            val_src.epochs = int(ds.get("val_epochs", 0))
            eval_x, eval_y = val_src.next_batch(eval_len, eval_batch)
        resumed = True
    else:
        trainer = StreamingTrainer(model, model.adapter, curriculum, lr=lr)
    ch = chunk if chunk is not None else max(eval_len // 4, 1)

    ckpt_file = Path(checkpoint_path) if checkpoint_path is not None \
        else root.parent / "phase1_ckpt.pt"

    def _data_state() -> Dict[str, object]:
        # val 游标存"评估批取用前"的位置：续训时会重取同一评估批，
        # 若存取用后的位置，重取会把评估批错位成另一段（首版续训轨迹
        # 不一致的根因）。回卷跨界的极端情形按模总量折算。
        eval_tokens = eval_batch * (eval_len + 1)
        pre = val_src.cursor - eval_tokens
        if pre < 0:
            pre += val_src.total_tokens
        return {"train_cursor": train_src.cursor, "train_epochs": train_src.epochs,
                "val_cursor": pre, "val_epochs": val_src.epochs,
                "corpus_sha256": compute_sha256(root / MANIFEST_NAME)}

    t0 = time.time()
    eval_history: List[Dict[str, Any]] = []
    first_eval: Optional[float] = None
    while not curriculum.exhausted(trainer.ledger.tokens_seen):
        if max_steps is not None and trainer.ledger.steps >= max_steps:
            break
        _, stage = trainer.stage()
        x, y = train_src.next_batch(stage.seq_len, batch)
        r = trainer.train_step(x, y)
        if trainer.ledger.steps % eval_every == 0 or curriculum.exhausted(
                trainer.ledger.tokens_seen):
            gate = trainer.maybe_commit(eval_x, eval_y, chunk=ch)
            first_eval = first_eval if first_eval is not None else float(
                gate["eval_loss"])
            eval_history.append({
                "step": trainer.ledger.steps,
                "stage": int(r["stage"]),
                "stage_seq_len": int(stage.seq_len),
                "train_loss": round(float(r["loss"]), 6),
                "eval_loss": round(float(gate["eval_loss"]), 6),
                "committed": bool(gate["committed"]),
                "rollback": bool(gate["rollback"]),
            })
        if checkpoint_every > 0 and trainer.ledger.steps % checkpoint_every == 0:
            trainer.save_checkpoint(ckpt_file, data_state=_data_state())

    final_eval = float(eval_history[-1]["eval_loss"]) if eval_history else None
    improved = bool(final_eval is not None and final_eval < first_eval)
    claim_permitted = bool(improved and trainer.ledger.commits >= 1 and not synthetic)
    if checkpoint_every > 0:
        trainer.save_checkpoint(ckpt_file, data_state=_data_state())
    report: Dict[str, Any] = {
        "protocol": "phase1-train-run",
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "resumed_from": (str(resume_from) if resumed else None),
        "checkpoint": ({"path": str(ckpt_file), "every_steps": checkpoint_every}
                       if checkpoint_every > 0 else None),
        "corpus": {
            "root": str(root),
            "manifest_sha256": compute_sha256(root / MANIFEST_NAME),
            "synthetic": synthetic,
            "train_tokens": train_src.total_tokens,
            "val_tokens": val_src.total_tokens,
            "splits_disjoint": True,
            "train_epochs": train_src.epochs,
            "val_epochs": val_src.epochs,
        },
        "curriculum": curriculum.as_dict(),
        "eval_protocol": {
            "eval_seq_len": eval_len,
            "eval_batch": eval_batch,
            "chunk": ch,
            "note": ("评估一律用固定 eval_seq_len 的同一 val 批；跨档训练 loss 不可比，"
                     "评估可比。first_eval_loss 是**本次运行**的首个评估（续训时基线"
                     "在续点重置）；回滚闸门的历史基线 best_eval 来自 checkpoint，"
                     "不受续训影响"),
        },
        "ledger": trainer.ledger.as_dict(),
        "eval_history": eval_history,
        "gate": {
            "first_eval_loss": None if first_eval is None else round(first_eval, 6),
            "final_eval_loss": None if final_eval is None else round(final_eval, 6),
            "improved": improved,
            "commits": trainer.ledger.commits,
            "rollbacks": trainer.ledger.rollbacks,
        },
        "honesty": {
            "claim_permitted": claim_permitted,
            "forbidden_checks": [
                "synthetic 语料不得用于外部宣称（只证明管线语义正确）",
                "训练 loss 不是评估（评估只认 val 分片固定批）",
                "报告必须携带语料指纹；指纹不符即换了数据",
            ],
        },
        "wall_seconds": round(time.time() - t0, 3),
    }
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 1 训练执行器：packed 语料 → 课程 → StreamingTrainer → 报告")
    ap.add_argument("--corpus", type=str, default=None,
                    help="packed 语料目录（含 index.json）；缺省走合成演示")
    ap.add_argument("--out", type=str, default="runs/phase1_report.json")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--train-tokens", type=int, default=20000)
    ap.add_argument("--val-tokens", type=int, default=4096)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)

    if args.corpus:
        root = args.corpus
        synth_note = ""
    else:
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="simplcore_phase1_"))
        spec = SyntheticCorpusSpec(vocab_size=32, seq_len=64,
                                   train_tokens=args.train_tokens,
                                   val_tokens=args.val_tokens, seed=args.seed)
        make_synthetic_corpus(tmp, spec)
        root = str(tmp)
        synth_note = "（合成演示语料：只证明管线语义正确，宣称许可=False）"

    report = run_phase1(root, batch=args.batch, max_steps=args.steps,
                        lr=args.lr, eval_every=args.eval_every, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    led = report["ledger"]
    g = report["gate"]
    print(f"=== Phase 1 训练执行器 {synth_note} ===")
    print(f" 语料: {report['corpus']['root']} (sha256 "
          f"{report['corpus']['manifest_sha256'][:16]}..., train "
          f"{report['corpus']['train_tokens']} tok)")
    for e in report["eval_history"]:
        print(f" step {e['step']:3d} | stage {e['stage']} (L={e['stage_seq_len']}) | "
              f"train {e['train_loss']:.4f} | eval {e['eval_loss']:.4f} | "
              f"{'提交' if e['committed'] else '—'}{'/回滚' if e['rollback'] else ''}")
    verdict = "改善 ✓" if g["improved"] else "未改善 ✗"
    print(f"== 评估 {g['first_eval_loss']} → {g['final_eval_loss']}（{verdict}）；"
          f"steps={led['steps']} tokens={led['tokens_seen']} "
          f"commits={g['commits']} rollbacks={g['rollbacks']} | "
          f"claim_permitted={report['honesty']['claim_permitted']} ==")
    print(f" 报告: {out_path}")
    return 0 if g["improved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
