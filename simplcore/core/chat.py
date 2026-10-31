# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""对话数据管线：特殊 token、chat 模板、带掩码的 SFT 数据集、序列打包。

为什么 30M 级模型必须先有这一层
--------------------------------
预训练目标（预测下一个 token）学的是"文本像什么"，对话能力学的是
"**轮到谁说、说到哪停**"。后者完全由格式承载：如果训练时把
``<|user|>`` 之后的内容也计入损失，模型就会学会自己扮演用户，
多轮对话里开始自问自答；如果不把 ``<|end|>`` 计入损失，模型就学不会停，
生成会一直续写到 max_new_tokens 为止。这两种失败在 loss 曲线上一模一样，
只能靠"标签掩码"从根上避免。

本模块提供的三件事：

1. :class:`SpecialTokenCodec` —— 在任意基础 codec（char / byte / BPE）上
   **叠加**特殊 token，不重训词表。ID 布局是
   ``[0, base.size())`` 为原词表（**ID 完全不变**），
   ``[base.size(), base.size()+n)`` 为特殊 token。
2. :class:`ChatTemplate` —— 唯一的对话渲染格式，训练与推理共用一份，
   避免"训练用 A 格式、推理用 B 格式"这种最难查的错配。
3. :class:`SFTDataset` + :func:`pack_sequences` —— 逐 token 标签掩码 +
   定长打包，把大量短对话拼成不浪费算力的定长序列。

格式
----
::

    <|system|>你是……<|end|><|user|>问……<|end|><|assistant|>答……<|end|>

只用 4 个特殊符号，且每个都是**单 token**。小模型对大段模板很敏感：
每多一个模板 token，就多一份"永远不用预测、只占上下文"的负担。

标签
----
``labels[i] == -100`` 的位置不参与损失（PyTorch 的 ``ignore_index``）。
具体地：system / user 的全部 token、以及每个 ``<|assistant|>`` 标记本身
都是 -100；只有 assistant 的**内容**和结尾的 ``<|end|>`` 被训练。
``<|end|>`` 必须被训练 —— 它是"停止"这个动作本身。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .tokenization import Tokenizer

IGNORE_INDEX = -100

#: 特殊 token。顺序即 ID 顺序，一旦发布**不得重排**（会静默改变已训权重含义）。
SPECIAL_TOKENS: Tuple[str, ...] = (
    "<|pad|>",        # 0 + offset：打包补齐
    "<|end|>",        # 1 + offset：一句话/一轮结束，也是生成停止符
    "<|user|>",       # 2 + offset
    "<|assistant|>",  # 3 + offset
    "<|system|>",     # 4 + offset
)
PAD_TOKEN = "<|pad|>"
END_TOKEN = "<|end|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
SYSTEM_TOKEN = "<|system|>"


class SpecialTokenCodec(Tokenizer):
    """在基础 codec 上叠加特殊 token 的包装器。

    不动原词表：``encode("你好") == base.encode("你好")``，特殊 token 追加在
    词表**尾部**（``base.size() + i``）。这对 30M 计划很关键：

    * 基础词表 ID 不变 → 换 codec（char → BPE）时特殊 ID 的定义方式一致，
      已有 checkpoint 的嵌入行语义不会整体错位；
    * 追加而非前置 → 若将来接入外部预训练的词表，前 ``base.size()`` 行
      仍可直接复用。
    """

    def __init__(self, base: Tokenizer, specials: Sequence[str] = SPECIAL_TOKENS,
                 allow_unknown_special: bool = False) -> None:
        self.base = base
        self.specials = tuple(specials)
        self._offset = int(base.size())
        self._ids = {s: self._offset + i for i, s in enumerate(self.specials)}
        if allow_unknown_special:
            raise NotImplementedError(
                "未登记的特殊 token 会被静默当作普通文本切碎，"
                "这比报错更难查；请把需要的符号加进 specials")
        # 长符号优先匹配，避免 "<|end|>" 被 "<|e" 之类的前缀抢走
        ordered = sorted(self.specials, key=len, reverse=True)
        self._pattern = re.compile(
            "(" + "|".join(re.escape(s) for s in ordered) + ")") if ordered else None

    # -- 查询 ---------------------------------------------------------------
    @property
    def offset(self) -> int:
        return self._offset

    def token_id(self, token: str) -> int:
        try:
            return self._ids[token]
        except KeyError:
            raise KeyError(f"{token!r} 不是已登记的特殊 token：{self.specials}") from None

    def id_is_special(self, idx: int) -> bool:
        return idx >= self._offset

    # -- 编解码 -------------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        if self._pattern is None:
            return list(self.base.encode(text))
        out: List[int] = []
        for part in self._pattern.split(text):
            if not part:
                continue
            if part in self._ids:
                out.append(self._ids[part])
            else:
                out.extend(self.base.encode(part))
        return out

    def decode(self, ids: Iterable[int], keep_special: bool = True) -> str:
        """ID → 文本。``keep_special=True`` 时特殊 token 以字面量出现（便于查看）。"""
        ids = [int(i) for i in ids]
        if not ids:
            return ""
        if not any(self.id_is_special(i) for i in ids):
            return self.base.decode(ids)
        out: List[str] = []
        buf: List[int] = []

        def flush() -> None:
            if buf:
                out.append(self.base.decode(list(buf)))
                buf.clear()

        for i in ids:
            if self.id_is_special(i):
                flush()
                if keep_special:
                    out.append(self.specials[i - self._offset])
            else:
                buf.append(i)
        flush()
        return "".join(out)

    def size(self) -> int:
        return self._offset + len(self.specials)

    def max_token_len(self) -> int:
        return max(self.base.max_token_len(), max(len(s) for s in self.specials))


# ---------------------------------------------------------------------------
# 模板
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChatTemplate:
    """唯一的对话格式定义。训练与推理都必须走它。

    Attributes:
        system: 默认 system 提示；``None`` 表示不注入（不再额外花 token）。
        add_end_after_assistant: 是否在 assistant 回复后补 ``<|end|>``。
            必须为 True，否则模型没有"停止"的概念。
        add_end_after_prompt: 是否在 user/system 后补 ``<|end|>``。
            必须为 True，否则 user 与 assistant 的边界只能靠标记区分，
            小模型很容易学混。
    """

    system: Optional[str] = None
    add_end_after_assistant: bool = True
    add_end_after_prompt: bool = True

    # -- 渲染 ---------------------------------------------------------------
    def render(self, messages: Sequence[Dict[str, str]],
               add_generation_prompt: bool = False) -> str:
        """把消息列表渲染成字符串（特殊 token 以字面量出现）。"""
        msgs = [dict(m) for m in messages]
        if self.system and not (msgs and msgs[0].get("role") == "system"):
            msgs.insert(0, {"role": "system", "content": self.system})
        parts: List[str] = []
        for m in msgs:
            role, content = m.get("role"), m.get("content", "")
            if role == "system":
                parts.append(f"{SYSTEM_TOKEN}{content}")
                if self.add_end_after_prompt:
                    parts.append(END_TOKEN)
            elif role == "user":
                parts.append(f"{USER_TOKEN}{content}")
                if self.add_end_after_prompt:
                    parts.append(END_TOKEN)
            elif role == "assistant":
                parts.append(f"{ASSISTANT_TOKEN}{content}")
                if self.add_end_after_assistant:
                    parts.append(END_TOKEN)
            else:
                raise ValueError(f"未知 role {role!r}（只支持 system / user / assistant）")
        if add_generation_prompt:
            parts.append(ASSISTANT_TOKEN)
        return "".join(parts)

    # -- 编码 + 掩码 ---------------------------------------------------------
    def encode_example(self, messages: Sequence[Dict[str, str]], codec: SpecialTokenCodec,
                       max_len: Optional[int] = None,
                       drop_if_no_target: bool = True,
                       truncate: str = "right") -> Optional[Tuple[List[int], List[int]]]:
        """编码一条多轮对话，返回 ``(input_ids, labels)``。

        ``labels`` 与 ``input_ids`` 等长；prompt 段全部为 :data:`IGNORE_INDEX`。

        Args:
            max_len: 截断长度（含特殊 token）。``None`` 不截断。
            drop_if_no_target: 截断后若一个 assistant token 都不剩，返回 ``None``
                —— 这种样本的损失是 0/0，留着只会让"有效 token 数"虚高。
            truncate: ``"right"`` 保留开头（优先保住 system + 首轮），
                ``"left"`` 保留结尾（优先保住最后一轮）。

        Returns:
            ``(ids, labels)`` 或 ``None``（无可用目标）。
        """
        ids: List[int] = []
        labels: List[int] = []

        def emit(text: str, train: bool) -> None:
            part = codec.encode(text)
            ids.extend(part)
            labels.extend(part if train else [IGNORE_INDEX] * len(part))

        msgs = [dict(m) for m in messages]
        if self.system and not (msgs and msgs[0].get("role") == "system"):
            msgs.insert(0, {"role": "system", "content": self.system})
        for m in msgs:
            role, content = m.get("role"), m.get("content", "")
            if role == "system":
                emit(SYSTEM_TOKEN, False)
                emit(content, False)
                if self.add_end_after_prompt:
                    emit(END_TOKEN, False)
            elif role == "user":
                emit(USER_TOKEN, False)
                emit(content, False)
                if self.add_end_after_prompt:
                    emit(END_TOKEN, False)
            elif role == "assistant":
                emit(ASSISTANT_TOKEN, False)
                emit(content, True)
                if self.add_end_after_assistant:
                    emit(END_TOKEN, True)      # 停止动作必须学
            else:
                raise ValueError(f"未知 role {role!r}")

        if max_len is not None and max_len > 0 and len(ids) > max_len:
            if truncate == "right":
                ids, labels = ids[:max_len], labels[:max_len]
            else:
                ids, labels = ids[-max_len:], labels[-max_len:]
        if drop_if_no_target and all(v == IGNORE_INDEX for v in labels):
            return None
        return ids, labels

    # -- 推理 ---------------------------------------------------------------
    def stop_ids(self, codec: SpecialTokenCodec) -> List[int]:
        return [codec.token_id(END_TOKEN)]

    def prompt_ids(self, codec: SpecialTokenCodec,
                   messages: Sequence[Dict[str, str]]) -> List[int]:
        """推理入口：渲染到 ``<|assistant|>`` 为止并编码。"""
        text = self.render(messages, add_generation_prompt=True)
        return codec.encode(text)


# ---------------------------------------------------------------------------
# 数据读取
# ---------------------------------------------------------------------------
def iter_sft_records(path: str) -> Iterator[Dict[str, Any]]:
    """逐行读 JSONL，兼容三种常见 schema。

    * ``{"messages": [{"role":..,"content":..}, ...]}``
    * ``{"conversations": [[user, assistant], ...]}``  （ShareGPT 简写）
    * ``{"prompt": "...", "response": "..."}``
    """
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} 不是合法 JSON：{exc}") from None
            yield normalise_record(rec)


def normalise_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """把三种 schema 归一成 ``{"messages": [...]}``。"""
    if "messages" in rec:
        msgs = rec["messages"]
    elif "conversations" in rec:
        msgs = []
        for turn in rec["conversations"]:
            if isinstance(turn, dict):
                role = turn.get("role") or turn.get("from")
                content = turn.get("content") or turn.get("value", "")
            else:
                role, content = turn[0], turn[1]
            role = {"human": "user", "gpt": "assistant",
                    "system": "system"}.get(str(role), str(role))
            msgs.append({"role": role, "content": content})
    elif "prompt" in rec and "response" in rec:
        msgs = [{"role": "user", "content": rec["prompt"]},
                {"role": "assistant", "content": rec["response"]}]
    else:
        raise ValueError(f"无法识别的样本 schema，键：{sorted(rec)}")
    return {"messages": msgs}


# ---------------------------------------------------------------------------
# 打包
# ---------------------------------------------------------------------------
def _pack_rows(examples: Sequence[Tuple[Sequence[int], Sequence[int]]],
               seq_len: int, pad_id: int,
               drop_remainder: bool = False,
               exclusive: Optional[Sequence[bool]] = None,
               ) -> Tuple[torch.Tensor, torch.Tensor, List[bool]]:
    """``pack_sequences`` 的实现，额外返回每行是否"独占"。"""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if exclusive is not None and len(exclusive) != len(examples):
        raise ValueError("exclusive 必须与 examples 等长")
    rows_ids: List[List[int]] = []
    rows_lab: List[List[int]] = []
    rows_excl: List[bool] = []
    cur_ids: List[int] = []
    cur_lab: List[int] = []
    cur_excl = False

    def flush(pad: bool) -> None:
        nonlocal cur_ids, cur_lab, cur_excl
        if not cur_ids and not pad:
            return
        rows_ids.append(cur_ids + [pad_id] * (seq_len - len(cur_ids)))
        rows_lab.append(cur_lab + [IGNORE_INDEX] * (seq_len - len(cur_lab)))
        rows_excl.append(cur_excl)
        cur_ids, cur_lab, cur_excl = [], [], False

    for i, (ids, lab) in enumerate(examples):
        ids, lab = list(ids), list(lab)
        if len(ids) > seq_len:
            ids, lab = ids[:seq_len], lab[:seq_len]
        alone = bool(exclusive[i]) if exclusive is not None else False
        # 独占样本必须一进一出：进来前把已攒的行收掉，出去后立刻收行
        if alone and cur_ids:
            flush(pad=True)
        if len(cur_ids) + len(ids) > seq_len:
            flush(pad=True)
        if alone:
            cur_excl = True
        cur_ids.extend(ids)
        cur_lab.extend(lab)
        if len(cur_ids) == seq_len or alone:
            flush(pad=False)
    if cur_ids:
        if drop_remainder:
            cur_ids, cur_lab = [], []
        else:
            flush(pad=True)
    if not rows_ids:
        return (torch.zeros((0, seq_len), dtype=torch.long),
                torch.zeros((0, seq_len), dtype=torch.long), [])
    return (torch.tensor(rows_ids, dtype=torch.long),
            torch.tensor(rows_lab, dtype=torch.long), rows_excl)


def pack_sequences(examples: Sequence[Tuple[Sequence[int], Sequence[int]]],
                   seq_len: int, pad_id: int,
                   drop_remainder: bool = False,
                   exclusive: Optional[Sequence[bool]] = None,
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """把变长样本贪心拼接成定长行，避免短对话浪费算力。

    打包不会破坏标签语义：prompt 段本来就是 -100，拼接点在
    ``<|end|>`` 之后，模型看到的是"上一轮结束、下一轮开始"，
    与真实部署时的多轮上下文一致。

    Args:
        exclusive: 与 ``examples`` 等长的掩码。为 ``True`` 的样本**独占一行**，
            前后都强制断开。这是给"带外部记忆"的样本用的：记忆是**整行**
            一个向量，一旦与别的样本混行，那半行的样本也会被注入本来
            不属于它的记忆。独占是唯一能保证对齐的做法。

    Returns:
        ``(input_ids, labels)``，均为 ``(n_rows, seq_len)``，
        尾巴用 ``pad_id`` / ``IGNORE_INDEX`` 填充。
    """
    ids, lab, _ = _pack_rows(examples, seq_len, pad_id, drop_remainder, exclusive)
    return ids, lab


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def load_memory_encoder(path: str, dim: Optional[int] = None):
    """从 ``memory_encoder.json`` 读回编码器（训练时落盘的那一份）。

    推理侧必须用它，不能用默认的未拟合编码器：训练出来的 ``memory_in``
    是在**拟合后**的向量分布上学的，换成未拟合的输入，记忆通道的语义
    整体偏移，而输出仍然是"看起来像人话"的东西，不会报错。
    """
    from .memory import TextKeyEncoder

    with open(path, "r", encoding="utf-8") as fh:
        state = json.load(fh)
    enc = TextKeyEncoder.from_state(state)
    if dim is not None and int(enc.dim) != int(dim):
        raise ValueError(f"编码器维度 {enc.dim} 与模型宽度 {dim} 不一致")
    return enc


def build_memory_encoder(path: str, dim: int, *, alpha: float = 1.0,
                         max_records: Optional[int] = None):
    """扫描 JSONL 里所有 ``memory`` 字段，拟合出**训练与评测共用**的编码器。

    为什么必须在 :class:`SFTDataset` 之外单独建：数据集按 train / val 分别
    构造，各自 ``fit`` 会得到两份不同的 df 统计，同一段记忆在两侧被编码成
    不同向量。形状对、loss 照降，但两侧的"记忆语言"不一样 —— 这类错不会
    让任何断言失败，只会让消融数字失去意义。

    所以调用方（训练脚本）应当：fit 一次 → 传给两个数据集 → 存盘 →
    评测/消融从盘里读回同一份。
    """
    from .memory import TextKeyEncoder

    texts: List[str] = []
    with open(path, "rb") as fh:
        for i, line in enumerate(fh):
            if max_records is not None and i >= max_records:
                break
            line = line.decode("utf-8").strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = rec.get("memory")
            if isinstance(m, str) and m.strip():
                texts.append(m)
    return TextKeyEncoder(dim).fit(texts, alpha=alpha)


class SFTDataset(Dataset):
    """定长打包后的 SFT 数据集。

    两段式：先扫一遍文件记录每条样本的字节偏移（不解析内容），
    再按需 ``seek`` 读取 —— 这样 100 万条对话也不必全部驻留内存。

    Args:
        path: JSONL 路径。
        codec: 带特殊 token 的 codec。
        template: 对话模板。
        seq_len: 打包后的定长。
        max_len: 单条对话截断长度（含特殊 token），``None`` 不截断。
        max_records: 只读前 N 条（调试用）。
        seed: 打包顺序的打乱种子（``None`` 不打乱）。
    """

    def __init__(self, path: str, codec: SpecialTokenCodec,
                 template: Optional[ChatTemplate] = None,
                 seq_len: int = 1024, max_len: Optional[int] = 2048,
                 max_records: Optional[int] = None,
                 shuffle: bool = False, seed: int = 0,
                 index_slice: Optional[Tuple[int, int]] = None,
                 memory_dim: int = 0,
                 memory_encoder: Optional[Any] = None) -> None:
        self.path = path
        self.codec = codec
        self.template = template or ChatTemplate()
        self.seq_len = int(seq_len)
        self.max_len = max_len
        self.memory_dim = int(memory_dim)
        # 编码器必须由**外部传入**（训练/评测共用同一份语料统计）。
        # 默认不拟合的编码器只为兼容旧调用：同一模板下只差一个词的记忆
        # 余弦高达 0.71，模型分不出"小林"和"阿岚"。
        self.memory_encoder = memory_encoder
        self._offsets: List[int] = []
        self._examples: List[Tuple[List[int], List[int]]] = []
        self._exclusive: List[bool] = []
        self._mem_texts: List[Optional[str]] = []
        self.n_skipped = 0
        self.n_tokens_kept = 0
        self.n_memory_rows = 0

        self._scan(max_records)
        if index_slice is not None:
            lo, hi = index_slice
            self._offsets = self._offsets[max(0, lo):max(0, hi)]
        if shuffle:
            import random
            random.Random(seed).shuffle(self._offsets)
        self._rows = self._pack()
        if self._rows[0].shape[0] == 0:
            raise ValueError(
                f"{path} 一条可用样本都没有（全部被截断成 0 个 assistant token？）")

    # -- 内部 --------------------------------------------------------------
    def _scan(self, max_records: Optional[int]) -> None:
        with open(self.path, "rb") as fh:
            pos = 0
            n = 0
            for line in fh:
                if line.strip():
                    self._offsets.append(pos)
                    n += 1
                    if max_records is not None and n >= max_records:
                        break
                pos += len(line)

    def _encode_offset(self, offset: int):
        """编码一条样本，返回 ``(ids, labels, memory_text, memory_required)``。

        带 ``memory`` 字段的样本记录外部事实；``memory_required`` 表示
        "没有这段记忆就根本答不出"（例如跨会话回忆人名）。后者在缺记忆时
        整行不监督 —— 否则等于逼模型把"小林"当默认答案背下来，
        换个人名就错，而且这个错误在 loss 曲线上完全看不见。
        """
        with open(self.path, "rb") as fh:
            fh.seek(offset)
            line = fh.readline().decode("utf-8").strip()
        if not line:
            return None
        raw = json.loads(line)
        rec = normalise_record(raw)
        mem = raw.get("memory")
        mem = str(mem).strip() if isinstance(mem, str) else None
        required = bool(raw.get("memory_required", False))
        enc = self.template.encode_example(rec["messages"], self.codec,
                                           max_len=self.max_len)
        if enc is None:
            return None
        ids, lab = enc
        return ids, lab, (mem or None), required

    def _pack(self) -> Tuple[torch.Tensor, torch.Tensor]:
        for off in self._offsets:
            ex = self._encode_offset(off)
            if ex is None:
                self.n_skipped += 1
                continue
            ids, lab, mem, required = ex
            if required and not mem:
                # 没有外部事实就没有可学的东西：整行置 -100，别制造噪声梯度
                lab = [IGNORE_INDEX] * len(lab)
            self._examples.append((ids, lab))
            self._exclusive.append(bool(mem))
            self._mem_texts.append(mem)
            self.n_tokens_kept += sum(1 for v in lab if v != IGNORE_INDEX)

        ids_t, lab_t, rows_excl = _pack_rows(
            self._examples, self.seq_len, self.codec.token_id(PAD_TOKEN),
            exclusive=self._exclusive if any(self._exclusive) else None)

        self._mem = None
        if self.memory_dim > 0 and any(self._exclusive):
            # 独占 ⟺ 有记忆文本（上面两者由同一个 ``mem`` 决定），
            # 所以按同样的过滤顺序取文本，行 ↔ 文本才对得上。
            texts = [t for t, e in zip(self._mem_texts, self._exclusive) if e]
            self._mem = self._build_memory_rows(rows_excl, texts)
        return ids_t, lab_t

    def _build_memory_rows(self, rows_excl: Sequence[bool],
                           texts: Sequence[Optional[str]]) -> torch.Tensor:
        """把"独占行"对应的记忆文本编码成 ``(n_rows, memory_dim)``。

        非记忆行全是零向量。``memory_in`` 是**无 bias 的线性层**，
        所以零向量经过它仍是零 —— "不注入"是数学保证的，不是靠开关。
        """
        from .memory import TextKeyEncoder

        enc = self.memory_encoder
        if enc is None:
            enc = TextKeyEncoder(self.memory_dim)
            # 纪律：只有"没给编码器"才走到这里，属于兼容路径。训练脚本
            # 必须显式传入拟合好的编码器，否则同模板记忆近乎共线。
            self.memory_encoder = enc
        if int(getattr(enc, "dim", self.memory_dim)) != self.memory_dim:
            raise ValueError(
                f"记忆编码器维度 {getattr(enc, 'dim', None)} 与 memory_dim "
                f"{self.memory_dim} 不一致")
        out = torch.zeros((len(rows_excl), self.memory_dim), dtype=torch.float32)
        assert len(texts) == sum(1 for e in rows_excl if e), \
            f"记忆行数 {sum(1 for e in rows_excl if e)} 与文本数 {len(texts)} 不一致"
        k = 0
        for row, alone in enumerate(rows_excl):
            if not alone:
                continue
            text = texts[k]
            k += 1
            if not text:
                continue
            vec = enc(text)
            if float(vec.norm()) > 0:
                out[row] = vec
                self.n_memory_rows += 1
        return out

    # -- Dataset -----------------------------------------------------------
    def __len__(self) -> int:
        return self._rows[0].shape[0]

    def __getitem__(self, index: int):
        """返回**已偏移一位**的 ``(input_ids, labels)``，有记忆时追加第三个元素。

        因果 LM 的目标必须错开一格：位置 ``t`` 的 logits 预测的是
        ``labels[t+1]``。这里返回 ``(row[:-1], labels[1:])``，
        与 :class:`simplcore.core.train.LMWindows` 的口径一致。

        少了这一步，``cross_entropy(logits, labels)`` 会退化成
        "用当前位置的输入预测当前位置的标签" —— 模型只需学会复制输入，
        损失几秒内就掉到 0.1 以下，生成时则表现为**把标记原样吐回来**
        （第一个 token 永远是 ``<|assistant|>``）。这个错误不会让任何形状
        断言失败，只会让"训练收敛"变成一句空话。
        """
        x = self._rows[0][index][:-1]
        y = self._rows[1][index][1:]
        if self._mem is not None:
            return x, y, self._mem[index]
        return x, y

    @property
    def with_memory(self) -> bool:
        return self._mem is not None

    def memory_tensor(self) -> Optional[torch.Tensor]:
        return self._mem

    # -- 统计 ---------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        ids = self._rows[0]
        lab = self._rows[1]
        supervised = int((lab != IGNORE_INDEX).sum())
        total = int(lab.numel())
        return {
            "records": len(self._offsets),
            "records_used": len(self._examples),
            "records_skipped": self.n_skipped,
            "rows": int(ids.shape[0]),
            "seq_len": self.seq_len,
            "supervised_tokens": supervised,
            "all_tokens": total,
            "supervised_ratio": round(supervised / max(total, 1), 4),
            "memory_rows": self.n_memory_rows,
        }


# ---------------------------------------------------------------------------
# 词表训练 + 适配
# ---------------------------------------------------------------------------
def build_sft_dataset(sft_jsonl: str, codec: SpecialTokenCodec,
                      template: Optional[ChatTemplate] = None,
                      seq_len: int = 1024, max_len: Optional[int] = 2048,
                      max_records: Optional[int] = None) -> SFTDataset:
    """便捷构造 + 一次性完成形状校验。"""
    ds = SFTDataset(sft_jsonl, codec, template, seq_len, max_len, max_records)
    if codec.size() <= 0:
        raise ValueError("codec.size() 必须为正")
    return ds


def collate_sft(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]):
    """``DataLoader`` 的默认 collate（已经打包成定长，直接 stack）。

    同时兼容带记忆的三元组 ``(x, y, mem)``：记忆按行 stack 成 ``(B, dim)``。
    没带记忆的样本行里是零向量，经 ``memory_in``（无 bias）后仍为零，
    因此同一个 batch 里"有记忆"和"无记忆"的样本可以混在一起训练。
    """
    xs = torch.stack([b[0] for b in batch])
    ys = torch.stack([b[1] for b in batch])
    if len(batch[0]) > 2:
        ms = torch.stack([b[2] for b in batch])
        return xs, ys, ms
    return xs, ys
