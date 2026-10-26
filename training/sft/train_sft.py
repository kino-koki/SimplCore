import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from itertools import chain
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer,
)
from qingxiaotuan import QingxiaotuanForCausalLM, QingxiaotuanTokenizer, QingxiaotuanConfig


IM_START_ID = 151644
IM_END_ID = 151645
EOS_ID = 151643


def safe_id(tensor, dim=-1):
    if tensor.numel() == 0:
        return None
    max_val = tensor.max().item()
    if max_val < 0:
        return None
    return max_val


@dataclass
class SFTArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "预训练模型ckpt路径"},
    )
    data_path: str = field(
        default=None,
        metadata={"help": "训练数据路径（JSON/JSONL）"},
    )
    output_dir: str = field(
        default="./output_sft",
        metadata={"help": "输出目录"},
    )
    max_length: int = field(
        default=524288,
        metadata={"help": "最大序列长度（默认512K）"},
    )
    num_epochs: int = field(
        default=3,
        metadata={"help": "训练轮数"},
    )
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "学习率"},
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "每设备batch size（512K下每卡1条）"},
    )
    gradient_accumulation_steps: int = field(
        default=12,
        metadata={"help": "梯度累积步数（默认12）"},
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "日志间隔步数"},
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "保存间隔步数"},
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "最多保存的ckpt数量"},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={"help": "DeepSpeed配置文件路径"},
    )
    flash_attention: bool = field(
        default=True,
        metadata={"help": "启用Flash Attention"},
    )
    pack_sequences: bool = field(
        default=True,
        metadata={"help": "启用动态序列打包"},
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "启用梯度检查点"},
    )

    # --- 消融实验 & 1M 上下文 ---
    attention_mode: str = field(
        default="standard",
        metadata={
            "help": (
                "注意力模式 (消融实验)：\n"
                "  standard  - GQA + YaRN RoPE (默认)\n"
                "  csa       - Contextual Sliding Attention\n"
                "  hca       - Hierarchical Context Attention\n"
                "  csa_hca   - 0.7*CSA + 0.3*HCA\n"
                "  ring_csa  - Ring Attention + CSA (超长序列)\n"
            )
        },
    )
    model_size: str = field(
        default="12b",
        metadata={"help": "模型规格: 12b / 100b"},
    )
    enable_progressive_length_warmup: bool = field(
        default=True,
        metadata={"help": "启用 Progressive Length Warmup (4K -> 16K -> 64K -> 256K -> 1M)"},
    )


class SFTDataset(Dataset):
    """青小团SFT数据集 - 支持512K长上下文和sequence packing

    加载指令数据，使用ChatML格式：
    <|im_start|>user
    {instruction}<|im_end|>
    <|im_start|>assistant
    {response}<|im_end|>

    只计算assistant部分的loss，prompt部分mask为-100。
    支持将多个短对话打包成一个512K长序列。
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 524288,
        pack_sequences: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pack_sequences = pack_sequences
        self.data = self._load_data(data_path)
        self.packed_indices = None

        if self.pack_sequences:
            self.packed_indices = self._build_packed_indices()

    def _load_data(self, data_path: str) -> List[Dict]:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        ext = os.path.splitext(data_path)[1].lower()
        if ext == ".json":
            with open(data_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        elif ext == ".jsonl":
            raw_data = []
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        raw_data.append(json.loads(line))
        else:
            raise ValueError(f"不支持的文件格式: {ext}，仅支持 .json 和 .jsonl")

        processed = []
        for item in raw_data:
            if "conversations" in item:
                conversations = item["conversations"]
                messages = []
                for turn in conversations:
                    role = turn.get("role", turn.get("from", ""))
                    content = turn.get("content", turn.get("value", ""))
                    messages.append({"role": role, "content": content})
                processed.append({"messages": messages})
            elif "instruction" in item:
                inp = item.get("input", "")
                instruction = item["instruction"]
                if inp:
                    instruction = instruction + "\n" + inp
                processed.append({
                    "messages": [
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": item["output"]},
                    ]
                })
            elif "messages" in item:
                processed.append(item)
            elif "prompt" in item and "response" in item:
                processed.append({
                    "messages": [
                        {"role": "user", "content": item["prompt"]},
                        {"role": "assistant", "content": item["response"]},
                    ]
                })
            else:
                print(f"跳过未知格式的数据: {item.keys()}")
        return processed

    def _apply_chatml(self, messages: List[Dict]) -> str:
        """将消息列表转换为ChatML格式字符串"""
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            else:
                text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        text += "<|endoftext|>"
        return text

    def _find_assistant_ranges(self, input_ids: List[int]) -> List[List[int]]:
        """找到所有assistant回复的token范围（包括<|im_start|>assistant）"""
        ranges = []
        i = 0
        while i < len(input_ids) - 1:
            if input_ids[i] == IM_START_ID:
                role_start = i + 1
                role_end = role_start
                while role_end < len(input_ids) and input_ids[role_end] != IM_END_ID and input_ids[role_end] != 198:
                    role_end += 1
                role_tokens = input_ids[role_start:role_end]
                if role_tokens and self.tokenizer.decode(role_tokens).strip() == "assistant":
                    msg_start = i
                    ranges.append([msg_start, -1])
                if role_end < len(input_ids) and input_ids[role_end] == IM_END_ID:
                    i = role_end + 1
                else:
                    i += 1
            else:
                i += 1

        for r in ranges:
            if r[1] == -1:
                end_idx = r[0]
                while end_idx < len(input_ids):
                    if input_ids[end_idx] == IM_START_ID:
                        if r[0] != end_idx:
                            break
                    elif input_ids[end_idx] == EOS_ID:
                        r[1] = end_idx + 1
                        break
                    end_idx += 1
                if r[1] == -1:
                    r[1] = len(input_ids)
        return ranges

    def _tokenize_item(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """对单条SFT数据进行tokenize，返回input_ids, labels, loss_weights"""
        item = self.data[idx]
        messages = item["messages"]
        text = self._apply_chatml(messages)

        tokens = self.tokenizer.encode(text)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()

        assistant_ranges = self._find_assistant_ranges(tokens)
        if assistant_ranges:
            mask = torch.ones(len(tokens), dtype=torch.bool)
            for start, end in assistant_ranges:
                mask[start:end] = False
            labels[mask] = -100
        else:
            labels[:] = -100

        return input_ids, labels

    def _build_packed_indices(self):
        """构建packed序列的索引

        将多个短对话顺序拼接成接近max_length的长序列，
        返回packed样本列表，每个样本包含多个数据索引。
        """
        packed = []
        current_pack = []
        current_len = 0

        for idx in range(len(self.data)):
            item = self.data[idx]
            messages = item["messages"]
            text = self._apply_chatml(messages)
            tokens = self.tokenizer.encode(text)
            seq_len = len(tokens) + 1

            if current_len + seq_len > self.max_length:
                if len(current_pack) > 0:
                    packed.append(current_pack)
                    current_pack = []
                    current_len = 0
                if seq_len > self.max_length:
                    packed.append([idx])
                    continue

            current_pack.append(idx)
            current_len += seq_len

        if len(current_pack) > 0:
            packed.append(current_pack)

        return packed

    def __len__(self):
        if self.packed_indices is not None:
            return len(self.packed_indices)
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if self.packed_indices is not None:
            return self._get_packed_item(idx)
        return self._get_single_item(idx)

    def _get_single_item(self, idx) -> Dict[str, torch.Tensor]:
        """获取单条SFT样本（非packing模式）"""
        item = self.data[idx]
        messages = item["messages"]
        text = self._apply_chatml(messages)

        tokens = self.tokenizer.encode(text)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()

        assistant_ranges = self._find_assistant_ranges(tokens)
        if assistant_ranges:
            mask = torch.ones(len(tokens), dtype=torch.bool)
            for start, end in assistant_ranges:
                mask[start:end] = False
            labels[mask] = -100
        else:
            labels[:] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
            "loss_weights": torch.ones(len(input_ids), dtype=torch.float),
        }

    def _get_packed_item(self, idx) -> Dict[str, torch.Tensor]:
        """获取packed序列样本

        将indices中的多条短对话拼接成一个长序列，
        生成对应的segment_ids和loss_weights。
        """
        indices = self.packed_indices[idx]
        all_input_ids = []
        all_labels = []
        all_segment_ids = []

        for seg_idx, data_idx in enumerate(indices):
            input_ids, labels = self._tokenize_item(data_idx)
            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_segment_ids.append(torch.full((input_ids.size(0),), seg_idx, dtype=torch.long))

        input_ids = torch.cat(all_input_ids)
        labels = torch.cat(all_labels)
        segment_ids = torch.cat(all_segment_ids)
        seq_len = input_ids.size(0)

        if seq_len < self.max_length:
            pad_len = self.max_length - seq_len
            input_ids = torch.cat([input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
            segment_ids = torch.cat([segment_ids, torch.full((pad_len,), -1, dtype=torch.long)])

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        unique_segments = segment_ids[segment_ids >= 0].unique()
        loss_weights = torch.zeros(self.max_length, dtype=torch.float)
        if len(unique_segments) > 0:
            weight_per_segment = 1.0 / len(unique_segments)
            for sid in unique_segments:
                mask = segment_ids == sid
                assistant_mask = labels[mask] != -100
                seg_assistant_len = assistant_mask.sum().float()
                if seg_assistant_len > 0:
                    loss_weights[mask] = weight_per_segment / seg_assistant_len

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "segment_ids": segment_ids,
            "loss_weights": loss_weights,
        }


class DataCollatorForSFT:
    """青小团SFT数据collator - 支持sequence packing

    如果启用了packing，每条样本已经是packed后的完整序列，
    直接stack成batch即可。
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 524288, pack_sequences: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pack_sequences = pack_sequences

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        if self.pack_sequences:
            return self._collate_packed(features)
        return self._collate_padded(features)

    def _collate_packed(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """Packing模式：样本已经是max_length长度，直接stack"""
        input_ids = torch.stack([f["input_ids"] for f in features])
        labels = torch.stack([f["labels"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])
        loss_weights = torch.stack([f["loss_weights"] for f in features])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "loss_weights": loss_weights,
        }

    def _collate_padded(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """非packing模式：padding到相同长度"""
        input_ids_list = [f["input_ids"] for f in features]
        labels_list = [f["labels"] for f in features]
        attention_mask_list = [f["attention_mask"] for f in features]

        max_len = min(max(ids.size(0) for ids in input_ids_list), self.max_length)

        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for i in range(len(input_ids_list)):
            ids = input_ids_list[i]
            labs = labels_list[i]
            mask = attention_mask_list[i]

            if ids.size(0) > max_len:
                ids = ids[:max_len]
                labs = labs[:max_len]
                mask = mask[:max_len]

            pad_len = max_len - ids.size(0)
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
                labs = torch.cat([labs, torch.full((pad_len,), -100, dtype=torch.long)])
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])

            batch_input_ids.append(ids)
            batch_labels.append(labs)
            batch_attention_mask.append(mask)

        return {
            "input_ids": torch.stack(batch_input_ids, dim=0),
            "labels": torch.stack(batch_labels, dim=0),
            "attention_mask": torch.stack(batch_attention_mask, dim=0),
        }


def main():
    parser = HfArgumentParser((SFTArguments,))
    sft_args = parser.parse_args_into_dataclasses()[0]

    # ========================================================================
    # 消融实验 + 1M 上下文 设置
    # ========================================================================
    print(f"[1M 上下文] attention_mode={sft_args.attention_mode!r}")
    print(f"[1M 上下文] model_size={sft_args.model_size!r}")
    print(f"[1M 上下文] progressive_length_warmup={sft_args.enable_progressive_length_warmup}")
    print(f"[1M 上下文] max_length={sft_args.max_length}")

    # ---- 1) Progressive Length Warmup: 按阶段动态调整 max_length ----------
    # 在训练循环 (via trainer_callback) 中动态调整 dataset.max_length
    try:
        from qxt.model.config import (
            QXTConfig,
            get_current_max_seq_len,
            get_yarn_factor_for_seq_len,
        )
    except Exception:  # pragma: no cover
        QXTConfig = None
        get_current_max_seq_len = None
        get_yarn_factor_for_seq_len = None

    # ---- 2) Tokenizer ----
    tokenizer = QingxiaotuanTokenizer.from_pretrained(
        sft_args.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 3) 模型 + 注入 attention_mode / 1M 上下文参数 ----
    model = QingxiaotuanForCausalLM.from_pretrained(
        sft_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if sft_args.flash_attention else "sdpa",
    )

    # 把 attention_mode 注入到 config，便于我们的 modeling.py 工厂函数使用
    if hasattr(model.config, "attention_mode"):
        model.config.attention_mode = sft_args.attention_mode
    if hasattr(model.config, "max_position_embeddings"):
        model.config.max_position_embeddings = max(
            int(model.config.max_position_embeddings),
            int(sft_args.max_length),
        )
    if hasattr(model.config, "rope_theta") and getattr(model.config, "rope_theta", 0) < 1_000_000:
        model.config.rope_theta = 1_000_000.0

    model.config.use_cache = False

    # 日志：当前 attention_mode
    _am = getattr(model.config, "attention_mode", "standard")
    print(f"[消融实验] model.config.attention_mode={_am!r}")

    if sft_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("梯度检查点已启用")

    dataset = SFTDataset(
        data_path=sft_args.data_path,
        tokenizer=tokenizer,
        max_length=sft_args.max_length,
        pack_sequences=sft_args.pack_sequences,
    )
    print(f"加载数据集完成，共 {len(dataset)} 条样本")
    if sft_args.pack_sequences:
        print(f"序列打包已启用，{len(dataset)} 条packed序列")

    data_collator = DataCollatorForSFT(
        tokenizer=tokenizer,
        max_length=sft_args.max_length,
        pack_sequences=sft_args.pack_sequences,
    )

    training_args = TrainingArguments(
        output_dir=sft_args.output_dir,
        num_train_epochs=sft_args.num_epochs,
        per_device_train_batch_size=sft_args.per_device_train_batch_size,
        gradient_accumulation_steps=sft_args.gradient_accumulation_steps,
        learning_rate=sft_args.learning_rate,
        logging_steps=sft_args.logging_steps,
        save_steps=sft_args.save_steps,
        save_total_limit=sft_args.save_total_limit,
        bf16=True,
        tf32=True,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        report_to=["tensorboard"],
        ddp_find_unused_parameters=False,
        gradient_checkpointing=sft_args.gradient_checkpointing,
        logging_dir=os.path.join(sft_args.output_dir, "logs"),
        deepspeed=sft_args.deepspeed,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.1,
        prediction_loss_only=True,
        max_grad_norm=1.0,
    )

    # ---- 4) Progressive Length Warmup Callback ----------------------------
    if sft_args.enable_progressive_length_warmup and get_current_max_seq_len is not None:
        from transformers import TrainerCallback

        class ProgressiveLengthCallback(TrainerCallback):
            """每 `steps_per_stage` 步调整 dataset.max_length (4K -> 16K -> 64K -> 256K -> 1M)."""

            STAGES = [4096, 16384, 65536, 262144, 1048576]
            STEPS_PER_STAGE = 1000

            def __init__(self, dataset_ref, sft_args_ref):
                self.dataset_ref = dataset_ref
                self.sft_args_ref = sft_args_ref

            def on_step_end(self, args, state, control, **kwargs):
                global_step = int(state.global_step)
                target = max(
                    4096,
                    min(
                        self.sft_args_ref.max_length,
                        self.STAGES[
                            min(len(self.STAGES) - 1,
                                global_step // max(1, self.STEPS_PER_STAGE))
                        ],
                    ),
                )
                if self.dataset_ref.max_length != target:
                    self.dataset_ref.max_length = target
                    if hasattr(self.dataset_ref, "packed_indices") and self.dataset_ref.pack_sequences:
                        self.dataset_ref.packed_indices = self.dataset_ref._build_packed_indices()
                    if global_step % 50 == 0:
                        print(
                            f"[ProgressiveLength] step={global_step}: dataset.max_length → {target}"
                        )

    else:
        ProgressiveLengthCallback = None  # type: ignore

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    if ProgressiveLengthCallback is not None:
        trainer.add_callback(ProgressiveLengthCallback(dataset, sft_args))
        print("[1M 上下文] Progressive Length Warmup callback 已注册")

    trainer.train()
    trainer.save_model(sft_args.output_dir)
    tokenizer.save_pretrained(sft_args.output_dir)
    print(f"SFT训练完成，模型已保存至: {sft_args.output_dir}")


if __name__ == "__main__":
    main()
