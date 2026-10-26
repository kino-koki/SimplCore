import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from torch.utils.data import Dataset
from transformers import HfArgumentParser, TrainingArguments
from qingxiaotuan import QingxiaotuanForCausalLM, QingxiaotuanTokenizer, QingxiaotuanConfig
from reward_model import QingxiaotuanRewardModel


IM_START_ID = 151644
IM_END_ID = 151645
EOS_ID = 151643


class QingxiaotuanModelWithValueHead(nn.Module):
    """青小团PPO Actor模型（带Value Head） - 青奈科技

    包装QingxiaotuanForCausalLM，添加value head用于PPO的critic网络。
    兼容trl库的PPOTrainer接口要求。
    """

    def __init__(self, config: QingxiaotuanConfig, checkpoint_path: Optional[str] = None):
        super().__init__()
        if checkpoint_path is not None:
            self.pretrained_model = QingxiaotuanForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        else:
            self.pretrained_model = QingxiaotuanForCausalLM(config)

        hidden_size = config.hidden_size
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.value_head.to(dtype=torch.bfloat16)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict:
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states[-1] if outputs.hidden_states is not None else None
        if hidden_states is None:
            model_outputs = self.pretrained_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = model_outputs.last_hidden_state

        if attention_mask is not None:
            last_token_indices = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            pooled_hidden = hidden_states[batch_indices, last_token_indices]
        else:
            pooled_hidden = hidden_states[:, -1, :]

        values = self.value_head(pooled_hidden).squeeze(-1)

        return {
            "logits": outputs.logits if hasattr(outputs, "logits") else outputs[0],
            "values": values,
            "loss": outputs.loss if hasattr(outputs, "loss") else None,
        }

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **generation_kwargs,
    ) -> torch.LongTensor:
        return self.pretrained_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )

    def save_pretrained(self, output_dir: str):
        self.pretrained_model.save_pretrained(output_dir)

    @property
    def config(self):
        return self.pretrained_model.config

    @property
    def device(self):
        return self.pretrained_model.device


@dataclass
class PPOArguments:
    sft_model_path: str = field(
        default=None,
        metadata={"help": "SFT模型路径"},
    )
    reward_model_path: str = field(
        default=None,
        metadata={"help": "Reward Model路径"},
    )
    data_path: str = field(
        default=None,
        metadata={"help": "PPO训练数据路径（JSON/JSONL）"},
    )
    output_dir: str = field(
        default="./output_rlhf",
        metadata={"help": "输出目录"},
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "最大序列长度"},
    )
    max_prompt_length: int = field(
        default=1024,
        metadata={"help": "最大prompt长度"},
    )
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "PPO学习率"},
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "PPO batch size"},
    )
    mini_batch_size: int = field(
        default=4,
        metadata={"help": "PPO mini batch size"},
    )
    kl_penalty: float = field(
        default=0.1,
        metadata={"help": "KL散度惩罚系数"},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "梯度累积步数"},
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
    num_ppo_epochs: int = field(
        default=4,
        metadata={"help": "PPO内部训练轮数"},
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "总训练轮数"},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={"help": "DeepSpeed配置文件路径"},
    )
    init_kl_coef: float = field(
        default=0.1,
        metadata={"help": "初始KL系数"},
    )
    target: float = field(
        default=6,
        metadata={"help": "KL目标值"},
    )
    horizon: float = field(
        default=10000,
        metadata={"help": "KL自适应horizon"},
    )
    gamma: float = field(
        default=1,
        metadata={"help": "GAE gamma"},
    )
    lam: float = field(
        default=0.95,
        metadata={"help": "GAE lambda"},
    )
    cliprange: float = field(
        default=0.2,
        metadata={"help": "PPO clip范围"},
    )
    cliprange_value: float = field(
        default=0.2,
        metadata={"help": "Value clip范围"},
    )
    vf_coef: float = field(
        default=0.1,
        metadata={"help": "Value function loss系数"},
    )


class PPODataset(Dataset):
    """青小团PPO数据集 - 青奈科技

    加载prompt数据，用于PPO训练时生成response并获取reward。
    """

    def __init__(self, data_path: str, tokenizer: QingxiaotuanTokenizer, max_prompt_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.prompts = self._load_data(data_path)

    def _load_data(self, data_path: str) -> List[str]:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        ext = os.path.splitext(data_path)[1].lower()
        prompts = []
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
            raise ValueError(f"不支持的文件格式: {ext}")

        for item in raw_data:
            if "prompt" in item:
                prompt_text = item["prompt"]
            elif "instruction" in item:
                inp = item.get("input", "")
                prompt_text = item["instruction"]
                if inp:
                    prompt_text = prompt_text + "\n" + inp
            elif "conversations" in item:
                conv = item["conversations"]
                prompt_text = ""
                for turn in conv:
                    role = turn.get("role", turn.get("from", ""))
                    content = turn.get("content", turn.get("value", ""))
                    if role == "user":
                        prompt_text += f"<|im_start|>user\n{content}<|im_end|>\n"
                    elif role == "system":
                        prompt_text += f"<|im_start|>system\n{content}<|im_end|>\n"
                    elif role == "assistant":
                        continue
                prompt_text += "<|im_start|>assistant\n"
            elif "messages" in item:
                prompt_text = ""
                for msg in item["messages"]:
                    if msg["role"] == "user":
                        prompt_text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
                    elif msg["role"] == "system":
                        prompt_text += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
                    elif msg["role"] == "assistant":
                        continue
                prompt_text += "<|im_start|>assistant\n"
            else:
                continue
            prompts.append(prompt_text)

        return prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx) -> Dict:
        prompt = self.prompts[idx]
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) > self.max_prompt_length:
            tokens = tokens[:self.max_prompt_length]
        return {
            "query": prompt,
            "input_ids": torch.tensor(tokens, dtype=torch.long),
        }


def compute_reward(
    reward_model: QingxiaotuanRewardModel,
    query_input_ids: torch.LongTensor,
    response_input_ids: torch.LongTensor,
    tokenizer: QingxiaotuanTokenizer,
    device: torch.device,
) -> torch.Tensor:
    """计算生成回复的奖励值"""
    full_input_ids = torch.cat([query_input_ids, response_input_ids], dim=-1)
    full_input_ids = full_input_ids.to(device)
    attention_mask = torch.ones_like(full_input_ids)

    with torch.no_grad():
        outputs = reward_model(
            input_ids=full_input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
    return outputs.logits


def main():
    parser = HfArgumentParser((PPOArguments,))
    ppo_args = parser.parse_args_into_dataclasses()[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = QingxiaotuanTokenizer.from_pretrained(
        ppo_args.sft_model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("加载SFT模型...")
    actor_model = QingxiaotuanModelWithValueHead(
        config=None,
        checkpoint_path=ppo_args.sft_model_path,
    )
    actor_model = actor_model.to(device)
    actor_model.train()

    print("创建参考模型...")
    ref_model = QingxiaotuanModelWithValueHead(
        config=None,
        checkpoint_path=ppo_args.sft_model_path,
    )
    ref_model = ref_model.to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    print("加载Reward Model...")
    reward_config = QingxiaotuanConfig.from_pretrained(
        ppo_args.reward_model_path,
        trust_remote_code=True,
    )
    reward_model = QingxiaotuanRewardModel(reward_config)
    reward_state_dict = torch.load(
        os.path.join(ppo_args.reward_model_path, "pytorch_model.bin"),
        map_location=device,
    )
    reward_model.load_state_dict(reward_state_dict, strict=False)
    reward_model = reward_model.to(device)
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False

    print("加载PPO数据集...")
    dataset = PPODataset(
        data_path=ppo_args.data_path,
        tokenizer=tokenizer,
        max_prompt_length=ppo_args.max_prompt_length,
    )
    print(f"数据集大小: {len(dataset)}")

    optimizer = torch.optim.AdamW(
        actor_model.parameters(),
        lr=ppo_args.learning_rate,
        weight_decay=0.1,
    )

    global_step = 0
    for epoch in range(ppo_args.num_train_epochs):
        print(f"\n开始 Epoch {epoch + 1}/{ppo_args.num_train_epochs}")

        for batch_idx in range(0, len(dataset), ppo_args.batch_size):
            batch_end = min(batch_idx + ppo_args.batch_size, len(dataset))
            batch_indices = list(range(batch_idx, batch_end))
            batch_queries = [dataset[i]["query"] for i in batch_indices]

            query_tokens = tokenizer(
                batch_queries,
                padding=True,
                truncation=True,
                max_length=ppo_args.max_prompt_length,
                return_tensors="pt",
            )
            query_input_ids = query_tokens["input_ids"].to(device)
            query_attention_mask = query_tokens["attention_mask"].to(device)

            with torch.no_grad():
                response_ids = actor_model.generate(
                    input_ids=query_input_ids,
                    attention_mask=query_attention_mask,
                    max_new_tokens=min(512, ppo_args.max_length - query_input_ids.size(1)),
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.8,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                response_only_ids = response_ids[:, query_input_ids.size(1):]
                response_attention = torch.ones_like(response_only_ids)

            full_input_ids = torch.cat([query_input_ids, response_only_ids], dim=1)
            full_attention_mask = torch.cat(
                [query_attention_mask, response_attention], dim=1
            )

            with torch.no_grad():
                rewards = compute_reward(
                    reward_model=reward_model,
                    query_input_ids=query_input_ids,
                    response_input_ids=response_only_ids,
                    tokenizer=tokenizer,
                    device=device,
                )

            num_mini_batches = max(1, ppo_args.batch_size // ppo_args.mini_batch_size)
            mini_batch_size = len(batch_indices) // num_mini_batches

            for mini_idx in range(num_mini_batches):
                start = mini_idx * mini_batch_size
                end = min((mini_idx + 1) * mini_batch_size, len(batch_indices))

                if start >= len(batch_indices):
                    break

                mb_input_ids = full_input_ids[start:end]
                mb_attention_mask = full_attention_mask[start:end]
                mb_rewards = rewards[start:end]
                mb_query_ids = query_input_ids[start:end]

                if mb_input_ids.size(0) == 0:
                    continue

                outputs = actor_model(
                    input_ids=mb_input_ids,
                    attention_mask=mb_attention_mask,
                )
                logits = outputs["logits"]
                values = outputs["values"]

                with torch.no_grad():
                    ref_outputs = ref_model(
                        input_ids=mb_input_ids,
                        attention_mask=mb_attention_mask,
                    )
                    ref_logits = ref_outputs["logits"]

                response_length = mb_input_ids.size(1) - mb_query_ids.size(1)
                if response_length <= 0:
                    continue

                log_probs = F.log_softmax(logits, dim=-1)
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)

                response_log_probs = []
                ref_response_log_probs = []
                for b in range(mb_input_ids.size(0)):
                    q_len = mb_query_ids.size(1)
                    resp_ids = mb_input_ids[b, q_len:]
                    if resp_ids.size(0) == 0:
                        response_log_probs.append(torch.tensor(0.0, device=device))
                        ref_response_log_probs.append(torch.tensor(0.0, device=device))
                        continue
                    rlp = log_probs[b, q_len - 1:-1].gather(
                        1, resp_ids.unsqueeze(-1)
                    ).squeeze(-1).mean()
                    ref_rlp = ref_log_probs[b, q_len - 1:-1].gather(
                        1, resp_ids.unsqueeze(-1)
                    ).squeeze(-1).mean()
                    response_log_probs.append(rlp)
                    ref_response_log_probs.append(ref_rlp)

                response_log_probs = torch.stack(response_log_probs)
                ref_response_log_probs = torch.stack(ref_response_log_probs)

                kl_div = (ref_response_log_probs - response_log_probs).mean()

                pg_loss = -(mb_rewards * response_log_probs).mean()

                value_loss = F.mse_loss(values, mb_rewards)

                loss = pg_loss + ppo_args.vf_coef * value_loss + ppo_args.kl_penalty * kl_div

                loss = loss / num_mini_batches

                loss.backward()

            torch.nn.utils.clip_grad_norm_(actor_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

            if global_step % ppo_args.logging_steps == 0:
                avg_reward = rewards.mean().item()
                print(
                    f"Step {global_step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Reward: {avg_reward:.4f} | "
                    f"KL: {kl_div.item():.4f} | "
                    f"PG Loss: {pg_loss.item():.4f} | "
                    f"Value Loss: {value_loss.item():.4f}"
                )

            if global_step % ppo_args.save_steps == 0:
                step_output_dir = os.path.join(ppo_args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(step_output_dir, exist_ok=True)
                actor_model.save_pretrained(step_output_dir)
                tokenizer.save_pretrained(step_output_dir)
                print(f"模型已保存至: {step_output_dir}")

    final_output_dir = ppo_args.output_dir
    os.makedirs(final_output_dir, exist_ok=True)
    actor_model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"PPO训练完成！模型已保存至: {final_output_dir}")


if __name__ == "__main__":
    main()
