import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from qingxiaotuan import QingxiaotuanConfig, QingxiaotuanForCausalLM, QingxiaotuanTokenizer


class QingxiaotuanRewardModel(PreTrainedModel):
    """青小团奖励模型 - 青奈科技

    基于QingxiaotuanForCausalLM，在最后一层隐藏状态上添加value head
    用于序列分类（奖励打分）。

    对于batch中的每个序列，生成一个标量奖励值。
    """

    config_class = QingxiaotuanConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["QingxiaotuanDecoderLayer"]

    def __init__(self, config: QingxiaotuanConfig):
        super().__init__(config)
        self.num_labels = 1
        self.model = QingxiaotuanForCausalLM(config)

        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 1),
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.model.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.model.embed_tokens = value

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return_dict = return_dict if return_dict is not None else True

        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state

        if attention_mask is not None:
            last_token_indices = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(
                hidden_states.size(0), device=hidden_states.device
            )
            pooled_hidden = hidden_states[batch_indices, last_token_indices]
        else:
            pooled_hidden = hidden_states[:, -1, :]

        rewards = self.value_head(pooled_hidden)
        rewards = rewards.squeeze(-1)

        loss = None
        if labels is not None:
            import torch.nn.functional as F
            if labels.size(0) % 2 == 0:
                chosen_rewards = rewards[0::2]
                rejected_rewards = rewards[1::2]
                loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

        if not return_dict:
            return (loss, rewards) if loss is not None else (rewards,)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=rewards,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )


def create_reward_model(
    config: QingxiaotuanConfig,
    pretrained_model_name_or_path: Optional[str] = None,
    load_pretrained_base: bool = True,
) -> QingxiaotuanRewardModel:
    """创建青小团奖励模型 - 青奈科技

    基于QingxiaotuanConfig创建序列分类模型，可选择加载预训练权重。

    Args:
        config: 模型配置
        pretrained_model_name_or_path: 预训练模型路径（可选）
        load_pretrained_base: 是否加载预训练模型的base权重

    Returns:
        初始化的QingxiaotuanRewardModel实例
    """
    model = QingxiaotuanRewardModel(config)

    if pretrained_model_name_or_path is not None and load_pretrained_base:
        base_model = QingxiaotuanForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.model = base_model
        print(f"已加载预训练模型权重: {pretrained_model_name_or_path}")

    return model


if __name__ == "__main__":
    config = QingxiaotuanConfig(
        vocab_size=152064,
        hidden_size=7168,
        num_hidden_layers=48,
        num_attention_heads=56,
        num_key_value_heads=8,
        intermediate_size=20480,
    )
    model = create_reward_model(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"奖励模型参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print("奖励模型创建成功！")
