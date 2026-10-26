"""
青小团品牌心智注入层测试 - 覆盖系统提示词、身份训练数据、apply_persona等
"""

import os
import json
import pytest
from qingxiaotuan.brand_persona import (
    QINGXIAOTUAN_SYSTEM_PROMPT,
    QINGXIAOTUAN_SYSTEM_PROMPT_ZH,
    QINGXIAOTUAN_SYSTEM_PROMPT_EN,
    QINGXIAOTUAN_PERSONA_TRAINING_DATA,
    apply_qingxiaotuan_persona,
    get_qingxiaotuan_training_data,
    generate_identity_dataset,
)


class TestQingxiaotuanSystemPrompt:

    def test_system_prompt_zh_exists(self):
        assert QINGXIAOTUAN_SYSTEM_PROMPT_ZH is not None
        assert len(QINGXIAOTUAN_SYSTEM_PROMPT_ZH) > 50

    def test_system_prompt_en_exists(self):
        assert QINGXIAOTUAN_SYSTEM_PROMPT_EN is not None
        assert len(QINGXIAOTUAN_SYSTEM_PROMPT_EN) > 50

    def test_system_prompt_dict_has_keys(self):
        assert "zh" in QINGXIAOTUAN_SYSTEM_PROMPT
        assert "en" in QINGXIAOTUAN_SYSTEM_PROMPT

    def test_system_prompt_contains_identity(self):
        assert "青小团" in QINGXIAOTUAN_SYSTEM_PROMPT_ZH
        assert "青奈科技" in QINGXIAOTUAN_SYSTEM_PROMPT_ZH

    def test_system_prompt_en_contains_identity(self):
        assert "Qingxiaotuan" in QINGXIAOTUAN_SYSTEM_PROMPT_EN
        assert "Qingnai" in QINGXIAOTUAN_SYSTEM_PROMPT_EN

    def test_system_prompt_mentions_moe(self):
        assert "MoE" in QINGXIAOTUAN_SYSTEM_PROMPT_ZH or "混合专家" in QINGXIAOTUAN_SYSTEM_PROMPT_ZH


class TestQingxiaotuanPersonaTrainingData:

    def test_training_data_count(self):
        assert len(QINGXIAOTUAN_PERSONA_TRAINING_DATA) >= 19

    def test_each_item_has_messages(self):
        for item in QINGXIAOTUAN_PERSONA_TRAINING_DATA:
            assert "messages" in item
            assert len(item["messages"]) == 2

    def test_each_item_has_user_and_assistant(self):
        for item in QINGXIAOTUAN_PERSONA_TRAINING_DATA:
            assert item["messages"][0]["role"] == "user"
            assert item["messages"][1]["role"] == "assistant"

    def test_no_other_brand_names(self):
        forbidden = ["qwen", "deepseek", "llama"]
        text = str(QINGXIAOTUAN_PERSONA_TRAINING_DATA).lower()
        for name in forbidden:
            assert name not in text, f"Found forbidden name: {name}"

    def test_identity_data_mentions_qingxiaotuan(self):
        for item in QINGXIAOTUAN_PERSONA_TRAINING_DATA:
            assistant_msg = item["messages"][1]["content"]
            assert "青小团" in assistant_msg or "Qingxiaotuan" in assistant_msg

    def test_identity_data_mentions_qingnai(self):
        for item in QINGXIAOTUAN_PERSONA_TRAINING_DATA:
            assistant_msg = item["messages"][1]["content"]
            assert "青奈" in assistant_msg or "Qingnai" in assistant_msg

    def test_english_identity_exists(self):
        english_items = [i for i in QINGXIAOTUAN_PERSONA_TRAINING_DATA
                        if i["messages"][0]["content"] == "Who are you?"]
        assert len(english_items) >= 1

    def test_architecture_description(self):
        arch_items = [i for i in QINGXIAOTUAN_PERSONA_TRAINING_DATA
                      if "架构" in i["messages"][0]["content"] or "architecture" in i["messages"][0]["content"]]
        assert len(arch_items) >= 1

    def test_version_info(self):
        version_items = [i for i in QINGXIAOTUAN_PERSONA_TRAINING_DATA
                         if "版本" in i["messages"][0]["content"]]
        assert len(version_items) >= 1


class TestQingxiaotuanApplyPersona:

    def test_apply_persona_returns_list(self):
        result = apply_qingxiaotuan_persona("你好")
        assert isinstance(result, list)
        assert len(result) == 2

    def test_apply_persona_first_is_system(self):
        result = apply_qingxiaotuan_persona("Hello")
        assert result[0]["role"] == "system"

    def test_apply_persona_second_is_user(self):
        result = apply_qingxiaotuan_persona("Test message")
        assert result[1]["role"] == "user"

    def test_apply_persona_preserves_message(self):
        msg = "这是一条测试消息"
        result = apply_qingxiaotuan_persona(msg)
        assert result[1]["content"] == msg

    def test_apply_persona_zh_default(self):
        result = apply_qingxiaotuan_persona("你好")
        assert "青小团" in result[0]["content"]

    def test_apply_persona_en(self):
        result = apply_qingxiaotuan_persona("Hello", language="en")
        assert "Qingxiaotuan" in result[0]["content"]

    def test_apply_persona_invalid_language_falls_back(self):
        result = apply_qingxiaotuan_persona("Hello", language="ja")
        assert "青小团" in result[0]["content"]


class TestQingxiaotuanGetTrainingData:

    def test_get_training_data_returns_list(self):
        data = get_qingxiaotuan_training_data()
        assert isinstance(data, list)
        assert len(data) >= 19

    def test_get_training_data_is_copy(self):
        data1 = get_qingxiaotuan_training_data()
        data2 = get_qingxiaotuan_training_data()
        assert data1 == data2

    def test_get_training_data_content(self):
        data = get_qingxiaotuan_training_data()
        first = data[0]
        assert first["messages"][1]["content"].startswith("我是青小团")


class TestQingxiaotuanGenerateIdentityDataset:

    def test_generate_identity_dataset_creates_file(self, temp_dir):
        output_path = os.path.join(temp_dir, "identity.jsonl")
        result = generate_identity_dataset(output_path=output_path)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0

    def test_generate_identity_dataset_format(self, temp_dir):
        output_path = os.path.join(temp_dir, "identity.jsonl")
        generate_identity_dataset(output_path=output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) >= 19
        first = json.loads(lines[0])
        assert "messages" in first
