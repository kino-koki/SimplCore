"""
青小团合成数据测试 - 覆盖指令、推理、代码、对话、知识问答生成器及数据验证
"""

import os
import json
import tempfile
import pytest


class TestQingxiaotuanInstructionGenerator:

    def test_instruction_generator_creation(self):
        from training.synthetic_data.base_generator import InstructionGenerator
        generator = InstructionGenerator()
        assert generator is not None

    def test_instruction_generator_single(self):
        from training.synthetic_data.base_generator import InstructionGenerator
        generator = InstructionGenerator()
        result = generator.generate(topic="写一篇关于人工智能的文章")
        assert result is not None
        assert "instruction" in result or "messages" in result

    def test_instruction_generator_batch(self):
        from training.synthetic_data.base_generator import InstructionGenerator
        generator = InstructionGenerator()
        topics = ["主题1", "主题2", "主题3"]
        results = generator.generate_batch(topics)
        assert len(results) == 3
        for r in results:
            assert r is not None

    def test_instruction_format_check(self):
        from training.synthetic_data.base_generator import InstructionGenerator
        generator = InstructionGenerator()
        result = generator.generate(topic="测试")
        assert isinstance(result, dict)

    def test_instruction_generator_empty_topic(self):
        from training.synthetic_data.base_generator import InstructionGenerator
        generator = InstructionGenerator()
        with pytest.raises((ValueError, Exception)):
            generator.generate(topic="")


class TestQingxiaotuanReasoningGenerator:

    def test_reasoning_generator_creation(self):
        from training.synthetic_data.base_generator import ReasoningGenerator
        generator = ReasoningGenerator()
        assert generator is not None

    def test_reasoning_generator_step_by_step(self):
        from training.synthetic_data.base_generator import ReasoningGenerator
        generator = ReasoningGenerator()
        result = generator.generate(difficulty="medium")
        assert result is not None

    def test_reasoning_generator_easy(self):
        from training.synthetic_data.base_generator import ReasoningGenerator
        generator = ReasoningGenerator()
        result = generator.generate(difficulty="easy")
        assert result is not None

    def test_reasoning_generator_hard(self):
        from training.synthetic_data.base_generator import ReasoningGenerator
        generator = ReasoningGenerator()
        result = generator.generate(difficulty="hard")
        assert result is not None

    def test_reasoning_step_by_step_format(self):
        from training.synthetic_data.base_generator import ReasoningGenerator
        generator = ReasoningGenerator()
        result = generator.generate()
        output = result.get("output", "")
        assert "因此" in output or "所以" in output or "步骤" in output or "过程" in output

    def test_reasoning_batch_generate(self):
        from training.synthetic_data.base_generator import ReasoningGenerator
        generator = ReasoningGenerator()
        results = generator.generate_batch(count=5)
        assert len(results) == 5

    def test_reasoning_difficulty_levels(self):
        from training.synthetic_data.base_generator import ReasoningGenerator
        generator = ReasoningGenerator()
        assert generator.difficulty_levels == ["easy", "medium", "hard"]


class TestQingxiaotuanCodeGenerator:

    def test_code_generator_creation(self):
        from training.synthetic_data.base_generator import CodeGenerator
        generator = CodeGenerator()
        assert generator is not None

    def test_code_generator_python(self):
        from training.synthetic_data.base_generator import CodeGenerator
        generator = CodeGenerator()
        result = generator.generate(language="python", task="实现二分查找")
        assert result is not None

    def test_code_generator_has_language_tags(self):
        from training.synthetic_data.base_generator import CodeGenerator
        generator = CodeGenerator()
        result = generator.generate(language="python", task="写一个函数")
        output = result.get("output", "")
        assert "python" in output.lower() or "def " in output

    def test_code_generator_multi_language(self):
        from training.synthetic_data.base_generator import CodeGenerator
        generator = CodeGenerator()
        for lang in ["python", "java", "cpp"]:
            result = generator.generate(language=lang, task="hello world")
            assert result is not None

    def test_code_batch_generation(self):
        from training.synthetic_data.base_generator import CodeGenerator
        generator = CodeGenerator()
        tasks = ["task1", "task2", "task3"]
        results = generator.generate_batch(language="python", tasks=tasks)
        assert len(results) == 3


class TestQingxiaotuanConversationGenerator:

    def test_conversation_generator_creation(self):
        from training.synthetic_data.base_generator import ConversationGenerator
        generator = ConversationGenerator()
        assert generator is not None

    def test_conversation_generator_multi_turn(self):
        from training.synthetic_data.base_generator import ConversationGenerator
        generator = ConversationGenerator()
        result = generator.generate(turns=4, topic="讨论人工智能")
        assert result is not None
        msgs = result.get("messages", [])
        assert len(msgs) == 4

    def test_conversation_turn_count(self):
        from training.synthetic_data.base_generator import ConversationGenerator
        generator = ConversationGenerator()
        result = generator.generate(turns=2, topic="测试")
        msgs = result.get("messages", [])
        assert len(msgs) == 2

    def test_conversation_role_alternation(self):
        from training.synthetic_data.base_generator import ConversationGenerator
        generator = ConversationGenerator()
        result = generator.generate(turns=6, topic="测试")
        msgs = result.get("messages", [])
        expected_roles = ["user", "assistant", "user", "assistant", "user", "assistant"]
        for i, msg in enumerate(msgs):
            assert msg["role"] == expected_roles[i]

    def test_conversation_batch_generation(self):
        from training.synthetic_data.base_generator import ConversationGenerator
        generator = ConversationGenerator()
        results = generator.generate_batch(count=4, turns=2)
        assert len(results) == 4


class TestQingxiaotuanKnowledgeQA:

    def test_knowledge_qa_creation(self):
        from training.synthetic_data.base_generator import KnowledgeQAGenerator
        generator = KnowledgeQAGenerator()
        assert generator is not None

    def test_knowledge_qa_generate(self):
        from training.synthetic_data.base_generator import KnowledgeQAGenerator
        generator = KnowledgeQAGenerator()
        result = generator.generate(domain="科学", subdomain="物理")
        assert result is not None
        assert "question" in result or "instruction" in result

    def test_knowledge_qa_domain_variety(self):
        from training.synthetic_data.base_generator import KnowledgeQAGenerator
        generator = KnowledgeQAGenerator()
        for domain in ["科学", "历史", "文学", "技术", "医学"]:
            result = generator.generate(domain=domain)
            assert result is not None


class TestQingxiaotuanDataAugmentation:

    def test_data_augmentation_creation(self):
        from training.synthetic_data.base_generator import DataAugmentor
        augmentor = DataAugmentor()
        assert augmentor is not None

    def test_data_augmentation_paraphrase(self):
        from training.synthetic_data.base_generator import DataAugmentor
        augmentor = DataAugmentor()
        text = "青小团是一个大语言模型。"
        result = augmentor.paraphrase(text)
        assert result is not None
        assert len(result) > 0

    def test_data_augmentation_no_op(self):
        from training.synthetic_data.base_generator import DataAugmentor
        augmentor = DataAugmentor()
        result = augmentor.paraphrase("")
        assert result == ""

    def test_data_augmented_count(self):
        from training.synthetic_data.base_generator import DataAugmentor
        augmentor = DataAugmentor()
        texts = ["文本1", "文本2", "文本3"]
        results = augmentor.augment_batch(texts)
        assert len(results) == 3


class TestQingxiaotuanDataValidator:

    def test_validator_creation(self):
        from training.synthetic_data.data_validator import DataValidator
        validator = DataValidator()
        assert validator is not None

    def test_validator_check_structure(self):
        from training.synthetic_data.data_validator import DataValidator
        validator = DataValidator()
        item = {"instruction": "test", "output": "test output"}
        result = validator.validate(item)
        assert result is True

    def test_validator_check_missing_fields(self):
        from training.synthetic_data.data_validator import DataValidator
        validator = DataValidator()
        item = {"instruction": "test"}
        result = validator.validate(item)
        assert result is False

    def test_validator_check_empty_output(self):
        from training.synthetic_data.data_validator import DataValidator
        validator = DataValidator()
        item = {"instruction": "test", "output": ""}
        result = validator.validate(item)
        assert result is False

    def test_validator_batch_check(self):
        from training.synthetic_data.data_validator import DataValidator
        validator = DataValidator()
        items = [
            {"instruction": "a", "output": "b"},
            {"instruction": "c", "output": ""},
            {"instruction": "", "output": "d"},
        ]
        results = validator.validate_batch(items)
        assert results == [True, False, False]

    def test_validator_checks_quality(self):
        from training.synthetic_data.data_validator import DataValidator
        validator = DataValidator(min_length=10)
        item = {"instruction": "a" * 20, "output": "b" * 20}
        result = validator.validate(item)
        assert result is True


class TestQingxiaotuanSyntheticDataFactory:

    def test_factory_creation(self):
        from training.synthetic_data.synthetic_dataset_factory import SyntheticDatasetFactory
        factory = SyntheticDatasetFactory()
        assert factory is not None

    def test_factory_register_generator(self):
        from training.synthetic_data.synthetic_dataset_factory import SyntheticDatasetFactory
        from training.synthetic_data.base_generator import InstructionGenerator
        factory = SyntheticDatasetFactory()
        factory.register_generator("instruction", InstructionGenerator)
        assert "instruction" in factory.generators

    def test_factory_create_instruction(self):
        from training.synthetic_data.synthetic_dataset_factory import SyntheticDatasetFactory
        from training.synthetic_data.base_generator import InstructionGenerator
        factory = SyntheticDatasetFactory()
        factory.register_generator("instruction", InstructionGenerator)
        dataset = factory.create("instruction", count=5)
        assert len(dataset) == 5

    def test_factory_create_conversation(self):
        from training.synthetic_data.synthetic_dataset_factory import SyntheticDatasetFactory
        from training.synthetic_data.base_generator import ConversationGenerator
        factory = SyntheticDatasetFactory()
        factory.register_generator("conversation", ConversationGenerator)
        dataset = factory.create("conversation", count=4)
        assert len(dataset) == 4

    def test_factory_create_mixed(self):
        from training.synthetic_data.synthetic_dataset_factory import SyntheticDatasetFactory
        from training.synthetic_data.base_generator import (InstructionGenerator, CodeGenerator, ReasoningGenerator)
        factory = SyntheticDatasetFactory()
        factory.register_generator("instruction", InstructionGenerator)
        factory.register_generator("code", CodeGenerator)
        factory.register_generator("reasoning", ReasoningGenerator)
        dataset = factory.create_mixed(count=10, ratios={"instruction": 0.4, "code": 0.3, "reasoning": 0.3})
        assert len(dataset) == 10

    def test_factory_save_dataset(self, temp_dir):
        from training.synthetic_data.synthetic_dataset_factory import SyntheticDatasetFactory
        from training.synthetic_data.base_generator import InstructionGenerator
        factory = SyntheticDatasetFactory()
        factory.register_generator("instruction", InstructionGenerator)
        dataset = factory.create("instruction", count=2)
        output_file = os.path.join(temp_dir, "synthetic.jsonl")
        factory.save(dataset, output_file)
        assert os.path.getsize(output_file) > 0

    def test_factory_generator_not_found(self):
        from training.synthetic_data.synthetic_dataset_factory import SyntheticDatasetFactory
        factory = SyntheticDatasetFactory()
        with pytest.raises(KeyError):
            factory.create("nonexistent")
