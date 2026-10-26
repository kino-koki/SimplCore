"""
青小团训练模块测试 - 覆盖DataLoader、Sequence Packing、SFT Dataset、Loss Masking、梯度累积等
"""

import os
import json
import torch
import pytest
import tempfile
from qingxiaotuan import QingxiaotuanConfig, QingxiaotuanForCausalLM


class TestQingxiaotuanPretrainDataset:

    def test_pretrain_dataset_creation(self, sample_pretrain_path):
        from training.pretrain.dataset import QingxiaotuanPretrainDataset
        from qingxiaotuan import QingxiaotuanTokenizer
        try:
            tokenizer = QingxiaotuanTokenizer.from_pretrained("qingxiaotuan")
        except Exception:
            pytest.skip("Tokenizer not available, skipping")
        dataset = QingxiaotuanPretrainDataset(
            data_files=sample_pretrain_path,
            tokenizer=tokenizer,
            max_length=512,
            pack_sequences=False,
        )
        assert dataset is not None
        assert dataset.pack_sequences is False

    def test_pretrain_dataset_packing(self, sample_pretrain_path):
        from training.pretrain.dataset import QingxiaotuanPretrainDataset
        from qingxiaotuan import QingxiaotuanTokenizer
        try:
            tokenizer = QingxiaotuanTokenizer.from_pretrained("qingxiaotuan")
        except Exception:
            pytest.skip("Tokenizer not available, skipping")
        dataset = QingxiaotuanPretrainDataset(
            data_files=sample_pretrain_path,
            tokenizer=tokenizer,
            max_length=512,
            pack_sequences=True,
        )
        assert dataset.pack_sequences is True

    def test_collate_fn_for_packing(self):
        from training.pretrain.dataset import collate_fn_for_packing
        batch = [
            {"input_ids": torch.randint(0, 1000, (128,)), "labels": torch.randint(0, 1000, (128,)),
             "attention_mask": torch.ones(128, dtype=torch.long), "loss_weights": torch.ones(128)},
            {"input_ids": torch.randint(0, 1000, (128,)), "labels": torch.randint(0, 1000, (128,)),
             "attention_mask": torch.ones(128, dtype=torch.long), "loss_weights": torch.ones(128)},
        ]
        collated = collate_fn_for_packing(batch)
        assert collated["input_ids"].shape == (2, 128)


class TestQingxiaotuanSequencePacking:

    def test_pack_short_sequences(self, sample_pretrain_path):
        from training.pretrain.dataset import QingxiaotuanPretrainDataset
        from qingxiaotuan import QingxiaotuanTokenizer
        try:
            tokenizer = QingxiaotuanTokenizer.from_pretrained("qingxiaotuan")
        except Exception:
            pytest.skip("Tokenizer not available, skipping")
        dataset = QingxiaotuanPretrainDataset(
            data_files=sample_pretrain_path,
            tokenizer=tokenizer,
            max_length=512,
            pack_sequences=True,
        )
        samples = []
        for i, sample in enumerate(dataset):
            if i >= 5:
                break
            samples.append(sample)
        assert len(samples) > 0

    def test_pack_sequences_segment_ids(self):
        from training.pretrain.dataset import QingxiaotuanPretrainDataset
        buffer = [
            {"input_ids": torch.randint(0, 1000, (50,)), "labels": torch.randint(0, 1000, (50,))},
            {"input_ids": torch.randint(0, 1000, (30,)), "labels": torch.randint(0, 1000, (30,))},
        ]
        dataset_instance = QingxiaotuanPretrainDataset.__new__(QingxiaotuanPretrainDataset)
        dataset_instance.max_length = 512
        dataset_instance.tokenizer = None
        packed = dataset_instance._pack_sequences(buffer)
        assert isinstance(packed, list)

    def test_finalize_pack_padding(self):
        from training.pretrain.dataset import QingxiaotuanPretrainDataset
        dataset_instance = QingxiaotuanPretrainDataset.__new__(QingxiaotuanPretrainDataset)
        dataset_instance.max_length = 128
        dataset_instance.tokenizer = None
        ids_list = [torch.randint(0, 1000, (50,))]
        labels_list = [torch.randint(0, 1000, (50,))]
        seg_ids_list = [torch.zeros(50, dtype=torch.long)]
        packed = dataset_instance._finalize_pack(ids_list, labels_list, seg_ids_list)
        assert packed["input_ids"].shape == (128,)
        assert packed["labels"].shape == (128,)
        assert "segment_ids" in packed
        assert "loss_weights" in packed
        assert "seg_mask" in packed


class TestQingxiaotuanSFTDataset:

    def test_sft_dataset_creation(self, sample_instruction_path):
        from training.sft.train_sft import SFTDataset
        from qingxiaotuan import QingxiaotuanTokenizer
        try:
            tokenizer = QingxiaotuanTokenizer.from_pretrained("qingxiaotuan")
        except Exception:
            pytest.skip("Tokenizer not available, skipping")
        dataset = SFTDataset(
            data_path=sample_instruction_path,
            tokenizer=tokenizer,
            max_length=512,
            pack_sequences=False,
        )
        assert len(dataset) > 0

    def test_sft_dataset_get_item(self, sample_instruction_path):
        from training.sft.train_sft import SFTDataset
        from qingxiaotuan import QingxiaotuanTokenizer
        try:
            tokenizer = QingxiaotuanTokenizer.from_pretrained("qingxiaotuan")
        except Exception:
            pytest.skip("Tokenizer not available, skipping")
        dataset = SFTDataset(
            data_path=sample_instruction_path,
            tokenizer=tokenizer,
            max_length=512,
            pack_sequences=False,
        )
        item = dataset[0]
        assert "input_ids" in item
        assert "labels" in item
        assert "attention_mask" in item

    def test_sft_dataset_conversations(self, sample_conversation_path):
        from training.sft.train_sft import SFTDataset
        from qingxiaotuan import QingxiaotuanTokenizer
        try:
            tokenizer = QingxiaotuanTokenizer.from_pretrained("qingxiaotuan")
        except Exception:
            pytest.skip("Tokenizer not available, skipping")
        dataset = SFTDataset(
            data_path=sample_conversation_path,
            tokenizer=tokenizer,
            max_length=512,
            pack_sequences=False,
        )
        assert len(dataset) > 0

    def test_sft_dataset_packing(self, sample_instruction_path):
        from training.sft.train_sft import SFTDataset
        from qingxiaotuan import QingxiaotuanTokenizer
        try:
            tokenizer = QingxiaotuanTokenizer.from_pretrained("qingxiaotuan")
        except Exception:
            pytest.skip("Tokenizer not available, skipping")
        dataset = SFTDataset(
            data_path=sample_instruction_path,
            tokenizer=tokenizer,
            max_length=512,
            pack_sequences=True,
        )
        assert dataset.packed_indices is not None
        assert len(dataset) > 0

    def test_sft_dataset_code_data(self, sample_code_path):
        from training.sft.train_sft import SFTDataset
        from qingxiaotuan import QingxiaotuanTokenizer
        try:
            tokenizer = QingxiaotuanTokenizer.from_pretrained("qingxiaotuan")
        except Exception:
            pytest.skip("Tokenizer not available, skipping")
        dataset = SFTDataset(
            data_path=sample_code_path,
            tokenizer=tokenizer,
            max_length=512,
            pack_sequences=False,
        )
        assert len(dataset) > 0

    def test_sft_dataset_chatml_format(self):
        from training.sft.train_sft import SFTDataset
        from qingxiaotuan import QingxiaotuanTokenizer
        try:
            tokenizer = QingxiaotuanTokenizer.from_pretrained("qingxiaotuan")
        except Exception:
            pytest.skip("Tokenizer not available, skipping")
        ds = SFTDataset.__new__(SFTDataset)
        ds.tokenizer = tokenizer
        messages = [
            {"role": "system", "content": "你是青小团"},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，我是青小团"},
        ]
        text = ds._apply_chatml(messages)
        assert "<|im_start|>system" in text
        assert "<|im_start|>user" in text
        assert "<|im_start|>assistant" in text
        assert "你是青小团" in text


class TestQingxiaotuanLossMasking:

    def test_loss_masking_assistant_only(self, sample_instruction_path):
        from training.sft.train_sft import SFTDataset
        from qingxiaotuan import QingxiaotuanTokenizer
        try:
            tokenizer = QingxiaotuanTokenizer.from_pretrained("qingxiaotuan")
        except Exception:
            pytest.skip("Tokenizer not available, skipping")
        dataset = SFTDataset(
            data_path=sample_instruction_path,
            tokenizer=tokenizer,
            max_length=512,
            pack_sequences=False,
        )
        item = dataset[0]
        labels = item["labels"]
        neg_count = (labels == -100).sum().item()
        assert neg_count > 0

    def test_loss_masking_labels_set(self, sample_instruction_path):
        from training.sft.train_sft import SFTDataset
        from qingxiaotuan import QingxiaotuanTokenizer
        try:
            tokenizer = QingxiaotuanTokenizer.from_pretrained("qingxiaotuan")
        except Exception:
            pytest.skip("Tokenizer not available, skipping")
        dataset = SFTDataset(
            data_path=sample_instruction_path,
            tokenizer=tokenizer,
            max_length=512,
            pack_sequences=False,
        )
        item = dataset[0]
        assert (item["labels"] != -100).sum().item() > 0

    def test_loss_masking_conversations(self, sample_conversation_path):
        from training.sft.train_sft import SFTDataset
        from qingxiaotuan import QingxiaotuanTokenizer
        try:
            tokenizer = QingxiaotuanTokenizer.from_pretrained("qingxiaotuan")
        except Exception:
            pytest.skip("Tokenizer not available, skipping")
        dataset = SFTDataset(
            data_path=sample_conversation_path,
            tokenizer=tokenizer,
            max_length=512,
            pack_sequences=False,
        )
        item = dataset[0]
        assert (item["labels"] == -100).sum().item() > 0

    def test_data_collator_sft(self, sample_instruction_path):
        from training.sft.train_sft import DataCollatorForSFT
        from qingxiaotuan import QingxiaotuanTokenizer
        try:
            tokenizer = QingxiaotuanTokenizer.from_pretrained("qingxiaotuan")
        except Exception:
            pytest.skip("Tokenizer not available, skipping")
        collator = DataCollatorForSFT(tokenizer=tokenizer, max_length=512, pack_sequences=False)
        from training.sft.train_sft import SFTDataset
        dataset = SFTDataset(sample_instruction_path, tokenizer, max_length=512, pack_sequences=False)
        features = [dataset[i] for i in range(min(2, len(dataset)))]
        if len(features) == 2:
            batch = collator(features)
            assert "input_ids" in batch


class TestQingxiaotuanSegmentedLoss:

    def test_segmented_loss_computation(self, small_qingxiaotuan_config):
        model = QingxiaotuanForCausalLM(small_qingxiaotuan_config)
        logits = torch.randn(1, 1024, 32000)
        labels = torch.randint(0, 32000, (1, 1024))
        loss = model._compute_segmented_loss(logits, labels)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_segmented_loss_chunk_size(self, small_qingxiaotuan_config):
        model = QingxiaotuanForCausalLM(small_qingxiaotuan_config)
        assert model.segmented_loss_chunk_size == 512

    def test_segmented_loss_with_masked_labels(self, small_qingxiaotuan_config):
        model = QingxiaotuanForCausalLM(small_qingxiaotuan_config)
        logits = torch.randn(1, 512, 32000)
        labels = torch.full((1, 512), -100, dtype=torch.long)
        labels[0, 256:] = torch.randint(0, 32000, (256,))
        loss = model._compute_segmented_loss(logits, labels)
        assert loss.ndim == 0


class TestQingxiaotuanDataCollator:

    def test_data_collator_packed_mode(self, sample_instruction_path):
        from training.sft.train_sft import DataCollatorForSFT
        from qingxiaotuan import QingxiaotuanTokenizer
        try:
            tokenizer = QingxiaotuanTokenizer.from_pretrained("qingxiaotuan")
        except Exception:
            pytest.skip("Tokenizer not available, skipping")
        collator = DataCollatorForSFT(tokenizer=tokenizer, max_length=512, pack_sequences=True)
        features = [
            {"input_ids": torch.randint(0, 1000, (512,)), "labels": torch.randint(0, 1000, (512,)),
             "attention_mask": torch.ones(512, dtype=torch.long), "loss_weights": torch.ones(512)},
        ]
        batch = collator(features)
        assert batch["input_ids"].shape == (1, 512)

    def test_data_collator_padded_mode(self, sample_instruction_path):
        from training.sft.train_sft import DataCollatorForSFT
        from qingxiaotuan import QingxiaotuanTokenizer
        try:
            tokenizer = QingxiaotuanTokenizer.from_pretrained("qingxiaotuan")
        except Exception:
            pytest.skip("Tokenizer not available, skipping")
        collator = DataCollatorForSFT(tokenizer=tokenizer, max_length=512, pack_sequences=False)
        features = [
            {"input_ids": torch.randint(0, 1000, (50,)), "labels": torch.randint(0, 1000, (50,)),
             "attention_mask": torch.ones(50, dtype=torch.long)},
            {"input_ids": torch.randint(0, 1000, (80,)), "labels": torch.randint(0, 1000, (80,)),
             "attention_mask": torch.ones(80, dtype=torch.long)},
        ]
        batch = collator(features)
        assert batch["input_ids"].shape[0] == 2


class TestQingxiaotuanCheckpointSaveLoad:

    def test_save_and_load_checkpoint(self, small_qingxiaotuan_model, dummy_inputs, temp_dir):
        model = small_qingxiaotuan_model
        save_path = os.path.join(temp_dir, "ckpt.pt")
        torch.save({"model_state_dict": model.state_dict()}, save_path)
        assert os.path.exists(save_path)
        loaded = torch.load(save_path, weights_only=False)
        assert "model_state_dict" in loaded

    def test_resume_from_checkpoint(self, small_qingxiaotuan_model, temp_dir):
        model = small_qingxiaotuan_model
        path = os.path.join(temp_dir, "resume.pt")
        torch.save({"model_state_dict": model.state_dict(), "step": 100}, path)
        checkpoint = torch.load(path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        assert checkpoint["step"] == 100

    def test_checkpoint_with_optimizer(self, small_qingxiaotuan_model, temp_dir):
        model = small_qingxiaotuan_model
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        path = os.path.join(temp_dir, "opt_ckpt.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, path)
        assert os.path.getsize(path) > 1000


class TestQingxiaotuanSFTArguments:

    def test_sft_arguments_defaults(self):
        from training.sft.train_sft import SFTArguments
        args = SFTArguments()
        assert args.max_length == 524288
        assert args.num_epochs == 3
        assert args.learning_rate == 1e-5
        assert args.per_device_train_batch_size == 1
        assert args.gradient_accumulation_steps == 12
        assert args.flash_attention is True
        assert args.pack_sequences is True
        assert args.gradient_checkpointing is True

    def test_sft_arguments_custom(self):
        from training.sft.train_sft import SFTArguments
        args = SFTArguments(
            model_name_or_path="qingxiaotuan",
            data_path="./data.jsonl",
            output_dir="./output",
            max_length=131072,
            num_epochs=5,
            learning_rate=5e-6,
        )
        assert args.max_length == 131072
        assert args.num_epochs == 5
        assert args.learning_rate == 5e-6


class TestQingxiaotuanTrainer:

    def test_training_args_creation(self, temp_dir):
        from transformers import TrainingArguments
        args = TrainingArguments(
            output_dir=temp_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            logging_steps=10,
            save_steps=50,
            bf16=False,
            remove_unused_columns=False,
        )
        assert args.output_dir == temp_dir
        assert args.learning_rate == 1e-4

    def test_trainer_with_model(self, small_qingxiaotuan_model, temp_dir):
        from transformers import Trainer, TrainingArguments
        args = TrainingArguments(
            output_dir=temp_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
        )
        trainer = Trainer(
            model=small_qingxiaotuan_model,
            args=args,
            train_dataset=None,
        )
        assert trainer is not None


class TestQingxiaotuanDeepSpeedConfig:

    def test_deepspeed_config_creation(self):
        from training.pretrain.train import build_deepspeed_config
        import argparse
        args = argparse.Namespace(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=12,
            learning_rate=3e-4,
            weight_decay=0.1,
            warmup_steps=2000,
            max_train_steps=100000,
            logging_steps=10,
            zero_stage=3,
            flash_attention=True,
            sequence_parallel=False,
        )
        config = build_deepspeed_config(args, world_size=8)
        assert config["train_micro_batch_size_per_gpu"] == 1
        assert config["gradient_accumulation_steps"] == 12
        assert config["zero_optimization"]["stage"] == 3
        assert config["bf16"]["enabled"] is True

    def test_deepspeed_zero_stage_variants(self):
        from training.pretrain.train import build_deepspeed_config
        import argparse
        args = argparse.ArgumentParser().parse_args([])
        args.per_device_train_batch_size = 1
        args.gradient_accumulation_steps = 4
        args.learning_rate = 3e-4
        args.weight_decay = 0.1
        args.warmup_steps = 1000
        args.max_train_steps = 50000
        args.logging_steps = 10
        args.zero_stage = 2
        args.flash_attention = True
        args.sequence_parallel = False
        config = build_deepspeed_config(args, world_size=4)
        assert config["zero_optimization"]["stage"] == 2

    def test_calculate_tokens_per_step(self):
        from training.pretrain.train import calculate_tokens_per_step
        import argparse
        args = argparse.ArgumentParser().parse_args([])
        args.per_device_train_batch_size = 1
        args.gradient_accumulation_steps = 12
        args.max_length = 524288
        tokens = calculate_tokens_per_step(args, world_size=8)
        expected = 1 * 8 * 12 * 524288
        assert tokens == expected
