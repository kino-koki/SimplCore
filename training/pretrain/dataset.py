import torch
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset
from itertools import islice
import random


class QingxiaotuanPretrainDataset(IterableDataset):
    """青小团预训练数据集，支持sequence packing和512K长序列

    将短序列动态打包(pack)成max_length长度的长序列，
    显著减少padding浪费，提升训练效率。
    """

    def __init__(self, data_files, tokenizer, max_length=524288, pack_sequences=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pack_sequences = pack_sequences
        self.dataset = load_dataset("json", data_files=data_files, streaming=True, split="train")

    def __iter__(self):
        buffer = []
        for raw_sample in self.dataset:
            text = raw_sample.get("text") or raw_sample.get("content", "")
            tokens = self._tokenize(text)
            if tokens is None:
                continue
            buffer.append(tokens)
            if self.pack_sequences:
                packed = self._pack_sequences(buffer)
                for item in packed:
                    yield item
            else:
                for t in buffer:
                    yield t
                buffer.clear()

    def _tokenize(self, text):
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(input_ids) == 0:
            return None
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}

    def _pack_sequences(self, buffer):
        """将buffer中的短序列打包成max_length长度的长序列

        返回打包后的样本列表，未被填满buffer的剩余序列会留在buffer中继续参与下次打包。
        """
        packed_samples = []
        current_ids = []
        current_labels = []
        current_segment_ids = []
        segment_idx = 0

        i = 0
        while i < len(buffer):
            item = buffer[i]
            ids = item["input_ids"]
            labs = item["labels"]
            seq_len = ids.size(0)

            if len(current_ids) + seq_len > self.max_length:
                if len(current_ids) > 0:
                    packed_samples.append(self._finalize_pack(
                        current_ids, current_labels, current_segment_ids
                    ))
                    current_ids = []
                    current_labels = []
                    current_segment_ids = []
                    segment_idx = 0
                else:
                    item_ids = ids[:self.max_length].clone()
                    item_labs = labs[:self.max_length].clone()
                    packed_samples.append({
                        "input_ids": item_ids,
                        "labels": item_labs,
                        "attention_mask": torch.ones(self.max_length, dtype=torch.long),
                        "segment_ids": torch.zeros(self.max_length, dtype=torch.long),
                        "loss_weights": torch.ones(self.max_length, dtype=torch.float),
                    })
                    i += 1
                    continue

            current_ids.append(ids)
            current_labels.append(labs)
            current_segment_ids.append(torch.full((seq_len,), segment_idx, dtype=torch.long))
            segment_idx += 1
            i += 1

        remaining = []
        if len(current_ids) > 0:
            remaining.append({
                "input_ids": torch.cat(current_ids),
                "labels": torch.cat(current_labels),
                "segment_ids": torch.cat(current_segment_ids),
            })
        for item in remaining:
            buffer.clear()
            buffer.append(item)
        if not remaining:
            buffer.clear()

        return packed_samples

    def _finalize_pack(self, ids_list, labels_list, segment_ids_list):
        input_ids = torch.cat(ids_list)
        labels = torch.cat(labels_list)
        segment_ids = torch.cat(segment_ids_list)
        seq_len = input_ids.size(0)

        if seq_len < self.max_length:
            pad_len = self.max_length - seq_len
            input_ids = torch.cat([input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
            segment_ids = torch.cat([segment_ids, torch.full((pad_len,), -1, dtype=torch.long)])

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        seg_mask = (segment_ids.unsqueeze(0) != segment_ids.unsqueeze(1)).long()
        seg_mask = seg_mask * attention_mask.unsqueeze(1)
        seg_mask = seg_mask * attention_mask.unsqueeze(0)

        unique_segments = segment_ids[segment_ids >= 0].unique()
        loss_weights = torch.zeros(self.max_length, dtype=torch.float)
        if len(unique_segments) > 0:
            weight_per_segment = 1.0 / len(unique_segments)
            for sid in unique_segments:
                mask = segment_ids == sid
                seg_len = mask.sum().float()
                if seg_len > 0:
                    loss_weights[mask] = weight_per_segment / seg_len

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "segment_ids": segment_ids,
            "loss_weights": loss_weights,
            "seg_mask": seg_mask,
        }

    def __len__(self):
        return 1000000


def collate_fn_for_packing(batch):
    """Sequence packing场景下的collate函数

    每条样本已经是packed后的完整序列，直接stack成batch。
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    loss_weights = torch.stack([item["loss_weights"] for item in batch])

    result = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "loss_weights": loss_weights,
    }

    if "seg_mask" in batch[0]:
        seg_mask = torch.stack([item["seg_mask"] for item in batch])
        result["seg_mask"] = seg_mask

    return result
