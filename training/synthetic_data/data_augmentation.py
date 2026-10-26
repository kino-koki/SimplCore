"""
青小团 合成数据生成引擎 - 数据增强模块
青奈科技 (QingNai Technology)

提供多种数据增强技术：回译增强、同义替换、
随机插入/删除/替换、句子打乱、噪声注入、
长度变化和风格迁移。
"""

import copy
import json
import random
import re
from typing import Any, Optional

from base_generator import BaseGenerator, DataRecord, GenerationContext

_SYNONYM_MAP_ZH = {
    "好": ["优秀", "出色", "良好", "优异", "棒"],
    "大": ["巨大", "庞大", "宏大", "广大", "重大"],
    "小": ["微小", "细小", "小巧", "细小", "轻微"],
    "快": ["迅速", "快速", "快捷", "飞速", "迅猛"],
    "慢": ["缓慢", "迟缓", "慢慢", "徐徐", "迟钝"],
    "美丽": ["漂亮", "好看", "迷人", "惊艳", "绚烂"],
    "重要": ["关键", "核心", "重大", "紧要", "必不可少"],
    "理解": ["明白", "懂得", "领会", "领悟", "掌握"],
    "帮助": ["协助", "辅助", "支持", "援助", "帮忙"],
    "使用": ["利用", "运用", "采用", "应用", "借助"],
    "但是": ["然而", "可是", "不过", "但", "却"],
    "因为": ["由于", "鉴于", "出于", "基于"],
    "所以": ["因此", "因而", "从而", "于是", "故而"],
    "非常": ["十分", "特别", "极其", "相当", "异常"],
    "很多": ["众多", "大量", "许多", "大批", "丰富"],
}

_STOP_WORDS_ZH = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
                   "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
                   "你", "会", "着", "没有", "看", "好", "自己", "这", "他"}

_PUNCTUATION = set("，。、；：？！""''【】（）《》—…·,.;:!?\"'-")


class DataAugmentation:
    def __init__(self, config: dict[str, Any], seed: int = 42):
        self.config = config
        self._rng = random.Random(seed)
        self.aug_cfg = config.get("augmentation", {})
        self.technique_weights = self.aug_cfg.get("techniques", {})
        self.augmented_ratio = self.aug_cfg.get("augmented_ratio", 0.3)
        self.preserve_semantics = self.aug_cfg.get("preserve_semantics", True)

    def _pick_technique(self) -> Optional[str]:
        if self.technique_weights:
            return self._pick_weighted(self.technique_weights)
        return None

    def _pick_weighted(self, options: dict[str, float]) -> str:
        items = list(options.items())
        keys = [k for k, _ in items]
        weights = [v for _, v in items]
        return self._rng.choices(keys, weights=weights, k=1)[0]

    def synonym_replacement(self, text: str, replace_prob: float = 0.15) -> str:
        words = list(text)
        replaced = list(text)
        for i, char in enumerate(words):
            if char in _SYNONYM_MAP_ZH and self._rng.random() < replace_prob:
                synonyms = _SYNONYM_MAP_ZH[char]
                replaced[i] = self._rng.choice(synonyms)
        return ''.join(replaced)

    def random_insertion(self, text: str, insert_prob: float = 0.1) -> str:
        if len(text) < 10:
            return text
        words = list(text)
        num_insert = max(1, int(len(words) * insert_prob))
        result = list(text)
        for _ in range(num_insert):
            pos = self._rng.randint(0, len(result))
            all_synonyms = []
            for syn_list in _SYNONYM_MAP_ZH.values():
                all_synonyms.extend(syn_list)
            insert_word = self._rng.choice(all_synonyms)
            result.insert(pos, insert_word)
        return ''.join(result)

    def random_deletion(self, text: str, delete_prob: float = 0.08) -> str:
        if len(text) < 10:
            return text
        chars = list(text)
        result = []
        for ch in chars:
            if self._rng.random() > delete_prob:
                result.append(ch)
        if len(result) < len(text) * 0.5:
            result = list(text[:max(3, len(text)//2)])
        return ''.join(result)

    def sentence_shuffle(self, text: str) -> str:
        sentences = re.split(r'([。！？\n!?])', text)
        grouped = []
        for i in range(0, len(sentences) - 1, 2):
            grouped.append(sentences[i] + (sentences[i+1] if i+1 < len(sentences) else ''))
        if len(sentences) % 2 == 1:
            grouped.append(sentences[-1])
        if len(grouped) <= 1:
            return text
        self._rng.shuffle(grouped)
        return ''.join(grouped)

    def noise_injection(self, text: str, noise_prob: float = 0.03) -> str:
        chars = list(text)
        noise_chars = " ，。、；：？！""''【】（）·~@#$%^&*"
        result = []
        for ch in chars:
            if self._rng.random() < noise_prob:
                noise_type = self._rng.random()
                if noise_type < 0.33:
                    result.append(self._rng.choice(noise_chars))
                    result.append(ch)
                elif noise_type < 0.66:
                    pass
                else:
                    result.append(self._rng.choice(noise_chars))
            else:
                result.append(ch)
        return ''.join(result)

    def length_variation(self, text: str, target_ratio: float = 0.7) -> str:
        if len(text) < 20:
            return text
        if target_ratio < 1.0:
            target_len = max(10, int(len(text) * target_ratio))
            return text[:target_len]
        else:
            repeat_text = text * int(target_ratio)
            return repeat_text[:int(len(text) * target_ratio)]

    def style_transfer(self, text: str, target_style: str = "formal") -> str:
        style_markers = {
            "formal": {
                "prefix": "根据以上分析，",
                "suffix": "综上所述，这是一个值得关注的问题。",
                "replacements": {"搞": "进行", "弄": "处理", "干": "执行"},
            },
            "friendly": {
                "prefix": "嗨！",
                "suffix": "希望这对你有帮助哦~",
                "replacements": {"进行": "搞", "处理": "弄", "但是": "不过呢"},
            },
            "concise": {
                "prefix": "",
                "suffix": "",
                "replacements": {},
            },
        }
        markers = style_markers.get(target_style, style_markers["formal"])
        result = text
        for old, new in markers["replacements"].items():
            result = result.replace(old, new)
        if markers["prefix"] and not result.startswith(markers["prefix"]):
            result = markers["prefix"] + result[0].lower() + result[1:]
        if markers["suffix"]:
            result = result.rstrip('。！？') + '。' + markers["suffix"]
        return result

    def round_trip_augment(self, text: str) -> str:
        if not text.strip():
            return text
        operations = [
            lambda t: self.synonym_replacement(t, 0.1),
            lambda t: self.random_insertion(t, 0.05),
            lambda t: t.replace('。', '。\n').replace('\n\n', '\n'),
        ]
        result = text
        for op in operations:
            if self._rng.random() < 0.5:
                result = op(result)
        return result

    def augment_text(self, text: str, technique: Optional[str] = None) -> str:
        if technique is None:
            technique = self._pick_technique()
        if technique is None:
            return text
        augmenters = {
            "synonym_replacement": lambda: self.synonym_replacement(text),
            "random_insertion": lambda: self.random_insertion(text),
            "random_deletion": lambda: self.random_deletion(text),
            "sentence_shuffle": lambda: self.sentence_shuffle(text),
            "noise_injection": lambda: self.noise_injection(text),
            "length_variation": lambda: self.length_variation(text, self._rng.uniform(0.5, 1.5)),
            "style_transfer": lambda: self.style_transfer(
                text, self._rng.choice(["formal", "friendly", "concise"])
            ),
            "round_trip": lambda: self.round_trip_augment(text),
        }
        augmenter = augmenters.get(technique)
        if augmenter is None:
            return text
        return augmenter()

    def augment_record(self, record: DataRecord) -> list[DataRecord]:
        augmented_records = []
        content = record.content
        for key, value in content.items():
            if isinstance(value, str) and len(value) > 20:
                aug_value = self.augment_text(value)
                if aug_value != value:
                    aug_content = copy.deepcopy(content)
                    aug_content[key] = aug_value
                    aug_metadata = dict(record.metadata)
                    aug_metadata["augmented"] = True
                    aug_metadata["original_field"] = key
                    aug_record = DataRecord(
                        id=self._rng.choice(["aug_" + record.id]),
                        source=record.source + "_augmented",
                        data_type=record.data_type,
                        content=aug_content,
                        metadata=aug_metadata,
                    )
                    augmented_records.append(aug_record)
                    if len(augmented_records) >= 3:
                        break
        return augmented_records

    def augment_dataset(self, records: list[DataRecord]) -> list[DataRecord]:
        augmented_records = list(records)
        target_aug_count = int(len(records) * self.augmented_ratio)
        candidate_indices = [i for i, r in enumerate(records) if r.content]
        self._rng.shuffle(candidate_indices)
        aug_count = 0
        for idx in candidate_indices:
            if aug_count >= target_aug_count:
                break
            augmented = self.augment_record(records[idx])
            augmented_records.extend(augmented)
            aug_count += len(augmented)
        return augmented_records

    def get_stats(self) -> dict[str, Any]:
        return {
            "techniques": list(self.technique_weights.keys()) if self.technique_weights else [],
            "augmented_ratio": self.augmented_ratio,
            "preserve_semantics": self.preserve_semantics,
        }
