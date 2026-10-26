# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
语音处理模块 (Speech Processing Module)
实现语音识别（ASR）和语音合成（TTS）功能

核心功能:
- 语音识别: 将语音转换为文字
- 语音合成: 将文字转换为语音
- 语音对话: 完整的语音交互能力
- 多语言支持: 支持中文、英文等多种语言
"""
from __future__ import annotations

from .asr import SpeechRecognizer, ASRResult
from .tts import SpeechSynthesizer, TTSResult
from .dialogue import VoiceDialogueManager

__all__ = [
    'SpeechRecognizer',
    'ASRResult',
    'SpeechSynthesizer',
    'TTSResult',
    'VoiceDialogueManager',
]
