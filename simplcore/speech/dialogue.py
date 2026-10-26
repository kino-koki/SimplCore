# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
语音对话管理器模块 (Voice Dialogue Manager)
实现完整的语音交互能力，整合ASR和TTS功能
"""
from __future__ import annotations

import abc
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .asr import SpeechRecognizer, ASRResult, create_recognizer
from .tts import SpeechSynthesizer, TTSResult, create_synthesizer


@dataclass
class DialogueTurn:
    """对话轮次"""
    
    user_text: str
    user_audio: Optional[bytes] = None
    assistant_text: str = ""
    assistant_audio: Optional[bytes] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_text': self.user_text,
            'assistant_text': self.assistant_text,
            'timestamp': self.timestamp.isoformat(),
        }


class VoiceDialogueManager:
    """语音对话管理器"""
    
    def __init__(
        self,
        recognizer: Optional[SpeechRecognizer] = None,
        synthesizer: Optional[SpeechSynthesizer] = None,
        model=None,
        enable_voice_input: bool = True,
        enable_voice_output: bool = True,
        language: str = "zh",
    ):
        self.recognizer = recognizer or create_recognizer("dummy")
        self.synthesizer = synthesizer or create_synthesizer("dummy")
        self.model = model
        self.enable_voice_input = enable_voice_input
        self.enable_voice_output = enable_voice_output
        self.language = language
        self.dialogue_history: List[DialogueTurn] = []
        self.is_listening = False
    
    def recognize(self, audio_data: bytes = None, file_path: str = None) -> ASRResult:
        """识别语音"""
        if audio_data is not None:
            return self.recognizer.recognize(audio_data, language=self.language)
        elif file_path is not None:
            return self.recognizer.recognize_file(file_path, language=self.language)
        else:
            return self.recognizer.recognize_mic(language=self.language)
    
    def synthesize(self, text: str, **kwargs) -> TTSResult:
        """合成语音"""
        return self.synthesizer.synthesize(text, lang=self.language, **kwargs)
    
    def generate_response(self, text: str) -> str:
        """生成AI响应"""
        if self.model is not None:
            try:
                return self.model.generate(text)
            except Exception as e:
                print(f"模型生成失败: {e}")
        
        return f"收到您的消息: {text}"
    
    def process_turn(
        self,
        user_input: str = "",
        audio_data: bytes = None,
        file_path: str = None,
        auto_respond: bool = True,
    ) -> Dict[str, Any]:
        """处理一轮对话"""
        user_text = user_input
        
        if self.enable_voice_input and not user_input:
            if audio_data is not None:
                asr_result = self.recognizer.recognize(audio_data, language=self.language)
            elif file_path is not None:
                asr_result = self.recognizer.recognize_file(file_path, language=self.language)
            else:
                asr_result = self.recognizer.recognize_mic(language=self.language)
            user_text = asr_result.text
            print(f"识别结果: {user_text} (置信度: {asr_result.confidence:.2f})")
        
        assistant_text = ""
        assistant_audio = None
        
        if auto_respond:
            assistant_text = self.generate_response(user_text)
            
            if self.enable_voice_output:
                tts_result = self.synthesizer.synthesize(assistant_text, lang=self.language)
                assistant_audio = tts_result.audio_data
        
        turn = DialogueTurn(
            user_text=user_text,
            user_audio=audio_data,
            assistant_text=assistant_text,
            assistant_audio=assistant_audio,
        )
        self.dialogue_history.append(turn)
        
        return {
            'user_text': user_text,
            'assistant_text': assistant_text,
            'has_audio': assistant_audio is not None,
            'turn_count': len(self.dialogue_history),
            'timestamp': turn.timestamp.isoformat(),
        }
    
    def start_conversation(self, prompt: str = "您好，请说"):
        """开始语音对话"""
        print("=== 语音对话开始 ===")
        if self.enable_voice_output:
            tts_result = self.synthesizer.synthesize(prompt, lang=self.language)
            tts_result.play()
        
        self.is_listening = True
        while self.is_listening:
            try:
                result = self.process_turn()
                print(f"\n用户: {result['user_text']}")
                print(f"助手: {result['assistant_text']}")
                
                if result['user_text'] in ["退出", "结束", "再见", "bye", "exit"]:
                    self.is_listening = False
                    if self.enable_voice_output:
                        tts_result = self.synthesizer.synthesize("再见！", lang=self.language)
                        tts_result.play()
                
            except KeyboardInterrupt:
                self.is_listening = False
                print("\n对话已结束")
    
    def stop_conversation(self):
        """停止语音对话"""
        self.is_listening = False
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取对话历史"""
        return [turn.to_dict() for turn in self.dialogue_history]
    
    def clear_history(self):
        """清除对话历史"""
        self.dialogue_history = []
    
    def set_model(self, model):
        """设置AI模型"""
        self.model = model
    
    def set_recognizer(self, recognizer: SpeechRecognizer):
        """设置语音识别器"""
        self.recognizer = recognizer
    
    def set_synthesizer(self, synthesizer: SpeechSynthesizer):
        """设置语音合成器"""
        self.synthesizer = synthesizer


class VoiceAssistant:
    """语音助手"""
    
    def __init__(
        self,
        recognizer_provider: str = "dummy",
        synthesizer_provider: str = "dummy",
        model=None,
        language: str = "zh",
    ):
        self.recognizer = create_recognizer(recognizer_provider)
        self.synthesizer = create_synthesizer(synthesizer_provider)
        self.dialogue_manager = VoiceDialogueManager(
            recognizer=self.recognizer,
            synthesizer=self.synthesizer,
            model=model,
            language=language,
        )
    
    def ask(self, prompt: str, auto_speak: bool = True) -> Dict[str, Any]:
        """提问并获取回答"""
        if auto_speak:
            tts_result = self.synthesizer.synthesize(prompt, lang=self.dialogue_manager.language)
            tts_result.play()
        
        result = self.dialogue_manager.process_turn()
        return result
    
    def say(self, text: str):
        """说话"""
        tts_result = self.synthesizer.synthesize(text, lang=self.dialogue_manager.language)
        tts_result.play()
    
    def listen(self) -> str:
        """监听并识别语音"""
        result = self.dialogue_manager.recognize()
        return result.text
    
    def chat(self):
        """开始聊天"""
        self.dialogue_manager.start_conversation()
    
    def stop(self):
        """停止聊天"""
        self.dialogue_manager.stop_conversation()
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取对话历史"""
        return self.dialogue_manager.get_history()


def create_voice_assistant(
    recognizer_provider: str = "dummy",
    synthesizer_provider: str = "dummy",
    model=None,
    language: str = "zh",
) -> VoiceAssistant:
    """创建语音助手"""
    return VoiceAssistant(
        recognizer_provider=recognizer_provider,
        synthesizer_provider=synthesizer_provider,
        model=model,
        language=language,
    )
