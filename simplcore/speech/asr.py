# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
语音识别模块 (Automatic Speech Recognition)
实现语音到文字的转换功能
"""
from __future__ import annotations

import abc
import wave
import io
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ASRResult:
    """语音识别结果"""
    
    text: str
    confidence: float = 0.0
    audio_duration: float = 0.0
    language: str = "zh"
    timestamp: datetime = field(default_factory=datetime.now)
    words: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'confidence': self.confidence,
            'audio_duration': self.audio_duration,
            'language': self.language,
            'timestamp': self.timestamp.isoformat(),
            'words': self.words,
        }
    
    def __repr__(self):
        return f"ASRResult(text={self.text[:30]}..., confidence={self.confidence:.2f})"


class SpeechRecognizer(abc.ABC):
    """语音识别器抽象基类"""
    
    @abc.abstractmethod
    def recognize(self, audio_data: bytes, **kwargs) -> ASRResult:
        """识别语音数据"""
        pass
    
    @abc.abstractmethod
    def recognize_file(self, file_path: str, **kwargs) -> ASRResult:
        """识别语音文件"""
        pass
    
    @abc.abstractmethod
    def get_name(self) -> str:
        """获取识别器名称"""
        pass
    
    def recognize_mic(self, duration: int = 5, **kwargs) -> ASRResult:
        """从麦克风识别语音"""
        try:
            import pyaudio
            import wave
            
            chunk = 1024
            format = pyaudio.paInt16
            channels = 1
            rate = 16000
            
            p = pyaudio.PyAudio()
            
            stream = p.open(
                format=format,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=chunk,
            )
            
            print(f"正在录音... ({duration}秒)")
            frames = []
            
            for _ in range(0, int(rate / chunk * duration)):
                data = stream.read(chunk)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            audio_data = b''.join(frames)
            return self.recognize(audio_data, sample_rate=rate, **kwargs)
        
        except ImportError:
            raise RuntimeError("请先安装pyaudio: pip install pyaudio")
        except Exception as e:
            raise RuntimeError(f"麦克风录音失败: {e}")


class WhisperRecognizer(SpeechRecognizer):
    """基于Whisper的语音识别器"""
    
    def __init__(self, model_name: str = "base", language: str = "zh"):
        self.model_name = model_name
        self.language = language
        self._model = None
    
    def _load_model(self):
        if self._model is None:
            try:
                import whisper
                self._model = whisper.load_model(self.model_name)
            except ImportError:
                raise RuntimeError("请先安装whisper: pip install openai-whisper")
    
    def recognize(self, audio_data: bytes, **kwargs) -> ASRResult:
        self._load_model()
        
        try:
            audio_file = io.BytesIO(audio_data)
            
            result = self._model.transcribe(
                audio_file,
                language=kwargs.get('language', self.language),
                **kwargs
            )
            
            words = []
            if 'words' in result:
                for word in result['words']:
                    words.append({
                        'word': word['word'],
                        'start': word['start'],
                        'end': word['end'],
                        'confidence': word.get('confidence', 0.0),
                    })
            
            return ASRResult(
                text=result['text'],
                confidence=result.get('confidence', 0.0),
                audio_duration=result.get('duration', 0.0),
                language=result.get('language', self.language),
                words=words,
            )
        
        except Exception as e:
            raise RuntimeError(f"Whisper识别失败: {e}")
    
    def recognize_file(self, file_path: str, **kwargs) -> ASRResult:
        self._load_model()
        
        try:
            result = self._model.transcribe(
                file_path,
                language=kwargs.get('language', self.language),
                **kwargs
            )
            
            words = []
            if 'words' in result:
                for word in result['words']:
                    words.append({
                        'word': word['word'],
                        'start': word['start'],
                        'end': word['end'],
                        'confidence': word.get('confidence', 0.0),
                    })
            
            return ASRResult(
                text=result['text'],
                confidence=result.get('confidence', 0.0),
                audio_duration=result.get('duration', 0.0),
                language=result.get('language', self.language),
                words=words,
            )
        
        except Exception as e:
            raise RuntimeError(f"Whisper文件识别失败: {e}")
    
    def get_name(self) -> str:
        return "whisper"


class PocketSphinxRecognizer(SpeechRecognizer):
    """基于PocketSphinx的语音识别器"""
    
    def __init__(self, language: str = "zh"):
        self.language = language
        self._model = None
    
    def _load_model(self):
        if self._model is None:
            try:
                from pocketsphinx import Pocketsphinx
                self._model = Pocketsphinx()
            except ImportError:
                raise RuntimeError("请先安装pocketsphinx: pip install pocketsphinx")
    
    def recognize(self, audio_data: bytes, **kwargs) -> ASRResult:
        self._load_model()
        
        try:
            audio_file = io.BytesIO(audio_data)
            
            with wave.open(audio_file, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
            
            result = self._model.decode_raw(frames)
            
            return ASRResult(
                text=result,
                confidence=0.5,
                audio_duration=0.0,
                language=self.language,
            )
        
        except Exception as e:
            raise RuntimeError(f"PocketSphinx识别失败: {e}")
    
    def recognize_file(self, file_path: str, **kwargs) -> ASRResult:
        self._load_model()
        
        try:
            result = self._model.decode(file_path)
            
            return ASRResult(
                text=result,
                confidence=0.5,
                audio_duration=0.0,
                language=self.language,
            )
        
        except Exception as e:
            raise RuntimeError(f"PocketSphinx文件识别失败: {e}")
    
    def get_name(self) -> str:
        return "pocketsphinx"


class DummyRecognizer(SpeechRecognizer):
    """虚拟语音识别器（用于测试）"""
    
    def __init__(self):
        self._responses = {
            'hello': '你好',
            'hi': '嗨',
            'how are you': '你好吗',
            'what is your name': '你叫什么名字',
        }
    
    def recognize(self, audio_data: bytes, **kwargs) -> ASRResult:
        sample_text = "这是一段测试语音识别结果"
        return ASRResult(
            text=sample_text,
            confidence=0.95,
            audio_duration=len(audio_data) / 16000 / 2,
            language=kwargs.get('language', 'zh'),
        )
    
    def recognize_file(self, file_path: str, **kwargs) -> ASRResult:
        return ASRResult(
            text="从文件识别的语音内容",
            confidence=0.90,
            audio_duration=5.0,
            language=kwargs.get('language', 'zh'),
        )
    
    def get_name(self) -> str:
        return "dummy"


def create_recognizer(provider: str = "dummy", **kwargs) -> SpeechRecognizer:
    """创建语音识别器"""
    providers = {
        'whisper': WhisperRecognizer,
        'pocketsphinx': PocketSphinxRecognizer,
        'dummy': DummyRecognizer,
    }
    
    provider_class = providers.get(provider.lower(), DummyRecognizer)
    return provider_class(**kwargs)
