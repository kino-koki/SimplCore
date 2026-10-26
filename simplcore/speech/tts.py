# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
语音合成模块 (Text-to-Speech)
实现文字到语音的转换功能
"""
from __future__ import annotations

import abc
import io
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TTSResult:
    """语音合成结果"""
    
    audio_data: bytes
    text: str
    language: str = "zh"
    sample_rate: int = 22050
    audio_format: str = "wav"
    duration: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'language': self.language,
            'sample_rate': self.sample_rate,
            'audio_format': self.audio_format,
            'duration': self.duration,
            'timestamp': self.timestamp.isoformat(),
            'audio_length': len(self.audio_data),
        }
    
    def save_to_file(self, file_path: str):
        """保存语音到文件"""
        with open(file_path, 'wb') as f:
            f.write(self.audio_data)
    
    def play(self):
        """播放语音"""
        try:
            import pygame
            pygame.mixer.init()
            audio_file = io.BytesIO(self.audio_data)
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except ImportError:
            raise RuntimeError("请先安装pygame: pip install pygame")
        except Exception as e:
            raise RuntimeError(f"播放失败: {e}")
    
    def __repr__(self):
        return f"TTSResult(text={self.text[:30]}..., duration={self.duration:.2f}s)"


class SpeechSynthesizer(abc.ABC):
    """语音合成器抽象基类"""
    
    @abc.abstractmethod
    def synthesize(self, text: str, **kwargs) -> TTSResult:
        """合成语音"""
        pass
    
    @abc.abstractmethod
    def get_name(self) -> str:
        """获取合成器名称"""
        pass
    
    def synthesize_and_play(self, text: str, **kwargs):
        """合成并播放语音"""
        result = self.synthesize(text, **kwargs)
        result.play()
        return result


class EdgeTTSSynthesizer(SpeechSynthesizer):
    """基于Edge TTS的语音合成器"""
    
    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural", rate: str = "+0%", volume: str = "+0%"):
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self._voices = None
    
    def _load_voices(self):
        if self._voices is None:
            try:
                import edge_tts
                self._voices = edge_tts.list_voices()
            except ImportError:
                raise RuntimeError("请先安装edge-tts: pip install edge-tts")
    
    def synthesize(self, text: str, **kwargs) -> TTSResult:
        try:
            import edge_tts
            
            voice = kwargs.get('voice', self.voice)
            rate = kwargs.get('rate', self.rate)
            volume = kwargs.get('volume', self.volume)
            
            communicate = edge_tts.Communicate(text, voice, rate=rate, volume=volume)
            
            audio_bytes = io.BytesIO()
            asyncio.run(self._save_audio(communicate, audio_bytes))
            
            return TTSResult(
                audio_data=audio_bytes.getvalue(),
                text=text,
                language=voice[:5] if '-' in voice else 'zh',
                sample_rate=22050,
                audio_format="wav",
                duration=len(audio_bytes.getvalue()) / 22050 / 2,
            )
        
        except ImportError:
            raise RuntimeError("请先安装edge-tts: pip install edge-tts")
        except Exception as e:
            raise RuntimeError(f"Edge TTS合成失败: {e}")
    
    async def _save_audio(self, communicate, audio_bytes):
        async for chunk in communicate.stream():
            if chunk['type'] == 'audio':
                audio_bytes.write(chunk['data'])
    
    def get_voices(self) -> List[Dict[str, Any]]:
        """获取可用语音列表"""
        self._load_voices()
        return self._voices
    
    def get_name(self) -> str:
        return "edge-tts"


class GTTSSynthesizer(SpeechSynthesizer):
    """基于Google TTS的语音合成器"""
    
    def __init__(self, lang: str = "zh", slow: bool = False):
        self.lang = lang
        self.slow = slow
    
    def synthesize(self, text: str, **kwargs) -> TTSResult:
        try:
            from gtts import gTTS
            
            lang = kwargs.get('lang', self.lang)
            slow = kwargs.get('slow', self.slow)
            
            tts = gTTS(text=text, lang=lang, slow=slow)
            
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            
            return TTSResult(
                audio_data=audio_bytes.getvalue(),
                text=text,
                language=lang,
                sample_rate=22050,
                audio_format="mp3",
                duration=len(audio_bytes.getvalue()) / 22050 / 2,
            )
        
        except ImportError:
            raise RuntimeError("请先安装gTTS: pip install gTTS")
        except Exception as e:
            raise RuntimeError(f"Google TTS合成失败: {e}")
    
    def get_name(self) -> str:
        return "gtts"


class Pyttsx3Synthesizer(SpeechSynthesizer):
    """基于pyttsx3的语音合成器（离线）"""
    
    def __init__(self, lang: str = "zh", rate: int = 150, volume: float = 1.0):
        self.lang = lang
        self.rate = rate
        self.volume = volume
        self._engine = None
    
    def _init_engine(self):
        if self._engine is None:
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
                self._engine.setProperty('rate', self.rate)
                self._engine.setProperty('volume', self.volume)
            except ImportError:
                raise RuntimeError("请先安装pyttsx3: pip install pyttsx3")
    
    def synthesize(self, text: str, **kwargs) -> TTSResult:
        self._init_engine()
        
        try:
            rate = kwargs.get('rate', self.rate)
            volume = kwargs.get('volume', self.volume)
            
            self._engine.setProperty('rate', rate)
            self._engine.setProperty('volume', volume)
            
            audio_bytes = io.BytesIO()
            
            def save_to_buffer(text):
                self._engine.save_to_file(text, audio_bytes)
            
            self._engine.save_to_file(text, audio_bytes)
            self._engine.runAndWait()
            
            audio_bytes.seek(0)
            
            return TTSResult(
                audio_data=audio_bytes.getvalue(),
                text=text,
                language=self.lang,
                sample_rate=22050,
                audio_format="wav",
                duration=len(audio_bytes.getvalue()) / 22050 / 2,
            )
        
        except Exception as e:
            raise RuntimeError(f"pyttsx3合成失败: {e}")
    
    def get_name(self) -> str:
        return "pyttsx3"


class DummySynthesizer(SpeechSynthesizer):
    """虚拟语音合成器（用于测试）"""
    
    def __init__(self):
        pass
    
    def synthesize(self, text: str, **kwargs) -> TTSResult:
        dummy_audio = b'\x00' * 10000
        
        return TTSResult(
            audio_data=dummy_audio,
            text=text,
            language=kwargs.get('lang', 'zh'),
            sample_rate=22050,
            audio_format="wav",
            duration=0.5,
        )
    
    def get_name(self) -> str:
        return "dummy"


def create_synthesizer(provider: str = "dummy", **kwargs) -> SpeechSynthesizer:
    """创建语音合成器"""
    providers = {
        'edge-tts': EdgeTTSSynthesizer,
        'gtts': GTTSSynthesizer,
        'pyttsx3': Pyttsx3Synthesizer,
        'dummy': DummySynthesizer,
    }
    
    provider_class = providers.get(provider.lower(), DummySynthesizer)
    return provider_class(**kwargs)

import asyncio
