# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimpFFT - 青小团自研FFT实现
替代 torch.fft 的核心功能
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
from .tensor import SimpTensor, DeviceType


class SimpFFT:
    """青小团自研FFT模块"""
    
    @staticmethod
    def fft(x: SimpTensor, dim: int = -1, norm: Optional[str] = None) -> SimpTensor:
        """快速傅里叶变换"""
        result = np.fft.fft(x._data, axis=dim, norm=norm)
        return SimpTensor(result, device=x.device)
    
    @staticmethod
    def ifft(x: SimpTensor, dim: int = -1, norm: Optional[str] = None) -> SimpTensor:
        """逆快速傅里叶变换"""
        result = np.fft.ifft(x._data, axis=dim, norm=norm)
        return SimpTensor(result, device=x.device)
    
    @staticmethod
    def rfft(x: SimpTensor, dim: int = -1, norm: Optional[str] = None) -> SimpTensor:
        """实数快速傅里叶变换"""
        result = np.fft.rfft(x._data, axis=dim, norm=norm)
        return SimpTensor(result, device=x.device)
    
    @staticmethod
    def irfft(x: SimpTensor, dim: int = -1, norm: Optional[str] = None) -> SimpTensor:
        """逆实数快速傅里叶变换"""
        result = np.fft.irfft(x._data, axis=dim, norm=norm)
        return SimpTensor(result, device=x.device)
    
    @staticmethod
    def fft2(x: SimpTensor, norm: Optional[str] = None) -> SimpTensor:
        """二维快速傅里叶变换"""
        result = np.fft.fft2(x._data, norm=norm)
        return SimpTensor(result, device=x.device)
    
    @staticmethod
    def ifft2(x: SimpTensor, norm: Optional[str] = None) -> SimpTensor:
        """二维逆快速傅里叶变换"""
        result = np.fft.ifft2(x._data, norm=norm)
        return SimpTensor(result, device=x.device)
    
    @staticmethod
    def fftn(x: SimpTensor, dim: Optional[Tuple[int, ...]] = None, norm: Optional[str] = None) -> SimpTensor:
        """N维快速傅里叶变换"""
        result = np.fft.fftn(x._data, axes=dim, norm=norm)
        return SimpTensor(result, device=x.device)
    
    @staticmethod
    def ifftn(x: SimpTensor, dim: Optional[Tuple[int, ...]] = None, norm: Optional[str] = None) -> SimpTensor:
        """N维逆快速傅里叶变换"""
        result = np.fft.ifftn(x._data, axes=dim, norm=norm)
        return SimpTensor(result, device=x.device)
    
    @staticmethod
    def fftshift(x: SimpTensor, dim: Optional[Tuple[int, ...]] = None) -> SimpTensor:
        """将零频率分量移到中心"""
        result = np.fft.fftshift(x._data, axes=dim)
        return SimpTensor(result, device=x.device)
    
    @staticmethod
    def ifftshift(x: SimpTensor, dim: Optional[Tuple[int, ...]] = None) -> SimpTensor:
        """逆fftshift"""
        result = np.fft.ifftshift(x._data, axes=dim)
        return SimpTensor(result, device=x.device)
    
    @staticmethod
    def fftfreq(n: int, d: float = 1.0) -> SimpTensor:
        """FFT频率采样"""
        result = np.fft.fftfreq(n, d=d)
        return SimpTensor(result)
    
    @staticmethod
    def rfftfreq(n: int, d: float = 1.0) -> SimpTensor:
        """实数FFT频率采样"""
        result = np.fft.rfftfreq(n, d=d)
        return SimpTensor(result)


def fft(x: SimpTensor, dim: int = -1, norm: Optional[str] = None) -> SimpTensor:
    return SimpFFT.fft(x, dim, norm)


def ifft(x: SimpTensor, dim: int = -1, norm: Optional[str] = None) -> SimpTensor:
    return SimpFFT.ifft(x, dim, norm)


def rfft(x: SimpTensor, dim: int = -1, norm: Optional[str] = None) -> SimpTensor:
    return SimpFFT.rfft(x, dim, norm)


def irfft(x: SimpTensor, dim: int = -1, norm: Optional[str] = None) -> SimpTensor:
    return SimpFFT.irfft(x, dim, norm)


def fft2(x: SimpTensor, norm: Optional[str] = None) -> SimpTensor:
    return SimpFFT.fft2(x, norm)


def ifft2(x: SimpTensor, norm: Optional[str] = None) -> SimpTensor:
    return SimpFFT.ifft2(x, norm)


def fftn(x: SimpTensor, dim: Optional[Tuple[int, ...]] = None, norm: Optional[str] = None) -> SimpTensor:
    return SimpFFT.fftn(x, dim, norm)


def ifftn(x: SimpTensor, dim: Optional[Tuple[int, ...]] = None, norm: Optional[str] = None) -> SimpTensor:
    return SimpFFT.ifftn(x, dim, norm)


def fftshift(x: SimpTensor, dim: Optional[Tuple[int, ...]] = None) -> SimpTensor:
    return SimpFFT.fftshift(x, dim)


def ifftshift(x: SimpTensor, dim: Optional[Tuple[int, ...]] = None) -> SimpTensor:
    return SimpFFT.ifftshift(x, dim)


def fftfreq(n: int, d: float = 1.0) -> SimpTensor:
    return SimpFFT.fftfreq(n, d)


def rfftfreq(n: int, d: float = 1.0) -> SimpTensor:
    return SimpFFT.rfftfreq(n, d)


if __name__ == "__main__":
    print("=" * 60)
    print("SimpFFT 自检")
    print("=" * 60)
    
    from .tensor import randn
    
    x = randn(4, 8)
    
    print(f"Input shape: {x.shape}")
    
    X = fft(x)
    print(f"FFT shape: {X.shape}")
    
    x_recovered = ifft(X)
    print(f"IFFT shape: {x_recovered.shape}")
    
    error = np.abs(x._data - x_recovered._data.real).mean()
    print(f"Reconstruction error: {error:.6f}")
    
    X2 = fft2(randn(4, 4))
    print(f"2D FFT shape: {X2.shape}")
    
    freqs = fftfreq(8)
    print(f"FFT frequencies: {freqs._data}")
    
    print("\nSimpFFT 自检 PASSED")
