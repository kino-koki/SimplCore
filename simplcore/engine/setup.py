# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore C++ Extension Setup
编译C++扩展模块，提供高性能张量操作
"""
from setuptools import setup, Extension
import numpy as np
import platform
import os

compile_args = [
    '-O3',
    '-std=c++11',
    '-march=native',
    '-mfma',
    '-mavx2',
]

if platform.system() == 'Windows':
    compile_args.extend([
        '/EHsc',
        '/O2',
        '/arch:AVX2',
    ])
elif platform.system() == 'Linux':
    compile_args.extend([
        '-fopenmp',
        '-ffast-math',
    ])
elif platform.system() == 'Darwin':
    compile_args.extend([
        '-fopenmp',
        '-ffast-math',
        '-mmacosx-version-min=10.15',
    ])

link_args = []
if platform.system() == 'Linux':
    link_args.append('-fopenmp')
elif platform.system() == 'Darwin':
    link_args.append('-fopenmp')

cSimplCore_module = Extension(
    'cSimplCore',
    sources=['cSimplCore.cpp'],
    include_dirs=[np.get_include()],
    libraries=[],
    library_dirs=[],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
    language='c++',
)

setup(
    name='cSimplCore',
    version='1.0.0',
    description='SimplCore C++ Extension Module - High-performance tensor operations',
    author='Lu\'an Qingnai Sciences Technology Co., Ltd.',
    author_email='support@qingnai.tech',
    ext_modules=[cSimplCore_module],
    install_requires=['numpy'],
    zip_safe=False,
)

