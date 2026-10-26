# Errors

Command failures and integration errors.

---

## [ERR-20260729-003] uppercase_simplcore_import_alias

**Logged**: 2026-07-29T00:00:00+08:00
**Priority**: medium
**Status**: resolved
**Area**: backend

### Summary
历史测试和脚本使用 `SimplCore.*` 大写包路径，但当前真实包目录是 `simplcore`，导致大小写敏感的 Python 导入失败。

### Error
```text
ModuleNotFoundError: No module named 'SimplCore'
```

### Context
- Command attempted: import from `SimplCore.architecture.config` and `SimplCore.architecture.SimplCore_block`.
- Affected old entrypoints: `test_world_model.py`, older architecture examples, world-model scripts.

### Suggested Fix
保留真实包名 `simplcore`，同时提供 `SimplCore.py` 兼容 shim，并在 `simplcore/architecture/__init__.py` 注册 `SimplCore.architecture.SimplCore_block` 与 `SimplCore.architecture.SimplCore_model` 别名。

### Metadata
- Reproducible: yes
- Related Files: SimplCore.py, simplcore/architecture/__init__.py

### Resolution
- **Resolved**: 2026-07-29T00:00:00+08:00
- **Notes**: 已新增大写包名 shim 和旧模块名 sys.modules 映射；旧导入路径验证通过。

---

## [ERR-20260729-002] uv_python_externally_managed

**Logged**: 2026-07-29T00:00:00+08:00
**Priority**: medium
**Status**: pending
**Area**: infra

### Summary
uv 管理的 Python 3.11.15 环境拒绝直接 `pip install`，需要创建项目虚拟环境后安装依赖。

### Error
```text
error: externally-managed-environment
This Python installation is managed by uv and should not be modified.
```

### Context
- 目标解释器路径为 `C:\Users\28726\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none\python.exe`。
- 用户要求使用 GPU 训练，不允许回退 CPU。
- `torch` 和 `torch_directml` 不在基础 3.11 环境中。

### Suggested Fix
使用 `uv venv` 创建项目虚拟环境，再用 `uv pip install --python .venv311\Scripts\python.exe torch-directml pyarrow` 安装 GPU 训练依赖。

### Metadata
- Reproducible: yes
- Related Files: simplcore/mix/training/train_150m_gpu.py

---

## [ERR-20260729-001] python_interpreter_target_mismatch

**Logged**: 2026-07-29T00:00:00+08:00
**Priority**: high
**Status**: pending
**Area**: infra

### Summary
默认命令行 `python` 指向 Python 3.10.11，且 `py -3.11` 不可用；项目后续目标已改为 Python 3.11。

### Error
```text
python -c ... failed with missing encodings from the embedded tool interpreter.
py -3.11 failed because py launcher is not recognized.
where.exe python reports Python 3.10.11.
```

### Context
- 用户明确要求以后使用 Python 3.11 开发。
- `python -m pytest ...` 当前仍可运行部分测试，但 7B 堆叠训练准备应把 Python 3.11 作为环境门禁。

### Suggested Fix
安装或配置可显式调用的 Python 3.11 解释器，并在 7B 训练准备脚本中检查 `sys.version_info[:2] == (3, 11)`。

### Metadata
- Reproducible: yes
- Related Files: simplcore/mix/runtime.py, simplcore/mix/TRAINING_HANDOFF.md

---

## [ERR-20260727-001] huggingface_hub_certificate_verify

**Logged**: 2026-07-27T00:00:00+08:00
**Priority**: medium
**Status**: pending
**Area**: infra

### Summary
Hugging Face dataset metadata query failed because Python certificate verification could not find a local issuer certificate.

### Error
```text
[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate
```

### Context
- Command attempted: huggingface_hub.HfApi().dataset_info(..., files_metadata=True)
- Repos: opencsg/chinese-fineweb-edu and opencsg/Fineweb-Edu-Chinese-V2.2
- WebFetch could access source documentation, so this is a local Python TLS trust issue rather than proof the dataset is unavailable.

### Suggested Fix
Try certifi-backed REQUESTS_CA_BUNDLE/SSL_CERT_FILE, or use an official alternate dataset endpoint such as ModelScope/OpenCSG if Hugging Face TLS remains blocked.

### Metadata
- Reproducible: yes
- Related Files: simplcore/mix/data_pipeline

---

