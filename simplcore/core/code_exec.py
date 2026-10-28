# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""代码执行沙箱 + pass@k 验证器 —— 解锁 ``DreamingLoop`` 的 ``weights`` 模式。

为什么需要它
------------
支柱 C（"去真实环境写代码 → 写回权重"）缺的不是训练代码，而是一个能**判定
代码对错**的独立评测器。``dream.py`` 的 ``DreamingLoop`` 在没有独立 verifier 时
会拒绝写回权重（``mode="weights"`` 直接 ROLLBACK），因为"让 gate 恒通过"
等于没有 gate。本模块提供这个缺失的评测器：

* :class:`LocalPythonSandbox` —— 在隔离子进程里跑一段 Python，带超时与输出截断；
* :func:`extract_code` —— 从模型输出里剥出 ```python ... ``` 代码块；
* :class:`CodeProblem` —— 一道 held-out 题（prompt + 断言测试）；
* :class:`CodeExecutionVerifier` —— ``(model) -> {"pass_rate": [逐题 0..1]}``，
  正好满足 :meth:`simplcore.core.dream.EvalScores.from_any` 的"逐样本分数"要求，
  可直接用于 no-regression gate 的配对 bootstrap；
* :func:`make_smoke_problems` —— 3 道内置小题，用于玩具闭环冒烟。

诚实边界
--------
这是**进程级**沙箱，不是安全边界：不拦网络、不拦文件系统、无 OS 级隔离。
资源约束比早期版本强：POSIX 用 rlimit 限 CPU/内存；Windows 用 Job Object
限进程内存配额，且句柄关闭时整个子进程组被终止（超时清理不再依赖单进程
kill）。它足以跑"模型自己生成的、你愿意信任的"自训代码，但仍**不能跑
不可信代码**——上生产的正确做法是换成容器 / gVisor / firejail，接口保持不变。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

__all__ = [
    "CODE_PROMPT",
    "TestOutcome",
    "SandboxResult",
    "LocalPythonSandbox",
    "CodeProblem",
    "CodeExecutionVerifier",
    "extract_code",
    "run_python_tests",
    "make_smoke_problems",
]

#: 给模型的最小代码提示。任务指令由 ``CodeProblem.prompt`` 填入。
CODE_PROMPT = (
    "请只输出一个 ```python 代码块，不要输出解释或多余文字。\n"
    "代码必须自包含、可直接运行，并严格按任务要求的函数名与签名实现。\n\n"
    "任务：\n{instruction}\n"
)

_FENCE_RE = re.compile(r"```[^\n`]*\n(.*?)```", re.S)
_CODE_START_RE = re.compile(r"^\s*(?:def |class |import |from |@)")


# ---------------------------------------------------------------------------
# 代码块抽取
# ---------------------------------------------------------------------------
def extract_code(text: str) -> str:
    """从模型输出里取出可执行代码。

    优先取第一个 ```...``` 围栏块（忽略语言标记）；没有围栏时，从第一行
    ``def``/``class``/``import``/``from``/``@`` 起截到末尾；再退化为原文。
    """
    if not text:
        return ""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip("\n")
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if _CODE_START_RE.match(ln):
            return "\n".join(lines[i:]).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# 沙箱
# ---------------------------------------------------------------------------
@dataclass
class TestOutcome:
    """单条测试的执行结果。"""
    passed: bool
    timed_out: bool = False
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""


@dataclass
class SandboxResult:
    """一道题（可能含多条测试）的执行汇总。"""
    passed: int
    total: int
    outcomes: List[TestOutcome] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.total > 0 and self.passed == self.total

    @property
    def timed_out(self) -> bool:
        return any(o.timed_out for o in self.outcomes)

    @property
    def score(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def detail(self) -> str:
        if self.ok:
            return f"{self.passed}/{self.total} passed"
        for i, o in enumerate(self.outcomes):
            if not o.passed:
                tag = "timeout" if o.timed_out else f"exit={o.exit_code}"
                msg = (o.stderr or o.stdout or "").strip().splitlines()
                tail = msg[-1][:200] if msg else ""
                return f"{self.passed}/{self.total} passed; first failure: test#{i} ({tag}) {tail}"
        return f"{self.passed}/{self.total} passed"


def _trunc(s: str, n: int) -> str:
    if not s:
        return ""
    if len(s) <= n:
        return s
    return s[:n] + f"\n... [truncated {len(s) - n} chars]"


def _as_text(b) -> str:
    if b is None:
        return ""
    if isinstance(b, bytes):
        return b.decode("utf-8", "replace")
    return str(b)


def _build_script(code: str, test_src: str) -> str:
    """拼出"用户代码 + 测试"的完整脚本。

    退出码约定：0=通过；1=编译/导入失败；3=断言失败；4=运行时异常。
    这样调用方不用解析 stderr 就能区分"答错"与"代码崩了"。
    """
    return (
        "# -*- coding: utf-8 -*-\n"
        "import sys, traceback\n"
        f"{code}\n\n"
        "# ---- held-out test ----\n"
        "try:\n"
        f"    exec(compile({test_src!r}, '<test>', 'exec'), globals())\n"
        "except AssertionError:\n"
        "    sys.exit(3)\n"
        "except BaseException:\n"
        "    traceback.print_exc()\n"
        "    sys.exit(4)\n"
        "sys.exit(0)\n"
    )


def _posix_limit_fn(cpu_s: Optional[float], mem_mb: Optional[int]):
    """POSIX 下用 rlimit 兜底 CPU/内存。"""
    if os.name != "posix":
        return None

    def _apply() -> None:
        try:
            import resource
            if cpu_s:
                hard = int(cpu_s) + 1
                resource.setrlimit(resource.RLIMIT_CPU, (hard, hard))
            if mem_mb:
                b = int(mem_mb) * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (b, b))
        except Exception:
            pass

    return _apply


class _WinJob:
    """Windows Job Object 封装：进程内存配额 + KILL_ON_JOB_CLOSE.

    以前 Windows 只有 timeout 一道防线（rlimit 是 POSIX 专属）。Job Object
    补上两件事：

    * ``ProcessMemoryLimit`` —— 子进程内存分配超过配额直接失败（Python 里
      表现为 MemoryError → 我们的脚本约定 exit 4），呼应 POSIX 的 RLIMIT_AS；
    * ``KILL_ON_JOB_CLOSE`` —— 句柄关闭即终止作业内全部进程，超时清理
      不再依赖 ``proc.kill()`` 单点（子进程再派生的进程也会一起被收掉）。

    失败语义：Windows 上创建/配置 Job 失败会抛 OSError 而不是静默降级——
    静默降级正是本项目在 ``_project_memory`` 等处明确杜绝的行为。
    """

    def __init__(self, mem_mb: Optional[int]) -> None:
        import ctypes
        from ctypes import wintypes

        self._ct = ctypes
        kernel32 = ctypes.windll.kernel32
        self._kernel32 = kernel32

        class _IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in
                        ("ReadOperationCount", "WriteOperationCount",
                         "OtherOperationCount", "ReadTransferCount",
                         "WriteTransferCount", "OtherTransferCount")]

        class _BASIC_LIMIT_INFO(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),          # ULONG_PTR
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _EXT_LIMIT_INFO(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BASIC_LIMIT_INFO),
                ("IoInfo", _IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        self._EXT_LIMIT_INFO = _EXT_LIMIT_INFO
        JobObjectExtendedLimitInformation = 9          # JOBOBJECTINFOCLASS
        JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise OSError(f"CreateJobObjectW failed: {ctypes.GetLastError()}")
        self.handle = handle
        try:
            info = _EXT_LIMIT_INFO()
            flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if mem_mb:
                flags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY
                info.ProcessMemoryLimit = int(mem_mb) * 1024 * 1024
            info.BasicLimitInformation.LimitFlags = flags
            if not kernel32.SetInformationJobObject(
                    handle, JobObjectExtendedLimitInformation,
                    ctypes.byref(info), ctypes.sizeof(info)):
                raise OSError(f"SetInformationJobObject failed: {ctypes.GetLastError()}")
        except Exception:
            self.close()
            raise

    def assign(self, pid: int) -> None:
        """把已启动的子进程纳入作业（须在子进程做任何有意义的工作之前）。"""
        PROCESS_SET_QUOTA = 0x0100
        PROCESS_TERMINATE = 0x0001
        kernel32 = self._kernel32
        h = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
        if not h:
            raise OSError(f"OpenProcess({pid}) failed: {kernel32.GetLastError()}")
        try:
            if not kernel32.AssignProcessToJobObject(self.handle, h):
                raise OSError(f"AssignProcessToJobObject failed: {kernel32.GetLastError()}")
        finally:
            kernel32.CloseHandle(h)

    def close(self) -> None:
        """关闭句柄；KILL_ON_JOB_CLOSE 使作业内所有进程随之终止。"""
        if getattr(self, "handle", None):
            self._kernel32.CloseHandle(self.handle)
            self.handle = None


class LocalPythonSandbox:
    """本地 Python 子进程沙箱。

    Args:
        timeout_s: 单条测试默认超时（秒）。
        max_output_chars: stdout/stderr 截断长度（防止模型代码刷屏）。
        python_exe: 解释器路径，默认当前解释器。
        memory_limit_mb: 资源上限——POSIX 用 RLIMIT_AS 限地址空间；
            Windows 用 Job Object 限进程内存配额（并随句柄关闭终止整个
            作业进程组）。``None`` 表示不限。
        inherit_env: 是否继承完整环境变量。默认只给最小集合（含 TEMP/TMP，
            遵循本机"缓存放 E 盘"的约定）。
    """

    def __init__(self, timeout_s: float = 5.0, max_output_chars: int = 4000,
                 python_exe: Optional[str] = None, memory_limit_mb: Optional[int] = 2048,
                 inherit_env: bool = False) -> None:
        self.timeout_s = float(timeout_s)
        self.max_output_chars = int(max_output_chars)
        self.python_exe = python_exe or sys.executable
        self.memory_limit_mb = memory_limit_mb
        self.inherit_env = inherit_env

    def _env(self) -> Dict[str, str]:
        if self.inherit_env:
            env = dict(os.environ)
        else:
            env = {}
            for k in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "HOME", "USERPROFILE"):
                v = os.environ.get(k)
                if v:
                    env[k] = v
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONHASHSEED"] = "0"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env

    def run(self, code: str, test_src: str,
            timeout_s: Optional[float] = None) -> TestOutcome:
        """执行 ``code`` 后跑 ``test_src``，返回单条测试结果。"""
        timeout = float(timeout_s if timeout_s is not None else self.timeout_s)
        with tempfile.TemporaryDirectory(prefix="sgc_codeexec_") as d:
            script = os.path.join(d, "case.py")
            with open(script, "w", encoding="utf-8") as f:
                f.write(_build_script(code, test_src))
            kw = dict(cwd=d, env=self._env(),
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            job = None
            try:
                if os.name == "nt":
                    # Windows：Job Object 内存配额 + 进程组终止。
                    # 注：即使 memory_limit_mb=None 也建作业，让
                    # KILL_ON_JOB_CLOSE 兜底超时清理（连同子进程的派生进程）。
                    job = _WinJob(self.memory_limit_mb)
                else:
                    pre = _posix_limit_fn(timeout, self.memory_limit_mb)
                    if pre is not None:
                        kw["preexec_fn"] = pre
                proc = subprocess.Popen([self.python_exe, "-I", "-B", script], **kw)
                if job is not None:
                    job.assign(proc.pid)
                try:
                    out, err = proc.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    out, err = proc.communicate()
                    return TestOutcome(
                        passed=False, timed_out=True, exit_code=None,
                        stdout=_trunc(_as_text(out), self.max_output_chars),
                        stderr=f"超出 {timeout:.1f}s 超时",
                    )
            except subprocess.TimeoutExpired:
                return TestOutcome(passed=False, timed_out=True, exit_code=None,
                                   stdout="", stderr=f"超出 {timeout:.1f}s 超时")
            except Exception as exc:  # noqa: BLE001
                return TestOutcome(
                    passed=False, timed_out=False, exit_code=None,
                    stdout="", stderr=f"沙箱启动失败: {type(exc).__name__}: {exc}",
                )
            finally:
                if job is not None:
                    job.close()          # KILL_ON_JOB_CLOSE：作业内进程全部终止
            return TestOutcome(
                passed=(proc.returncode == 0), timed_out=False,
                exit_code=proc.returncode,
                stdout=_trunc(_as_text(out), self.max_output_chars),
                stderr=_trunc(_as_text(err), self.max_output_chars),
            )


def run_python_tests(code: str, tests: List[str], timeout_s: float = 5.0,
                     sandbox: Optional[LocalPythonSandbox] = None,
                     max_output_chars: int = 4000) -> SandboxResult:
    """对 ``code`` 逐条跑 ``tests`` 里的断言，返回汇总。

    每条测试独立进程，因此一条崩溃（``sys.exit``/段错）不会污染其余测试。
    """
    tests = list(tests)
    if not tests:
        return SandboxResult(passed=0, total=0)
    box = sandbox or LocalPythonSandbox(timeout_s=timeout_s,
                                       max_output_chars=max_output_chars)
    outs = [box.run(code, t, timeout_s=timeout_s) for t in tests]
    return SandboxResult(passed=sum(1 for o in outs if o.passed),
                         total=len(outs), outcomes=outs)


# ---------------------------------------------------------------------------
# 题目 + 验证器
# ---------------------------------------------------------------------------
@dataclass
class CodeProblem:
    """一道 held-out 代码题。

    Args:
        name: 题目标识（报告用）。
        prompt: 给模型的自然语言任务描述。
        tests: 断言代码列表；每条是能 ``exec`` 的 Python 片段，失败即抛异常。
        timeout_s: 覆盖沙箱默认超时。
        meta: 任意元信息（来源/难度/标签）。
    """
    name: str
    prompt: str
    tests: List[str]
    timeout_s: Optional[float] = None
    meta: Dict[str, object] = field(default_factory=dict)


class CodeExecutionVerifier:
    """用真实执行结果给模型打分的 verifier。

    ``generator`` 是 ``(instruction) -> 模型原始输出`` 的可调用对象；本类负责
    抽代码、跑测试、换算成逐题分数。``__call__`` 的签名 ``(model=None)`` 与
    ``DreamingLoop`` 的 verifier 契约一致（模型已绑定在 generator 上，参数仅为兼容）。

    返回 ``{"pass_rate": [s1, s2, ...]}``：每题 0..1，**逐样本**，可直接喂给
    :meth:`simplcore.core.dream.EvalScores.from_any` 做配对 bootstrap。
    """

    def __init__(self, problems: List[CodeProblem],
                 generator: Callable[[str], str],
                 timeout_s: float = 5.0,
                 max_output_chars: int = 2000) -> None:
        self.problems = list(problems)
        if not self.problems:
            raise ValueError("CodeExecutionVerifier 至少需要一道题")
        self.generator = generator
        self.timeout_s = float(timeout_s)
        self.max_output_chars = int(max_output_chars)
        #: 最近一次评测的逐题明细（审计用）
        self.results: List[Dict[str, object]] = []

    def evaluate(self) -> List[float]:
        """跑全部题，返回逐题分数（0..1）。"""
        scores: List[float] = []
        self.results = []
        for p in self.problems:
            raw = self.generator(p.prompt) or ""
            code = extract_code(raw)
            res = run_python_tests(
                code, p.tests,
                timeout_s=p.timeout_s if p.timeout_s is not None else self.timeout_s,
                max_output_chars=self.max_output_chars,
            )
            score = (res.passed / res.total) if res.total else 0.0
            scores.append(float(score))
            self.results.append({
                "name": p.name,
                "score": float(score),
                "passed": res.passed,
                "total": res.total,
                "timed_out": res.timed_out,
                "detail": res.detail,
            })
        return scores

    def pass_rate(self) -> float:
        """平均通过率（报告用；gate 判定请用 :meth:`evaluate` 的逐题分数）。"""
        scores = self.evaluate()
        return sum(scores) / len(scores) if scores else 0.0

    def __call__(self, model=None) -> Dict[str, List[float]]:  # noqa: ARG002
        return {"pass_rate": self.evaluate()}

    @classmethod
    def from_model(cls, model, codec, problems: List[CodeProblem],
                   prompt_template: str = CODE_PROMPT,
                   max_new_tokens: int = 256,
                   timeout_s: float = 5.0) -> "CodeExecutionVerifier":
        """用 ``SimplCoreModel.generate_text`` 作为生成器构造 verifier。

        ``codec`` 是模型对应的 ``Tokenizer``（char/byte/BPE）。解码后由
        :func:`extract_code` 抽取代码块。
        """
        def _gen(instruction: str) -> str:
            prompt = prompt_template.format(instruction=instruction)
            return model.generate_text(prompt, codec,
                                       max_new_tokens=max_new_tokens,
                                       temperature=1.0, top_k=1)

        return cls(problems, _gen, timeout_s=timeout_s)


def make_smoke_problems() -> List[CodeProblem]:
    """3 道内置小题：用于在没有外部数据时冒烟整条闭环。"""
    return [
        CodeProblem(
            name="add",
            prompt="实现函数 add(a, b)，返回 a + b（支持整数与浮点数）。",
            tests=["assert add(2, 3) == 5", "assert add(-1.5, 0.5) == -1.0"],
        ),
        CodeProblem(
            name="is_even",
            prompt="实现函数 is_even(n)，n 为整数，偶数返回 True，否则返回 False。",
            tests=["assert is_even(0) == True", "assert is_even(3) == False",
                   "assert is_even(-4) == True"],
        ),
        CodeProblem(
            name="reverse_str",
            prompt="实现函数 reverse_str(s)，返回字符串 s 的反转。",
            tests=['assert reverse_str("abc") == "cba"', 'assert reverse_str("") == ""'],
        ),
    ]
