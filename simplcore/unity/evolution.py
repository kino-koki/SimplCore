# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplUnity 后台自进化管理器：模型在真实环境（Computer Use）学代码并把经验写进权重时，
后台并行长出新的 Unity 小专家——每个新域一份**不同**的数据，训练完原子换入活动模型。

设计要点（对应需求「后台应该同时训练新的模型模块」）：
* 训练在**独立的候选专家**上进行（``train_standalone_expert``），不触碰活动模型的参数，
  因此后台训练与主线前向/推理互不竞态；
* 一个域训完，``UnityModel.publish_expert`` 在预算守卫下把候选换入活动模型
  （``spawn_expert`` 原子新增 + 载入权重），路由门随后逐步学会在新域激活它；
* 受 ``WeightBudget`` 约束：总预算（默认 3B）/ 可学上限（2B）耗尽时拒绝生长；
* 提供 ``save`` / ``load`` 封装，使「自进化」跨会话累积（growth 落盘）。

> 生产部署建议：本管理器用线程在进程内并发。若需跨进程或多卡，可把 worker 改为
> 独立进程，通过队列/共享存储传递候选状态；本模块给出线程版并发安全 API。
"""
from __future__ import annotations

import queue
import threading
from typing import Callable, Iterator, List, Optional, Tuple

import torch


class UnitySelfEvolution:
    """后台训练新专家的管理器。

    用法::

        evo = UnitySelfEvolution(unity)
        evo.start()
        evo.register_domain("python_syntax", loader_for_python)
        evo.register_domain("bash_cli",     loader_for_bash)
        ...
        evo.stop()            # 等队列排空后退出
        evo.save("runs/tuan-1b/unity.pt")
    """

    def __init__(self, unity, device=None, amp_dtype=None,
                 steps_per_domain: int = 200, lr: float = 1e-2,
                 num_workers: int = 1) -> None:
        self.unity = unity
        self.device = device
        self.amp_dtype = amp_dtype
        self.steps_per_domain = int(steps_per_domain)
        self.lr = float(lr)
        self.num_workers = max(1, int(num_workers))
        self._queue: "queue.Queue[dict]" = queue.Queue()
        self._lock = threading.Lock()      # 保护 publish（活动模型专家列表变更）
        self._running = False
        self._workers: List[threading.Thread] = []
        self._pending: List[str] = []
        self._done: List[str] = []
        self._rejected: List[str] = []
        self._errors: List[Tuple[str, str]] = []

    # -- 生命周期 ----------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._workers = [
            threading.Thread(target=self._loop, daemon=True,
                             name=f"unity-evo-{i}")
            for i in range(self.num_workers)]
        for w in self._workers:
            w.start()

    def stop(self, timeout: float = 30.0) -> None:
        self._running = False
        for w in self._workers:
            w.join(timeout=timeout)
        self._workers = []

    # -- 任务提交 ----------------------------------------------------------
    def register_domain(self, name: str, loader: Iterator,
                        eval_loader=None, max_regression: float = 0.02) -> None:
        """提交一个新域：``loader`` 产出训练 batch；``eval_loader`` 用于无回归门。

        ``eval_loader`` 必须是**可重复**的 batch 序列（list/tuple）。候选专家先在
        该集合上与"无专家"基线比较；CE 上升超过 ``max_regression`` 则拒绝发布，
        活动模型不变。``max_regression=None`` 关闭门控（旧行为）。
        """
        self._queue.put({
            "name": str(name), "loader": loader,
            "eval_loader": None if eval_loader is None else list(eval_loader),
            "max_regression": (None if max_regression is None
                               else float(max_regression)),
        })
        with self._lock:
            self._pending.append(str(name))

    @property
    def num_pending(self) -> int:
        return self._queue.qsize()

    @property
    def num_experts(self) -> int:
        return len(self.unity.experts)

    def status(self) -> dict:
        with self._lock:
            return {
                "running": self._running,
                "workers": self.num_workers,
                "pending": list(self._pending),
                "done": list(self._done),
                "rejected": list(self._rejected),
                "errors": list(self._errors),
                "num_experts": len(self.unity.experts),
                "budget": self.unity.as_dict(),
            }

    # -- worker ------------------------------------------------------------
    def _loop(self) -> None:
        while self._running or not self._queue.empty():
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            name = str(task.get("name", "<unnamed>"))
            try:
                accepted, avg_ce, base_ce, cand_ce = self._train_one(task)
                with self._lock:
                    if name in self._pending:
                        self._pending.remove(name)
                    (self._done if accepted else self._rejected).append(name)
            except Exception as exc:  # 单域失败不拖垮后台 worker
                with self._lock:
                    if name in self._pending:
                        self._pending.remove(name)
                    self._errors.append((name, repr(exc)))
                print(f"[unity-evo] domain '{name}' failed: {exc!r}")
            finally:
                self._queue.task_done()

    def _train_one(self, task: dict):
        name = str(task["name"])
        loader = task["loader"]
        eval_loader = task.get("eval_loader")
        max_regression = task.get("max_regression")
        r = self.unity.cfg.resolve(self.unity.dim)
        # 1) 独立候选专家（不参与活动模型前向）→ 无竞态
        candidate = type(self.unity.experts[0])(
            self.unity.dim, r.expert_hidden, r.expert_layers, r.expert_dropout,
            name=name)
        if self.device is not None:
            candidate = candidate.to(self.device)
        opt = torch.optim.AdamW(candidate.parameters(), lr=self.lr)
        from .model import evaluate_standalone, train_standalone_expert
        avg_ce = train_standalone_expert(
            self.unity.base, self.unity.head, candidate, loader, opt,
            steps=self.steps_per_domain, device=self.device,
            amp_dtype=self.amp_dtype)
        # 2) 无回归门：候选必须不劣于"无专家"基线，否则拒绝发布
        base_ce = cand_ce = None
        if eval_loader and max_regression is not None:
            base_ce = evaluate_standalone(
                self.unity.base, self.unity.head, None, eval_loader,
                device=self.device, amp_dtype=self.amp_dtype)
            cand_ce = evaluate_standalone(
                self.unity.base, self.unity.head, candidate, eval_loader,
                device=self.device, amp_dtype=self.amp_dtype)
            if cand_ce > base_ce + float(max_regression):
                print(f"[unity-evo] reject domain '{name}': "
                      f"candidate_ce={cand_ce:.4f} > base_ce={base_ce:.4f} "
                      f"+ {max_regression:.4f}")
                return False, avg_ce, base_ce, cand_ce
        # 3) 原子换入活动模型（预算守卫在 spawn_expert 内）
        with self._lock:
            idx = self.unity.publish_expert(candidate, name=name)
        gate = "" if base_ce is None else \
            f", gate base={base_ce:.3f} cand={cand_ce:.3f}"
        print(f"[unity-evo] published domain '{name}' as expert#{idx} "
              f"(avg_ce={avg_ce:.3f}{gate}, total_experts={len(self.unity.experts)})")
        return True, avg_ce, base_ce, cand_ce

    # -- 持久化 ------------------------------------------------------------
    def save(self, path: str) -> None:
        self.unity.save_learned(path)

    @staticmethod
    def load(base, path, cfg=None):
        from .model import UnityModel
        return UnityModel.load_learned(base, path, cfg=cfg)
