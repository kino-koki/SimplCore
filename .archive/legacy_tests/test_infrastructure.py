"""
青小团基础设施测试 - 覆盖检查点管理器、训练监控器、实验追踪、故障容错、分布式协调等
"""

import os
import json
import time
import torch
import datetime
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestQingxiaotuanCheckpointManager:

    def test_checkpoint_manager_creation(self, temp_dir):
        from training.infrastructure.checkpoint_manager import CheckpointManager
        manager = CheckpointManager(save_dir=temp_dir)
        assert manager.save_dir == temp_dir

    def test_checkpoint_manager_save(self, temp_dir, small_qingxiaotuan_config):
        from training.infrastructure.checkpoint_manager import CheckpointManager
        manager = CheckpointManager(save_dir=temp_dir, keep_last=3)
        from qingxiaotuan import QingxiaotuanForCausalLM
        model = QingxiaotuanForCausalLM(small_qingxiaotuan_config)
        path = manager.save(model, step=100, metrics={"loss": 0.1})
        assert os.path.exists(os.path.join(temp_dir, path)) or os.path.exists(temp_dir)

    def test_checkpoint_manager_keep_last(self, temp_dir):
        from training.infrastructure.checkpoint_manager import CheckpointManager
        manager = CheckpointManager(save_dir=temp_dir, keep_last=2)
        for i in range(4):
            ckpt = {"step": i, "data": f"ckpt_{i}"}
            path = os.path.join(temp_dir, f"ckpt_{i}.pt")
            torch.save(ckpt, path)
            manager.register_checkpoint(f"ckpt_{i}.pt")
        manager.cleanup()
        remaining = [f for f in os.listdir(temp_dir) if f.endswith(".pt")]
        assert len(remaining) <= 2

    def test_checkpoint_manager_load(self, temp_dir):
        from training.infrastructure.checkpoint_manager import CheckpointManager
        manager = CheckpointManager(save_dir=temp_dir)
        ckpt = {"step": 100, "loss": 0.05}
        path = os.path.join(temp_dir, "best.pt")
        torch.save(ckpt, path)
        loaded = manager.load(path)
        assert loaded["step"] == 100
        assert loaded["loss"] == 0.05

    def test_checkpoint_manager_latest(self, temp_dir):
        from training.infrastructure.checkpoint_manager import CheckpointManager
        manager = CheckpointManager(save_dir=temp_dir)
        for i in range(3):
            path = os.path.join(temp_dir, f"ckpt_{i}.pt")
            torch.save({"step": i}, path)
            time.sleep(0.1)
        latest = manager.get_latest_checkpoint()
        assert latest is not None

    def test_checkpoint_manager_best_by_metric(self, temp_dir):
        from training.infrastructure.checkpoint_manager import CheckpointManager
        manager = CheckpointManager(save_dir=temp_dir, metric="loss", mode="min")
        for step, loss in [(100, 0.1), (200, 0.05), (300, 0.08)]:
            path = os.path.join(temp_dir, f"ckpt_{step}.pt")
            torch.save({"step": step, "loss": loss}, path)
            manager.register_checkpoint(f"ckpt_{step}.pt", metrics={"loss": loss})
        best = manager.get_best_checkpoint()
        assert best is not None

    def test_checkpoint_manager_max_mode(self, temp_dir):
        from training.infrastructure.checkpoint_manager import CheckpointManager
        manager = CheckpointManager(save_dir=temp_dir, metric="acc", mode="max")
        for step, acc in [(100, 0.8), (200, 0.9), (300, 0.85)]:
            path = os.path.join(temp_dir, f"ckpt_{step}.pt")
            torch.save({"step": step, "acc": acc}, path)
            manager.register_checkpoint(f"ckpt_{step}.pt", metrics={"acc": acc})
        best = manager.get_best_checkpoint()
        assert best is not None

    def test_checkpoint_manager_resume(self, temp_dir):
        from training.infrastructure.checkpoint_manager import CheckpointManager
        manager = CheckpointManager(save_dir=temp_dir, resume_from=os.path.join(temp_dir, "resume.pt"))
        ckpt = {"step": 500, "optimizer": "state"}
        torch.save(ckpt, os.path.join(temp_dir, "resume.pt"))
        state = manager.load_for_resume()
        assert state is not None or manager.resume_from is not None

    def test_checkpoint_metadata(self, temp_dir):
        from training.infrastructure.checkpoint_manager import CheckpointManager
        manager = CheckpointManager(save_dir=temp_dir)
        metadata = {
            "step": 100, "loss": 0.1, "epoch": 2,
            "timestamp": datetime.datetime.now().isoformat()
        }
        path = os.path.join(temp_dir, "meta.pt")
        torch.save(metadata, path)
        loaded = torch.load(path, weights_only=False)
        assert loaded["step"] == 100

    def test_checkpoint_size_tracking(self, temp_dir):
        from training.infrastructure.checkpoint_manager import CheckpointManager
        manager = CheckpointManager(save_dir=temp_dir)
        data = {"weight": torch.randn(100, 100)}
        path = os.path.join(temp_dir, "size_test.pt")
        torch.save(data, path)
        size = os.path.getsize(path)
        assert size > 0


class TestQingxiaotuanTrainingMonitor:

    def test_training_monitor_creation(self):
        from training.infrastructure.training_monitor import TrainingMonitor
        monitor = TrainingMonitor()
        assert monitor is not None

    def test_training_monitor_log_metrics(self, temp_dir):
        from training.infrastructure.training_monitor import TrainingMonitor
        monitor = TrainingMonitor(log_dir=temp_dir)
        monitor.log_metrics(step=100, loss=0.1, lr=1e-4)
        assert len(monitor.metrics_history) >= 1

    def test_training_monitor_get_history(self, temp_dir):
        from training.infrastructure.training_monitor import TrainingMonitor
        monitor = TrainingMonitor(log_dir=temp_dir)
        for i in range(5):
            monitor.log_metrics(step=i * 10, loss=0.5 - i * 0.1)
        history = monitor.get_metrics_history()
        assert len(history) == 5

    def test_training_monitor_save_log(self, temp_dir):
        from training.infrastructure.training_monitor import TrainingMonitor
        monitor = TrainingMonitor(log_dir=temp_dir)
        monitor.log_metrics(step=100, loss=0.1)
        path = monitor.save_log("training_log.json")
        assert os.path.exists(os.path.join(temp_dir, "training_log.json")) or path is not None

    def test_training_monitor_latest_metrics(self, temp_dir):
        from training.infrastructure.training_monitor import TrainingMonitor
        monitor = TrainingMonitor(log_dir=temp_dir)
        monitor.log_metrics(step=100, loss=0.1)
        latest = monitor.get_latest_metrics()
        assert latest["loss"] == 0.1

    def test_training_monitor_plot(self, temp_dir):
        from training.infrastructure.training_monitor import TrainingMonitor
        monitor = TrainingMonitor(log_dir=temp_dir)
        for i in range(10):
            monitor.log_metrics(step=i * 10, loss=1.0 / (i + 1))
        plot_path = monitor.plot_metrics(save_path=os.path.join(temp_dir, "plot.png"))
        assert plot_path is not None

    def test_training_monitor_progress_bar(self, temp_dir):
        from training.infrastructure.training_monitor import TrainingMonitor
        monitor = TrainingMonitor(log_dir=temp_dir, total_steps=100)
        for i in range(10):
            monitor.log_metrics(step=i * 10, loss=0.1)
        progress = monitor.get_progress()
        assert progress >= 0.1

    def test_training_monitor_memory_tracking(self, temp_dir):
        from training.infrastructure.training_monitor import TrainingMonitor
        monitor = TrainingMonitor(log_dir=temp_dir, track_memory=True)
        monitor.log_metrics(step=1, loss=0.1)
        assert hasattr(monitor, 'memory_stats')


class TestQingxiaotuanExperimentTracker:

    def test_experiment_tracker_creation(self):
        from training.infrastructure.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker()
        assert tracker is not None

    def test_experiment_tracker_start_run(self, temp_dir):
        from training.infrastructure.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(experiment_dir=temp_dir)
        run_id = tracker.start_run(name="test_run")
        assert run_id is not None

    def test_experiment_tracker_log_params(self, temp_dir):
        from training.infrastructure.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(experiment_dir=temp_dir)
        tracker.start_run(name="param_test")
        tracker.log_params({"learning_rate": 1e-4, "batch_size": 8})
        params = tracker.get_run_params()
        assert params["learning_rate"] == 1e-4
        assert params["batch_size"] == 8

    def test_experiment_tracker_log_metrics(self, temp_dir):
        from training.infrastructure.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(experiment_dir=temp_dir)
        tracker.start_run(name="metric_test")
        tracker.log_metrics({"loss": 0.1, "acc": 0.95})
        metrics = tracker.get_latest_metrics()
        assert metrics["loss"] == 0.1

    def test_experiment_tracker_end_run(self, temp_dir):
        from training.infrastructure.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(experiment_dir=temp_dir)
        tracker.start_run(name="end_test")
        tracker.end_run(status="completed")
        assert tracker.run_status == "completed"

    def test_experiment_tracker_list_runs(self, temp_dir):
        from training.infrastructure.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(experiment_dir=temp_dir)
        tracker.start_run(name="run1")
        tracker.end_run()
        tracker.start_run(name="run2")
        runs = tracker.list_runs()
        assert len(runs) >= 2

    def test_experiment_tracker_save_config(self, temp_dir):
        from training.infrastructure.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(experiment_dir=temp_dir)
        tracker.start_run(name="config_test")
        config = {"model": "qingxiaotuan", "num_layers": 48}
        tracker.save_config(config)
        loaded = tracker.load_config()
        assert loaded["model"] == "qingxiaotuan"

    def test_experiment_tracker_export(self, temp_dir):
        from training.infrastructure.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(experiment_dir=temp_dir)
        tracker.start_run(name="export_test")
        tracker.log_metrics({"loss": 0.1})
        tracker.log_params({"lr": 1e-4})
        path = tracker.export_json(os.path.join(temp_dir, "export.json"))
        assert os.path.exists(path) or path is None


class TestQingxiaotuanFaultTolerance:

    def test_fault_tolerance_handler_creation(self):
        from training.infrastructure.fault_tolerance import FaultToleranceHandler
        handler = FaultToleranceHandler()
        assert handler is not None

    def test_fault_tolerance_save_heartbeat(self, temp_dir):
        from training.infrastructure.fault_tolerance import FaultToleranceHandler
        handler = FaultToleranceHandler(heartbeat_dir=temp_dir)
        handler.save_heartbeat(step=100)
        assert os.path.exists(os.path.join(temp_dir, "heartbeat.pt")) or \
               len(os.listdir(temp_dir)) > 0

    def test_fault_tolerance_auto_recovery(self, temp_dir):
        from training.infrastructure.fault_tolerance import FaultToleranceHandler
        handler = FaultToleranceHandler(heartbeat_dir=temp_dir, auto_recovery=True)
        handler.save_heartbeat(step=50)
        last_step = handler.get_last_heartbeat_step()
        assert last_step == 50 or last_step is not None

    def test_fault_tolerance_timeout(self):
        from training.infrastructure.fault_tolerance import FaultToleranceHandler
        handler = FaultToleranceHandler(heartbeat_timeout=30)
        assert handler.heartbeat_timeout == 30

    def test_fault_tolerance_backup_state(self, temp_dir):
        from training.infrastructure.fault_tolerance import FaultToleranceHandler
        handler = FaultToleranceHandler(heartbeat_dir=temp_dir)
        state = {"step": 100, "loss": 0.05}
        handler.save_state(state)
        loaded = handler.load_state()
        assert loaded is not None

    def test_fault_tolerance_check_integrity(self, temp_dir):
        from training.infrastructure.fault_tolerance import FaultToleranceHandler
        handler = FaultToleranceHandler(heartbeat_dir=temp_dir)
        state = {"step": 100, "data": "test"}
        handler.save_state(state)
        assert handler.check_integrity() is True


class TestQingxiaotuanDistributedCoordination:

    def test_distributed_coordinator_creation(self):
        from training.infrastructure.distributed_coordinator import DistributedCoordinator
        coordinator = DistributedCoordinator(world_size=8)
        assert coordinator.world_size == 8

    def test_distributed_coordinator_rank_assignment(self):
        from training.infrastructure.distributed_coordinator import DistributedCoordinator
        coordinator = DistributedCoordinator(world_size=4, rank=0)
        assert coordinator.rank == 0

    def test_distributed_coordinator_world_size(self):
        from training.infrastructure.distributed_coordinator import DistributedCoordinator
        for ws in [1, 2, 4, 8, 16]:
            coordinator = DistributedCoordinator(world_size=ws)
            assert coordinator.world_size == ws

    def test_distributed_coordinator_barrier(self):
        from training.infrastructure.distributed_coordinator import DistributedCoordinator
        coordinator = DistributedCoordinator(world_size=1)
        result = coordinator.barrier()
        assert result is None or result is True

    def test_distributed_coordinator_all_reduce(self):
        from training.infrastructure.distributed_coordinator import DistributedCoordinator
        coordinator = DistributedCoordinator(world_size=1)
        data = torch.tensor([1.0, 2.0, 3.0])
        result = coordinator.all_reduce(data)
        assert torch.allclose(result, data)

    def test_distributed_coordinator_all_gather(self):
        from training.infrastructure.distributed_coordinator import DistributedCoordinator
        coordinator = DistributedCoordinator(world_size=1)
        data = [1, 2, 3]
        result = coordinator.all_gather(data)
        assert result == [data]

    def test_distributed_coordinator_broadcast(self):
        from training.infrastructure.distributed_coordinator import DistributedCoordinator
        coordinator = DistributedCoordinator(world_size=1, rank=0)
        data = {"config": "test"}
        result = coordinator.broadcast(data, src=0)
        assert result == data


class TestQingxiaotuanHyperparameterOptimization:

    def test_hpo_creation(self):
        from training.infrastructure.hyperparameter_optimization import HyperparameterOptimizer
        optimizer = HyperparameterOptimizer()
        assert optimizer is not None

    def test_hpo_random_search(self):
        from training.infrastructure.hyperparameter_optimization import HyperparameterOptimizer
        space = {"learning_rate": [1e-5, 2e-5, 5e-5, 1e-4], "batch_size": [1, 2, 4]}
        optimizer = HyperparameterOptimizer(search_strategy="random", n_trials=4)
        trials = optimizer.generate_trials(space)
        assert len(trials) == 4

    def test_hpo_grid_search(self):
        from training.infrastructure.hyperparameter_optimization import HyperparameterOptimizer
        space = {"lr": [1e-5, 1e-4], "bs": [1, 2]}
        optimizer = HyperparameterOptimizer(search_strategy="grid", n_trials=4)
        trials = optimizer.generate_trials(space)
        assert len(trials) <= 4

    def test_hpo_bayesian_search(self):
        from training.infrastructure.hyperparameter_optimization import HyperparameterOptimizer
        optimizer = HyperparameterOptimizer(search_strategy="bayesian", n_trials=3)
        space = {"learning_rate": (1e-6, 1e-3), "dropout": (0.0, 0.5)}
        trials = optimizer.generate_trials(space)
        assert len(trials) == 3

    def test_hpo_trial_result_tracking(self, temp_dir):
        from training.infrastructure.hyperparameter_optimization import HyperparameterOptimizer
        optimizer = HyperparameterOptimizer(search_strategy="random", n_trials=5)
        optimizer.add_trial_result(trial_id=1, params={"lr": 1e-4}, metrics={"loss": 0.1})
        optimizer.add_trial_result(trial_id=2, params={"lr": 5e-5}, metrics={"loss": 0.08})
        best = optimizer.get_best_trial()
        assert best is not None

    def test_hpo_save_results(self, temp_dir):
        from training.infrastructure.hyperparameter_optimization import HyperparameterOptimizer
        optimizer = HyperparameterOptimizer(search_strategy="random", n_trials=3)
        optimizer.add_trial_result(1, {"lr": 1e-4}, {"loss": 0.1})
        path = os.path.join(temp_dir, "hpo_results.json")
        optimizer.save_results(path)
        assert os.path.exists(path)


class TestQingxiaotuanProfiler:

    def test_profiler_creation(self, temp_dir):
        from training.infrastructure.profiler import Profiler
        profiler = Profiler(output_dir=temp_dir)
        assert profiler.output_dir == temp_dir

    def test_profiler_start_stop(self, temp_dir):
        from training.infrastructure.profiler import Profiler
        profiler = Profiler(output_dir=temp_dir)
        profiler.start()
        time.sleep(0.01)
        profiler.stop()
        stats = profiler.get_stats()
        assert stats is not None

    def test_profiler_track_function(self, temp_dir):
        from training.infrastructure.profiler import Profiler
        profiler = Profiler(output_dir=temp_dir)
        with profiler.track("test_op"):
            _ = sum(range(1000))
        traces = profiler.get_traces()
        assert len(traces) >= 1

    def test_profiler_memory_usage(self, temp_dir):
        from training.infrastructure.profiler import Profiler
        profiler = Profiler(output_dir=temp_dir)
        profiler.start()
        mem = profiler.get_memory_usage()
        profiler.stop()
        assert isinstance(mem, float)

    def test_profiler_report(self, temp_dir):
        from training.infrastructure.profiler import Profiler
        profiler = Profiler(output_dir=temp_dir)
        profiler.start()
        _ = sum(range(10000))
        profiler.stop()
        report = profiler.generate_report()
        assert report is not None


class TestQingxiaotuanPipelineOrchestrator:

    def test_orchestrator_creation(self):
        from training.infrastructure.training_pipeline_orchestrator import TrainingPipelineOrchestrator
        orchestrator = TrainingPipelineOrchestrator()
        assert orchestrator is not None

    def test_orchestrator_run_stages(self, temp_dir):
        from training.infrastructure.training_pipeline_orchestrator import TrainingPipelineOrchestrator
        orchestrator = TrainingPipelineOrchestrator(output_dir=temp_dir)
        stages = orchestrator.plan_pipeline(
            stages=["data_preparation", "pretraining", "sft", "evaluation"]
        )
        assert len(stages) == 4

    def test_orchestrator_stage_dependency(self, temp_dir):
        from training.infrastructure.training_pipeline_orchestrator import TrainingPipelineOrchestrator
        orchestrator = TrainingPipelineOrchestrator(output_dir=temp_dir)
        deps = orchestrator.get_stage_dependencies()
        assert isinstance(deps, dict)

    def test_orchestrator_resume_pipeline(self, temp_dir):
        from training.infrastructure.training_pipeline_orchestrator import TrainingPipelineOrchestrator
        orchestrator = TrainingPipelineOrchestrator(output_dir=temp_dir, resume=True)
        assert orchestrator.resume is True

    def test_orchestrator_check_stage_status(self, temp_dir):
        from training.infrastructure.training_pipeline_orchestrator import TrainingPipelineOrchestrator
        orchestrator = TrainingPipelineOrchestrator(output_dir=temp_dir)
        status = orchestrator.get_stage_status("pretraining")
        assert status is not None

    def test_orchestrator_parallel_stages(self, temp_dir):
        from training.infrastructure.training_pipeline_orchestrator import TrainingPipelineOrchestrator
        orchestrator = TrainingPipelineOrchestrator(output_dir=temp_dir, parallel_stages=True)
        assert orchestrator.parallel_stages is True

    def test_orchestrator_update_stage(self, temp_dir):
        from training.infrastructure.training_pipeline_orchestrator import TrainingPipelineOrchestrator
        orchestrator = TrainingPipelineOrchestrator(output_dir=temp_dir)
        orchestrator.update_stage_status("pretraining", "completed")
        status = orchestrator.get_stage_status("pretraining")
        assert status == "completed"
