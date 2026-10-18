"""Aynı fiziksel parametre bütçeli Kronecker ve rank-1 baseline deneyi.

Bu modül PyTorch'u import anında zorunlu kılmaz. Deney iki öğretmen görevi
kullanır; böylece tek bir Kronecker-uyumlu görevden evrensel üstünlük sonucu
çıkarılmaz:

* ``kronecker_teacher``: Y = A* X B* — Kronecker inductive bias lehine.
* ``rank1_teacher``: vec(Y) = u (v^T vec(X)) — rank-1 baseline lehine.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, Sequence


def _torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as error:  # pragma: no cover - ortama bağlı
        raise ImportError("Kronecker benchmark için PyTorch gereklidir") from error
    return torch, nn


def kronecker_capacity_contract(n: int) -> Dict[str, Any]:
    """Torch gerektirmeyen fiziksel/sanal kapasite ve rank üst sınırları."""
    if n < 2:
        raise ValueError("n >= 2 olmalı")
    dimension = n * n
    return {
        "n": n,
        "matrix_dimension": dimension,
        "physical_parameter_budget_each": 2 * n * n,
        "full_operator_entries_n4": n ** 4,
        "full_dense_parameters_without_bias": n ** 4,
        "operator_entries_are_parameters": False,
        "kronecker_max_effective_operator_rank": dimension,
        "rank1_bottleneck_max_effective_operator_rank": 1,
    }


def _parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _parameter_bytes(model) -> int:
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _regression_metrics(torch, prediction, target) -> Dict[str, float]:
    error = prediction - target
    mse = float(torch.mean(error.square()).item())
    target_energy = float(torch.mean(target.square()).item())
    centered_energy = float(torch.mean((target - target.mean()).square()).item())
    tolerance = 0.1 * float(torch.mean(target.abs()).item()) + 1e-12
    return {
        "mse": mse,
        "normalized_mse": mse / max(target_energy, 1e-12),
        "r2": 1.0 - mse / max(centered_energy, 1e-12),
        "tolerance_accuracy": float(torch.mean((error.abs() <= tolerance).float()).item()),
        "tolerance": tolerance,
    }


def run_kronecker_dense_trial(
    n: int = 16,
    steps: int = 100,
    batch_size: int = 16,
    test_samples: int = 128,
    learning_rate: float = 0.01,
    seed: int = 42,
    device: str = "cpu",
    tasks: Sequence[str] = ("kronecker_teacher", "rank1_teacher"),
) -> Dict[str, Any]:
    """İki eşit bütçeli modeli aynı veri ve optimizer protokolünde eğit."""
    torch, nn = _torch()
    if n < 2 or steps < 1 or batch_size < 1 or test_samples < 1:
        raise ValueError("n>=2; steps, batch_size ve test_samples >=1 olmalı")
    if not tasks:
        raise ValueError("En az bir görev gerekli")
    allowed_tasks = {"kronecker_teacher", "rank1_teacher"}
    unknown = set(tasks) - allowed_tasks
    if unknown:
        raise ValueError(f"Bilinmeyen görevler: {sorted(unknown)}")

    target_device = torch.device(device)
    capacity = kronecker_capacity_contract(n)
    dimension = capacity["matrix_dimension"]

    class KroneckerBilinearLayer(nn.Module):  # type: ignore[name-defined]
        def __init__(self):
            super().__init__()
            self.A = nn.Parameter(torch.empty(n, n))
            self.B = nn.Parameter(torch.empty(n, n))
            nn.init.xavier_uniform_(self.A)
            nn.init.xavier_uniform_(self.B)

        def forward(self, inputs):
            return torch.matmul(self.A, torch.matmul(inputs, self.B))

    class RankOneFactorizedLayer(nn.Module):  # type: ignore[name-defined]
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(dimension, 1, bias=False)
            self.fc2 = nn.Linear(1, dimension, bias=False)

        def forward(self, inputs):
            flat = inputs.reshape(inputs.shape[0], dimension)
            return self.fc2(self.fc1(flat)).reshape(inputs.shape[0], n, n)

    def make_data(task: str, task_index: int):
        generator = torch.Generator(device="cpu").manual_seed(seed + 10_000 * task_index)
        train_x = torch.randn(steps, batch_size, n, n, generator=generator)
        test_x = torch.randn(test_samples, n, n, generator=generator)
        if task == "kronecker_teacher":
            left = torch.randn(n, n, generator=generator) / math.sqrt(n)
            right = torch.randn(n, n, generator=generator) / math.sqrt(n)

            def teacher(inputs):
                return torch.matmul(left, torch.matmul(inputs, right))
        else:
            output_vector = torch.randn(dimension, generator=generator)
            input_vector = torch.randn(dimension, generator=generator) / math.sqrt(dimension)

            def teacher(inputs):
                flat = inputs.reshape(*inputs.shape[:-2], dimension)
                scalar = torch.matmul(flat, input_vector)
                return (scalar.unsqueeze(-1) * output_vector).reshape(*inputs.shape[:-2], n, n)

        train_y = teacher(train_x)
        test_y = teacher(test_x)
        return (
            train_x.to(target_device), train_y.to(target_device),
            test_x.to(target_device), test_y.to(target_device),
        )

    def train_model(model, train_x, train_y, test_x, test_y):
        model = model.to(target_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        losses = []
        max_gradient_norm = 0.0
        nonfinite_steps = 0
        if target_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(target_device)
            torch.cuda.synchronize(target_device)
        started = time.perf_counter()
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(train_x[step])
            loss = torch.mean((prediction - train_y[step]).square())
            if not bool(torch.isfinite(loss)):
                nonfinite_steps += 1
                continue
            loss.backward()
            squared_norm = 0.0
            gradients_finite = True
            for parameter in model.parameters():
                if parameter.grad is None:
                    continue
                gradients_finite = gradients_finite and bool(torch.isfinite(parameter.grad).all())
                squared_norm += float(parameter.grad.detach().square().sum().item())
            if not gradients_finite:
                nonfinite_steps += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            max_gradient_norm = max(max_gradient_norm, math.sqrt(squared_norm))
            optimizer.step()
            losses.append(float(loss.item()))
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        elapsed = time.perf_counter() - started
        with torch.no_grad():
            train_metrics = _regression_metrics(torch, model(train_x[-1]), train_y[-1])
            test_metrics = _regression_metrics(torch, model(test_x), test_y)
        tail = losses[-min(10, len(losses)):]
        tail_mean = sum(tail) / len(tail) if tail else float("nan")
        tail_variance = (
            sum((value - tail_mean) ** 2 for value in tail) / len(tail) if tail else float("nan")
        )
        parameter_bytes = _parameter_bytes(model)
        result = {
            "physical_parameters": _parameter_count(model),
            "parameter_bytes": parameter_bytes,
            "estimated_adam_training_bytes": parameter_bytes * 4,
            "train": train_metrics,
            "test": test_metrics,
            "final_step_loss": losses[-1] if losses else float("nan"),
            "tail_loss_std": math.sqrt(tail_variance),
            "max_gradient_norm": max_gradient_norm,
            "nonfinite_steps": nonfinite_steps,
            "optimization_stable": nonfinite_steps == 0 and bool(losses),
            "training_seconds": elapsed,
            "steps_per_second": steps / max(elapsed, 1e-12),
        }
        if target_device.type == "cuda":
            result["peak_cuda_bytes"] = int(torch.cuda.max_memory_allocated(target_device))
        return model, result

    task_results: Dict[str, Any] = {}
    for task_index, task in enumerate(tasks):
        train_x, train_y, test_x, test_y = make_data(task, task_index)
        models = {}
        trained = {}
        for model_index, (name, constructor) in enumerate((
            ("kronecker", KroneckerBilinearLayer),
            ("rank1_bottleneck", RankOneFactorizedLayer),
        )):
            torch.manual_seed(seed + 100_000 * task_index + model_index)
            model, metrics = train_model(constructor(), train_x, train_y, test_x, test_y)
            models[name] = metrics
            trained[name] = model

        kron = trained["kronecker"]
        dense = trained["rank1_bottleneck"]
        rank_a = int(torch.linalg.matrix_rank(kron.A.detach()).item())
        rank_b = int(torch.linalg.matrix_rank(kron.B.detach()).item())
        models["kronecker"]["factor_rank_a"] = rank_a
        models["kronecker"]["factor_rank_b"] = rank_b
        models["kronecker"]["effective_operator_rank"] = rank_a * rank_b
        models["kronecker"]["max_effective_operator_rank"] = dimension
        dense_rank = min(
            int(torch.linalg.matrix_rank(dense.fc1.weight.detach()).item()),
            int(torch.linalg.matrix_rank(dense.fc2.weight.detach()).item()),
        )
        models["rank1_bottleneck"]["effective_operator_rank"] = dense_rank
        models["rank1_bottleneck"]["max_effective_operator_rank"] = 1
        winner = min(models, key=lambda name: models[name]["test"]["normalized_mse"])
        task_results[task] = {"models": models, "winner_by_test_normalized_mse": winner}

    expected_budget = 2 * n * n
    budgets = {
        name: task_results[next(iter(task_results))]["models"][name]["physical_parameters"]
        for name in ("kronecker", "rank1_bottleneck")
    }
    if any(value != expected_budget for value in budgets.values()):
        raise AssertionError(f"Parametre bütçesi eşit değil: {budgets}, beklenen={expected_budget}")
    return {
        "protocol": "kronecker-vs-rank1-param-matched-v2",
        "seed": int(seed),
        "n": n,
        "matrix_dimension": dimension,
        "steps": steps,
        "batch_size": batch_size,
        "test_samples": test_samples,
        "learning_rate": learning_rate,
        "device": str(target_device),
        "capacity": capacity,
        "fairness": {
            "same_physical_parameter_budget": True,
            "same_training_and_test_inputs_per_task": True,
            "same_optimizer": "Adam",
            "single_task_universal_superiority_claim": False,
            "baseline_name": "rank1_bottleneck (not unrestricted dense)",
        },
        "tasks": task_results,
    }


__all__ = ["kronecker_capacity_contract", "run_kronecker_dense_trial"]
