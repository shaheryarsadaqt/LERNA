"""Deterministic sampled, post-decision, lagged LER diagnostics."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

SAMPLED_LAGGED_MODE = "sampled_lagged"
SAMPLED_LAGGED_TIMING = "post_decision_after_backward"

_LER_FLOOR = 1e-8
_EMA_KEEP = 0.7
_EMA_NEW = 0.3


class SampledLaggedLERTracker:
    """Track lagged LER signals from a bounded deterministic parameter sample."""

    def __init__(
        self,
        task: str = "unknown",
        window_size: int = 5,
        parameter_sample_size: int = 4096,
        sample_seed: int = 0,
    ):
        if int(parameter_sample_size) < 1:
            raise ValueError("parameter_sample_size must be >= 1")
        if int(window_size) < 2:
            raise ValueError("window_size must be >= 2")
        self.task = task
        self.window_size = int(window_size)
        self.parameter_sample_size = int(parameter_sample_size)
        self.sample_seed = int(sample_seed)

        self.loss_history: List[float] = []
        self.entropy_history: List[float] = []
        self.ler_history: List[float] = []
        self.ler_raw_history: List[float] = []
        self.rho_vg_history: List[float] = []
        self.velocity_history: List[float] = []

        self._optimizer = None
        self._adam_eps_fallback = 1e-8
        self._sample_plan: Optional[List[Tuple[str, torch.Tensor]]] = None
        self._sample_index: Dict[str, torch.Tensor] = {}
        self._realized_sample_size = 0
        self._sampled_tensor_count = 0
        self._prev_samples: Optional[Dict[str, torch.Tensor]] = None

        self.n_updates = 0
        self.n_decisions = 0
        self.last_update_decision: Optional[int] = None
        self._decisions_at_last_update: Optional[int] = None
        self._last_noted_decision: Optional[int] = None

    def set_optimizer(self, optimizer) -> None:
        self._optimizer = optimizer

    def note_decision(self, decision_index: int) -> None:
        """Advance observation age without changing any signal."""
        self.n_decisions += 1
        self._last_noted_decision = int(decision_index)

    def update(
        self,
        loss: float,
        logits: torch.Tensor,
        accuracy: Optional[float] = None,
        model: Optional[torch.nn.Module] = None,
        optimizer=None,
        decision_index: Optional[int] = None,
    ) -> None:
        """Commit one observation after a successful real backward pass."""
        del accuracy
        opt = optimizer if optimizer is not None else self._optimizer
        self.loss_history.append(float(loss))
        self.entropy_history.append(self._compute_entropy(logits))

        velocity: Optional[float] = None
        rho: Optional[float] = None
        if model is not None:
            if self._sample_plan is None:
                self._build_sample_plan(model)
            if self._prev_samples is None:
                self._prev_samples = self._capture_samples(model)
            else:
                velocity, rho, new_samples = self._sampled_dynamics(model, opt)
                self._prev_samples = new_samples

        if velocity is not None:
            self.velocity_history.append(velocity)
        if rho is not None:
            self.rho_vg_history.append(rho)
        self._maybe_append_ler(velocity)

        self.n_updates += 1
        if decision_index is not None:
            self.last_update_decision = int(decision_index)
        elif self._last_noted_decision is not None:
            self.last_update_decision = self._last_noted_decision
        self._decisions_at_last_update = self.n_decisions

    def get_diagnostics(self) -> Dict:
        if self._decisions_at_last_update is None:
            observation_age = self.n_decisions
        else:
            observation_age = self.n_decisions - self._decisions_at_last_update
        return {
            "mode": SAMPLED_LAGGED_MODE,
            "timing": SAMPLED_LAGGED_TIMING,
            "parameter_sample_size_requested": self.parameter_sample_size,
            "parameter_sample_size_realized": self._realized_sample_size,
            "sampled_tensor_count": self._sampled_tensor_count,
            "n_updates": self.n_updates,
            "n_decisions": self.n_decisions,
            "last_update_decision": self.last_update_decision,
            "observation_age_decisions": observation_age,
            "ler": self.ler_history[-1] if self.ler_history else None,
            "ler_raw": self.ler_raw_history[-1] if self.ler_raw_history else None,
            "rho_vg": self._window_mean(self.rho_vg_history),
            "rho_vg_raw": self.rho_vg_history[-1] if self.rho_vg_history else None,
            "param_velocity": self._window_mean(self.velocity_history),
            "n_velocity_samples": len(self.velocity_history),
            "n_rho_vg_samples": len(self.rho_vg_history),
            "full_parameter_snapshots": False,
            "full_vector_concatenation": False,
        }

    def _build_sample_plan(self, model: torch.nn.Module) -> None:
        named = [
            (name, param.numel())
            for name, param in model.named_parameters()
            if param.requires_grad
        ]
        total = sum(numel for _, numel in named)
        budget = min(self.parameter_sample_size, total)
        generator = random.Random(self.sample_seed)

        plan: List[Tuple[str, torch.Tensor]] = []
        remaining_budget = budget
        remaining_total = total
        for name, numel in named:
            if remaining_budget <= 0 or numel == 0:
                remaining_total -= numel
                continue
            k = (remaining_budget * numel + remaining_total - 1) // remaining_total
            k = min(k, numel, remaining_budget)
            if k > 0:
                if k == numel:
                    indices = torch.arange(numel, dtype=torch.long)
                else:
                    selected = sorted(generator.sample(range(numel), k))
                    indices = torch.tensor(selected, dtype=torch.long)
                plan.append((name, indices))
                remaining_budget -= k
            remaining_total -= numel

        self._sample_plan = plan
        self._sample_index = {name: indices for name, indices in plan}
        self._realized_sample_size = sum(indices.numel() for _, indices in plan)
        self._sampled_tensor_count = len(plan)

    def _iter_sampled(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            indices = self._sample_index.get(name)
            if indices is None or not param.requires_grad:
                continue
            yield name, param, indices.to(device=param.device)

    def _capture_samples(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        samples: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, param, indices in self._iter_sampled(model):
                samples[name] = (
                    param.detach().reshape(-1).index_select(0, indices).float()
                )
        return samples

    def _sampled_signal(self, param, indices, state_lookup) -> Optional[torch.Tensor]:
        entry = state_lookup.get(id(param))
        if entry is not None:
            group, state = entry
            exp_avg = state.get("exp_avg")
            exp_avg_sq = state.get("exp_avg_sq")
            step = state.get("step", 0)
            step_value = float(step.item() if torch.is_tensor(step) else step)
            if exp_avg is not None and exp_avg_sq is not None and step_value > 0:
                beta1, beta2 = group.get("betas", (0.9, 0.999))
                eps = group.get("eps", self._adam_eps_fallback)
                avg_sample = (
                    exp_avg.detach().reshape(-1).index_select(0, indices).float()
                )
                sq_sample = (
                    exp_avg_sq.detach().reshape(-1).index_select(0, indices).float()
                )
                avg_hat = avg_sample / (1.0 - beta1 ** step_value)
                sq_hat = sq_sample / (1.0 - beta2 ** step_value)
                return -(avg_hat / (sq_hat.sqrt() + eps))
        if param.grad is not None:
            return param.grad.detach().reshape(-1).index_select(0, indices).float()
        return None

    def _sampled_dynamics(self, model, optimizer):
        state_lookup = {}
        if optimizer is not None:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    state_lookup[id(param)] = (
                        group,
                        optimizer.state.get(param, {}),
                    )

        new_samples: Dict[str, torch.Tensor] = {}
        delta_sq_sum = 0.0
        scalar_count = 0
        dot_sum = 0.0
        signal_sq_sum = 0.0
        velocity_sq_for_rho = 0.0
        has_signal = False

        with torch.no_grad():
            for name, param, indices in self._iter_sampled(model):
                current = (
                    param.detach().reshape(-1).index_select(0, indices).float()
                )
                new_samples[name] = current
                previous = self._prev_samples.get(name)
                if previous is None or previous.numel() != current.numel():
                    continue
                delta = current - previous
                delta_sq = float(delta.pow(2).sum().item())
                delta_sq_sum += delta_sq
                scalar_count += int(delta.numel())

                signal = self._sampled_signal(param, indices, state_lookup)
                if signal is not None:
                    has_signal = True
                    dot_sum += float((delta * signal).sum().item())
                    signal_sq_sum += float(signal.pow(2).sum().item())
                    velocity_sq_for_rho += delta_sq

        velocity = None
        if scalar_count > 0:
            velocity = (delta_sq_sum / scalar_count) ** 0.5

        rho = None
        if has_signal:
            denominator = velocity_sq_for_rho ** 0.5 * signal_sq_sum ** 0.5
            rho = 0.0 if denominator < 1e-12 else dot_sum / denominator
        return velocity, rho, new_samples

    @staticmethod
    def _compute_entropy(logits: torch.Tensor) -> float:
        detached = logits.detach()
        if detached.dim() >= 2 and detached.size(-1) > 1:
            probabilities = F.softmax(detached.float(), dim=-1)
            return float(
                -(
                    probabilities * torch.log(probabilities + 1e-10)
                ).sum(dim=-1).mean().item()
            )
        predictions = detached.float().flatten()
        if predictions.numel() > 1:
            spread = float(predictions.std().item()) / (
                abs(float(predictions.mean().item())) + 1e-6
            )
            return max(min(spread, 3.0), 0.05)
        return 0.1

    def _loss_gain(self) -> Optional[float]:
        if len(self.loss_history) < 2:
            return None
        window_size = min(self.window_size, len(self.loss_history))
        if window_size >= 4:
            half = window_size // 2
            older = self.loss_history[-window_size:-half]
            newer = self.loss_history[-half:]
            gain = sum(older) / len(older) - sum(newer) / len(newer)
        else:
            gain = self.loss_history[-2] - self.loss_history[-1]
        return max(float(gain), 0.0)

    def _maybe_append_ler(self, velocity: Optional[float]) -> None:
        gain = self._loss_gain()
        if gain is None:
            return
        entropy_window = self.entropy_history[-self.window_size:]
        average_entropy = sum(entropy_window) / len(entropy_window)
        if velocity is not None and velocity > 0:
            ler = velocity * gain * average_entropy
        else:
            ler = gain * average_entropy
        ler = max(float(ler), _LER_FLOOR)
        self.ler_raw_history.append(ler)
        if self.ler_history:
            ler = _EMA_KEEP * self.ler_history[-1] + _EMA_NEW * ler
        self.ler_history.append(ler)

    def _window_mean(self, history: List[float]) -> Optional[float]:
        if not history:
            return None
        recent = history[-self.window_size:]
        return float(sum(recent) / len(recent))
