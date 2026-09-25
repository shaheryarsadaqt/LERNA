"""Deterministic sampled, post-decision, lagged LER diagnostics."""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

SAMPLED_LAGGED_MODE = "sampled_lagged"
SAMPLED_LAGGED_TIMING = "post_decision_after_backward"

_LER_FLOOR = 1e-8
_EMA_KEEP = 0.7
_EMA_NEW = 0.3


class TerminalInvalidObservation(Exception):
    """Raised when a sampled observation contains non-finite values.

    This is a terminal event for the attempt. The attempt must stop before
    the pending optimizer/scheduler action.
    """
    def __init__(self, event_type: str, decision_index: int, operands: Dict[str, Any], valid_prefix: Dict[str, Any]):
        self.event_type = event_type
        self.decision_index = decision_index
        self.operands = operands
        self.valid_prefix = valid_prefix
        super().__init__(f"Terminal invalid observation: {event_type} at decision {decision_index}")


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

        # Terminal contract counters
        self.observation_attempts = 0          # Every update() call
        self.warmup_updates = 0                # Finite update without LER emission (insufficient history)
        self.valid_emissions = 0               # Successfully emitted LER (finite)
        self.invalid_emission_events = 0       # Non-finite inputs rejected
        self.terminal_invalid_event = None     # Dict with event details, or None
        self.last_valid_emission_decision: Optional[int] = None  # Zero-based decision index
        self.terminal_forensics = None         # Forensics captured at first terminal event

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
        """Commit one observation after a successful real backward pass.

        Validates all inputs before committing any state. On first non-finite
        observation, raises TerminalInvalidObservation with the valid prefix
        preserved and invalid operands recorded.

        Once a terminal invalid event is recorded, the tracker enters an
        absorbing terminal state: all subsequent updates immediately raise
        TerminalInvalidObservation without modifying state.
        """
        del accuracy
        opt = optimizer if optimizer is not None else self._optimizer

        # Absorbing terminal state: if already terminal, re-raise immediately
        if self.terminal_invalid_event is not None:
            raise TerminalInvalidObservation(
                "terminal_invalid_already_recorded",
                decision_index if decision_index is not None else self._last_noted_decision or 0,
                {"note": "tracker already in terminal state"},
                self._get_valid_prefix()
            )

        # Increment attempt counter at start
        self.observation_attempts += 1

        # --- Phase 1: Compute all values without committing state ---
        # Validate loss
        loss_val = float(loss)
        if not math.isfinite(loss_val):
            tagged = {"loss": self._make_tagged_operand(loss_val)}
            self._capture_forensics_best_effort("nonfinite_loss", decision_index, loss_val, logits)
            self._record_terminal_invalid("nonfinite_loss", decision_index, tagged)
            raise TerminalInvalidObservation(
                "nonfinite_loss",
                decision_index if decision_index is not None else self._last_noted_decision or 0,
                tagged,
                self._get_valid_prefix()
            )

        # Compute and validate entropy
        entropy = self._compute_entropy(logits)
        if not math.isfinite(entropy):
            tagged = {"entropy": self._make_tagged_operand(entropy)}
            self._capture_forensics_best_effort("nonfinite_entropy", decision_index, loss_val, logits)
            self._record_terminal_invalid("nonfinite_entropy", decision_index, tagged)
            raise TerminalInvalidObservation(
                "nonfinite_entropy",
                decision_index if decision_index is not None else self._last_noted_decision or 0,
                tagged,
                self._get_valid_prefix()
            )

        # Compute velocity and rho (but don't commit yet)
        velocity: Optional[float] = None
        rho: Optional[float] = None
        new_samples = None
        candidate_prev_samples = None
        if model is not None:
            if self._sample_plan is None:
                self._build_sample_plan(model)
            if self._prev_samples is None:
                # First update with model: capture initial samples, no dynamics yet
                self._prev_samples = self._capture_samples(model)
            else:
                velocity, rho, new_samples = self._sampled_dynamics(model, opt)
                candidate_prev_samples = new_samples
                # Validate velocity and rho before committing
                if velocity is not None and not math.isfinite(velocity):
                    tagged = {"velocity": self._make_tagged_operand(velocity)}
                    self._capture_forensics_best_effort("nonfinite_velocity", decision_index, loss_val, logits)
                    self._record_terminal_invalid("nonfinite_velocity", decision_index, tagged)
                    raise TerminalInvalidObservation(
                        "nonfinite_velocity",
                        decision_index if decision_index is not None else self._last_noted_decision or 0,
                        tagged,
                        self._get_valid_prefix()
                    )
                if rho is not None and not math.isfinite(rho):
                    tagged = {"rho": self._make_tagged_operand(rho)}
                    self._capture_forensics_best_effort("nonfinite_rho", decision_index, loss_val, logits)
                    self._record_terminal_invalid("nonfinite_rho", decision_index, tagged)
                    raise TerminalInvalidObservation(
                        "nonfinite_rho",
                        decision_index if decision_index is not None else self._last_noted_decision or 0,
                        tagged,
                        self._get_valid_prefix()
                    )

        # Pre-validate LER emission (would be emitted if sufficient history)
        # We need to simulate _maybe_append_ler logic to check if LER would be non-finite
        # This ensures atomic commit: either all histories updated or none
        # Check if AFTER appending this loss/entropy, there would be sufficient history
        loss_history_len_after = len(self.loss_history) + 1
        # Need at least 2 losses for gain calculation
        will_emit_ler = loss_history_len_after >= 2
        if will_emit_ler:
            # Compute gain as it would be after append (matching _loss_gain logic)
            if loss_history_len_after >= 4:
                window_size = min(self.window_size, loss_history_len_after)
                half = window_size // 2
                older_start = loss_history_len_after - window_size
                older_end = loss_history_len_after - half
                # Build older and newer windows as _loss_gain would
                older_losses = list(self.loss_history[older_start:older_end])
                newer_losses = list(self.loss_history[older_end:]) + [loss_val]
                # If the new loss_val falls in older window, adjust
                if older_end == loss_history_len_after:
                    older_losses.append(loss_val)
                    newer_losses = list(self.loss_history[older_end:])
                gain = (sum(older_losses) / len(older_losses)) - (sum(newer_losses) / len(newer_losses))
            else:
                # Simple difference between last two losses
                if len(self.loss_history) >= 1:
                    gain = self.loss_history[-1] - loss_val
                else:
                    gain = 0.0
            gain = max(float(gain), 0.0)
            
            # Entropy window: take up to window_size entries from history + new entropy
            entropy_window = list(self.entropy_history[-self.window_size:]) + [entropy]
            average_entropy = sum(entropy_window) / len(entropy_window)
            if velocity is not None and velocity > 0:
                ler_candidate = velocity * gain * average_entropy
            else:
                ler_candidate = gain * average_entropy
            if not math.isfinite(ler_candidate):
                tagged = {"ler": self._make_tagged_operand(ler_candidate)}
                self._record_terminal_invalid("nonfinite_ler", decision_index, tagged)
                raise TerminalInvalidObservation(
                    "nonfinite_ler",
                    decision_index if decision_index is not None else self._last_noted_decision or 0,
                    tagged,
                    self._get_valid_prefix()
                )
            ler_candidate = max(float(ler_candidate), _LER_FLOOR)

        # --- Phase 2: All validations passed, commit atomically ---
        self.loss_history.append(loss_val)
        self.entropy_history.append(entropy)

        if candidate_prev_samples is not None:
            self._prev_samples = candidate_prev_samples
        if velocity is not None:
            self.velocity_history.append(velocity)
        if rho is not None:
            self.rho_vg_history.append(rho)

        # Actually emit LER if we would
        if will_emit_ler:
            self.ler_raw_history.append(ler_candidate)
            if self.ler_history:
                ler_candidate = _EMA_KEEP * self.ler_history[-1] + _EMA_NEW * ler_candidate
            self.ler_history.append(ler_candidate)
            emitted_ler = True
        else:
            emitted_ler = False

        # Update counters
        self.n_updates += 1
        if decision_index is not None:
            self.last_update_decision = int(decision_index)
        elif self._last_noted_decision is not None:
            self.last_update_decision = self._last_noted_decision
        self._decisions_at_last_update = self.n_decisions

        # Track whether we emitted a valid LER.
        if emitted_ler:
            self.valid_emissions += 1
            if self.last_update_decision is not None:
                self.last_valid_emission_decision = self.last_update_decision
        else:
            self.warmup_updates += 1

    def _get_valid_prefix(self) -> Dict[str, Any]:
        """Return the finite accepted prefix for failure evidence."""
        return {
            "observation_attempts": self.observation_attempts,
            "warmup_updates": self.warmup_updates,
            "valid_emissions": self.valid_emissions,
            "ler_raw_history": list(self.ler_raw_history),
            "ler_history": list(self.ler_history),
            "velocity_history": list(self.velocity_history),
            "rho_vg_history": list(self.rho_vg_history),
            "loss_history": list(self.loss_history),
            "entropy_history": list(self.entropy_history),
        }

    def _make_tagged_operand(self, value: float) -> Dict[str, str]:
        """Create tagged operand for terminal contract (kind='nonfinite', value='NaN'|'+Infinity'|'-Infinity')."""
        return {"kind": "nonfinite", "value": "NaN" if math.isnan(value) else ("+Infinity" if value > 0 else "-Infinity")}

    def _record_terminal_invalid(self, event_type: str, decision_index: Optional[int], operands: Dict[str, Any]) -> None:
        """Record terminal invalid event (only first one)."""
        if self.terminal_invalid_event is None:
            self.invalid_emission_events += 1
            self.terminal_invalid_event = {
                "event_type": event_type,
                "decision_index": decision_index if decision_index is not None else self._last_noted_decision,
                "observation_attempt": self.observation_attempts,
                "operands": operands,
            }

    def _capture_forensics_best_effort(self, event_type: str, decision_index: Optional[int], loss_val: float, logits: Optional[torch.Tensor]) -> None:
        """Capture forensics at first terminal event (best effort, never raises)."""
        if self.terminal_forensics is not None:
            return
        info = {
            "event_type": event_type,
            "decision_index": decision_index if decision_index is not None else self._last_noted_decision,
            "observation_attempt": self.observation_attempts,
            "loss_repr": repr(loss_val),
        }
        try:
            if logits is not None:
                t = logits.detach()
                f = t.float()
                fin = torch.isfinite(f)
                info["logits"] = {
                    "dtype": str(t.dtype),
                    "shape": list(t.shape),
                    "n_nan": int(torch.isnan(f).sum()),
                    "n_posinf": int(torch.isposinf(f).sum()),
                    "n_neginf": int(torch.isneginf(f).sum()),
                    "max_abs_finite": float(f[fin].abs().max()) if bool(fin.any()) else None,
                }
        except Exception as exc:
            info["forensics_error"] = f"{type(exc).__name__}: {exc}"
        self.terminal_forensics = info

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
            # Terminal contract fields
            "observation_attempts": self.observation_attempts,
            "warmup_updates": self.warmup_updates,
            "valid_emissions": self.valid_emissions,
            "invalid_emission_events": self.invalid_emission_events,
            "terminal_invalid_event": self.terminal_invalid_event,
            "last_valid_emission_decision": self.last_valid_emission_decision,
            "ler_raw_history": list(self.ler_raw_history),
            "ler_history": list(self.ler_history),
            "velocity_history": list(self.velocity_history),
            "rho_vg_history": list(self.rho_vg_history),
            "loss_history": list(self.loss_history),
            "entropy_history": list(self.entropy_history),
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
            if math.isfinite(denominator) and denominator >= 1e-12:
                rho = dot_sum / denominator
            else:
                rho = 0.0
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

    def _window_mean(self, history: List[float]) -> Optional[float]:
        if not history:
            return None
        recent = history[-self.window_size:]
        return float(sum(recent) / len(recent))