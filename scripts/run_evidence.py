"""Run terminal contract evidence generation and write results to files."""
import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVIDENCE = os.path.join(BASE, "review_evidence")
os.makedirs(EVIDENCE, exist_ok=True)

def run(cmd, label):
    log = os.path.join(EVIDENCE, f"{label}.log")
    with open(log, "w") as f:
        r = subprocess.run(cmd, cwd=BASE, stdout=f, stderr=subprocess.STDOUT, text=True)
    return r.returncode, log

# 1. Run focused pytest
code, log = run([sys.executable, "-m", "pytest", "tests/test_terminal_invalid_observation.py", "-v"], "focused_terminal_tests")
with open(os.path.join(EVIDENCE, "focused_terminal_tests.summary.json"), "w") as f:
    json.dump({"exit_code": code, "log": os.path.basename(log)}, f, indent=2)

# 2. Generate coherent fixture
import torch
import torch.nn as nn
from lerna.utils.lagged_ler import SampledLaggedLERTracker, TerminalInvalidObservation

model = nn.Linear(10, 2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
tracker = SampledLaggedLERTracker(task="fixture", window_size=5, parameter_sample_size=100, sample_seed=42)
tracker.set_optimizer(optimizer)

logits = torch.randn(4, 2)
losses = [0.5, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43]
for i, loss in enumerate(losses):
    tracker.update(loss=loss, logits=logits, model=model, optimizer=optimizer, decision_index=i)

try:
    tracker.update(loss=float("nan"), logits=logits, model=model, optimizer=optimizer, decision_index=7)
except TerminalInvalidObservation:
    pass

diag = tracker.get_diagnostics()
fixture = {
    "mode": diag["mode"],
    "timing": diag["timing"],
    "parameter_sample_size_requested": diag["parameter_sample_size_requested"],
    "parameter_sample_size_realized": diag["parameter_sample_size_realized"],
    "sampled_tensor_count": diag["sampled_tensor_count"],
    "n_updates": diag["n_updates"],
    "n_decisions": diag["n_decisions"],
    "last_update_decision": diag["last_update_decision"],
    "observation_age_decisions": diag["observation_age_decisions"],
    "ler": diag["ler"],
    "ler_raw": diag["ler_raw"],
    "rho_vg": diag["rho_vg"],
    "rho_vg_raw": diag["rho_vg_raw"],
    "param_velocity": diag["param_velocity"],
    "n_velocity_samples": diag["n_velocity_samples"],
    "n_rho_vg_samples": diag["n_rho_vg_samples"],
    "full_parameter_snapshots": diag["full_parameter_snapshots"],
    "full_vector_concatenation": diag["full_vector_concatenation"],
    "observation_attempts": diag["observation_attempts"],
    "warmup_updates": diag["warmup_updates"],
    "valid_emissions": diag["valid_emissions"],
    "invalid_emission_events": diag["invalid_emission_events"],
    "terminal_invalid_event": diag["terminal_invalid_event"],
    "last_valid_emission_decision": diag["last_valid_emission_decision"],
    "ler_raw_history": diag["ler_raw_history"],
    "ler_history": diag["ler_history"],
    "velocity_history": diag["velocity_history"],
    "rho_vg_history": diag["rho_vg_history"],
    "loss_history": diag["loss_history"],
    "entropy_history": diag["entropy_history"],
}
with open(os.path.join(EVIDENCE, "coherent_fixture.json"), "w") as f:
    json.dump(fixture, f, indent=2)

# Verify fixture consistency
checks = {
    "attempt_accounting": fixture["observation_attempts"] == fixture["warmup_updates"] + fixture["valid_emissions"] + fixture["invalid_emission_events"],
    "final_values_preserved": fixture["ler"] is not None and fixture["ler_raw"] is not None,
    "rho_vg_raw_matches_last_history": fixture["rho_vg_raw"] == (fixture["rho_vg_history"][-1] if fixture["rho_vg_history"] else None),
    "terminal_event_attempt": fixture["terminal_invalid_event"]["observation_attempt"] == fixture["observation_attempts"],
    "last_valid_emission_decision": fixture["last_valid_emission_decision"] == 6,
}
# Verify EMA recurrence
raw = fixture["ler_raw_history"]
ema = fixture["ler_history"]
expected_ema = []
for i, r in enumerate(raw):
    if i == 0:
        expected_ema.append(r)
    else:
        expected_ema.append(0.7 * expected_ema[-1] + 0.3 * r)
checks["ema_recurrence"] = len(expected_ema) == len(ema) and all(abs(a - b) < 1e-9 for a, b in zip(expected_ema, ema))
with open(os.path.join(EVIDENCE, "coherent_fixture_checks.json"), "w") as f:
    json.dump(checks, f, indent=2)

# 3. Mutation tests
mutation_results = {}

# Mutation 1: Disable loss guard
import lerna.utils.lagged_ler as ll
with open(ll.__file__) as f:
    src = f.read()
mutated_src = src.replace("if not math.isfinite(loss_val):", "if False:  # MUTATION")
m1_dir = os.path.join(EVIDENCE, "mutation1")
os.makedirs(m1_dir, exist_ok=True)
m1_path = os.path.join(m1_dir, "lagged_ler_mut.py")
with open(m1_path, "w") as f:
    f.write(mutated_src)

import importlib.util
spec = importlib.util.spec_from_file_location("lagged_ler_mut", m1_path)
mut_ll = importlib.util.module_from_spec(spec)
sys.modules["lagged_ler_mut"] = mut_ll
spec.loader.exec_module(mut_ll)

try:
    m_model = nn.Linear(10, 2)
    m_opt = torch.optim.AdamW(m_model.parameters(), lr=1e-3)
    m_tracker = mut_ll.SampledLaggedLERTracker(task="m1", window_size=5, parameter_sample_size=100, sample_seed=42)
    m_tracker.set_optimizer(m_opt)
    m_logits = torch.randn(4, 2)
    try:
        m_tracker.update(loss=float("nan"), logits=m_logits, model=m_model, optimizer=m_opt, decision_index=0)
        mutation_results["mutation1_disable_loss_guard"] = {"mutation_survived": True, "detail": "NaN accepted without raising"}
    except mut_ll.TerminalInvalidObservation:
        mutation_results["mutation1_disable_loss_guard"] = {"mutation_survived": False, "detail": "Guard still active"}
except Exception as e:
    mutation_results["mutation1_disable_loss_guard"] = {"mutation_survived": True, "detail": f"Error: {type(e).__name__}: {e}"}

# Mutation 2: Disconnect signal validation in policy
import lerna.trainers.policies as pol
with open(pol.__file__) as f:
    pol_src = f.read()
mutated_pol = pol_src.replace(
    "raise TerminalInvalidPolicyError(",
    "return False  # MUTATION: disconnected\n        raise TerminalInvalidPolicyError(",
)
m2_dir = os.path.join(EVIDENCE, "mutation2")
os.makedirs(m2_dir, exist_ok=True)
m2_path = os.path.join(m2_dir, "policies_mut.py")
with open(m2_path, "w") as f:
    f.write(mutated_pol)

spec2 = importlib.util.spec_from_file_location("policies_mut", m2_path)
mut_pol = importlib.util.module_from_spec(spec2)
sys.modules["policies_mut"] = mut_pol
spec2.loader.exec_module(mut_pol)

try:
    from lerna.utils.lagged_ler import SampledLaggedLERTracker as SLT, TerminalInvalidObservation as TIO
    m_model = nn.Linear(10, 2)
    m_opt = torch.optim.AdamW(m_model.parameters(), lr=1e-3)
    m_tracker = SLT(task="m2", window_size=5, parameter_sample_size=100, sample_seed=42)
    m_tracker.set_optimizer(m_opt)
    m_logits = torch.randn(4, 2)
    try:
        m_tracker.update(loss=float("nan"), logits=m_logits, model=m_model, optimizer=m_opt, decision_index=0)
    except TIO:
        pass

    m_policy = mut_pol.LERGuidedStratifiedPolicy(
        ler_tracker=m_tracker, min_step=0, target_skip_rate=0.2,
        total_steps=100, probe_interval=8, ler_guidance_strength=1.0,
    )
    from unittest.mock import MagicMock
    mock_trainer = MagicMock()
    mock_trainer.state.global_step = 1
    mock_trainer._pre_clip_grad_norm = 0.5
    try:
        result = m_policy.should_skip(mock_trainer, m_model, {"input_ids": torch.randn(2, 10), "labels": torch.tensor([0, 1])})
        mutation_results["mutation2_disconnect_signal_validation"] = {"mutation_survived": True, "detail": f"Returned {result} instead of raising"}
    except mut_pol.TerminalInvalidPolicyError:
        mutation_results["mutation2_disconnect_signal_validation"] = {"mutation_survived": False, "detail": "Still raises"}
except Exception as e:
    mutation_results["mutation2_disconnect_signal_validation"] = {"mutation_survived": True, "detail": f"Error: {type(e).__name__}: {e}"}

with open(os.path.join(EVIDENCE, "mutation_results.json"), "w") as f:
    json.dump(mutation_results, f, indent=2)

# 4. Trainer evidence
from lerna.trainers.true_skip_trainer import (
    TrueBackwardSkippingTrainer, ObservationFailureError, ONLINE_LER_MODE_SAMPLED_LAGGED,
)
from lerna.utils.run_provenance import write_manifest_running, finalize_manifest_failed
from transformers import TrainingArguments
import tempfile, shutil

temp_dir = tempfile.mkdtemp(prefix="terminal_evidence_")
try:
    class TerminalLossModel(nn.Module):
        def __init__(self, fail_at_step=1):
            super().__init__()
            self.linear = nn.Linear(10, 2)
            self.step = 0
            self.fail_at_step = fail_at_step
        def forward(self, input_ids, labels=None, **kwargs):
            self.step += 1
            logits = self.linear(input_ids)
            loss = None
            if labels is not None:
                if self.step >= self.fail_at_step:
                    loss = torch.tensor(float("nan"), device=input_ids.device)
                else:
                    loss = torch.nn.functional.cross_entropy(logits, labels)
            return type("Outputs", (), {"loss": loss, "logits": logits})()

    model = TerminalLossModel(fail_at_step=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = TrainingArguments(
        output_dir=temp_dir, per_device_train_batch_size=2, max_steps=10,
        logging_steps=1, report_to="none", save_strategy="no",
        remove_unused_columns=False, gradient_accumulation_steps=1,
    )
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self): return 4
        def __getitem__(self, idx):
            return {"input_ids": torch.randn(10), "labels": torch.tensor(idx % 2)}

    trainer = TrueBackwardSkippingTrainer(
        model=model, args=args, train_dataset=DummyDataset(), optimizers=(optimizer, None),
    )
    trainer.online_ler_mode = ONLINE_LER_MODE_SAMPLED_LAGGED
    trainer.online_ler_update_interval = 1
    trainer.capture_logits = True
    tracker = SampledLaggedLERTracker(task="evidence", window_size=5, parameter_sample_size=100, sample_seed=42)
    tracker.set_optimizer(optimizer)
    trainer.ler_tracker = tracker

    opt_count = [0]
    sched_count = [0]
    scaler_count = [0]
    orig_opt = optimizer.step
    def spy_opt(*a, **kw):
        opt_count[0] += 1
        return orig_opt(*a, **kw)
    optimizer.step = spy_opt
    if trainer.lr_scheduler is not None:
        orig_sched = trainer.lr_scheduler.step
        def spy_sched(*a, **kw):
            sched_count[0] += 1
            return orig_sched(*a, **kw)
        trainer.lr_scheduler.step = spy_sched
    if hasattr(trainer.accelerator, "scaler") and trainer.accelerator.scaler is not None:
        orig_scaler = trainer.accelerator.scaler.step
        def spy_scaler(*a, **kw):
            scaler_count[0] += 1
            return orig_scaler(*a, **kw)
        trainer.accelerator.scaler.step = spy_scaler

    write_manifest_running(
        temp_dir, argv=["python", "-m", "pytest"], task="evidence", model_id="test-model",
        model_revision="rev1", seed=42, controller_name="ler_guided_stratified",
        controller_seed=42, target_skip_rate=0.2, planned_quota=20, total_steps=100,
        warmup_steps=50, skip_update_mode="freeze",
        controller_config_effective={"policy_class": "LERGuidedStratifiedPolicy"},
        matched_budget_planned=True, budget_classification="matched",
        output_paths={"results": "results.json", "instrumentation": "instrumentation.json", "ler_diagnostics": "ler_diagnostics.json"},
        requested_classification="local_development",
        identity_inputs={"task": "evidence", "seed": 42}, fingerprint="a" * 16, attempt=1,
    )

    try:
        trainer.train()
        error = None
    except ObservationFailureError as e:
        error = e

    evidence = {
        "optimizer_step_count": opt_count[0],
        "scaler_step_count": scaler_count[0],
        "scheduler_step_count": sched_count[0],
        "error_type": type(error).__name__ if error else None,
        "error_cause_type": type(error.__cause__).__name__ if error and error.__cause__ else None,
        "tracker_event": error.tracker_event if error else None,
    }
    for name in ["instrumentation.json", "ler_diagnostics.json"]:
        p = os.path.join(temp_dir, name)
        if os.path.exists(p):
            with open(p) as f:
                evidence[name] = json.load(f)
    if error is not None:
        finalize_manifest_failed(temp_dir, error)
        p = os.path.join(temp_dir, "run_manifest.json")
        if os.path.exists(p):
            with open(p) as f:
                evidence["failed_manifest"] = json.load(f)
    with open(os.path.join(EVIDENCE, "trainer_evidence.json"), "w") as f:
        json.dump(evidence, f, indent=2)
finally:
    shutil.rmtree(temp_dir)

print("EVIDENCE GENERATED")