"""End-to-end and failure-path tests for the offline Phase 1.3 harness."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from lerna.trainers.policies import AlwaysFalsePolicy
from lerna.utils import run_provenance
from lerna.utils.phase1_3_completed_matrix import CompletedMatrixError
from lerna.utils.phase1_3_matrix import MatrixPlanError
from scripts import run_phase1_3_synthetic as synthetic

ARMS = (
    "full_finetune",
    "exact_random",
    "fixed_phase_strat",
    "phase_strat_guarded",
    "ler_guided_stratified",
    "ler_guided_stratified_safe",
)
ONLINE_ARMS = {
    "phase_strat_guarded",
    "ler_guided_stratified",
    "ler_guided_stratified_safe",
}
RATES = (0.30, 0.40)


def _load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _clean_git_state():
    return {
        "commit_sha": synthetic._resolve_git_sha(),
        "dirty": False,
        "tracked_changes": [],
        "untracked_paths": [],
    }


def _install_clean_provenance(monkeypatch) -> None:
    monkeypatch.setattr(synthetic, "collect_git_state", lambda *_args: _clean_git_state())
    monkeypatch.setattr(
        run_provenance,
        "collect_git_state",
        lambda *_args: _clean_git_state(),
    )


def _semantic_runs(root: Path):
    report = _load(root / "matrix_validation.json")
    semantic = []
    for run in report["runs"]:
        attempt = Path(run["attempt_dir"])
        results = _load(attempt / "results.json")
        instrumentation = _load(attempt / "instrumentation.json")
        eval_metrics = {
            key: value
            for key, value in results["eval_metrics"].items()
            if key
            not in {
                "eval_runtime",
                "eval_samples_per_second",
                "eval_steps_per_second",
            }
        }
        semantic.append(
            {
                "cell_id": run["cell_id"],
                "attempt": run["attempt"],
                "eval_metrics": eval_metrics,
                "train_loss": results["train_loss"],
                "forward_calls": instrumentation["forward_calls"],
                "backward_calls": instrumentation["backward_calls"],
                "skipped_backward_steps": instrumentation["skipped_backward_steps"],
                "optimizer_step_attempts": instrumentation[
                    "optimizer_step_attempts"
                ],
                "scheduler_step_calls": instrumentation["scheduler_step_calls"],
                "policy_diagnostics": results["policy_diagnostics"],
                "online_diagnostics": results["online_diagnostics"],
            }
        )
    return semantic


@pytest.fixture(scope="module")
def completed_matrices(tmp_path_factory):
    monkeypatch = pytest.MonkeyPatch()
    _install_clean_provenance(monkeypatch)
    roots = (
        tmp_path_factory.mktemp("phase1_3_synthetic_a") / "matrix",
        tmp_path_factory.mktemp("phase1_3_synthetic_b") / "matrix",
    )
    try:
        for root in roots:
            assert synthetic.main(["--output-dir", str(root), "--seed", "7"]) == 0
    finally:
        monkeypatch.undo()
    return roots


def _build_context(base_output_dir: Path):
    train_dataset = synthetic.SyntheticDataset(
        size=synthetic.SYNTHETIC_TRAIN_SIZE,
        width=synthetic.SYNTHETIC_WIDTH,
        seed=synthetic.SYNTHETIC_DATA_SEED,
    )
    eval_dataset = synthetic.SyntheticDataset(
        size=synthetic.SYNTHETIC_EVAL_SIZE,
        width=synthetic.SYNTHETIC_WIDTH,
        seed=synthetic.SYNTHETIC_DATA_SEED + 1,
    )
    facts = synthetic._data_facts(train_dataset, eval_dataset)
    plan = synthetic._plan_matrix(
        base_output_dir=str(base_output_dir),
        training_seed=7,
        facts=facts,
        git_sha=synthetic._resolve_git_sha(),
    )
    return plan, facts, train_dataset, eval_dataset


def _copy_completed_matrix(source: Path, target: Path):
    shutil.copytree(source, target)
    plan_path = target / "matrix_plan.json"
    plan = _load(plan_path)
    for cell in plan:
        cell["planned_arm_dir"] = str(
            target / cell["arm"] / cell["fingerprint"]
        )
    _write(plan_path, plan)
    return plan


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _first_attempt(root: Path) -> Path:
    return next((root / "full_finetune").glob("*/attempt-001"))


def _validate_matrix(plan, root: Path):
    return synthetic.validate_phase1_3_completed_matrix(
        plan,
        tasks=[synthetic.SYNTHETIC_TASK],
        seeds=[7],
        target_skip_rates=list(synthetic.STRICT_TARGET_SKIP_RATES),
        minimum_seed_count=1,
        base_output_dir=str(root),
    )


def test_real_matrix_has_exact_completed_contract(completed_matrices):
    root = completed_matrices[0]
    report = _load(root / "matrix_validation.json")
    assert report["status"] == "completed"
    assert report["plan_valid"] is True
    assert report["completed_matrix_valid"] is True
    assert report["n_cells"] == report["n_valid_runs"] == 12
    assert tuple(report["arms"]) == ARMS
    assert tuple(report["target_skip_rates"]) == RATES

    online_count = 0
    for index, run in enumerate(report["runs"]):
        rate = RATES[index // len(ARMS)]
        arm = ARMS[index % len(ARMS)]
        expected_skips = 0 if arm == "full_finetune" else round(rate * 100)
        expected_backward = 100 - expected_skips
        assert run["cell_id"] == [synthetic.SYNTHETIC_TASK, 7, rate, arm]
        assert run["attempt"] == 1
        assert run["backward_calls"] == expected_backward
        assert run["skipped_backward_steps"] == expected_skips
        assert run["valid_for_matched_budget"] is True

        attempt = Path(run["attempt_dir"])
        manifest = _load(attempt / "run_manifest.json")
        results = _load(attempt / "results.json")
        instrumentation = _load(attempt / "instrumentation.json")
        assert manifest["status"] == "completed"
        assert manifest["provenance_classification"] == "matched_claim"
        assert manifest["validation"]["ok"] is True
        assert manifest["validation"]["valid_for_matched_budget"] is True
        assert manifest["fingerprint"] == results["fingerprint"]
        assert manifest["attempt"] == results["attempt"] == 1
        assert results["task"] == synthetic.SYNTHETIC_TASK
        assert results["seed"] == 7
        assert results["ablation"] == arm
        assert results["model"] == synthetic.SYNTHETIC_MODEL_ID
        assert results["model_revision"] == synthetic.SYNTHETIC_MODEL_REVISION
        assert manifest["run"]["model_id"] == synthetic.SYNTHETIC_MODEL_ID
        assert manifest["run"]["model_revision"] == synthetic.SYNTHETIC_MODEL_REVISION
        assert instrumentation["forward_calls"] == 100
        assert instrumentation["backward_calls"] == expected_backward
        assert instrumentation["skipped_backward_steps"] == expected_skips
        assert instrumentation["optimizer_step_attempts"] == expected_backward
        assert instrumentation["scheduler_step_calls"] == expected_backward
        assert instrumentation["skip_update_mode"] == "freeze"
        assert (
            instrumentation["scheduler_step_policy"]
            == synthetic.SCHEDULER_STEP_POLICY
        )
        assert instrumentation["parameters_may_change_on_skipped_step"] is False
        has_ler_artifact = (attempt / "ler_diagnostics.json").is_file()
        assert has_ler_artifact == (arm in ONLINE_ARMS)
        online_count += int(has_ler_artifact)

    assert online_count == 6
    plan = _load(root / "matrix_plan.json")
    completed = _validate_matrix(plan, root)
    assert len(completed["valid_runs"]) == 12


def test_real_matrix_is_semantically_deterministic(completed_matrices):
    first, second = completed_matrices
    assert _semantic_runs(first) == _semantic_runs(second)


def test_nonempty_output_root_is_rejected_without_overwrite(
    tmp_path,
    monkeypatch,
):
    _install_clean_provenance(monkeypatch)
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("preserve me\n", encoding="utf-8")
    monkeypatch.setattr(
        synthetic,
        "_execute_cell",
        lambda **_kwargs: pytest.fail("training must not start"),
    )

    with pytest.raises(RuntimeError, match="absent or empty"):
        synthetic.main(["--output-dir", str(output), "--seed", "7"])

    assert sentinel.read_text(encoding="utf-8") == "preserve me\n"
    assert sorted(path.name for path in output.iterdir()) == ["sentinel.txt"]


def test_dirty_checkout_rejected_before_dataset_or_output_creation(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "dirty-output"
    dirty = _clean_git_state()
    dirty["dirty"] = True
    dirty["tracked_changes"] = ["tests/test_phase1_3_synthetic.py"]
    monkeypatch.setattr(synthetic, "collect_git_state", lambda *_args: dirty)
    monkeypatch.setattr(
        synthetic,
        "SyntheticDataset",
        lambda **_kwargs: pytest.fail("dataset construction must not start"),
    )

    with pytest.raises(RuntimeError, match="clean tracked tree"):
        synthetic.main(["--output-dir", str(output), "--seed", "7"])

    assert not output.exists()


def test_invalid_plan_rejected_before_output_creation(tmp_path, monkeypatch):
    _install_clean_provenance(monkeypatch)
    output = tmp_path / "invalid-plan"
    monkeypatch.setattr(synthetic, "build_phase1_3_matrix_plan", lambda **_kwargs: [])
    monkeypatch.setattr(
        synthetic,
        "_execute_cell",
        lambda **_kwargs: pytest.fail("training must not start"),
    )

    with pytest.raises(MatrixPlanError):
        synthetic.main(["--output-dir", str(output), "--seed", "7"])

    assert not output.exists()


def test_runtime_drift_rejected_before_attempt_model_or_trainer(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "runtime-drift"
    plan, facts, train_dataset, eval_dataset = _build_context(output)
    original_runtime_cell = synthetic._runtime_cell

    def drifted_runtime_cell(**kwargs):
        runtime = original_runtime_cell(**kwargs)
        runtime["fingerprint"] = "0" * 16
        return runtime

    monkeypatch.setattr(synthetic, "_runtime_cell", drifted_runtime_cell)
    monkeypatch.setattr(
        synthetic,
        "_allocate_attempt",
        lambda *_args: pytest.fail("attempt allocation must not occur"),
    )
    monkeypatch.setattr(
        synthetic,
        "TinyClassifier",
        lambda **_kwargs: pytest.fail("model construction must not occur"),
    )
    monkeypatch.setattr(
        synthetic,
        "AblationTrainer",
        lambda **_kwargs: pytest.fail("trainer construction must not occur"),
    )

    with pytest.raises(ValueError, match="Planned/runtime mismatch"):
        synthetic._execute_cell(
            cell=plan[0],
            facts=facts,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            git_sha=synthetic._resolve_git_sha(),
            base_output_dir=str(output),
        )

    assert not output.exists()


def test_trainer_failure_publishes_failed_manifest(tmp_path, monkeypatch):
    _install_clean_provenance(monkeypatch)
    output = tmp_path / "trainer-failure"
    plan, facts, train_dataset, eval_dataset = _build_context(output)

    class FailingTrainer:
        def __init__(self, **_kwargs):
            pass

        def train(self):
            raise RuntimeError("synthetic trainer failure")

    monkeypatch.setattr(synthetic, "AblationTrainer", FailingTrainer)
    with pytest.raises(RuntimeError, match="synthetic trainer failure"):
        synthetic._execute_cell(
            cell=plan[0],
            facts=facts,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            git_sha=synthetic._resolve_git_sha(),
            base_output_dir=str(output),
        )

    attempt = _first_attempt(output)
    manifest = _load(attempt / "run_manifest.json")
    assert manifest["status"] == "failed"
    assert "validation" not in manifest
    assert not (attempt / "results.json").exists()


def test_policy_decision_failure_never_publishes_completion(
    tmp_path,
    monkeypatch,
):
    _install_clean_provenance(monkeypatch)
    output = tmp_path / "policy-failure"
    plan, facts, train_dataset, eval_dataset = _build_context(output)

    def fail_policy(*_args, **_kwargs):
        raise RuntimeError("synthetic policy failure")

    monkeypatch.setattr(AlwaysFalsePolicy, "should_skip", fail_policy)
    with pytest.raises(RuntimeError):
        synthetic._execute_cell(
            cell=plan[0],
            facts=facts,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            git_sha=synthetic._resolve_git_sha(),
            base_output_dir=str(output),
        )

    manifest = _load(_first_attempt(output) / "run_manifest.json")
    assert manifest["status"] == "failed"
    assert "validation" not in manifest


def test_piece5_failure_prevents_completed_manifest(tmp_path, monkeypatch):
    _install_clean_provenance(monkeypatch)
    output = tmp_path / "piece5-failure"
    plan, facts, train_dataset, eval_dataset = _build_context(output)

    class RejectedReport:
        ok = False
        valid_for_matched_budget = False
        findings = ()

        @staticmethod
        def to_dict():
            return {
                "ok": False,
                "valid_for_matched_budget": False,
                "findings": [],
            }

    monkeypatch.setattr(
        synthetic,
        "validate_results",
        lambda *_args, **_kwargs: RejectedReport(),
    )
    with pytest.raises(RuntimeError, match="Piece 5 rejected"):
        synthetic._execute_cell(
            cell=plan[0],
            facts=facts,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            git_sha=synthetic._resolve_git_sha(),
            base_output_dir=str(output),
        )

    attempt = _first_attempt(output)
    manifest = _load(attempt / "run_manifest.json")
    assert manifest["status"] == "failed"
    assert "validation" not in manifest
    assert (attempt / "results.json").is_file()


@pytest.mark.parametrize(
    "corruption",
    (
        "hash_tamper",
        "missing_results",
        "running_manifest",
        "malformed_attempt",
        "duplicate_attempt",
        "mismatched_results",
        "malformed_manifest",
    ),
)
def test_completed_matrix_corruptions_are_rejected_read_only(
    completed_matrices,
    tmp_path,
    corruption,
):
    root = tmp_path / corruption
    plan = _copy_completed_matrix(completed_matrices[0], root)
    attempt = _first_attempt(root)
    manifest_path = attempt / "run_manifest.json"
    results_path = attempt / "results.json"

    if corruption == "hash_tamper":
        results_path.write_text(
            results_path.read_text(encoding="utf-8") + " ",
            encoding="utf-8",
        )
    elif corruption == "missing_results":
        results_path.unlink()
    elif corruption == "running_manifest":
        manifest = _load(manifest_path)
        manifest["status"] = "running"
        _write(manifest_path, manifest)
    elif corruption == "malformed_attempt":
        attempt.rename(attempt.with_name("attempt-1"))
    elif corruption == "duplicate_attempt":
        shutil.copytree(attempt, attempt.with_name("attempt-002"))
    elif corruption == "mismatched_results":
        results = _load(results_path)
        results["seed"] = 8
        _write(results_path, results)
    elif corruption == "malformed_manifest":
        _write(manifest_path, [])
    else:
        raise AssertionError(corruption)

    before = _tree_digest(root)
    with pytest.raises(CompletedMatrixError):
        _validate_matrix(plan, root)
    assert _tree_digest(root) == before


def test_harness_source_excludes_external_execution_paths():
    source = Path(synthetic.__file__).read_text(encoding="utf-8")
    for prohibited in (
        "load_dataset(",
        "from_pretrained(",
        "wandb.init(",
        "requests.",
        "http://",
        "https://",
        "MRPC",
        "/raid/hf_cache",
    ):
        assert prohibited not in source
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = ""' in source
    assert 'os.environ["WANDB_DISABLED"] = "true"' in source
    assert "if torch.cuda.is_available():" in source


def test_synthetic_model_revision_is_stable_local_identity():
    assert synthetic.SYNTHETIC_MODEL_REVISION == "synthetic-cpu-local-v1"
    assert synthetic.SYNTHETIC_MODEL_REVISION != "45d08642849e5c5701b162671ac811b7654bfd9f"
    assert synthetic.SYNTHETIC_MODEL_ID == "tiny-linear-cpu"
    assert synthetic.SYNTHETIC_MODEL_ID != "jhu-clsp/ettin-encoder-150m"
