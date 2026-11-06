"""Step 3C close-out: regression tests for three behaviours that no other test pins.

The Step 3C acceptance review (bundle 817502b4) verified each of these by
running the code, and found no test for any of them:

1. Direct exact-gate seed validation. phase_exact_v1 rejects fractional,
   boolean and non-finite seeds; legacy_v4 keeps its int() conversion.
2. Output-path overwrite refusal. allocate_output_dir() raises before
   anything is written, and main() --dry-run returns REJECTED
   output_path_exists and leaves the first manifest unchanged.
3. The non-dry immutable-revision preflight. ExperimentConfig(dry_run=False)
   rejects branch names, short hashes, upper case and whitespace in every
   revision field; dry-run does not enforce it.

CPU only. No model, no Trainer, no dataset, no download. main() is called
with --dry-run only.
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from lerna.utils.forward_gate import ScoreQuantileGate
from scripts.run_controller_ablation import ExperimentConfig, allocate_output_dir, main

_BASE = dict(task="sst-2", arm="gate_s1_p1", controller_version="phase_exact_v1", score_source="s1",
             n_phases=1, total_steps=20, min_step=2, target_skip_rate=0.5)
_PINS = dict(model_revision="a" * 40, tokenizer_revision="b" * 40, dataset_revision="c" * 40)


def _gate(seed, version):
    return ScoreQuantileGate(rate=0.5, total_steps=20, min_step=2, n_phases=1, max_consecutive_skips=4,
                             seed=seed, score_source=lambda *args: 0.5, controller_version=version)


class TestExactGateSeedValidation:
    @pytest.mark.parametrize("seed, exc", [(11.9, ValueError), (True, TypeError),
                                           (float("nan"), ValueError), (float("inf"), ValueError)],
                             ids=["fractional", "bool", "nan", "inf"])
    def test_exact_rejects_bad_seed(self, seed, exc):
        with pytest.raises(exc):
            _gate(seed, "phase_exact_v1")

    @pytest.mark.parametrize("seed", [12, 12.0])
    def test_exact_accepts_integral_seed(self, seed):
        gate = _gate(seed, "phase_exact_v1")
        assert gate.seed == 12 and type(gate.seed) is int

    def test_legacy_keeps_int_conversion(self):
        assert _gate(11.9, "legacy_v4").seed == 11
        assert _gate(True, "legacy_v4").seed == 1


class TestOverwriteRefusal:
    def test_allocate_output_dir_refuses_existing_path(self, tmp_path):
        cfg = ExperimentConfig(**_BASE, dry_run=True, output_dir=str(tmp_path), attempt_id="same")
        out = allocate_output_dir(cfg)
        (out / "marker.txt").write_text("first")
        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            allocate_output_dir(cfg)
        assert (out / "marker.txt").read_text() == "first"
        assert [p for p in tmp_path.rglob("*") if p.is_file()] == [out / "marker.txt"]

    def test_dry_run_main_rejects_second_run_and_keeps_first_manifest(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", [
            "run_controller_ablation.py", "--dry-run", "--task", "sst-2", "--arm", "gate_s1_p1",
            "--controller-version", "phase_exact_v1", "--score-source", "s1", "--total-steps", "20",
            "--min-step", "2", "--n-phases", "1", "--skip-rate", "0.5",
            "--output-dir", str(tmp_path), "--attempt-id", "same"])
        assert main() == 0
        assert json.loads(capsys.readouterr().out)["status"] == "OK"
        manifests = list(tmp_path.rglob("dry_run_manifest.json"))
        assert len(manifests) == 1
        before = manifests[0].read_bytes()
        assert main() == 1
        second = json.loads(capsys.readouterr().out)
        assert second["status"] == "REJECTED" and second["reason"] == "output_path_exists"
        assert second["provenance"]["execution_mode"] == "plan_only"
        assert manifests[0].read_bytes() == before
        assert [p for p in tmp_path.rglob("*") if p.is_file()] == manifests


class TestNonDryRevisionPreflight:
    @pytest.mark.parametrize("field", ["model_revision", "tokenizer_revision", "dataset_revision"])
    @pytest.mark.parametrize("bad", ["main", "abc1234", "A" * 40, " " + "a" * 39],
                             ids=["branch_main", "short_sha", "upper_case", "leading_space"])
    def test_non_dry_rejects_unpinned_revision(self, field, bad, tmp_path):
        kwargs = dict(_PINS, **{field: bad})
        with pytest.raises(ValueError, match="Real execution blocked") as info:
            ExperimentConfig(**_BASE, **kwargs, dry_run=False, output_dir=str(tmp_path))
        assert field in str(info.value)

    def test_non_dry_accepts_full_sha_pins(self, tmp_path):
        cfg = ExperimentConfig(**_BASE, **_PINS, dry_run=False, output_dir=str(tmp_path))
        assert (cfg.model_revision, cfg.tokenizer_revision, cfg.dataset_revision) == ("a" * 40, "b" * 40, "c" * 40)

    def test_dry_run_does_not_enforce_pins(self, tmp_path):
        cfg = ExperimentConfig(**_BASE, dry_run=True, output_dir=str(tmp_path))
        assert cfg.model_revision == "main"
