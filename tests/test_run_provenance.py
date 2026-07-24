"""Dependency-light tests for Piece 6A run provenance and runner wiring."""

import ast
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "run_provenance",
    REPO_ROOT / "lerna" / "utils" / "run_provenance.py",
)
provenance = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(provenance)

CLASSIFICATION_LOCAL_DEVELOPMENT = provenance.CLASSIFICATION_LOCAL_DEVELOPMENT
CLASSIFICATION_MATCHED_CLAIM = provenance.CLASSIFICATION_MATCHED_CLAIM
MANIFEST_FILENAME = provenance.MANIFEST_FILENAME
MANIFEST_SCHEMA_VERSION = provenance.MANIFEST_SCHEMA_VERSION
ProvenanceError = provenance.ProvenanceError
classify_run = provenance.classify_run
finalize_manifest_completed = provenance.finalize_manifest_completed
finalize_manifest_failed = provenance.finalize_manifest_failed
load_manifest = provenance.load_manifest
manifest_path = provenance.manifest_path
write_manifest_running = provenance.write_manifest_running
FAKE_GIT_CLEAN = lambda: {
    "commit_sha": "abc123",
    "dirty": False,
    "tracked_changes": [],
    "untracked_paths": [".kilo/settings.json"],
}
FAKE_GIT_DIRTY = lambda: {
    "commit_sha": "abc123",
    "dirty": True,
    "tracked_changes": [" M lerna/trainers/policies.py"],
    "untracked_paths": [],
}
FAKE_VERSIONS = lambda: {
    "python": "3.11.0",
    "torch": "2.4.0",
    "transformers": "4.44.0",
    "cuda": "12.1",
    "device": "MockGPU",
}


class RunProvenanceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="lerna-provenance-test-")
        self.run_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_running(self, **overrides):
        kwargs = dict(
            argv=[
                "run_ablation_study.py",
                "--api-token",
                "ARGV_SECRET",
                "--password=INLINE_SECRET",
                "--mode",
                "smoke",
            ],
            task="sst2",
            model_id="modernbert",
            seed=42,
            controller_name="RandomSkipPolicy",
            controller_seed=42,
            target_skip_rate=0.3,
            planned_quota=30,
            total_steps=100,
            warmup_steps=10,
            skip_update_mode="freeze",
            controller_config_effective={
                "arm": "exact_random",
                "wandb_api_key": "CONFIG_SECRET",
            },
            matched_budget_planned=True,
            budget_classification="fixed_epoch",
            output_paths={"results": "results.json"},
            git_provider=FAKE_GIT_CLEAN,
            version_provider=FAKE_VERSIONS,
        )
        kwargs.update(overrides)
        return write_manifest_running(str(self.run_dir), **kwargs)

    def _write_artifacts(self):
        for name, content in {
            "results.json": '{"ok": true}',
            "instrumentation.json": '{"batches": 100}',
            "ler_diagnostics.json": '{"ler": 0.1}',
        }.items():
            (self.run_dir / name).write_text(content, encoding="utf-8")

    def test_running_manifest_written(self):
        manifest = self._write_running()
        self.assertEqual(load_manifest(str(self.run_dir)), manifest)
        self.assertEqual(manifest["schema_version"], MANIFEST_SCHEMA_VERSION)
        self.assertEqual(manifest["status"], "running")
        self.assertEqual(
            manifest["provenance_classification"],
            CLASSIFICATION_MATCHED_CLAIM,
        )
        self.assertEqual(manifest["run"]["planned_quota"], 30)
        self.assertIn("start_time_utc", manifest)

    def test_secrets_are_redacted_from_config_and_argv(self):
        self._write_running()
        raw = Path(manifest_path(str(self.run_dir))).read_text(encoding="utf-8")
        for secret in ("ARGV_SECRET", "INLINE_SECRET", "CONFIG_SECRET"):
            self.assertNotIn(secret, raw)
        self.assertIn("<redacted>", raw)

    def test_dirty_tree_refuses_matched_claim(self):
        with self.assertRaises(ProvenanceError):
            self._write_running(git_provider=FAKE_GIT_DIRTY)

    def test_early_stopping_refuses_matched_claim(self):
        with self.assertRaises(ProvenanceError):
            self._write_running(matched_budget_planned=False)

    def test_momentum_refuses_matched_claim(self):
        with self.assertRaises(ProvenanceError):
            self._write_running(skip_update_mode="momentum")

    def test_local_development_allows_dirty_unmatched_momentum(self):
        manifest = self._write_running(
            git_provider=FAKE_GIT_DIRTY,
            matched_budget_planned=False,
            skip_update_mode="momentum",
            requested_classification=CLASSIFICATION_LOCAL_DEVELOPMENT,
        )
        self.assertEqual(
            manifest["provenance_classification"],
            CLASSIFICATION_LOCAL_DEVELOPMENT,
        )

    def test_unknown_classification_rejected(self):
        with self.assertRaises(ProvenanceError):
            classify_run(
                git_dirty=False,
                matched_budget_planned=True,
                skip_update_mode="freeze",
                requested_classification="production",
            )

    def test_start_refuses_existing_manifest(self):
        self._write_running()
        with self.assertRaises(ProvenanceError):
            self._write_running()

    def test_start_refuses_stale_canonical_artifact(self):
        (self.run_dir / "results.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ProvenanceError):
            self._write_running()

    def test_completed_transition_hashes_artifacts(self):
        self._write_running()
        self._write_artifacts()
        manifest = finalize_manifest_completed(
            str(self.run_dir),
            realized_skips=28,
            realized_skip_rate=0.28,
            validation_status={
                "valid_for_matched_budget": True,
                "ok": True,
                "findings": [{
                    "severity": "info",
                    "field": "example",
                    "actual": "VALIDATION_SECRET",
                    "message": "token=VALIDATION_SECRET",
                }],
            },
        )
        self.assertEqual(manifest["status"], "completed")
        self.assertGreaterEqual(manifest["duration_seconds"], 0.0)
        self.assertEqual(
            manifest["realized"], {"skipped_steps": 28, "skip_rate": 0.28}
        )
        expected = hashlib.sha256(
            (self.run_dir / "results.json").read_bytes()
        ).hexdigest()
        self.assertEqual(manifest["artifacts"]["results.json"]["sha256"], expected)
        self.assertEqual(
            manifest["validation"]["findings"],
            [{"severity": "info", "field": "example"}],
        )
        self.assertNotIn("VALIDATION_SECRET", json.dumps(manifest))

    def test_matched_completion_requires_piece5_approval(self):
        self._write_running()
        self._write_artifacts()
        with self.assertRaises(ProvenanceError):
            finalize_manifest_completed(
                str(self.run_dir),
                validation_status={"valid_for_matched_budget": False},
            )
        self.assertEqual(load_manifest(str(self.run_dir))["status"], "running")

    def test_local_development_records_failed_validation(self):
        self._write_running(
            requested_classification=CLASSIFICATION_LOCAL_DEVELOPMENT
        )
        self._write_artifacts()
        manifest = finalize_manifest_completed(
            str(self.run_dir),
            validation_status={"valid_for_matched_budget": False, "ok": False},
        )
        self.assertEqual(manifest["status"], "completed")
        self.assertFalse(manifest["validation"]["valid_for_matched_budget"])

    def test_terminal_manifest_cannot_be_overwritten(self):
        self._write_running()
        self._write_artifacts()
        finalize_manifest_completed(
            str(self.run_dir),
            validation_status={"valid_for_matched_budget": True},
        )
        with self.assertRaises(ProvenanceError):
            finalize_manifest_failed(str(self.run_dir), RuntimeError("late"))

    def test_failure_redacts_exception_and_preserves_artifacts(self):
        self._write_running()
        partial = self.run_dir / "instrumentation.json"
        partial.write_text('{"partial": true}', encoding="utf-8")
        manifest = finalize_manifest_failed(
            str(self.run_dir), RuntimeError("api_key=EXCEPTION_SECRET")
        )
        raw = json.dumps(manifest)
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["error"]["type"], "RuntimeError")
        self.assertNotIn("EXCEPTION_SECRET", raw)
        self.assertTrue(partial.exists())
        self.assertTrue(manifest["artifacts"]["instrumentation.json"]["exists"])

    def test_atomic_publish_failure_keeps_running_manifest(self):
        self._write_running()
        before = load_manifest(str(self.run_dir))
        self._write_artifacts()
        with mock.patch("os.replace", side_effect=OSError("simulated crash")):
            with self.assertRaises(OSError):
                finalize_manifest_completed(
                    str(self.run_dir),
                    validation_status={"valid_for_matched_budget": True},
                )
        self.assertEqual(load_manifest(str(self.run_dir)), before)
        leftovers = [name for name in os.listdir(self.run_dir) if name.endswith(".tmp")]
        self.assertEqual(leftovers, [])


class RunnerWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner_path = REPO_ROOT / "scripts" / "run_ablation_study.py"
        cls.source = cls.runner_path.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source, filename=str(cls.runner_path))

    def test_runner_invokes_piece5_and_manifest_lifecycle(self):
        called_names = {
            node.func.id
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertTrue(
            {
                "write_manifest_running",
                "validate_skip_results",
                "finalize_manifest_completed",
                "finalize_manifest_failed",
            }.issubset(called_names)
        )

    def test_runner_catches_baseexception_for_terminal_failure(self):
        catches_base = any(
            isinstance(handler.type, ast.Name) and handler.type.id == "BaseException"
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Try)
            for handler in node.handlers
            if handler.type is not None
        )
        self.assertTrue(catches_base)

    def test_runner_exposes_explicit_classification_flag(self):
        self.assertIn("--provenance-classification", self.source)
        self.assertIn("provenance_classification=args.provenance_classification", self.source)


if __name__ == "__main__":
    unittest.main()
