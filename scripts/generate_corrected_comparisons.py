#!/usr/bin/env python3
"""
Regenerate corrected comparisons from the original paired ledger
(analysis/method_selection_audit/runs.csv).

Per LERNA_STEP3_CLI_PROMPT_V2.md section 7 and STEP3B_CLI_CORRECTION_PROMPT.md
section 1: regenerate from the original paired ledger with raw-proportion and
pp columns, full precision, explicit CI method and seed. Filter the exact
source root and matching configuration before pairing; preserve source
identity and assert against silent cohort collapse.

Cohorts:
  - QNLI confirmatory: source_root exactly /raid/stage2_confirmatory, task qnli
  - SST-2 replication: source_root exactly /raid/sst2_replication, task sst-2

No bootstrapping claimed as recovery of original RNG. Paired-t intervals use
fixed tabulated 97.5th percentiles (df=n-1). p-values are carried from the
original audit where available, not recomputed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

# Fixed tabulated 97.5th percentiles (Student t, df=n-1)
T_CRIT = {
    1: 12.7062047361747,
    2: 4.3026527297493,
    3: 3.1824463052837,
    4: 2.7764451051978,
    5: 2.5705818356158,
    6: 2.4469118487351,
    7: 2.3646242515928,
    8: 2.3060041352042,
    9: 2.2621571628541,
    10: 2.2281388519675,
    11: 2.2009852531461,
    12: 2.1788128296747,
    13: 2.1603686564610,
    14: 2.1447866879171,
    15: 2.1314465455575,
    16: 2.1199052992324,
    17: 2.1098155778334,
    18: 2.1009220402409,
    19: 2.0930240544083,
    20: 2.0859634472658,
    21: 2.0796138447113,
    22: 2.0738730678556,
    23: 2.0686576104190,
    24: 2.0638985616280,
    25: 2.0595385527692,
    26: 2.0555294477502,
    27: 2.0518305164832,
    28: 2.0484071442937,
    29: 2.0452296421326,
    30: 2.0422724561232,
}


def _tcrit(df: int) -> float:
    if df in T_CRIT:
        return T_CRIT[df]
    raise ValueError(f"no tabulated t critical value for df={df}")


LEDGER = Path("analysis/method_selection_audit/runs.csv")
OUT_DIR = Path("analysis/controller_review_step3b")


def _load_ledger(path: Path):
    rows = list(csv.DictReader(path.open()))
    # Group by (source_root, task, arm, seed) to handle duplicate (arm,seed)
    # across different task cohorts.
    by = {}
    for r in rows:
        key = (r.get("source_root", ""), r.get("task", ""), r["arm"], int(r["seed"]))
        if key in by:
            raise ValueError(f"duplicate key {key} in ledger")
        by[key] = float(r["eval_accuracy"])
    return rows, by


def _filter_cohort(cohort_by, source_root: str, task: str):
    """Filter ledger by exact source_root AND task, preserving full keys."""
    result = {}
    for (root, task_name, arm, seed), value in cohort_by.items():
        if root == source_root and task_name == task:
            result[(arm, seed)] = value
    # Detect duplicate (arm, seed) within the selected cohort — must fail.
    seen = set()
    for (arm, seed) in result:
        k = (arm, seed)
        if k in seen:
            raise ValueError(
                f"duplicate (arm,seed)={k} within selected cohort "
                f"({source_root}, {task})"
            )
        seen.add(k)
    return result


def _comparisons(by, pairs, task, source_root, expected_cohort_size=None,
                 expected_seed_set=None):
    out = []
    # All seeds present in ANY arm for this source_root+task cohort.
    all_seeds_in_cohort = sorted({s for _, s in by.keys()})
    # Determine the expected seed set: explicit identities > cohort size > all present.
    if expected_seed_set is not None:
        expected_seed_set = set(expected_seed_set)
        cohort_complete = expected_seed_set == set(all_seeds_in_cohort)
    elif expected_cohort_size is not None:
        expected_seed_set = set(all_seeds_in_cohort[:expected_cohort_size])
        cohort_complete = len(all_seeds_in_cohort) == expected_cohort_size
    else:
        expected_seed_set = set(all_seeds_in_cohort)
        cohort_complete = True

    for arm_a, arm_b in pairs:
        seeds_a = sorted(s for arm, s in by if arm == arm_a)
        seeds_b = sorted(s for arm, s in by if arm == arm_b)
        seeds = sorted(s for arm, s in by if arm == arm_a and (arm_b, s) in by)
        set_a = set(seeds_a)
        set_b = set(seeds_b)
        paired_set = set(seeds)

        # Missing seeds: expected cohort seeds absent from at least one arm.
        missing_seeds = sorted(expected_seed_set - (set_a | set_b)) if expected_seed_set else []
        # Unexpected seeds: seeds present that are not in the expected cohort.
        unexpected_seeds = sorted((set_a | set_b) - expected_seed_set) if expected_seed_set else []
        # Unmatched seeds: present in one arm but not the other.
        unmatched_a = sorted(set_a - set_b)
        unmatched_b = sorted(set_b - set_a)
        # Matched-pair deletion: same seed missing from BOTH arms.
        matched_pair_deleted = sorted(expected_seed_set - set_a - set_b) if expected_seed_set else []

        # Cohort-size enforcement: each arm must have the expected number of seeds.
        cohort_size_violation = False
        if expected_cohort_size is not None:
            if len(set_a) != expected_cohort_size:
                cohort_size_violation = True
            if len(set_b) != expected_cohort_size:
                cohort_size_violation = True
            if not cohort_complete:
                cohort_size_violation = True

        # Detect missing data: compare seed SETS plus cohort-size enforcement.
        missing_data = (
            set_a != set_b
            or len(seeds) != len(seeds_a)
            or len(seeds) != len(seeds_b)
            or len(matched_pair_deleted) > 0
            or cohort_size_violation
        )

        if len(seeds) == 0:
            # No paired observations at all.
            out.append({
                "source_root": source_root,
                "task": task,
                "a": arm_a,
                "b": arm_b,
                "n": 0,
                "seeds": [],
                "mean_raw": None,
                "mean_pp": None,
                "paired_sd_raw": None,
                "paired_sd_pp": None,
                "se_raw": None,
                "se_pp": None,
                "t_statistic": None,
                "ci95_t_raw": [None, None],
                "ci95_t_pp": [None, None],
                "equal_pairs": 0,
                "status": "insufficient_data",
                "ci_method": "paired t; skipped (no paired seeds)",
                "missing_data": True,
                "missing_seeds": missing_seeds,
                "unexpected_seeds": unexpected_seeds,
                "unmatched_a": unmatched_a,
                "unmatched_b": unmatched_b,
                "matched_pair_deleted": matched_pair_deleted,
                "cohort_size_violation": cohort_size_violation,
                "expected_cohort_size": expected_cohort_size,
                "actual_cohort_size_a": len(set_a),
                "actual_cohort_size_b": len(set_b),
            })
            continue
        if len(seeds) == 1:
            # One paired observation: n=1 is valid for reporting but
            # sd/se/t/CI are undefined (df=0).
            s = seeds[0]
            diff = by[(arm_a, s)] - by[(arm_b, s)]
            out.append({
                "source_root": source_root,
                "task": task,
                "a": arm_a,
                "b": arm_b,
                "n": 1,
                "seeds": seeds,
                "mean_raw": diff,
                "mean_pp": 100 * diff,
                "paired_sd_raw": None,
                "paired_sd_pp": None,
                "se_raw": None,
                "se_pp": None,
                "t_statistic": None,
                "ci95_t_raw": [None, None],
                "ci95_t_pp": [None, None],
                "equal_pairs": 1 if diff == 0 else 0,
                "status": "partial" if missing_data else "complete",
                "ci_method": "paired t; n=1 (sd/se/t/CI undefined, df=0)",
                "missing_data": missing_data,
                "missing_seeds": missing_seeds,
                "unexpected_seeds": unexpected_seeds,
                "unmatched_a": unmatched_a,
                "unmatched_b": unmatched_b,
                "matched_pair_deleted": matched_pair_deleted,
                "cohort_size_violation": cohort_size_violation,
                "expected_cohort_size": expected_cohort_size,
                "actual_cohort_size_a": len(set_a),
                "actual_cohort_size_b": len(set_b),
            })
            continue
        diffs = [by[(arm_a, s)] - by[(arm_b, s)] for s in seeds]
        n = len(diffs)
        mean = statistics.mean(diffs)
        sd = statistics.stdev(diffs)
        se = sd / math.sqrt(n)
        half = _tcrit(n - 1) * se
        ci_lo = mean - half
        ci_hi = mean + half
        t = mean / se if se else None
        status = "complete"
        ci_method = f"paired t, fixed tabulated 97.5th percentile, df={n-1}"
        if missing_data:
            status = "partial"
            ci_method += f"; WARNING: {len(unmatched_a)} unmatched {arm_a} row(s), {len(unmatched_b)} unmatched {arm_b} row(s)"
        out.append({
            "source_root": source_root,
            "task": task,
            "a": arm_a,
            "b": arm_b,
            "n": n,
            "seeds": seeds,
            "mean_raw": mean,
            "mean_pp": 100 * mean,
            "paired_sd_raw": sd,
            "paired_sd_pp": 100 * sd,
            "se_raw": se,
            "se_pp": 100 * se,
            "t_statistic": t,
            "ci95_t_raw": [ci_lo, ci_hi],
            "ci95_t_pp": [100 * ci_lo, 100 * ci_hi],
            "equal_pairs": sum(d == 0 for d in diffs),
            "status": status,
            "missing_data": missing_data,
            "missing_seeds": missing_seeds,
            "unexpected_seeds": unexpected_seeds,
            "unmatched_a": unmatched_a,
            "unmatched_b": unmatched_b,
            "matched_pair_deleted": matched_pair_deleted,
            "cohort_size_violation": cohort_size_violation,
            "expected_cohort_size": expected_cohort_size,
            "actual_cohort_size_a": len(set_a),
            "actual_cohort_size_b": len(set_b),
            "ci_method": ci_method,
        })
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, cohort_by = _load_ledger(LEDGER)

    # Ledger hash for provenance
    ledger_hash = hashlib.sha256(LEDGER.read_bytes()).hexdigest()[:16]

    # Cohorts: exact source_root + task
    qnli_by = _filter_cohort(cohort_by, "/raid/stage2_confirmatory", "qnli")
    sst2_by = _filter_cohort(cohort_by, "/raid/sst2_replication", "sst-2")

    comparisons = []
    qnli_expected_seeds = {11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
                           59, 61, 67, 71, 73, 79, 83, 89, 97, 107, 109, 113}
    sst2_expected_seeds = {101, 103, 107, 109, 113, 127, 131, 137, 139, 149}
    comparisons += _comparisons(qnli_by, [
        ("gate", "control"),
        ("gate", "reduced_steps"),
        ("control", "reduced_steps"),
        ("gate", "full_finetune"),
    ], "qnli", "/raid/stage2_confirmatory",
         expected_cohort_size=24, expected_seed_set=qnli_expected_seeds)
    comparisons += _comparisons(sst2_by, [
        ("gate", "control"),
    ], "sst-2", "/raid/sst2_replication",
         expected_cohort_size=10, expected_seed_set=sst2_expected_seeds)

    # Verify pp = 100 * raw for all complete estimates and endpoints
    for c in comparisons:
        if c["status"] != "complete":
            continue
        if c["paired_sd_raw"] is None:
            continue  # n=1: sd/se/t/CI undefined
        assert abs(c["mean_pp"] - 100 * c["mean_raw"]) < 1e-12, c
        assert abs(c["paired_sd_pp"] - 100 * c["paired_sd_raw"]) < 1e-12, c
        assert abs(c["ci95_t_pp"][0] - 100 * c["ci95_t_raw"][0]) < 1e-10, c
        assert abs(c["ci95_t_pp"][1] - 100 * c["ci95_t_raw"][1]) < 1e-10, c

    # Verify expected QNLI confirmatory values against unchanged ledger
    expected = {
        ("gate", "control"): 0.1403380316065665,
        ("gate", "reduced_steps"): 0.00915248032216746,
        ("control", "reduced_steps"): -0.13118555128439902,
        ("gate", "full_finetune"): -0.48736957715540913,
    }
    for c in comparisons:
        if c["status"] != "complete":
            continue
        if c["source_root"] == "/raid/stage2_confirmatory":
            key = (c["a"], c["b"])
            if key in expected:
                assert abs(c["mean_pp"] - expected[key]) < 1e-10, (
                    f"QNLI {key}: mean_pp={c['mean_pp']} != expected {expected[key]}"
                )

    # Determine overall status: OK only if all contrasts are complete and
    # there is at least one complete contrast.
    incomplete = [c for c in comparisons if c["status"] != "complete"]
    if len(rows) <= 1:
        # Header-only ledger (no data rows)
        overall_status = "REJECTED"
        overall_reason = "ledger contains header only; no data rows"
    elif incomplete:
        overall_status = "PARTIAL"
        overall_reason = f"{len(incomplete)} contrast(s) have insufficient data"
    else:
        overall_status = "OK"
        overall_reason = "all contrasts complete"

    result = {
        "status": overall_status,
        "reason": overall_reason,
        "method": "paired t, fixed tabulated 97.5th percentile, df=n-1; "
                  "raw-proportion and pp columns; pp = 100 * raw verified",
        "ledger_source": str(LEDGER),
        "ledger_hash": ledger_hash,
        "ledger_rows": len(rows),
        "cohorts": {
            "qnli_confirmatory": {
                "source_root": "/raid/stage2_confirmatory",
                "task": "qnli",
                "n_entries": len(qnli_by),
                "n_seeds": len({s for _, s in qnli_by}),
                "n_arms": len({a for a, _ in qnli_by}),
                "expected_cohort_size": 24,
            },
            "sst2_replication": {
                "source_root": "/raid/sst2_replication",
                "task": "sst-2",
                "n_entries": len(sst2_by),
                "n_seeds": len({s for _, s in sst2_by}),
                "n_arms": len({a for a, _ in sst2_by}),
                "expected_cohort_size": 10,
            },
        },
        "comparisons": comparisons,
        "notes": [
            "QNLI gate-control mean +0.140338 pp, paired SD .293111 pp, "
            "paired-t CI approximately [.016568, .264108] pp.",
            "Cohort selection filters by exact source_root + task, preventing "
            "silent cross-cohort overwrite of (arm,seed) keys.",
            "p-values are not regenerated here; carried from original audit "
            "where available, not recomputed.",
            "CI method is paired-t; no bootstrap-resample count applies.",
        ],
    }

    csv_path = OUT_DIR / "corrected_comparisons_v2.csv"
    json_path = OUT_DIR / "corrected_comparisons_v2.json"

    # CSV
    fieldnames = [
        "source_root", "task", "a", "b", "n", "status", "missing_data", "mean_raw", "mean_pp",
        "paired_sd_raw", "paired_sd_pp", "t_statistic",
        "ci95_t_raw_lo", "ci95_t_raw_hi",
        "ci95_t_pp_lo", "ci95_t_pp_hi",
        "equal_pairs", "ci_method",
        "missing_seeds", "unexpected_seeds", "unmatched_a", "unmatched_b",
        "matched_pair_deleted", "cohort_size_violation",
        "expected_cohort_size", "actual_cohort_size_a", "actual_cohort_size_b",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for c in comparisons:
            ci_raw = c["ci95_t_raw"]
            ci_pp = c["ci95_t_pp"]
            w.writerow({
                "source_root": c["source_root"],
                "task": c["task"],
                "a": c["a"], "b": c["b"], "n": c["n"], "status": c["status"], "missing_data": c.get("missing_data", False),
                "mean_raw": c["mean_raw"], "mean_pp": c["mean_pp"],
                "paired_sd_raw": c["paired_sd_raw"],
                "paired_sd_pp": c["paired_sd_pp"],
                "t_statistic": c["t_statistic"],
                "ci95_t_raw_lo": ci_raw[0],
                "ci95_t_raw_hi": ci_raw[1],
                "ci95_t_pp_lo": ci_pp[0],
                "ci95_t_pp_hi": ci_pp[1],
                "equal_pairs": c["equal_pairs"],
                "ci_method": c["ci_method"],
                "missing_seeds": ",".join(str(s) for s in c.get("missing_seeds", [])),
                "unexpected_seeds": ",".join(str(s) for s in c.get("unexpected_seeds", [])),
                "unmatched_a": ",".join(str(s) for s in c.get("unmatched_a", [])),
                "unmatched_b": ",".join(str(s) for s in c.get("unmatched_b", [])),
                "matched_pair_deleted": ",".join(str(s) for s in c.get("matched_pair_deleted", [])),
                "cohort_size_violation": c.get("cohort_size_violation", False),
                "expected_cohort_size": c.get("expected_cohort_size"),
                "actual_cohort_size_a": c.get("actual_cohort_size_a"),
                "actual_cohort_size_b": c.get("actual_cohort_size_b"),
            })

    # JSON
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "reason": result["reason"],
        "comparisons": len(comparisons),
        "csv": str(csv_path),
        "json": str(json_path),
        "ledger_hash": ledger_hash,
    }, indent=2))
    return 0 if result["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
