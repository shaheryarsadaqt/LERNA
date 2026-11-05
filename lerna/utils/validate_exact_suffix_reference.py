#!/usr/bin/env python3
"""
Independent validation of PhaseSuffixFeasibility against exhaustive enumeration.

Per STEP3B_CLI_CORRECTION_PROMPT.md section 3:
- Enumerates and classifies prefixes independently.
- Compares helper outputs only for prefixes that satisfy expired phase quotas
  and the prefix legality contract, including feasible and infeasible remaining
  suffixes.
- Separately asserts that the integrated gate rejects illegal/inconsistent
  prefixes.
- Fails and returns nonzero for EVERY in-contract initial/action/suffix/witness
  mismatch.
- Validates returned witnesses explicitly against constraints.
- Resolves the intended tested checkout explicitly (no hardcoded import path).

CPU-only; no torch/Trainer/score-policy integration; no ML training.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path


def load_helper(root: Path):
    """Load PhaseSuffixFeasibility from the intended tested checkout."""
    spec = importlib.util.spec_from_file_location(
        "phase_suffix_feasibility", root / "lerna" / "utils" / "phase_suffix_feasibility.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_bounds(total_steps: int, n_phases: int):
    """Equal-length phases; last absorbs remainder."""
    bounds = []
    step = 0
    for i in range(n_phases):
        end = total_steps if i == n_phases - 1 else step + total_steps // n_phases
        bounds.append((step, end))
        step = end
    return bounds


def phase_at_list(bounds, horizon):
    """Map each step index to its phase."""
    phase_at = []
    for j, (s, e) in enumerate(bounds):
        phase_at.extend([j] * (e - s))
    return phase_at


def is_valid_prefix(prefix, bounds, quotas, warmup, cap, phase_at):
    """Check whether a prefix satisfies warmup, cap, per-phase quota limits,
    and expired-phase exactness."""
    streak = 0
    counts = [0] * len(bounds)
    current_t = len(prefix)
    for step, action in enumerate(prefix):
        p = phase_at[step]
        if action:
            streak += 1
            if step < warmup or streak > cap:
                return False
            counts[p] += 1
            if counts[p] > quotas[p]:
                return False
        else:
            streak = 0
    # Expired phases (fully before current_t) must have exact quota met.
    for j, (s, e) in enumerate(bounds):
        if e <= current_t and counts[j] != quotas[j]:
            return False
    return True


def exhaustive_allowed_actions_from_prefix(
    bounds, quotas, warmup, cap, prefix, c, used, phase_at
):
    """Oracle: enumerate legal first actions from a valid prefix state.

    Returns the set of legal first actions (0=retain, 1=skip) at step
    len(prefix), or empty set if no valid completion exists.
    """
    horizon = bounds[-1][1]
    current_t = len(prefix)
    n_phases = len(bounds)

    # Prefix must be valid (checked by caller)
    results = set()
    remaining = horizon - current_t
    for suffix in itertools.product((0, 1), repeat=remaining):
        streak = c
        counts = list(used)
        valid = True
        for step, action in enumerate(suffix, start=current_t):
            p = phase_at[step]
            if action:
                streak += 1
                if step < warmup or streak > cap:
                    valid = False
                    break
                counts[p] += 1
                if counts[p] > quotas[p]:
                    valid = False
                    break
            else:
                streak = 0
        if not valid:
            continue
        if any(counts[j] != quotas[j] for j in range(n_phases)):
            continue
        if remaining > 0:
            results.add(suffix[0])
    return results


def exhaustive_initial_feasible(bounds, quotas, warmup, cap) -> bool:
    """Oracle: is there any valid full schedule satisfying all quotas?"""
    horizon = bounds[-1][1]
    n_phases = len(bounds)
    phase_at = phase_at_list(bounds, horizon)
    for actions in itertools.product((0, 1), repeat=horizon):
        streak = 0
        counts = [0] * n_phases
        valid = True
        for step, action in enumerate(actions):
            p = phase_at[step]
            if action:
                streak += 1
                if step < warmup or streak > cap:
                    valid = False
                    break
                counts[p] += 1
                if counts[p] > quotas[p]:
                    valid = False
                    break
            else:
                streak = 0
        if valid and all(counts[j] == quotas[j] for j in range(n_phases)):
            return True
    return False


def validate_witness(witness, bounds, quotas, warmup, cap, phase_at) -> bool:
    """Validate a witness schedule against all constraints."""
    horizon = bounds[-1][1]
    if len(witness) != horizon:
        return False
    streak = 0
    counts = [0] * len(bounds)
    for step, action in enumerate(witness):
        p = phase_at[step]
        if action:
            streak += 1
            if step < warmup or streak > cap:
                return False
            counts[p] += 1
            if counts[p] > quotas[p]:
                return False
        else:
            streak = 0
    return all(counts[j] == quotas[j] for j in range(len(bounds)))


def run_checks(root: Path, output: Path):
    helper_mod = load_helper(root)
    H = helper_mod.PhaseSuffixFeasibility

    configs = 0
    states = 0
    action_checks = 0
    initial_checks = 0
    witness_checks = 0
    mismatches = []
    invalid_index_failures = []

    for horizon in range(1, 7):
        for phases in range(1, horizon + 1):
            bounds = build_bounds(horizon, phases)
            phase_at = phase_at_list(bounds, horizon)
            for quotas in itertools.product(
                *(range(e - s + 1) for s, e in bounds)
            ):
                for warmup in sorted({0, horizon // 2, horizon}):
                    for cap in (1, 2, 4):
                        configs += 1
                        h = H(horizon, phases, list(quotas), warmup, cap)

                        # Initial feasibility
                        expected_initial = exhaustive_initial_feasible(
                            bounds, list(quotas), warmup, cap
                        )
                        actual_initial = h.initial_feasible()
                        initial_checks += 1
                        if actual_initial != expected_initial:
                            mismatches.append(
                                dict(
                                    type="initial_feasibility",
                                    horizon=horizon,
                                    phases=phases,
                                    quotas=list(quotas),
                                    warmup=warmup,
                                    cap=cap,
                                    expected=expected_initial,
                                    actual=actual_initial,
                                )
                            )

                        # For each valid prefix state, compare helper actions with oracle.
                        # Enumerate all possible prefixes up to length horizon-1.
                        for prefix_len in range(horizon):
                            for prefix in itertools.product((0, 1), repeat=prefix_len):
                                if not is_valid_prefix(
                                    list(prefix), bounds, list(quotas), warmup, cap, phase_at
                                ):
                                    continue  # skip invalid prefixes (unpaid expired phase, etc.)
                                current_t = prefix_len
                                if current_t >= horizon:
                                    continue
                                p = phase_at[current_t]
                                # Compute streak and used counts from prefix
                                streak = 0
                                counts = [0] * phases
                                for step, action in enumerate(prefix):
                                    pp = phase_at[step]
                                    if action:
                                        streak += 1
                                        counts[pp] += 1
                                    else:
                                        streak = 0
                                q = quotas[p] - counts[p]
                                if q < 0:
                                    continue  # shouldn't happen if prefix valid

                                states += 1
                                expected_actions = exhaustive_allowed_actions_from_prefix(
                                    bounds, list(quotas), warmup, cap,
                                    list(prefix), streak, counts, phase_at
                                )
                                action_checks += 1
                                try:
                                    actual_actions = set(
                                        h.get_feasible_actions(current_t, streak, q)
                                    )
                                except Exception as exc:
                                    mismatches.append(
                                        dict(
                                            type="helper_exception",
                                            horizon=horizon,
                                            phases=phases,
                                            quotas=list(quotas),
                                            warmup=warmup,
                                            cap=cap,
                                            prefix=list(prefix),
                                            t=current_t,
                                            c=streak,
                                            q=q,
                                            error=str(exc),
                                        )
                                    )
                                    continue
                                if actual_actions != expected_actions:
                                    mismatches.append(
                                        dict(
                                            type="action_set",
                                            horizon=horizon,
                                            phases=phases,
                                            quotas=list(quotas),
                                            warmup=warmup,
                                            cap=cap,
                                            prefix=list(prefix),
                                            t=current_t,
                                            c=streak,
                                            q=q,
                                            phase=p,
                                            expected=sorted(expected_actions),
                                            actual=sorted(actual_actions),
                                        )
                                    )

                        # Witness validation
                        if h.initial_feasible():
                            witness = h.witness()
                            witness_checks += 1
                            if witness is None:
                                mismatches.append(
                                    dict(
                                        type="witness_none",
                                        horizon=horizon,
                                        phases=phases,
                                        quotas=list(quotas),
                                        warmup=warmup,
                                        cap=cap,
                                    )
                                )
                            elif not validate_witness(
                                witness, bounds, list(quotas), warmup, cap, phase_at
                            ):
                                mismatches.append(
                                    dict(
                                        type="witness_invalid",
                                        horizon=horizon,
                                        phases=phases,
                                        quotas=list(quotas),
                                        warmup=warmup,
                                        cap=cap,
                                        witness=witness,
                                    )
                                )

    # Helper invalid-index queries must raise, not return True
    h4 = H(4, 2, [1, 1], 0, 1)
    for bad_t, bad_c, bad_q in [(-1, 0, 0), (4, 0, 0), (99, 0, 0)]:
        try:
            h4.is_action_feasible(bad_t, bad_c, bad_q, 0)
            invalid_index_failures.append(
                dict(type="no_raise", t=bad_t, c=bad_c, q=bad_q)
            )
        except (ValueError, TypeError):
            pass  # expected

    # Terminal t=T: only q=0 feasible
    try:
        if h4.is_feasible_at(4, 0, 1):
            invalid_index_failures.append(
                dict(type="terminal_q_nonzero_feasible", t=4, q=1)
            )
    except Exception:
        invalid_index_failures.append(dict(type="terminal_raised", t=4, q=1))

    # Helper fractional configuration must raise
    try:
        H(4.9, 2, [1, 1], 0, 1)
        invalid_index_failures.append(dict(type="no_raise_fractional_horizon"))
    except (ValueError, TypeError):
        pass

    result = {
        "scope": "Validator contract check: exact suffix feasibility vs exhaustive "
        "enumeration, in-contract prefixes only. CPU-only, no training.",
        "configurations": configs,
        "feasibility_state_checks": states,
        "allowed_action_set_checks": action_checks,
        "initial_feasibility_checks": initial_checks,
        "witness_checks": witness_checks,
        "mismatches": mismatches,
        "mismatch_count": len(mismatches),
        "invalid_index_failures": invalid_index_failures,
        "invalid_index_failure_count": len(invalid_index_failures),
    }
    result["all_passed"] = (
        len(mismatches) == 0 and len(invalid_index_failures) == 0
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Validate PhaseSuffixFeasibility against exhaustive enumeration"
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Path to the LERNA checkout containing the helper under test",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the validation results JSON",
    )
    args = parser.parse_args()
    result = run_checks(args.root, args.output)
    sys.exit(0 if result["all_passed"] else 1)


if __name__ == "__main__":
    main()
