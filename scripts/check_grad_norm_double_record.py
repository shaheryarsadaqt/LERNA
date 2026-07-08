#!/usr/bin/env python3
"""Read-only: check whether completed grad_norm runs double-recorded grad norms.
ratio = grad_norm_samples_collected / backward_calls
  ~2.0 -> pre-clip AND post-clip both recorded (M1 double-record)
  ~1.0 -> single-record
Usage: python scripts/check_grad_norm_double_record.py [dir]
"""
import json, sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/phase1_2_baselines/grad_norm")
ratios = []
for p in sorted(root.rglob("results.json")):
    try:
        d = json.loads(p.read_text())
    except Exception as e:
        print(f"skip {p}: {e}"); continue
    if d.get("baseline") != "grad_norm":
        continue
    s = (d.get("baseline_stats") or {}).get("grad_norm_samples_collected")
    b = d.get("backward_calls")
    if not s or not b:
        continue
    s, b = float(s), float(b)
    r = s / b
    ratios.append(r)
    print(f"{str(d.get('task')):<8} seed={d.get('seed')}  samples={s:.0f}  backward={b:.0f}  ratio={r:.3f}")

print("-" * 60)
if ratios:
    m = sum(ratios) / len(ratios)
    verdict = ("DOUBLE-RECORD -> keep as-is for consistency" if m > 1.5
               else "single-record" if m < 1.25
               else "AMBIGUOUS -> inspect individually")
    print(f"n_runs={len(ratios)}  mean_ratio={m:.3f}  => {verdict}")
else:
    print(f"No grad_norm results.json found under {root}")
