# Gate A Decision — LER informativeness oracle test

Date:            2026-10-04 (template prepared; Stage 0 NOT YET RUN — no numbers populated)
Runner SHA:      v4-prereg (tag at commit 93839a1)
Score source:    S1 (||p-y||) / S2 (alignment) — selected by Stage 0
Selected score:  PENDING Stage 0 GPU oracle probe (shadow mode, seeds 101/103)

## Stage 0 oracle probe (seeds 101, 103 — disjoint from Stage 2)

Run type: full_finetune shadow mode (no skipping, backward every step)
Probe P_s: 256 train examples (for S2 gradient computation)
Probe P_e: 256 train examples, disjoint from P_s (for realized gain measurement)
Probe exclusion: asserted (P_s ∩ train_sampler == ∅, P_e ∩ train_sampler == ∅)
Log interval: every 8 steps
S2 probe refresh: every 64 steps (fp32, eval mode)

### Seed 101 — S1 (forward gain bound)
  local Spearman ρ(S1, gain on P_e | 64-step window): ____ (list per window)
  bootstrap 95% CI: [____, ____]
  eligible (ρ ≥ 0.10, CI above 0): YES / NO

### Seed 101 — S2 (last-layer alignment)
  local Spearman ρ(S2, gain on P_e | 64-step window): ____ (list per window)
  bootstrap 95% CI: [____, ____]
  eligible (ρ ≥ 0.10, CI above 0): YES / NO

### Seed 103 — S1 (forward gain bound)
  local Spearman ρ(S1, gain on P_e | 64-step window): ____ (list per window)
  bootstrap 95% CI: [____, ____]
  eligible (ρ ≥ 0.10, CI above 0): YES / NO

### Seed 103 — S2 (last-layer alignment)
  local Spearman ρ(S2, gain on P_e | 64-step window): ____ (list per window)
  bootstrap 95% CI: [____, ____]
  eligible (ρ ≥ 0.10, CI above 0): YES / NO

### Selection (fixed rule)
  Per-seed ρ values: S1 seed101=____  S1 seed103=____  S2 seed101=____  S2 seed103=____
  Higher S1 ρ: ____  |  Higher S2 ρ: ____
  Selected score: S1 / S2  (ties → S1)

## DECISION

  Selected score: S1 / S2

This score is carried into Stage 2 (rate 0.60, 24 seeds). Stage 0 is a
descriptive/mechanism probe; it never cancels Stage 2 regardless of outcome.
