# Supervisor Sign-Off — LERNA v4 Pre-Registration

> **Decision framing.** Both hard gates from the draft brief are answered below by the
> stand-in (the lead author). The pre-registration may not be finalized, and no confirmatory
> or Stage-0 GPU cell may run, until this page bears the actual supervisor's or committee
> member's countersignature. The stand-in answers are binding on procedure only; they do not
> transfer authorship or override a later written veto.

---

## Q1 — Same line or new method?

**Answer:** Same line.

LER's objective — minimize forward loss subject to a backward-step budget; i.e. minimize
(loss ÷ compute) rather than loss alone — is unchanged from v3 / Selective Backprop
(Jiang et al., 2019) and the gradient-norm bound it uses (Katharopoulos & Fleuret,
2018). What moves is the **estimator**: v3 uses a backward-pass gradient-norm (LER),
v4 uses a forward-only proxy (the gradient-norm upper bound already used in Selective
Backprop), computed before the backward is spent. No new score or method is introduced.
The contribution v4 registers is a **matched, pre-registered comparison of a
forward-predicted-gain gate vs a matched random control**, under an identical budget
and identical strict (M2c) cell protocol. v3 LER results are retrospective context
with known confounds (lagged sampling, momentum carryover); they carry no confirmatory
claim in v4. The supervisor should treat "forward gate vs random control" as a
measurement, not an innovation claim.

## Q2 — Is a two-sided null acceptable?

**Answer:** Yes, provided the pre-registration is followed exactly and both sides of the
distribution are reported.

The primary test is a two-sided α = 0.05 paired t-test (df = n−1). Separately, a
pre-registered TOST equivalence bound (two one-sided tests, α = 0.05 each) is reported as
a secondary analysis to bound the effect size in the event of a null. The Directionality
statement is fixed in writing before the first Stage-1 cell runs, and neither the null nor
a significant result is selectively emphasized. This is consistent with Kaddour et al.
(NeurIPS 2023), in which the batch-selection gain vanished against a baseline whose learning
rate had fully decayed — i.e. a null is a publishable, non-trivial outcome, not a failure
of the protocol. Accepting a two-sided null does **not** accept "no signal exists"
without the power bound; the thesis must still report the upper bound on effect at n seeds
(Path B wording is mandatory).

## Effective gates

| Gate | Answer | Effective? |
|---|---|---|
| Same line vs new method | Same line (forward gate vs random control) | Yes |
| Two-sided null acceptable | Yes, with pre-reg fidelity + paired TOST reporting | Yes |

---

**Stand-in (lead author):** ______________________  date: 2026-10-04  
**Real supervisor / committee (required before thesis submission):** ______________________  date: ___________
