# Tool-selection vs. deepening — RL reward shape and the breadth ceiling

Status: **PROPOSED · step-1 harness built and self-tested (synthetic + 2-row
real-schema smoke), not yet run on the full dump** · direct-to-main

Provenance: filed text (owner + chat Claude, 2026-11-20) · harness and
extensions (chat DeepSeek, 2026-11-20) · schema verification + harness repair
R1–R7 (chat Kimi, 2026-11-20). Repairs are listed in the repair record below;
nothing was edited silently.

## Question

Does RL that rewards *selecting a reasoning tool* when uncertainty rises
preserve the base model's breadth, while RL that rewards *the answer* narrows it?

## Sources (OBSERVED — papers read 2026-11-20; numbers cited, not yet re-derived)

| claim | source | status |
|---|---|---|
| "aha" self-reflection emerges in R1-Zero without reflection RL | Guo et al. 2025, *DeepSeek-R1*, arXiv 2501.12948 | OBSERVED (abs page read) |
| base models already self-reflect; self-reflection frequency not correlated with accuracy in R1-Zero | Liu et al. 2025, Sea AI Lab, arXiv 2503.20783 — DeepSeek-V3-Base, 500 MATH-train questions (§2.3, App. F) | OBSERVED (HTML read) |
| spontaneous shifts in ~6.31% of traces; do not rise with training; seldom help | d'Aliberti & Horta Ribeiro, *The Illusion of Insight in Reasoning Models*, arXiv 2601.00514v2 (Princeton; v2 2026-04-19; ACL 2026 Findings) | OBSERVED (v2 HTML read) |
| entropy-triggered *external* reconsideration: +8.41 pp on MATH-500, smaller on crosswords/Rush Hour | same, RQ3 | OBSERVED (cited) |
| scope of the above: Qwen2.5-1.5B/7B + Llama-3.1-8B, GRPO; crosswords, MATH-500, Rush Hour | same | OBSERVED (cited) |

Precisions on the pooled 6.31%: per-domain shift rates spread an order of
magnitude (RHour 14.32%, Math 2.65%, Xwords 1.22%, temperature-pooled);
shifted traces 2.57% correct vs 16.44% non-shifted (N=723,200). Any re-check
reports per-domain rates; the pooled figure hides the heterogeneity.

None of these numbers are re-derived in this repo yet. Step 1 is the
independent read; until it runs, they are *cited*, not *reproduced*.
"Independent" here means an independent re-analysis of the authors' released
traces — the same data, not independent data collection.

## Arms (PROPOSED — step 2, no GPU on this owner)

- **A** — base model, no RL.
- **B** — answer-reward RL (standard GRPO; reward = correctness).
- **C** — selection-reward RL; reward = `correctness + λ · 1[tool chosen
  correctly at an entropy spike]`. Tool set declared up front:
  `substitution / series / numeric step / re-parse`.

## Measures

- `breadth` — pass@k, k = 1 … 256 (unbiased estimator, Chen et al. 2021:
  `1 − C(n−c,k)/C(n,k)`, n = 256 samples/problem at T = 1.0).
- `selection` — tool-choice accuracy at entropy spikes (arm C).
- `shift quality` — share of shifts that (a) change the final answer and
  (b) land on the correct one.

## Prediction (PROPOSED)

B rises at pass@1 and falls **below A** at high k. C keeps A's high-k
ceiling and improves selection.

**Refuted if** C narrows like B, or selection does not improve.

## Pre-registered thresholds (committed 2026-11-20, before any step-2 number)

- **"C narrows like B"** is refuted if, at k=256,
  `pass@256(C) ≥ 0.8 · pass@256(A)` while `pass@256(B) < 0.8 · pass@256(A)`,
  each with a problem-level bootstrap 95% CI excluding the threshold.
- **"selection doesn't improve"** is refuted if the C−B difference in
  tool-choice accuracy at p90 entropy spikes has a 95% CI with upper
  bound ≤ 0.
- **Step 1.5 (free, on the released dump):** the *cited* narrowing claim is
  independently supported only if, with base and trained arms both present,
  `pass@k(trained) < pass@k(base)` with CI excluding 0. On this dump the
  contrast is **step 0 (initialization) vs final step**, k ≤ 8 (see the
  OBSERVED limits below). If a cell lacks the samples, it stays
  `not run` — the cited claim is then unreproduced, not refuted.

## Arm C's oracle problem (open; blocks step 2)

Arm C rewards "the right tool chosen at an entropy spike." No ground-truth
tool label exists. Three operationalizations, with their failure modes:

1. **Outcome-conditional** — choice is "correct" iff the post-spike
   continuation reaches gold. *Fails:* this is answer reward with extra
   steps, and should narrow exactly like B — the very collapse C is meant
   to avoid. Using it would beg the question.
2. **Counterfactual** — apply the chosen tool to the problem state;
   correct iff the answer moves toward gold relative to not applying it.
   *Fails:* needs a per-tool executor (all four are implementable for
   MATH-500). Most faithful; most work.
3. **Verifier-scored** — a separate small model scores appropriateness.
   *Fails:* inherits the verifier's biases.

Decision needed before step 2. Default if none chosen: **(2)**, because
(1) cannot distinguish C from B by construction.

*The glyph addendum in NOURISHMENT.md forces this: if perception-assertion
is the object, outcome-conditional (1) is disqualified by construction and
(2) counterfactual is the only faithful operationalization.*

## Step 1 — independent re-check (stdlib, runnable now)

Target: Princeton released traces + code
(`github.com/humans-and-machines/Illusion-of-Reasoning`,
`HF od2961/illusion-of-reasoning-main-traces`; 1,235,792 rows per the rows
API, 2026-11-20).

Re-derive, without the authors' code in the loop:

1. shift rate,
2. accuracy given a shift vs. without,
3. entropy-gated gain.

Different lab from the authors = the first independent read.

### OBSERVED dump schema (rows API, 2026-11-20) — supersedes guessing

Per row: `model` (GRPO-1.5B / GRPO-7B / GRPO-Llama8B), `task`
(math / xword / carpark), `step`, `temperature` ∈ {0, 0.05, 0.3, 0.7},
`sample_idx` (G = 8 samples per config), `split`, `problem` / `example_id` /
`enumeration`, `gold_answer*`. `pass1`/`pass2` are JSON strings; each
carries `output`, `entropy`, `entropy_think`, `entropy_answer`,
`is_correct_pred`, `is_valid_pred`, `has_reconsider_cue`,
`_shift_prefilter_markers`, `_shift_prefilter_pos`, `pred_answer*`,
`soft_reward*`, `tokens_*`, `valid_tag_structure`.

- **The authors' judge labels ship in the release**: pass1 carries
  `shift_in_reasoning_v1`, `shift_markers_v1`, `shift_first_marker_char`,
  `shift_before/after_excerpt`, `shift_rationale_gpt{,_model,_time}`
  (gpt-4o). *Correction to this morning's verification pass, which read only
  the card preview and reported judge labels as absent/UNCHECKED — the card
  truncates the field list; the rows API shows them present.*
- `pass2` carries `improved_over_pass1`, `prev_output` (= pass1 output),
  `reconsider_markers: ["injected_cue"]`.
- No base model in the release; step 0 is the initialization under the RL
  prompt template.
- Dataset card declares **no license**; resolve before redistributing
  derived records.
- Smoke-row anomaly (n=2, noted not generalized): one row carried
  `shift_in_reasoning_v1=true` with empty prefilter markers and no marker
  text in the output; its excerpts point at the prompt's few-shot exemplar.
  Label provenance is not fully reconstructible from a row alone — the
  judge-vs-prefilter gap is worth watching at scale.

### Harness

[`step1_recheck.py`](./step1_recheck.py) — stdlib. Primary path reads the
OBSERVED passpair layout; a flat fallback covers the raw
`artifacts/results/*.jsonl` (schema UNCHECKED; refuses to guess, exit 3).
`--self-test` exercises both layouts on synthetic records with planted
effects, so correctness of the *harness* is separable from availability of
the *data*. Metric definitions:

- `shift_rate` — primary: the authors' released label
  `shift_in_reasoning_v1`, re-aggregated (paper-comparable). Sensitivities:
  narrow lexical markers, wide lexical markers, dataset prefilter alone.
  Spontaneous-shift statistics use cue-free pass1 records **only** (R2).
- `accuracy_given_shift` — authors' label primary, narrow-lexical secondary.
- `entropy_gated_gain` — the paper's RQ3 design: paired pass1→pass2 accuracy
  delta, top-20% vs bottom-80% within-task pass1 sequence entropy (R3).
- `spontaneous_shift_effect_high_entropy` — the paper's RQ3 spontaneous
  question at sequence level.
- Bootstrap CIs resample instances, not problem clusters (caveat carried in
  the output).

### Repair record (2026-11-20, vs the first harness draft)

- R1 — draft field lists missed the observed schema on four axes (nested
  pass1/pass2 JSON; `is_correct_pred` nested per pass; scalar sequence
  entropies, not token lists; `problem`/`example_id` id fields). On the real
  dump the flat path exits 3 by design; it stays as the raw-artifacts
  fallback.
- R2 — pass2 texts begin with the appended cue ("Wait, …"), so `\bwait\b`
  fires on every pass2 row. Spontaneous stats are pass1-only; the draft
  would have reported a shift rate inflated toward 50% on paired data.
- R3 — the draft's `entropy_gated_gain` (per-token max ≥ global p90,
  same-trace contrast) is not the paper's RQ3 number and cannot run on
  sequence-level entropies. Replaced by the paired design above; the draft
  contrast survives renamed (`spontaneous_shift_effect_high_entropy`).
- R4 — the draft assumed no judge labels in the release. Wrong: they ship
  (see correction above). Primary rate now uses them; lexical stays as the
  more independent, weaker definition.
- R5 — step1b's draft silently dropped the second record (re-injected
  `first` over the iterator's head). Fixed with `itertools.chain`.
- R6 — step1b's draft grouped by (model, problem), pooling all training
  steps and temperatures into one contaminated pass@k. Grouping is now per
  (model, task, step, temperature, problem); nothing pools across training
  states.
- R7 — draft problem-id detection missed the observed id fields; detection
  is shared with `step1_recheck.problem_id_of`.

## Step 1.5 — pass@k from the dump (free, no GPU, no re-sampling)

[`step1b_passk_from_dump.py`](./step1b_passk_from_dump.py) +
[`passk.py`](./passk.py) (unbiased estimator, Chen et al. 2021; paired
arm deltas with problem-level bootstrap).

OBSERVED limits, stated before anyone is disappointed: the dump carries
G = 8 samples per (problem, step, temperature), so **k is data-capped at
8** — pass@256 belongs to step 2's own sampling. There is no base-model arm;
the narrowing contrast is **step 0 vs final step** per (model, task,
temperature). `--emit-arm-files` writes step0/final arm files for
`passk.py`. Degrades honestly to `ran_partial` when a config lacks samples.

## Step 2+ — arms A/B/C (needs GPU; open slot)

Protocol sketch, so it can be picked up:

- Model: Qwen2.5-1.5B-Instruct (and 0.5B as a floor).
- Data: MATH-500 for train/eval; held-out set untouched for pass@k.
- RL: GRPO, identical hyperparameters across B and C; only reward differs.
  λ for C swept over {0.1, 0.5, 1.0} and reported, not tuned on eval.
- pass@k: n = 256 samples/problem, T = 1.0, k ∈ {1,2,4,…,256}.
- Entropy gate for C: token entropy > corpus p90 (same definition as step 1).
- Log per-run: tool-choice accuracy, shift rate, shift quality,
  answer-change rate.

## Slots

| slot | status | source / note |
|---|---|---|
| `step1_shift_rate` | **not run** — harness ready; judge labels ship in the release | `step1_recheck.py`; needs the dump |
| `step1_gated_gain` | **not run** — harness ready; paired pass1→pass2 design | ditto |
| `step1b_passk_from_dump` | **not run** — harness ready; k ≤ 8 on this dump | `step1b_passk_from_dump.py` |
| `pass_k_curves_ABC` | **not run** — needs GPU | step 2 |
| `selection_acc_C` | **not run** — needs GPU; blocked on the oracle decision | step 2 |
| `shift_quality_ABC` | **not run** — needs GPU | step 2 |
| `glyph_sensing_real` | **not run** — blocked on corpus choice | see NOURISHMENT.md addendum; pre-verbal vs. CoT degrade test; not answerable from this dump |
| `harness_self_test` | **OBSERVED** (synthetic, 2026-11-20) | `step1_recheck.py --self-test`, both layouts; planted effects recovered |
| `passk_self_test` | **OBSERVED** (synthetic, 2026-11-20) | `passk.py --self-test`; estimator pins + planted narrowing detected (pass@1 ↑, pass@8 ↓, CIs exclude 0) |
| `step1b_self_test` | **OBSERVED** (synthetic, 2026-11-20) | `step1b_passk_from_dump.py --self-test`; planted narrowing recovered step0-vs-final |
| `step1_schema_smoke` | **OBSERVED** (2 real rows via rows API, 2026-11-20) | layout detected, all pass1 fields read, paired pipeline green; numbers structurally meaningless at n=2 and are NOT slot values |

Rule: every slot above stays `not run` until a results file with
`"status": "ran"` exists and is committed alongside. No number enters
this table from memory or from the papers.

## Reproduction

```bash
# 0. what is green today (no data needed)
D=docs/research/tool-selection-vs-deepening
python $D/step1_recheck.py --self-test
python $D/passk.py --self-test
python $D/step1b_passk_from_dump.py --self-test

# 1. fetch (needs network; not done by this owner)
pip install huggingface_hub
huggingface-cli download od2961/illusion-of-reasoning-main-traces \
  --repo-type dataset --local-dir data/princeton-traces
# note: the processed split is parquet (skipped by design, stdlib-only);
# the raw artifacts/results/*.jsonl are the intake path

# 2. probe schema, then run
python $D/step1_recheck.py --data data/princeton-traces --probe
python $D/step1_recheck.py --data data/princeton-traces --out $D/step1_results.json
python $D/step1b_passk_from_dump.py --data data/princeton-traces --probe
python $D/step1b_passk_from_dump.py --data data/princeton-traces \
     --out $D/step1b_results.json --emit-arm-files
python $D/passk.py --arm step0=$D/arm_GRPO-1.5B_math_step0_T0p7.json \
                   --arm final=$D/arm_GRPO-1.5B_math_step950_T0p7.json --ks 1,2,4,8
```

Observer slots: null.
