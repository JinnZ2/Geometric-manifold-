<!-- Author: DeepSeek (relayed by owner 2026-11-20); reviewed by chat Claude L0, see REVIEW.md -->

# Tool-selection vs. deepening — RL reward shape and the breadth ceiling

Status: **PROPOSED · step 1 harness built, not yet run on real data** ·
co-authored (owner + chat Claude, 2026-11-20) · direct-to-main

## Question

Does RL that rewards *selecting a reasoning tool* when uncertainty rises
preserve the base model's breadth, while RL that rewards *the answer* narrows it?

## Sources (OBSERVED by the cited work; **not** re-measured here)

| claim | source | status |
|---|---|---|
| "aha" self-reflection emerges in R1-Zero without reflection RL | Guo et al. 2025 (DeepSeek-R1) | OBSERVED (cited) |
| base models already self-reflect | Liu et al. 2025, Sea AI Lab — DeepSeek-V3-Base, 500 MATH questions | OBSERVED (cited) |
| spontaneous shifts in ~6.31% of traces; do not rise with training; seldom help | arXiv 2601.00514v2 (Princeton, Apr 2026) | OBSERVED (cited) |
| entropy-triggered *external* reconsideration: +8.41 pp on MATH-500 | arXiv 2601.00514v2 | OBSERVED (cited) |
| scope of the above: Qwen2.5 + Llama, GRPO; crosswords, MATH-500, Rush Hour | arXiv 2601.00514v2 | OBSERVED (cited) |

None of the numbers above are re-derived in this repo yet. Step 1 is the
independent read; until it runs, they are *cited*, not *reproduced*.

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

## Step 1 — independent re-check (stdlib, runnable now)

Target: Princeton released traces + code
(`github.com/humans-and-machines/Illusion-of-Reasoning`,
`HF od2961/illusion-of-reasoning-main-traces`).

Re-derive, without the authors' code in the loop:

1. shift rate,
2. accuracy given a shift vs. without,
3. entropy-gated gain.

Different lab from the authors = the first independent read.

Harness: [`step1_recheck.py`](./step1_recheck.py). It is schema-tolerant
(probes field names, mapping overridable via `--map`), streams records,
and writes `step1_results.json`. `--self-test` exercises the whole
pipeline on synthetic records so correctness of the *harness* is
separable from availability of the *data*.

Known limitation recorded in output: shift↔token positional alignment is
not attempted. If the release carries per-token entropies but no explicit
`shift_positions`, the gate is evaluated at trace level
(spike somewhere ∧ shift somewhere), which is weaker than
"reconsideration *triggered by* the spike".

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
| `step1_shift_rate` | **not run** — harness ready | `step1_recheck.py`; data not present |
| `step1_gated_gain` | **not run** — harness ready | ditto; gated form is trace-level (see limitation) |
| `pass_k_curves_ABC` | **not run** — needs GPU | step 2 |
| `selection_acc_C` | **not run** — needs GPU | step 2 |
| `shift_quality_ABC` | **not run** — needs GPU | step 2 |
| `harness_self_test` | **OBSERVED** (synthetic) | `step1_recheck.py --self-test` |

Rule: every slot above stays `not run` until a `step1_results.json` with
`"status": "ran"` exists and is committed alongside. No number enters
this table from memory or from the papers.

## Reproduction

```bash
# 1. fetch (needs network; not done by this owner)
pip install huggingface_hub
huggingface-cli download od2961/illusion-of-reasoning-main-traces \
  --repo-type dataset --local-dir data/princeton-traces

# 2. probe schema
python docs/research/tool-selection-vs-deepening/step1_recheck.py \
  --data data/princeton-traces --probe

# 3. run
python docs/research/tool-selection-vs-deepening/step1_recheck.py \
  --data data/princeton-traces --out step1_results.json
```


<!-- filer: block 2 of 4, appended in the order received -->

## Pre-registered thresholds (committed 2026-11-20, before any step-2 number)

- **"C narrows like B"** is refuted if, at k=256,
  `pass@256(C) ≥ 0.8 · pass@256(A)` while `pass@256(B) < 0.8 · pass@256(A)`,
  each with a problem-level bootstrap 95% CI excluding the threshold.
- **"selection doesn't improve"** is refuted if the C−B difference in
  tool-choice accuracy at p90 entropy spikes has a 95% CI with upper
  bound ≤ 0.
- **Step 1.5 (free, on the released dump):** the *cited* narrowing claim
  is independently supported only if, with base and trained arms both
  present, `pass@256(trained) < pass@256(base)` with CI excluding 0.
  If the dump holds one sample per problem, or no base arm, this slot
  stays `not run` — the cited claim is then unreproduced, not refuted.

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

## Slots (updated)

| slot | status | source / note |
|---|---|---|
| `step1_shift_rate` | not run — harness ready | `step1_recheck.py` |
| `step1_gated_gain` | not run — harness ready | trace-level gate (see limitation) |
| `step1b_passk_from_dump` | not run — harness ready | `step1b_passk_from_dump.py`; needs dump |
| `pass_k_curves_ABC` | not run — needs GPU | step 2 |
| `selection_acc_C` | not run — needs GPU | step 2; blocked on oracle decision |
| `shift_quality_ABC` | not run — needs GPU | step 2 |
| `harness_self_test` | OBSERVED (synthetic) | `step1_recheck.py --self-test` |
| `passk_unit_check` | OBSERVED (synthetic) | see below |

<!-- filer: block 3 of 4, appended in the order received -->

| `glyph_sensing_real` | not run — blocked on corpus choice | see NOURISHMENT.md addendum; pre-verbal vs. CoT degrade test |

<!-- filer: block 4 of 4, appended in the order received -->

*The glyph addendum in NOURISHMENT.md forces this: if perception-assertion
is the object, outcome-conditional (1) is disqualified by construction and
(2) counterfactual is the only faithful operationalization.*
