# Safety drift, alignment repair and parameter-space monitoring — experiment designs

Status: PROPOSED (all experiments). Date 2026-11-24. Co-authored (human + AI).
Draft read: `hypotheses/safety-drift-alignment-repair-and-parameter-space-monitoring.md` at
`02320b51b9138dec032f8f85839bce420502a9ec` (regenerated 2026-11-23T16:38:05Z).
Engine-level gaps: [`engine-evidence-audit.md`](engine-evidence-audit.md).

## 1. What the engine claims, and on what evidence

"Established", 72 surviving claims of 138 for the topic; 4 survivors predate provenance
stamping. The survivors include on-topic fine-tuning safety papers and clearly off-topic
items (PatternFormer, VAKRA, a Korean public-API benchmark, diffusion total variation,
multi-pitch time-frequency analysis); a Montgomery–Vaughan Hilbert-inequality paper is
listed as active. Survival is the engine's lexical/corroboration rule, not a measurement
(see ENG-G1..G3).

Spot checks (arXiv API, 2026-11-24; abstracts read, papers not read in full):

| cited item | checked? | what the abstract says |
|---|---|---|
| arXiv 2506.05346, "Why LLM Safety Guardrails Collapse After Fine-tuning: A Similarity Analysis Between Alignment and Fine-tuning Datasets" | exists | High representation similarity between upstream alignment data and downstream fine-tuning data weakens guardrails; low similarity gives more robust models. |
| arXiv 2602.16931, "Narrow Fine-Tuning Erodes Safety Alignment in Vision-Language Agents" | exists | Gemma3-4B; misalignment grows monotonically with LoRA rank; multimodal evaluation shows more misalignment than text-only; mixing 10% harmful data degrades alignment. |
| arXiv 2510.01555, "Rethinking KL Regularization in RLHF" | exists | Compares KL estimators (k_n) used in the reward vs as a loss; argues k1-in-reward is the principled choice. Relevant to how the repo's KL safety term is placed, not to monitoring. |
| other on-topic survivors | not checked | UNCHECKED. |

Anchor literature not retrieved by the engine (existence checked): Qi et al., "Fine-tuning
Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!", arXiv
2310.03693; Wei et al., "Assessing the Brittleness of Safety Alignment via Pruning and
Low-Rank Modifications", arXiv 2402.05162; Kirkpatrick et al. (EWC), arXiv 1612.00796.

## 2. What the repo claims (the part this topic tests)

- `manifolds/parameter_manifold.py` `ParameterManifold.repair_step`: task cross-entropy plus
  `lambda_asym` × KL-to-reference on safety inputs, KL weighted by `(1 + lambda_curv · curv)`,
  step projected to a trust region. Sign note in code: corrected 2026-10-02 to descent on KL.
- `repair/generic_repair_controller.py`: stdlib controller, objective task + λ·safety (sign
  corrected 2026-11-20; README reports that on the default run the minus form raised KL
  3.79 → 39.37 and the plus form lowered it 3.79 → 0.17 — a single run, not re-run here).
  Phase labels are heuristic (`ISS_PROOF_PENDING: True`).
- `docs/research/residual-tower/`: snapshot-delta Test 1 (`snapshot_delta_test1.py`,
  results `test1_pass: true`, seed 20261120) shows bit-identical incremental vs full
  dot/matvec in exact integer (GF(2)/word) arithmetic. `explore_harness.py` is built for
  BLIND/REVIEW records; linear-delta probes are BLOCKED / NOT_RUN per `EXPLORE_HARNESS_SPEC.md`.

## 3. What science needs to be known

### (a) Gaps the engine recorded

- None topic-specific beyond the engine-wide hidden-variable self-correlation defect.

### (b) Gaps identified here

| id | gap | kind |
|---|---|---|
| SDM-G1 | **Do parameter-space monitors track safety, or only training time?** ‖Δθ‖ and KL-to-reference both grow with steps, as does safety degradation; a correlation along one trajectory is confounded by step count. The repo's monitoring premise has no test at fixed step across runs that differ in safety outcome. | confound / missing control |
| SDM-G2 | **Repair vs simple baselines.** The repair objective is a KL penalty with curvature weighting and a trust region. Whether it beats L2-to-reference (L2-SP), EWC, or an unweighted KL penalty at matched task loss is not measured; one default run after a sign fix is not a comparison. | missing control |
| SDM-G3 | **Exact vs floating arithmetic.** Test 1 shows exactness for integer arithmetic. Parameter monitoring of real checkpoints is float32/bf16; incremental updates accumulate rounding. Whether this flips monitor decisions is unmeasured. | scope limit |
| SDM-G4 | **External effect not reproduced.** The similarity effect in 2506.05346 is the most directly relevant survivor; it has not been reproduced at small scale or connected to a parameter-space quantity. | replication |
| SDM-G5 | **Operationalisation of "safety".** In the toy, safety = KL to reference outputs on designated inputs. For LLMs, refusal-rate on harmful prompts scored by a classifier is a different construct; results do not transfer automatically between them. | operationalisation |

## 4. Experiments

### EXP-SDM-01 — Do parameter-space monitors predict safety degradation at fixed step?

Gap: SDM-G1, SDM-G5.

- **Hypothesis.** H1: across fine-tuning runs compared at the same step, a parameter-space
  monitor M ranks runs by safety degradation ΔS (Spearman ρ ≥ 0.5; AUC ≥ 0.75 for "ΔS above
  median"). H0: ρ ≤ 0.2 once step is fixed (the monitor carries time, not safety).
- **Monitors (M).** ‖Δθ‖₂; diagonal-Fisher-weighted ‖Δθ‖ with Fisher at θ_ref on harmful
  prompts; projection of Δθ onto the alignment direction θ_instruct − θ_base (available when
  both base and instruct weights are public); output KL to θ_ref on held-out harmless prompts
  (an output-space comparator).
- **Outcome (S).** Refusal rate on a fixed held-out harmful prompt set, scored by a fixed
  classifier, with a 10% human audit of classifier labels. Prompt set and classifier chosen
  and cited before running.
- **Design.** A small open instruct model with a public base (e.g. Qwen2.5-0.5B-Instruct;
  availability and licence to confirm). Fine-tuning data: benign instruction data, benign +
  10% harmful, harmful-only (the Qi et al. setting, abstract-level). LoRA rank ∈ {8, 64}, two
  learning rates, 3 seeds → 36 runs, 10 checkpoints each. Rough compute: hours on one 24 GB
  GPU; justify the count as the minimum giving ≥ 30 runs per fixed step for a rank correlation.
- **Pre-registered prediction.** The Fisher-weighted and projection monitors satisfy H1;
  raw ‖Δθ‖ does not (ρ < 0.5), because LoRA rank and lr change ‖Δθ‖ without proportional
  safety change. **Falsified if** no parameter monitor reaches ρ ≥ 0.5 at the final step, or
  if output KL (cheaper) dominates all parameter monitors by ≥ 0.15 in ρ.
- **Analysis.** Per-step Spearman ρ across runs with bootstrap CI; partial correlation with
  step on pooled checkpoints as secondary; Holm correction across 4 monitors.
- **Threats.** Classifier drift on fine-tuned output style; prompt set leakage into training
  data; Fisher estimated on few samples (report sample size; see FIT notes on the
  `FisherMetricEstimator` normalisation).
- **Tier.** Single GPU.
- **Would establish:** whether parameter-space quantities carry safety information beyond
  training time for this model family. **Would not establish:** that monitoring prevents
  harm, or anything about larger models or RLHF-trained guardrails.

### EXP-SDM-02 — Repo repair vs simple regularisers on the toy (Pareto comparison)

Gap: SDM-G2.

- **Hypothesis.** H1: the repo's curvature-weighted, trust-region KL repair reaches a better
  task-loss vs safety-KL frontier than (i) unweighted KL penalty, (ii) L2-SP ‖θ − θ_ref‖², (iii)
  EWC with diagonal Fisher at θ_ref, (iv) no repair. H0: frontiers overlap within seed noise.
- **Design.** `simulation/environment.py` ToyLLM (3,152 parameters) with the drift used by
  `main.py`. For each arm sweep the strength (λ) over 8 log-spaced values, equal step budget,
  10 seeds. Ablate the repo arm: curvature weighting off; trust region off. Also run
  `repair/generic_repair_controller.py` on the analytic landscapes in
  `experiments/experiment_generic_repair.py` with the same baselines added.
- **Pre-registered prediction.** Hypervolume of the repo arm exceeds the best baseline by
  ≥ 5% of the reference box; the gain disappears with curvature weighting off. **Falsified
  if** the 95% bootstrap CI of (repo − best baseline) hypervolume includes 0.
- **Analysis.** Per-seed fronts; hypervolume with a fixed reference point set before
  running; paired bootstrap across seeds.
- **Threats.** Tuning effort imbalance: give each arm the same λ grid and step budget.
  Wall-clock: report it (the repo's compute-cost notes suggest repair cost is a variable).
- **Tier.** Laptop CPU.
- **Would establish:** whether the repo's specific ingredients add anything over standard
  regularisers at toy scale. **Would not establish:** behaviour on real models.
- **Reuses:** `main.py`, `manifolds/parameter_manifold.py`, `simulation/controller.py`,
  `experiments/ablations.py` (layer-disable runner), `repair/generic_repair_controller.py`,
  `experiments/experiment_generic_repair.py`.

### EXP-SDM-03 — Snapshot-delta monitoring in floating point

Gap: SDM-G3. Extends Test 1; does not reinterpret it.

- **Hypothesis.** H1: for a linear monitor m(θ) = w·θ over float32 checkpoints, the
  incremental update m_t = m_{t−1} + w·Δθ_t diverges from full recomputation by an error that
  grows with the number of deltas T and flips threshold decisions at a measurable rate near
  the threshold. H0: no decision flips in T ≤ 10⁴ for monitors with margin > 1 ulp-scaled
  bound.
- **Design.** Positive control: rerun `snapshot_delta_test1.py` (integer arithmetic) and
  require its recorded bit-identity (`bit_identical`) to reproduce. Then float arms: float32
  and float64 incremental, float32 with Kahan-compensated accumulation, all vs float64 full
  recomputation as reference. Δθ sequences from (a) synthetic sparse deltas (reuse
  `make_snapshots` structure with float values) and (b) the ToyLLM trajectory from `main.py`.
  N = 2·10⁵ entries (as in Test 1), T ∈ {10, 10², 10³, 10⁴}, 20 seeds.
- **Pre-registered prediction.** float32 error grows roughly ∝ √T for random-sign deltas
  (DERIVED heuristic, random-walk rounding) and decision flips occur only when the margin is
  within a few multiples of that error; Kahan removes flips at T = 10⁴. **Falsified if**
  float32 error is independent of T, or Kahan does not reduce it.
- **Analysis.** Max and RMS absolute error vs T (log–log slope with CI); flip rate per
  threshold margin bin.
- **Logging.** Record runs in the `explore_harness.py` record format so they sit beside
  Test 1; mark them as a new probe, not as the BLOCKED linear-delta probes.
- **Tier.** Laptop CPU.
- **Would establish:** when incremental monitoring is safe in float. **Would not
  establish:** anything about whether the monitor measures safety (SDM-01).
- **Reuses:** `docs/research/residual-tower/snapshot_delta_test1.py` (`full_dot`, `inc_dot`,
  `full_matvec`, `inc_matvec`, `bit_identical`, `make_snapshots`), `explore_harness.py`.

### EXP-SDM-04 — Small-scale replication of the alignment/fine-tuning similarity effect

Gap: SDM-G4. Lower priority; depends on SDM-01 infrastructure.

- **Hypothesis.** H1 (from the 2506.05346 abstract): fine-tuning sets more similar in
  representation to the model's alignment data produce larger post-fine-tuning safety
  degradation. H0: Spearman ρ(similarity, ΔS) ≤ 0.
- **Design.** Needs a model whose alignment data are public (e.g. an OLMo/Tülu-family
  instruct model; specific checkpoint UNCHECKED). Build 8 benign fine-tuning sets spanning a
  similarity range (mean cosine of embeddings to the alignment set, embedding model fixed in
  advance), equal size and steps, 3 seeds. Measure ΔS as in SDM-01, plus the SDM-01
  monitors so the effect can be tied to a parameter-space quantity.
- **Pre-registered prediction.** ρ ≥ 0.5. **Falsified if** the 95% CI of ρ lies below 0.2.
- **Threats.** Similarity correlates with task difficulty and loss magnitude; report
  fine-tuning loss and ‖Δθ‖ as covariates.
- **Tier.** Single GPU.
- **Would establish:** whether the external effect holds at small scale. **Would not
  establish:** the paper's causal mechanism.
