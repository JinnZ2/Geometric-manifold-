# Hidden-variable detection / causal discovery from residuals — experiment designs

Status: PROPOSED (all experiments). Date 2026-11-24. Co-authored (human + AI).
Draft read: `hypotheses/hidden-variable-detection-causal-discovery-from-residuals.md` at
`02320b51b9138dec032f8f85839bce420502a9ec` (regenerated 2026-11-23T16:38:05Z).
Engine-level gaps: [`engine-evidence-audit.md`](engine-evidence-audit.md).

## 1. What the engine claims, and on what evidence

"Established" with 4 surviving claims. Statement line: the nested-prediction-models paper.

| cited item | checked? | what the record shows |
|---|---|---|
| Non-Invariance in Nested Prediction Models under Selective Predictor Availability, arXiv 2609.02836 | exists (arXiv API; current version v6) | Clinical prediction with a selectively measured predictor; decomposes non-invariance into omission, residual non-invariance and interaction; extends to collider structures from multiple selection routes. On-topic in the sense of selection/missingness producing spurious structure. |
| Early fault detection for DFIG via multiscale latent variable regression, doi 10.1109/TIM.2025.3542110 | not checked | UNCHECKED. Title indicates latent-variable regression for fault detection (process monitoring, not causal discovery). |
| Quality-oriented dynamic sparse latent variable detection for industrial IoT, doi 10.1109/JIOT.2025.3538627 | not checked | UNCHECKED. Same reading from title. |
| WeightBench: LMs as weight-space programmers, arXiv 2611.18553 | title only | Unrelated to hidden-variable detection. |
| Identification of Latent Variables From Graphical Model Residuals (Semantic Scholar 4fd0e767…) | API rate-limited on 2026-11-24 | UNCHECKED at source. The engine's logged abstract (2021, `findings_log.jsonl`) describes causal discovery when causal sufficiency fails. This is the most on-topic item in the log, and its claim is `active` with 0 passes, not surviving. |

The topic's own engine stage (stage 6, `stage_hidden`) is recorded as non-functional in
`docs/hypothesis_engine.md` ("Known defect"): the two `pearson_r = 1.0` rows in
`data/hidden_variables.jsonl` (2026-08-13) were self-correlations, and with endogenous
candidates excluded and the constant one skipped the stage cannot produce a suggestion.

## 2. What science needs to be known

### (a) Gaps the engine recorded

- HVD-G1 (recorded in `docs/hypothesis_engine.md`): no valid exogenous candidate; claims
  need dates on a real time axis and at least one candidate series not computed from
  `beta_confidence`.
- HVD-G2 (recorded): `findings_rate` was aligned by `i % len(periods)`, so pairings were
  arbitrary.

### (b) Gaps identified here

| id | gap | kind |
|---|---|---|
| HVD-G3 | **Detection power of the trigger is unknown** even with valid inputs. The residual `|beta − 0.5|` folds the sign, `beta` takes few distinct values for claims with ≤ 5 tests, and the rule `mean|res| ≥ 0.1 ∧ |r| > 0.5` has no stated false-alarm rate or power at the topic sizes seen (20 to 262 claims per topic at this commit). | measurement / operationalisation |
| HVD-G4 | **The date axis exists.** Every claim id joins to a dated finding (492 / 492 at this commit), so HVD-G1's plumbing is available without changing the claim schema. Caveat: `date` is publication date from each API, with source-dependent semantics (Crossref gives year-level dates for some rows); `retrieved_at` is the run time. Which axis a hidden driver would act on is unstated. | operationalisation |
| HVD-G5 | **Known nuisance covariates were never checked.** Pass counts depend on lexical overlap (ENG-G2), so abstract length, missing abstracts, source and lane are candidate "hidden variables" for the engine's own residuals. They are measurable today. | confound |
| HVD-G6 | **Identifiability.** "Residual structure implies a hidden cause" holds only under assumptions (linearity, noise distribution, no selection, no measurement error). Measurement error and selection (the collider structures in arXiv 2609.02836) can produce the same residual pattern with no latent cause. Neither the engine nor `sims/dark_constraint/` states which assumptions it relies on. | unstated assumption |
| HVD-G7 | **Q7 (`docs/research/OPEN_QUESTIONS.md`) is half-open:** K = 1 absorption is established; K ≥ 2 recovery is untested because the null arm was degenerate (null residual a structural zero). | missing control (recorded in repo, not by the engine) |

## 3. Experiments

### EXP-HVD-01 — Power and false-alarm rate of the engine's trigger on planted data

Gap: HVD-G3.

- **Hypothesis.** H1: with a valid exogenous candidate, the rule detects a planted driver of
  moderate size (log-odds effect β_z = 1 on per-test pass probability) with power ≥ 0.8 at
  ≥ 30 claims per topic, at a false-alarm rate ≤ 0.05. H0: power < 0.5 or false-alarm rate
  > 0.1.
- **Variables.** Planted: binary z per claim (rate 0.3), β_z ∈ {0, 0.5, 1, 2}, tests per claim
  ∈ {3, 5, 10}, claims per topic ∈ {10, 30, 100, 300}, base pass logit ∈ {−1, 0, 1}.
  Outcome: fire rate over 500 replicate topics per cell, with the folded residual (engine)
  and a signed residual (`beta − 0.5`) as an ablation.
- **Pre-registered prediction.** The folded residual has lower power than the signed one
  whenever base logit ≈ 0 (beta straddles 0.5). Power at β_z = 1 is < 0.8 for ≤ 30 claims,
  because |r| > 0.5 is a strict bar for a binary candidate. **Falsified if** folded and signed
  residuals have equal power (±0.05) across all cells, or power ≥ 0.8 at 30 claims and β_z = 1.
- **Analysis.** Fire rates with Wilson CIs; power curves per cell. 500 replicates give a CI
  half-width ≤ 0.045.
- **Threats.** Synthetic outcomes are binomial and independent across tests; real engine
  passes are neither (ENG-G5). The result bounds the rule's power from above.
- **Tier.** Laptop CPU (seconds).
- **Would establish:** whether the trigger could work once fed. **Would not establish:**
  that any real hidden variable exists.
- **Sketch:** [`sketches/hvd_planted_trigger.py`](sketches/hvd_planted_trigger.py) (imports
  the engine's `pearson_r` read-only; `--selftest` passes; the sweep has not been run).

### EXP-HVD-02 — The engine's own residuals against measurable nuisance covariates

Gap: HVD-G4, HVD-G5.

- **Hypothesis.** H1: claim outcomes (`passed`, `survived`) are explained to a large degree
  by text-mechanical covariates. H0: no covariate beats a permutation null.
- **Variables.** Per claim (join on id): abstract length (tokens), `(no abstract)` flag,
  number of topic-name tokens in the abstract, source, lane of the finding, number of same-topic
  findings logged by the claim's first test run, publication date, `retrieved_at`.
- **Design.** Offline on `data/`. Negative-binomial regression of `passed`; logistic
  regression of `survived`; topic as a fixed effect.
- **Pre-registered prediction.** Number of same-topic findings at test time and abstract
  token count each have positive coefficients with 95% CI excluding 0; publication date has
  none after those are included. **Falsified if** no text-mechanical covariate has a CI
  excluding 0 and the permutation p for the full model is > 0.05 (10,000 permutations).
- **Threats.** Truncation of `passed` at survival (use `survived` as the main outcome);
  pre-provenance claims (analyse with and without); reformulated claims reset counts
  (`reformulate()` zeroes passed/failed).
- **Tier.** Laptop CPU.
- **Would establish:** which nuisance variables drive the engine's confidence numbers, and a
  first valid candidate series for stage 6. **Would not establish:** anything about the
  scientific topic itself.

### EXP-HVD-03 — When does residual structure license a latent cause? (linear SEM benchmark)

Gap: HVD-G6.

- **Hypothesis.** H1: a residual-dependence test detects a planted latent confounder in a
  linear SEM with power ≥ 0.8 at n = 1,000 and confounding strength ≥ 0.3, and its
  false-alarm rate under measurement error with no latent is ≤ 0.1. H0: the test does not
  distinguish latent confounding from measurement error (false-alarm rate under measurement
  error ≥ 0.5 × power under latent).
- **Arms.** (1) No latent, exact measurement (null). (2) Latent L → X, L → Y with strength c ∈
  {0.1, 0.3, 0.5}. (3) No latent, measurement error on X with reliability ∈ {0.9, 0.7, 0.5}.
  (4) No latent, selection on a collider (keep rows with X + Y + noise > q), q ∈ {0, 0.5, 1}.
  Each with Gaussian and with non-Gaussian (uniform, Laplace) noise.
- **Tests.** Residual correlation after regressing Y on X's observed parents; an HSIC-type
  independence test of residuals against regressors (detects dependence that linear
  correlation cannot, under non-Gaussian noise); report both.
- **Scale.** n ∈ {100, 300, 1,000, 3,000}; 500 replicates per cell (CI half-width ≤ 0.045).
- **Pre-registered prediction.** Under Gaussian noise, arms 2 and 3 are indistinguishable by
  either test (the standard non-identifiability result for linear Gaussian models). Under
  non-Gaussian noise, HSIC separates them. **Falsified if** arm 3 under Gaussian noise
  produces a false-alarm rate ≤ 0.1 while arm 2 is detected with power ≥ 0.8.
- **Tier.** Laptop CPU. numpy/scipy; HSIC with a permutation null (200 permutations) is the
  cost driver.
- **Would establish:** the operating conditions under which "residual ⇒ hidden variable" is
  a valid inference, as an input to stage 6 and to `sims/dark_constraint/`. **Would not
  establish:** anything about nonlinear or time-series causal discovery, which needs a
  separate design.

### EXP-HVD-04 — Q7: does K ≥ 2 recover an unmodelled constraint against a non-degenerate null?

Gap: HVD-G7. Design follows the experiment already written in
`docs/research/OPEN_QUESTIONS.md` Q7; this entry fixes the parts left open there.

- **Hypothesis.** H1: with K ≥ 2 load cases, the residual of the visible-only fit scales
  monotonically with dark-constraint strength and exceeds the null-arm floor by ≥ 3×. H0:
  residual does not scale (Spearman ρ < 0.5) or stays within 3× of the floor.
- **Design.** Re-run `sims/dark_constraint/run.py` with: a null arm carrying non-zero visible
  residuals (the specific change Q7 asks for) so linearisation and convergence error set the
  floor; K ∈ {1, 2, 3, 4}; dark strength on a 6-point grid bracketing where K = 1 absorption
  breaks down; three locations and two constraint kinds; 10 seeds. Write the change as a new
  config and a REFUTE addendum before running; do not overwrite the existing FINDING.
- **Pre-registered prediction.** K = 1 continues to absorb (residual ≤ floor) at all
  strengths; K ≥ 2 residual scales with strength. **Falsified if** K ≥ 2 residual is flat in
  strength once the null is non-degenerate.
- **Tier.** Laptop CPU.
- **Would establish:** whether multiple independent probes give the instrument a handle on a
  dark constraint in this toy. **Would not establish:** anything about astrophysical dark
  matter (the analogy that motivated Q7).
- **Reuses:** `sims/dark_constraint/` (`run.py`, `REFUTE.md`, `NULL.md`, `config.json`).
