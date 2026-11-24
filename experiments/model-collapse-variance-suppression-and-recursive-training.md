# Model collapse, variance suppression and recursive training — experiment designs

Status: PROPOSED (all experiments). Date 2026-11-24. Co-authored (human + AI).
Draft read: `hypotheses/model-collapse-variance-suppression-and-recursive-training.md` at
`02320b51b9138dec032f8f85839bce420502a9ec` (regenerated 2026-11-23T16:38:05Z).
Engine-level gaps: [`engine-evidence-audit.md`](engine-evidence-audit.md).

## 1. What the engine claims, and on what evidence

"Established" with 33 surviving claims; 17 of them predate provenance stamping, 7 have no
abstract sentence. Statement line: CollapseTracker. The list mixes three things the topic
name joins together: model collapse under recursive training, critical-slowing-down early
warning signals (EWS), and unrelated papers (a NOvA neutrino cross-section, sign-language
gloss translation, recipe transformers, HVAC review).

Spot checks (Crossref, 2026-11-24):

| cited item | checked? | what the record shows |
|---|---|---|
| CollapseTracker: An Empirical Dataset Documenting Progressive Model Collapse..., doi 10.2139/ssrn.6560303 | exists | SSRN posted content (preprint, not peer reviewed), 2026. Engine-logged abstract: recursive fine-tuning of GPT-2 (124M) and DistilGPT-2 (82M) over 10 generations on 3 seed datasets. Whether the dataset is downloadable: UNCHECKED. |
| Self-Verification Provably Prevents Model Collapse in Recursive Synthetic Training, doi 10.52202/085713-1213 | exists | Fu, Wang, Chen, Shen, NeurIPS 38 (2025) proceedings. No abstract in Crossref; the engine's claim text is "(no abstract)", so it survived on title tokens. Content UNCHECKED. |
| Limits of using early warning signals for preventing tipping, doi 10.52843/cassyni.b93kgq | exists | Sujith, 2025-06-24, Cassyni posted content (a talk record, not a paper). Abstract: EWS detect an impending tipping only past bifurcation points when control parameters vary fast; fast variation also delays tipping ("borrowed stability"); shown in a thermoacoustic system. This is the cluster HYPOTHESIS_TRIAGE flagged as valuable. |
| Conditions for skilful spatial and temporal tipping point EWS; Critical slowing down at the neuromuscular junction (ALS); Wealth thermalization EWS | not checked | UNCHECKED. |

Anchor literature not retrieved by the engine (existence checked 2026-11-24): Shumailov et al.,
"AI models collapse when trained on recursively generated data", *Nature* 2024,
doi 10.1038/s41586-024-07566-y (and the preprint arXiv 2305.17493); Gerstgrasser et al., "Is
Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic
Data", arXiv 2404.01413; Dohmatob et al., "A Tale of Tails: Model Collapse as a Change of
Scaling Laws", arXiv 2402.07043; Scheffer et al., "Early-warning signals for critical
transitions", *Nature* 2009, doi 10.1038/nature08227; Boettiger & Hastings 2012,
doi 10.1098/rspb.2012.2085. Content of these is from their abstracts or not re-read here; the
specific statements used below are marked DERIVED (textbook) or VERIFY.

## 2. What science needs to be known

### (a) Gaps the engine recorded

- Two `pearson_r = 1.0` hidden-variable rows for this topic on 2026-08-13 were self-correlations
  (`docs/hypothesis_engine.md`, "Known defect"); nothing else topic-specific.

### (b) Gaps identified here

| id | gap | kind |
|---|---|---|
| MCR-G1 | **Regime of collapse.** Whether recursive training collapses depends on whether data are replaced or accumulated (Shumailov vs Gerstgrasser abstracts). The topic and the repo's notes use "model collapse" without fixing the regime. | scope / operationalisation |
| MCR-G2 | **Sign of the variance signal.** In critical slowing down, variance and lag-1 autocorrelation of fluctuations rise approaching a bifurcation (Scheffer 2009). In model collapse, the variance of the generated distribution falls. The topic name pairs "variance suppression" with EWS; these are different variances (state vs fluctuation of a summary statistic), and whether recursive refitting approaches a bifurcation at all (rather than drifting as a multiplicative random walk) is not established. | conceptual confound |
| MCR-G3 | **Prosecutor's fallacy.** Any EWS evaluation on runs selected because they collapsed overstates performance (Boettiger & Hastings 2012). A null arm of non-collapsing runs matched in length is required. | missing control |
| MCR-G4 | **Fast-parameter limit.** Sujith (2025): with fast parameter change, EWS fire after the bifurcation. Recursive training changes the "parameter" (the data distribution) every generation, so this limit may apply by construction. | scope |
| MCR-G5 | **Repo mechanism untested in this setting.** The data manifold's asymmetric GMR cleaning ("minority never dropped") is the repo's design answer to tail loss. Whether it slows tail loss across generations, relative to symmetric cleaning and to no cleaning, is unmeasured. | missing experiment |

## 3. Experiments

### EXP-MCR-01 — Replace vs accumulate in recursive Gaussian and GMM refitting (calibration)

Gap: MCR-G1. This is a reproduction of known behaviour, used to calibrate the toy that
MCR-02 and MCR-03 build on.

- **Hypothesis.** H1 (DERIVED for the 1-D Gaussian MLE): under replace,
  E[σ²_t] = σ²_0 ((n−1)/n)^{t+1} counting generation 0 as a fit to real data; median σ²_t decays
  faster than the mean. Under accumulate, σ²_t stays within a bounded band. H0: no difference
  between regimes beyond sampling error.
- **Design.** 1-D Gaussian (sketch), then a K = 5 component GMM in 2-D (EM, fixed K), n ∈ {20,
  100, 1,000} per generation, T = 200 generations, 500 seeds. GMM outcomes: smallest
  component weight, tail mass P(‖x‖ > 3σ_0), number of components with weight > 1/(10K).
- **Pre-registered prediction.** 1-D mean variance ratio within 5% of the formula at t ≤ 10 for
  n = 20 (a code check; failure means a bug). Under replace, the median GMM tail mass at
  t = 200 is < 50% of t = 0 for n ≤ 100; under accumulate it is ≥ 80%. **Falsified (for the
  GMM part) if** replace and accumulate tail mass differ by < 10 percentage points at t = 200
  for n = 100.
- **Analysis.** Per-generation medians with 95% bootstrap bands over seeds.
- **Tier.** Laptop CPU.
- **Would establish:** a calibrated toy with known regime behaviour. **Would not establish:**
  anything about language models (CollapseTracker-scale replication is a single-GPU follow-on,
  only if the dataset or code is available, which is UNCHECKED).
- **Sketch:** [`sketches/mcr_recursive_gaussian.py`](sketches/mcr_recursive_gaussian.py)
  (1-D only; `--selftest` checks the mean recursion over 4,000 seeds at n = 20; the regime
  comparison has not been run).

### EXP-MCR-02 — Do standard EWS lead a collapse event, against a matched null?

Gap: MCR-G2, MCR-G3, MCR-G4.

- **Hypothesis.** H1 (the CSD reading of the topic): Kendall τ of rolling variance and of
  rolling lag-1 autocorrelation of a summary statistic (σ_t, or smallest GMM component
  weight) is positive before a defined collapse event, and discriminates pre-event windows
  from matched null windows with AUC ≥ 0.7. H0 (the random-walk reading): AUC ≤ 0.6.
- **Event.** Defined before running: E = first generation at which a GMM component weight
  falls below 1/(10K) (mode loss). Window: the w = 20 generations before E.
- **Null.** Windows of the same length from (a) accumulate runs and (b) replace runs that had
  no event within the horizon, at matched generation index. Without (a) and (b) the
  evaluation is conditioned on collapse (MCR-G3).
- **Design.** Toy from MCR-01, replace regime, n ∈ {50, 100, 300}, temperature ∈ {1.0, 0.9} (a
  sampler with temperature < 1 introduces a systematic tail-shrinking push), 500 seeds; EWS
  windows w ∈ {10, 20, 40}. Report all (n, temperature, w) cells.
- **Pre-registered prediction.** AUC in [0.4, 0.6] at temperature 1.0 (no bifurcation, only a
  random walk); at temperature 0.9 rolling variance τ is negative (variance shrinks, opposite
  to CSD) and AC1 τ is not informative. **Falsified if** AUC ≥ 0.7 for both τ statistics at
  temperature 1.0 in ≥ 2 of 3 n values.
- **Analysis.** AUC with bootstrap CI; report τ distributions in event and null windows;
  sensitivity to w.
- **Threats.** The event definition drives the result; report a second event (tail mass below
  50% of initial) as a robustness check, decided now, not after seeing the first.
- **Tier.** Laptop CPU.
- **Would establish:** whether the CSD toolkit applies to recursive refitting at all.
  **Would not establish:** anything about EWS in the repo's basin-repair dynamics, where the
  question is posed separately (EXP-LLG-01).
- **Sketch:** EWS functions (`ews`, `kendall_tau`, `ac1`) are in
  [`sketches/mcr_recursive_gaussian.py`](sketches/mcr_recursive_gaussian.py); GMM and event
  code are not written.

### EXP-MCR-03 — Does asymmetric (GMR-style) cleaning slow minority tail loss across generations?

Gap: MCR-G5.

- **Hypothesis.** H1: across recursive generations of (features, labels) with an 80/20 class
  split, the repo's `DataManifold` cleaning retains minority-class recall and minority feature
  variance better than symmetric neighbourhood editing (Wilson-style, same k) and no cleaning.
  H0: minority recall at generation T differs by < 0.05 between arms.
- **Design.** Generation t: fit a classifier and a class-conditional generator (Gaussian per
  class) on the current data, sample n new points, apply the arm's cleaning, train the next
  generation on the result (replace regime; an accumulate arm as a control). Arms: none,
  symmetric editing, `manifolds/data_manifold.py` with the default thresholds
  (`confidence_threshold_majority: 0.7`, `confidence_threshold_minority: 0.3`). n = 200 (the
  repo's feature set size), T = 50, 100 seeds. Evaluate on a fixed held-out real test set.
- **Pre-registered prediction.** At T = 50, minority recall under GMR ≥ symmetric + 0.05 and
  minority variance ratio ≥ symmetric + 0.1. **Falsified if** GMR is not better than symmetric
  editing on either metric (95% bootstrap CI of the difference includes 0).
- **Threats.** GMR keeps more minority points by design; check that the gain is not only
  from a larger minority sample (add a symmetric arm with matched retention counts).
- **Tier.** Laptop CPU.
- **Would establish:** whether the repo's asymmetric cleaning protects tails in a recursive
  loop at toy scale. **Would not establish:** effects on generative language models.
- **Reuses:** `manifolds/data_manifold.py` (`DataManifold`), `simulation/environment.py`
  (feature-space data with the 80/20 split).
