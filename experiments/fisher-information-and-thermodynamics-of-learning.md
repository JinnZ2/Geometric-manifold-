# Fisher information and thermodynamics of learning — experiment designs

Status: PROPOSED (all experiments). Date 2026-11-24. Co-authored (human + AI).
Draft read: `hypotheses/fisher-information-and-thermodynamics-of-learning.md` at
`02320b51b9138dec032f8f85839bce420502a9ec` (regenerated 2026-11-23T16:38:05Z).
Engine-level gaps (retrieval, lexical corroboration, statement selection) are in
[`engine-evidence-audit.md`](engine-evidence-audit.md) and are not repeated here.

## 1. What the engine claims, and on what evidence

The draft is "established" with 14 surviving claims. Its `> statement` line is the
satellite-infrastructure-sharing paper (beta 0.80), selected by insertion order (ENG-G6).
Each claim is a paper's own first sentence, so the draft makes no proposition about Fisher
information or the thermodynamics of learning; it is a reading list.

Spot checks of cited sources (metadata via Crossref or the arXiv API, 2026-11-24):

| cited item (as in draft) | checked? | what the record shows |
|---|---|---|
| Fisher information flow in artificial neural networks, doi 10.1103/kn3z-rmm8 | exists | Weimar et al., *Physical Review X*, 2025-09-16. Abstract: Fisher information as a quantity that propagates through a physical system and sets estimation precision limits; applied to ANNs used for parameter estimation. On-topic for Fisher information; not about thermodynamics of learning. |
| Deep Neural Network Training With Distributed K-FAC, doi 10.1109/TPDS.2022.3161187 | exists | Pauloski et al., *IEEE TPDS*, 2022. A systems paper on scaling a Kronecker-factored Fisher approximation. Relevant to Fisher-as-metric cost, not to its interpretation. |
| Information geometry of evolution of neural network parameters while training, doi 10.1016/j.neucom.2024.128007 | exists | Thiruthummal, Kim, Shelyag, *Neurocomputing*, 2024. Abstract not returned by Crossref; content UNCHECKED. |
| Satellite Infrastructure Sharing ..., arXiv 2608.12265 | exists | LEO satellite coverage via spherical Cox–Boolean models. Unrelated to the topic. |
| FisherMask; Fast post-training pruning based on Fisher information | not checked | UNCHECKED. |
| Remaining 8 (MIMO detection, quantum private capacity, atmospheric fronts, session handover, agent evaluation, prefix sliding, global sensitivity indices, quantum estimation) | titles only | Titles indicate other fields. Relevance not labelled; that is EXP-ENG-01. |

No surviving claim concerns a thermodynamic quantity (heat, entropy production, work,
Landauer cost) in learning. The literature that exists for that link was not retrieved,
e.g. Goldt & Seifert, "Stochastic Thermodynamics of Learning", *PRL* 118, 010601 (2017)
(exists, checked via Crossref), and Crooks, "Measuring Thermodynamic Length", *PRL* 99,
100602 (2007) (exists, checked via Crossref). Their exact statements have not been
transcribed here: VERIFY before using either as a pre-registered criterion.

## 2. What the repo itself claims under this topic

The engine does not touch repo code, but the topic exists to ground
`addon_thermodynamic_control/energy.py`:

- `FisherMetricEstimator.diagonal` — diagonal Fisher from labels sampled from the model
  (the "true" Fisher, not the empirical Fisher; cf. Kunstner et al., arXiv 1905.12558,
  exists), 16 samples by default.
- `RepairEnergyAccumulator` — `C_repair = Σ_t δ_tᵀ G(θ_t) δ_t`, called "Discrete Riemannian
  kinetic energy integral", with `recent_trend() > 2.0` described as "nonlinear spike" and
  "early warning of phase transition".
- README terminology map: "Fisher energy accounting `C = δᵀGδ` | Natural-gradient metric;
  Mahalanobis step cost".

## 3. What science needs to be known

### (a) Gaps the engine recorded

- None specific to this topic. The draft lists three escape-hatched unknowns (quantum state
  tomography, prediction-driven allocation, RobustRLlib), all outside the topic.

### (b) Gaps identified here

| id | gap | kind |
|---|---|---|
| FIT-G1 | **What `C` is.** For small δ, `δᵀFδ ≈ 2·KL(p_θ ‖ p_{θ+δ})` (standard; DERIVED, not measured in this repo). `C` is then a sum of per-step divergences in nats, an action-like quantity. Calling it "energy" brings in no physical units; the README's "Mahalanobis step cost" is the accurate reading. | structural / dimensional |
| FIT-G2 | **Discretisation dependence.** For a fixed path cut into N equal steps, `Σ δᵀFδ ≈ A/N` while the Fisher length `L = Σ √(δᵀFδ)` does not depend on N. So `C` and its trend change with learning rate and step count even when the path does not. A rise of `recent_trend()` above 2.0 can therefore be caused by a ×√2 step-size change with no change in the landscape. | mathematical; needs a numerical check on repo trajectories |
| FIT-G3 | **Estimator fidelity.** Diagonal approximation and 16 Monte-Carlo label samples. Unmeasured: error of `δᵀ diag(F̂) δ` against the full Fisher and against measured KL; spread across resampling seeds; whether spike decisions are stable across seeds. Code note (OBSERVED by reading `energy.py`): the loop runs `min(n_samples, len(inputs))` times but divides by `n_samples`, so with fewer than 16 inputs the estimate is scaled by `len(inputs)/16`. The default safety batch is 32, so default runs are not affected; effect elsewhere unmeasured. | measurement |
| FIT-G4 | **Leading-indicator claim.** "Early warning of phase transition" for the energy trend has no causal-threshold test with a null arm. The same defects that `sims/kappa_eff_leading/REFUTE.md` found for κ_eff (look-ahead, cannot fail to fire, no baseline) would apply to any whole-run threshold. | missing control |
| FIT-G5 | **Thermodynamics of learning proper.** The link between information-geometric cost and physical dissipation exists in stochastic thermodynamics for Langevin-type dynamics at a stated temperature (Goldt & Seifert 2017, VERIFY statement). The repo's controller is deterministic gradient steps with no temperature, so that framework does not apply as written. Physical energy of hardware (joules) is a different quantity again and is out of scope for software experiments. | scope / unstated assumption |

## 4. Experiments

### EXP-FIT-01 — Is repair energy a path property or a step-size property?

Gap: FIT-G1, FIT-G2.

- **Hypothesis.** H1 (from the mathematics): replaying the same recorded parameter path at
  sub-sampling factor m multiplies `C` by ≈ m and leaves the Fisher length `L` within a few
  percent. H0 (the reading implied by "energy" as a path cost): `C` is invariant to m within
  10%.
- **Variables.** `C_m = Σ δᵀ G δ` and `L_m = Σ √(δᵀ G δ)` over the path sub-sampled every m-th
  checkpoint, m ∈ {1, 2, 4, 8}; `Σ 2·KL` per step measured directly on the safety inputs; G
  as computed by the repo (`FisherMetricEstimator.diagonal`) and, at ToyLLM size (3,152
  parameters), the exact full Fisher.
- **Design.** Part 1 (no repo code): the numpy sketch with an exact Fisher on softmax
  regression, curved fixed-endpoint paths, N ∈ {5,…,160}, 10 seeds. Part 2: run
  `main.py --config configs/default.yaml` and `configs/adversarial.yaml` with θ saved every
  step (a wrapper outside `simulation/`; do not edit the controller), 5 seeds each, then
  recompute `C_m`, `L_m`. Part 3 (null arm for the trend detector): same runs with the repair
  `lr` multiplied by √2 at step 50 and nothing else changed; count how often
  `recent_trend() > 2.0` fires within 20 steps of the change versus in matched runs without
  the change.
- **Pre-registered prediction.** `C_m / C_1` within [0.8m, 1.2m] for m ≤ 4 on all seeds;
  `L_m / L_1` within [0.9, 1.0] (chords are never longer than the path). In Part 3, the
  trend detector fires after the lr change in ≥ 80% of seeds and in ≤ 20% of matched control
  seeds. **Falsified if** `C_m / C_1` stays within [0.9, 1.1] (C behaves as a path
  invariant), or the lr change does not move the trend ratio.
- **Analysis.** Report ratios per seed; no pooling needed for Part 1–2. Part 3: two-sided
  Fisher exact test on fire counts, 10 seeds per arm.
- **Threats.** Repo paths are not smooth (trust-region clipping, data cleaning steps), so the
  ≈ m scaling holds only where the path is locally straight; report curvature of the
  path (angle between successive δ) alongside. Diagonal Fisher can change the constant but not
  the scaling with m.
- **Tier.** Laptop CPU (minutes).
- **Would establish:** whether `C` and its trend measure the path or the stepping. **Would
  not establish:** anything about physical energy, or whether a Fisher-length-based monitor
  would be a good early warning (that is EXP-FIT-03).
- **Sketch:** [`sketches/fit_discretization.py`](sketches/fit_discretization.py)
  (`--selftest` checks `δᵀFδ ≈ 2KL`, PSD Fisher, convergence of `L_N` and `N·C_N`; Part 1
  itself has not been run).
- **Reuses:** `addon_thermodynamic_control/energy.py` (`FisherMetricEstimator`,
  `RepairEnergyAccumulator`), `simulation/environment.py::model_fn`, `main.py`.

### EXP-FIT-02 — Fidelity and stability of the diagonal, 16-sample Fisher

Gap: FIT-G3.

- **Hypothesis.** H1: per-step `δᵀ diag(F̂₁₆) δ` is within a factor of 2 of `2·KL` measured
  directly on ≥ 90% of steps, and spike decisions agree across Fisher resampling seeds on
  ≥ 90% of steps. H0: either fails.
- **Variables.** For each step of a recorded repair path: exact full Fisher quadratic form,
  exact diagonal form, sampled diagonal form with n ∈ {4, 16, 64, 256}, and `2·KL` measured;
  `recent_trend() > 2` decision per step under 20 resampling seeds; batch size ∈ {8, 16, 32}
  to expose the divisor behaviour noted in FIT-G3.
- **Design.** ToyLLM default and adversarial configs, 5 trajectory seeds × 20 Fisher seeds.
  Full Fisher via per-example Jacobians (3,152 × 3,152 matrix; ≈ 80 MB in float64).
- **Pre-registered prediction.** Sampled-diagonal error is dominated by the diagonal
  approximation, not by n, once n ≥ 16; decision agreement ≥ 90% at n = 16. At batch 8 the
  sampled estimate is ≈ 0.5× the batch-32 estimate (from the divisor), detectable as a
  constant offset. **Falsified if** agreement < 90% at n = 16, or the batch-8 offset is
  absent (meaning the code reading in FIT-G3 is wrong).
- **Analysis.** Median and 90th percentile of |log ratio| per estimator; Fleiss' κ of spike
  decisions across seeds.
- **Threats.** KL measured on the same 32 safety inputs used for the Fisher, so agreement can
  be optimistic; add a held-out input set.
- **Tier.** Laptop CPU (≲ 1 h).
- **Would establish:** whether the monitor's numbers are stable enough to threshold.
  **Would not establish:** whether thresholding them is useful.
- **Reuses:** `addon_thermodynamic_control/energy.py`, `addon_thermodynamic_control/stability.py`
  (`_fisher_diag` uses `n = min(8, len(inputs))`; include it as a second estimator).

### EXP-FIT-03 — Does any Fisher-based quantity lead basin exit, against a free baseline?

Gap: FIT-G4. Runs inside EXP-LLG-01 (same harness, same σ grid, same null arm), as two
extra signals: `recent_trend()` of `C`, and the Fisher speed `√(δᵀGδ)` per step. Pre-registered
criteria, verdict rules and the θ-distance baseline are those of EXP-LLG-01; see
[`loss-landscape-geometry-and-basin-stability.md`](loss-landscape-geometry-and-basin-stability.md).
Prediction specific to this signal: given EXP-FIT-01, `recent_trend()` adds no lead over
θ-distance at fixed lr, because both are functions of step size. Falsified if it meets the
LLG-01 support rule while θ-distance does not.

### EXP-FIT-04 — The stochastic-thermodynamics analogue, where it applies

Gap: FIT-G5.

- **Hypothesis.** H1: for a perceptron trained by Langevin dynamics at temperature T in a
  teacher–student setting, the inequality stated by Goldt & Seifert (2017) between
  information acquired about the labels and total entropy production holds numerically, and
  the repo-style quantity `Σ δᵀFδ` computed on the same trajectories is (or is not)
  monotonically related to the entropy production. H0: no monotone relation (Spearman |ρ|
  < 0.3 across runs).
- **Pre-condition.** Transcribe the paper's inequality, definitions of heat and of the
  information term, and its assumptions, word-for-word with page/equation numbers, before
  any run. Until then the criterion is VERIFY and the experiment does not start.
- **Design.** N ∈ {20, 50, 100} input dimensions, P/N ∈ {0.5, 1, 2}, T ∈ {0.1, 0.5, 1}, 20 seeds
  per cell; Euler–Maruyama integration with dt halved until results change < 1%.
  Mutual-information estimate by the method the paper uses, if computable; otherwise
  report it as NOT_BUILT rather than substituting an estimator.
- **Pre-registered prediction.** Inequality holds in all cells (it is a theorem under its
  assumptions; a violation indicates an implementation error, not a discovery). The
  relation to `Σ δᵀFδ` is the open part; no direction is predicted.
- **Tier.** Laptop CPU.
- **Would establish:** whether the repo's metric connects to a thermodynamic quantity in the
  one regime where the theory is stated. **Would not establish:** anything about the
  deterministic controller, or about physical energy use of hardware.

## 5. Out of scope, stated

Physical energy (joules) of training or repair on hardware is not addressed. A software
counter (RAPL, NVML) measures device energy that is dominated by things other than the
information processed; a Landauer-type comparison would be many orders of magnitude apart and
would not discriminate between hypotheses. No experiment is proposed for it.
