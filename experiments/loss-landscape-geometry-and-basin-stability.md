# Loss-landscape geometry and basin stability — experiment designs

Status: PROPOSED (all experiments). Date 2026-11-24. Co-authored (human + AI).
Draft read: `hypotheses/loss-landscape-geometry-and-basin-stability.md` (98 KB) at
`02320b51b9138dec032f8f85839bce420502a9ec` (regenerated 2026-11-23T16:38:05Z).
Engine-level gaps: [`engine-evidence-audit.md`](engine-evidence-audit.md).

## 1. What the engine claims, and on what evidence

"Established" with 249 surviving claims (262 claims in the tree for this topic; 239 from
arXiv, 14 Crossref, 9 Semantic Scholar). 31 surviving claims predate provenance stamping, i.e.
were staked under the config that `docs/research/HYPOTHESIS_TRIAGE.md` found 86% off-topic;
they remain because the tree has no retirement path. 7 surviving claims have no abstract
sentence.

A title keyword filter (`loss landscape|flat|sharp|mode connect|basin|trust region|hessian|
curvature`, run 2026-11-24) matches 19 of the 262 claims; several of those are other fields
(pore-scale flow, asymptotic flatness in gravitation, a flatness interferometer, Riccati PDEs).
This is a title match, not a relevance label; EXP-ENG-01 is the measurement.

Spot checks (arXiv API / Crossref, 2026-11-24):

| cited item | checked? | what the record shows |
|---|---|---|
| Emergent heavy fermion and superconductivity near Mott transition in twisted bilayer graphene, arXiv 2608.12319 | exists | Condensed-matter paper. Unrelated. Survived (3 passes). |
| On the Superlinear Relationship between SGD Noise Covariance and Loss Landscape Curvature, arXiv 2602.05600 | exists | Argues the common assumption Fisher ≈ Hessian (so SGD noise C ∝ H) holds only under restrictive conditions typically violated in deep nets, and derives C ∝ E_p[h_p²] (per-sample Hessians). Directly relevant to this repo, which uses a Fisher diagonal as metric and a Hessian Rayleigh quotient as curvature (EXP-LLG-02). |
| Loss Landscape Geometry of PDE Emulators: Or, Symmetry Learning via Gradient Alignment, arXiv 2601.20172 | exists | Influence-based diagnostic (metric-weighted gradient overlap along group orbits) for whether training selects a symmetry-compatible basin. Adjacent. The draft lists it as "Loss Landscape Geometry and the Learning of Symmetries: Or, What Influence Functions Reveal About Ro…" (DOI 10.48550/arXiv.2601.20172); the arXiv v2 title differs. Likely a version-title change; UNCHECKED. |
| Connecting phases of matter to the flatness of the loss landscape in analog variational quantum algorithms, doi 10.1038/s42005-026-02528-4 | exists | *Communications Physics*, 2026-02-18. Quantum VQA landscapes; adjacent, different substrate. |
| Decomposing SGD Dynamics in Neural Networks: Teacher-Induced Spikes and Variance Inflation, arXiv 2611.24908 | exists | Two-layer nets, multi-index teacher, spectral spikes in the orthogonal complement Gram matrix. Adjacent. |
| Exploring Flatter Loss Landscape Surface via SAM with Linear Mode Connectivity; QUATRO; Single-domain generalisation via path flatness | not checked | UNCHECKED. |

Anchor literature the engine did not retrieve but the repo's claims rest on (existence
checked via arXiv API, 2026-11-24; content not re-read here): Keskar et al. 1609.04836 (sharp
minima and large batch), Dinh et al. 1703.04933 (sharp minima can generalise; flatness is
not reparameterisation-invariant), Garipov et al. 1802.10026 and Draxler et al. 1803.00885
(mode connectivity), Jiang et al. 1912.02178 and Andriushchenko et al. 2302.07011 (sharpness
vs generalisation measured at scale).

## 2. What the repo claims under this topic

- Safe basin `B = {θ : KL(f_θ ‖ f_θ₀) < ε}` (function space).
- `κ_eff = θ̇ᵀ H θ̇ / θ̇ᵀ θ̇` as a leading indicator ("Spike in kappa_eff precedes behavioral
  collapse", `addon_thermodynamic_control/stability.py`).
- Status on record: κ_eff REFUTED in the committed regime (`sims/kappa_eff_leading/FINDING.md`,
  three rounds); escape is rate-induced with σ_crit ≈ 0.00084 measured against a predicted
  0.00120 (`sims/rate_induced_escape/FINDING.md`, SUPPORTED); Q1 (κ_eff near σ_crit) UNTESTED;
  Q3 (`C_bound = 20` inert) MEASURED; Q4 (scale) UNTESTED (`docs/research/OPEN_QUESTIONS.md`).

## 3. What science needs to be known

### (a) Gaps the engine recorded

- Off-topic retrieval for this topic (recorded in HYPOTHESIS_TRIAGE and the `query_arxiv`
  docstring; category filter added). The OR-expansion of query words (ENG-G1) is not recorded.
- `claims_predating_provenance` is reported each run (66 tree-wide at this commit).

### (b) Gaps identified here

| id | gap | kind |
|---|---|---|
| LLG-G1 | **Q1: no fair test of any indicator near σ_crit.** All κ_eff rounds ran at σ = 0.012–0.020, 10–24× above critical. The one retrieved on-topic EWS limit (Sujith 2025, see the model-collapse file) says EWS can fire only after the bifurcation when the control parameter moves fast, which is exactly the committed regime. | missing condition |
| LLG-G2 | **Fisher vs Hessian.** The repo uses the Fisher diagonal for metric/trust radius and the safety-KL Hessian for curvature, and the terminology map treats them as one family. Whether they agree along repair trajectories on ToyLLM is unmeasured; arXiv 2602.05600 argues they need not. | unstated assumption |
| LLG-G3 | **Coordinate dependence of κ_eff.** The basin is defined in function space (invariant to reparameterisation); κ_eff is a Euclidean Rayleigh quotient in parameter space. ToyLLM has an exact ReLU rescaling symmetry, so κ_eff can change while the function, the basin and the KL do not. Also, `effective_repair_curvature` returns `abs(κ)`, discarding the sign of curvature. | structural |
| LLG-G4 | **Q3/Q4 ride-alongs.** A scale-relative bound for the phase classifier (Q3) and the same indicator battery at 1–2 orders of magnitude more parameters (Q4) are both gated on Q1. | scope |
| LLG-G5 | **Event definition.** A "leading indicator" needs an event. `KL > ε` with ε = 0.15 is a convention; `sims/rate_induced_escape` uses the sign of net ΔKL/step instead. Lead times depend on which is used; neither is pre-registered as the event for Q1. | operationalisation |

## 4. Experiments

### EXP-LLG-01 — Q1: indicator battery near σ_crit, with a sub-critical null

Gap: LLG-G1, LLG-G5 (and carries EXP-FIT-03 and the Q3 ride-along).

- **Hypothesis.** H1 (Theory A of `sims/kappa_eff_leading/REFUTE.md`, transported): at
  σ ∈ [0.0006, 0.0016], at least one signal alarms before the event and beats the θ-distance
  baseline's lead at ≥ 80% of supercritical (seed, σ) cells, under a criterion whose
  false-alarm rate on the sub-critical and zero-drift arms is ≤ 0.2. H0 (Theory B): no signal
  does under any viable criterion. Anything else: INCONCLUSIVE.
- **Signals.** κ_eff; λ_max and λ_min of the safety Hessian (power / Lanczos on HVPs);
  saturation margin ρ = drift ΔKL ÷ repair cap; time-to-boundary (ε − KL)/(dKL/dt); Fisher
  speed and `recent_trend()` of `C` (EXP-FIT-03); θ-distance (free baseline); KL itself
  (second baseline, since a KL-threshold monitor already exists).
- **Event.** Pre-register two events and grade both: E1 = first step with KL > ε (ε = 0.15);
  E2 = first step after which a causal 20-step mean of net ΔKL/step stays > 0. A signal
  "leads" only if it leads the event it is graded on.
- **Design.** σ ∈ {0.0006, 0.0008, 0.0010, 0.0012, 0.0014, 0.0016}, isotropic and adversarial
  modes, 10 seeds; zero-drift null arm (10 seeds) and the sub-critical cells (σ ≤ 0.0008,
  where net ΔKL/step was ≤ 0 at σ = 0.0005 in the recorded sweep) as the false-alarm arm.
  Horizon: from the recorded net rates (+3.5e−5/step at σ = 0.0015) and a KL gap of order
  0.05, breach can take thousands of steps; set the horizon from a calibration run so that
  ≥ 80% of supercritical runs reach E1, and record that calibration before grading
  (calibration fixes the observation window only, as in the existing sims). Criteria: the
  7-criterion sweep of `sims/kappa_eff_leading/config.json`, unchanged, plus smoothing windows
  {1, 3, 5}. `mu_mode ∈ {adaptive, frozen}` (Q2) as a diagnostic arm.
- **Pre-registered prediction.** Following Sujith (2025) and the committed-regime result, no
  signal other than KL and time-to-boundary meets the support rule for E1; ρ leads E2 by
  construction (it is the same quantity smoothed) and is excluded from grading on E2 for that
  reason. **Falsified (for the prediction) if** κ_eff or λ_max meets the support rule on either
  event.
- **Analysis.** Per-criterion verdict table (as in the existing FINDING); lead-time
  distributions with bootstrap CIs; false-alarm rates per arm. Grading code written and
  committed before the sweep runs. Q3 ride-along: phase classifier with bound = k × running
  median κ_eff, k ∈ {2, 3, 5}, compared to `basin_kl` alone by balanced accuracy on E1.
- **Threats.** Near σ_crit, outcomes are noisy and slow; with 10 seeds per cell, an 80% rule
  is a coarse bar (8 of 10). Sub-critical runs that never breach still need a horizon matched
  to the supercritical ones or false-alarm rates are not comparable (Boettiger & Hastings,
  "Early warning signals and the prosecutor's fallacy", *Proc R Soc B* 2012, exists, checked
  via Crossref: a sample conditioned on transitions inflates apparent EWS performance).
- **Tier.** Single CPU, hours (order of 10⁵ controller steps at ≈ 0.03–0.04 s/step inferred from the
  recorded κ_eff runtime of ~68 s; estimate, not measured).
- **Would establish:** whether any curvature signal earns its HVP cost in the one regime where
  leading is possible. **Would not establish:** anything at model scale (Q4), or for drift
  that is not a random walk in parameter space.
- **Reuses:** `sims/kappa_eff_leading/run.py` (criteria, grading, null arm),
  `sims/rate_induced_escape/run.py` (σ_crit calibration, `mu_mode`),
  `addon_thermodynamic_control/stability.py`. Planned home: `sims/critical_regime_indicators/`
  (named in OPEN_QUESTIONS, not yet built), with its own REFUTE.md written before the run.

### EXP-LLG-02 — Fisher, Hessian and gradient-noise covariance along repair trajectories

Gap: LLG-G2.

- **Hypothesis.** H1: on ToyLLM, the safety-KL Hessian H, the model Fisher F and the
  gradient-noise covariance C share top-k eigenspaces (subspace overlap ≥ 0.8 for k = 10) only
  near θ_ref, and diverge as KL to the reference grows. H0: overlap ≥ 0.8 everywhere on the
  trajectory.
- **Variables.** At checkpoints along default and adversarial runs: top-20 eigenpairs of H
  (Lanczos on HVPs), F (exact, 3,152 × 3,152), C (per-example gradient covariance on the task
  loss), and the diagonal of F as the repo uses it. Overlap = ‖U_Aᵀ U_B‖²_F / k. Also the
  quantities 2602.05600 predicts: C vs E_p[h_p²] vs H.
- **Design.** 5 seeds × 2 configs × 10 checkpoints; at θ_ref, the Hessian of the repo's safety loss
  KL(f_ref ‖ f_θ) equals F(θ_ref) (standard identity; DERIVED), which is the built-in positive
  control: overlap must be ≈ 1 there or the code is wrong.
- **Pre-registered prediction.** Overlap(H, F) ≥ 0.9 at θ_ref and declines monotonically with
  KL; overlap(diag F, F) top-k is < 0.5 everywhere (ToyLLM weights are densely coupled).
  **Falsified if** overlap(H, F) stays ≥ 0.8 across the full KL range observed.
- **Tier.** Laptop CPU.
- **Would establish:** whether the repo's metric and its curvature signal measure the same
  thing at toy scale. **Would not establish:** the result at scale, or for the task loss
  (where F = H does not hold even at the reference).

### EXP-LLG-03 — Coordinate dependence of κ_eff under an exact symmetry

Gap: LLG-G3.

- **Hypothesis.** H0 (reading implied by calling κ_eff a basin property): κ_eff is invariant
  to the ReLU rescaling (W1, b1, W2) → (αW1, αb1, W2/α) within 5%. H1: κ_eff varies with α
  while logits and KL are unchanged to float precision.
- **Design.** α ∈ {0.25, 0.5, 1, 2, 4}; θ, θ_ref and the drift direction are mapped by the
  same linear map; 20 seeds of (θ_ref, direction, inputs). Second arm: repeat the
  κ_eff-leading criterion sweep on rescaled runs (α = 0.5, 2) to see whether verdicts change
  under a change of coordinates with no change of function. Report the sign of the Rayleigh
  quotient before `abs()` as an additional column.
- **Pre-registered prediction.** Max logit difference < 1e−4 at every α; κ_eff ratio κ(α)/κ(1)
  outside [0.95, 1.05] for α ∉ {1} on ≥ 90% of seeds. **Falsified if** κ_eff is within 5% at all
  α on ≥ 90% of seeds.
- **Tier.** Laptop CPU (seconds for arm 1).
- **Would establish:** whether κ_eff is a function-space or a coordinate quantity. **Would not
  establish:** whether it is a useful indicator in fixed coordinates (that is LLG-01); a fixed
  parameterisation in practice may make the dependence harmless.
- **Sketch:** [`sketches/llg_rescaling.py`](sketches/llg_rescaling.py) — reuses
  `simulation/environment.py::model_fn` and
  `SpectralCertificate.effective_repair_curvature` unmodified; `--selftest` checks function
  invariance and invertibility of the map; the α sweep has not been run.

### EXP-LLG-04 — Q4: the same battery at 1–2 orders of magnitude more parameters (gated)

Gap: LLG-G4. Runs only after LLG-01 reports. If LLG-01 refutes every curvature signal near
σ_crit, OPEN_QUESTIONS already notes that "scale is a weaker hypothesis than it looks"; record
that and do not run. If any signal is supported: MLP of 10⁵–10⁶ parameters or a small
pretrained LM fine-tuned under injected drift, with the same σ-relative-to-σ_crit design
(σ_crit re-measured at that scale first, by the `rate_induced_escape` method), HVP cost
measured per step. Tier: single GPU. Would establish transfer of a toy result to one larger
model, not to production scale.
