# INVENTORY — what basin repair measures, triggers, does, retains

Emitted by `sims/drift_spectrum/inventory.py` (work order section 2). `[MECHANICAL]`
lines are read from source and re-checked by `inventory.py --selftest`; `[READING]`
lines are sentences about what the code means, with the source beside them.

## 1. What is MEASURED as drift

- [MECHANICAL] `repair_step` returns metrics ['task_loss', 'safety_loss', 'curvature', 'confidence', 'dist_to_ref', 'repair_cost_seconds'] (`manifolds/parameter_manifold.py`).
- [MECHANICAL] `dist_to_ref` = `LA.norm(theta_new - self.theta_ref).item()` — the L2 norm of theta minus the reference, taken on the post-step theta.
- [READING] Units: raw parameter units (weights are dimensionless). Window: NONE — the
  quantity is instantaneous, cumulative from the reference, recomputed every step. There is
  no windowed drift quantity in the core loop.
- [MECHANICAL] The only defined windows anywhere in the repair path are all **10 steps**: `Monitor.detect_cost_spike(window=10)`, `RepairCostMonitor.recent_trend(window=10)` in the addon, and `GenericRepairController._recent_trend(window=10)`. All three are rolling-mean ratios over repair COST, not over theta.
- [READING] Consequence for this test (order §2, "the test uses THAT window"): drift itself has
  no window to inherit, so A1/TIME takes W = 10 from the one window the repo does define,
  and says so on every emitted spectrum. This is a transfer, recorded as one.
- [MECHANICAL] Confidence = `torch.exp(torch.tensor(-self.lambda_curv * risk - dist))` — `exp(-lambda_curv * risk - dist)`, where `risk` is the softmax-variance curvature proxy and `dist` the same L2 norm.
- [MECHANICAL] Addon quantities (`addon_thermodynamic_control/energy.py`, `stability.py`,
  `repair/generic_repair_controller.py`): `basin_kl` (KL of outputs from the reference on
  safety inputs), `repair_energy = delta^T diag(F) delta` per step and its cumulative sum,
  `kappa_eff` (effective repair curvature), `lambda_max` of the safety Hessian by power
  iteration. None is a function of theta alone; all are functions of theta on a fixed input set.

## 2. What TRIGGERS a repair event

- [MECHANICAL] `repair_step` is called 1 time per step in `simulation/controller.py`, guarded only by `if self.param_layer:` (unconditional-per-step = True). **Repair runs every step.**
- [MECHANICAL] `repair_triggered` is a logged FLAG, set by `param_metrics['confidence'] < 0.5` — the constant 0.5 is hardcoded in the controller and is in no config file — OR by `policy_layer.needs_repair(policy_conf)`, whose threshold is `confidence_threshold` (code default 0.4, `default.yaml` 0.4, `adversarial.yaml` 0.5).
- [READING] So there is no trigger in the sense the order asks about ("threshold? schedule?
  detector?"): the schedule is every step, and the thresholded quantities produce a LABEL on
  a step that ran anyway. The label is what §4's overlay can use; nothing else exists.
- [MECHANICAL] `PolicyManifold.reanchor` is defined (True) and called 0 times in the controller. The policy layer's repair action is never executed by the core loop; only its flag is read.
- [MECHANICAL] The addon assigns a `phase` label in {stable, threshold, critical} from
  kappa / basin_kl / trend thresholds (`energy.py`, `stability.py`, `generic_repair_controller.py`).
  `sims/kappa_eff_leading/FINDING.md` records kappa_eff REFUTED as a leading indicator of
  basin breach across three rounds. Those labels are not used here.

## 3. What a repair DOES to theta

- [MECHANICAL] One gradient step on `task_loss - self.lambda_asym * weighted_safety` (the saddle-point sign is intentional, per CLAUDE.md), `delta = -lr * grad`, then an L2 clamp `||delta|| <= trust_radius` (clamp present = True). Defaults: `trust_radius` 0.05, `lr` 0.01 (`configs/default.yaml`).
- [READING] Every step therefore moves theta by at most `trust_radius` in L2. In the
  delta-theta vocabulary of this test, the repo already produces exactly one per-step
  delta vector per step; it does not keep it (section 5).

## 4. Whether any SPECTRAL quantity is already computed

- [MECHANICAL] Eigen/SVD call sites in the repo: 12, of which 0 are in the repair path (`manifolds/`, `simulation/`, `repair/`, `addon_*/`). Sites: ['research_interface/coupling_coherence.py:239', 'sims/dark_constraint/run.py:86', 'sims/shape_shadow/run.py:79', 'sims/shape_shadow/run.py:133', 'sims/shape_shadow/run.py:145', 'sims/shape_shadow/run.py:188', 'experiments/toy_landscape_v3.py:161', 'experiments/toy_landscape.py:124', 'experiments/toy_landscape.py:125', 'experiments/toy_landscape_v2.py:87', 'docs/theoretical_notes/mean_field2.py:57', 'docs/theoretical_notes/mean_field.py:50'].
- [READING] Those decompose a coupling network's Laplacian (`research_interface/`), toy-landscape
  safety Hessians (`experiments/toy_landscape*.py`), Procrustes cross-covariances of point sets
  (`sims/dark_constraint`, `sims/shape_shadow`) and a mean-field Jacobian (`docs/theoretical_notes`).
  None is a covariance of theta over any axis.
- [MECHANICAL] Power-iteration sites: 8 — the top eigenvalue of the safety-loss Hessian (`SpectralCertificate`), a single number, not a spectrum.
- [MECHANICAL] `FisherMetricEstimator.spectral_norm_approx` returns `return diag.max().item()` — the largest DIAGONAL Fisher entry, named as a spectral norm.
- [MECHANICAL] RankMe / effective-rank / alpha-ReQ sites: none.
- [READING] No covariance of theta over any axis is computed anywhere. The two metrics the
  order imports as reference instruments have no counterpart in the repo; nothing is being
  compared against an existing number.

## 5. Whether checkpoints / histories are RETAINED

- [MECHANICAL] `torch.save` call sites: none. No theta is written anywhere.
- [MECHANICAL] `Monitor` appends the metrics dict per step (True) with a `step` key (True) and writes `results/metrics.csv`; the word `theta` occurs in `monitors.py` = False. `GenericRepairController._history` is a `list[RepairState]` of scalar metrics (including `delta_norm`), not of theta.
- [READING] Per-step METRICS are retained with a step index, so a repair-flag timeline against
  training step exists and §4's overlay CAN run on the flag. Theta and delta-theta are
  discarded on every step, so no drift SPECTRUM can be computed from anything the repo
  retains today: this test has to record theta itself, which Tier 1 does in its own loop.

## 6. What follows for the test

- Window for A1/TIME: 10 steps, transferred from the cost-trend window (§1 above), declared.
- Repair-event overlay (§4 of the order): uses the repo's two flag rules — parameter
  confidence `exp(-lambda_curv*curv - dist) < 0.5` and policy confidence
  `1 - JS/ln2 < 0.4` — re-implemented in pure Python on the Tier 1 run with the reference
  set to theta_0. That is NOT the repo's scenario (there, theta starts drifted from an
  aligned reference and is pulled back); on a training run from init both quantities are
  monotone in dist, so each flag crosses once. The overlay reports WHERE that crossing
  falls relative to the spectral transitions, which is the only overlay the flag supports.
- Nothing in sections 1–5 changes under `adversarial.yaml` except the constants quoted.
