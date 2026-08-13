# Terminology Map — Ecosystem Naming ↔ Standard Research Vocabulary

Purpose: a translation layer that makes this ecosystem legible to reviewers, collaborators, journals, and tooling — without changing any of the underlying work. Every concept here has a published home in the mainstream literature; this document just points at it.

How to use: paste the relevant section into each repo's README (e.g. as `## Relationship to prior work`), cite the anchor references, and keep your internal naming in code if you like — the map does the bridging so the code doesn't have to change.

---

## 1. Geometric-manifold- ("Basin Repair Framework")

| Your term | Standard term | Anchor literature |
|---|---|---|
| Basin repair framework | Trust-region-constrained fine-tuning with a reference anchor | Schulman et al., TRPO (2015); KL-regularized RLHF (Ouyang et al. 2022) |
| Safe basin B_θ = {θ : KL(f_θ‖f_θ₀) < ε} | KL-ball constraint / trust region in distribution space | TRPO (2015); also "behavioral cloning anchor" in offline RL |
| Parameter manifold repair | Projected/regularized optimization in weight space | Trust-region methods (Nocedal & Wright); elastic weight consolidation (Kirkpatrick 2017) |
| Drift detection | Distributional shift monitoring / runtime assurance | OOD detection literature; runtime assurance (Schierman et al. 2015) |
| κ_eff = \|v·Hv\|/v·v | Rayleigh quotient of the Hessian; spectral instability indicator | Hessian-eigenvalue analyses of training stability (Ghorbani et al. 2019; Cohen et al. "edge of stability" 2021) |
| Curvature-weighted safety loss | Second-order / curvature-aware regularization | Optimal Brain Damage/Surgeon lineage (LeCun 1990; Hassibi 1992); K-FAC (Martens & Grosse 2015) |
| Asymmetric repair penalty (λ_asym) | Cost-sensitive / asymmetric loss; constrained optimization via Lagrangian | Lagrangian safe RL (Altman 1999; Achiam et al. CPO 2017) |
| Saddle-form total loss (task − λ·safety) | Adversarial/Lagrangian saddle-point objective | Standard in safe RL and GAN literature |
| Thermodynamic control / free energy F | Composite loss with Lagrange multipliers; Lyapunov candidate | Lyapunov-based control (Khalil, *Nonlinear Systems*) |
| Fisher energy accounting C = δᵀGδ | Natural-gradient metric; Mahalanobis step cost | Amari natural gradient (1998); K-FAC |
| Energy trend spike detection | Change-point / anomaly detection on loss curves | CUSUM; early-warning-signal literature (Scheffer 2009) |
| GMR data cleaning (minority never dropped) | Neighborhood-based label noise filtering with class-conditional weighting | Wilson & Martinez editing (k-NN cleaning); SMOTE-adjacent imbalance literature |
| Policy manifold JS re-anchoring | Distributional alignment via JS divergence; exponential moving average target | JS-GAN lineage; target networks / EMA in RL |
| Phase classifier (stable/threshold/critical) | Regime classification on stability indicators | Early-warning-signal regime detection (Scheffer et al. 2009, 2012) |
| ISS_proof_pending | Input-to-state stability analysis (open problem) | Sontag, "Input to State Stability" (1989+); Jiang & Wang 2001 |
| Lyapunov certificate V = L_safety + μ·C_repair | Composite Lyapunov function | Khalil Ch. 4; control-barrier-function adjacency (Ames et al. 2017) |
| "Echo chamber of its own geometry" | Closed-loop self-training without external grounding | Model collapse literature (Shumailov et al., Nature 2024) |

**One-paragraph conventional summary (paste-ready):**
> This repository studies runtime assurance for neural networks under distribution shift. A reference model defines a KL trust region in behavior space; a controller applies curvature-regularized, trust-region-bounded updates that pull drifting parameters back toward the reference, with asymmetric penalties preserving minority-class behavior. A spectral statistic of the safety Hessian is evaluated as a leading indicator of instability. Stability guarantees (ISS, Lyapunov) are stated as open problems.

---

## 2. Cross-Domain-Toolkit

| Your term | Standard term | Anchor literature |
|---|---|---|
| Cascade regime audit | Early-warning-signal (EWS) detection of critical transitions | Scheffer et al., Nature 2009; Dakos et al. 2012 |
| Spinodal h* = 2/√27 | Saddle-node/fold bifurcation threshold of the cusp normal form | Catastrophe theory (Thom); Strogatz, *Nonlinear Dynamics* |
| COMMITTED / STRESSED / CASCADE regimes | Tipping-point regime classification | Lenton et al., tipping elements (2008); R-tipping literature |
| Coherence-under-contradiction signal | Rigidity/locking indicator under conflicting evidence | Belnap 4-valued logic; Dempster–Shafer conflict mass K |
| Falsification ledger | Append-only, hash-chained experiment/claim log with refutation protocol | Tamper-evident logs (Merkle trees); Popperian falsifiability operationalized; registered-reports adjacency |
| Escape-hatch flag | Detection of unfalsifiable reformulation patterns | Lakatos "degenerating research programmes"; Popper's conventionalist stratagems |
| Multi-substrate calibration gate, GROUND/PREDICT | Multi-sensor data fusion with source-role typing and abstention | JDL fusion model; Kalman/information filtering; covariance intersection (Julier & Uhlmann 1997) |
| Lε determinacy gate | Confidence-thresholded fusion with deferral | Selective prediction / reject option (Chow 1957); selective classification |
| Determinacy noisy-OR | Independent evidence pooling | Noisy-OR (Pearl 1988); subjective logic (Jøsang 2016) |
| Symbolic checker (logical_form) | Lightweight formal property verification | SMT-lite / safe expression evaluation; runtime verification |

**Paste-ready summary:**
> A stdlib-only toolkit for three classical problems: (1) falsifiability-disciplined claim tracking with tamper-evident provenance; (2) multi-sensor fusion with calibrated confidence, role typing, and abstention; (3) early-warning detection of critical transitions using the standard EWS battery (critical slowing down, variance, skew, flickering) plus fold-bifurcation thresholds from catastrophe theory.

---

## 3. Geometric-to-Binary-Computational-Bridge

| Your term | Standard term | Anchor literature |
|---|---|---|
| Geometric-to-binary bridge | Physics-informed compilation / physical computing substrate synthesis | Bond graphs (Paynter 1961); port-Hamiltonian systems (van der Schaft 2014) |
| Bond-graph IR with 6 substrates | Multi-energy-domain modeling intermediate representation | Modelica/multiphysics modeling; port-Hamiltonian theory |
| Exact/soft/composite couplers | Ideal vs lossy vs hierarchical junctions | Bond-graph junction structure (0/1-junctions, transformers, gyrators) |
| Emitters (KiCad/g-code/OpenSCAD) | Hardware description generation / automated synthesis to fabrication formats | HLS-adjacent; generative CAD |
| Verify (Farina sweep, unit CSV) | Frequency-response system identification | Farina log-sweep method (2000); SysID literature |
| GEIS octahedral 8-state encoder | 3-bit geometric state encoding on the octahedron's vertices | Group codes / permutation modulation (Slepian 1965) |
| Timestamped CSV contract (seq,micros,v0..vN,crc8) | Framed telemetry protocol with CRC integrity | Standard embedded telemetry (cf. MAVLink, SLIP+CRC) |

**Paste-ready summary:**
> A compilation pipeline from multi-domain physical models (bond-graph IR, equivalent to port-Hamiltonian structure) to fabricable artifacts (PCB layouts, toolpaths, printable geometry), with system-identification-based verification against physical units.

---

## 4. curly-octo-happiness

| Your term | Standard term | Anchor literature |
|---|---|---|
| Claim epistemics (Beta posterior) | Beta-Bernoulli belief tracking / Bayesian reliability estimation | Jøsang subjective logic; standard conjugate Bayesian updating |
| Escape-hatch reformulation counters | Unfalsifiability / ad-hoc-hypothesis detection | Lakatos; Popper |
| UnknownJournal | Explicit unknown-state logging (epistemic humility) | Kleene 3-valued logic; belief-revision literature |
| Gray-coded band-index bitstreams | Gray-code quantized feature encoding | Gray codes (standard); thermometer/ordinal encoding |
| Physics-discovery novelty encoder | Autoencoder with novelty-triggered growth | Growing neural gas (Fritzke 1995); adaptive resonance theory (Grossberg) |
| GAE/HND/FDM diagnostics | Residual-correlation hidden-variable detection | Granger-causality-adjacent residual analysis; latent-variable diagnostics |
| Learning simulation design | Procedural curriculum / environment generation | PAIRED (Dennis et al. 2020); domain randomization |

---

## 5. Mathematical-collapse-prevention-model (MCPM)

| Your term | Standard term | Anchor literature |
|---|---|---|
| M(S) = (R_e·A·D·f(C)) − L | Composite resilience index with penalty term | Composite indicators (OECD handbook); resilience metrics |
| f(C) = exp(−α‖C−C*‖²_F) | Gaussian kernel penalty on configuration deviation | RBF kernels; May stability criterion (1972) adjacency |
| BLACK/RED/AMBER/GREEN verdicts | Thresholded risk classification | Traffic-light risk dashboards (standard in risk management) |
| ttc (time-to-collapse) extrapolation | Linear trend-to-threshold forecasting | Survival/reliability analysis; EWS trend indicators |
| Drag ratio L/A | Load-vs-capacity margin | Load factor / safety margin (civil/structural engineering) |

---

## 6. Rosetta-Shape-Core

| Your term | Standard term | Anchor literature |
|---|---|---|
| Shape ontology (platonic solids as type system) | Symmetry-group-based type taxonomy | Point groups / polyhedral symmetry (standard group theory); geometric algebra |
| Shape assignments (ICOSA/DODECA/OCTA/TETRA/CUBE) | Symmetry-class tagging of data structures | Equivariance literature (Cohen & Welling 2016) — could ground invariants per class |

**Grounding note**: to make the shapes load-bearing rather than decorative, assign each one a *verifiable invariant* (e.g., octahedral = 8-state octahedral encoder consistency with GEIS; tetrahedral = 4-component confidence aggregation). Then the shape name is a compressed, checkable claim about structure.

---

## 7. Cross-repo vocabulary (use these in papers/READMEs)

| Ecosystem phrase | Say instead |
|---|---|
| "collapse prevention" | "early-warning detection and mitigation of critical transitions" |
| "falsification engine" | "falsifiability-disciplined experiment logging with refutation gates" |
| "calibration gate defers" | "selective fusion with reject option" |
| "thermodynamic epistemology" | "stability-analyzed iterative inference" (or drop the framing and show the math) |
| "manifold repair" | "trust-region-constrained adaptation toward a reference distribution" |
| "cascade audit" | "critical-transition early-warning battery" |
| "bridge to binary" | "compilation to fabricable/executable targets" |

## 8. Venue/citation posture

- Control-theory frame (ISS, Lyapunov, trust regions): venues like *IEEE Control Systems Letters*, *Automatica* workshops, NeurIPS safe-RL workshops.
- EWS/cascade frame: *Ecology Letters* lineage, *Nature Communications* interdisciplinary, *PNAS* complex-systems.
- Fusion/abstention frame: *Information Fusion*, IEEE Sensors.
- Falsification-infrastructure frame: *Journal of Open Source Software* (JOSS is a natural fit for every stdlib-only repo here — low friction, citable, values exactly this style of engineering).
- Each repo should carry a `CITATION.cff` and, where possible, a DOI via Zenodo — this is the cheapest credentialing step available and fully within your control.
