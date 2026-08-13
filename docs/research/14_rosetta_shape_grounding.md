# 14 — Grounding Rosetta-Shape-Core: Shapes as Deformable Containers of Equation-Complexes

Date: 2026-08-13 · Prompt: the user's design intent — "each shape holds a combination of equations and constants in its complexity; the shape distorts when out of balance (a prediction system); drill down into the shape to diagnose which equation or interconnection failed; intended as a meta-meta structure."
Verdict up front: **yes, ground it — the intent maps onto three established formalisms almost word-for-word, and the sim below shows the mechanism working end-to-end in ~80 lines of stdlib Python.**

---

## 1. The three formal anchors

### Anchor A — Geometric morphometrics: shape deviation IS a measured quantity
D'Arcy Thompson's program, formalized as **geometric morphometrics** (Bookstein; Kendall): a configuration of landmarks has a *shape* = its equivalence class under translation/rotation/scale. The distance between shapes is the **Procrustes distance**; deformation between reference and measured shape is decomposed by thin-plate splines into orthogonal warp components. This is exactly "the shape gets distorted when out of balance, allowing for a prediction system" — with a century of statistics behind the measurement.

### Anchor B — Equivariant bifurcation theory: symmetry breaking classifies failure modes
For a system with symmetry group G, instabilities don't happen generically — they happen *along irreducible representations of G* (Golubitsky–Stewart–Schaeffer, *Singularities and Groups in Bifurcation Theory*). For a polyhedral symmetry group, the deformation modes of the shape decompose into irreps, and each irrep is a **symmetry-adapted failure channel**: uniform breathing, axis-elongation, shear, chirality flips. "Drill down" = project the measured distortion onto the irrep basis → which *kind* of imbalance; then onto edges/faces → which *specific* equation or interconnection. This is the rigorous version of the drill-down intent.

### Anchor C — Shape spaces are manifolds: the meta-meta level is real
**Kendall's shape space** — the space of all shapes of k landmarks — is itself a manifold (for 2D landmarks, a complex projective space; in general a quotient of a sphere). "The space of shapes" is not a figure of speech. A system evolving through imbalance trajectories is a **point moving in shape space**; collapse/tipping = the trajectory crossing a boundary in that space. This grounds the "meta-meta structure" intent: shapes hold equations (level 0), the shape deforms (level 1 — shape as indicator), and the *trajectory in shape space* is itself an auditable dynamical object (level 2 — meta-meta). EWS machinery (notes/08, cascade audit) applies directly to the shape-space trajectory.

### (Supporting) Force-density / Maxwell reciprocal figures
Polyhedral shapes as equilibrium of edge tensions is classical structural engineering (force-density method; Maxwell's reciprocal diagrams 1864): an imbalanced edge demand literally deforms the polyhedron. So "shape distortion under imbalance" is also physically realizable — a shape can be *built* (GBCB emitters) such that its physical deformation mirrors the equation residuals.

---

## 2. The grounded scheme (concrete)

A **Rosetta shape** S = (P, G, A) where:
- **P** = polyhedron (e.g., octahedron): vertices V, edges E, faces F.
- **G** = symmetry group of P (octahedral group O_h for the octahedron).
- **A** = assignment map: equations/constants → elements of P (equations on edges, constants on vertices, interconnections on faces — or per-repo convention).

**State**: each edge e carries a residual r_e = (observed − predicted) of its equation. Balanced ⟺ all r_e = 0 ⟺ shape = reference polyhedron.

**Deformation operator**: residuals act as edge-length demands d_e = d_ref·(1 + κ·r_e); relax the spring network (force-density equilibrium) → deformed configuration V′. (This is the sim below.)

**Measurement**:
- Global: Procrustes distance D(S′, S_ref) — one scalar "balance" reading, composable with the cascade audit as a signal pressure.
- Modal: project the displacement field onto graph-Laplacian eigenvectors (finite-size stand-in for the irrep decomposition; on the full symmetry group, onto irreps of G) → *which failure channel*.
- Local: per-edge residuals r_e → *which equation*; per-face aggregates → *which interconnection*.

**Meta-meta layer**: the sequence S₀, S₁, S₂, … is a trajectory in shape space. Track its velocity/acceleration/AR(1)/variance — the standard EWS battery — to predict *approaching* imbalance before the shape visibly deforms (rate-induced tipping detection on the shape trajectory itself).

**Ledger integration**: each drill-down conclusion ("edge e's equation is the imbalance source") is a falsifiable claim — enters the CDT ledger with refutation condition (e.g., "correcting equation e restores D < τ").

---

## 3. Simulation evidence (stdlib, ~80 LOC — sims/rosetta_shape_sim.py)

Octahedron (6 vertices, 12 edges), one edge's equation driven out of balance by residual δ:

**Dose-response (prediction system works):**
| residual δ | Procrustes distortion |
|---|---|
| 0.05 | 0.0234 |
| 0.10 | 0.0470 |
| 0.20 | 0.0955 |
| 0.40 | 0.1986 |

Near-linear through this range (≈ 0.235·δ + small quadratic term) — distortion is a *readable gauge* of imbalance magnitude, exactly the prediction-system intent.

**Drill-down (δ = 0.2 on edge (0,2)):**
- Per-vertex displacement: **[0.1445, 0.0225, 0.1445, 0.0225, 0.0774, 0.0774]** — the two vertices of the faulted edge carry 6× the displacement of their neighbors. **Localization works.**
- Eigenmode decomposition: octahedral graph Laplacian eigenvalues [0, 4×3, 6×2]; displacement concentrated in the low modes (λ=4 triplet: amplitudes 0.097, −0.043, 0.060 vs λ=6 pair: ~0.003–0.006). **Smooth modes dominate** — the distortion signature is low-order, hence detectable from few probes, and mode-projection compresses the diagnosis.

This is the shape doing precisely what the README-intent says: holding the equation-complex, deforming under imbalance, supporting drill-down diagnosis — with numbers.

---

## 4. Why the shape assignments in Geometric-manifold- become load-bearing

Current bridge maps: data→ICOSA, parameter→DODECA, policy→OCTA, confidence→TETRA, thermo→CUBE. Under this grounding, the choice stops being decorative if each shape's complexity matches its content:

- **OCTA (policy, 6 vertices / 12 edges)**: policy alignment has few components (P, Q, mixture M; JS terms) — small shape, appropriate.
- **TETRA (confidence, 4 vertices)**: four confidence channels (data/param/policy/combined) → one vertex each; balanced tetrahedron = four-way agreement; flattening = one channel dissenting. Clean.
- **CUBE (thermo, 12 edges)**: Fisher energy terms (task, safety, policy, repair) + cross-terms — edge count matches the term structure.
- **ICOSA/DODECA (data/parameter, 30 edges each)**: the rich layers get the rich shapes — many features/weights need many carriers. Dodecahedron–icosahedron duality even gives a formal relationship between the data and parameter layers (dual polyhedra = dual constraint systems).

The grounding rule (from TERMINOLOGY_MAP §6): **a shape assignment is a claim; make it checkable.** E.g., "TETRA-confidence" is falsifiable: if confidence aggregation ever carries ≠4 components, the shape is wrong and the bridge file must change — the ledger records it.

---

## 5. Research & build agenda

1. **Shape-space EWS** (days, stdlib): trajectory of D(S′,S_ref) + mode amplitudes through the CDT cascade audit — does AR(1)/variance on the shape trajectory precede visible distortion? (Ties notes/12 S6 protocol: pre-register kill criteria.)
2. **Full irrep decomposition** (week): replace graph-Laplacian modes with true O_h irreps for the octahedron — cleaner channel semantics ("breathing mode" = total-energy imbalance; "shear modes" = cross-term imbalance).
3. **Face-level interconnect diagnosis** (week): faces as interconnections — residual aggregation over faces detects coupling failures that edge analysis misses (XOR-type contradictions between equations — Minsky–Papert blind spot of edge-only analysis, notes/10).
4. **Physical shape demonstrator** (quarter, GBCB): a literal strut-and-node octahedron whose edge actuators realize equation residuals — the shape as a physical dashboard (force-density method makes this mechanically sound).
5. **Dual-shape coupling**: data (ICOSA) ↔ parameter (DODECA) as dual polyhedra — investigate whether dual-edge incidence gives a principled data↔parameter influence map.
6. **Paper path**: "Polyhedral symmetry-breaking as a diagnostic representation for composite dynamical systems" — morphometrics + equivariant bifurcation + the sim evidence; venues: complex-systems or visualization/diagnostics tracks; JOSS for the tooling.

## 6. What this does NOT claim
Shapes are not consciousness detectors, not energy fields, not symbolic magic. They are **symmetry-structured constraint carriers with a measurable deformation semantics** — a visualization *and computation* layer grounded in morphometrics, bifurcation theory, and structural mechanics. Stated this way, every claim is falsifiable — which is the ecosystem's house rule anyway.

---

## 7. Simulation results: shape as predictor and diagnostician (2026-08-13, agenda item 1)

Three sims (stdlib-only; sims/shape_trajectory_ews.py, shape_fold_ews.py, shape_fold_nullfix.py, shape_fold_roc.py). Octahedron spring network, edge (0,2) carries the imbalanced equation.

### 7.1 Diagnosis (drill-down): works cleanly
- Dose-response near-linear: distortion ≈ 0.235·δ (R² visually high over δ∈[0.05,0.4]).
- Edge localization: faulted edge's vertices carry ~6× the displacement of neighbors.
- **Face-level (XOR-type interconnection)**: two opposing edge faults on one face (+15%/−15%) — individually small, but face-aggregated displacement mass = **2.81× off-face mass**. Interconnection faults are visible at face level that edge-level analysis underweights — the drill-down hierarchy (global Procrustes → modal decomposition → face → edge) is functional at every level.

### 7.2 Prediction (shape-space EWS): the shape transmits fold dynamics, with the classic tradeoff
- Under **fold (saddle-node) approach** driving the imbalance, absolute-threshold alarms on windowed variance/AR(1) of the distortion trajectory fired with **mean lead ≈ 280 steps (variance, 8/10 trials) and ≈ 490 steps (AR(1), 10/10)** before the snap — the shape trajectory genuinely carries critical-slowing-down information.
- But specificity was poor (null false-positive 0.60 with lax thresholds), and null-calibrated Kendall-τ alarms showed the sensitivity/specificity knife-edge:

| alarm criterion | detection (fold) | false positive (null) |
|---|---|---|
| τ>0.2 | 0.08 | 0.25 |
| τ>0.3 | 0.08 | 0.08 |
| τ>0.4 | 0.08 | 0.00 |

At these window counts (short trajectories, 12 trials) the τ statistic over windowed variance barely separates drift from null (mean +0.07 vs +0.05) — **the windowed-tau operationalization dilutes a late-rising signal**. This is exactly the Boettiger–Hastings/Dakos finding (notes/12 S6) reproduced in shape space: EWS reliability is operationalization-bound, not free.

### 7.3 The fix the geometry itself suggests
The diagnosis sim showed distortion concentrates in the low-order symmetry modes (λ=4 triplet carried ~85% of modal amplitude). **Mode-filtered EWS**: project D(t) onto the fault-bearing modes *before* window statistics — the modal projection raises SNR by discarding orthogonal noise directions. This is the shape-native advantage over scalar EWS: the geometry tells you which subspace to monitor. (Next sim: mode-filtered τ statistics; predicted detection ↑ at matched FP.)

### 7.4 Honest status of the three claims
| Claim (from design intent) | Status after sims |
|---|---|
| Shape distorts under imbalance, readable as gauge | **Supported** — near-linear dose-response, stable across noise |
| Drill-down diagnoses the equation/interconnection | **Supported** — vertex (6×), face (2.8×), and modal (85% low-mode) localization all work |
| Shape trajectory predicts imbalance in advance | **Conditionally supported** — long alarm leads under fold dynamics with absolute thresholds; sensitivity/specificity tradeoff at small samples; mode filtering predicted to improve it (untested) |

All three conclusions are ledger-ready claims with stated refutation conditions; the third is exactly the kind of conditional, operationalization-dependent claim the falsification ledger exists to hold honestly.

---

## 8. The shape AS the dynamical system — mechanical critical slowing down (sims/shape_csd_probes.py)

The 7.2 weakness (shape as passive readout attenuates CSD) is resolved by making the shape carry its own bistability: octahedron with one **bistable strut** (quartic energy E(l)=a(l−1.2)²(l−1.8)² — two stable lengths), external compression ramped until the strut snaps to its short branch.

**Measured:**
- Snap at compression ≈ **0.495** (strut 1.8 → 1.45, first-order jump).
- **Recovery-time CSD**: impulse-probe recovery time is flat (70 steps) through 62% of snap compression, rises to 80 at 62%, then diverges (600+, probe no longer returns) at 75% — **a leading signature ~25% of the compression range before the snap**. The non-recovery of probes past 75% is itself the flickering/loss-of-resilience signature (the strut no longer recaptures its state — kicks knock it into the other basin).
- **Fluctuation variance**: flat 7×10⁻⁶ through 75%, then 3× jump at 88% — real but *late*; recovery time is the earlier instrument here (consistent with literature: recovery-rate CSD typically leads variance).

**This fixes the 7.2 gap**: when the shape's own mechanics hold the bistability, critical slowing down is directly measurable on the shape by active probing — impulse → recovery time. The shape is not a readout of a hidden process; it *is* the process. Procrustes distance monitors balance; probe-recovery time predicts failure; drill-down localizes it.

### Updated claim table
| Claim | Status |
|---|---|
| Shape distorts under imbalance, readable gauge | Supported (linear dose-response) |
| Drill-down diagnoses equation/interconnection | Supported (vertex 6×, face 2.8×, modal 85% low-mode) |
| Shape trajectory predicts imbalance (passive monitoring) | Weak at short samples (classic EWS tradeoff) |
| **Active probe-recovery on a bistable shape predicts its snap** | **Supported** — ~25% lead via recovery-time divergence, ~12% via variance |
| Mode-filtered passive EWS | **Refuted at this regime** (modal variance monotone *decreased*, τ=−0.49 — stiffness filtering; recorded as falsified rather than abandoned: retest under different noise/stiffness ratios) |

### CDT cascade-audit wiring (the instrument mapping)
Shape signals → SignalReads pressures: probe recovery time → `critical_slowing_down`; strut fluctuation variance → `variance_inflation`; Procrustes distortion rate → `skew_to_alt_well` (approach to the alternate configuration); snap events → `flickering` (basin hopping); face/edge localization consistency → `coherence_under_contradiction`; number of distinct deformed modes visited → `diversity_collapse`. h_eff = compression/snap-compression ratio → the fold control parameter. The shape becomes a six-signal physical instrument with a spinodal at the strut's barrier point.
