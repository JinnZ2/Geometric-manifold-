# Integration Points Across the JinnZ2 Repo Ecosystem

Date: 2026-08-13 · Basis: notes/00–10, PLAN_FORWARD.md, HARDWARE_INTEGRATION_PLAN.md, CROSS_DOMAIN_TOOLKIT_PROPOSALS.md, hypothesis-engine.

## The five repos and what each contributes

| Repo | Role in ecosystem | Key assets |
|---|---|---|
| **curly-octo-happiness** (COH) | AI/learning science core | Claim epistemics (Beta posterior, escape-hatch counters, UnknownJournal), Gray-coded band-index bitstreams, GAE/HND/FDM diagnostics, physics-discovery auto-encoders |
| **hypothesis-engine** (HE) | Automation layer | explore→log→claim→test→modify→hidden-variables→consolidate loop as GitHub Action (cron, issue-on-new-hypothesis) |
| **Geometric-to-Binary-Computational-Bridge** (GBCB) | Hardware bridge | Bond-graph IR (6 substrates), emitters (KiCad/g-code/OpenSCAD/coil/loom), verify (Farina WAV sweep), GEIS octahedral encoder, timestamped CSV contract `seq,micros,v0..vN,crc8` |
| **Mathematical-collapse-prevention-model** (MCPM) | Collapse metric | M(S) = (R_e·A·D·f(C)) − L, f(C)=exp(−α‖C−I_n/φ‖²_F), BLACK/RED/AMBER/GREEN verdicts, ttc extrapolation |
| **Cross-Domain-Toolkit** (CDT) | Portable kernels | falsification_ledger (hash chain, gated refute, symbolic checker), multi_substrate_calibration (Lε gate, GROUND/PREDICT), cascade_regime_audit (6 EWS + spinodal 2/√27) |

---

## Integration matrix (who plugs into whom, and how)

### IP-1 · CDT falsification_ledger ← everywhere
Every repo's claims should live in one ledger format. COH's Beta-posterior claim tracking, MCPM's verdict history, GBCB's hardware acceptance tests, and HE's consolidated hypotheses all map onto Claim→Prediction→Observation→Mismatch. **Concrete step**: adopt CDT's ledger as the serialization layer; add `.npy`/blob support via SHA-256 content-addressed manifests (notes/10 §2.3) so model checkpoints and WAV sweeps enter the chain.

### IP-2 · Subjective-logic opinions unify COH Beta posteriors and CDT confidence binding
COH tracks claims as Beta(1+passed, 1+failed); CDT binds confidence = reliability × warp(native). Jøsang's opinion ω=(b,d,u,a) is **exactly isomorphic** to Beta (b=r/(W+r+s), d=s/(W+r+s), u=W/(W+r+s), W=2). One object, two parameterizations → cross-repo interchange format; u drives CDT's gate; cumulative vs averaging fusion chosen by substrate dependence (notes/10 §1.5).

### IP-3 · Calibration gate ← GBCB hardware substrates
GBCB's verify loop (Farina sweeps, unit CSV) produces exactly the GROUND/PREDICT readings CDT's gate consumes: sensor = GROUND with reliability earned from its ledger track record (EigenTrust loop, CDT P0.3); the bond-graph model = PREDICT. The timestamped CSV contract (I1) is the wire format; `bounds=` DEFER handles incoherent reads (e.g., clamped phone-mic AGC).

### IP-4 · Covariance Intersection for dependent substrates
GBCB multi-sensor rigs and CDT's gate both currently assume independence (noisy-OR). Notes/10 §1.5: **CI is the correct fusion under unknown cross-correlation**; information-filter summation double-counts. Add CI as an optional fusion mode in CDT (pure Python, ~60 LOC); D-S conflict mass K feeds cascade audit signal S5 (coherence-under-contradiction) — the computed bridge from gate to audit (CDT P0.2).

### IP-5 · Cascade audit ← MCPM as a domain instantiation
MCPM's M(S) decomposition supplies h_eff and signal reads for CDT's CascadeAudit: A(alertness) from AR(1)/variance EWS (notes/07), diversity D as signal S6, drag ratio L/A > 1 as a COMMITTED proxy. Conversely, CDT's spinodal 2/√27 gives MCPM a principled threshold where it currently has verdict bands. **One shared example** (`mcpm_as_cascade_audit.py`) demonstrates both.

### IP-6 · Spinodal lineage closed loop
CDT's H_SPINODOAL descends from JinnZ2/JinnZ2 field_collapse.py and monoculture_collapse_predictor (Kramers escape). Eight atlas domains are cusp-structured (notes/08 Part B): each gets a reference h_eff mapper as a CDT example — the collapse repos become the *validated instantiations* of the portable kernel.

### IP-7 · Hypothesis engine ← all repos as topic sources
HE's config/topics.json gains: compression-order noncommutativity (notes/09 H4), manifold-aware quantization, Grassmannian distillation, CI-vs-noisy-OR overconfidence, per-layer ID rank budgets. Each consolidated hypothesis is filed as a CDT ledger claim with refutation_set + logical_form — HE becomes the cron front-end, CDT the persistence/epistemics back-end.

### IP-8 · Compression research program on the tiered stack (notes/09 + notes/10)
- **Tier 0 (stdlib core, new module candidates)**: quantization simulator (INT8/INT4/ternary integer emulation), power-iteration SVD ≤256², CKA/effective-rank battery, .npy reader/writer, content-addressed manifests. All fit the measured stdlib envelope (256² matmul ~1.5 s; CKA N=300 ~1.1 s).
- **Tier 2 (torch)**: the memo's research matrix (ResNet-18 + small decoder; INT8 PTQ / 4-bit GPTQ-style / structured / unstructured / low-rank arms; CKA + erank + Procrustes + mutual-kNN + TWO-NN geometry battery; Pareto atlas output).
- **Tier 3**: distilled int8 MLP → RP2040 via TFLM (≤~150K params, notes/10 §2.4) — end-to-end demo closing the loop from COH learning science to GBCB hardware.

### IP-9 · Geometry battery as a shared CDT example package
The notes/09 §2.7 battery (principal angles, linear CKA, whitened Procrustes, mutual-kNN, TWO-NN ID, R_M/D_M) is computable stdlib-only at probe scale → `cross_domain_toolkit/examples/representation_geometry/`. Reuses CDT's mapper pattern (series → [0,1] pressure), letting "geometric drift of a model under compression" become a cascade-audit signal (S1/S2 analog on representation space).

### IP-10 · Knowledge-system spine shared across repos
CDT P1 stack (content-addressed triple store, PROV-O emission, Belnap 4-valued labels, ATMS, Dung grounded extension, hitting-set repair) serves: COH (claim dependency graphs + hidden-variable tracing), MCPM (assumption provenance per verdict), HE (hypothesis lineage), GBCB (calibration provenance per unit CSV). One `integrate/` package, four consumers.

### IP-11 · Embedded telemetry back into ledgers
GBCB's serial_csv.py (I2) + MicroPython sensor nodes stream UART/CSV → CDT ledgers on the host. Hardware claims ("this coil meets spec across temperature") become continuously re-tested ledger entries — falsification as a living process, not a one-time check (notes/10 Tier 3 ↔ Tier 0 contract).

### IP-12 · Language discipline as ecosystem law
Tier 0 stdlib-only forever; numpy+pytest only in Tier 1 packages; torch only in Tier 2 experiments; formats (NDJSON/CSV/.npy/manifest) defined and tested in Tier 0, all tiers must round-trip in CI. ONNX for non-LLM models, GGUF only if local LLM inference enters scope. Mojo: watch at 1.0 (notes/10 §2.5).

---

## Suggested build order
1. **IP-1 + IP-2** (ledger + opinions interchange) — cheap, unblocks everything.
2. **IP-3 + IP-4** (gate hardware hookup + CI fusion) — makes the gate honest under dependence.
3. **IP-5 + IP-6** (MCPM/collapse instantiations) — validates the audit on real domain math.
4. **IP-8 Tier-0 modules + IP-9** (geometry battery) — new capability, stdlib-bounded.
5. **IP-7** (HE topics wired to ledger) — automation last, once the epistemic substrate is stable.
6. **IP-10 + IP-11** (knowledge spine, telemetry loop) — the long-horizon integration.

---

## Addendum 2026-08-13 — Geometric-manifold- (Basin Repair Framework) + Rosetta-Shape-Core

New ecosystem member: **GM** = Basin Repair Framework (3-layer data/parameter/policy manifolds, KL-basin, κ_eff curvature early-warning, Fisher energy accounting, CLAIM_TABLE exports; torch-based; stdlib GenericRepairController). Fieldlink peer: **Rosetta-Shape-Core** (shape ontology repo).

### IP-13 · GM CLAIM_TABLE → CDT falsification ledger (adapter ~60 LOC)
GM already exports falsifiable claims (`falsification_condition`, `status: OPEN`, `ISS_proof_pending`). Adapter converts to CDT ledger Claims — hash-chained, refute-gated. ISS/Lyapunov pending proofs become OPEN claims with refutation sets.

### IP-14 · GM signals → CDT cascade audit
κ_eff, energy trend ratio, JS confidence, combined confidence = four signal reads; h_eff = KL/ε. GM's heuristic phase classifier (κ>20 / KL>2ε / trend>3) replaced by principled six-signal + spinodal machinery.

### IP-15 · GM 3-layer consistency → sheaf λ₁ meta-signal (notes/12 S1)
Data/parameter/policy layers as stalks; inter-layer disagreement = λ₁ lift; leave-one-layer-out localizes the deceptive layer. ~100 LOC stdlib. Directly addresses GM's "echo chamber" admission.

### IP-16 · GM repair controller + integral feedback (notes/12 S2)
Add integral term on basin violation; ISS proof (GM's stated open problem) reframed via internal-model principle. GM's μ-adaptation is already a discrete integrator — name it, analyze it.

### IP-17 · GM trust regions → multi-basin atlas repair (notes/12 S4)
Trust region = chart. Multiple reference basins {θ_refⁱ} + gating partition of unity = atlas repair; makes GM's atlas/ directory literally true. Connects to SliceGPT chart-aware compression (notes/09).

### IP-18 · GM κ_eff leading-indicator claim → kill-criteria test (notes/12 S6)
Pre-registered Theory A (κ_eff leads KL breach) vs Theory B (coincident/lagging; trivial-baseline comparison) using GM's existing energy_sweep apparatus. Cheapest high-value experiment in the ecosystem.

### IP-19 · GM planned intake valve → citation-bias stack (notes/12 S5)
curiosity_loop/math_stepper provenance weights become computed ledger quantities (escape-hatch flag + supportive-fraction + reference-rot), validated on the CMN/wood-wide-web case.

### IP-20 · GM repair events → anti-unification basin generalization (notes/12 S7)
Third repair mode: expand basin (LGG-style) instead of pulling back when drift is beneficial; every expansion is a refutable claim.

### IP-21 · GBCB hardware → GM physical grounding loop
GM's to-do wants live external execution friction updating the reference basin; GBCB's CSV contract + serial adapters are the intake — hardware-in-the-loop basin repair demo.

### IP-22 · Rosetta-Shape-Core grounding
Shape assignments (ICOSA/DODECA/OCTA/TETRA/CUBE) currently decorative; ground via notes/11 geometry — what invariants do the shapes encode? Treat as schema, not physics, until load-bearing.

Updated build order: IP-13 (ledger adapter) and IP-18 (κ_eff test) first — both days-scale; then IP-15; then IP-16/17 as quarter-scale projects.
