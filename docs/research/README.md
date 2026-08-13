# Research Notes & Forward Plans

Consolidated research output from a cross-repo literature pass (2026-08-13), landed
here as the reference material behind the hypothesis engine (`scripts/hypothesis_engine.py`)
and the roadmap items below.

**Read the provenance carefully.** Most of these notes were compiled against sibling
repos in the same ecosystem, not against this one. They are kept verbatim — each file
names its own source repo in its header — because the machinery they describe is the
machinery this repo is converging on.

| File | Subject repo | What it is |
|---|---|---|
| `RESEARCH_PLAN.md` | — | The staged plan that produced everything else here. |
| `00_INDEX.md` | curly-octo-happiness | Index + cross-cutting thesis for notes 01–05. |
| `01_newest_hypotheses.md` | curly-octo-happiness | Epistemic grounding, falsification, calibration, self-model hypotheses. |
| `02_ai_training.md` | curly-octo-happiness | Learning/update rules, confidence propagation, RLVR/GRPO, optimizers, scaling laws. |
| `03_transformer_design.md` | curly-octo-happiness | Gray-code bitstream encodings, attention variants, MoE, norm-free design. |
| `04_neural_architecture.md` | curly-octo-happiness | GAE/HND/FDM diagnostics, geometric inference, NAS, KAN, MoE routing. |
| `05_learning_simulation_design.md` | curly-octo-happiness | World models, curiosity, dreams, skill libraries, falsification-driven environments. |
| `06_complexity_cybernetics_robotics.md` | curly-octo-happiness | Complexity engineering, cybernetics (VSM/Ashby), advanced robotics — the basis for `PLAN_FORWARD.md`. |
| `07_MCPM_collapse_research.md` | Mathematical-collapse-prevention-model | Calibration sources for the M(S) collapse metric — the closest external match to this repo's collapse work in `docs/theoretical_notes/`. |
| `08_cross_domain_toolkit.md` | Cross-Domain-Toolkit | Code-verified read of the toolkit (spinodal 2/√27, six EWS signals, calibration gate, falsification ledger), a 30-domain equation atlas, and a logic/knowledge-systems layer. |
| `09_nn_compression_manifolds.md` | Cross-repo (theory) | Compression science verified to paper IDs, representation-geometry battery, and four falsifiable hypotheses. Includes corrections to its own seed memo. |
| `10_integration_theories_languages.md` | Cross-repo (theory) | Perceptron/fusion theory (Novikoff, Cover/VC, Conant–Ashby, Covariance Intersection) and the measured stdlib-Python performance envelope behind the tier discipline. |
| `11_meta_structures_consciousness_bio_intelligence.md` | Cross-repo (theory) | Meta-structure formalisms, parallel-processing paradigms, geometric overlays, consciousness-studies status, biological compute paradigms. Claims carry [E]/[C]/[S] flags. |
| `12_seven_questions_shape.md` | Cross-repo (sims) | **The empirical basis for IP-15…IP-20.** Seven stdlib sims (S1–S7), each with a measured dose-response curve and a confirmed literature gap. |
| `PLAN_FORWARD.md` | curly-octo-happiness | Phased roadmap (Phase 0–4) formalizing existing heuristics in validated theory. |
| `HARDWARE_INTEGRATION_PLAN.md` | curly-octo-happiness × Geometric-to-Binary-Computational-Bridge | Physical-grounding plan: measurement schema, cheap instrument rack, safety runtime, power claims. |
| `14_rosetta_shape_grounding.md` | Rosetta-Shape-Core | Shapes as deformable containers of equation-complexes: morphometrics, equivariant bifurcation, Kendall shape space; the sims behind the 6× vertex / 2.8× face / 85% low-mode localization figures. §4 grounds this repo's shape assignments. |
| `15_physical_shape_instrument.md` | Cross-repo (physical build) | The $15 octahedral bistable instrument: build, protocols E-P1–E-P8, and the E-P2 pre-registration arc. |
| `17_fractals_bio_cosmo_trig.md` | Cross-repo (theory) | Fold normal form across fractals/biology/cosmology/trigonometry; ten cross-domain tests, two of them aimed at this repo. |
| `TERMINOLOGY_MAP.md` | All six repos | Ecosystem naming ↔ standard research vocabulary, with anchor citations and venue posture. §1 is this repo. |
| `INTEGRATION_POINTS.md` | All six repos | The integration matrix (IP-1…IP-22) and build order. IP-13…IP-21 are this repo. |
| `HARNESS.md` | All six repos | Sim Harness Standard v1: required directory layout, config manifest, execution contract, and verdict discipline for any sim whose result enters the ledger. |

**Note on numbering:** the notes series is the sibling ecosystem's and arrived here in
fragments. Still absent and cited by the files above: **notes 13, 16, and 18** — those
cross-references will not resolve locally. Note 18 §3 is the origin of the E-P8
snap-latency protocol in notes 15, so it is the one most worth chasing.

## Imported sims

Two sims from the same research pass live in `experiments/`. Both are numpy, neither
conforms to `HARNESS.md`, and both are named in its retrofit queue:

| Script | Status | Verified here |
|---|---|---|
| `experiments/fractal_basin_sim.py` | Retrofit queue #4. Deficiency: alpha measured at a single damping — `gamma` is a parameter but only ever called at its 0.25 default, so the mandated sweep is missing. | **Reproduces notes/17 §1 exactly**: α=0.688 (double well), α=0.392 and 8.0% Wada (triple well). Deterministic under its fixed seed. |
| `experiments/ep2_prereg_sim.py` | Retrofit queue #1, and **superseded**: this is v2, which notes/15 records as REFUTED. | Runs, and prints "PASS (predicted)" with detection in 200/200 trials — because it has **no null arm**. That is the point: the uncontrolled version cannot fire a false positive, which is exactly why v1/v2 were refuted and v3 (two-arm, one pre-committed checkpoint) exists. v3 was not provided. |

Both were landed verbatim apart from two mechanical changes: `fractal_basin_sim.py` had
its output directory hardcoded to a nonexistent agent workspace (`/mnt/agents/output/figures/`,
now `results/fractal_basin`, overridable via `FRACTAL_BASIN_OUT`), and both were
reformatted for ruff (semicolon statements, one lambda). Outputs are byte-identical
before and after reformatting.

## Where this repo sits in the tier discipline

`INTEGRATION_POINTS.md` IP-12 states the ecosystem's dependency law, and notes 10 §2.2
gives the measured justification: pure-Python runs ~9M MAC/s and one MNIST epoch on a
784→64→10 MLP takes 20–25 minutes, so stdlib "covers algorithm prototyping + pedagogy +
auditability, not evidence on realistic models."

This repo is a **Tier 2** project — `requirements.txt` puts torch at the core, which is
exactly what the tiering prescribes for real compression/training evidence. What matters
is that the material landed here stays in its own lane, and currently does:

| Tier | Deps | What lives here |
|---|---|---|
| 0 | stdlib only | `scripts/hypothesis_engine.py`, `repair/generic_repair_controller.py` |
| 1 | numpy | `experiments/fractal_basin_sim.py`, `experiments/ep2_prereg_sim.py` |
| 2 | torch | the manifold pipeline: `manifolds/`, `simulation/`, `repair/`, `addon_thermodynamic_control/` |

The rule worth keeping: the hypothesis engine must not acquire a numpy dependency, and
the Tier-1 sims must not acquire a torch one. Notes 12's seven sims are all Tier 0 — if
IP-15/16/17/20 get built here from those blueprints, they can stay stdlib even though the
repo around them is torch.

## Shape assignments: verified against `bridges/rosetta-bridge.json`

Notes 14 §4 summarizes this repo's bridge as `data→ICOSA, parameter→DODECA, policy→OCTA,
confidence→TETRA, thermo→CUBE`. **All five are accurate** — the first three under
`layer_shape_map`, the last two under the separate `confidence_aggregation` and
`thermodynamic_extension` keys.

One thing to pin down before treating the shapes as load-bearing. Notes 14 makes each
assignment a falsifiable claim — "if confidence aggregation ever carries ≠4 components,
the shape is wrong" — but the two documents count differently: notes 14 justifies TETRA
by **four channels** (data/param/policy/combined, one per vertex), while the bridge file's
own rationale justifies it by **three weights** ("the irreducible three-weight
combination", a simplex on 3 signals). Both arrive at TETRA; they disagree on what would
refute it. The refutation condition has to name one count to be a claim at all.

## Relevance to this repo

Three threads carry over directly:

1. **Early-warning signals** (Notes 07 §2, "A" term) — AR(1) and variance rise as the
   recovery rate |λ| → 0 near a tipping point. This is the cheapest unimplemented
   monitor for `repair/monitors.py`, and it is stdlib-only.
2. **Model collapse as variance suppression** (Notes 07 §2, "D" term) — Shumailov 2024
   (tails vanish first, σ² → 0 recursively) is the literature anchor for
   `repair/smoothing_auditor.py` and `docs/theoretical_notes/run_collapse_experiment.py`.
3. **Safety envelopes as claims** (`PLAN_FORWARD.md` §3.1, `HARDWARE_INTEGRATION_PLAN.md`
   I10) — CBF-style safe sets are the embodied analogue of this repo's trust region:
   both are hard constraints that bound a step rather than penalize it.

## Action items aimed squarely at this repo

`INTEGRATION_POINTS.md` and `17_fractals_bio_cosmo_trig.md` name specific, testable work
here. Ordered by cost:

| ID | Item | Empirical basis now in-repo |
|---|---|---|
| IP-18 | Pre-registered test of the κ_eff leading-indicator claim: Theory A (κ_eff leads the KL breach) vs Theory B (coincident/lagging, vs a trivial baseline). | **Notes 12 S6** ran exactly this protocol on a fold series: variance-τ fires on 60% of trials, AC1-τ on 47%, **0% false alarms on the null**, OR-battery ~76%. The lesson is that neither marker is reliable alone and the battery's value is additive coverage. Runs on the existing `energy_sweep` apparatus. |
| X4 | Is repair *navigation* or *teleportation*? Check linear/mode connectivity between the drifted checkpoint and the repaired weights. | Notes 11 §2.2 supplies the standard: model soups work only within a basin (linear mode connectivity; permutation alignment, Entezari 2022). If successful repairs are **not** path-connected, the navigation metaphor fails and the stability claims need the ISS proof outright. |
| X7 | Do independent repair runs (different seeds/subsets) converge on the same weight sites? Jaccard overlap across N runs. | Reuses the existing pipeline. Both outcomes informative — funnelled repair space, or idiosyncratic repair. |
| IP-15 | Treat the three manifold layers as sheaf stalks; inter-layer disagreement (λ₁ lift) with leave-one-layer-out localizing the deceptive layer. | **Notes 12 S1**, measured: λ₁ ≈ 0.176·φ² (quadratic in fault magnitude) and **exact** fault localization — compensating the true edge collapses λ₁ to 0.000 vs 0.024–0.038 for wrong edges. Note the consequence: quadratic sensitivity means faults below noise·√(1/0.176) are invisible. ~100 LOC stdlib; addresses the "echo chamber of its own geometry" admission in `Claude-to-do.md`. |
| IP-16 | Name and analyze the existing μ-adaptation as the discrete integrator it already is; reframe the open ISS problem via the internal-model principle. | **Notes 12 S2**: the integral loop holds RMSE 0.067–0.076 under drift and is robust across a ×10 gain perturbation, where the fitted-offset baseline degrades. The no-drift noise penalty is a few percent, not orders of magnitude. Theory: Yi et al. PNAS 2000 (integral feedback is *necessary* for robust perfect adaptation). Analysis, not new subsystems. |
| IP-13 | Adapter from `docs/theoretical_notes/CLAIM_TABLE.fab.json` to a hash-chained, refute-gated ledger. | **Notes 08 §A.3** specifies the target format precisely: SHA-256 over canonical JSON, gated `refute()`, refutation_set guards, escape-hatch detection. ~60 LOC. `ISS_proof_pending` becomes an OPEN claim with a refutation set. |
| IP-20 | Third repair mode: *expand* the basin (LGG-style anti-unification) when drift is beneficial, instead of always pulling back. | **Notes 12 S7**: LGG refinement achieved 0% error with 100% true-alarm retention, while every deletion repair lost 15–20%. Refinement dominated deletion on both axes. |
| IP-17 | Multiple reference basins + gating partition of unity = atlas repair, making `atlas/` literally true. | **Notes 12 S4**, matched-budget: the 2-chart advantage is unbounded at zero curvature (global PCA cannot represent bimodality at all), 15.4× at c=0.5, 4.8× at c=1.0, 2.2× at c=2.0 — **monotone decreasing in curvature**. Charts pay most when the data is clustered, least when curvature dominates. |

Two caveats on the above, from reading the code rather than the notes:

- IP-14 describes this repo's phase classifier as `κ>20 / KL>2ε / trend>3`. The actual
  gate in `addon_thermodynamic_control/stability.py:441` is
  `kappa > C_bound or basin_kl > 2·epsilon_s or dV_dt > 0.1` — `C_bound` is configurable,
  not a hardcoded 20, and the third term is a Lyapunov derivative, not a trend ratio.
- IP-21's hardware-in-the-loop grounding depends on repos not present here (GBCB's
  `serial_csv.py`). It is a cross-repo item, not actionable from this checkout alone.

## Caveat

These notes cite literature that was compiled by an automated research pass. Citations
carry the confidence of that pass, not of a page-level re-verification — Notes 07 §5
flags its own known gaps. Treat every citation as a claim to check, which is exactly
the posture the hypothesis engine encodes.
