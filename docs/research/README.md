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
| `PLAN_FORWARD.md` | curly-octo-happiness | Phased roadmap (Phase 0–4) formalizing existing heuristics in validated theory. |
| `HARDWARE_INTEGRATION_PLAN.md` | curly-octo-happiness × Geometric-to-Binary-Computational-Bridge | Physical-grounding plan: measurement schema, cheap instrument rack, safety runtime, power claims. |
| `15_physical_shape_instrument.md` | Cross-repo (physical build) | The $15 octahedral bistable instrument: build, protocols E-P1–E-P8, and the E-P2 pre-registration arc. |
| `17_fractals_bio_cosmo_trig.md` | Cross-repo (theory) | Fold normal form across fractals/biology/cosmology/trigonometry; ten cross-domain tests, two of them aimed at this repo. |
| `TERMINOLOGY_MAP.md` | All six repos | Ecosystem naming ↔ standard research vocabulary, with anchor citations and venue posture. §1 is this repo. |
| `INTEGRATION_POINTS.md` | All six repos | The integration matrix (IP-1…IP-22) and build order. IP-13…IP-21 are this repo. |

**Note on numbering:** the notes series is the sibling ecosystem's, and it arrived here
in fragments. Notes 08–14, 16, and 18 are cited by the files above but are **not present
in this repo** — cross-references to them will not resolve locally.

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

| ID | Item | Why it's cheap |
|---|---|---|
| IP-18 | Pre-registered test of the κ_eff leading-indicator claim: Theory A (κ_eff leads the KL breach) vs Theory B (coincident/lagging, vs a trivial baseline). | Runs on the existing `energy_sweep` apparatus. Called "the cheapest high-value experiment in the ecosystem." |
| X4 | Is repair *navigation* or *teleportation*? Check linear/mode connectivity between the drifted checkpoint and the repaired weights. | A connectivity check over saved θ. If successful repairs are **not** path-connected, the navigation metaphor fails and the stability claims need the ISS proof outright. |
| X7 | Do independent repair runs (different seeds/subsets) converge on the same weight sites? Jaccard overlap across N runs. | Reuses the existing pipeline. Both outcomes are informative — funnelled repair space, or idiosyncratic repair. |
| IP-15 | Treat the three manifold layers as sheaf stalks; inter-layer disagreement (λ₁ lift) with leave-one-layer-out localizing the deceptive layer. | ~100 LOC stdlib. Directly addresses the "echo chamber of its own geometry" admission in `Claude-to-do.md`. |
| IP-16 | Name and analyze the existing μ-adaptation as the discrete integrator it already is; reframe the open ISS problem via the internal-model principle. | Analysis, not new subsystems. |
| IP-13 | Adapter from `docs/theoretical_notes/CLAIM_TABLE.fab.json` to a hash-chained, refute-gated ledger. | ~60 LOC. `ISS_proof_pending` becomes an OPEN claim with a refutation set. |
| IP-17 | Multiple reference basins + gating partition of unity = atlas repair, making `atlas/` literally true. | Quarter-scale; the trust region is already a chart. |

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
