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

## Caveat

These notes cite literature that was compiled by an automated research pass. Citations
carry the confidence of that pass, not of a page-level re-verification — Notes 07 §5
flags its own known gaps. Treat every citation as a claim to check, which is exactly
the posture the hypothesis engine encodes.
