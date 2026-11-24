# Experiment designs for the hypothesis engine's drafts (2026-11-24)

Status of every experiment below: **PROPOSED**. Nothing here is a result. Co-authored (human + AI).

- Source read: `hypotheses/*.md` and `data/` (claim tree, findings, unknowns, reformulations,
  engine report) at commit `02320b51b9138dec032f8f85839bce420502a9ec` (engine digest
  2026-11-23T16:38Z). Date written: 2026-11-24.
- Main advanced while this was written: engine digest `33458e3` (2026-11-24) changed the
  surviving counts to FIT 15, LLG 259, SDM 76 (HVD, MCR: one-line changes). The analysis and
  counts in these files are pinned to `02320b51` and were not re-derived for `33458e3`.
  The gaps concern the engine's method, so later digests do not change them.
- Scope: five topic drafts plus one cross-cutting audit of the engine's evidence pipeline.
- Note on location: `experiments/` already existed with earlier scripts (`Ideas.md`,
  `ablations.py`, `toy_landscape*.py`, `experiment_generic_repair.py`, ...). Those files are
  unchanged and are not indexed here; the design docs are added alongside them, and the
  sketches live in `experiments/sketches/`. No engine-generated file, config or script was
  edited.
- Markers used: PROPOSED, VERIFY, UNCHECKED, DERIVED (textbook derivation, not measured),
  SKETCH (runnable code that has passed only its own `--selftest`).
- Source spot checks: existence of cited items was checked via Crossref / arXiv on
  2026-11-24 and abstracts were read where available; papers were not read in full. Items
  not checked are marked UNCHECKED in each file.

## Index

| topic | file | gaps | experiments | tier | status |
|---|---|---|---|---|---|
| Engine evidence pipeline (all topics) | [engine-evidence-audit.md](engine-evidence-audit.md) | ENG-G1..G6 | EXP-ENG-01 (retrieval precision audit), EXP-ENG-02 (corroboration ablation, sketch), EXP-ENG-03 (survival vs relevance) | laptop CPU (all) | PROPOSED |
| Fisher information and thermodynamics of learning | [fisher-information-and-thermodynamics-of-learning.md](fisher-information-and-thermodynamics-of-learning.md) | FIT-G1..G5 | EXP-FIT-01 (repair-energy discretization, sketch), EXP-FIT-02, EXP-FIT-03 (runs inside EXP-LLG-01), EXP-FIT-04 | laptop CPU (all) | PROPOSED |
| Hidden-variable detection / causal discovery from residuals | [hidden-variable-detection-causal-discovery-from-residuals.md](hidden-variable-detection-causal-discovery-from-residuals.md) | HVD-G1..G7 | EXP-HVD-01 (planted trigger, sketch), EXP-HVD-02, EXP-HVD-03, EXP-HVD-04 | laptop CPU (all) | PROPOSED |
| Loss-landscape geometry and basin stability | [loss-landscape-geometry-and-basin-stability.md](loss-landscape-geometry-and-basin-stability.md) | LLG-G1..G5 | EXP-LLG-01 (EWS before escape), EXP-LLG-02, EXP-LLG-03 (κ_eff rescaling, sketch), EXP-LLG-04 (gated) | LLG-01 single CPU hours; LLG-02/03 laptop; LLG-04 single GPU | PROPOSED |
| Model collapse, variance suppression, recursive training | [model-collapse-variance-suppression-and-recursive-training.md](model-collapse-variance-suppression-and-recursive-training.md) | MCR-G1..G5 | EXP-MCR-01 (replace vs accumulate, sketch), EXP-MCR-02 (EWS vs matched null), EXP-MCR-03 (GMR cleaning in a recursive loop) | laptop CPU (all) | PROPOSED |
| Safety drift, alignment repair, parameter-space monitoring | [safety-drift-alignment-repair-and-parameter-space-monitoring.md](safety-drift-alignment-repair-and-parameter-space-monitoring.md) | SDM-G1..G5 | EXP-SDM-01 (monitors at fixed step), EXP-SDM-02 (repair vs baselines), EXP-SDM-03 (snapshot-delta in float), EXP-SDM-04 (similarity replication) | SDM-01/04 single GPU; SDM-02/03 laptop | PROPOSED |

Totals: 33 gaps (6 engine-wide + 27 topic), 22 experiments, 5 sketches.

## Suggested order (cheap and informative first)

1. **EXP-ENG-02** — ablate the engine's corroboration rule. Seconds on a laptop; decides how
   much any "surviving" count means for all five topics.
2. **EXP-LLG-03** — does κ_eff change under a function-preserving rescaling of ToyLLM? Seconds;
   a yes means κ_eff reads coordinates, not the landscape.
3. **EXP-FIT-01** — does the repo's repair energy depend on step size along a fixed path?
   Minutes; separates a path property from a discretization artefact.

Highest scientific value at moderate cost: **EXP-LLG-01** (do EWS precede basin escape, with a
matched null), which also carries EXP-FIT-03.

## Sketches

See [sketches/README.md](sketches/README.md). Each runs offline; only `--selftest` has been run.
