# Sim Harness Standard v1 — JinnZ2 ecosystem
From bottleneck audit: results must be reproducible, falsifiable, and ledger-native
BY CONSTRUCTION, not by culture. Every sim that wants its result in the ledger
conforms to this standard. Stdlib-only for the harness itself.

## 1. Directory layout
```
sims/<name>/
  run.py            # the sim; reads ONLY config.json + CLI overrides
  config.json       # full parameter set (see manifest below)
  NULL.md           # the null model: what result would mean "no effect"
  REFUTE.md         # explicit falsification threshold(s), written BEFORE running
  results/          # timestamped runs: results/2026-08-14T01Z/
    metrics.json    # machine-readable outputs
    summary.md      # human-readable, auto-generated
    ledger_entry.jsonl  # appended to central ledger by ledger_hook.py
```

## 2. config.json manifest (required fields)
```json
{
  "name": "shape_fold_ews",
  "seeds": [0, 1, 2, 3, 4],
  "sweeps": {"gamma": [0.1, 0.25, 0.5]},
  "null_model": "shuffle_labels",
  "refute_if": "detection lead < 15% of load range at >=3 of 5 seeds",
  "tier": 0,
  "runtime_estimate_s": 120,
  "depends_on": []
}
```
Rules:
- **seeds: minimum 5.** A single-seed result is a pilot, marked PILOT in the ledger.
- **sweeps: at least one parameter** that the claim should be robust to.
- **null_model: mandatory.** If you can't name the null, you don't have an experiment.
- **refute_if: mandatory, quantitative, pre-committed.** The ledger import adapter
  already rejects claims without refutation conditions — the harness enforces it
  upstream.

## 3. Execution contract
`python3 run.py` must:
1. Read config.json, run ALL seeds x ALL sweep points x null model.
2. Write metrics.json with per-seed, per-sweep-point, and null distributions.
3. Evaluate refute_if AGAINST THE DATA and write verdict: SUPPORTED / REFUTED /
   INCONCLUSIVE (with reason). The sim grades itself; humans may dispute the grade,
   not compute it.
4. Emit ledger_entry.jsonl: {"type":"PREDICT"|"MEASURE","claim":...,"refute_if":...,
   "verdict":...,"metrics_hash":sha256(metrics.json),"seeds":n,"config_hash":...}

## 4. Verdict discipline
- SUPPORTED requires: effect in predicted direction, at >=4/5 seeds, above null
  by the pre-committed margin.
- Any post-hoc change to refute_if after seeing data → the entry is marked
  EXPLORATORY, never PREDICT. Exploratory entries are welcome but wear the label.
- INCONCLUSIVE is a first-class verdict (see kappa_eff random-ray result).

## 5. Retrofit queue (existing sims, priority order)
1. ep2_prereg_sim.py -> becomes the physical instrument's formal pre-registration
2. shape_csd_probes.py (headline CSD claim)
3. kappa_eff_kill_test.py (verdict flipped on criterion choice — needs the full
   criterion-sweep recorded in config)
4. fractal_basin_sim.py (alpha measured at single damping — sweep gamma mandatory)
5. s3_s7.py, shape_fold_*.py, snap_information_sim.py

## 6. What this buys
- Every claim in every notes/*.md becomes traceable to a config hash.
- The ledger stops being a document and becomes a database.
- External replication = clone + python3 run.py. That is the whole JOSS story.
