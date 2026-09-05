# drift_spectrum — parameter-drift spectral signature (Tier 1)

Built to the work order of 2026-09-04 (the one surviving item of a literature-roundup
audit). Asks whether Li et al.'s representation-geometry instruments (RankMe, alpha-ReQ;
arXiv 2509.23024) read anything when pointed at PARAMETER drift, which has no sample
dimension — so three candidate axes supply one, and every spectrum names its axis.

Stdlib only, pure Python, ~40 s for the full sweep on this machine. Nothing here is a
claim about a real network or about whether basin repair is doing the right thing.

## Read in this order

1. `INVENTORY.md` — what the repo measures, triggers, does, retains (order §2). Emitted by
   `inventory.py`; its `[MECHANICAL]` lines are re-checked by `inventory.py --selftest`.
2. `RESULTS.md` §0 — the licence. **S4 fails**: the reference channel reproduces two of the
   three published phases (collapse, expansion) and never the third (compression), on any
   setting tried (`config.json` → `calibration_record`). Section 4 of the order is therefore
   not run and no LOCKED / ANTI-PHASE / DECOUPLED verdict exists.
3. `RESULTS.md` §2 — the drift axes, per axis, per seed, as observations. They disagree on
   whether drift has non-monotone spectral structure, which per §8 opens `BRANCH.md` entry 02.
4. `BRANCH.md`.

## Run

```
python3 inventory.py --selftest      # the repo facts INVENTORY.md rests on
python3 selftest.py                  # S1-S5 + align known answers; exits 1 on S4, which is the recorded state
python3 run_t1.py [--quick]          # -> results/<stamp>/raw.json
python3 report.py                    # -> RESULTS.md (+ BRANCH.md if the axes disagree)
python3 -m pytest tests/test_drift_spectrum.py   # from the repo root; pins the S4 state
```

## Files (as the order lists them)

```
inventory.py   §2  repo inventory -> INVENTORY.md
drift.py       §3  MLP, skewed data, the three axes (A1 TIME / A2 UNIT / A3 SEED, raw + permutation-aligned)
spectrum.py    §3  Jacobi eigensolver, RankMe, alpha-ReQ, Spectrum (refuses a missing axis)
run_t1.py      §3  the sweep; also re-implements the repo's two repair-flag rules
align.py       §4  three-leg detector, alignment, repair-flag overlay
report.py      §3  RESULTS.md, per axis, no averaging
selftest.py    §6  S1-S5
config.json        every constant, every [CHOICE], and the calibration record
```

## Two defects found by the folder's own checks, kept on record

- The first A2 implementation handed the axis `theta - prev` at a point where `prev` had
  already been advanced to `theta`, so A2 read exactly 0.00 on every step. Found by looking at
  the first quick run's table, fixed before the recorded run.
- The first three-leg detector took the global maximum as the peak and searched for the
  trough before it, so any series starting at its maximum could never show three legs. Found
  by the fall-rise-fall known-answer fixture in `selftest.py`, fixed; the change moved only
  control-side drift rows and left S4 and the skewed reference rows unchanged.

## Tier 2

Specified in the order (§5): the same A1/A2 machinery on an open checkpoint series such as
Pythia, where the reference channel is already published. Not attempted here; compute and
egress both exceed this environment. Tier 2 confirms or breaks Tier 1 and does not replace it.
