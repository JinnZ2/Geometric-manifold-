# Experiment results — run of 2026-11-27

These are the laptop-CPU experiments from `experiments/*.md`, run on 2026-11-27.

- Data and code are as of commit e814887bed510ef0d4bd67c54820399f0ad088d6.
- Environment: Python 3.14.8, numpy 2.5.5, scipy 1.18.1, torch 2.15.0+cpu, statsmodels 0.15.0.
- Machine: Intel Xeon @ 2.90 GHz, 4 cores, shared with other jobs.
- Each `EXP-*/` directory holds `run.py`, raw output (`raw.json`, plus logs where relevant) and `results.md`. Each `results.md` lists its deviations from the pre-registered design before the results.
- Shared helper: [`common.py`](common.py).
- The designs themselves were not edited. Where a design was wrong, the result file carries a dated note (2026-11-27).

Note: the repo `.gitignore` has `results/` on line 7. That pattern also matches this directory, so these files were added with `git add -f`. If this directory should be tracked, add an exception (`!experiments/results/`). The `.gitignore` was not changed here.

## Index

Verdicts are against each design's pre-registered prediction and falsifier. Everything here is a toy- or repo-scale result. None of it is a claim about real models.

| Experiment | Verdict | Key result |
|---|---|---|
| [EXP-LLG-03](EXP-LLG-03/results.md) | SUPPORTED | κ(α)/κ(1) falls outside [0.95, 1.05] on 20/20 seeds at every α≠1, while the logit difference is exactly 0. The ledger verdict changes with the coordinates. |
| [EXP-FIT-01](EXP-FIT-01/results.md) | Mixed: parts 1–2 SUPPORTED on the falsifier, but bands missed; part 3 FALSIFIED | C scales about ×m; full-Fisher m=4 ratio is 2.59–2.80. The lr×√2 drift probe fires 0/10, because the trust-region clip absorbs it. |
| [EXP-HVD-01](EXP-HVD-01/results.md) | SUPPORTED | Folded power at β_z=1 is at most 0.46. Signed beats folded in 30/36 cells, with the rest tied. |
| [EXP-ENG-02](EXP-ENG-02/results.md) | Mixed: (i) SUPPORTED, (ii) FALSIFIED in the opposite direction, (iii) SUPPORTED | A prefix-only D gives ≥3 passes for every claim in 4/5 lanes. Removing the prefix raises the ≥3-pass counts in LLG, FIT and SDM. |
| [EXP-HVD-02](EXP-HVD-02/results.md) | FALSIFIED as stated; H0 rejected | same_topic_at has a negative logit coefficient, −4.0 [−5.5, −2.5] per SD. abs_tokens is null; semantic_scholar is positive. Permutation p ≤ 5e-4. A claim-age confound is likely (UNCHECKED). |
| [EXP-MCR-01](EXP-MCR-01/results.md) | Mixed, PILOT scale: code check SUPPORTED; replace collapses; accumulate n=20 misses its band | The 1-D check matches within 0.9% at 20k seeds. Accumulate tail retention is 120% at n=100 and 66% at n=20. The 10pp falsifier is mis-scaled (dated note). |
| [EXP-MCR-02](EXP-MCR-02/results.md) | INCONCLUSIVE, PILOT (degenerate event timing) | Events fire early, at median generation 4–13. No T=1.0 cell has both AUCs ≥ 0.7. |
| [EXP-HVD-04](EXP-HVD-04/results.md) | SUPPORTED on the K≥2 criteria, with a qualification | ρ is 0.59–0.85 in 18/18 cells. Detection is mostly a presence step at s=0.01. For K=1, the brace breaks at s=1.0. |
| [EXP-SDM-03](EXP-SDM-03/results.md) | Error growth and Kahan SUPPORTED; flips INCONCLUSIVE | float32 incremental drift ∝ T^0.50 [0.43, 0.54], and Kahan makes it 64× smaller at T=10⁴. No flips at any non-zero margin. |

## Not run (UNCHECKED)

| Experiment | Reason |
|---|---|
| EXP-ENG-01 | Needs human rater labels. |
| EXP-ENG-03 | Needs human labels for the gold set. |
| EXP-FIT-04 | Precondition not met: it requires a word-for-word transcription of Goldt & Seifert, which is not available here. |
| EXP-LLG-01 | Hours-scale tier (about 10⁵ controller steps). Not run within the budget. |
| EXP-LLG-04, EXP-SDM-01, EXP-SDM-04 | GPU tier. |
| EXP-HVD-03, EXP-SDM-02, EXP-MCR-03, EXP-FIT-02, EXP-FIT-03, EXP-LLG-02 | Laptop tier, but not run today. The compute budget on the shared machine went to the cheaper picks above. They remain runnable later. |
