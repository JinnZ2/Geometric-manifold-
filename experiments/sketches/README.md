# Sketches (SKETCH, not results)

Written 2026-11-24 against `02320b51`. Each script runs offline (standard library + numpy;
`llg_rescaling.py` also needs torch, as the repo's tests do). For each, only `--selftest`
has been run by the author; full runs have NOT been done and no numbers from them are
reported anywhere in `experiments/`. Scripts that touch the engine or the toy model import
repo code read-only.

| sketch | experiment | selftest command | full run |
|---|---|---|---|
| `eng_lexical_audit.py` | EXP-ENG-02 | `python3 experiments/sketches/eng_lexical_audit.py --selftest` | `--data-dir data` — NOT run |
| `fit_discretization.py` | EXP-FIT-01 | `python3 experiments/sketches/fit_discretization.py --selftest` | `--Ns ...` — NOT run |
| `hvd_planted_trigger.py` | EXP-HVD-01 | `python3 experiments/sketches/hvd_planted_trigger.py --selftest` | NOT run |
| `llg_rescaling.py` | EXP-LLG-03 | `python3 experiments/sketches/llg_rescaling.py --selftest` | `--alphas ...` — NOT run |
| `mcr_recursive_gaussian.py` | EXP-MCR-01/02 | `python3 experiments/sketches/mcr_recursive_gaussian.py --selftest` | `--n --T --seeds` — NOT run |

Selftests check code correctness against known answers (e.g. the DERIVED Gaussian-MLE
variance recursion, exact function preservation under rescaling). Passing a selftest is a
structural check, not empirical support for any hypothesis.
