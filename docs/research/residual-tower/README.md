# residual-tower

**Status: v0 prototype. Passes its own 22 selftests and a 4-claim replay-checked demo within tested scope. Not yet reviewed by a second reader. Not wired into any of the user's repositories.**

Co-authored (human + AI). Stdlib only, phone-runnable, Python 3.8+. License: CC0 1.0.

In plain words: arithmetic where every number is stored as a **coarse-to-fine tower of codes on declared grids, plus an exact residual** — and where precision is only ever reduced by a declared, measured, archived `truncate()`. It borrows its shape from the audited `experiments/experiment_bridge` bench in [Geometric-to-Binary-Computational-Bridge](https://github.com/JinnZ2/Geometric-to-Binary-Computational-Bridge) (itself prompted by the Stanford NCM preprint, arXiv 2610.14162) and its conscience from [formula-audit](https://github.com/JinnZ2/formula-audit): explicit rejection, VOID instead of guesses, errors as data.

Try it (Python 3, nothing to install):

```sh
python3 residual_tower.py selftest
python3 residual_tower.py demo demo_record.jsonl
python3 residual_tower.py check demo_record.jsonl   # printed-vs-executed diff
```

## What it is

A `Tower` is a list of codes — one per declared uniform `Book` — plus a terminal residual kept as an exact `Fraction`. The invariant

```
value == sum(accepted code values) + terminal residual        (exact, always)
```

holds whether or not any level rejected. Example (from `demo`-grade code, see below):

```
355/113 = 3 + 5/32 − 1/64 + ESCAPE + 7/7232        (exact)
```

Level 3 ESCAPEd there: the incoming residual `7/7232 ≈ 0.000968` sat just outside that book's acceptance band, so the level contributed 0 and recorded the escape instead of silently snapping. Nothing was lost; the residual carries it.

Arithmetic (`add`, `sub`, `mul`, `div`, `dot_tower`) computes exact values first, then re-encodes. Division by a tower whose value is exactly zero is `TowerVoid`. Operations on towers built from different book stacks are `TowerVoid` — alignment across stacks is NOT_BUILT, not guessed.

## The one destructive operation

`truncate(keep_levels)` drops finer levels and **returns the measured dropped magnitude and its fraction of the value**. That is the whole point of the design: floats discard precision silently on every operation; a tower discards only here, on purpose, with a number attached. The report is labeled `split_truncation_defect (arithmetic only; not power conjugacy, not physics)`.

## Traps pinned as selftests

Each of these is a finding from the bridge audit or its specimen audits, encoded as a test so it stays caught:

| Trap (source) | Selftest |
|---|---|
| Printed numbers diverged from executed ones (bridge code audit) | `check` subcommand diffs printed records against re-execution; demo ships a record it regenerates |
| Contribution fraction divided by the wrong split's total (Kimi round-1 correction) | T7: captured fractions use the same-split denominator, never exceed-by-mislabel |
| `measured` hardcoded as a literal (STRUCTURAL finding, rounds 1–2) | T11: `claim()` refuses non-callable `measured`; replay re-computes and compares, timestamps excluded |
| Tolerances without stated reasons (18 of 21 UNSOURCED) | T10: missing `tolerance_reason` is flagged `UNSOURCED` in the record |
| NaN thresholds bypassing rejection | T4: NaN/±inf are `TowerVoid` at the door |
| `zip()` silently dropping coordinates on mismatched vectors | T5: length mismatch is `TowerVoid`, never truncated |
| Reserved zero/escape code competing with real codes | T3: `ESCAPE` is not an `int`, never decodes as a code, and preserves the value whole |
| Cross-term statistic misread as power conjugacy (semantic-physics audit) | T8: `cross_term_defect` carries the fixed label "vector-split orthogonality defect; not power conjugacy" |

## Documented behavior that will surprise you once

**The edge band.** A book's acceptance band ends at its top grid point plus tau, which is narrower than its declared `[lo, hi)` span. A residual sitting within one step of the edge ESCAPEs the next level and cascades, e.g. `1/4` in the demo stack gives codes `[8, ESCAPE, ESCAPE, ESCAPE]` and terminal residual exactly `1/4`. This is recorded behavior, not a defect: the value is exact, the escapes are visible in `per_level()`, and the demo's fourth claim exists to keep it that way. Also: `captured_fraction` can exceed 1 when a level overshoots a small residual (see level 1 of the 355/113 trace) — arithmetic is exact regardless; the fraction describes scale distribution, not error.

## Scope limits

- Uniform grids only. Learned codebooks (the actual RVQ training loop): **NOT_BUILT**. If learned books ever arrive, they belong behind an adapter with seed, corpus, and calibration split declared.
- Vector towers are componentwise with one shared stack; geometric codebooks are NOT_BUILT.
- `z = reconstruction + residual` is an additive identity. Nothing here is energy, effort/flow, or physics.
- No NCM semantics, no Lojban, no COBOL. The coarse-to-fine shape is borrowed; the meanings are not.
- The demo claims are self-referential (the tool checking its own construction). First external use should be a claim about something else.

## Where it could plug in (proposals, nothing wired)

- **formula-audit**: `numcheck`'s intervals answer "is the formula in range?"; towers answer "which scale carries the value?" — the two compose naturally (interval endpoints as towers).
- **residual-audit**: towers give `d_*` rungs an exact arithmetic substrate when claims are numeric.
- **explorative-math**: scale-decomposed sums for exploration records, with `per_level()` as the archive format.

## Files

- `residual_tower.py` — the whole thing: books, towers, arithmetic, claims, replay, check, selftest. One file, stdlib only.
- `demo_record.jsonl` — the shipped demo record; `check` regenerates and diffs it.
