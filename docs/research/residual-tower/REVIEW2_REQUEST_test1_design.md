# REVIEW-2 request: design of snapshot-delta Test 1

Status: request, open. Nothing is ordered from it: no code, slot or doc line changes on the
strength of this file alone. Filed 2026-11-20 at the owner's order. Co-authored (JinnZ2 +
Claude).

Independence of the work under review: L1 (Claude Code, fresh session). The same session chose
the design, wrote the script and ran it.

## What to read

- `SNAPSHOT_DELTA_EXPERIMENT.md` (this folder): the "Test 1" section and the "checks" entries.
- `snapshot_delta_test1.py` (this folder): the `[CHOICE]` constants, `planted_cases()`,
  `planted_wrong_cases()`.
- `snapshot_delta_test1_results.json` (this folder): `choices`, `labels`, `speedup_rows`.

## Scope

Two things, and only these.

**A. The choices list.** The doc leaves open which computation, its size, the snapshots, how
cost is measured and how many runs. The builder fixed:

1. snapshots = lists of unsigned 64-bit words, exact integer arithmetic (no tower codes);
2. computations = sum, dot with a fixed weight vector, matrix (8 x 20,000) times vector;
3. sizes = 200,000 entries; 1, 100, 10,000, 100,000, 200,000 changed;
4. cost = operation counts, and wall time as the median of 5 repeats;
5. "bit-identical" = equal canonical bytes of the exact integer results;
6. one seed (20261120);
7. carry-ripple pairs: steps uniform in [1, 65536], random sign;
8. Gray: 10,000 clean unit steps; 1000 with one random bit of the new value flipped;
9. parity: 8 x 8 row-major, bit index = row * 8 + col, bit 0 least significant; 1000 random
   chunks per planted pattern; the second check = difference against the known-good chunk.

For each: does it answer what Test 1 asks, or a different question? Is anything the doc states
contradicted? Which choice, if changed, would change a slot's reading?

**B. The planted cases.** 34 with known answers, 8 planted-wrong. For each rule in the doc's
checks entries: is there a case, is its known answer right, and is a case missing?

## Out of scope

The claim of the experiment, tower snapshots, the VOID slots, the code's style.

## Return

Per item A1-A9 and per rule in B: AGREE / DISAGREE / GUESS, with the line read. Then the
exposure attestation (model, memory, files read, anything fetched, prior exposure, anything
relayed about expected answers).

Observer slots: null.
