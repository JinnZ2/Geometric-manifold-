# Snapshot-delta experiment (residual-tower)

Status: PROPOSED. Held under the 2026-11-11 feature freeze: no code is run for this until the map
REVIEW-2 clears (Simulators `FREEZE_CLEANUP_MEASURAND.md`, section 6). Nothing below has been
run. Source of every line in "Claim" through "Slots": owner work order in chat, 2026-11-20.
Co-authored (JinnZ2 + Claude).

## Claim (PROPOSED)

Binary snapshots are stored as dyadic tower codes plus exact residuals. The XOR of two snapshots
gives the changed bits; the highest changed bit is the coarsest level that moved.

## Linear operations (PROPOSED)

For sum, dot and matmul: f(new) = f(old) + f(delta). Compute on the delta only.

## Nonlinear operations (PROPOSED)

No split. Recompute, or interval-bound the change.

## Known risk

Grid-edge values flip codes on tiny changes. Example: the 1/4 tie in `encode()`. So a diff must
check residuals, not codes alone.

What is on record for the example (rerun from the filed copy, 2026-11-20, see
`../residual-audit/INTERIM_HOME.md`): `encode()` sends a value exactly halfway between two grid
points to the lower one; `encode_tower(1/4)` on the demo stack gives codes
[8, ESCAPE, ESCAPE, ESCAPE] with residual 1/4. The tie rule is unstated in the tower's README.
That a tiny change across such an edge flips codes is the risk as stated in the order; it has
not been measured.

## Test 1 (not run)

A large linear computation, done two ways: incremental from the diff, and full recompute.
Pass = bit-identical results. Record the cost of each.

Not stated in the order, so not fixed here: which computation, its size, what the two snapshots
are, how cost is measured, and how many runs.

## Slots (VOID until run)

| slot | value |
|---|---|
| speedup | VOID |
| mismatch count | VOID |
| edge-flip count | VOID |
| nonlinear bound width | VOID |

### Addition 2026-11-20: tie_flip_count

Source: owner work order in chat, 2026-11-20.

| slot | value |
|---|---|
| tie_flip_count | VOID |

`encode()` rounds exact halves to the lower grid point (OBSERVED 2026-11-20, rerun from the filed
copy). Count flips at tie points separately from edge-band flips.

## Addition 2026-11-20: known risks, second entry

Source: owner work order in chat, 2026-11-20. Placed at the end of the file so the earlier
sections stay as written; it belongs under "Known risk".

- Carry ripple (DERIVED, not measured): a minimal change can flip many bits (0111 -> 1000), so
  the highest set bit of the XOR overstates the coarsest level moved.
- Claim amended: the coarsest level moved is read from the arithmetic difference (new - old),
  not from the XOR. The XOR is kept only as a fast upper bound. The "Claim" section above is kept
  as first written; this line replaces its second sentence.
- Non-dyadic values (e.g. 1/3) never terminate in binary -> always ESCAPE to the residual; count
  them per run.

Note on the last line, from what is on record: the one non-dyadic value rerun so far, 355/113,
gave codes [14, 13, 0, ESCAPE] with terminal residual 7/7232. Three levels accepted a code and one
level ESCAPEd; the value did not ESCAPE at every level. What held is that the terminal residual is
not zero. So "always ESCAPE to the residual" reads here as "always leaves a nonzero terminal
residual"; whether a level ESCAPEs depends on the value. 1/3 itself has not been run.

| slot | value | counts |
|---|---|---|
| carry_ripple_count | VOID | cases where the XOR level and the difference level disagree |
| ~~nondyadic_escape_count~~ | VOID | non-dyadic values per run (renamed 2026-11-20, see the correction at the end) |

## Addition 2026-11-20: checks

Source: owner work order in chat, 2026-11-20. The file had no "Checks" section; this is it.

- Gray cross-check (DERIVED, not run), scoped to unit steps: g = b XOR (b >> 1) for both
  snapshots. A +/-1 step must give exactly one differing Gray bit, at index = binary ripple length
  (trailing-ones count). Agreement = clean step; disagreement = flag a corrupt bit. Independent of
  the arithmetic path, NOT of the input (a bad source bit corrupts both paths).
- Out of scope: for steps > 1, Gray's highest differing bit = the binary XOR's highest bit, so it
  does not fix the overstatement; the subtraction stays authoritative.
- Geometry note (PROPOSED): Gray = Hamiltonian walk on the hypercube, each new bit made by
  reflecting the existing walk; the bit flipped at step n follows the ruler sequence = binary
  carry length at step n.

Notes on the derivation (worked by hand for this file, 2026-11-20; nothing run):

- g(b) XOR g(b') = x XOR (x >> 1), with x = b XOR b'. For b' = b + 1, x is a run of k + 1 ones at
  the bottom (k = trailing ones of b), and x XOR (x >> 1) is the single bit at index k, counting
  from 0. So the index is the trailing-ones count of the LOWER of the two values. For a -1 step
  the lower value is the new one; read from the old value it is the trailing-zeros count.
- The top set bit of x XOR (x >> 1) is the top set bit of x, for any x. That is the "out of scope"
  line.
- The check is stated for unsigned integer codes. How it reads on a tower (one code per level,
  ESCAPE at a level, a carry between levels) is not stated in the order and is not fixed here.

| slot | value |
|---|---|
| gray_unit_agree_count | VOID |
| gray_unit_disagree_count | VOID |

## Correction 2026-11-20: non-dyadic values

Source: chat Claude, relayed in the owner work order of 2026-11-20.

The line "Non-dyadic values (e.g. 1/3) never terminate in binary -> always ESCAPE to the
residual" (second known-risks entry, above) overstated it. That line and the 355/113 note under it
are kept as written.

- Values take codes at each level.
- What holds is a nonzero terminal residual at every finite depth. DERIVED: dyadic codes sum only
  to dyadic values, so a non-dyadic value minus any finite sum of them is not zero. Not run.
- Slot renamed: `nondyadic_escape_count` -> `nondyadic_nonzero_residual_count`. The old name is
  struck in its table and kept.

| slot | value |
|---|---|
| nondyadic_nonzero_residual_count | VOID |

## Addition 2026-11-20: checks, second entry (index rule and scope)

Source: chat Claude, relayed in the owner work order of 2026-11-20. The first checks entry above
is kept as written.

1. Index rule: read the flipped Gray bit from the LOWER of the two values (the trailing-ones
   count of the lower value). This supersedes "trailing-ones count" for both directions in the
   first checks entry. Credit: the hand-worked note under that entry (first bullet of "Notes on
   the derivation").
2. Scope (DERIVED): the Gray cross-check applies to flat binary snapshots only. Tower codes
   (signed, nearest-grid, ESCAPE, exact residual) are not positional digits, so the relation does
   not transfer. Tower snapshots compare by per-level arithmetic difference. This settles the
   third bullet of the hand-worked note ("how it reads on a tower ... is not fixed here").

Nothing run.

## Addition 2026-11-20: checks, third entry (2D chunk geometry)

Source: owner work order in chat, 2026-11-20. Nothing run.

- 2D chunk geometry cross-check (PROPOSED, not run): lay each snapshot chunk as a square
  (64 bits -> 8x8, 256 -> 16x16); store parity per row and per column.
  - single bad bit -> located at the crossing of the bad row and column (detect + correct)
  - two bad bits -> detected, not located
  - blind spot: four errors at the corners of a rectangle cancel in every row and column ->
    undetected (DERIVED)
- Line operations (PROPOSED): shifts move a whole row / column / diagonal in one op (+/-1 row,
  +/-side column, +/-side+/-1 diagonals), as chess bitboards do; candidate for shortening
  per-line calculations. Masks needed at the edges (wraparound), as in bitboards.
- Independent of the arithmetic path and the Gray check; NOT of the source bits.

| slot | value | note |
|---|---|---|
| parity_located_count | VOID | |
| parity_detected_unlocated_count | VOID | |
| rectangle_miss_count | VOID | needs planted 4-corner errors |
| line_op_speedup | VOID | |

Notes (worked by hand for this file, 2026-11-20; nothing run):

- Two bad bits, checked by hand: in one row, the row parity holds and two column parities fail;
  in one column, the reverse; otherwise two rows and two columns fail, which fits two different
  pairs of positions. Detected in each case, located in none. Agrees with the line above.
- "+/-1 row, +/-side column" is read here for a row-major layout: a shift by 1 moves a bit along
  its row, a shift by `side` moves it along its column. The order does not state the layout.
- Not stated in the order, so not fixed here: an error in a stored parity bit itself, and where
  the parity bits are kept.

## Addition 2026-11-20: checks, fourth entry (three-corner miscorrection)

Source: owner message in chat, 2026-11-20, continuing the third checks entry. Nothing run.

- three bad bits on three corners of a rectangle -> miscorrection: parity points at the 4th
  corner and flips a good bit (DERIVED).
- Rule: correct only with a second independent check after correcting.

| slot | value | note |
|---|---|---|
| miscorrection_count | VOID | needs planted 3-corner errors |

Note (worked by hand for this file, 2026-11-20; nothing run): with bad bits at (r1,c1), (r1,c2)
and (r2,c1), row r1 and column c1 each hold two errors and pass; row r2 and column c2 each hold
one and fail. The crossing is (r2,c2), a good bit. Flipping it leaves four errors on the corners
of a rectangle, which is the blind spot of the third entry: every row and column then passes. So
the parity check alone reports the miscorrection as a clean repair, and the "single bad bit ->
detect + correct" line of the third entry holds only when exactly one bit is bad.

## Addition 2026-11-20: checks, fifth entry (2D chunk check: layout, parity bits, storage)

Source: chat Claude, relayed in the owner work order of 2026-11-20. Nothing run. This entry
settles the three points the third entry's hand-worked notes left open.

- Layout: row-major (bitboard convention); shift 1 = along a row, shift `side` = along a column,
  `side`+/-1 = diagonals.
- Three bad bits on three corners of a rectangle -> miscorrection, the rule, and the slot
  `miscorrection_count`: already filed as the fourth checks entry above (same wording; the order
  adds "planted 3-corner errors", which that slot's note carries). Not filed twice.
- Parity-bit error (DERIVED): a data-bit error flags its row AND its column; a lone row flag or a
  lone column flag = the parity bit itself is bad -> leave the data unchanged.
- Parity storage: beside the snapshot in its record, with its own sha256.

Note (worked by hand for this file, 2026-11-20; nothing run): the number of failing rows and the
number of failing columns are both odd or both even, because each equals the number of bad data
bits modulo 2. So data errors alone cannot produce exactly one failing row and no failing
column, or the reverse. A lone flag therefore means at least one parity bit is bad. It does not
show that the data bits are all good: bad data bits can sit under a bad parity bit.

## Addition 2026-11-20: checks, sixth entry (lone parity flag)

Source: chat Claude, relayed in the owner work order of 2026-11-20. It goes with the
parity-bit-error line of the fifth entry, which is kept as written. Nothing run.

- A lone flag proves a parity bit is bad, not that the data is clean; certify the data only with
  an independent path (arithmetic difference or the Gray check), the same rule as the
  three-corner case.

## Addition 2026-11-20: Test 1 run (flat binary snapshots)

Source: owner work order in chat, 2026-11-20 (the order for this held item). Script
`snapshot_delta_test1.py`, results `snapshot_delta_test1_results.json` (status "ran"), both
beside this file. Run by Claude Code, Python 3.13.18. Everything above this entry is kept as
written; "Nothing below has been run" in the status line and "not run" in the entries above
were true when written. Every number below is from the results file.

What Test 1 names, quoted from above: inputs "A large linear computation, done two ways:
incremental from the diff, and full recompute"; measures "Pass = bit-identical results. Record
the cost of each"; slots `speedup`, `mismatch count`, `edge-flip count`,
`nonlinear bound width`.

Choices made by the builder, because this file leaves them open (which computation, size,
snapshots, cost, runs): snapshots are lists of unsigned 64-bit words with exact integer
arithmetic; computations are sum, dot (200,000 entries) and matrix-times-vector (8 x 20,000);
1 to all entries changed; cost is operation counts, plus wall time as the median of 5 repeats;
seed 20261120. Tower codes are not used: the run covers flat binary snapshots only.

Self-test (runs before the run and gates it): 34 planted cases, each with its known answer, all
pass; 8 planted-wrong cases, all reported as failing.

| slot | value | from |
|---|---|---|
| mismatch count | 0 of 13 comparisons (Test 1 pass) | run |
| speedup | not one number; see the table below | run |
| carry_ripple_count | 7434 of 10,000 random pairs (steps up to 65,536); XOR level below the difference level: 0 | run |
| gray_unit_agree_count | 10,000 of 10,000 clean unit steps | run |
| gray_unit_disagree_count | 0 of 10,000 clean unit steps | run |
| parity_located_count | 1000 of 1000 one-error chunks | run |
| parity_detected_unlocated_count | 1000 of 1000 two-error chunks | run |
| rectangle_miss_count | 1000 of 1000 planted 4-corner chunks | run |
| miscorrection_count | 1000 of 1000 planted 3-corner chunks; the second check caught 1000 | run |
| edge-flip count | VOID | not built |
| tie_flip_count | VOID | not built |
| nonlinear bound width | VOID | not built |
| nondyadic_nonzero_residual_count | VOID | not built |
| line_op_speedup | VOID | not built |

Speedup (full cost / incremental cost). "With scan" adds the cost of finding the diff by
comparing all entries of the two snapshots. Wall figures are this machine, this run.

| computation | changed / entries | ops | ops, with scan | wall | wall, with scan |
|---|---|---|---|---|---|
| sum | 1 / 200,000 | 200000 | 1.0 | 3454.57 | 0.4409 |
| sum | 10,000 / 200,000 | 20 | 0.9524 | 7.42 | 0.3248 |
| sum | 200,000 / 200,000 | 1 | 0.5 | 0.71 | 0.1055 |
| dot | 1 / 200,000 | 200000 | 1.0 | 19501.34 | 1.9206 |
| dot | 10,000 / 200,000 | 20 | 0.9524 | 11.48 | 1.3412 |
| dot | 200,000 / 200,000 | 1 | 0.5 | 1.84 | 0.4417 |
| matvec 8 x 20,000 | 1 / 20,000 | 20000 | 7.9968 | 4308.38 | 15.2979 |
| matvec 8 x 20,000 | 20,000 / 20,000 | 1 | 0.8889 | 1.04 | 0.8277 |

Observed in the run, not in the entries above:

- With the diff scan counted, the incremental sum was slower than the full sum at every size
  run, and the incremental dot was faster only while few entries changed. The gain shown without
  the scan needs the list of changed entries to come from somewhere other than a full compare.
- Gray check on corrupted steps: 1000 unit steps with one random bit of the new value flipped;
  996 flagged, 4 not flagged. In all 4 the flip turned the upper neighbour into the lower one
  (or the reverse), which is itself a unit step. Planted case: 5 -> 6 with bit 1 flipped gives 4.
- Lone parity flag: 1000 of 1000 classified as a bad parity bit, data left unchanged in all.

Limits: one seed, one machine; the snapshots are random words, not tower output; the second
check in the parity cases compares against the known-good chunk, which only a test has.

## Addition 2026-11-20: Test 1 labels (second entry)

Source: owner work order in chat, 2026-11-20. Labels only: the run was repeated and the
deterministic sha256 is unchanged (`b88473c48a6edacaa8948fda8347ffab5f2cd2dc76c7f1f71288cbec24f85769`).
The first Test 1 entry above is kept as written. Its wall figures are from the first run (commit
`59143f2`); the results file now holds the repeat run's wall figures, shown here.

Independence L1 (Claude Code, fresh session). Test design choices are judgment; REVIEW-2 of the
design pending. Request: `REVIEW2_REQUEST_test1_design.md`, beside this file.

**1. Speedup, split.** The slot `speedup` is replaced by two; each row in the results file
(`speedup_rows`) declares which one it is and its `diff_source`.

- `speedup_given_diff`: op count. diff_source = given: the list of changed entries is supplied;
  finding it is not counted.
- `speedup_with_scan`: op count and wall time. diff_source = full compare: both snapshots
  compared entry by entry (n compares), counted.

| computation | changed / entries | speedup_given_diff (op count) | speedup_with_scan (op count) | speedup_with_scan (wall) |
|---|---|---|---|---|
| sum | 1 / 200,000 | 200000.0 | 1.0 | 0.3985 |
| sum | 100 / 200,000 | 2000.0 | 0.9995 | 0.4526 |
| sum | 10,000 / 200,000 | 20.0 | 0.9524 | 0.3378 |
| sum | 100,000 / 200,000 | 2.0 | 0.6667 | 0.1442 |
| sum | 200,000 / 200,000 | 1.0 | 0.5 | 0.0951 |
| dot | 1 / 200,000 | 200000.0 | 1.0 | 1.1922 |
| dot | 100 / 200,000 | 2000.0 | 0.9995 | 1.8588 |
| dot | 10,000 / 200,000 | 20.0 | 0.9524 | 1.5325 |
| dot | 100,000 / 200,000 | 2.0 | 0.6667 | 0.507 |
| dot | 200,000 / 200,000 | 1.0 | 0.5 | 0.3582 |
| matvec | 1 / 20,000 | 20000.0 | 7.9968 | 14.7684 |
| matvec | 100 / 20,000 | 200.0 | 7.6923 | 13.3708 |
| matvec | 20,000 / 20,000 | 1.0 | 0.8889 | 0.8442 |

Wall time for the given-diff case is in `timing_this_run` in the results file; it is not a slot.

**2. Parity, three-corner case.**

| field | value | basis |
|---|---|---|
| miscorrection_count | 1000 of 1000 planted 3-corner chunks | run |
| test_reference_catch | 1000 of 1000 | run; the catch compares against the known-good chunk, which exists only in a test (was "the second check caught 1000") |
| miscorrection_detected_in_field | 0 of 1000 | DERIVED, not run: no reference outside a test; after the flip every row and column passes |

**3. Aliasing.** One class name for both rows: *aliasing* = the error lands on a state the check
accepts as valid, so the check cannot detect it.

| row | value | check | class |
|---|---|---|---|
| rectangle_miss_count | 1000 of 1000 planted 4-corner chunks | 2D parity | aliasing |
| gray corrupted_not_flagged | 4 of 1000 corrupted unit steps | Gray unit-step check | aliasing: corruption landing on a valid unit step, undetectable by the unit-step check |

**4. carry_ripple_count, pair rule.** 7434 of 10,000 pairs. Rule: old = uniform random 64-bit
word; step = uniform integer in [1, 65536] with a random sign; new = old + step, or old - step
if that leaves the 64-bit range; a pair is counted when the XOR level differs from the
difference level. The count belongs to this rule; another step range gives another count.

Observer slots: null.
