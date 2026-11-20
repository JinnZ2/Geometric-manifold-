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
| nondyadic_escape_count | VOID | non-dyadic values per run |

Observer slots: null.
