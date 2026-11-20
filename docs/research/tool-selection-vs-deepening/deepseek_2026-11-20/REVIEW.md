# REVIEW of the DeepSeek tool-selection files (2026-11-20)

Status: review record. Filed by Claude Code (session trailer on the commit) at the owner's
order. Observer slots: null.

## 1. What is filed here, and how

| file | source | body sha256 (before the header line was added) |
|---|---|---|
| `EXPERIMENT.md` | 4 fenced blocks of the owner's message, in the order received | `c099602f…` / `67fd1518…` / `2d7e5174…` / `754b65a6…` (first 8 hex each) |
| `step1_recheck.py` | 1 fenced block | `de986ca5c221a6ea…` |
| `passk.py` | 1 fenced block | `9883e1bb17238a75…` |
| `step1b_passk_from_dump.py` | 1 fenced block | `ee455abf2a79d2fb…` |
| `NOURISHMENT.md` | 2 fenced blocks (note, then addendum), in the order received | `2ebd081b…` / `b22c9149…` |

- The text was taken from the fenced blocks of the owner's chat message of 2026-11-20. No file
  attachment was found on disk; the message is the only source.
- Additions by the filer, and nothing else: the header line (`<!-- … -->` on line 1 of the `.md`
  files; a `#` comment on line 2 of the `.py` files, under the shebang), and one
  `<!-- filer: block N of M … -->` line between blocks in the two `.md` files.
- The blocks are not merged into each other. In `EXPERIMENT.md` the later blocks ("Append to
  EXPERIMENT.md", the `glyph_sensing_real` row, the one line for the oracle section) follow the
  first block as received; the row is not moved into either slots table.
- The prose around the blocks in the message (DeepSeek's commentary) is not filed.
- The voice-order `../EXPERIMENT.md` (commit `82e86b8`) is not touched.

## 2. Slots: what the filed copy says, and what was observed

The slots tables in `EXPERIMENT.md` are verbatim. They say `harness_self_test` and
`passk_unit_check` are OBSERVED (synthetic). Read them with this section.

| slot | filed text | observed here |
|---|---|---|
| `harness_self_test` | OBSERVED (synthetic) | **VOID.** `step1_recheck.py --self-test` crashes (R1 below). It has not passed anywhere this record knows of. |
| `passk_unit_check` | OBSERVED (synthetic) | Unrun until someone runs it; run by the filer on 2026-11-20: **PASS** (section 4). |
| all other slots | not run | not run |

## 3. Findings

### Chat Claude's findings (R1–R2, D1–D9, P1–P4, Holds): text not located

The order names findings R1–R2 (OBSERVED by chat Claude, sandbox), D1–D9 (DERIVED), P1–P4 and
Holds. Their text was not in the message and is not in any repository file the filer could
find. They are **not written here**; nothing below is chat Claude's wording. When the owner
relays them, they go in this section unchanged.

The order's own items 3 and 4 name two faults: the self-test crash and the step1b record drop.
The filer takes these to be R1 and R2. That mapping is the filer's reading, not confirmed.

### R1 (as read from the order): `step1_recheck.py --self-test` crashes

- Chat Claude: OBSERVED in sandbox, per the order (result text not relayed).
- Filer, 2026-11-20, Python 3.13.18, filed copy: **reproduced.** Exit 1,
  `TypeError: run() missing 1 required positional argument: 'mapping'` at `self_test()`,
  `res, err = run(recs)`.

### R2 (as read from the order): `step1b_passk_from_dump.py` drops one record

- Chat Claude: OBSERVED in sandbox, per the order (result text not relayed).
- Filer, 2026-11-20, filed copy: **reproduced.** Input: one `.jsonl`, 3 records, same
  `problem_id`, same model, all correct. Output: `{"n": 2, "c": 2}`,
  `n_samples_correctness_known: 2`, `max samples/problem = 2`.
- Cause, read from the code: `first = next(it)` takes record 1; the loop then runs over the
  rest and at `i == 0` replaces the record it just read (record 2) with `first`. Record 2 is
  never counted.

### Filer's own notes (not chat Claude's; from reading and running the filed copies)

- F-a. `step1b_passk_from_dump.py` prints "run --probe and pin --map" but has no `--map` option.
- F-b. `step1_recheck.py` `run()`: a `--map` file with no `text` key is replaced wholly by
  auto-detection.
- F-c. `step1_recheck.py` imports `urllib.request` and does not use it.
- F-d. The self-test's planted effect (+0.10 accuracy when shifted) is not asserted; with 400
  records and a 10% shift rate the two groups' sampling noise is of the same order as 0.10.

## 4. passk unit check (run by the filer)

Run from this folder, 2026-11-20, Python 3.13.18, against the filed `passk.py`:

```
assert pass_at_k(10,10,1)==1.0 and pass_at_k(10,0,1)==0.0
assert pass_at_k(1,1,1)==1.0
assert abs(pass_at_k(5,2,1)-0.4)<1e-9
assert pass_at_k(2,0,1)==0.0 and pass_at_k(3,1,3)==1.0
```

Result: `passk unit check: PASS`, exit 0. The asserts are DeepSeek's, from the same message.

## 5. Citation for the "CoT can hurt" prior

`NOURISHMENT.md` (addendum) flags this prior as PROPOSED, not cited. The citation the owner
relayed: Liu et al. 2025, arXiv 2410.21333 (ICML 2025). Recorded here only; the filed
`NOURISHMENT.md` is not changed. The filer did not read the paper; identifier and venue are as
relayed.
