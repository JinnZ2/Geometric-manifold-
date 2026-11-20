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

### 2b. Context for the "OBSERVED (synthetic)" labels (owner, relayed 2026-11-20)

The DeepSeek app on the owner's phone has no code execution. Its "OBSERVED (synthetic)" labels
on `harness_self_test` and `passk_unit_check` mark checks it could not run. They are recorded
here as a **mislabel from an environment limit, not a run claim**: no run by DeepSeek is
asserted or disputed. Source: the owner's statement; the filer did not see the DeepSeek session.
This applies to R1: the crash was not seen by the author because the self-test was never
executed there. The filed slots tables stay verbatim.

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

## 6. `_v2` files (separate commit; originals not edited)

`step1_recheck_v2.py` and `step1b_passk_from_dump_v2.py` sit beside the originals. Changes, by
Claude Code at the owner's order, 2026-11-20:

| # | change | file |
|---|---|---|
| 1 | `self_test()` calls `run(recs, {})` (R1) | step1_recheck_v2 |
| 2 | the first record is chained back in front of the rest instead of replacing record 2 (R2); the loop is now `collect()` | step1b_v2 |
| 3 | `"logprobs"` removed from `ENTROPY_FIELDS` and `"score"` from `CORRECT_FIELDS`; they are used only when pinned with `--map` | step1_recheck_v2 |
| 4 | results and console output carry `records_seen`, `limit`, `limit_applied` (`limit_applied` is true only when a record was left unread) | both |
| 5 | metric keys renamed: `shift_rate` → `marker_rate`, `entropy_gated_gain` → `spike_shift_assoc` | step1_recheck_v2 |
| 6 | self-test is deterministic (exact counts: 200 with a marker, 130 correct; 1800 without, 990 correct) and asserts the planted accuracy effect `0.65 − 0.55 == 0.10`, the limit fields, and that `score` / `logprobs` are not auto-detected | step1_recheck_v2 |
| 7 | `--self-test` added: 3 records, one problem, one model; asserts `n == 3, c == 3` (3 of 3), and `--limit 2` gives `records_seen == 2, limit_applied == true` | step1b_v2 |
| 8 | `--map` added (the original names it in an error message but has no such option; needed once change 3 makes pinning the only route for those fields); imports from `step1_recheck_v2` | step1b_v2 |

Changes 7 (as a CLI flag) and 8 go beyond the wording of the order; they are listed so they can
be struck. `accuracy_given_shift` keeps its name; the order named two renames. `passk.py` has no
`_v2`.

Runs by the filer, 2026-11-20, Python 3.13.18:

- `python3 step1_recheck_v2.py --self-test` → exit 0, `"self_test": "PASS"`,
  `marker_rate 0.1`, `with_shift 0.65`, `without_shift 0.55`, `planted_effect 0.1`.
- `python3 step1b_passk_from_dump_v2.py --self-test` → exit 0,
  `{"self_test": "PASS", "counted": "3 of 3"}`.
- The 3-record input from R2 through `step1b_passk_from_dump_v2.py` → `records seen = 3`,
  `max samples/problem = 3`.

These are synthetic runs of the `_v2` files. They do not change the slots in the filed
`EXPERIMENT.md`, which stay verbatim; `harness_self_test` for the original stays VOID. Neither
script has been run on the Princeton dump.
