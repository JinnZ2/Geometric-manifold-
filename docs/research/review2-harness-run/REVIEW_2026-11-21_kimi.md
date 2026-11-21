# REVIEW: new files in Geometric-manifold-, d6d1eda..a440c12 (15 commits, 23 files)

Status: review record. Reviewer: Kimi K3 (Moonshot AI), with code execution.
Independence: L2 (different model family from the Claude-family builder and first
runner; exposure declared in `exposure.json`, beside this file). Written 2026-11-21.
Co-reviewed against the standing house rules: void ≠ pass; a mismatch is data;
no number from memory.

Method note: github.com git endpoints were unreachable from this sandbox; the repo
was fetched as a tarball of `a440c12` via the GitHub API. Consequence: my explore-
harness records carry `commit: "void"` (no .git present); the code under test is
pinned instead by file sha256 (below) and by `harness_sha256` inside each record.

Environment: Linux x86_64 (kernel 6.18.44), CPython 3.12.12. The builder/filer ran
CPython 3.13.18. Cross-version agreement is reported where it matters.

Per item: OBSERVED (ran or read here, source named) / not run.

---

## 1. `tool-selection-vs-deepening/deepseek_2026-11-20/` — the filed DeepSeek files and REVIEW.md

Every runnable claim in the filed `REVIEW.md` was re-run here. OBSERVED:

| REVIEW.md claim | re-run here | result |
|---|---|---|
| R1: filed `step1_recheck.py --self-test` crashes, `TypeError: run() missing 'mapping'` | ran | **reproduced**, exit 1, same TypeError (3.12.12; filer saw 3.13.18) |
| R2: filed `step1b_passk_from_dump.py` drops the 2nd record (3 in, 2 out) | ran | **reproduced**: `{"n": 2, "c": 2}`, `max samples/problem = 2` |
| passk unit check PASS | ran | **PASS**, exit 0 |
| `_v2` step1_recheck self-test PASS (marker_rate 0.1, 0.65/0.55, planted 0.1) | ran | **PASS**, exit 0, same numbers |
| `_v2` step1b self-test PASS ("3 of 3"); `--limit 2` → `limit_applied=True` | ran | **PASS**, exit 0, same |
| `_v2` step1b on the R2 3-record input → 3 of 3 | ran | **reproduced** |

REVIEW.md is accurate on everything executable. Slot consequence (as REVIEW.md states):
`harness_self_test` for the *filed original* is VOID; the `_v2` self-test passes are
synthetic; neither script has run on the Princeton dump. The filed slots tables stay
verbatim and are consistent with that.

Section 2b (the "OBSERVED (synthetic)" labels are a mislabel from an environment
limit — the DeepSeek phone app cannot execute code): the environment claim is the
owner's statement; not verifiable here. Consistent with everything runnable.

### 1b. Three repair lineages now exist — comparison (OBSERVED)

The repo holds DeepSeek's originals + Claude Code's `_v2`. A third lineage was
delivered earlier the same day (`/mnt/agents/output/tool-selection-vs-deepening/`,
Kimi): also fixes R1/R2, independently. Convergences and differences:

| point | DeepSeek original | Claude `_v2` | Kimi delivered |
|---|---|---|---|
| R1 crash | present | fixed (`run(recs, {})`) | fixed (self-test passes) |
| R2 record drop | present | fixed (`collect()`) | fixed (same bug found independently; repair R5) |
| D2/D3 (`logprobs`/`score` auto-read) | present | removed; `--map`-pin only | never present (OBSERVED dump fields used: `is_correct_pred`, `entropy`) |
| D1 (marker ≠ judged shift) | marker-only | marker-only (`marker_rate`) | judge labels primary (`shift_in_reasoning_v1` ships in the release — rows API, 2026-11-20), markers as sensitivities |
| layout | flat only | flat only | passpair (OBSERVED dump layout) + flat fallback; detection over ≤500 rows, ≥80% parseable |
| D5 (one pooled gate) | pooled p90 | pooled p90 (kept) | per-task p80 gates, paired design |
| limit transparency (filer change 4) | absent | `records_seen`/`limit`/`limit_applied` | **absent** — `_v2` is ahead here |
| F-d (planted effect unasserted) | present | deterministic exact-count assert | planted +0.20 with CI-excludes-0 assert |

Data, not a ranking: for the actual dump run, the judge-label path exists only in
the third lineage; the limit-transparency fields exist only in `_v2`. A merge would
serve; not built here (no order).

### 1c. Claude's D7 and D9 land on the Kimi-delivered docs too (owned)

- D9 (`0.8 · pass@256(A)` threshold: chosen, no source): the Kimi-delivered
  `EXPERIMENT.md` carries the same threshold pre-registered **without** an explicit
  "chosen, no external source" label. Accepted finding; a one-line label is the fix.
- D7 (NOURISHMENT capacity probe (2): majority vote cannot raise pass@k): the probe
  text is DeepSeek's, carried verbatim in both the repo filing and the Kimi
  delivery. The critique stands against both; not caught in the Kimi pass.

---

## 2. REVIEW-2 return: explore harness on the pin (per `REVIEW2_REQUEST_harness_run.md`)

Pin: `residual_tower.py` sha256 `94fcd1e8679a6f65375f4034b00cd8a88f93c0bd14a39128eb47f6e85ece92d9`
— **matches** (OBSERVED, computed locally). Harness sha256:
`5f151501f57164ae616d32796ab744e6e1352cfb22b94b3facfba27ed311bbde`.

| command | exit | result |
|---|---|---|
| `explore_harness.py self-test` | 0 | PASS, 26 checks |
| `run --mode BLIND ... --sealed-out /tmp/outside/full_records.jsonl` | 0 | 9 records appended (OBSERVED 7, VOID 2) |
| `run --mode REVIEW ...` | 0 | 9 records appended; **all 7 runnable probes `diff=[]` (hash match)**; 2 `diff=void` |
| `validate` | 0 | `records 36   invalid 0` |

OBSERVED: the 7 RUNNABLE probes reproduce bit-for-bit (by sealed-hash comparison)
on a different model family, machine, OS and Python minor version than the
builder's first run. The two VOIDs are the designed ones (`linear-delta` BLOCKED —
no incremental path in v0; `snapshot-delta-*` NOT_RUN — artifact unpinned); they
are measurements about the run, not failures.

Record discipline (OBSERVED in the appended records): all 18 `raw_output` values
are `sealed:<sha256>` or `void`; `expected` is `void` in BLIND; exposure list
carried verbatim; `runner_version` recorded as declared (`kimi-k3`) — the fixed
"void" reason was not applied because a version was supplied; seal enforcement is
code (`is_sealed` refusal in `append_record`, fsync before close, `IMPLEMENT` in
`REJECTED_MODES`, an open-hook recording `expected/` touches — the repo tests
exercise it).

Return items: `records_after_kimi_run.jsonl` (the earlier 18 lines + my 18) and
`exposure.json` sit beside this file. The plain-output file stays here, unshared,
per the request. I did not open `expected/` myself; the harness did in REVIEW mode
only. Branch `claude/explore-harness` @ 4d79e45: never fetched (stated in
exposure). No AUDIT or SUGGEST output produced — not requested.

## 3. REVIEW-2 return: snapshot-delta Test 1 design (per `REVIEW2_REQUEST_test1_design.md`)

Files read: `SNAPSHOT_DELTA_EXPERIMENT.md` (full), `snapshot_delta_test1.py`
(`[CHOICE]` constants, `planted_cases()`, `planted_wrong_cases()`, run paths),
`snapshot_delta_test1_results.json` (`choices`, `labels`, `speedup_rows`).

Independent rerun first: `--run` reproduced here, exit 0. **`deterministic_sha256`
`b88473c48a6edacaa8948fda8347ffab5f2cd2dc76c7f1f71288cbec24f85769` — exact match**
to the committed file and to the value quoted in the doc's labels entry
(OBSERVED, 3.12.12 vs the builder's 3.13.18). Entire `deterministic` section
byte-equal; self-test 34 planted pass + 8 planted-wrong caught, as committed. Wall
figures differ — declared machine/run-specific, not slots; consistent.

### A. The choices list (A1–A9)

- **A1 (u64 words, exact integers, no tower codes): AGREE.** Test 1 asks for "a
  large linear computation, done two ways"; the doc does not state the snapshots.
  Exact integer arithmetic is load-bearing for "bit-identical": floats would fail
  by associativity and measure arithmetic, not the delta method. Narrower than the
  Claim (tower codes untested) — the script says so itself, and the tower-only
  slots stay VOID. Consistent.
- **A2 (sum, dot, matvec 8×20,000): AGREE.** The doc's "Linear operations" names
  sum, dot and matmul. Matvec is a linear matmul special case; declared [CHOICE].
- **A3 (200,000 entries; changed 1…200,000): AGREE.** "Large" satisfied; the
  changed-ladder is what makes the speedup crossover legible. Slot readings belong
  to this ladder — stated in the labels entry.
- **A4 (op counts + wall median of 5): AGREE.** Op counts deterministic; wall
  demoted from slot status in the labels entry — the right call.
- **A5 (bit-identical = canonical bytes): AGREE.** Matches the pass criterion.
- **A6 (one seed 20261120): AGREE, with a measured caveat.** The seed is recorded.
  The seed-dependent slot is `carry_ripple_count`. Cross-check here: three fresh
  streams (seeds 1, 999, and a fresh-stream 20261120) give 7534, 7488, 7514 per
  10,000 vs the committed 7434 — spread ≈ binomial noise around ~0.75 (σ≈44). The
  count is a draw of the stated pair rule; the labels entry's "another step range
  gives another count" extends to "another seed gives another count (within noise)".
- **A7 (carry-ripple rule: old uniform u64, step uniform ±[1, 65536]): AGREE.**
  Operationalizes the doc's second known-risks entry. Bound confirmed: XOR level
  never below difference level — 0 violations in the committed 10,000 plus 30,000
  further pairs here (40,000 total, OBSERVED).
- **A8 (Gray: 10,000 clean; 1,000 one-bit-corrupted): AGREE.** Matches the checks
  entries. The 4/1000 not-flagged are fully accounted: all 4 land on the other
  unit neighbour (`corrupted_not_flagged_other_unit_neighbour: 4`), and the planted
  case (5→6 flip bit 1 → 4) pins the mechanism. Aliasing class, honestly labeled.
- **A9 (parity 8×8 row-major, bit = row*8+col, bit 0 LSB; 1,000 per pattern;
  second check vs known-good): AGREE.** Layout matches checks entry 5. The
  test-only nature of the second check is declared in limits and sharpened in the
  labels entry (`test_reference_catch` vs `miscorrection_detected_in_field: 0 of
  1000, DERIVED`). That split is the honest version of the number.

### B. Planted cases vs the doc's checks rules

Per rule: difference-authoritative (5 cases + 1 planted-wrong) ✓; Gray unit-step
index-from-lower, both directions, corrupted, aliasing, scope, >1 top-bit relation
(9 + 2 wrong) ✓; 2D parity — 1 located/corrected, 2 detected-not-located in all
three geometries, 3-corner point-at-4th + flip-passes-parity + second-check catch,
4-corner blind spot, lone row/column flag = bad parity bit, data unchanged, lone
flag over 4 bad data bits not certified, "parity never certifies" (14 + 4 wrong)
✓; linear two-ways incl. negative delta (4 + 1 wrong) ✓. Known answers spot-
recomputed by hand here: correct (e.g. gray(7)^gray(8) = 1<<3, k = trailing-ones
of the lower value).

Cases missing: none found for any *testable* rule. The "a bad source bit corrupts
both paths" line (checks entry 1) is a scope statement, not testable by the check
itself — recorded as an observation, not a missing case.

### Exposure attestation (this return)

Model: Kimi K3 (Moonshot AI). Memory: project history notes prior Kimi-family
sessions drafted the residual-tower prototype and EXPLORE_HARNESS_SPEC; no text
from those sessions was in context. Files read: as listed in `exposure.json` plus
`SNAPSHOT_DELTA_EXPERIMENT.md`, `snapshot_delta_test1.py`,
`snapshot_delta_test1_results.json`, `CHANNEL_SCOPE.md`, `SPRINT_STATE_2026-11-20.md`,
the deepseek_2026-11-20 folder, `subliminal_guard_v2.py`, both repo test files.
Fetched: the GitHub API compare and tarball. Nothing relayed about expected
answers by any party; the rerun above was executed before reading the planted-case
expected values in detail.

## 4. `subliminal_guard_v2.py` + `CHANNEL_SCOPE.md` addition

OBSERVED (read + ran): the regime enum matches `CHANNEL_SCOPE.md:22` exactly
(`init_dependent_body`, `vocab_geometry`, `conditional_body`, `mixed`, `unknown`).
Both dated rules are implemented as ordered: out-of-regime regimes
(`init_dependent_body`, `conditional_body`, `mixed` — declared [CHOICE] with the
reason) return `{"status": "VOID", ...}` naming the regime, never a pass; a pass
under `unknown` reads `holds_regime_unknown`, never `holds`. `SURFACE_CHANNELS`
names what `audit()` actually reads, with the invariant-3 justification in the
code — matching the addition's "(c) applied by transfer" note. Schema checks
raise on: missing field, regime outside the enum, `body` dropped from
`blind_channels`, a channel both reached and blind. `--regime` defaults to
`unknown`, never inferred. Self-test: 20 checks, PASS, exit 0. Repo tests
(`tests/test_subliminal_guard_v2.py` + `tests/test_explore_harness.py`): **254
passed, 0 failed** (pytest 9.x, installed for this review). `subliminal_guard.py`
untouched — confirmed by the commit diff (add-only).

One environment note: the two repo test files import `pytest` — the only non-
stdlib dependency in this batch (tests only; every shipped script stays stdlib-
only). Recorded as data; phone-runners can't run the test files.

## 5. `SPRINT_STATE_2026-11-20.md` (audit record)

OBSERVED: spot-checked 2 of 4 recorded file sha256s (`INTERIM_HOME.md`
`c87facdc…`, `CHANNEL_SCOPE.md` `be6ecc7d…`) — exact match at a440c12. The record's
central claims are consistent with the diff: `channel_scope` applied by transfer
to `subliminal_guard_v2.py` only; §4's own target (`residual_audit.py`
audit/selftest/compare, T12) still unbuilt — `channel_scope`, `SURFACE_CHANNELS`,
`--regime`, T12 confirmed ABSENT from `residual_audit.py` (grep, this review). §3
backfill not run — consistent with the EXP1 archive being untouched in the diff.

## 6. Stale status lines (data, not errors)

- `EXPLORE_HARNESS_SPEC.md` header still reads "Status: PROPOSED · no code yet"
  while `explore_harness.py` (791 lines) sits merged beside it. The house
  kept-as-written convention explains it; a cold reader will misread it. A dated
  "code landed at cbf8eea0" marker would close it.
- `SNAPSHOT_DELTA_EXPERIMENT.md` header: "Held under the 2026-11-11 feature
  freeze… Nothing below has been run" — true when written; the dated additions
  supersede it. Same suggestion.
- `passk.py` has no `_v2` and needed none (unit check passes as filed) — noted so
  its absence isn't read as an omission.

## 7. Not done in this review

- Branch `claude/explore-harness` @ 4d79e45 not fetched (would expose plain outputs).
- The Princeton dump: unreachable from this sandbox (HF hosts blocked); all
  tool-selection slots that need it stay `not run`.
- `residual_tower.py` source not read beyond its hash (BLIND discipline kept until
  after the runs; only the harness's seal/append paths were spot-read afterward).
- AUDIT / SUGGEST / AUDIT-2 modes: not requested, not produced.
- The owner's statement behind REVIEW.md §2b (phone app has no code execution):
  relayed, not independently verifiable here.

Observer slots: null.
