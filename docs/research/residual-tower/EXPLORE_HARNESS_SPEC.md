# EXPLORE_HARNESS_SPEC — explore/audit harness for residual-tower + snapshot-delta

**Status: PROPOSED · no code yet. Freeze lifted 2026-11-20 (Simulators 56a208e, FREEZE §6); code needs its own order.**
**Amended 2026-11-20 by chat Claude (L0): F1–F6, from uploaded copy sha256 1282196c…3368. Amendment log at the end.**
**Target path in Geometric-manifold-: `docs/research/residual-tower/EXPLORE_HARNESS_SPEC.md`.**
Read first: the repo's READING_PROTOCOL. Map, not prescription. Errors are data. Check it.

This document specifies a harness that does not exist yet. Anything written here
is a claim about a design, tagged PROPOSED, until a run says otherwise.

## Purpose

One stdlib script any AI can run to explore, audit, and re-audit the
residual-tower and snapshot-delta work, logging every mode. The harness never
decides what is true; it records what was run, by whom, at what stage, and what
came out.

## Modes

Non-hierarchical. Authority attaches to **mode + stage**, never to the runner.
A record's weight comes from what kind of run it documents and where that run
sits in the ladder — not from which model or human produced it.

| Mode | Semantics | Allowed status_tag values | May write expected? |
|---|---|---|---|
| BLIND | Run probes with no expected values shown; record raw output | OBSERVED, VOID | Never — field is the string `void` |
| REVIEW | Compare BLIND outputs to pinned expected snapshots; list diffs only | OBSERVED, VOID | No — reads expected, writes diffs, **no verdict key** |
| AUDIT | Re-derive each claim from source; tag every claim OBSERVED / DERIVED / PROPOSED with its source | OBSERVED, DERIVED, PROPOSED, VOID | No |
| SUGGEST | Propose new probes or claims; written as PROPOSED records, never applied by the harness | PROPOSED only | No |
| IMPLEMENT | **None by the harness.** Records with mode=IMPLEMENT are schema-invalid and must be rejected. Humans and other sessions act. | — | — |
| REVIEW-2 | Re-run by a *different model family* on the same pinned inputs; links to the run it confirms via `rerun_of` | OBSERVED, VOID | No |
| AUDIT-2 | Compare runs across families; agreement counted **once per `independence_level`** (multiple same-family confirmations of one probe at one level collapse to a single vote) | OBSERVED, DERIVED, VOID | No |

## Record format

One JSONL record per run, append-only, fsync before close. All fields required;
`void` is a value, not an omission.

| Field | Content |
|---|---|
| `run_id` | sha256 of the record's canonical JSON (sort_keys) with the `run_id` field removed, hex; `recorded_at` is inside the hash, so identical repeat runs get distinct ids |
| `mode` | one of the seven modes above |
| `runner_family` | model family or `human`; free text, declared, not verified |
| `runner_version` | version string as declared by the runner |
| `stack_level` | tower book level L0–L4 the run exercised (whole-stack runs record the finest level reached). **Not** the independence ladder. |
| `stack_spec` | `{top_range, levels, codes_per_level}` of the stack under test, plus `residual_tower.py` sha256 |
| `independence_level` | independence ladder L0–L4 as declared by the runner (L0 same session · L1 fresh session, same family · L2 different family · L3 + no exposure · L4 + a non-model check) |
| `recorded_at` | ISO 8601 timestamp, declared; set before `run_id` is computed |
| `harness_sha256` | sha256 of the harness script that wrote the record |
| `input_sha256s` | map of filename → sha256 for every input the run read |
| `commit` | repo commit of the code under test, or `void` when probing a pinned external specimen |
| `probe` | probe id from the probe table, or `ad_hoc:<sha256 of probe definition>` |
| `raw_output` | verbatim stdout/result blob |
| `expected` | sha256 of the pinned expected snapshot, or the string `void` in BLIND |
| `diff` | list of differences (REVIEW/REVIEW-2/AUDIT-2), else `void` |
| `status_tag` | from the mode's allowed set |
| `exposure` | list of files the runner saw *before* running (BLIND runs must show an empty expected-snapshot exposure) |
| `notes` | free text or `void` |
| `observer` | `{human_interest: void, ai_interest: void, divergence_note: void}` — slots stay void until an external party fills them |

## Canonical probe stack

All pinned probes run on `dyadic_stack(top_range=4, levels=5, codes_per_level=16)`
from `residual_tower.py`, levels named L0–L4. Reference pin at time of writing:
`residual_tower.py` sha256 `94fcd1e8679a6f65375f4034b00cd8a88f93c0bd14a39128eb47f6e85ece92d9`
(v0, 4-level demo stack — records from it are legal but carry `stack_level ≤ L3`;
cross-stack comparisons are AUDIT-2 business and must say so).

Expected snapshots live in a separate `expected/` file set, produced once per
pin, sha256-pinned, and **never shipped to a BLIND runner**. This doc names
probe inputs only; it does not print expected outputs.

## Probe start set

| id | input (exact) | exercises | state |
|---|---|---|---|
| `selftest` | `python3 residual_tower.py selftest` | the pinned traps (count 22: chat Claude sandbox run, 2026-11-20, L0, file sha256 = pin) | RUNNABLE on pin |
| `check-demo` | `check` on the shipped `demo_record.jsonl` | printed-vs-executed discipline | RUNNABLE on pin |
| `pi-approx` | 355/113 | multi-level decomposition; ESCAPE from L3 onward (on the 5-level stack that is L3 and L4) | RUNNABLE |
| `quarter-tie` | 1/4 | exact midpoint tie at L0 (resolves to the **lower** index) + edge-band cascade | RUNNABLE |
| `third-nondyadic` | 1/3 | non-dyadic rational; terminal residual nonzero; exactness holds | RUNNABLE |
| `gray-unit-step` | 1.0 and 3/2 (one L0 step apart) | index adjacency: codes must differ by exactly 1, lower value → lower index. (Name kept from the bridge's Gray-coded bands; the property under test is unit-step monotonicity.) | RUNNABLE |
| `carry-ripple` | 2 − 1/7232 and 2 | **input under review.** Intended: snap-up at a grid boundary (tower analogue of 0111→1000). Sandbox run (L0): both inputs take the same L0 code; they differ only from L3 down, so this pair exercises a fine-level deficit, not a coarse carry. 7232 matches the denominator of the pi-approx residual (DERIVED; no stated source for the choice). New value not chosen here. | RUNNABLE, purpose open |
| `linear-delta` | `add_delta(t, d)` vs full `encode_tower(t.value()+d)`, bit-identical | incremental update path | **BLOCKED — PROPOSED.** No incremental path exists in v0; this probe is its acceptance test if one is ever built. |
| `snapshot-delta-*` | pending | the snapshot-delta specimen lives in the gbcb bridge evidence folder, not yet pinned here | **NOT_RUN — PROPOSED.** Pin the artifact by sha256 first. |

## Exposure detection — VOID, source not located

An earlier addition (held-answer probes, a planted defect, REVIEW-2 of the probe
set, OWNER TODO) is named in session records but its text was not found in the
repo, on any branch, or in this copy. Not rewritten from memory. Slots:

| slot | state |
|---|---|
| `held_answer_probes` | VOID — source not located |
| `planted_defect` | VOID — source not located |
| `probe_set_review2` | VOID — source not located |
| `owner_todo` | VOID — source not located |

Until filled: `exposure` is self-declared only. A run whose exposure cannot be
checked cannot claim `independence_level` L3 or higher.

## Rules

1. Inputs are pinned by sha256. A run on unpinned inputs is VOID, not wrong.
2. The harness never edits source files. Runs are read-only; records go to a
   separate `runs/` directory.
3. `void ≠ pass`. A VOID record is a measurement about the run, not a verdict
   about the code.
4. A mismatch is data. It is recorded, never hidden, never repaired in place.
5. Same-family repeat runs add exposure records, not authority. Cross-family
   agreement is counted once per ladder level (AUDIT-2 dedupe).

## Report format

Per item: OBSERVED / not run, plus the source of each claim (probe id, run_id,
or `void`). The report prints what is missing before what is supported, in the
study-aperture convention. It gives no quality score and no verdict on whether
residual-tower is right.

## What this is not

- Not a benchmark and not a leaderboard; runner hierarchy is explicitly denied.
- Not a certification of residual_tower.py; it is a way to keep re-checking it.
- Not an implementation. IMPLEMENT mode does not exist here on purpose.

## Known gaps (as proposed)

- The `linear-delta` probe blocks on an unbuilt code path; do not "resolve" this
  by writing the path inside the harness.
- Snapshot-delta probes need their artifact pinned into this repo first.
- The v0 demo stack has 4 levels; the canonical probe stack has 5. Records
  carry it in `stack_spec`.
- Exposure detection is VOID (see section); exposure is self-declared until it is filled.
- `carry-ripple` input is under review.

## Amendment log

| id | change | source |
|---|---|---|
| F1 | `ladder_level` split into `stack_level` (tower) and `independence_level` (ladder); AUDIT-2 dedupe keys on `independence_level` | label collision: one label, two measurands |
| F2 | `stack_spec` field | known-gaps line required it; record table lacked it |
| F3 | `recorded_at`, `harness_sha256` | identical runs collided on `run_id`; no time or harness pin |
| F4 | exposure-detection section, VOID | addition not located |
| F5 | dyadic_stack L0 = [−4, 4), step 1/2 (residual_tower.py lines 94–102, Book lines 64–91) → quarter-tie is a tie, gray-unit-step is one step; pi-approx wording; carry-ripple under review | chat Claude sandbox run, L0, sha256 94fcd1e8… = pin |
| F6 | selftest count 22, sha256 94fcd1e8… | same run; source stated in the probe row |

Probe outputs from that sandbox run are not printed here. Any reader who saw the
chat session that produced this amendment is exposed and is not BLIND.

## Addition 2026-11-20: raw_output is sealed in committed records

Dated addition, appended; the text above is unchanged (sha256 of the file before
this addition: `fada7cb0fa8726a6b03bad5a5ae13866ab9cb12893701b45aca6ed73317e4f7c`, at d6d1eda).

- In a public repository, `raw_output` in a committed record is
  `sealed:<sha256 of the canonical output>`, or `void`. The harness refuses to
  append a record with plain output.
- Reason: BLIND runners read the repository. A plain `raw_output` in `runs/`
  shows a later BLIND runner the outputs that `expected/` holds only as hashes.
- The full records (plain `raw_output`) are written to a sealed file outside the
  repository. The outputs are deterministic for a pinned input, so that file can
  be regenerated.
- REVIEW compares the sealed hash to the commitment in `expected/`.
- Plain text appears only at the reveal step, which is a separate, later step.
- `runner_version` may be `void`; the reason is then stated in `notes`.

Exposure note: branch `claude/explore-harness` @ 4d79e45 (PR #34, closed unmerged)
holds plain outputs in `runs/records.jsonl`. Any reader of that branch is exposed
and is not BLIND.

Source: owner order, 2026-11-20 (the first order asked both that plain outputs not
be committed and that the records be committed; see Simulators FIRING_LOG FL09).

## Addition 2026-11-20 (second): what the seal does for deterministic probes

Dated addition, appended; the text above is unchanged.

For deterministic probes the seal blocks reading, not deriving; REVIEW-2 of them
measures reproducibility across runner, machine and environment. Exposure is
load-bearing in AUDIT and SUGGEST.

Source: owner order, 2026-11-20.
