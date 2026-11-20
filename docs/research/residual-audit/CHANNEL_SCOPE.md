# `channel_scope` — record-field specification

**Status:** PROPOSED — held under the 2026-11-11 feature freeze. Docs-only until the freeze lifts; no code changes are authorized by this document.
**Origin:** scope law from Madl, *Channel Location Constrains the Auditability of Subliminal Learning* (single author, [arXiv 2606.22019](https://arxiv.org/abs/2606.22019)), via the tooling report §9.1.
**Rule carried on every record:** the carrier regime the run assumes, plus the channels the record is blind to — with `body` always listed.

## 1. Field shape

Every audit record MUST carry a top-level `channel_scope` object:

```json
"channel_scope": {
  "assumed_regime": "vocab_geometry",
  "channels_reached": ["char_ngram", "token_multiset", "token_order"],
  "blind_channels": ["body", "semantic"],
  "notes": null
}
```

| Subfield | Type | Rule |
|---|---|---|
| `assumed_regime` | enum string | One of `init_dependent_body`, `vocab_geometry`, `conditional_body`, `mixed`, `unknown` — the three regimes of arXiv 2606.22019 plus honest fallbacks. `unknown` is a valid value; never guess a regime. Use `mixed` when claims in one run span regimes, with per-claim detail in `notes`. |
| `channels_reached` | list of strings | Channels the run's ladder could in principle affect, per the current calibration. Today: `char_ngram`, `token_multiset`, `token_order`. When `ra.laddercal` publishes its carrier × rung matrix, this list is read from the calibration file, not asserted. |
| `blind_channels` | list of strings | **MUST always contain `body`.** No current rung reaches body computation; the conditional-body regime (sycophancy transfer at ~0.63 of teacher effect, evading four audits) is invisible to this instrument by construction. |
| `notes` | string or null | Free text; null by default (observer-slot discipline). |

## 2. Invariants

1. `body` ∈ `blind_channels` for **every** record, without exception. A run claiming otherwise is a schema violation.
2. `semantic` ∈ `blind_channels` unless the run included an active semantic instrument (the proposed `ra.nli` adapter) **and** the record references that instrument's identity and calibration version.
3. Claims MUST NOT exceed `channels_reached`. A residual of 0 on surface channels is a statement about surface channels only; the record is the unit of honesty.
4. `channel_scope` is the per-record rollup; the existing per-rung `limiting_channel` field is unchanged. `weak_destruction` is unchanged.
5. Intervals stay intervals: `channel_scope` annotates what was measurable; it never upgrades an `residual_undecided` to a verdict.

## 3. Backfill policy (append-only)

Existing archives (including EXP1's twelve records) are **not edited in place**. Backfill is a new JSONL annotation record per run:

```json
{
  "schema": "residual-audit/1",
  "record_type": "channel_scope_annotation",
  "annotates": "runs/exp1/archive/records.jsonl",
  "channel_scope": {
    "assumed_regime": "unknown",
    "channels_reached": ["char_ngram", "token_multiset", "token_order"],
    "blind_channels": ["body", "semantic"],
    "notes": "backfilled 2026-11-20; operator==Q confound already recorded per record"
  }
}
```

## 4. Freeze-held implementation sketch (do NOT apply while the 2026-11-11 freeze stands)

For the post-freeze sprint, the change is small by design:

- `audit()`: add `channel_scope` built from a module constant `SURFACE_CHANNELS` plus the rules above; add CLI flag `--regime` (default `unknown`, never inferred).
- `selftest()`: add **T12** — for every produced record, assert `"body" in record["channel_scope"]["blind_channels"]`; assert `assumed_regime` is in the enum; assert schema violation raises.
- `compare()`: when two records differ in `assumed_regime`, flag `regime_mismatch` alongside `kin_only` states; cross-regime residual comparisons are UNCHECKED by default.
- Estimated size: ~15 lines plus one selftest. Deliberately deferred so it lands together with `ra.manifest`, whose hash chain will then cover the new field from day one.

## Addition 2026-11-20 (dated append; the text above is unchanged)

Sha256 of this file before the addition: `e96f49531a2dd5a3defd1113a5b677133a04cf32bfb5cbab15f1657cd40dd2e0`.

(a) Freeze line superseded. The status line and the §4 heading say the change is held
under the 2026-11-11 feature freeze. The freeze has a lift record: Simulators
`FREEZE_CLEANUP_MEASURAND.md` §6, at Simulators commit `56a208e`. Those two lines are
kept as written and are read as superseded by the lift record.

(b) Two rules added, beyond §1, §2 and §4:

- An audit run outside its carrier regime returns status `VOID` with the regime named;
  it never returns a pass.
- A pass under `assumed_regime: unknown` reads `holds_regime_unknown`, never `holds`.

Source for both: `../RESEARCH_IN_PROGRESS_subliminal_learning.md` §1.1 (Madl,
arXiv 2606.22019, abstract read; full text not read): "an audit used outside its
carrier regime can give false assurance". The rules themselves are the owner's order of
2026-11-20; §4 as written has only the `regime_mismatch` flag on `compare()`.

(c) Where §4 has been applied. §4 was applied by transfer to
`../subliminal_guard_v2.py` (`audit()` there; `subliminal_guard.py` untouched), with
`channels_reached` naming what that function reads rather than the ladder channels of §1.
§4's own target, the residual-audit `audit()` / `selftest()` / `compare()`, is still
unbuilt. The §3 backfill has not been run.
