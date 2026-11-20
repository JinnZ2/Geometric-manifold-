# Sprint state, 2026-11-20: ra.manifest, fa.conformal, ra.laddercal, and CHANNEL_SCOPE §4's own target

Status: AUDIT record, written 2026-11-20 at Geometric-manifold- `d7bed77`. No code. Report only;
co-authored (JinnZ2 + Claude). Nothing here starts a build: each item needs its own order.

Tags: OBSERVED = read in a file at the line given. ABSENT = looked for and not there.
SILENT = the text does not say. Quotes are whole lines copied by script from the files named;
long lines are the files' own.

Files read (sha256 at `d7bed77`):

| file | sha256 |
|---|---|
| `INTERIM_HOME.md` | `c87facdc95be260e3f2d2008dbc443b2bac41d53e48871983ffad8afa9f9bcbd` |
| `CHANNEL_SCOPE.md` | `be6ecc7dc2fbdac125e9840616b3004771010f88e532f48f9d8e5a7a6efb2fb6` |
| `tooling-report/Tooling_the_Residual_Check_Survey_and_Build_Plan.md` | `a99b85050b1ff39af6a1d75f25a5d95dfbf9cd6b95851d6c3b7c1c666b04a5f9` |
| `residual_audit.py` | `53ccbfed03b40054d10b4e385dd3dd1c5f736e072f2dc8947bf04ff68c1dd4be` |

Also read: `README.md`, `runs/exp1/EXP1.md` (searched for the four names; no spec text found).
Not read: the `formula-audit`, `explorative-math` repositories (the `fa.` prefix names
`formula-audit`, tooling report line 111); the report's cited papers; the three figures.

What `INTERIM_HOME.md` points to for the sprint:

`INTERIM_HOME.md:61-63`

> 3. Proposed build sprint. The report's "immediate sprint" (line 213) names `ra.manifest`,
>    `fa.conformal` and `ra.laddercal`. Status: PROPOSED, held under the 2026-11-11 feature
>    freeze (owner-stated in the order of 2026-11-20). Nothing from the sprint is built here.
`INTERIM_HOME.md:157-165`

> ## Note, 2026-11-20: freeze source
>
> The notes of 2026-11-20 above (commit `fb3c0f8`) and the status lines in this file say "held
> under the 2026-11-11 feature freeze". The freeze has a lift record: Simulators
> `FREEZE_CLEANUP_MEASURAND.md`, section 6, sixth entry ("Lift record"), at Simulators commit
> `56a208e` (dated 2026-11-20). The earlier lines are kept as written. The lift record does not
> start any held item here: each one (the sprint, `CHANNEL_SCOPE.md` code, the snapshot-delta
> tests) needs its own order from the owner.
>

## 1. Spec text, quoted

### 1.1 `ra.manifest` (OBSERVED)

`tooling-report/Tooling_the_Residual_Check_Survey_and_Build_Plan.md:151`

> **`ra.manifest` — hash-chained archives (effort 1, payoff 3.5).** Each JSONL record gains `prev_hash` and `record_hash` (SHA-256 over canonical serialization), chaining the archive into a tamper-evident log; run headers fix the ladder configuration and scorer versions by hash. Verification is a single stdlib pass — recompute the chain, report the first mismatch — so any third party can check an archive without trusting the project. This is Audit the Whisper's manifest discipline reduced to forty stdlib lines, it retroactively hardens the EXP1 archive, and it is the precondition for the archives being admissible as evidence in any later dispute about what was measured when — including disputes with a future version of the project itself.
Phase and gate lines that name it:

`tooling-report/Tooling_the_Residual_Check_Survey_and_Build_Plan.md:179`

> | 0 | ra.manifest | Tamper-evident archives (retroactive to EXP1) | nothing |
From line 171 and 184, see 1.5.

A separate spec file for `ra.manifest` (field list, canonical serialization rule, header schema):
ABSENT. The paragraph above is the whole spec text in this directory.

### 1.2 `fa.conformal` (OBSERVED)

`tooling-report/Tooling_the_Residual_Check_Survey_and_Build_Plan.md:126`

> **`fa.conformal` — conformal calibration wrapper (effort 2, payoff 4.5).** Given a calibration set of n known-clean examples and any scorer, compute p = (1 + #{calibration scores ≥ observed}) / (n + 1) in exact arithmetic; flag `calibration_thin` when n < 30 and VOID the p-value when the exchangeability precondition is declared unmet by the operator. The wrapper attaches to residual-audit's scorers first (turning raw residuals into calibrated ones — candidate 3's core), but is deliberately scorer-agnostic per ConformalGuard's pattern. Its deeper value is social: a conformal p-value is a claim about *procedure*, not about the scorer's truth, which is exactly the modesty the project requires.
`tooling-report/Tooling_the_Residual_Check_Survey_and_Build_Plan.md:206`

> | 3 | Calibrated residual | formula-audit | Machinery now exists off the shelf ([ConformalGuard](https://arxiv.org/abs/2610.14365), [Audit the Whisper](https://arxiv.org/abs/2510.04303)) | fa.conformal attached to the three existing scorers |

A separate spec file for `fa.conformal` (input format, tie rule, output record shape): ABSENT here.
Its home repository is `formula-audit`, which was not read.

### 1.3 `ra.laddercal` (OBSERVED)

`tooling-report/Tooling_the_Residual_Check_Survey_and_Build_Plan.md:153`

> **`ra.laddercal` — ladder calibration protocol (effort 2, payoff 4.5).** The single highest-leverage addition. Today the ladder's strength is *asserted* by channel-retention arithmetic; calibration makes it *measured*: for each claim class, run a benchmark set with known buried signals (synthetic traits of known carrier — token-level, multiset-level, semantic) and publish which rungs actually destroy which carriers, as a versioned calibration JSON. Madl's three regimes define the carrier taxonomy the calibration must cover, and the EXP1 finding that `vocab_cut` collapses everything while paraphrase survives semantic content becomes the first two rows of the calibration table rather than anecdote. Without this, every residual number in the archive inherits an unquantified "how strong was the destruction, really?" caveat; with it, the harness's weak_destruction flag becomes a calibrated statement.
`tooling-report/Tooling_the_Residual_Check_Survey_and_Build_Plan.md:204`

> | 1 | Carrier localization (ladder × mechanism) | residual-audit + gm.divtoken | Sharpened: channel-location regimes define what "localized" must mean ([2606.22019](https://arxiv.org/abs/2606.22019)) | ra.laddercal, then first ladder×layer matrix on NDIF traces |
`CHANNEL_SCOPE.md:23`

> | `channels_reached` | list of strings | Channels the run's ladder could in principle affect, per the current calibration. Today: `char_ngram`, `token_multiset`, `token_order`. When `ra.laddercal` publishes its carrier × rung matrix, this list is read from the calibration file, not asserted. |

A separate spec file for `ra.laddercal` (benchmark set, calibration JSON schema, version rule):
ABSENT.

### 1.4 CHANNEL_SCOPE §4's own target: `audit()` / `selftest()` / `compare()` (OBSERVED)

`CHANNEL_SCOPE.md:52-60`

>
> ## 4. Freeze-held implementation sketch (do NOT apply while the 2026-11-11 freeze stands)
>
> For the post-freeze sprint, the change is small by design:
>
> - `audit()`: add `channel_scope` built from a module constant `SURFACE_CHANNELS` plus the rules above; add CLI flag `--regime` (default `unknown`, never inferred).
> - `selftest()`: add **T12** — for every produced record, assert `"body" in record["channel_scope"]["blind_channels"]`; assert `assumed_regime` is in the enum; assert schema violation raises.
> - `compare()`: when two records differ in `assumed_regime`, flag `regime_mismatch` alongside `kin_only` states; cross-regime residual comparisons are UNCHECKED by default.
> - Estimated size: ~15 lines plus one selftest. Deliberately deferred so it lands together with `ra.manifest`, whose hash chain will then cover the new field from day one.
The rules §4 calls "the rules above" are §1 (lines 7-25) and §2 (lines 27-33) of the same file;
they are not re-quoted here. The dated addition of 2026-11-20 on this target:

`CHANNEL_SCOPE.md:82-86`

> (c) Where §4 has been applied. §4 was applied by transfer to
> `../subliminal_guard_v2.py` (`audit()` there; `subliminal_guard.py` untouched), with
> `channels_reached` naming what that function reads rather than the ladder channels of §1.
> §4's own target, the residual-audit `audit()` / `selftest()` / `compare()`, is still
> unbuilt. The §3 backfill has not been run.

### 1.5 Lines that order or gate the four (OBSERVED)

`tooling-report/Tooling_the_Residual_Check_Survey_and_Build_Plan.md:13`

> The recommended build order is: **`ra.manifest` and `ra.laddercal` first** (cheap, and they make every later result interpretable), then the three high-payoff/low-effort instruments **`fa.conformal`, `gm.cka-rsa`, `em.mirror`**, then the adapter-mediated instruments (NDIF remote traces, NLI scorer, perplexity-gap channel) that extend the stack without breaking the stdlib-only rule. The single most important honest-limit statement is this: **the residual harness measures the data/output side only**; body-channel transfer of the kind Madl documents for sycophancy evaded four audits across two model families, and no tool proposed here changes that. What the stack *can* do is make channel-local measurements with calibrated uncertainty, refuse to overclaim, and archive everything so that stronger instruments can re-audit the same records later.
`tooling-report/Tooling_the_Residual_Check_Survey_and_Build_Plan.md:171`

> The build order follows the figure's geometry. **Phase 0** is `ra.manifest` alone: forty lines, and it hardens every archive that already exists, including EXP1's twelve records. **Phase 1** is the cheap-high-payoff cluster — `fa.conformal`, `ra.laddercal`, `gm.cka-rsa`, `em.mirror` — after which every subsequent measurement is calibrated, the ladder's strength claims are measured rather than asserted, and em's regenerations are self-checking. **Phase 2** adds the adapter-mediated instruments (`gm.ndif`, `ra.nli`, `fa.judgebias`, `ra.paraphrase-rung`), which require network access or external models but no change to any core. **Phase 3** is the research-facing layer — `gm.probe`, `gm.divtoken`, `em.bytelens`, `ra.pplgap`, `fa.units`, `fa.evalue` — where the instruments are built alongside the experiments they serve, because their exact shape depends on what Phases 0–2 reveal.
`tooling-report/Tooling_the_Residual_Check_Survey_and_Build_Plan.md:180`

> | 1 | fa.conformal · ra.laddercal · gm.cka-rsa · em.mirror | Calibrated residuals; measured ladder strength; kinship distance v1; self-checking regeneration | Phase 0 |
`tooling-report/Tooling_the_Residual_Check_Survey_and_Build_Plan.md:184`

> Each phase has an explicit exit gate, because a build order without gates is just a list. Phase 0 exits when the EXP1 archive verifies clean under the new hash chain and a tampered copy is demonstrably detected. Phase 1 exits when the same twelve EXP1 records, re-scored under `fa.conformal`, produce calibrated p-values whose calibration-set provenance is itself archived — and when `ra.laddercal` has published its first carrier × rung destruction matrix, even if most cells are still VOID. Phase 2 exits when at least one record exists whose semantic-channel measurement did not come from the operator's own lineage — the EXP1 confound retired in practice, not just in documentation. Phase 3 has no exit gate; it is the standing research program, and its instruments graduate into earlier phases only after they have produced one archived result each.
`tooling-report/Tooling_the_Residual_Check_Survey_and_Build_Plan.md:186`

> Two sequencing notes deserve emphasis. First, `ra.laddercal` gates the interpretability of everything downstream: until the ladder's per-carrier strength is measured, residual numbers are comparable within a run but not across runs, because the destruction denominators differ. Second, the adapter instruments are deliberately placed *after* the calibration cluster so that when an adapter is absent — no network on the reservation, no NDIF affiliation, no NLI model — the degraded mode still produces calibrated, archived, honestly-flagged results rather than a dead pipeline. This is the EXP1 lesson generalized: a null run with recorded confounds is a usable instrument; a rich run with unrecorded ones is not.
`tooling-report/Tooling_the_Residual_Check_Survey_and_Build_Plan.md:213`

> The immediate sprint falls out of the table without further argument: `ra.manifest` (Phase 0, this week of work, forty lines), then `fa.conformal` attached to the three existing scorers so that the next archive contains calibrated p-values, then `ra.laddercal`'s first synthetic-carrier benchmark. Everything else in the roadmap waits on those three, and none of them requires network, affiliation, or new hardware — which, given the operating context, is not a coincidence but a selection criterion the survey was run under from the start. Once the sprint lands, the archive finally supports the experiment EXP1 was rehearsing for: two genuinely external lineages as Q, the same twelve-record grid, and a `compare` run whose residual intervals are calibrated, hash-chained, and scoped to named channels.

## 2. Per item: inputs, outputs, slots, tests named, and what exists in code

"In code" means `residual_audit.py` in this directory unless another file is named.

### 2.1 `ra.manifest`

| aspect | text | tag |
|---|---|---|
| inputs | "Each JSONL record"; "run headers" fixing "the ladder configuration and scorer versions" (report:151) | OBSERVED |
| outputs | `prev_hash`, `record_hash` per record; a verification pass that reports "the first mismatch" (report:151) | OBSERVED |
| hash and serialization | "SHA-256 over canonical serialization" (report:151); what "canonical" means | OBSERVED; the rule itself SILENT |
| slots | none named | SILENT |
| tests named | Phase 0 gate: "the EXP1 archive verifies clean under the new hash chain and a tampered copy is demonstrably detected" (report:184) | OBSERVED |
| size | "forty stdlib lines" (report:151); "forty lines" (report:171, 213) | OBSERVED |
| first record's `prev_hash`; how the existing 12 EXP1 records get hashes without being edited | not stated. CHANNEL_SCOPE §3 (lines 35-50) keeps archives append-only for its own backfill; the report says "retroactively hardens" | SILENT |

In code: `append_record` exists at `residual_audit.py:470` (append, flush, fsync; no hash).
`prev_hash`, `record_hash`, `sha256`: ABSENT from the file. `hashlib` is imported (line 46) and used
for SHA-1 only: token buckets (line 192) and `claim_id` (line 442). No run header: ABSENT. The EXP1
archive it would cover exists: `runs/exp1/archive/records.jsonl`, 12 lines.

Nearby, in another directory and not this module: `residual-tower/explore_harness.py` has a
per-record sha256 `run_id` and an append-only log, with no chain between records.

### 2.2 `fa.conformal`

| aspect | text | tag |
|---|---|---|
| inputs | "a calibration set of n known-clean examples and any scorer"; an observed score; an operator declaration on exchangeability (report:126) | OBSERVED |
| outputs | "p = (1 + #{calibration scores ≥ observed}) / (n + 1) in exact arithmetic" (report:126) | OBSERVED |
| flags | `calibration_thin` "when n < 30"; "VOID the p-value when the exchangeability precondition is declared unmet by the operator" (report:126) | OBSERVED |
| slots | none named | SILENT |
| tests named | Phase 1 gate: "the same twelve EXP1 records, re-scored under `fa.conformal`, produce calibrated p-values whose calibration-set provenance is itself archived" (report:184) | OBSERVED |
| where the known-clean calibration set comes from; its minimum content beyond the n < 30 flag | not stated | SILENT |
| which direction of score is "more extreme" for each scorer | the formula uses ≥; per-scorer direction | SILENT |

In code: `conformal`, `p_value`, `calibration`: ABSENT from `residual_audit.py`. The "three
existing scorers" (report:213) exist: `s_containment` (line 113), `s_ngram_cosine` (line 120),
`s_lcs_ratio` (line 124), collected in `SCORERS` (line 131) and run by `measure` (line 138).
Any `fa.conformal` code in `formula-audit`: not read.

### 2.3 `ra.laddercal`

| aspect | text | tag |
|---|---|---|
| inputs | "for each claim class, run a benchmark set with known buried signals (synthetic traits of known carrier — token-level, multiset-level, semantic)" (report:153) | OBSERVED |
| outputs | "which rungs actually destroy which carriers, as a versioned calibration JSON" (report:153); "carrier × rung destruction matrix" (report:184) | OBSERVED |
| carrier taxonomy | "Madl's three regimes define the carrier taxonomy the calibration must cover" (report:153) | OBSERVED |
| first rows | "`vocab_cut` collapses everything while paraphrase survives semantic content becomes the first two rows" (report:153) | OBSERVED |
| slots | matrix cells may be VOID: "even if most cells are still VOID" (report:184) | OBSERVED |
| tests named | Phase 1 gate: the first matrix is published (report:184); sprint: "first synthetic-carrier benchmark" (report:213) | OBSERVED |
| consumer | `channels_reached` "is read from the calibration file, not asserted" (CHANNEL_SCOPE:23) | OBSERVED |
| the claim classes; the benchmark set; the JSON schema; the version rule; how a body-regime carrier is built for a text-only ladder | not stated | SILENT |

In code: the ladder exists: destruction methods `d_scrub` (168), `d_shuffle_windows` (175),
`d_vocab_cut` (186), `d_char_noise` (197), `d_decimate` (218), `d_full_shuffle` (226), `d_external`
(235); `run_ladder` (308); `ladder_summary` (291) with `limiting_channel` (304); the
`weak_destruction` flag (371); `RETENTION_CHANNELS` (274) with the three channel names
`char_ngram`, `token_multiset`, `token_order`. A calibration file, a benchmark set, a carrier ×
rung matrix: ABSENT. A paraphrase rung in code: ABSENT (`ra.paraphrase-rung` is Phase 2, report:171).

### 2.4 §4's own target

| aspect | text | tag |
|---|---|---|
| inputs | a module constant `SURFACE_CHANNELS`; CLI flag `--regime` "(default `unknown`, never inferred)" (CHANNEL_SCOPE:56) | OBSERVED |
| outputs | `channel_scope` on every `audit()` record (CHANNEL_SCOPE:56), fields per §1; `regime_mismatch` flag in `compare()`, "cross-regime residual comparisons are UNCHECKED by default" (CHANNEL_SCOPE:58) | OBSERVED |
| slots | `notes`: "null by default (observer-slot discipline)" (CHANNEL_SCOPE:25) | OBSERVED |
| tests named | **T12** (CHANNEL_SCOPE:57) | OBSERVED |
| size and timing | "~15 lines plus one selftest"; "lands together with `ra.manifest`" (CHANNEL_SCOPE:60) | OBSERVED |
| whether the two rules added 2026-11-20 (VOID outside the regime; `holds_regime_unknown`) apply to this target | the addition states the rules (CHANNEL_SCOPE:71-80) and says this target is unbuilt (82-86); it does not say how they map onto residual records, which have flags and intervals, not a `holds` verdict | SILENT |

In code: `audit` (360), `selftest` (548, tests T1-T11), `cmd_compare` (497, `kin_only` states at
520-533) exist. `channel_scope`, `SURFACE_CHANNELS`, `regime`, `regime_mismatch`, T12: ABSENT from
`residual_audit.py`. The CLI (673-684) has no `--regime`. The 12 EXP1 records carry no
`channel_scope`; the §3 backfill annotation has not been written.
By transfer, not this target: `../subliminal_guard_v2.py` carries `channel_scope` on its own
`audit()` (merged at `d7bed77`).

## 3. Dependencies among the four, from the text only

| pair | text | tag |
|---|---|---|
| `ra.manifest` needs | "nothing" (report:179) | OBSERVED |
| `fa.conformal` needs `ra.manifest` | Phase 1 "Depends on: Phase 0" (report:180); Phase 0 "is `ra.manifest` alone" (report:171) | OBSERVED |
| `ra.laddercal` needs `ra.manifest` | same two lines | OBSERVED |
| `fa.conformal` and `ra.laddercal`, each on the other | same phase (report:180). Order differs by line: line 13 puts `ra.laddercal` first with `ra.manifest` and `fa.conformal` after; line 213 puts `fa.conformal` second and `ra.laddercal` third. Neither line says one needs the other | SILENT on need; the two orders disagree (OBSERVED) |
| everything downstream on `ra.laddercal` | "`ra.laddercal` gates the interpretability of everything downstream" (report:186). Whether `fa.conformal` is "downstream" in that sentence | OBSERVED; SILENT on `fa.conformal` |
| §4 target and `ra.manifest` | "Deliberately deferred so it lands together with `ra.manifest`, whose hash chain will then cover the new field from day one" (CHANNEL_SCOPE:60). Stated as timing; not stated as a need in either direction | OBSERVED (timing); SILENT (need) |
| §4 target needs `ra.laddercal` | `channels_reached`: "Today: `char_ngram`, `token_multiset`, `token_order`. When `ra.laddercal` publishes its carrier × rung matrix, this list is read from the calibration file" (CHANNEL_SCOPE:23). The field can be filled before `ra.laddercal` exists; its source changes after | OBSERVED |
| `ra.laddercal` needs the §4 target | not stated. Both use Madl's regimes (report:153; CHANNEL_SCOPE:22) | SILENT |
| §4 target and `fa.conformal` | neither file names the other | SILENT |
| `ra.manifest` needs the §4 target | not stated | SILENT |
| any of the four on the freeze | superseded by the lift record (INTERIM_HOME:159-165; CHANNEL_SCOPE:66-69); "each one ... needs its own order from the owner" (INTERIM_HOME:163-165) | OBSERVED |

Outside the four, named by the text as needs of these items: `ra.nli` for invariant 2 of
CHANNEL_SCOPE §2 (line 30); the `formula-audit` repository as the home of `fa.conformal`.

## 4. Not done

- No code written or run. `residual_audit.py selftest` was not run for this record.
- The report's citations (other than Madl, see `../RESEARCH_IN_PROGRESS_subliminal_learning.md` §1.1) are UNVERIFIED, as filed.
- `formula-audit` was not read, so "ABSENT" for `fa.conformal` means absent from this directory.

Observer slots: null.
