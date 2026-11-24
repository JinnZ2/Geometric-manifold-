# DEVIATIONS — the catch matrix of 2026-11-20

Status: analysis record, docs-only. Written 2026-11-21. Co-authored (JinnZ2 + chat
Kimi). Nothing here is new code; every row cites a record that exists.
Observer slots: null.

Tags: CONVERGENT = found independently by more than one reviewer. DIVERGENT = found
by exactly one. IMMUNITY = the defect never existed in that lineage (construction,
not detection). MISS = the defect is present in that lineage's artifact and that
reviewer did not find it. SELF = found by the same reviewer who made it, later.

The reviewers and their stances:

| id | who | stance | channel limits |
|---|---|---|---|
| DS | DeepSeek (phone app, chat) | author | no code execution (owner's statement, REVIEW.md §2b) |
| CC | chat Claude (L0, sandbox + reading) + Claude Code (filer) | auditor: file the claim, check against sources | no live fetch of the HF rows API on record |
| KM | chat Kimi (sandbox) | rebuilder: fetch ground truth, regenerate | single-family execution of its own artifacts |

The natural experiment: CC's findings (R1–R2, D1–D9, P1–P4, F-a–F-d, filed in
`deepseek_2026-11-20/REVIEW.md`) and KM's repairs (R1–R7, in the delivered scripts'
docstrings) were made the same day, in parallel sessions, neither seeing the other.
The artifacts under review were the same DeepSeek drafts. Overlap and non-overlap
are therefore blind measurements, not coordinated ones.

## 1. The matrix

Mechanism classes: RUN = revealed by execution. FETCH = revealed by external data
absent from the artifact. THEORY = revealed by a model of the estimator/domain.
HYGIENE = revealed by attention policy (sourcing, provenance, CLI surfaces,
defaults). OWNER = revealed only by the owner's channel.

| # | finding | class | DS | CC | KM | shape |
|---|---|---|---|---|---|---|
| 1 | self-test crashes (`run()` missing 'mapping') | RUN | could not run → mislabeled OBSERVED | caught (ran), filer + KM reproduced | avoided by construction (ran pre-delivery) | CONVERGENT |
| 2 | step1b drops the 2nd record (3 in, 2 out) | RUN | — | caught (R2) | caught independently (R5) | **CONVERGENT, the blind datapoint** |
| 3 | planted +0.10 effect unasserted, same order as noise | RUN | — | caught (F-d); fixed deterministically | same class fixed differently (+0.20, CI assert) | CONVERGENT (fix) |
| 4 | judge labels ship in the release (`shift_in_reasoning_v1` + gpt-4o rationale fields on every pass1 row) | FETCH | — | not on record | caught (rows API, 2026-11-20) | DIVERGENT, **landscape-changing** (see §3) |
| 5 | full OBSERVED dump schema (28 pass1 keys; pass2 cue fields; G=8; temps; no base arm) | FETCH | guessed (4 axes wrong) | not on record | caught | DIVERGENT |
| 6 | smoke-row anomaly: judged shift, empty prefilter markers, excerpts point at the few-shot exemplar — label provenance not reconstructible from a row | FETCH | — | not on record | caught, recorded open | DIVERGENT, unresolved |
| 7 | KM's Phase-1 note "labels absent" was wrong (card preview truncates the field list) | FETCH | — | — | caught (same fetch) | SELF |
| 8 | capacity probe (2): majority vote cannot raise pass@k; incoherent | THEORY | authored | caught (D7) | MISS (carried verbatim) | DIVERGENT |
| 9 | glyph_sensing_real decides neither way | THEORY | authored | caught (D8) | partial MISS (kept as falsifier) | DIVERGENT |
| 10 | `--limit 20000`: silent file-order subsample | HYGIENE | authored | caught (D4); `_v2` adds `records_seen/limit/limit_applied` | MISS (no transparency fields) | DIVERGENT |
| 11 | 0.8·pass@256(A) threshold: chosen, no source | HYGIENE | authored | caught (D9) | MISS (pre-registered without the label) | DIVERGENT |
| 12 | marker rate ≠ judged shift; assoc ≠ intervention (measurand substitution) | HYGIENE | authored | caught as risk (D1) | addressed as opportunity (labels path + paired design) | complementary halves, §3 |
| 13 | `logprobs` auto-read as entropy | HYGIENE | authored | caught (D2) | IMMUNITY (OBSERVED fields) | DIVERGENT |
| 14 | `score ≥ 0.5` auto-read as correctness (fail-open) | HYGIENE | authored | caught (D3) | IMMUNITY (`is_correct_pred`) | DIVERGENT |
| 15 | one pooled p90 gate across models | HYGIENE | authored | caught (D5); renamed `spike_shift_assoc` | caught independently; remedy differs (per-task p80, paired) | CONVERGENT, remedies diverge |
| 16 | field detection from 1 record | HYGIENE | authored | caught (D6) | addressed by construction (≥80% of ≤500 rows) | CONVERGENT-by-construction |
| 17 | provenance header named the wrong author | HYGIENE | — | caught (P1) | fixed independently (full provenance line) | CONVERGENT |
| 18 | `--map` named in error text, absent from CLI | HYGIENE | authored | caught (F-a) | present by construction | CONVERGENT-by-construction |
| 19 | `--map` file lacking `text` is wholly replaced | HYGIENE | authored | caught (F-b) | UNCHECKED in KM lineage | open |
| 20 | unused `urllib` import | HYGIENE | authored | caught (F-c) | n/a | DIVERGENT (cosmetic) |
| 21 | "glyph": owner's sense (constraint-sensing) vs DS's reduction (tool-choice) | OWNER | reduction authored | recorded (P2) | carried DS's reduction | DIVERGENT, scope finding |
| 22 | direct-to-main path collision with the voice-order file | OWNER/process | — | caught (P3), resolved by folder | n/a (delivery was to output/) | process |
| 23 | DS cites DS's own R1 paper; `observer.ai_interest` fillable externally | observer-slot | — | recorded (P4) | — | observer-slot discipline |

Rows 19–23 are included for completeness; the analysis below uses rows 1–20.

## 2. What the deviations say

**Overlap concentrates on the shared channel.** Every RUN-class finding (rows 1–3)
is CONVERGENT. Execution is a shared channel: anyone who runs, finds; the only
filter is whether the reviewer can run at all (DS could not — and the failure mode
was not silence but a *mislabel*: OBSERVED without a run. Environment limits
manufacture false positives of assurance, not just absences).

**Every divergent finding is divergent by mechanism, not by care.** FETCH catches
(rows 4–6) required leaving the artifact for live data; THEORY catches (8–9)
required the estimator's shape in the head; HYGIENE-only catches (10–11) required
a particular attention policy. CC and KM each missed entire classes the other
harvested. Nobody was sloppy; the channels are simply not the same channels.

**This is `channel_scope` for reviewers.** Each stance has `channels_reached` and
`blind_channels`, and care inside a stance does not remove its blind spots —
exactly the structure CHANNEL_SCOPE.md §1 forces onto instruments. The deviation
matrix is the empirical channel map, and the residual disagreement is where
undetected defect classes live. The deviations are the instrument reading, not
instrument error.

**Stance is a variable with its own risk profile.** The auditor (CC) catches
claim–data mismatches but inherits the draft's anchored design (the `_v2` keeps
the flat layout and pooled gate). The rebuilder (KM) catches reality–doc
mismatches (rows 4–5) but ships new code paths executed by one family only —
under AUDIT-2's own dedupe rule, KM's execution evidence is one level deep.
Immunity-by-regeneration is real (rows 13–14) and so is its price.

**Numerators without denominators.** ~20 catches, unknown totals per class. The
pattern "execution converges, everything else diverges" is an anecdote shaped
like a law. The instrument that supplies denominators is `PLANTED_DEFECT_SPEC.md`
(beside this file): seeded defects with known answers, catch = reproduced or
mechanism-named, false alarms counted. Until that runs, every cell above is a
sighting, not a rate.

## 3. The landscape-changing deviation (row 4, read slowly)

CC's D1 was correct given what CC could see: markers ≠ judged shift, so the
draft's metric could not confirm or refute 6.31%. KM's fetch then showed the
judged labels *ship in the release* — so the draft could have been
paper-comparable all along, and the right fix was not (only) a cautious rename
but a primary path the draft never knew existed.

One deviation, three lessons: a finding can be locally correct and globally
overtaken; the reviewer who fetches changes the other reviewer's conclusions
*retroactively*; and KM's row 7 (the same fetch refuting KM's own morning note)
is the same lesson pointed inward. None of this is error-as-shame. It is the
measurement system working at the reviewer level the same way it works at the
script level: a mismatch is data, not a failure to hide.

## 4. What this record does not claim

- No rates. Denominators are VOID until the planted-defect instrument runs.
- No severity weighting: row 20 counts like row 2 here; they are not the same size.
- Rows 12, 15–18 involve judgment about what counts as "the same finding" across
  stances; the tags above are KM's reading and are correctable.
- DS's channel limit is the owner's statement (REVIEW.md §2b); not independently
  verifiable from here.

## 5. Sources

`docs/research/tool-selection-vs-deepening/deepseek_2026-11-20/REVIEW.md` (R1–R2,
D1–D9, P1–P4, F-a–F-d; filer runs); `REVIEW_2026-11-21_kimi.md` (reproductions of
every runnable REVIEW.md claim; the three-lineage comparison; D7/D9 owned);
KM-delivered script docstrings (repairs R1–R7); the rows-API fetch of 2026-11-20
(2 rows); `EXPLORE_HARNESS_SPEC.md` (the VOID slot this analysis feeds).

## 6. Dated addition, 2026-11-24 UTC: one row from another review (append only)

Nothing above this heading is changed; the first 9455 bytes are the file as
received (sha256 01e805a7e757f5adc0edc0a63cba5c26eb18997596d930622be6dd515eb87f98).
Added by the filing session (Claude Code) on the owner's order 336-F3,
co-authored (JinnZ2 + Claude). This row is about a different artifact from
rows 1-23 and does not use the DS / CC / KM columns.

New class, added to this file's class list by this section:
"inherited-from-artifact": "reviewer's claim matches an error already in the artifact's own text."

| reviewer | artifact | confirmed items | reproduced | inherited-from-artifact | differs | not checkable | channel | volume |
|---|---|---|---|---|---|---|---|---|
| GPT | register 2026-11-23 (Simulators #336, `REGISTER_AS_RECEIVED.md` sha 33092afccd1f1081f087edd2cfdf3537375419cb8c82e6612c887a85802eea39) | 6 | 3 | 1 (confirmed 1: matches repo docstring, not code) | 1 | 1 | FETCH/HYGIENE | thin |

Notes:
- Source of every count: Simulators PR #336,
  `audit/gpt_register_2026-11-23/REPRODUCTION.md`, sections C5, D1 and E3
  (not merged at the time of writing).
- differs 1 is confirmed 6 (`META-PROTOCOL.md` read in full, 673 lines; the
  claimed rule is not in it). not checkable 1 is confirmed 5 (the register
  names no path).
- reproduced 3 is as ordered: confirmed 2, 3 and 4. In the source record
  confirmed 3 carries the label "order RECORDED only; no-results STATEMENT
  PRESENT", not REPRODUCED; confirmed 2 and 4 carry REPRODUCED.
- The checks were run by a Claude session; the register's author is GPT.
  No rates; n = 6. Observer slots: null.

## 7. Correction, 2026-11-24 UTC

Appended; every line above is unchanged (first 11117 bytes sha256
`16fe4bea9a25c57b3a75bc8f1aef37d6291967da9dfeedc990bee331b8c1ba1b`).

Class added: "statement-present": the artifact contains the statement the
reviewer points to; what the statement says was not checked.

"Correction 2026-11-23: reproduced 2 (not 3); statement-present 1
 (confirmed 3). Row totals: reproduced 2, statement-present 1,
 inherited-from-artifact 1, differs 1, not checkable 1 = 6.
 The '3' was an ordering error by chat Claude."

The counts of the section 6 row, as corrected (reviewer, artifact, channel
and volume cells are as in section 6):

| confirmed items | reproduced | statement-present | inherited-from-artifact | differs | not checkable |
|---|---|---|---|---|---|
| 6 | 2 (confirmed 2, 4) | 1 (confirmed 3) | 1 (confirmed 1) | 1 (confirmed 6) | 1 (confirmed 5) |

The correction text is the owner's order 336-F4, quoted. Co-authored, Kavik +
Claude. License CC0-1.0. Observer slots: null.
