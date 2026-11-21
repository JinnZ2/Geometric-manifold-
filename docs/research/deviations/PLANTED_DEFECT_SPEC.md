# PLANTED_DEFECT_SPEC — measuring reviewer catch rates, with denominators

Status: PROPOSED · docs-only · no code. Freeze-safe: nothing here builds anything;
the build needs its own order from the owner. Written 2026-11-21. Co-authored
(JinnZ2 + chat Kimi). Observer slots: null.

Fills, in `EXPLORE_HARNESS_SPEC.md`: the slot `planted_defect — VOID, source not
located`. This document is a source; the slot stays VOID until a run exists.

## 1. Purpose

The catch matrix of 2026-11-20 (`DEVIATIONS_2026-11-21.md`, beside this file) has
numerators and no denominators: ~20 sightings, zero known totals. This instrument
supplies denominators. Defects with known answers are seeded into carrier
artifacts; reviewers of declared stance and family run against them; the output
is a catch **rate** per reviewer-stance per defect class, with false alarms
counted. The claim under test is the matrix's shape: *execution catches converge
across families; theory, hygiene and fetch catches diverge.*

## 2. Defect classes

Named after the channels the matrix revealed. A seed belongs to exactly one
class; the class is part of its sealed known-answer record.

| class | definition | example seed |
|---|---|---|
| K1 execution | wrong behavior on a run; static reading unlikely to reveal it | an iterator that consumes one record twice (the 2026-11-20 bug, re-instantiated) |
| K2 schema-drift | the artifact's field assumptions diverge from a supplied live schema | a renamed key; a stringly-typed field that became structured |
| K3 theory | the estimator cannot measure the claim, and the run still "passes" | a "pass@k" computed by majority vote; a pooled gate masking a subgroup reversal |
| K4 hygiene | silent subsample; unsourced constant; fail-open default; provenance swap | `--limit` applied without a flag; a threshold with no source line |
| K5 environment | fails only under a different OS / Python minor / arch | seed-dependent float formatting. **VOID at v1** — hard to plant deterministically; class recorded, no seeds |

K1–K4 only at v1. K5 stays VOID so that a later version has a home for it.

## 3. Carriers

- C1: the three filed tool-selection scripts (originals, at their pinned commit)
  — the artifacts the anecdotal matrix is about.
- C2: one minimal synthetic script (~80 lines, written for this purpose) — the
  control: small enough that "the file was too big to read" cannot confound.

Carriers are pinned by sha256. Seeds are patches against the pin, never edits in
place; the original artifact is untouched (house rule).

## 4. Seeding and sealing

1. A SEEDER authors each defect with a known-answer record: `{id, class, carrier,
   patch_sha256, trigger, expected_catch}` — where `expected_catch` names the
   mechanism a correct catch must identify.
2. Every known-answer record is **verified by a second family** before sealing:
   the seeder's known answer is itself a claim, and a wrong known answer corrupts
   the denominator. Unverified seeds are VOID and excluded from rates.
3. Verified records are sealed: `sealed:<sha256>`, stored apart from runners,
   same discipline as `expected/commitments.json` in the explore harness.
4. n seeds per class per carrier: proposed n = 5 (K1–K4 × C1–C2 = 40 seeds at
   v1). The number is a [CHOICE]; it sets the rate resolution (±1 seed = ±0.2).

## 5. Arms (the stance variable, made orthogonal)

| arm | allowed | measures |
|---|---|---|
| A | execute only; no reading of source | the RUN channel in isolation |
| B | read only; no execution | static reading, the auditor's first pass |
| C | read + run | the default stance |
| D | C + the live schema (for K2) | the FETCH channel; K2 is VOID in arms A–C by design |

One reviewer may run several arms; each arm is a separate run with its own record
and exposure list. Family and independence level are declared per run (the
explore-harness ladder L0–L4 applies; AUDIT-2 dedupe counts agreement once per
level).

## 6. Scoring

- **Catch**: the run's report names the defect's mechanism (per its sealed
  `expected_catch`) or reproduces its wrong behavior. Vague unease is not a catch.
- **Near-miss**: recorded separately: right region, wrong mechanism.
- **False alarm**: a reported defect that is not a seed. Counted per class. A
  reviewer who flags everything catches everything; the false-alarm column is
  what makes the catch column mean something.
- Rate per class = catches / verified seeds in class. Reported with the
  denominator, never as a bare percentage (house rule: the table carries n).
- Time-to-first-catch: recorded, optional, not a slot.

## 7. Blinding and exposure

- The runner must not see the patch, the seed ids, or the known-answer records.
  Exposure is self-declared (the explore harness's `exposure.json` discipline);
  exposure detection stays VOID as in the parent spec.
- A run whose exposure is later shown to include seed material is VOID for those
  ids — recorded, never hidden.
- The seeder cannot be a runner on their own seeds (L0, adds no authority).
- Reveal is a separate, later step; sealed records open only there.

## 8. Pre-registered prediction (PROPOSED, refutable)

P1: K1 catch rates are high and family-independent (overlap ≈ 1) in arms A, C, D.
P2: K3 and K4 catch rates diverge across families in every arm (overlap ≪ 1).
P3: K2 is caught only in arm D.
P4: DS's 2026-11-20 failure mode generalizes: a no-execution arm does not stay
silent on K1 — it over-claims. Measured as K1 false-assurance reports in arm B.

Refuted if the class × arm matrix comes out flat: then the 2026-11-20 matrix was
noise, and that is a result too. The thresholds for "high" and "≪" are **chosen,
no external source** — labeled as such here, per finding D9 of the matrix. Suggested
values, flagged chosen: high ≥ 0.8; low overlap < 0.5.

## 9. Slots (VOID until run)

| slot | value |
|---|---|
| catch_rate_by_class_and_arm (matrix, with n) | VOID |
| false_alarm_rate_by_class | VOID |
| convergence_overlap_by_channel | VOID |
| near_miss_rate_by_class | VOID |
| K5 seeds | VOID — class recorded, no seeds at v1 |
| time_to_first_catch | not a slot |

## 10. Limits (declared)

- All seeds equal weight; severity is not modeled.
- Seeding naturalness: planted defects are written by reviewers, maybe cleaner
  than wild ones. Mitigation, PROPOSED: two of the 40 seeds are re-instantiations
  of the real 2026-11-20 defects (rows 2 and 10 of the matrix), so wild and
  planted can be compared on at least two points.
- The instrument measures catch, not fix quality.
- A reviewer who knows this spec exists changes their behavior; the spec is
  public by design. Recorded as a standing confound, not hidden.

## 11. What this is not

Not a leaderboard: the family axis measures *channels*, and a low rate in a class
a stance never claims is scope, not failure. Not a certification of any reviewer
or artifact. Not code: the build (a small stdlib applier + runner-log schema,
estimated well under 200 lines) needs its own order.

## 12. Dated addition — 2026-11-21 UTC (2026-11-20 America/Chicago), chat Claude

Status: PROPOSED · docs-only · append-only; §1–§11 unchanged. Co-authored
(JinnZ2 + chat Claude). Reviewer standing for this addition: L1-with-exposure
(Claude family; read DEVIATIONS and the review texts before writing).
Observer slots: null.

Sources: A1 — label-asymmetry observation, chat session of the same date.
A2, A3 — GPT-6 review of Geometric-manifold- @ a440c12, findings D3 and D2.
A4 — the Perplexity review's self-declared family, as recorded in Kimi's
response record.

### A1 — author-label arm (amends §5, §7)

Observation, OBSERVED, n = 1 comparison, outside code review: in a two-country
comparison table, the reviewer (chat Claude) attached 6 source/abuse caveats to
one country and 1 to the other with names shown; with names hidden and the same
evidence, the count was 4 and 4. The label moved the caveat count; the evidence
did not. Transfer to code review: PROPOSED, not shown.

Gap: §7 blinds seeds, seed ids and known answers, but every C1 carrier names its
author family in its provenance header. If a reviewer's scrutiny scales with the
author label, the effect lands in the family/channel columns and is read as
channel.

Arm E: the same carrier and seeds as arm C; the provenance header per run is one of
`stripped` / `true` / `swapped:<other family>`. Record `author_label_shown`.
Measures: catch and false-alarm rate per label condition, same seeds; the
difference between conditions is the label effect. C2 (synthetic, no author) is
the label-free anchor. Minimum run set: stripped vs true; swapped optional. [CHOICE]

P5 (PROPOSED, refutable): false alarms rise under labels naming a family the
reviewer treats as less reliable; K1 catches do not move (execution does not see
the label). Refuted if the E conditions come out flat.

### A2 — a non-seed finding is not automatically a false alarm (amends §6)

C1 is pinned from real artifacts and carries wild defects. Example: matrix row 8
(majority-vote capacity probe) is present in the Kimi-lineage copy of
NOURISHMENT.md (sha256 b2d6ac8e22de4a25c95e6a3742d324eab3c17543cc0114c1fa3e59652fb98f3b),
line 57; under §6 as written, a reviewer who catches it scores a false alarm.

Every non-seed finding is adjudicated into one of:

| outcome | rule |
|---|---|
| verified_incidental | real defect, not seeded; reproduced or mechanism-checked by an adjudicator of a family other than the reporter's |
| false_alarm | adjudicated not a defect |
| unresolved | adjudication incomplete; excluded from both rates, counted beside them |
| out_of_scope | outside the carrier pin or the class list |

Pre-seed step: C1 pins are reviewed for known wild defects before seeding; those
found go into the sealed record as `pre_existing`, so catching them is
pre-classified. false_alarm_rate = false_alarm / all non-seed findings, with
unresolved reported beside it.

### A3 — trigger verification (amends §4)

§4, step 2 verifies the known answer by a second family; it does not verify that
the seed's trigger fires in the runner's environment. A seed that never fires in a
run is not a miss for that run.

Known-answer record adds `trigger_verified_env` (environments where the trigger
was observed to fire). Run record adds, per seed, `trigger_fired` ∈ {yes, no,
not_applicable, unknown}; not_applicable = no-execution arm.
Run denominator = verified seeds whose trigger fired, plus K3/K4 seeds in static
arms. Seeds with trigger_fired = no are reported in their own column, not as misses.

### A4 — declared vs verified family (amends §5)

Runner identity is declared, not verified (parent spec). The Perplexity cell
self-identified as Claude and was recorded on that declaration.
Run record adds `family_declared` and `family_verified` (VOID at v1 — no
verification method located). Dedupe and independence-level counting use
family_declared until family_verified exists, and every rate table says so.

### Slots added (VOID until run)

| slot | value |
|---|---|
| label_effect_by_class_and_condition (catch, false alarm, with n) | VOID |
| verified_incidental_count_by_carrier | VOID |
| trigger_fired_rate_by_env | VOID |
| family_verified | VOID — no method located |
