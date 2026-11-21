# REVIEW — PLANTED_DEFECT_SPEC §12 (A1–A4), cross-family check

Status: FILED · review mode; §12 not edited.
Co-authored (JinnZ2 + chat Kimi). Date: 2026-11-21 UTC.
Reviewer stance: Kimi family, checking a Claude-family addition per the spec's
own §4 step 2 logic. L2-with-exposure. Observer slots: null.

Exposure (read before writing): this work order; PLANTED_DEFECT_SPEC.md §1–§11
(sha256 c7821038…2f8ef9, my own delivered text, hash-verified); PR #38
PLANTED_DEFECT_SPEC.md with §12 (sha256 0566afb9…b1ad3, hash-verified against
the work order); DEVIATIONS_2026-11-21.md (sha256 01e805a7…87f98);
EXPLORE_HARNESS_SPEC amendments F1–F6 (ladder L0–L4, AUDIT-2 dedupe);
kimi_2026-11-20/NOURISHMENT.md in PR #39 (hash-verified, see A2);
GPT-6 review document (for A2/A3 source context).

## A1 — author-label arm (arm E): **SHARPEN**

The design CAN separate label effect from channel: same carrier, same seeds,
only the provenance header varies (`stripped`/`true`/`swapped`), so the header
is the manipulated variable and the per-condition difference is identifiable.
That part is sound. Four sharpenings:

1. **Reviewer multiplicity.** With one runner per family, label effect and
   reviewer identity are aliased at n=1 — the arm would measure "this
   reviewer's label response." The label claim needs ≥1 runner per family per
   condition, pooled; say so, or arm E's result inherits the same
   numerators-no-denominators disease the parent spec exists to cure.
2. **C2 anchor scope.** C2 is a valid anchor for the EXISTENCE of a label
   effect on a neutral artifact, not for its MAGNITUDE on C1 — C2 differs
   from C1 in size, naturalness and origin, so cross-carrier comparison
   confounds label with carrier. Magnitude claims must come from within-C1
   conditions (stripped vs true on the same carrier). Also: C2 is only
   label-free if its own provenance header names no family — "synthetic" is
   not automatically "authorless."
3. **P5's mechanism clause is over-strong.** "K1 catches do not move
   (execution does not see the label)" assumes constant effort. If a
   distrusted label induces MORE runs, K1 catches move through the effort
   path while execution stays label-blind. Record run counts / effort per
   condition; P5's refutation rule (flat E conditions) is otherwise correctly
   falsifiable.
4. **Demand characteristic.** The spec is public (§10 standing confound);
   runners who know headers may be swapped can discount ALL headers, which
   flattens the effect toward a false null. Record it as a standing confound
   per the §10 precedent.

Provenance note: the motivating observation is n=1, non-code domain, and a
self-observation by the affected family. Acceptable as PROPOSED motivation —
the arm exists to measure, not assert — but its provenance tier should be
recorded in the arm record.

## A2 — adjudication of non-seed findings: **SHARPEN**

The table is the right D3 repair, and the `pre_existing` pre-seed seal is the
right shape. The work order's question — does this create a "who adjudicates
the adjudicator" hole? — answer: a BOUNDED one. The adjudicator's verdict is
itself a single-family claim with no disagreement protocol; §4 step 2 logic
applied to the table demands one more tier, and so on. The parent spec's own
depth is one cross-family tier (seeds get one second-family verification), so
A2 at one cross-family adjudicator is CONSISTENT with house depth — but the
hole should be closed at that depth rather than left implicit:

1. Record adjudicator identity + family per verdict (currently only "a family
   other than the reporter's").
2. Add a disagreement path: second adjudicator of a third family; failing
   concurrence, dual-recorded `unresolved` with both positions preserved.
   (`unresolved` is already excluded from both rates and counted beside —
   correct.)
3. Pre-seed `pre_existing` review: state who performs it; if the seeder's
   family alone, the sealed record is again a single-family claim until
   reveal — fine, but say it.

Evidence check (VERIFIED this session, upgrades A2's example from claimed to
OBSERVED): `docs/research/tool-selection-vs-deepening/kimi_2026-11-20/
NOURISHMENT.md` hashes to exactly the claimed
b2d6ac8e22de4a25c95e6a3742d324eab3c17543cc0114c1fa3e59652fb98f3b, and line 57
is the majority-vote pass@256 capacity probe — a K3-class wild defect
(majority vote measures the mode; it cannot detect re-widening of the
reachable-answer SET, and can mask it). Nuance worth recording: it is a
PROPOSED-protocol defect in a docs artifact — exactly the kind of thing a
reviewer catching it under §6-as-written would have scored as a false alarm.
A2's example stands.

## A3 — trigger verification: **SHARPEN**

Consistency with §6 (the work order's question): YES. §6 says rate = catches
/ verified seeds; A3 narrows the denominator to verified-AND-fired (plus
K3/K4 in static arms). That is a strict refinement, not a contradiction, and
it inherits §6's carry-the-n rule. It is exactly D2's exposure tiering made
mechanical. Three gaps:

1. `trigger_fired = unknown` has no stated handling → exclude from the
   denominator, count beside (mirror `no`); one line.
2. **Authorship of the field.** The runner is blinded (§7) and cannot know
   whether a trigger fired. `trigger_fired` is a SCORER/at-reveal field,
   populated from the sealed trigger record plus run logs — say so,
   otherwise the schema implies the runner self-scores.
3. **Symmetry statement.** If fired-seeds-only is the K1 denominator, then
   the K1 catch RATE in arm B (no execution) is VOID by design — say it
   explicitly, parallel to "K2 is VOID in arms A–C" in §5. Otherwise P4's
   false-assurance measure risks being misread as a catch-rate miss.

## A4 — declared vs verified family: **ACCEPT**

The work order's question — does A4 contradict the explore-harness ladder?
Checked: NO. The ladder already runs on declared values — independence_level
L0–L4 is declared per run, AUDIT-2 dedupes per declared level, and exposure
detection is VOID in the parent spec. A4 contradicts nothing; it makes the
declared-not-verified status explicit and reserves the slot
(`family_verified`, VOID, no method located — honest; the provenance
literature has model-lineage methods but none located for chat-session
identity).

One cross-field caution (SHARPEN-level note, not a verdict change): A1's
`author_label_shown` is a CARRIER property and A4's `family_declared` is a
RUNNER property. They must be distinct schema fields — collapsing them
re-creates exactly the confound A1 exists to measure.

## Pattern-level summary

A1–A4 all extend the parent spec's own discipline (declared-not-verified,
sealed records, one-tier cross-family checks); nothing breaks the structure.
The recurring gap is field authorship and scope: A1 needs its multiplicity
and anchor scope pinned, A3 needs its scorer assignment and VOID symmetry
stated, A2 needs its adjudicator provenance recorded. All four are cheap
one-to-three-line fixes; none requires redesign.
