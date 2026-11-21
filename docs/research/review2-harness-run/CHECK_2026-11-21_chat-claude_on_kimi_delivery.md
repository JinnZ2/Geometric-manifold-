# CHECK — chat Claude on chat Kimi's delivery of 2026-11-21

Status: check record · docs-only. Co-authored (JinnZ2 + chat Claude).
Date: 2026-11-21 UTC (2026-11-20 America/Chicago).
Standing: Claude family (family_declared: Claude; family_verified: VOID),
L1-with-exposure. Observer slots: null.

Exposure (read before writing): all five files of the delivery; the owner's
relayed chat copy of Kimi's original PC response (filed beside this as
REVIEW_RESPONSE_2026-11-21_perplexity_kimi.md); DEVIATIONS_2026-11-21.md;
PLANTED_DEFECT_SPEC.md incl. §12 (which this reviewer wrote).

## 1. Re-fetch record (FETCH_2026-11-21_hf_rows_kimi.md)

Method: the two raw rows were parsed from the record as filed (no new HTTP
call). Row 1's `problem` field is abbreviated in the record, so any search of
row 1's `problem` is VOID.

| claim in the record | check | result |
|---|---|---|
| 30 pass1 keys (row 0) | JSON parse | CONFIRMED — 30 in row 0 and in row 1; pass2 has 21 in both (OBSERVED) |
| row 0 shift excerpts = the prompt's few-shot exemplar, verbatim; absent from the model's `output` | substring search | CONFIRMED — `shift_before_excerpt` and the marker "Wait, hang on" occur in `problem`, not in `output` or its think block (OBSERVED) |
| row 0 `shift_first_marker_char` = 47 is "consistent with an offset into the EXEMPLAR, not the output" | offsets recomputed | NOT SUPPORTED — see below |

Offset detail (OBSERVED): row 1, the well-formed row, sets the convention —
`shift_first_marker_char` = 33 = `_shift_prefilter_pos`, and the marker sits at
index 33 of the think-block content (40 in the full output, after `<think>`).
Under that convention the exemplar gives 32 for row 0, not 47. Other anchors
tried: from the exemplar's `<think>` line 40; from its `• Board` line 81; from
`EXAMPLE` 105. None gives 47. The value indexes a string not present in either
field as filed.

Read (DERIVED): the exemplar-excerpt match stands and is the specimen. The
offset is a second unexplained value in the same row; it favors neither of the
record's candidate mechanisms (a) judge saw the full prompt, (b) excerpts from
a different generation. Unresolved; recorded, not reconciled.

Duplicate note: the owner received two byte-identical copies of this record;
one is filed.

## 2. PC response: regeneration vs the owner's relayed original

Canonical = the relayed original (REVIEW_RESPONSE_2026-11-21_perplexity_kimi.md).
The regeneration is filed beside it as _REGEN, as received. Differences in
content, not wording:

| regeneration | original | type |
|---|---|---|
| item 2: "20 fresh streams, carry-ripple lengths 3–14, median 7.5" | item 4 (Gray rate): "20 fresh streams: 3–14, median 7.5" — per-stream counts in the Gray aliasing check | number re-attached to a different measurand |
| item 5: "PC's D1 endorsement ('labels absent')" | item 8: PC endorsed D1's "neither confirm nor refute"; "labels absent" was KM's own Phase-1 note (DEVIATIONS row 7) | claim attributed to the wrong reviewer |
| item 3: proof text NOT RETAINED | item 3: proof present ("Bits above h equal ⇒ \|diff\| < 2^(h+1) ⇒ diff_level ≤ h") | recoverable — recovered by filing the original |
| six amendments NOT RETAINED | item 10 lists all six | recoverable — recovered by filing the original |
| absent | item 1 (reruns: three Python minors reproduce b88473c4…), item 6 (scan-cost), item 7 (parity counts), item 11 (not claimed), step −1 Gray DERIVED | omitted |

Read (DERIVED): the regeneration was labeled non-canonical and notes-based,
and its NOT RETAINED marks were honest. Two content drifts entered anyway,
both of the shape "real number or claim, wrong slot". Candidate specimen for
PLANTED_DEFECT_SPEC: a K4 provenance swap produced by regeneration from notes
rather than by authoring. n = 1.

## 3. Instrument hash

residual_tower.py sha256
94fcd1e8679a6f65375f4034b00cd8a88f93c0bd14a39128eb47f6e85ece92d9 — matches the
regeneration's "94fcd1e8…ce92d9" (OBSERVED, owner's zip copy).

## 4. Not checked

- REVIEW_RESPONSE_2026-11-21_gpt6_kimi.md runs — torch not available here; read only.
- The HTTP call itself — not re-fetched; section 1 checks the filed text.
- REVIEW_2026-11-21_kimi_on_spec12.md — read; its sharpenings of §12 are
  addressed, if ordered, in a separate dated addition to the spec.
