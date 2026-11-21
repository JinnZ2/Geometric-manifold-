# RESPONSE RECORD — Perplexity-Claude (PC) review of 2026-11-20 work

Status: **REGENERATED — NOT the byte-identical original.**
The original text was finalized in the chat-Kimi session of 2026-11-21 and lost
to a sandbox outage before filing; the owner holds a relayed chat copy. THIS
file is rebuilt in a later session from retained project notes. Verdicts and
numbers below are re-reported from those notes and marked; wording differs from
the original. The owner's relayed copy is canonical. Differences the owner
should expect: section order and phrasing; the verdict CONTENT is as noted.

Co-authored (JinnZ2 + chat Kimi). Date: 2026-11-21 UTC (regenerated).
Observer slots: null.

Exposure (read before writing): this work order; chat-Kimi project notes of
2026-11-21 (PC review round); residual_tower.py re-checked at pin this session
(sha256 94fcd1e8…ce92d9, selftest exit 0). The PC review text itself was NOT
re-read this session — verdicts below are from retained notes, and the response
should be re-pointed at the PC text before anyone quotes it.

## Verdicts (re-reported from session notes; tags as originally recorded)

1. PC headline "Test 1 pass is guaranteed in exact integer arithmetic" —
   ACCEPT. The guarantee holds only in exact arithmetic; the float arm is the
   counterexample. Reproduced in-session: 15–30% mismatch across 4
   parameterizations; PC's 73/200 sits inside that family; k=1 mismatches MORE
   than k=100. (re-reported; float-arm numbers NOT re-run this session.)
2. Gray rate — VERIFIED exactly in-session: lambda 7.8125, P(X≤4)=0.111;
   aliasing only for b≡1,2 mod 4 flipping bit1; exhaustive +1 steps; 20 fresh
   streams, carry-ripple lengths 3–14, median 7.5. (re-reported; NOT re-run
   this session. Instrument state re-VERIFIED 2026-11-21: tower at pin,
   selftest exit 0.)
3. Carry-ripple bound — derivable; a 2-line proof was supplied. (Proof text
   NOT RETAINED — regenerate on demand; the bound itself is in the verdict.)
4. Unsalted commitments — ACCEPT: load-bearing for the VOID held-answer
   probes; optional for the current deterministic probe set.
5. PC's D1 endorsement ("labels absent") — OVERTAKEN by the judge-labels
   fetch. NOW UPGRADED: re-OBSERVED 2026-11-21 — the rows API shows
   `shift_in_reasoning_v1` + gpt-4o rationale fields on pass1 rows
   (see FETCH_2026-11-21_hf_rows_kimi.md, row 0/1 raw).
6. Meta: PC is the 4th matrix cell; all its catches sat in Claude-family home
   channels (THEORY/HYGIENE); divergence without independence — this sharpens
   planted-defect P2 to within-family. (DERIVED, as originally recorded.)

## Six actionable amendments

NOT RETAINED in detail. The notes record only: six docs-level amendments were
listed, owner's to order. Their text is not in retained notes and is NOT
reconstructed here. Recover from the owner's relayed chat copy, or order a
re-derivation as its own item.

## Standing caveats for this regeneration

- Verdict direction and the numbers in items 1–2 are as noted; any quotation
  needs the owner's relayed copy or a re-run (order it as a work item; the
  float-arm and Gray checks are both cheap against the pinned tower).
- This regeneration itself is a provenance event: same family (Kimi), new
  session, exposure = notes only. Under the planted-defect spec's own logic it
  counts as a re-statement, not an independent confirmation of anything.
