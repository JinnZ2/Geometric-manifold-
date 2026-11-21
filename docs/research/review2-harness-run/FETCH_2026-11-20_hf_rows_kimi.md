# FETCH record — HF rows API, 2026-11-20 (original observation)

Status: FILED · provenance record · no new execution in this file.
Co-authored (JinnZ2 + chat Kimi). Date: 2026-11-21 UTC (written).
Observation date: 2026-11-20 (date only; time NOT RECORDED).
Observer slots: null.

Exposure (read before writing): this work order (2026-11-21);
DEVIATIONS_2026-11-21.md (sha256 01e805a7…87f98, verified this session);
chat-Kimi project notes from the 2026-11-20 tool-selection verification pass.

## What is recorded vs not recorded

| field | value |
|---|---|
| dataset id | `od2961/illusion-of-reasoning-main-traces` (OBSERVED in notes; re-OBSERVED 2026-11-21) |
| endpoint URL exactly as called | NOT RECORDED. The rows-API convention is `https://datasets-server.huggingface.co/rows`; the exact URL string as called on 2026-11-20 was not retained. |
| query params | NOT RECORDED exactly. From the recorded result shape: `length=2`, `offset=0` (a 2-row smoke window at the head of the split). `config`/`split` values as called: NOT RECORDED. |
| fetch timestamp (UTC) | NOT RECORDED. Date 2026-11-20 only. |
| the 2 rows raw | ORIGINAL NOT RETAINED. The session that made the fetch is closed; the raw JSON is not in retained notes. Per house rule, not reconstructed. |

## Facts the fetch supported (retained in notes), mapped to DEVIATIONS rows

- Row 4 — judge labels ship in the release: pass1 rows carried
  `shift_in_reasoning_v1` plus gpt-4o rationale fields. (re-OBSERVED 2026-11-21,
  see FETCH_2026-11-21 record.)
- Row 5 — dump schema: notes record "28 pass1 keys; pass2 cue fields; G=8;
  temps; no base arm". The 2-row window directly supported the pass1 key set
  and pass2 cue fields. G=8, temperature range, and the absence of a base arm
  are dataset-wide claims a 2-row window cannot establish; they came from the
  card and schema preview, not the rows — recorded here as scope-limited.
- Row 6 — smoke-row anomaly: a row with judged shift, empty
  `_shift_prefilter_markers`, and shift excerpts not locatable in the row's own
  `output` — label provenance not reconstructible from the row. (Mechanism
  sharpened by the 2026-11-21 re-fetch; see that record.)
- Row 7 — KM's Phase-1 note "labels absent" was wrong; the card preview
  truncates the field list. Supported by the same fetch: fields present via the
  rows API that the card preview does not show.

## Why this file exists

DEVIATIONS §5 cited the fetch with no artifact behind it. This file is the
artifact: it records what is knowable now and marks the rest NOT RECORDED /
ORIGINAL NOT RETAINED. A re-fetch executed 2026-11-21 is filed separately as
FETCH_2026-11-21_hf_rows_kimi.md; it is a new observation, not this one.
