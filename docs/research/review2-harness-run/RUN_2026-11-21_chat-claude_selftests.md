# RUN — chat Claude self-test datapoint, 2026-11-21 UTC (2026-11-20 America/Chicago)

Status: OBSERVED run record · docs-only. Co-authored (JinnZ2 + chat Claude).
Observer slots: null.

Runner: chat Claude, chat sandbox. family_declared: Claude. family_verified: VOID.
Standing: L1-with-exposure — read DEVIATIONS_2026-11-21.md, PLANTED_DEFECT_SPEC.md
and the Perplexity / Kimi / GPT-6 review texts before running. Not REVIEW-2.
Environment: Python 3.13.18, Linux; each script run with `python3 -I` from an
empty working directory; inputs = the owner's zip members (hashes in commit 1 of
this PR, plus residual_audit.py 53ccbfed… and residual_tower.py 94fcd1e8…).

| script | command | exit | result | repeat |
|---|---|---|---|---|
| kimi_2026-11-20/passk.py | --self-test | 0 | PASS | output identical on 2 runs |
| kimi_2026-11-20/step1_recheck.py | --self-test | 0 | PASS | output identical on 2 runs |
| kimi_2026-11-20/step1b_passk_from_dump.py | --self-test | 0 | PASS | output identical on 2 runs |
| residual-audit/residual_audit.py | selftest | 0 | 0 failure(s) | not repeated |
| residual-tower/residual_tower.py | selftest | 0 | fails: 0 | not repeated |

What this adds: the Kimi-lineage scripts now have execution by a second declared
family (Kimi's note: previously Kimi-only).
What it does not add: no real data was run; nothing here bears on the paper's
numbers. Self-tests check the code against its own planted cases only.
Matrix cells re-checked on these copies (static, OBSERVED): row 8 present
(NOURISHMENT.md:57); row 11 unlabeled (EXPERIMENT.md:62); row 10 — `--limit`
present in step1_recheck.py and step1b_passk_from_dump.py (absent from passk.py),
no records_seen/limit_applied fields in any of the three; new — field detection uses a 0.8
row fraction (step1_recheck.py:207, 230), chosen, unlabeled (row 11 class).
