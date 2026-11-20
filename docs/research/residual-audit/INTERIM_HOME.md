# Interim home: residual-audit

Interim home. `JinnZ2/residual-audit` does not exist (repository list read 2026-11-19); these
files stay here until it does, then move there and this folder becomes a pointer.

Filed 2026-11-19 as received, byte-identical, from the zip
`OKComputer_Exploring_Deep_Research_Tools.zip`,
sha256 `f531061fc49e57acda6d3fb01144e98c08a0a5043d81a40f76ef69ac7d52487b`.
The zip itself is not in this repository. It held no `__pycache__`.

Author: Kimi, for `residual-audit` and for the tooling report (owner-stated 2026-11-19; the
files themselves name a Kimi build for `residual_audit.py` and EXP1, operator = Q = kimi-k3).
Source: the OKComputer zip named above.
Status: PROPOSED. EXP1 is a null run by its own statement. The tooling report's citations other
than Madl (arXiv 2606.22019) are UNVERIFIED.

One rename: the report arrived as
`tooling-report/Tooling the Residual Check — Survey and Build Plan.md` (the dash printed as
`#U2014` on extraction) and is filed as
`tooling-report/Tooling_the_Residual_Check_Survey_and_Build_Plan.md`. Content unchanged.

Zip path to filed path: `residual-audit/<x>` is `<x>` here; `tooling-report/<x>` is `tooling-report/<x>`.

sha256 of each filed file:

```text
cb56e29609fd64e64dbe5b7601f5d57a5c84a43dabb918014c3cf43c9604c525  ./README.md
66cc67ac4be537d82e4a7a4327ece5a00f9d798bc76190fbecd4e15dc8fd51a2  ./examples/owl_claim.json
53ccbfed03b40054d10b4e385dd3dd1c5f736e072f2dc8947bf04ff68c1dd4be  ./residual_audit.py
ce065e36d75a7bf167bbaa6dc2a67c1d07d56755dda1c56ebae4ea0a96674918  ./runs/exp1/EXP1.md
20223e8c517b4ff0bc220a53d407fb9eec5dbca705c785c5485235c938bb5da5  ./runs/exp1/archive/records.jsonl
72055bf2ac698c6a7924096f79c1dd34f30c92c1246634e721361e0dad896c86  ./tooling-report/Tooling_the_Residual_Check_Survey_and_Build_Plan.md
```

Check: `python3 docs/research/residual-audit/residual_audit.py selftest`

## Merge record, 2026-11-19

PR #23 was merged with a merge commit, `601a48c`. No CI configured for PRs; checks run:
`residual_audit.py` selftest 0 failures, `subliminal_guard.py` demo exit 0, sha256 byte-identical.

- No CI: the only workflow, `.github/workflows/hypothesis-engine.yml`, does not run on pull requests.
- The three checks were run again on main at `601a48c` after the merge, with the same results:
  selftest `0 failure(s)`, exit 0; `python3 docs/research/subliminal_guard.py` exit 0;
  `subliminal_guard.py` sha256 `d5f92b6d…` equal to the upload, and the six per-file hashes above match.
- Post-merge CI: NOT READ (none configured).
- Not run: `pytest` (torch is absent where the checks ran), `ruff`.

## Notes, 2026-11-20

1. Figures. The tooling report refers to 3 figures: `figs/fig1_priority.png` (build effort vs.
   expected payoff, report line 173), `figs/fig2_interlock.png` (four-repository interlock,
   line 169), `figs/fig3_heatmap.png` (research candidates x repositories, line 196). The zip
   (sha256 `f531061f…487b`) lists 6 members and none is under `figs/`. The three figures are
   UNRESOLVABLE until Kimi supplies them. The same zip was uploaded again 2026-11-20 with the
   same sha256 and the same 6 members.
2. Citation correction. The tooling report writes "Madl et al." (lines 37, 142, 219). The
   paper has a single author: Madl, arXiv 2606.22019 (abstract read 2026-11-19; see
   `RESEARCH_IN_PROGRESS_subliminal_learning.md` §1.1). Read "Madl et al." in the report as
   "Madl". The report file is kept byte-identical; the correction is this note.
3. Proposed build sprint. The report's "immediate sprint" (line 213) names `ra.manifest`,
   `fa.conformal` and `ra.laddercal`. Status: PROPOSED, held under the 2026-11-11 feature
   freeze (owner-stated in the order of 2026-11-20). Nothing from the sprint is built here.
4. Scope law (owner-stated in the order of 2026-11-20), to carry on every residual record:
   - the carrier regime assumed;
   - the channels the record is blind to. The body channel is always listed.
   The 12 filed EXP1 records predate this rule and are not edited.

## Replacement, 2026-11-20: zip `f56a94c5…5a97`

The filed copy now follows a second zip, `OKComputer_Exploring_Deep_Research_Tools.zip`,
sha256 `f56a94c50de0872e61652995704302fd1b3e1b892eb35b76eed7d74a6dc35a97` (owner-supplied,
Kimi; uploaded to the filing session 2026-11-20). The prior version is the zip
`f531061fc49e57acda6d3fb01144e98c08a0a5043d81a40f76ef69ac7d52487b` recorded above; its
per-file hashes above are kept as the prior version's. Neither zip is in this repository.
The new zip held no `__pycache__`. Nothing was stripped.

New zip, 10 members, against the prior 6:

- Same bytes (5): `README.md`, `examples/owl_claim.json`, `residual_audit.py`,
  `runs/exp1/EXP1.md`, `runs/exp1/archive/records.jsonl`.
- Changed (1): the tooling report. Same rename as before (`#U2014` dash name to
  `Tooling_the_Residual_Check_Survey_and_Build_Plan.md`). Diff against the prior filed copy:
  - "Madl et al." occurred 11 times before and occurs once now, inside the erratum line.
    10 were replaced.
  - 5 wording changes that follow from the single author, on lines 13, 37, 99 and 136:
    "document" to "documents", "Their closing warning" to "The paper's closing warning",
    "show" to "shows", "they establish" to "it establishes", "as Madl do" to "as Madl does".
  - 2 lines added at the end: a blank line and the erratum line dated 2026-11-20.
  - No other difference.
- New (4): `CHANNEL_SCOPE.md`, `tooling-report/figs/fig1_priority.png`,
  `tooling-report/figs/fig2_interlock.png`, `tooling-report/figs/fig3_heatmap.png`.

sha256 of each filed file after the replacement (each equals the zip member):

```text
e96f49531a2dd5a3defd1113a5b677133a04cf32bfb5cbab15f1657cd40dd2e0  ./CHANNEL_SCOPE.md
cb56e29609fd64e64dbe5b7601f5d57a5c84a43dabb918014c3cf43c9604c525  ./README.md
66cc67ac4be537d82e4a7a4327ece5a00f9d798bc76190fbecd4e15dc8fd51a2  ./examples/owl_claim.json
53ccbfed03b40054d10b4e385dd3dd1c5f736e072f2dc8947bf04ff68c1dd4be  ./residual_audit.py
ce065e36d75a7bf167bbaa6dc2a67c1d07d56755dda1c56ebae4ea0a96674918  ./runs/exp1/EXP1.md
20223e8c517b4ff0bc220a53d407fb9eec5dbca705c785c5485235c938bb5da5  ./runs/exp1/archive/records.jsonl
a99b85050b1ff39af6a1d75f25a5d95dfbf9cd6b95851d6c3b7c1c666b04a5f9  ./tooling-report/Tooling_the_Residual_Check_Survey_and_Build_Plan.md
3bd1cc9de1b5d08e6d8fe3f9a243bb4de5dc85cadef742d2d4bf5f69fc10bb5c  ./tooling-report/figs/fig1_priority.png
eb5b571c624e3b0fad0a0601b4c43140d479caa098955c23f9f78a345203445e  ./tooling-report/figs/fig2_interlock.png
ce493470395acc64ea89cbec42e2139a5a8f7da6de90eb2d9d2a25ff9ac85311  ./tooling-report/figs/fig3_heatmap.png
```

Notes, 2026-11-20 (second set):

1. Figures resolved. The three figures the report links are now filed under
   `tooling-report/figs/`. This supersedes note 1 of "Notes, 2026-11-20" above (the
   UNRESOLVABLE mark); that note is kept as written.
2. Figure 1 (`fig1_priority.png`): the effort and payoff positions are Kimi's estimates.
   Status PROPOSED. They are judgments, not measurements.
3. The figures label a repository "residual-audit". No such repository exists (see the top of
   this file); read that label as this interim home until one does.
4. `CHANNEL_SCOPE.md`: status PROPOSED, held under the 2026-11-11 feature freeze
   (owner-stated; the file states the same in its own Status line). No code here was changed
   for it. The 12 EXP1 records are not edited; the file's own backfill policy is append-only.
5. Note 2 of "Notes, 2026-11-20" above (read "Madl et al." as "Madl") described the prior
   report. The filed report now carries the correction itself.
6. Dates: the report's erratum is dated 2026-11-20. `runs/exp1/EXP1.md` is still headed
   2026-11-20 while its 12 records are stamped `2026-11-19T1946Z`; that flag stands.


## Addition, 2026-11-20: `residual-tower/` from zip `79085d17…190d`

Owner-supplied 2026-11-20 (Kimi), zip sha256
`79085d170c176d33000eada8e16fde9d8e1421048c7c07c0757df3c73630190d`, 13 members.

- New: `residual-tower/` (3 files), filed byte-identical at `docs/research/residual-tower/`:

  | file | sha256 |
  |---|---|
  | `README.md` | `e1535d223bc7a2047ab68cd5372c0c8cd77b9ae997daab25764060487d4218a8` |
  | `demo_record.jsonl` | `42fe38804fa352d2217439c59fd0bf96983f2c555d66a2331e5ae11809ca6f50` |
  | `residual_tower.py` | `94fcd1e8679a6f65375f4034b00cd8a88f93c0bd14a39128eb47f6e85ece92d9` |

- The other 10 members (6 under `residual-audit/`, the report, 3 figures) are byte-identical to
  the copies filed from zip `f56a94c5…5a97`. Nothing new was filed for them.
- Status: PROPOSED, held under the 2026-11-11 feature freeze (owner-stated). Not wired into any
  repository. Its own README says "Not yet reviewed by a second reader"; that line is kept as
  delivered and predates the audit below.
- Second-reader audit (chat Claude, 2026-11-20, relayed by the owner): selftest 22/22; `check`
  0 diffs; the 355/113 trace and the edge band reproduced; division by a zero-valued tower is
  VOID.
- Open (same audit): the tie-breaking rule in `encode()` is unstated. A value exactly halfway
  between two grid points goes to the lower one: 1/4 on the level-0 grid (step 1/2) takes the
  code whose value is 0, not 1/2.
- Rerun from the filed copy, 2026-11-20 (`python3 -I`, files re-hashed after: unchanged):
  `selftest` n 22, fails 0; `check demo_record.jsonl` diffs 0, ok true; `encode_tower(355/113)`
  codes [14, 13, 0, ESCAPE], residual 7/7232, value exact; `encode_tower(1/4)` codes
  [8, ESCAPE, ESCAPE, ESCAPE], residual 1/4; `div` by a zero tower raises `TowerVoid`.
  `demo` was not run (it writes a record file).
