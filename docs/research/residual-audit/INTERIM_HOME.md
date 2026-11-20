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
