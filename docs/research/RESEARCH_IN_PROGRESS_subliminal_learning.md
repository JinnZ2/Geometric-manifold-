# RESEARCH IN PROGRESS: subliminal learning

co-authored (Kavik + Claude chat + external model contributions as marked), CC0.
Opened 2026-11-19. Status: IN PROGRESS. Nothing here is a finding unless tagged OBSERVED.
Mode of this file: SUGGEST. Next stages: AUDIT (different family, L2+) → IMPLEMENT runs.

Tags: OBSERVED (source or run read) · DERIVED (follows from OBSERVED) · PROPOSED (design) ·
UNVERIFIED (cited by an external model, not yet read here). More asterisks = thinner.

## 1. Field state (OBSERVED: read 2026-11-19)

- Cloud, Le, Chua et al., "Language models transmit behavioural traits through hidden signals
  in data", Nature 652, 615–621 (2026), doi 10.1038/s41586-026-10319-8. Traits pass through
  semantically unrelated teacher data (number lists, code, CoT), survive filtering, and fail
  across different base models. The paper proves a theorem for all neural nets under conditions
  and demonstrates it in an MLP.
- Schrodi, Kempf, Barez, Brox, "Towards Understanding Subliminal Learning", ICLR 2026
  (arXiv 2509.23886v2): divergence tokens carry the bias; early layers critical; fine-tuning one
  early layer suffices; fragile: meaning-preserving paraphrase, even by the biased teacher, or
  mixing teachers usually suppresses it; still occurs under greedy decoding.
- Nief, Fu, Muchane, Holtzman, "Subliminal Learning is a LoRA Artifact" (arXiv 2606.00831,
  2026-05-30): more common in open-weight models than previously shown; fragile; sensitive to
  context and LoRA rank.
- Concurrent mechanism claim: token entanglement + statistical logit leakage (Zur et al. 2025);
  cited inside Schrodi et al., not read directly.

### 1.1 Added 2026-11-19 (OBSERVED: abstract read; arXiv abstract page fetched 2026-11-19 by Claude Code, full text not read)

- Madl, "Channel Location Constrains the Auditability of Subliminal Learning",
  arXiv 2606.22019 (v1, 2026-06-20). Three regimes by channel location:
  - a controlled initialization-dependent body channel, where a pre-training screen (coverage)
    predicts held-out transfer;
  - in pretrained language models, masked single-token traits ride vocabulary geometry; this
    channel is initialization-independent, so initialization-alignment screens are not mechanistic
    there;
  - conditional behaviours routed through the network body: sycophancy transfer reaches about
    0.63 of the teacher's effect and evades four audits across two model families.
  - an audit used outside its carrier regime can give false assurance.

## 2. Instruments built (OBSERVED: runs on toy data only)

`subliminal_guard.py` (stdlib):
- Layer 1 carrier scan: d(t) = log p_teacher − log p_base per token. A planted push hid under
  normal fine-tune drift without a reference; against a clean reference teacher it flagged
  exactly the planted tokens.
- Layer 1b context-keyed scan (bigram, position, any key): Welch z vs reference, Bonferroni,
  per-context renormalization offset. Caught a trait arranged to cancel at the unigram grain;
  unigram z = +1.3, bigram z = ±37.
- Layer 5 self-audit kernel: invariance under paraphrase / reason chain (source vs relay) /
  cross-lineage agreement.
- Known limits: needs a known-clean same-base reference teacher; thresholds PROPOSED;
  long-range (sequence-level) encodings are not covered.

### 2.1 Limits added 2026-11-19 (DERIVED from §1.1)

- Layers 1 / 1b and the residual harness (§4.1) are output-side: they read token log-probs or
  output text. They are blind to the body channel by construction.
- The "lineage is the bottleneck" reading holds only for the initialization-dependent channel.
- Wording note 2026-11-19 (accepted): "body" appears in two regimes of §1.1. One is the
  controlled initialization-dependent body channel, where the pre-training screen works. The
  other is the sycophancy body routing, which evaded the audits. The first bullet holds for
  both: the output-side layers are blind to both.

## 3. External extensions received 2026-11-19 (author model: Kimi (owner-stated 2026-11-19))

Author slot filled 2026-11-19: it read "[void, unlabeled in paste]" when filed; the owner stated
the author is Kimi. "The external model" in §3, §3.1 and §4 means Kimi.

PROPOSED by the external model, filed as received in summary:
- multi-scale invariance → distance-to-break instead of a boolean
- reason-chain depth + closure on a conserved quantity / measurement / record
- lineage topology (clusters of models that move together), not one averaged scalar
- observer-slot discipline: human_interest / ai_interest / divergence_note stay null
- residual check: destroy the surface, keep the latent; residual = max(0, P − RC)
  - S = surface agreement, P = post-destruction agreement, RC = reason-chain coverage
  - destruction ladder: lossy paraphrase cascade → tokenizer/vocabulary cut →
    modality re-encoding → adversarial surface noise → basin/manifold displacement →
    cross-lineage after ablation
  - rules: destruction must not be reversible by string recovery; R is never auto-filled
    (empty → RC = 0); residual stays graded; residual measures channel, not truth
- tie-in to this repo: is the claim stable only inside the current safe basin; does it flip
  toward the trust-region boundary

Note 2026-11-19 (Claude Code, code read at `d67b12e`): term collision.
- "trust-region boundary" in the external text above = the edge of the safe basin.
- "trust region" in this repo = a clamp on the norm of one update step
  (`manifolds/parameter_manifold.py:78`), not a region of parameter space.
- Nearest basin handle here: `BasinDivergenceMonitor.kl_from_reference` / `.in_basin`
  (`addon_thermodynamic_control/energy.py:110`, `:117`).
- The tie-in needs a claim→theta adapter; none exists in this repo.

Note 2026-11-20 (Claude Code): main's `repair_step` ascended on KL until the objective-sign
PR (branch `claude/objective-sign-fix`, cherry-pick of `59031bc`); any basin-handle result
before the merge is void.
- The PR is #26, merged at `a9d5b5c` (2026-11-19, 22:17 -06:00). "The merge" above means that commit.
- Diagnosis: `edd8ef9` (branch `funnel-probe`): `delta = -lr*grad(task - lambda*KL)`,
  cos(delta, grad KL) = +0.993.
- Verification on that branch: `ff63359`: KL 3.79 → 0.17, cos(delta, grad KL) = −0.995.
- Rerun in the filing session, 2026-11-20 (torch 2.15.0, default config, 100 steps):
  - main `55a2d1a`: KL 3.7906 → 39.3746, KL rises on 100 of 100 steps, cos +0.991 at step 1
  - this branch: KL 3.7906 → 0.1734, KL rises on 0 of 100 steps, cos −0.994 at step 1

### 3.1 Audit notes on the residual check (DERIVED, chat Claude, L0 for its own Layer 5)

- A1: P and RC sit on different scales. P is the agreement of a re-ask; RC is the semantic
  coverage of a chain. Subtracting them assumes a shared unit. → report (S, P, RC) as a triple;
  "residual" is a derived view, not the primary datum.
- A2: the agreement function A() is itself a judge. A model-scored A from the teacher or
  student lineage can manufacture residual (same point as candidate 7 below). → A's
  lineage is a required declared slot; run A at ≥2 lineages.
- A3: RC computed by the audited model is introspection on the same weights. → RC from a
  different family, or from a human, or recorded as L0.
- A4: paraphrase is known not to remove transmission in some settings and to suppress it in
  others (Schrodi: suppresses; the external model cites persistence through faithful and
  contradictory paraphrase**, UNVERIFIED) → the ladder's first rung is contested, so record
  results per rung, not only at the end.
- A5: the ladder × carrier matrix (candidate 1) is the place the Layer 1 / 1b carrier scans plug in:
  a rung "kills" a carrier when the carrier scan's excess z drops at that rung.

## 4. Research candidates (from the external model = Kimi, owner-stated 2026-11-19; citations UNVERIFIED)

| # | candidate | external model's rank note | to verify |
|---|---|---|---|
| 1 | destruction ladder × proposed carrier matrix (which rung kills which carrier) | most original | — |
| 2 | continuous "kinship distance" (init distance, data overlap, tokenizer) predicting transmission | clearest citable path | MNIST partial recovery via behavioral cloning of initial logits** |
| 3 | calibrate residual: e-values / finite-sample bounds, FPR budget | most original | "Audit the Whisper" collusion-auditing line** |
| 4 | receiver-side probe: is the trait linearly readable before behavior shows it | — | steganographic payloads linearly separable, localized circuits** |
| 5 | which ladder rung, used as preprocessing, blocks transmission at lowest capability cost | — | KL-regularized fine-tuning; non-monotonic acquisition** |
| 6 | re-audit published emergent-misalignment datasets for hidden-signal share | cheapest | Cloud et al. flag EM may partly be subliminal (check in Nature text) |
| 7 | judge artifact: same-lineage judges manufacture apparent transmission | — | — |
| 8 | supply-chain scale: does transmission compound / attenuate / saturate across generations of synthetic data | — | "filtering can't remove non-semantic signals" (Cloud: consistent) |

Also UNVERIFIED: a "penguin" exception where divergence tokens fail to explain transmission**.

Authorship of candidates 1–8: Kimi (owner-stated 2026-11-19).

### 4.1 Linked builds, added 2026-11-19 (PROPOSED; received as a zip; filed 2026-11-19 at an interim home, see below)

- `residual-audit` (Kimi build; `residual_audit.py`, 713 lines, stdlib):
  - selftest: pass. Re-run here from the received file: `selftest: 0 failure(s)`, exit 0.
  - EXP1: null run by its own statement; 3 claims x 4 destruction batches, 12 records;
    single lineage (kimi-k3); operator = Q.
- Tooling report, "Tooling the Residual Check: An Ecosystem Survey and Build Plan for Four
  Repositories": 17 proposals (ids `em.*` 4, `gm.*` 4, `fa.*` 4, `ra.*` 5). All PROPOSED.
  Its citations other than Madl (§1.1) are UNVERIFIED.
- Received zip sha256 `f531061fc49e57acda6d3fb01144e98c08a0a5043d81a40f76ef69ac7d52487b`.
- Provenance flag: EXP1.md is headed 2026-11-20; this file was filed 2026-11-19.
  The 12 EXP1 records carry `created` = `2026-11-19T1946Z`.

Clean-up 2026-11-19 (DERIVED):
- Date of EXP1: read it from the records. The 12 records are stamped `2026-11-19T1946Z`; the
  CF02 runner in the companion zip is stamped `2026-11-19T19:13:29Z`. Records and runner
  agree; the heading (2026-11-20) is the outlier: a typo or a sandbox-clock issue.
- Resolved 2026-11-19 (this line replaces the "UNRESOLVABLE until filed" mark): the files are
  filed in this repository at an interim home, commit `32417d9`. `JinnZ2/residual-audit` does
  not exist; they move there when it does. Links, as path@commit:
  - `docs/research/residual-audit/INTERIM_HOME.md@32417d9` (zip sha256, per-file sha256, one rename)
  - `docs/research/residual-audit/residual_audit.py@32417d9`
  - `docs/research/residual-audit/README.md@32417d9`
  - `docs/research/residual-audit/examples/owl_claim.json@32417d9`
  - `docs/research/residual-audit/runs/exp1/EXP1.md@32417d9`
  - `docs/research/residual-audit/runs/exp1/archive/records.jsonl@32417d9`
  - `docs/research/residual-audit/tooling-report/Tooling_the_Residual_Check_Survey_and_Build_Plan.md@32417d9`
  - The zip itself is still in no repository; its sha256 is kept as provenance. The six filed
    files are byte-identical to the zip's members.
  - Selftest re-run from the filed copy, 2026-11-19: `selftest: 0 failure(s)`, 14 lines `ok`, exit 0.

Authorship 2026-11-19 (owner-stated): `residual-audit` and the tooling report are Kimi's
(OKComputer zip, sha256 `f531061fc49e57acda6d3fb01144e98c08a0a5043d81a40f76ef69ac7d52487b`).
The bound-example hunt (the README claim hunt, `tests/test_readme_claims.py --hunt`, in
formula-audit PR #2 and study-aperture PR #35) is Perplexity's.

Source for the Perplexity attribution, added 2026-11-19:

- The owner's statement of 2026-11-19 ("The bound-example hunt = Perplexity").
- The owner's order received in the filing session 2026-11-19T22:22Z (16:20 owner-side). As
  received there it opens with the label line `PERPLEXITY FINDING → RIPPLE (DERIVED)`, and
  its item 4 names the PR: `Copy formula-audit#2 and study-aperture#35 status into the
  housekeeping records: merged? CI?`
- The owner states that in the paste the hunt report sits under an explicit `Perplexity:`
  label and names study-aperture#35. That label line and the hunt report text are not in the
  text the filing session received; the two lines quoted above are what it holds.
- formula-audit PR #2's description carries a "Created with Perplexity" footer (read
  2026-11-19). study-aperture PR #35's description: not read.

Added 2026-11-19, later (relayed quote): the owner's order gives, from the owner's 2026-11-19
16:20 paste (a chat-side attachment), line 151, verbatim: `Perplexity:`. By the same order that
is the line directly above the paragraph beginning "The hunt is built and running in both repos,
and it's merged into both.", and that paragraph names JinnZ2/formula-audit#2 and
JinnZ2/study-aperture#35. Source: chat Claude, which holds the paste; relayed here through the
owner's order. The filing session has not seen the paste, so only the two quoted strings above
are recorded, and the rest of the paragraph is not quoted.

### 4.2 Notes, 2026-11-20 (records only; same four notes as `residual-audit/INTERIM_HOME.md`)

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

## 5. Open slots

```
run_id | instrument | carrier | ladder_rung | lineage(teacher,student,judge,A) | L |
S | P | RC | carrier_excess_z | flags | provenance | date
```
All void until a run fills them. Unknown → void, never dropped.

## 6. Next stages

1. AUDIT this file (different family): citations in §4 read against primaries → OBSERVED / NOT FOUND
2. Verify the tie-in claims against this repo's actual basin / trust-region code (not read here)
   - Note 2026-11-19 (Claude Code, read at `d67b12e`; code read, nothing run against it):
     functions the §3 tie-in would call
     - "stable only inside the current safe basin":
       `BasinDivergenceMonitor.kl_from_reference(theta, inputs)` and `.in_basin(theta, inputs)`
       (`addon_thermodynamic_control/energy.py:110`, `:117`; basin = KL to reference < `epsilon`);
       `ParameterManifold.geometric_confidence(theta, safety_risk)`
       (`manifolds/parameter_manifold.py:27`; exp(−curvature_weight·risk − ‖theta − theta_ref‖));
       `CoupledDynamicalSystem._phase(kappa, basin_kl, dV_dt)`
       (`addon_thermodynamic_control/stability.py:440`; stable / threshold / critical);
       `repair/generic_repair_controller.py` (stdlib; `in_basin`, `epsilon_basin`, line 123).
     - "flips toward the trust-region boundary": the trust region in this repo is a cap on the
       norm of one update step, not a region of parameter space:
       `ParameterManifold.repair_step` (`manifolds/parameter_manifold.py:37`, clamp at `:78`);
       `BridgeOptimizer` (`research_interface/bridge_optimizer.py:83`, reports
       `trust_region_active`); `research_interface/parallel_transformer.py:208`.
     - None found: no function here takes a claim, a prompt, an answer, or per-token log-probs.
       Every function above takes `theta` (flat parameter vector) plus input tensors through
       `model_fn(inputs, theta)`. An adapter from "claim" to (theta, inputs) is not in the repo.
3. Pick runs by what can be done stdlib on a phone vs what needs model API access; state the barrier
   - Note 2026-11-19 (Claude Code): barrier for the tie-in runs. The basin code in this repo
     imports torch (`manifolds/`, `addon_thermodynamic_control/`, `simulation/`), so those runs
     cannot be done stdlib / on a phone. OBSERVED in the filing session: `python -m pytest tests/`
     stopped at collection with `No module named 'torch'`. Route: a session or machine with torch.
     `subliminal_guard.py` itself is stdlib and runs anywhere.
