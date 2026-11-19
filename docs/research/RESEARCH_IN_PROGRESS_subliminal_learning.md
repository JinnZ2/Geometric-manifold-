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

## 3. External extensions received 2026-11-19 (author model: [void, unlabeled in paste])

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

## 4. Research candidates (from the external model; citations UNVERIFIED)

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
