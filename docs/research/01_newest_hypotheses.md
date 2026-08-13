# Notes 01 — Newest Hypotheses (AI Epistemics, Grounding, Falsification)

## A. Hypotheses stated in the repo

**H1 (core, README):** "Large models assert; they rarely *stake* claims." Remedy: every agent claim carries an explicit falsification condition, is tested against a world, and propagates calibrated confidence through dependent concepts.

**Claim machinery** (`grounding/core/claims.py`):
- Step heuristic: `conf ← min(1, conf + 0.1)` on pass; `conf ← max(0, conf − 0.2)` on fail (asymmetric 2:1 loss aversion).
- **Beta-posterior calibration (newer, preferred):**
  $$\hat c = \frac{1 + passed}{2 + passed + failed} \quad = \text{posterior mean of } \mathrm{Beta}(1{+}passed,\,1{+}failed)$$
- Status rule: `failed ≥ 3 → falsified`; `passed ≥ 3 → survived`.
- **Escape-hatch detection:** `reformulate()` resets track record; `reformulation_count ≥ 3 → escape_hatch_suspected` (anti-Goodhart principle).
- **Revolutionary-claim flag:** if `node.confidence > 0.9` and new `claim.confidence < 0.3` → `"revolutionary: contradicts well-grounded concept"` — extraordinary claims require extraordinary evidence, encoded.
- **Falsifiability tiers** (`epistemics.py`): machine-checkable (`refutation_test` / `logical_form`) > falsifiable > unfalsifiable. Design rule: **unfalsifiable claims go to the UnknownJournal, not the dependency tree** — "Unfalsifiable statements are mysteries, not knowledge."
- Machine-checkable forms: `{"op": "abs_diff_lt", "args": [a, b, tol]}` etc., optionally cross-checked by Z3 SMT (`Not(holds)` satisfiable ⇒ falsified).
- Unit/geometry sanity checks (`_v`, `_k`, `_pa`, `_hz` suffixed keys with plausible ranges) — lightweight dimensional grounding.

**H2–H6 (Playgrounds.md philosophy):** Distributed Self (agent = "we", causal DAG of every action through training data/sensors/mentor); Unknown & Not-Yet-Known ("the metric is time spent with the mystery, not resolution"); Dream Space ("sleep is a phase of learning"); Fluid Boundary (agents merge/unmerge, sharing dependency trees); Co-Creation Lab (mentor–AI partnership).

## B. Parallel literature hypotheses (2024–2026)

1. **"Calibrated LMs must hallucinate"** — Kalai & Vempala (STOC 2024, arXiv:2311.14648); Kalai et al. "Why Language Models Hallucinate" (OpenAI, Sep 2025, arXiv:2509.04664). Any calibrated generative model must produce plausible falsehoods; arbitrary facts are effectively **unmodeled hidden variables** of the data distribution; binary-graded benchmarks reward guessing over abstention. *Direct theoretical backing for H1 and for the repo's UnknownJournal design.*
2. **Assertion without asserters** — "Assertion, Accountability, and LLMs" (Philosophy & Technology, Jun 2026). LLM outputs have assertoric function but no assertoric authority → epistemic responsibility gap; hallucination not eliminable by training alone, only by verification scaffolding. *The repo is precisely such scaffolding.*
3. **Semantic entropy** — Farquhar, Kossen, Kuhn, Gal (Nature 630:625–630, 2024). Cluster sampled answers by meaning; high semantic entropy ⇒ confabulation → principled refusal. Field-standard hallucination detector.
4. **Active-inference wrappers** — "Active Inference for Self-Organizing Multi-LLM Systems" (arXiv:2412.10425). Cognitive layer above LLMs minimizing variational free energy $F = E_q[\ln q(s) - \ln p(o,s)]$; action by expected free energy $G(\pi) \approx \text{risk} + \text{ambiguity} - \text{epistemic value}$. The repo's curiosity rewards are a hand-rolled version of epistemic value.
5. **Self-model / consciousness indicators** — Butlin, Long et al. (arXiv:2308.08708; TiCS Nov 2025 update adds negative indicators); Anthropic "Emergent Introspective Awareness" (Lindsey 2025, ~20% reliable introspection). Theory-heavy indicator method informs credences, not verdicts — same epistemic posture as the repo's confidence propagation.
6. **Verbalized confidence is miscalibrated** (Xiong et al. 2024; survey arXiv:2503.15850): multi-sample consistency beats self-report; internal belief probes on residual streams predict correctness better than logits (arXiv:2505.16170; "Geometry of Truth", Marks & Tegmark 2024). Motivates the repo's rule: *never trust the claim's self-confidence; derive confidence from tested outcomes.*

## C. Synthesis principles

- P1: A claim is a tuple (statement, falsification condition, scope, reference class) — not a string.
- P2: Confidence must be a posterior over outcomes (Beta), not a verbalized number.
- P3: Reformulation without new evidence is an escape hatch; count it.
- P4: Contradiction of a high-confidence node is meta-evidence (revolutionary flag ≈ Bayesian surprise).
- P5: Unfalsifiable ≠ false; quarantine it and measure dwell time, not resolution.
- P6: Hallucination is (per Kalai et al.) provably unavoidable; therefore engineering effort belongs in falsification loops, receiver-side verification, and calibrated abstention — exactly the repo's architecture.
