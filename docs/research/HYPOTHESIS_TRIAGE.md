# Triaging the engine's output into tests worth running

Basis: the two live runs of 2026-08-13 (65 claims in `data/claim_tree.json`, 5 hypothesis
drafts). The question is which of these become pre-registered experiments under
`HARNESS.md`, and the honest answer is: **almost none of them, and not for the reason the
counts suggest.**

## The headline numbers do not mean what they look like

The report says 65 claims, 37 surviving, 2 topics flagged NEW HYPOTHESIS. Read literally
that sounds like 37 candidate hypotheses. It is not.

**A "claim" is a template-distilled sentence, not a proposition.** The design doc says so:
claims are built as `On topic {topic}, {title} reports: {first sentence of abstract}`.
There is no independent content to test — the claim is a citation with a falsification
condition stapled to it.

**"Surviving" is a weak predicate.** It means cross-source corroboration heuristics did not
contradict a paper's own abstract sentence. For a sentence a paper wrote about itself, that
is close to automatic. Survival here measures *absence of textual contradiction*, not
support.

## Three of five topics produced nothing at all

| topic | claims | surviving |
|---|---|---|
| loss-landscape geometry and basin stability | 29 | 22 |
| model collapse, variance suppression, recursive training | 23 | 15 |
| safety drift, alignment repair, parameter-space monitoring | 8 | **0** |
| Fisher information and thermodynamics of learning | 3 | **0** |
| hidden variable detection / causal discovery from residuals | 2 | **0** |

The three empty ones are the three closest to this repo's actual subject. Whatever the
engine is retrieving, it is not retrieving the literature this framework sits in.

## The loss-landscape topic is 86% off-topic

3 of 22 surviving claims are plausibly about neural network loss landscapes. The rest are
keyword collisions on *flatness*, *curvature* and *basin* — words that belong to every
quantitative field:

- twisted bilayer graphene near a Mott transition (condensed matter)
- interface phases in 2-D quantum magnets (condensed matter)
- pore-scale flow curvature (fluid dynamics)
- asymptotic flatness and galaxy rotation curves (general relativity)
- radius of curvature with a low-coherence flatness interferometer (optical metrology)
- completeness of quasicontinuous function spaces (pure maths)
- environmental quenching of galaxy star formation (astronomy)
- default mode network connectivity in geriatric depression (clinical neuroscience)

**Fixed at the source rather than downstream:** `query_arxiv` now accepts arXiv subject
categories, and `configs/topics.json` pins each topic to its classes (`cs.LG`, `stat.ML`,
`cs.NE`, …). Filtering after retrieval would have meant tuning a relevance heuristic
against the same corpus that produced the problem; restricting the query is the cheaper and
more honest fix. The three empty topics also had their queries retargeted, since phrases
like "thermodynamic cost of learning free energy" match almost nothing on arXiv.

## The one genuinely valuable find

Buried in the model-collapse topic is a **critical-slowing-down / early-warning-signals
cluster**, including a limits paper:

- *Critical Slowing Down Theory Gives Precursors…*
- *Critical slowing down theory provides early warning…*
- **Limits of using early warning signals for…**
- *When Tails Are Heavy: The Benefits of Variance…*

This is the one place the engine earned its runtime. That cluster bears directly on
`OPEN_QUESTIONS` **Q1** — the κ_eff test near σ_crit — and on the whole EWS thread running
through notes 07, notes 12 S6 and `sims/kappa_eff_leading/`. A limits-of-EWS paper is
exactly the prior art for a result that has already been reproduced here three times over:
detection and false-alarm rate trade against each other, and the operationalization decides
the verdict.

**Action:** read the limits paper before running Q1, and pre-register its stated failure
conditions in that sim's `REFUTE.md` rather than inventing new ones. That is the same move
that made `sims/ep8_snap_latency/` credible — transcribed criteria beat authored ones.

## The triage, stated plainly

| category | count | disposition |
|---|---|---|
| Testable claims requiring no new apparatus | **0** | — |
| Claims that are propositions at all (not citations) | **0** | the template guarantees this |
| Genuinely on-topic literature pointers | ~8 | read them; they are references, not experiments |
| Off-topic keyword collisions | ~30 | addressed by category filtering |
| Topics returning nothing | 3 of 5 | queries retargeted |

**No experiment in this repo should be derived from the current output.** The engine is
working as designed — it explores, stakes, tests and consolidates without error — but what
it produces is *reading*, and the design doc says as much: "hypothesis drafts are starting
points for human review."

The testable work continues to come from `OPEN_QUESTIONS.md`, where each entry originated
in a measurement rather than a retrieval. Q1 remains the highest-value experiment, and the
engine's contribution to it is one citation cluster, which is a real contribution and a
much smaller one than "37 surviving claims" implies.

## What would make the engine produce testable output

1. **Extract quantities, not sentences.** A claim worth testing needs a number, a
   comparison and a condition — "method A beats B by x% on benchmark C". The current
   template cannot express that. Mining abstracts for `(metric, delta, dataset)` triples
   would be a different and much harder stage.
2. **Corroboration is not replication.** The design doc already flags this. Until a claim
   can be checked against something other than another abstract, "survived" will keep
   meaning "nothing textually contradicted it".
3. **Report precision, not volume.** The most useful number this analysis produced is 86%
   off-topic, and the engine does not compute anything like it. A per-topic relevance
   estimate would have surfaced the problem on run one instead of run two.
