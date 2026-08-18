# The off-topic results were the transfer, not the noise

`HYPOTHESIS_TRIAGE.md` reported that 86% of the loss-landscape topic's surviving claims
were off-topic keyword collisions, and treated that as a precision failure to be filtered
away. Filtering was the wrong first instinct. Reading what those papers actually say
changes the verdict.

## What the "off-topic" set is actually about

| retrieved paper | its actual subject | what it maps onto here |
|---|---|---|
| Mott transition in twisted bilayer graphene | quasiparticle residue **Z vanishes** at the transition | λ_min → 0 — critical slowing down, `DOMAIN_PHYSICS.md` §4, Q1 |
| Interface phases in 2-D quantum magnets | **interfaces separating ordered bulk domains** | basin boundaries — `sims/fractal_basin_damping/` |
| Pore-scale flow curvature, Lewis number | **stability of a propagating front** | the same fold/bifurcation machinery |
| Beyond asymptotic flatness: pressure-induced curvature, rotation curves | **modified dynamics vs unseen mass** | the *other horn* of `sims/dark_constraint/` |
| Impurity quantum criticality with **entanglement witnesses** | an observable that **certifies** proximity to criticality | exactly what κ_eff was meant to be — Q1 |
| Contiguity approach to replica symmetric marginals | **mean-field Gibbs systems**, cavity method | `docs/theoretical_notes/mean_field.py` already exists here |
| Environmental quenching of galaxy star formation | environment/coupling determines state | `research_interface/coupling_coherence.py` |
| Default mode network connectivity → treatment response | network **connectivity predicts recovery** | the eigenratio criterion in `coupling_coherence.py` |

Eight papers, eight disciplines, and every one lands on a piece of work this repo is
already doing. That is not a keyword accident to be engineered away. It is
`17_fractals_bio_cosmo_trig.md`'s central claim arriving unbidden: *"each domain
independently arrived at the same saddle-node/fold machinery."*

The words that caused the collisions — *flatness*, *curvature*, *basin*, *stability* — are
field-agnostic **because the mathematics is**. A retrieval system that filters on
discipline is filtering out precisely the Rosetta transfer the ecosystem exists to find.

## Two of these are immediately useful

**Entanglement witnesses.** Quantum information has a rigorous theory of *witnesses*:
observables constructed so that a particular value certifies a property of the state, with
explicit necessary/sufficient conditions and a notion of an optimal witness. That is the
formal version of the question Q1 asks and κ_eff failed: *what observable certifies
proximity to a transition?* Three rounds of `sims/kappa_eff_leading/` established that
κ_eff is not such an observable at this scale. The witness literature is where to look for
what the construction requires, rather than trying another ad-hoc scalar.

**The rotation-curve paper is the alternative hypothesis in `sims/dark_constraint/`.**
That sim established that at one load case an unmodelled constraint is perfectly absorbed
into fictitious visible residuals — the degeneracy between "my modelled components are
misbehaving" and "there is a component I have not modelled". *Beyond the assumption of
asymptotic flatness* is a live attempt at the modified-dynamics horn of exactly that
degeneracy in the astrophysical case. It is the closest thing to prior art for the sim's
open half.

## What changed in the engine

A second, unrestricted retrieval pass now runs alongside the category-filtered one. Matches
that appear only without the category restriction are written to `data/cross_domain.jsonl`
and **not staked as claims**.

That split matters in both directions:

- The claim tree stays clean, so "surviving" counts mean what they say. A paper on
  combustion-front stability is not evidence about loss landscapes, and staking it as
  though it were is what produced the misleading 86% in the first place.
- The transfer candidates are preserved with their topic and abstract, so the material is
  there to read rather than silently dropped by a filter.

Neither lane contaminates the other, and the report counts both.

## The honest caveat

**Nothing about this makes cross-domain matches automatically valuable.** The eight above
were selected by hand, after reading them, because a shared mechanism was visible. Most
unrestricted matches will be genuine noise, and the same *flatness* keyword that found
interfaces in quantum magnets also found a flatness interferometer with no abstract and
nothing to transfer.

What the lane buys is the chance to look. The judgement stays manual, which is the correct
place for it — the engine cannot tell a shared mechanism from a shared word, and this
document exists because a person noticed the difference.

---

## First autonomous run (2026-08-17): the lane split, measured

The first *scheduled* cron run with every fix in place. Three of them worked unattended:

- `new_hypotheses: []` — the announced-topics fix held, no third duplicate issue.
- `hidden_variable_suggestions: []` — no self-correlations.
- `new_findings: 32 → claims_staked: 16 + cross_domain_candidates: 16`. The split is live
  and exactly accounts for the intake.

**But the split is not sorting by what it was meant to sort by.** Reading both lanes:

| lane | plausibly on-topic | rate |
|---|---|---|
| primary (category-filtered, **staked as claims**) | ~3 of 16 | ~19% |
| cross-domain (unrestricted, **logged only**) | ~3 of 16 | ~19% |

The two lanes have **the same topical precision**. Categories removed the cross-*discipline*
noise — no more general relativity or geriatric depression — but inside `cs.LG` the queries
match nearly anything: GPU sharing for vision-language models, cytometry foundation models,
sex estimation from skeletal remains, MIMO detection. arXiv category turns out to be almost
uncorrelated with whether a paper is about the topic.

That makes the asymmetry between the lanes unjustified as it stands. The primary lane
**stakes** its 16 into the claim tree, where they count toward "surviving"; the cross-domain
lane merely logs its 16. Two streams of equal precision are being treated as though one were
evidence and the other were not. The primary lane is contaminating the tree at the same rate
the cross-domain lane was — which is the problem the category filter was introduced to solve.

**The gate should be relevance, not category.** That is a harder thing to build than a
`cat:` prefix, and it is the actual bottleneck; nothing about retrieval improves until a
claim has to earn its way into the tree on something better than keyword overlap.

## The hand-picked set was survivorship-biased, and now it is quantified

`CROSS_DOMAIN_YIELD.md` above stated the caveat plainly — "the eight were selected by hand,
after reading them" — and the autonomous run measures it. Hand-picking gave 8 of 8 clean
mechanism transfers. The lane running by itself gives roughly 3 of 16. The caveat was
correct and the correction factor is about 5×.

The three worth reading from this run:

- **Spatiotemporal Tube-Based Safety-Certificate for Autonomous Navigation** — the strongest
  hit. A safety *certificate* is exactly the witness concept from the entanglement-witness
  thread (an observable whose value certifies a property), and tube-based robust control is
  the CBF/safety-envelope machinery in `PLAN_FORWARD.md` §3.1 and `HARDWARE_INTEGRATION_PLAN.md`
  I10. Two independent threads of this repo meet in one paper.
- **Joint Estimation of Sparse Multilayer Networks via Graph Limits** — multilayer networks
  and spectral limits, which is `research_interface/coupling_coherence.py`'s territory and
  IP-15's three-layers-as-stalks framing.
- **On quantitative sufficient second-order optimality conditions** — second-order
  sufficiency is the formal version of the question `sims/objective_sign/` just answered
  empirically about what makes a point a minimum rather than an escape.

## Also measured: 82% of the claim tree is unattributable

`claims_predating_provenance: 72` of 88. Only the 16 staked this run carry a config hash, so
the staleness check added last time can only see 18% of the tree. That is not a defect in the
check — it is the honest state of a ledger that ran for three sessions before anyone recorded
where its entries came from, and it will decay as the tree turns over.
