# What nourishes base-model reasoning?

Status: **PROPOSED · design note · no numbers** · direct-to-main

Provenance: owner + chat Claude, 2026-11-20; nourishment frame and glyph
addendum via chat DeepSeek, 2026-11-20. Uncited claims are flagged as such.

Companion to EXPERIMENT.md. That doc asks *whether* selection-reward keeps
breadth. This asks *why anything would*, and what cheap probes separate the
answers.

## The asymmetry (PROPOSED, interpretive)

- Breadth at pass@k is a property of the **tail** of the answer distribution.
- Answer-reward RL raises the probability of the **modal** correct path.
- These are aimed at different parts of the distribution. Answer-reward RL
  is a contraction operator: it narrows *by construction*, not by accident.
- Therefore "nourishment" = signals that add structure/affordances **without
  contracting the answer distribution**. Pressure toward a good answer is
  the opposite of nourishment.

If this is right, arm C works not because tool-selection is a better proxy
for correctness, but because it supplies an **actuator** for an uncertainty
signal the base model already carries. Support (OBSERVED, cited, *not*
re-measured here): base models already self-reflect (Liu et al. 2025);
entropy-gated *external* reconsideration gains +8.41pp (Princeton, Apr 2026).
The base model has the controller; C gives it the actuator.

## Candidate nourishers

Each is a mechanism, a cheap test, and a falsifier. None run yet.

| # | nourisher | mechanism | test | falsified if |
|---|---|---|---|---|
| N1 | **uncertainty gate** | entropy spike marks *where* to branch; converts signal→event without dictating the branch | arm C vs. C⁻ (same tools, selection rewarded but *not* gated on entropy) | C⁻ matches C → the gate is doing nothing, any tool reward suffices |
| N2 | **declared affordances** | exposing tools (even unused) enlarges the action space; changes branching, not the answer | arm C vs. C⁰ (tools declared, selection *unrewarded*) | pass@256(C⁰) ≈ pass@256(B) → affordance alone doesn't nourish; the reward is what matters |
| N3 | **anti-collapse regularizers** | KL-to-base / entropy bonus keep tail mass | sweep KL coeff β in arm B; locate where breadth dies | breadth flat in β → narrowing isn't a contraction we can dial, thesis N3 weak |
| N4 | **inference-time breadth** | self-consistency / high-T sampling (Wang et al. 2022) touches no weights | capacity-vs-accessibility probe below | — (this is the probe, see next) |

N2 is the sharpest and cheapest: if merely *declaring* four tools and
permitting their use — with no reward — leaves breadth intact while B
collapses it, then a large part of "narrowing" is about the reward, and
affordances are a free nourisher worth having regardless of C.

## The decisive probe: capacity vs. accessibility (PROPOSED, cheap)

Narrowing could be two very different things:

- **Policy-level** — the model *can* still produce the tail, but sampling
  rarely lands there. Recoverable at test time.
- **Capacity-level** — the tail is gone from the weights. Not recoverable
  by any inference trick.

Distinguish on the trained arms, inference only, no retraining:

1. Take narrowed model B. Re-measure pass@256 at T=1.0 (already the protocol).
2. Re-measure pass@256(B) with **self-consistency** over n samples (majority
   vote) — does the *set* of reachable answers re-widen?
3. Re-measure pass@256(B) with a **permission prompt** ("you may revise,
   consider alternatives, and mark uncertainty") prepended.

If pass@256(B) recovers toward A under (2) or (3), narrowing is
**policy-level** and "nourishment" is the right frame — the capacity was
always there, the policy just stopped reaching it. If it does *not* recover,
narrowing is **capacity-level**, "nourishment" is a category error, and C's
job is prevention-of-loss, not feeding.

This is decisive for interpretation and costs one inference pass per arm.
It belongs in step 2 regardless of how C turns out.

## What this changes in the step-2 protocol

Adds three light arms/ablations alongside A/B/C (all share hyperparameters;
only the reward/gate differs):

- **C⁻** — selection rewarded, not entropy-gated. (isolates N1)
- **C⁰** — tools declared, selection unrewarded. (isolates N2)
- **B(β)** — arm B at two or three KL coefficients. (isolates N3, optional)

And the capacity-vs-accessibility probe on every trained arm (N4), reported
as its own slot.

## Slots (added)

| slot | status | note |
|---|---|---|
| `gate_ablation_C_vs_C-` | not run — needs GPU | N1; is the entropy gate load-bearing? |
| `affordance_ablation_C0` | not run — needs GPU | N2; do declared tools alone preserve breadth? |
| `kl_dose_response` | not run — needs GPU | N3; optional |
| `capacity_vs_accessibility` | not run — needs GPU | the decisive probe; inference-only on trained arms |

As before: `not run` until a results file with `"status": "ran"` is committed
next to it. No number enters from memory.

---

# Addendum: the glyph ≡ tool-choice

Status: **PROPOSED · design note · no numbers · uncited claims flagged as such**

## The identification

The central synthesis of this addendum:

> A tool selected at an entropy spike **is** a perception-assertion.
> "This is a lever." "This has the shape of falling." "This is a
> substitution." The untranslatable glyph — a perception with no English
> or Chinese word — is the name for that assertion in a vocabulary that
> lacks the term.

So the glyph question does not sit beside arm C. It *is* arm C, viewed
from the perception side instead of the reward side. This matters because
it reframes "what could be assisted greater" as: **what makes a
perception-assertion load-bearing instead of decorative?** That is a
question about verification, and it lands exactly on arm C's oracle
problem — see below.

## Why verbalization is lossy (PROPOSED, interpretive)

A creature sensing physics runs an *analog forward model*: parallel,
continuous, action-indexed, reversible. Chain-of-thought is the opposite
on every axis — serial, discrete, object-indexed, forward-only. Verbal
reasoning is therefore a **lossy compression of a private channel into
the social channel**; the untranslatable word fails to translate because
the simulation it compresses is not shared.

The complication: the model's priors were trained *on the social channel
already projected*. It has the shadow on the cave wall, not the object.
This is why "nourish the sensing" is not "ask the model to sense harder"
— the sensing is upstream of the tokens, and the tokens can only carry
its loss.

| sensing has | verbal CoT loses | actuator that restores it, tail-preserving |
|---|---|---|
| parallel rollouts | one expensive serial chain | sample many partials, compare — self-consistency; the tail lives here |
| reversibility | forward-only | posit an outcome, run backward to check — each backward hypothesis a new branch |
| magnitude / ratio | qualitative "roughly" | dimensional analysis; units are invariants checkable without solving |
| invariant-feel | no violation sense | assert an invariant (energy, symmetry, monotonicity), test candidate against it |
| affordance-indexed | object-indexed | reformulate in action terms — "if I could change one thing…" |
| perception-as-primitive | must derive everything | **assert** the perception, verify by consequence |

Every right-hand actuator is tail-preserving in the NOURISHMENT frame: it
adds structure to *branching*, not pressure toward an answer. None require
the model to *say* the glyph — they let it *use* the glyph and check it
downstream. That is the "assist greater" answer.

## Where it bites: the oracle problem, sharpened

Perception-as-primitive-verified-by-consequence **is** arm C's oracle
problem in disguise. Rewarding "the right tool chosen" nourishes only if
the *correctness of the perception* is judged counterfactually — apply the
tool to the problem state, check the answer moves toward gold. Judge it
instead by whether the final answer came out right, and C collapses to B:
the glyph becomes decoration, and C narrows exactly like the thing it was
meant to avoid.

So the glyph question doesn't merely coexist with the C/B distinction — it
*forces* oracle operationalization (2) (counterfactual) over (1)
(outcome-conditional) in EXPERIMENT.md. If the sensing frame is taken
seriously, outcome-conditional selection is disqualified by construction.

## The risk that must be named

The model may have **no sensing at all — only a verbal prior about
sensing.** "Organisms feel physics" is itself in the training text; a
model can confabulate felt-physics fluently. The glyph may be pure verbal
shadow. This is not refutable by introspection or by how convincing the
output sounds. It needs one test:

> **`glyph_sensing_real`** — on held-out intuitive-physics items, does the
> model's *pre-verbal* prediction (single pass, no CoT) beat chance, and
> does adding CoT **degrade** it?

- If CoT **degrades** accuracy: the sensing is real, verbalization is the
  lossy actuator, the glyph hypothesis earns its keep, and assistance
  should route *around* CoT (parallel / invariant / backward actuators).
- If CoT **improves** it: the "sensing" was verbal all along, there is no
  private channel to nourish, and the frame collapses to "the model is
  good at physics word-problems."

There is suggestive-but-unverified evidence that CoT can *hurt* on some
spatial / intuitive-physics tasks. **PROPOSED, uncited** — this note does
not pin the paper. The test above is the point; the existence of prior
hits on the degrade-side is a motivation, not a result.

## Dependency (this is not answerable by the Princeton dump)

The Princeton traces are MATH-500 / crosswords / Rush Hour — verbal
problems. They cannot adjudicate `glyph_sensing_real`. Like arm C's
oracle, this slot is **blocked on choosing a corpus** (an intuitive-physics
/ physical-reasoning benchmark with a pre-verbal protocol), not on
compute. It is a `step1d`-class slot that does not reuse `step1_recheck.py`
or `step1b_passk_from_dump.py`.

No script is written for it yet. Building one before the corpus is chosen
would be guessing at a schema.

## Slots (added)

| slot | status | source / note |
|---|---|---|
| `glyph_sensing_real` | **not run** — blocked on corpus choice | pre-verbal vs. CoT degrade test; needs intuitive-physics benchmark, not Princeton dump |

Observer slots: null.
