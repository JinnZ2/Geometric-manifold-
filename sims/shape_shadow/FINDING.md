# FINDING — shadow faults are real but weak, and one of my hypotheses was malformed

Run 2026-08-13T2109Z · octahedron, 12 edges, 4 residual magnitudes × 8 seeds · ~55 s ·
graded by `run.py` against `REFUTE.md`.

## Verdict: REFUTED — H3 failed, and it deserved to

| hypothesis | result |
|---|---|
| **H1** full-displacement Jacobian is rank 12 | **holds** — rank 12, as Maxwell counting predicts for an isostatic framework |
| **H2** vertex-magnitude Jacobian has rank ≤ 6 | **holds, but the test was worthless** — see below |
| **H3** the least-visible direction casts a shadow ≤ 1/10 of typical | **fails** — it casts **1.05× typical**, i.e. it is not blind at all |
| **H4** collisions exist at finite amplitude (added after H3 failed) | **partially** — concealment 0.21× / 0.39× / 0.66×, real but far short of the 0.1× bar |

## Two errors of mine, recorded because they change how the result reads

**H2 was tautological.** A 6×12 Jacobian has rank ≤ 6 by dimension alone. Predicting "rank
≤ 6" and observing rank 6 confirms arithmetic, not physics. It should have been stated as a
claim about *concealment*, which is what H4 ended up testing.

**H3 linearized at the wrong point.** The vertex-magnitude observable is `|disp_i|`, and at
r = 0 the displacement is zero, where the norm is not differentiable. The resulting
"kernel" is an artifact of that, and the numbers say so plainly: the magnitude Jacobian's
largest singular value is **0.0001** against **1.41** for the full field — four orders of
magnitude smaller, because it is measuring second-order effects at a non-smooth point.
Directions that look blind in that linearization are perfectly visible at finite amplitude
(1.05× typical). The pre-registered test was measuring an artifact of its own base point.

Both are why H4 was added and labelled as a post-hoc addition rather than folded into the
verdict.

## What is actually true about the shadow

The map from 12 edge residuals to 6 vertex magnitudes **cannot be injective** — that is
certain by dimension count, no experiment required. So collisions exist. The question worth
measuring is how *deep* the concealment is, and the answer is: shallow.

| step along the null direction | shadow moves | typical for same-size change | concealment |
|---|---|---|---|
| 0.025 | 4.21e-03 | 1.98e-02 | **0.21×** |
| 0.050 | 1.56e-02 | 4.05e-02 | **0.39×** |
| 0.100 | 5.13e-02 | 7.76e-02 | **0.66×** |

Two things to read here. Concealment is real — a fault change along the null direction
shows up about **5× more quietly** than an arbitrary change of the same size. And it decays
fast: by a step comparable to the base fault itself, concealment is nearly gone (0.66×).
The linearized null direction curves away from the true level set almost immediately, so
there is no corridor a fault can travel along while staying hidden.

## What this means for notes 14

**It supports the drill-down claim more than it qualifies it.** notes 14 measured vertex
localization at 6×, face aggregation at 2.8× and low-mode dominance at 85%, all on faults
that cast shadows. The obvious worry — that a 12 → 6 projection leaves half of fault space
invisible — does not survive contact with the geometry. Every direction produces an
ordinary-sized shadow at finite amplitude, and the concealment that does exist is a factor
of about two to five, bounded and shrinking with fault size.

The honest caveat is narrower than the one I set out to test: **two faults differing by a
small step along a null direction are harder to tell apart than two arbitrary faults, by
roughly 5× at small separations.** That is a resolution limit on discriminating *similar*
faults, not a blind spot that hides faults outright.

## Limits

- **One shape, one base fault, one null direction.** The collision test uses the first null
  vector at a single random base residual. A systematic scan over the 6-dimensional null
  space and over base faults would give a distribution rather than three numbers.
- **Isotropic prior.** "Typical" means typical under a uniform prior over residual
  directions. Whether concealed pairs correspond to faults that actually co-occur is a
  different question this cannot answer.
- **The definition is mine.** "Shadow shape" had no prior definition in this repo or the
  notes; the operational one in `REFUTE.md` — observable as a reduced statistic, shadow
  fault as a residual that moves the shape without moving the statistic — is a proposal.
  A different intended meaning would need a different experiment, and this one should not
  be read as settling a question it may not have been asked.
