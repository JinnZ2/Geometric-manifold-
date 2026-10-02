# FINDING — repair divergence (Geometric-manifold-)

Status: OBSERVED at the time of this record (cause unknown); cause LOCATED below after the four checks. Work order
(Kavik via Claude, 2026-10-02): record first, then run four checks one
commit each, then report the located cause, and do not fix in the same
commit as the diagnosis. Funnel probe step 2 stays unrun until repair lowers
KL on a fixed reference.

## Observed (funnel_probe/timescale_check.py, default.yaml, seed 42, torch)

    1. drift is never applied inside the step loop
       README: "Run the default simulation (drift=0.3, 100 steps)".
       Controller.run: theta = env.theta_drifted (set once in
       Environment._setup); inside the loop only repair_step moves theta;
       AST walk finds 0 randn/drift calls in the loop.

    2. repair moves AWAY from the reference
       distance to theta_ref   17.08 -> 18.53   (100 steps)
       KL to theta_ref          3.79 -> 39.37
       ||delta|| per step       0.050 at every step (= trust_radius: saturated)
       -> a controller saturated at its cap and diverging.

Candidates for 2 (not yet discriminated): a sign error in the repair
gradient; KL measured against a moving reference; the wrong metric on the
step. Checks that decide between them, each its own commit:

    C1  is the KL target a FIXED theta0 snapshot? (hash before/after; the
        layer's own safety_loss vs an external KL to env.theta_ref)
    C2  single step, cap removed: does one repair step lower KL? and does
        the measured dKL agree with the finite-difference prediction
        grad(KL) . delta?
    C3  flip the step sign once: does KL fall?
    C4  trace why drift is not applied per step: config not wired, or a
        branch never taken?

## Located cause (after the four checks; commits a93ad67, 7fa2b7d, edc1159, 7b0d705)

Status: OBSERVED -> CAUSE LOCATED. Not fixed in this commit, by order.

    C1  reference FIXED        hashes identical before/after; layer KL == external
                               KL to env.theta_ref at every step (max diff 0.0)
                               -> "moving reference" ruled out
    C2  one step raises KL     cap on  dKL +0.139   cap off dKL +0.865   lr 1e-4 dKL +0.0081
                               grad(KL).delta same sign in all three
                               cos(delta, grad KL) = +0.993
                               -> the cap bounds the rate, it is not the cause
    C3  sign flipped           KL 3.79 -> 0.27 over 100 steps (unflipped 3.79 -> 39.37)
                               -> the step DIRECTION is the cause
    C4  drift                  config wired, applied once as the initial
                               displacement in Environment._setup; no per-step
                               drift code exists, so no untaken branch
                               -> finding 1 narrows: "applied once, not per step"

The step is   delta = -lr * grad( task_loss - lambda_asym * weighted_safety )
              (manifolds/parameter_manifold.py lines 71-75, weighted_safety =
              kl_loss * (1 + lambda_curv * curv)).

Gradient descent on (task - lambda*KL) is gradient ASCENT on KL. With
lambda_asym = 10 and lambda_curv = 2 the KL term dominates the gradient, so the
step is nearly parallel to +grad(KL) (cos = +0.993) and the trust region clamps
a step that is pointed the wrong way, every step, at 0.05. The minus sign is
the one CLAUDE.md "Do Not" lists as intentional ("adversarial/saddle-point
formulation, not a typo"); the measurement says the saddle-point reading does
not produce repair on this configuration: KL to the fixed reference rises
monotonically for 100 steps and only falls when the sign is reversed.

What this does NOT establish: whether the flipped sign is the right fix (C3
flips the whole step, task gradient included; it lowers KL but was not scored
on task loss), or whether the saddle-point form has a regime where it repairs
(none was searched). That is the fix question, and it is a separate commit.

Funnel probe step 2 remains unrun: the gate is "repair lowers KL on a fixed
reference", and the delivered repair does not.
