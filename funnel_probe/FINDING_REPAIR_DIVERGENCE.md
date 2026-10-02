# FINDING — repair divergence (Geometric-manifold-)

Status: OBSERVED. Cause: unknown at the time of this record. Work order
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
