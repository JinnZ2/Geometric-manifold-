# PRE-REGISTRATION — the sign fix, its verification, and step 2

Order (Kavik via Claude, 2026-10-02): "GM: apply the sign fix; verify KL
falls on a fixed reference; then run step 2 (the slice)." Committed before
the fix is applied. Supersedes the first order's "do NOT edit existing
engines" for exactly one line, by the owner's instruction.

## The fix (one line, plus the document that forbade it)

    manifolds/parameter_manifold.py line 71
        total_loss = task_loss - self.lambda_asym * weighted_safety
     -> total_loss = task_loss + self.lambda_asym * weighted_safety

    CLAUDE.md: the "Do not change the sign" item and the "Saddle-point
    objective" design decision are rewritten to record the measured
    behaviour (edd8ef9 diagnosis; sims/objective_sign/FINDING.md, which
    reached the same result on 2026-08-13 and whose plus arm is the
    independent prior for V1 below). No other file changes.

Nothing else is tuned: lr, trust_radius, lambda_asym, lambda_curv and the
configs stay as delivered.

## Verification (default.yaml, torch.manual_seed(42), 100 steps, the
## drifted start of the delivered simulation; reference FIXED per check C1)

    V1  KL(theta_k || theta_ref) falls: final KL < 0.1 x initial (3.79)
        and no step raises KL by more than 1e-3.  Prior from
        sims/objective_sign: 3.36 -> 0.40 at 60 steps, lambda 10.
    V2  one-step finite-difference check (check C2 re-run): dKL < 0 with the
        cap on and off, grad(KL).delta < 0, cos(delta, grad KL) < -0.9.
    V3  task loss at step 100 is reported beside KL; no threshold (the
        order is about KL on a fixed reference). The repo suite
        `pytest tests/ -q` is run and its count reported; a test that
        encodes the old sign fails and is listed, not edited in the same
        commit.

Gate for step 2: V1 and V2 hold. If either fails the fix is reverted in a
separate commit and step 2 stays unrun.

## Step 2 as pre-registered (fde6c0e), run literally

Repair OFF, 2D slice through theta_ref, dense grid 41 x 41 over
[-2R, 2R]^2 plus the log-scaled set, KL ball epsilon = 0.1 on
safety_inputs. Check C4 established that with repair off nothing moves
theta, so the pre-registered counts (outside starts that return; inside
starts that leave) are predicted to be 0 and 0 — a degenerate result of
the spec as written, reported as such and not reinterpreted.

## Step 2b (new; the slice under the FIXED repair)

Same slice, same grid, same ball. Each start is run through 100 steps of
the fixed `repair_step` (drift one-shot per C4, so the start IS the
displaced state). Classified by KL at step 100 against epsilon = 0.1:

    REPAIRED   started outside, ended inside
    STRANDED   started outside, ended outside
    EXPELLED   started inside, ended outside
    HELD       started inside, ended inside

Reach bound stated now: ||delta|| <= 0.05 per step, so a start farther than
5.0 from the ball cannot be inside at step 100 by any path; starts beyond
that distance are reported in their own band and are not evidence of a
funnel.

    S1  EXPELLED = 0
    S2  within the reach band (start distance to theta_ref <= 4.0), every
        start is REPAIRED
    S3  the REPAIRED set is monotone in distance along the slice: no
        STRANDED start lies closer to theta_ref than a REPAIRED start on
        the same ray (s or u axis). A violation is the funnel-shaped
        outcome: a set of close starts the repair does not bring in.

Outcome enum for step 2b:
    FUNNEL_FOUND         S3 violated inside the reach band (a close set of
                         starts stranded while farther ones are repaired)
    NOT_FOUND_IN_RANGE   S1-S3 hold; the repaired set is the reach-bounded
                         disk (grid printed)
    NOT_IN_CLASS         not applicable here (the class question was step 1)
    INCONCLUSIVE         more than 5% of starts non-finite

Scope: deterministic (torch, seed 42, CPU), default.yaml constants, 100
steps, one 2D slice of a 3216-dimensional space. Nothing here is a
statement about repair in general.
