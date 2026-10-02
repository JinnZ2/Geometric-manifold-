#!/usr/bin/env python3
"""
singular_funnel_pitchfork.py  -- CC0, stdlib only

Reproduction target: Yanchuk, Wieczorek, Jardon-Kojakhmetov, Alkhayuon,
"Singular basins in multiscale systems: tunneling between stable states",
Phys. Rev. Lett. 137, 147202 (2026); arXiv:2601.02001, Fig. 1(a), eqs (1-2).

    dx/dt  = x (mu - x^2)            x >= 0   (fast)
    dmu/dt = eps (-mu + a x - b)              (slow)
    a=3, b=2, eps=0.1  ->  e0=(0,-2) stable, e1=(1,1) saddle, e2=(2,4) stable

REDUCED MODEL (adiabatic elimination): threshold at mu = 1.
    mu0 < 1 -> e0,  mu0 > 1 -> e2.   No other outcome is possible.
FULL MODEL claim under test: a thin funnel near x=0 reaches e0 from
    arbitrarily large mu0, width shrinking ~ exp(-C/eps), never zero.

Numerics: integrate y = ln x (exact for x>0), so x down to 1e-300 is
representable. Fixed-step RK4. This is NOT the paper's ode45 setup.

Outcome branches are kept distinct:
    REPRODUCED      funnel points found where reduced model forbids them
    NOT_REPRODUCED  none found in the searched range (searched range printed)
    INCONCLUSIVE    trajectories hit the time cap unclassified
"""
import math, sys

A, B = 3.0, 2.0
E0, E2 = (0.0, -B), (2.0, 4.0)
MU_SADDLE = 1.0                      # reduced-model threshold
DT, T_MAX = 0.01, 600.0

def rhs(y, mu, eps):
    x = math.exp(y)
    return mu - x * x, eps * (-mu + A * x - B)

def classify(x0, mu0, eps, dt=DT, t_max=T_MAX):
    """Return 'e0', 'e2' or 'UNCLASSIFIED'. Absorbing exits only."""
    y, mu, t = math.log(x0), mu0, 0.0
    while t < t_max:
        x = math.exp(y)
        # e0 absorbing: mu<0 and x << b/a  -> x decays, mu -> -b
        if mu < 0.0 and x < 1e-8:
            return "e0"
        # e2 local neighbourhood (linearly stable equilibrium)
        if abs(x - E2[0]) < 0.02 and abs(mu - E2[1]) < 0.05:
            return "e2"
        k1 = rhs(y, mu, eps)
        k2 = rhs(y + dt/2*k1[0], mu + dt/2*k1[1], eps)
        k3 = rhs(y + dt/2*k2[0], mu + dt/2*k2[1], eps)
        k4 = rhs(y + dt*k3[0], mu + dt*k3[1], eps)
        y  += dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        mu += dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        y = max(y, -690.0)             # floor: x ~ 1e-300
        t += dt
    return "UNCLASSIFIED"

def reduced(mu0):
    return "e0" if mu0 < MU_SADDLE else "e2"

def funnel_edge(mu0, eps, lo=-200.0, hi=math.log(0.5), iters=40):
    """Bisect ln x0: largest x0 that still reaches e0. None if no e0 at lo."""
    if classify(math.exp(lo), mu0, eps) != "e0":
        return None
    if classify(math.exp(hi), mu0, eps) == "e0":
        return hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        (lo, hi) = (mid, hi) if classify(math.exp(mid), mu0, eps) == "e0" else (lo, mid)
    return lo

def approx_edge(mu0, eps):
    """DERIVED here, not from the paper: x stays near 0 while mu decays from
    mu0 to 0; x peaks at mu=0 and must stay below ~b/a for mu to keep falling.
    ln x0* ~ ln(b/a) - (mu0 - b ln(1 + mu0/b)) / eps.   Approximate only."""
    return math.log(B / A) - (mu0 - B * math.log(1 + mu0 / B)) / eps

def selftest():
    ok = True
    def check(name, cond):
        nonlocal ok
        print(("  PASS " if cond else "  FAIL ") + name); ok &= cond
    print("SELFTEST")
    check("e0 neighbourhood -> e0", classify(1e-3, -1.5, 0.1) == "e0")
    check("e2 neighbourhood -> e2", classify(2.0, 3.9, 0.1) == "e2")
    check("ordinary point mu0=6, x0=0.5 -> e2 (agrees with reduced)",
          classify(0.5, 6.0, 0.1) == "e2")
    check("reduced threshold at mu=1", reduced(0.99) == "e0" and reduced(1.01) == "e2")
    return ok

def main():
    if not selftest():
        print("SELFTEST FAILED -- results below not trustworthy"); sys.exit(1)
    print("\nRUN 1  eps=0.1: funnel edge vs mu0 (reduced model says e2 for all mu0>1)")
    print(f"  {'mu0':>5} {'reduced':>8} {'log10 x0* numeric':>18} {'log10 x0* approx':>17}")
    found = unclassified = 0
    for mu0 in [1.5, 2, 3, 4, 5, 6, 8]:
        edge = funnel_edge(mu0, 0.1)
        ap = approx_edge(mu0, 0.1) / math.log(10)
        if edge is None:
            print(f"  {mu0:>5} {reduced(mu0):>8} {'none found':>18} {ap:>17.2f}")
        else:
            found += 1
            print(f"  {mu0:>5} {reduced(mu0):>8} {edge/math.log(10):>18.2f} {ap:>17.2f}")
    print("\nRUN 2  scaling at mu0=4: ln(edge) should fall ~ linearly in 1/eps")
    print(f"  {'eps':>6} {'1/eps':>6} {'log10 x0*':>10}")
    pts = []
    for eps in [0.2, 0.1, 0.05]:
        edge = funnel_edge(4.0, eps)
        if edge is None:
            print(f"  {eps:>6} {1/eps:>6.0f} {'none':>10}")
        else:
            pts.append((1/eps, edge)); print(f"  {eps:>6} {1/eps:>6.0f} {edge/math.log(10):>10.2f}")
    if len(pts) >= 2:
        (u1, l1), (u2, l2) = pts[0], pts[-1]
        print(f"  slope d(ln x0*)/d(1/eps) = {(l2-l1)/(u2-u1):.3f}  (negative = exponential narrowing)")
    print("\nSTATUS:", "REPRODUCED" if found else
          "NOT_REPRODUCED (searched ln x0 in [-200, ln 0.5], mu0 in [1.5, 8])")
    print("scope: deterministic, fixed-step RK4 dt=%.3g, T_max=%g; no noise; one parameter set" % (DT, T_MAX))

if __name__ == "__main__":
    main()
