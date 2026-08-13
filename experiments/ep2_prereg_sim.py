# ============================================================================
# SUPERSEDED — this is v2, which the ecosystem's own pre-registration arc
# records as REFUTED. Do not cite its output as a result.
#
# Per docs/research/15_physical_shape_instrument.md, "Pre-registration arc":
#   v1 (absolute t-test, sequential scan)         REFUTED — null arm fired at 96%
#   v2 (differential ratio, sequential scan)      REFUTED — null arm fired at 82%
#   v3 (differential ratio, ONE pre-committed checkpoint at compression 0.44)
#                                                 SUPPORTED — 5/5 seeds, null FP 0.05
#
# What this script actually is: the single-arm version. It has NO null arm and NO
# rigid-strut control, so its "PASS (predicted)" line is not evidence — it is the
# uncontrolled measurement whose failure motivated v2 and v3. Running it prints
# detection in 200/200 trials and a 48.3% lead precisely because nothing here can
# fire a false positive. Sequential scanning over 15 compression steps for any
# t > 1.86 is the multiple-comparisons machine that v3 exists to eliminate.
#
# v3 itself was not provided with this file. Retrofitting to
# docs/research/HARNESS.md (item 1 in its retrofit queue) means: two arms
# (bistable + identical rigid), one pre-committed checkpoint, >= 5 seeds, a named
# null model, and a self-graded SUPPORTED/REFUTED/INCONCLUSIVE verdict.
#
# Kept in the repo as the provenance of a refutation, not as a working experiment.
# ============================================================================
# E-P2 pre-registration simulator (v2): fold normal-form stiffness law.
# Near a saddle-node fold, effective stiffness k_eff ~ sqrt(1 - c/c_snap)
# (mean-field exponent 1/2). Probe-flick recovery time tau ~ 1/k_eff.
# Matches shape_csd_probes.py: tau 70 -> 600+ steps as compression -> snap.
import numpy as np

SNAP = 0.495          # measured in shape_csd_probes.py
TAU0 = 70.0           # recovery time at baseline compression 0.30 (calibrated)
C0   = 0.30

def tau_true(comp, creep=0.0):
    k_rel = np.sqrt(max(1e-9, 1.0 - comp/SNAP)) * (1.0 - creep)
    return TAU0 * np.sqrt(1.0 - C0/SNAP) / k_rel

def run_trial(seed):
    r = np.random.default_rng(seed)
    comps = np.arange(0.30, 0.4949, 0.01)
    taus = []
    for i, c in enumerate(comps):
        t = tau_true(c, creep=0.004*i)          # PETG creep 0.4%/step, dwell<60 s
        taus.append((t*(1+0.05*r.standard_normal(5))).mean())
    taus = np.array(taus)
    base = taus[:5]
    for i in range(5, len(comps)):
        cur = taus[i]*(1+0.05*r.standard_normal(5))
        tstat = (cur.mean()-base.mean())/np.sqrt(cur.var(ddof=1)/5+base.var(ddof=1)/5)
        if tstat > 1.86:
            return comps[i]
    return None

det = [d for d in (run_trial(s) for s in range(200)) if d is not None]
print(f"detection in {len(det)}/200 trials")
print(f"median first detection: compression {np.median(det):.3f}")
print(f"expected lead before snap: {(SNAP-np.median(det))/(0.60-0.30)*100:.1f}% of load range")
print(f"10-90 pct detection: {np.percentile(det,10):.3f} - {np.percentile(det,90):.3f}")
print(f"refutation check: lead >= 15% required -> {'PASS (predicted)' if (SNAP-np.median(det))/(0.30) >= 0.15 else 'FAIL at 5% noise / 0.4% creep'}")

print("\ncompression  tau_true  (no-creep / with-creep)")
for c in [0.30,0.35,0.40,0.42,0.44,0.46,0.48,0.49]:
    i = int(round((c-0.30)/0.01))
    print(f"  {c:.2f}       {tau_true(c):7.1f} / {tau_true(c,0.004*i):7.1f}")

# sensitivity: what noise level preserves the 15% lead?
for noise in [0.02,0.05,0.10,0.15]:
    leads=[]
    for s in range(100):
        r=np.random.default_rng(1000+s)
        comps=np.arange(0.30,0.4949,0.01)
        taus=[]
        for i,c in enumerate(comps):
            taus.append((tau_true(c,0.004*i)*(1+noise*r.standard_normal(5))).mean())
        taus=np.array(taus)
        base=taus[:5]
        d=None
        for i in range(5,len(comps)):
            cur=taus[i]*(1+noise*r.standard_normal(5))
            t=(cur.mean()-base.mean())/np.sqrt(cur.var(ddof=1)/5+base.var(ddof=1)/5)
            if t>1.86:
                d=comps[i]
                break
        if d:
            leads.append((SNAP-d)/0.30)
    print(f"noise {noise:.0%}: median lead {np.median(leads)*100:.1f}%  (detection {len(leads)}/100)")
