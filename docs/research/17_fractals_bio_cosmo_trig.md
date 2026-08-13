# Notes 17 — Creative exploration: fractals, biological solution search, cosmology, trigonometry
2026-08-14 · four parallel research briefs + one new sim, synthesized
Full briefs: subagent workspaces (fractal, bio, cosmology, trig). Sim: sims/fractal_basin_sim.py, figures/fractal_basins.png

## 0. The headline: one theorem ties all four domains together

The fold normal form we've been living in (k_eff ∝ √(1−c/c_snap), h* = 2/√27) is not
*analogous to* results in these four fields — it is *the same mathematics*:

- **Cosmology**: Jeans gravitational-instability growth time τ ∝ (k_J² − k²)^(−1/2) —
  literally the same square-root divergence as our recovery time. Caustics of cosmic
  structure formation are Thom-classified folds/cusps/swallowtails
  (Arnold–Shandarin–Zel'dovich 1982, theorem-grade). The QCD phase diagram in (T, μ_B)
  is a conjectured cusp with a hunted critical endpoint.
- **Fractals**: at the fold, the basin boundary's uncertainty exponent α and the
  stiffness collapse are two readings of the same saddle-node geometry (measured below).
- **Biology**: convergent evolution's "few paths, many arrivals" (C4: >60 origins, ~4
  trajectories) is what a solution space organized around a small set of normal-form
  attractors *looks like*.
- **Trigonometry**: the fold's structural stability (codimension-1 = generic) is why the
  octahedron's symmetry group O_h can classify failure channels by irrep — and why the
  first octahedral-invariant harmonic appears only at ℓ=4 (Bethe selection rule).

The RG↔catastrophe bridge (Goldenfeld's framing) says it precisely: **folds are
structurally stable (geometry is universal); exponents are renormalized (numbers are
not).** That split is the license for the whole Rosetta program — and its boundary.

## 1. Fractal geometry — verified core + new measurement

Verified: uncertainty exponent f(ε) ~ ε^α, α = D − D_boundary (Grebogi/McDonald/Ott/
Yorke 1983–85); Wada property (Kennedy–Yorke 1991; grid test Daza et al. 2015);
riddled basins → formal uncomputability (Sommerer–Ott 1996); basin entropy
(Daza et al. 2016); Mandelbrot boundary has Hausdorff dimension 2 (Shishikura 1998);
Feigenbaum δ = 4.669 as RG fixed point.

**New sim result (fractal_basin_sim.py, our own double/triple well):**
- Bistable (the physical strut's potential): **α = 0.69** → boundary dimension 1.31.
  The boundary is *already fractal* at moderate damping (chaotic saddle on the barrier).
- Tristable: **α = 0.39**, D_b = 1.61; **8% of boundary cells are Wada** (touch all
  three basins) — partial Wada at γ=0.25 damping.
- Physical reading: α is the exchange rate between measurement precision and outcome
  certainty. α=1: 2× better instrument → 2× certainty. α=0.39: 2× better instrument →
  1.31× certainty. **Near the snap region of our instrument there is a precision floor
  that no better accelerometer can cross.** This is a *design constraint*, not a nuisance.

## 2. Biological solution search — verified core

Verified: C4 photosynthesis >60 independent origins but ~4 acquisition paths (Sage 2011;
Williams 2013 eLife); electric organs 6× with shared genetic module (Gallant 2014
Science); carcinisation ≥5×; Prestin: >25 identical amino-acid replacements in
bats/dolphins — "very limited ways for a mammal to hear high-frequency sounds"
(Liu/Li 2010); neutral networks (Schuster–Fontana 1994); arrival bias/simplicity bias
(Dingle, Schaper & Louis 2015); chemotaxis = Barkai–Leibler integral feedback (already
the ecosystem's calibration substrate — now confirmed as *evolution's own search
algorithm*); affinity maturation = Darwinian search in days with public clonotypes
recurring across donors (convergence within individuals).
Contested (flagged): genome-wide convergence statistics; robustness→evolvability in
proteins; Gould vs Conway Morris.

## 3. Cosmology — verified core

Verified: EW and QCD transitions are crossovers at physical parameters, with
critical endpoints (EW: m_H≈72–80 GeV endpoint, Kajantie 1996/Csikor 1999; QCD
endpoint conjectured, actively hunted); Coleman thin-wall bubble nucleation = Kramers
escape, B ~ ε⁻³; Zel'dovich–Arnold caustic classification (folds=pancakes,
cusps=filaments, umbilics=nodes; Hidding et al. 2014 revival); Press–Schechter =
first-passage barrier statistics with analytic ground truth; hybrid-inflation waterfall
is a genuine bifurcation.
Contested: QCD critical endpoint; DESI evolving dark energy (2–3σ, disputed); Hubble
tension.

## 4. Trigonometry — verified core

Verified/recomputed: Descartes angular defect Σδ = 2πχ = 4π (discrete Gauss–Bonnet;
octahedron per-vertex defect 2π/3 ✓); exact dihedral angles recomputed; octahedron
Gram spectrum {2,2,2,0,0,0} — rotation-invariant shape fingerprint; addition formulas =
circle group law; Chebyshev = SU(2) characters; spherical harmonics = SO(3) irreps;
**O_h selection rule: no A1g (octahedral-invariant) content for ℓ=1,2,3; first at ℓ=4**;
Majorana constellations: platonic solids are literally quantum states (octahedral
"kings of quantumness"); hyperbolic rapidity addition; arcsine/Chebyshev–Beta(½,½)
conjugacy — which yields an **exact Beta-family correction to our VSA bundling law
Φ(1/√(k−1))**.

## 5. Cross-domain intersections (the creative payload)

**X1. Uncertainty exponent as a new instrument channel (E-P7 candidate).**
Claim: α(c) — the fractal exponent of the snap/no-snap basin boundary in (flick
position, flick strength) space — collapses toward 0 as c → c_snap, *independently* of
the stiffness signal. Test: on the physical instrument, Monte-Carlo flick outcomes at
fixed compression; fit f(ε,c); compare lead time vs recovery-time channel.
Refuted if: α(c) is flat across the load range. Our sim says α<1 already exists at
baseline — the sweep is the experiment. Cheap: needs only the flick pad + phyphox.

**X2. Cusp as structure detector, not just failure detector (cosmology inversion).**
Arnold–Zel'dovich classifies caustics = structures by catastrophe type. Our cascade
audit asks "is failure near?"; the same normal-form fitting can ask "**what class of
structure is forming?**" Fold→sheet, cusp→filament, umbilic→node. Any spatial field in
the ecosystem (sensor grids, attention maps, GM basin atlases) gets a caustic census.
Refuted if: fitted catastrophe types don't predict subsequent structural features
better than chance.

**X3. Convergent evolution as the existence proof for Rosetta shapes.**
C4's ">60 origins, ~4 paths" is what a funnelled solution space looks like from inside.
If equation/constant complexes really cluster into a few normal-form attractors, then
independent domains arriving at the same complex (our 30-domain equation atlas) should
show C4-style statistics: many arrivals, few paths. Test: path-classify the equation
atlas entries by derivation route (not just final form); count distinct routes per
complex. Refuted if: routes are as diverse as arrivals (no funnelling) — which would
weaken the Rosetta compression claim.

**X4. Basin repair = neutral-network navigation (bio × GM).**
Evolution navigates between solutions along neutral networks without fitness loss.
GM's repair navigates within a KL-basin without accuracy loss — same topology.
Prediction: successful repairs should trace *connected paths* in parameter space (you
can interpolate between pre- and post-repair weights without crossing a loss barrier).
Test: linear/mode connectivity check between checkpoint and repaired weights.
Refuted if: repairs that work are not path-connected (then repair is teleportation, not
navigation — and GM's stability claims need the ISS proof, not the navigation metaphor).

**X5. Descartes defect as a conserved checksum for shape diagnostics.**
Σδ_v = 4π always. Any measured distortion of the octahedral instrument must redistribute
defect, not create it. So: (a) a measurement that violates Σδ=4π is a *sensor fault*,
detected for free; (b) damage localizes as a **defect dipole** (defect stolen from one
vertex appears at another). Drill-down gains a conservation law — the shape version of
sin²+cos²=1. Test: sim — inject vertex damage, compute defect field, check dipole
signature vs the 6× vertex-localization result from notes/14.

**X6. O_h ℓ=4 selection rule as a distortion fingerprint.**
Octahedral symmetry forbids invariant angular content below ℓ=4. A distortion's
spherical-harmonic spectrum should therefore show characteristic low-ℓ leakage patterns
per failure channel (irrep), and the ℓ=4 invariant amplitude measures *total departure
from octahedrality* in one scalar. This upgrades drill-down from vertex-localization
(notes/14: 6× concentration) to a full angular spectrum — the shape's Fourier transform.
Test: decompose sim distortions from notes/14 into Y_ℓm; check irrep-spectrum
separation of the known failure channels.

**X7. Prestin effect for repair algorithms (bio × GM).**
Bats and dolphins independently reused >25 identical Prestin substitutions — solution
space for high-frequency hearing is tiny. Analog: do independent repair runs (different
seeds/data subsets) converge on the *same weight sites*? If yes, repair space is
funnelled and a repair-site atlas is compressible; if no, repair is idiosyncratic.
Test: N repair runs on the same damaged model; measure site overlap (Jaccard).
Either outcome is informative — a true both-ways result.

**X8. Crossover shoulder = bounded flickering (cosmology × EWS).**
At a crossover (EW/QCD at physical parameters) there is no fold — but EWS literature
and our own flickering channel predict a *shoulder*: variance rises and falls without
divergence. This gives the cascade audit a third regime classification (fold / crossover
/ smooth) detectable from the same six signals, with the cusp's (a,b) control-plane
position as the classifier. Test: synthetic cusp normal form with cubic term swept
through crossover; check the audit classifies correctly. Refuted if: crossover and fold
are indistinguishable at realistic noise.

**X9. Jeans law as cross-validation of the fold exponent.**
τ ∝ (k_J²−k²)^(−1/2) is the same exponent-1/2 divergence in a field with 400 years of
independent evidence. If our instrument measures an exponent ≠ 1/2 robustly, that's
not "our instrument is weird" — it's either renormalized exponents (RG caveat,
fractal agent §5) or a misidentified bifurcation. Either way the Jeans anchor turns the
instrument's exponent measurement into a real contribution.

**X10. VSA bundling law exact correction (trig × notes/12 S3).**
Chebyshev–Beta(½,½) conjugacy gives an exact finite-d correction to bit-acc =
Φ(1/√(k−1)). S3 measured exact match at d=1024; the Beta correction predicts where it
*departs* at small d — a pre-registered deviation curve for a future sweep.

## 6. Standing contested-claims ledger (all four domains)
- WBE 3/4 metabolic scaling universality — contested (Glazier 2005; Banavar 1999
  derives 3/4 without fractals)
- Brain criticality — "near-critical" is the honest claim (Beggs–Timme 2012)
- Cosmological homogeneity scale — not closed (2025 DESI conditional-density minority
  result vs Scrimgeour 2012)
- Genome-wide molecular convergence statistics — contested (Parker 2013 critiques)
- QCD critical endpoint — conjectured, actively hunted
- O_h ℓ-decomposition table — standard theory, no live citation retrieved this session

## 7. What this changes in the ecosystem
1. **E-P7 added to the instrument protocol**: uncertainty-exponent sweep (X1) — the
   fractal channel is the cheapest addition with the most novel output.
2. **Cascade audit gains regime classification**: fold/crossover/smooth (X8).
3. **Drill-down gains two new layers**: Descartes checksum (X5) and angular spectrum
   (X6) — both sim-testable today against notes/14 data.
4. **GM gets two honest tests**: path-connectivity of repair (X4), repair-site
   convergence (X7) — both outcomes informative, neither assumable.
5. **The four-brief meta-observation**: each domain independently arrived at the same
   saddle-node/fold machinery. That *is* the convergent-evolution pattern (X3) acting
   on mathematics itself — the Rosetta hypothesis observing its own mechanism.
