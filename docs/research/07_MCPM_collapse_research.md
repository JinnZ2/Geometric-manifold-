# Notes 07 — Research Grounding for MCPM (Mathematical-collapse-prevention-model)

Date: 2026-08-13. Repo: github.com/JinnZ2/Mathematical-collapse-prevention-model.
Core metric: **M(S) = (R_e × A × D × f(C)) − L**, Value = M(S)/Energy_Cost. f(C) = exp(−α‖C − C*‖²_F), C* = I_n/φ. Verdicts: BLACK (any of R_e/A/D ≤ 0 — irreversible), RED (M<0 or ttc≤1), AMBER (ttc≤5 or degrading & M<0.5), GREEN.

## 1. Repo state (from deep read)

- All four M(S) inputs are **raw user-supplied floats in [0,1]** — no calibration or measurement pipeline exists. README's `EnergyIntegration` is phantom; docs links are phantom; install points at nonexistent repo.
- Four standalone audit subsystems (business/dependency/premise/substrate) don't consume M(S).
- Sharpest existing pieces: BLACK = multiplicative zero (irreversibility); drag ratio L/A > 1; discretionary-effort leading indicator (6–18 mo ahead of turnover); premise fragility = confidence × (1−evidence); brittleness = import_fraction × log(1+replacement_days) × SPOF-multiplier; golden-ratio trust chambers; verdict layer borrowed from metabolic-accounting.

## 2. Calibration sources per M(S) term (the missing measurement pipelines)

**R_e (resonance/constructive energy):**
- Aerobic scope AS = MMR − SMR → 0 at T_crit (OCLTT; Pörtner) — usable-energy headroom with collapse at zero.
- ATP % of baseline: <15–25% switches death mode (Lieberthal 1998, PMID 9486226) — hard energy floor.
- HRO "preoccupation with failure" as maintained feedback resonance (Weick & Sutcliffe).
- The 2003 blackout's stalled alarm process = R_e collapse converting survivable N-1 into 61.8 GW cascade.

**A (adaptability/recovery rate):**
- Critical slowing down: recovery rate |λ|→0 at tipping; lag-1 autocorrelation α→1, variance σ²/(1−α²) rises (Scheffer 2009; Dakos 2024 meta: 67.8% detection rate). Return time T_r ≈ 1/|λ|.
- Measurable decades pre-collapse: Amazon VOD AR(1) rising across >75% of basin (Boulton 2022); AMOC EWS tipping estimate 2057 [2025–2095] (Ditlevsen 2023); physical indicator F_ovS leads ~25 yr (van Westen 2024).
- Hormesis caps adaptive gain at 1.3–1.6× baseline (Calabrese) — A is bounded plasticity, don't let it grow unbounded in-model.
- Gray-wave: A(t) ≈ A₀·0.9^t from 10%/yr maintainer retirement; proteostasis: A deliberately down-regulated at reproductive maturity (Labbadia & Morimoto 2015) — timed collapse dataset.

**D (diversity):**
- *Response* diversity, not richness (Elmqvist 2003); stability = φ·mean_stability/synchrony (Loreau & de Mazancourt 2013).
- Model collapse (Shumailov 2024 Nature): tails vanish first, σ²→0 recursively; Dohmatob 2024: phase transition, ~1% synthetic contamination can trigger; Gerstgrasser 2024: **accumulate (keep real data) → collapse avoided** — A sustains D against L.
- Polycentricity = institutional D feeding A (Ostrom); cascade fragility in belief systems (Bikhchandani 1992; MusicLab: identical systems diverge under social coupling).

**f(C) (coupling, optimal at intermediate):**
- May's stability ceiling: σ√(SC) < d — hard quantitative bound on coupling.
- Buldyrev 2010 Nature: interdependent nets collapse abruptly at p_c = 2.4554/⟨k⟩ vs 1/⟨k⟩ isolated; Parshani: partial coupling q interpolates — first-order jump means no early-warning gradient from L alone.
- Kauffman NK: evolvability peaks ~K=2; Ethiraj & Levinthal 2004; Orton & Weick 1990 — the interior-optimum canon for f(C).
- Wunderling 2023: tipping interactions add +49% tipped elements; overshoot raises risk +72% — coupling matrix between subsystems, measured.
- Strangler-fig migration beats big-bang ~4× (TSB 2018 as hard anchor) — preserving operational feedback (R_e) during transition.

**L (loss/entropy rate):**
- Planetary boundaries: transgression distance as rising-risk rate (Richardson 2023: 6/9 transgressed).
- Allostatic load index — composite multi-system dysregulation predicting collapse before any single variable crosses (template for L as integral).
- Eigen error threshold: q* = ln(σ)/L_genome — the analytically solvable M(S)=0 case.
- Deferred maintenance as compounding interest; bathtub/Weibull β>1 wear-out.
- Link rot 13–22% within 2 yr; knowledge half-life λ = ln2/t½ (field-specific, 7–13 yr monograph).
- Replication crisis: 36% (psychology OSC 2015), ~21% (preclinical Begley & Ellis) — audited false-content fraction of stored knowledge.
- Vosoughi 2018 Science: falsehood spreads 1.7× faster, 6× slower truth — measured asymmetric R0; epistemic L grows superlinearly with coupling unless filtered.

**Value = M(S)/energy:**
- EROEI ladder (Hall): 3:1 bare civilization, ~10:1 for modern services — empirical denominator thresholds.
- Tainter: collapse when marginal ROC = ΔE/ΔS < 0 — the *derivative* of the value ratio; 2024 formalization in Entropy (PMC11154394).
- Turchin PSI = MMP×EMP×SFD — multiplicative structural template; note the counter-test (PMC10621949): only as good as independently measured inputs.

## 3. Structural upgrades the literature demands

1. **Path-dependence & rate-dependence.** Hysteresis (AMOC, ice sheets) and R-tipping (Ashwin 2012) show collapse can occur at M(S)>0 if forcing outpaces A. Add d(forcing)/dt term and history to M(S); static M(S)=0 underestimates overshoot risk.
2. **A as observable early-warning.** AR(1)/variance on any monitored series is the best-calibrated generic EWS (67.8% detection). This is implementable stdlib-only — the repo's biggest cheap win.
3. **First-order transitions.** Coupled systems (Buldyrev) jump discontinuously — L gives no gradient near threshold; the multiplicative coherence terms *are* the leading indicators. Keep them; add abrupt-jump flag when interdependency detected.
4. **D must be response-diversity.** Raw counts are wrong; measure variance of responses across the stress axis.
5. **faIR-style operationalization precedent.** FAIR principles + FAIRness indices prove "coherence" can be audited per-field without being optimized — the measurement-not-control stance has a mature working exemplar.
6. **Goodhart warrant.** Manheim & Garrabrant (arXiv:1803.04585) four-category taxonomy + Campbell's law (indicator corrupts the process itself, not just the measurement) = the formal citations for "measure, never target." The repo's ethical stance is the mainstream conclusion of 90 years of literature (Merton 1936 → Deming → Austin → Krakovna).

## 4. Proposed next steps for the repo

| Priority | Item | Effort |
|---|---|---|
| P0 | AR(1)/variance EWS module on M(S) history + any monitored series (stdlib, ~100 lines) | days |
| P0 | Wire audit subsystems to actually consume/produce M(S) terms | days |
| P1 | Calibration adapters: each literature source above as a named, cited derivation of R_e/A/D/L from real data (e.g. `A_from_timeseries()`, `D_response_diversity()`, `L_linkrot_rate()`) | 1–2 wks |
| P1 | Rate-term: track dM/dt and forcing rate; R-tipping flag when forcing > A | days |
| P1 | Fix phantom docs/README (EnergyIntegration, install, license reconciliation) | hours |
| P2 | Ostrom design-principles scorer for governance systems (8 principles → f(C) integrity) | 1 wk |
| P2 | EROEI/ROC layer formalizing the value denominator with Tainter marginal-collapse detector (dM/dcost < 0) | 1 wk |
| P2 | Uncertainty on M(S): propagate input intervals (literature gives ranges, e.g. AMOC 1.4–8 °C) instead of point floats | 1 wk |
| P3 | Model-collapse case study as flagship example: accumulate-vs-replace sim showing D decay and L compounding — the one domain where M(S)'s mechanism is exactly provable | days |

## 5. Honest grounding gaps flagged by the research

- No formal "corruption as entropy" canon (inference only); no canonical quantitative schema-drift study; misinformation R0 relies on standard SIS/SIR theory; CHAOS figures contested (Eveleens & Verhoef 2010) — use Flyvbjerg's n=1,471 dataset as the hard anchor; coupling-optimum org-theory cites are canonical but were not re-verified page-level.
