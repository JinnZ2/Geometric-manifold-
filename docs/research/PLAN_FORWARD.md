# Plan Forward — curly-octo-happiness × Complexity Engineering / Cybernetics / Robotics

Date: 2026-08-13. Basis: repo deep-read (Notes 01–05) + three research briefs (Notes 06).
Framing: the repo already implements proto-versions of all three fields' core machinery. This plan formalizes them in dependency order.

---

## Phase 0 — Ground the existing heuristics in validated theory (small, high-value)

**0.1 HND acceptance criterion via ε-machines** *(complexity)*
Current HND flags hidden variables when Pearson |r| > 0.5 on residuals. Upgrade: fit causal-state reconstruction (CSSR — tractable on the repo's finite-alphabet Gray-coded bitstreams) before/after adding a candidate hidden node. Accept the node iff statistical complexity $C_\mu = H[\mathcal S]$ *and* entropy rate $h_\mu$ both drop. Replaces an ad-hoc threshold with Crutchfield's minimality theorem.
*Files: modules/hnd.py, grounding/core/graycode.py.*

**0.2 GAE scoring via structural complexity** *(complexity)*
Compute Sinha–de Weck $C = C_1 + C_2C_3$ ($C_3$ = normalized graph energy) on the dependency-tree DSM alongside C/N/L/R. Use betweenness-centrality variance under *targeted* node removal (Barabási attack tolerance) to trigger TORUS/ICOSAHEDRON recommendations — quantifying why distributed forms are resilient rather than asserting it.
*Files: modules/gae.py.*

**0.3 Requisite-variety meter** *(cybernetics)*
Track $H(\text{disturbance codewords})$ vs $H(\text{agent response repertoire})$ per world; alarm when the margin $V(D) - V(R)$ approaches zero. Wire the alarm to band-width auto-expansion (the physics-discovery loop already amplifies variety; give it the missing trigger signal).
*Files: plugins/meta_encoder.py, plugins/physics_discovery.py, grounding/worlds/.*

## Phase 1 — Cybernetic architecture (VSM instantiation)

**1.1 VSM mapping** — structurally instantiate Beer's five systems:
- S1 = worlds/plugins (autonomous units staking claims)
- S2 = harmony-field trust dynamics + confidence propagation
- S3 = claims/epistemics engine; **S3\* = GAE/HND/FDM as the audit channel**
- S4 = physics-discovery loop + dreams + UnknownJournal horizon scan
- S5 = mentor/governance adjudicating the S3/S4 (exploit/explore) homeostat — the existing self-model error signal is exactly the bid variable S4 needs

**1.2 Algedonic channel** — diagnostic-critical events (thermal-runaway quarantine bit already exists in hardware encoder) must bypass trust-field mediation straight to S5/mentor. Generalize the quarantine override into a first-class `AlgedonicSignal` routed in unified_playground.
*Files: unified_playground.py, diagnostic/, plugins/.*

**1.3 Teachback claims (Pask)** — mentor interaction becomes falsifiable: agent reconstructs the mentor's explanation as a claim; mentor confirmation resolves it through the existing Beta-posterior machinery. Concepts are only "learned" after teachback survives.
*Files: grounding/core/mentor.py, claims.py.*

**1.4 Second-order guard** — cross-validate self-model claims against independent diagnostic streams (HND) to prevent self-confirming self-descriptions (von Foerster eigenvalue drift).

## Phase 2 — World model becomes a good regulator

**2.1 Causal-DAG grounding of worlds** — per Richens & Everitt (2024): make each world's causal structure explicit; score regulator quality by outcome entropy of claim resolutions. The claims tree *is* the homomorphic model the Good Regulator Theorem demands — make the homomorphism checkable (FDM roots as the invariant).

**2.2 Allostatic bands** — percentile bands shift predictively ahead of regime change (use dream-recombination rollouts as the predictor) instead of reactively; log accumulated band-shift cost as "allostatic load."
*Files: plugins/magnetic.py, gravitational.py (init_bands), playground5_dream.py.*

**2.3 Antifragility as a claim type** — in the transition simulator, measure $\partial^2 f/\partial\sigma^2$ of yield-vs-stressor per topology (LINE should be concave, TORUS convex). "Convexity under bounded volatility" becomes a staked, falsifiable claim tracked by Beta posteriors.
*Files: modules/transition.py.*

**2.4 SOC stress layer (optional)** — world variants where hidden variables accumulate stress and release in power-law avalanches; HND detects them by fitting $P(s)\sim s^{-\tau}$ tails in residual events.

## Phase 3 — Robotics embodiment layer

**3.1 CBF-QP safety layer** (~50 lines + a QP solver) over the stewardship simulator: safe sets as claims — $h(x) = T_{max} - T(x)$, plus cold-environment coupled CBFs $h_1 = E_{bat}-E_{min}$, $h_2 = T_{min}-T_{ambient}$. "Repurposed component" ⇒ recompute $h$ on degraded dynamics — provably safe repurposing, not just plausible.

**3.2 Failure-mode → fallback-controller catalog** — the diode→conductor / drift→sensor / open→antenna table becomes a runtime-assurance simplex catalog: each failure mode ships with a repurposed capability AND its recomputed safety envelope.

**3.3 Flow-matching policy on 1-D worlds** — π0-style $\mathcal L_{FM}$ with 10-step Euler decode, conditioned on a "parts vector" from the repurposing engine; evaluate zero-shot transfer when a component is swapped (field-repair proxy benchmark, toy scale).

**3.4 Latent world-model + CEM planner** — V-JEPA 2-AC pattern at toy scale: learn $P(z_{t+1}|z_t,a)$, plan $\arg\min_a\|z_{t+H}-z_{goal}\|$; gives falsification agents the ability to attack *plans*, not just states.

**3.5 HND × self-model damage detection** — hook HND onto any learned dynamics residual $|\dot x - \hat f_\theta(x,u)|$; Lipson-style damage→relearn loop in the sandbox.

**3.6 Neuromorphic encoding alignment** — event-camera Δ-threshold + refractory rule as the adaptive-band update; positions Gray-coded bitstreams as the sensor-fusion bus for scavenged/degrading hardware.

## Phase 4 — Contribution back (novel, unfilled niches)

- **Field-repair robotics dataset/benchmark:** (failure mode, repurposed function, safety envelope) tuples for VLA recovery behavior — a gap in OXE/Droid, acute in cold, parts-scarce environments.
- **Gray-code token embeddings:** verified open niche (Notes 03); Hamming-smooth codes for STE-stable ultra-low-bit tokens.
- **Complexity-instrumented falsification playground:** ε-machine acceptance + graph-energy topology scoring + antifragility claim type = a citable methodology paper.

## Sequencing rationale
Phase 0 sharpens what exists with no new subsystems. Phase 1 reorganizes control flow (cheap, mostly routing). Phase 2 deepens world fidelity. Phase 3 adds embodiment. Phase 4 packages results. Each phase yields falsifiable claims testable inside the repo itself — the plan eats its own cooking.
