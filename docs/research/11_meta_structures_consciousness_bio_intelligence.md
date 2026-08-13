# 11 — Meta-Structures · Parallel Processing · Geometric Overlays · Consciousness · Biological Intelligences

Research date: 2026-08-13. Flag legend: **[E]** established · **[C]** contested · **[S]** speculative.
Companion: integration/EXPLORE_AND_EXPERIMENT.md (pattern intersections + experiment list).

---

## Part 1 — Meta-structures & meta-meta integration

### 1.1 Learning structures that learn structures
- **MAML** (Finn 2017, 1703.03400): bi-level optimization min_θ Σ L_Ti(θ − α∇L_Ti(θ)) — outer loop searches the space of learning procedures.
- **Learned optimizers** (Andrychowicz 2016, 1606.04474; VeLO 2022): gradient descent on gradient descent itself.
- **In-context learning as meta-learning**: transformers provably implement gradient descent steps in forward pass (von Oswald 2023; Akyürek 2023); mesa-optimizer evidence is synthetic-task-only (PNAS 2024, 2309.05858) **[E in controlled settings; S for production LLMs]**.
- **Hypernetworks** (Ha 2017, 1609.09106): networks emitting network weights — the space of networks becomes a differentiable object.
- **Neural algorithmic reasoning** (Veličković & Blundell 2021): nets executing classical algorithms; sharp OOD degradation.

### 1.2 Mathematics of structure-on-structure
- **Category theory**: functor = map between structures; natural transformation = map between maps of structures (first true meta-structure); **pushout/colimit = the canonical gluing operation**.
- **Operads** (Spivak wiring diagrams, 1305.0297): operations whose operands are operations; systems-of-systems composition = nesting typed boxes. Applied to real systems engineering (Baez-Foley, 2009.12647).
- **Applied category successes**: DisCoCat compositional semantics; functorial passive linear networks / reaction networks (Baez & Fong 1504.05625; 1704.02051); double categories of open systems.
- **Institutions** (Goguen–Burstall 1992): "logic over logics" — (Sign, Sen, Mod, ⊨) with truth invariant under change of notation. Institution morphisms translate between logics (HETS/DOL).

### 1.3 Self-reference limits and the engineering dodges
- Tarski undefinability, Gödel II, Löb: **no system soundly asserts its own total correctness**. Dodges: (a) stratification (verifier simpler than verified; Type₀ : Type₁ universes); (b) bounded-fidelity self-models with calibrated error budgets (Tarski blocks exact truth, not approximate prediction); (c) explicit level tags (staging/MetaML); (d) **ledger anchoring — ground self-descriptions in an append-only external record** (natural fit: falsification ledger).

### 1.4 Physical meta-structure
- Metamaterials: effective-medium theory (structure ≪ λ → emergent ε_eff, μ_eff); mechanical metamaterials (auxetics, origami bistability, Kane–Lubensky topological floppy modes); bone/nacre 7-level hierarchies = emergent effective parameters at each scale — **the physical instance of Simon's near-decomposability** (1962: hierarchy is the condition for evolvability).
- Bond graphs are already a meta-structure: one port formalism instantiating across energy domains; functorial treatment makes compilation compositional (EPHS, 2402.17640).

### 1.5 Meta-meta integration patterns (implementable)
1. **Schema pushout**: integrate two schemas as colimit over an explicit interface schema.
2. **Functorial pivot**: n functors into a common semantic category, not n² translators (PROV event graph as pivot).
3. **Common-refinement search**: anti-unification / least-general-generalization (Plotkin 1970) when models disagree.
4. **Port contracts**: open systems + assume–guarantee at typed ports.
5. **Ledger-anchored stratification**: each level's claims sealed by hashes at the level below.

---

## Part 2 — Parallel processing

### 2.1 Paradigms with formal character (all admit stdlib toy implementations)
- **Kahn process networks**: blocking read / non-blocking write FIFO ⇒ **determinacy theorem** (output independent of schedule) — the gold standard for parallel sensor pipelines.
- **BSP** (Valiant 1990): cost model T = w + g·h + L — predictable enough to compile against.
- **CSP/π-calculus**: channels as compositional unit; deadlock-freedom via channel-rank orderings.
- **Cellular automata**: computation-universal (Rule 110, Cook 2004); systolic arrays → TPU ancestry.
- **Neuromorphic 2026 status**: Hala Point (1,152 Loihi 2, 1.15B neurons, ~2.6 kW); SpiNNaker2 Leipzig (~650M neurons); vendor claims (15+ TOPS/W) are company-reported, not independently benchmarked.

### 2.2 Compositional safety results
- **CALM theorem** (Hellerstein & Alvaro, CACM 2020): coordination-free consistency ⟺ monotonicity. Non-monotone ops (outlier rejection, resets) are exactly the coordination points. **Direct design rule for the ecosystem: monotone fusion ops (growing confidence sets, max-threshold alarms) can run coordination-free; non-monotone ops (calibration resets) must synchronize.**
- CRDTs = join-semilattice + monotone merge; Amdahl/Gustafson bounds; **model soups** (Wortsman 2022) — weight averaging valid only within a loss basin (linear mode connectivity; permutation alignment, Entezari 2022).

### 2.3 Geometry link
Parallel branches ≈ **atlas/chart decomposition** of a manifold (gating network = partition of unity); MoE = learned parallel decomposition; multi-head attention = parallel rank-limited channels additively overlaid.

---

## Part 3 — Geometric overlays

### 3.1 Machinery for multiple structures on one space
- **Fiber bundles** (gauge fields = connections; curvature F = dA + A∧A).
- **Cellular sheaves** (Curry thesis, 1303.3255): stalks on graph nodes/edges + restriction maps; **sheaf condition = formal local-to-global consistency**; sheaf Laplacian kernel = globally consistent states.
- **Sheaf neural networks** (Hansen & Gebhart 2020, 2012.06333; neural sheaf diffusion 2202.02479) — learn the restriction maps; solves heterophily/oversmoothing.
- **Multiphysics coupling**: operator splitting (Strang: e^{hA/2}e^{hB}e^{hA/2}), FMI co-simulation standard.

### 3.2 Bond graphs ARE a geometric overlay formalism [verified]
Port-Hamiltonian form ẋ = (J(x) − R(x))∂H + g(x)u with J skew-symmetric, R ≥ 0; **Dirac structures** (maximally isotropic subbundles under the power pairing ⟨e₁,f₂⟩+⟨e₂,f₁⟩) are the unified interconnection object; 0/1-junctions are discrete Dirac structures (van der Schaft & Jeltsema 2014, doi:10.1561/2600000002). One graph substrate, multiple energy-domain overlays, TF/GY as transition maps.

### 3.3 Overlays inside neural nets
- **Residual stream = shared additive bus** (Elhage et al., transformer-circuits framework 2021): layers write/read a common vector space.
- **Superposition** (Toy Models 2022, 2209.10652): n ≫ d features in almost-orthogonal directions (JL regime), sparsity tolerates interference.
- **SAEs** as the disentangling overlay: scaled to 34M features (Anthropic 2024); **2025–26 critique**: DeepMind deprioritized after SAEs underperformed cheap linear probes on some tasks; causal relevance and coverage remain open **[C]**.
- **VSA/hyperdimensional computing** (Kanerva 2009): binding + bundling algebra; capacity ~√d items before crosstalk — the cleanest formal overlay capacity result.

### 3.4 Physical overlay compute
Photonic matmul (Lightmatter Passage L20/L200 announced Mar 2026; Q.ANT NPU 2 first cloud-purchasable photonic accelerator; **no independent photonic-vs-GPU LLM benchmark exists**); FDTD acoustic/wave compute; DNA strand displacement; morphological computation (body-as-reservoir).

---

## Part 4 — Consciousness studies, 2024–2026

### 4.1 Theory standing (full survey: subagent file consciousness_survey_2024_2026.md)
- **Cogitate (Nature 642:133–142, 2025)** is the anchor event: IIT's sustained-posterior-synchronization prediction failed; GNW's offset-ignition failed, PFC decoding underperformed. Neither refuted, both weakened **[C]**. Relative beneficiaries: Local Recurrence, HOT.
- **IIT**: 2023 letter (124 signatories) called it pseudoscience; defenders (Tononi/Boly, Nat. Neurosci. 2025) cite 16 supportive studies. Do not use Φ as an instrument.
- **AST (attention schema)** is the most directly implementable theory.
- **Machine consciousness**: Butlin et al. 2023 indicator-properties framework (2308.08708) remains the reference; current LLMs fail most indicators (no within-token recurrence, no persistent workspace, no embodiment). 2025–26: Anthropic model-welfare program; Eleos assessments. Preprint introspection results **[S]**.
- **Recurrence is the convergence point** of RPT, IIT's unfolding argument, and Chalmers' X-factors.
- Honest program: **access-consciousness metrics + integration-structure measurement + falsification ledgers. The hard problem is untouched; nothing here detects consciousness.**

### 4.2 Transferable methodology
Pre-registered divergent predictions with kill criteria; theory-neutral marker batteries; proponent-co-design with independent execution; budget for Duhem–Quine auxiliary revisions. **These are exactly the falsification-ledger discipline — the ecosystem's epistemics already match the field's best practice.**

### 4.3 Markers computable cheaply
PCI/PCIst (Casali 2013): normalized Lempel–Ziv of binarized perturbation response; clinical cutoff ~0.31, validated across sleep/anesthesia/DoC **[E clinically; theory-ambiguous]**. Computable from time series in stdlib.

---

## Part 5 — Biological intelligences (compute paradigm per system)

| System | Compute paradigm | Key equation/model | Status flags |
|---|---|---|---|
| **Physarum** | flow-adaptive graph optimization; stigmergic external memory | **Tero law: dD/dt = \|Q\|^μ − γD**, Q ∝ D·Δp (Science 2010, Tokyo rail) | [E] |
| **Plants** | multi-timescale parallel signaling (AP + Ca²⁺ wave ~1 mm/s + hormone) | GLR-gated Ca²⁺ waves (Science 2018) | Signaling [E]; "plant neurobiology" label [C]; Mimosa habituation [C]; pea conditioning failed replication (Markel 2020) |
| **Immune system** | selection-trained anomaly classifier; affinity maturation = iterated optimization with rising stringency | clonal selection; negative selection → Forrest 1994 artificial immune systems | [E] |
| **Bacterial chemotaxis** | **integral feedback = robust perfect adaptation** (topology, not tuning) | dm/dt = g(a − a₀); Barkai–Leibler 1997; Yi et al. PNAS 2000 (integral feedback is the *only* linear strategy) | [E] — flagship result |
| **Quorum sensing** | analog population comparator, bistable switch | dA/dt = k₀N − γA; Hill output V·Aⁿ/(Kⁿ+Aⁿ) | [E] |
| **Biofilms** | K⁺ action-potential-like waves coordinate millions of cells | Prindle et al., Nature 2015 | [E] |
| **Fungi** | grown memristive sensor substrate; spike telemetry | Adamatzky 2022 spike "language"; 2024 fungal-controlled robot (Sci. Robotics) | Spikes [E]; "language" reading [C]; wood-wide-web mother-tree claims **[C — Karst et al. 2023 documented positive citation bias]** |
| **Levin bioelectricity** | voltage-graded rewritable computation layer between genome and anatomy; morphogenesis as setpoint control | V_mem instructive patterns; gap junctions = edge weights; TAME framework (2022); xenobots/anthrobots | [E] experiments; framing interpretations [C] |
| **Octopus** | federated control: ~350M of 500M neurons in arms; segmented arm nerve cord (Nat. Commun. 2025) | central goals, peripheral sensorimotor autonomy | [E] |
| **Basal cognition** | two-state stochastic learning machine | Stentor habituation = step-like Markov switch (Curr. Biol. 2022/23); Spirostomum habituates *without a nucleus* | [E] |

### Repeated engineering patterns across biology
1. Stigmergic indirect coordination (environment as blackboard)
2. Integral-feedback robust adaptation (tuning-free by topology)
3. Reaction–diffusion patterning (Turing conditions: trace<0, det>0, (f_uD_v + g_vD_u)² > 4D_uD_v·det)
4. Flow-adaptive network growth (Tero)
5. Voltage-graded signaling (not a neuron monopoly)
6. Quorum thresholds (Hill-function analog voters)

All six admit stdlib-Python toy models (~20–100 LOC each).

---

## Cross-cutting intersections (the meta-observation)

The same formal objects recur across all five research streams:
- **Pushouts/sheaf gluing** = schema integration (Part 1) = local-to-global sensor consistency (Part 3) = the calibration gate's DEFER condition.
- **Integral feedback / internal model** = bacterial chemotaxis (Part 5) = Conant–Ashby PREDICT mandate (notes/10) = drift-free sensor calibration.
- **Additive overlay buses** = transformer residual stream (Part 3) = multiphysics operator splitting = stigmergic blackboards (Part 5).
- **Superposition capacity ~√d** (VSA) bounds how many calibration codes / feature overlays one substrate carries.
- **Stratification vs diagonalization** (Part 1) = ledger-anchored trust (ecosystem) = why consciousness theories need marker batteries rather than self-certification (Part 4).
