# 10 — Perceptron & Integration Theories · Language/Stack Strategy

Research date: 2026-08-13. Companion: notes/09 (compression & manifolds) · integration/INTEGRATION_POINTS.md.

---

## Part 1 — Perceptron & integration theories

### 1.1 Perceptron foundations (ESTABLISHED)
- **Novikoff 1962 convergence**: separable with margin γ, ‖x‖ ≤ R ⇒ at most (R/γ)² mistakes. *Reading: margin γ of fused substrate evidence is a live health metric; convergence cost diverges as 1/γ² as substrates become ambiguous.*
- **Minsky–Papert 1969**: XOR/parity not linearly separable. *Reading: a single linear gate is provably blind to interaction-type contradictions (each substrate nominal, jointly contradictory) — the most dangerous multi-source failure mode. Contradiction detection needs ≥1 hidden layer.*
- **Cover 1965 function counting**: separable dichotomies = 2·C(N−1, d); capacity threshold N ≈ 2(d+1). **VC dim of linear classifiers = d+1**. *Reading: each substrate fed to one gate consumes VC capacity; Cover/VC bound the number of independent falsifiable claims a single integration layer can track before memorization.*
- **Manifold capacity** (Chung–Sompolinsky PRX 2018): α_M decreases with R_M√D_M — shrink each substrate's representation radius/dimension before fusion; rising effective manifold dimension = pre-collapse signature.

### 1.2 Modern integrators
- Random features (Rahimi–Recht 2007): random expansion + linear readout ≈ kernel machine.
- ELM (Huang 2006): fixed random hidden layer, ridge readout W = H†T.
- Reservoir/ESN (Jaeger 2001): **echo state property** — contraction (spectral radius < 1 with margin) is the stability condition for temporal integration; loss of contraction precedes divergence of fused streams.

### 1.3 Neuroscience integration theories (status-flagged)
| Theory | Core formal statement | 2025–26 status |
|---|---|---|
| **IIT** (Tononi) | Φ = min over partitions of D(cause–effect structure ‖ partitioned structure); IIT 4.0: Albantakis et al. 2023, PLOS Comput. Biol. 19(10):e1011465 | **CONTESTED**: 124-signatory "pseudoscience" letter (2023, Nature news d41586-023-02971-1); Cogitate adversarial collab (Nature 642:133–142, 2025): neither IIT nor GNW vindicated; IIT posterior emphasis fared relatively better. **Do not use Φ as an instrument; partition-perturbation testing is a legitimate theory-neutral probe.** |
| **GNW** (Baars/Dehaene) | Ignition (~300 ms nonlinear threshold) + frontoparietal broadcast; Dehaene & Changeux, Neuron 70:200–227, 2011 | Functional/computational — most portable to engineering (broadcast bus with all-or-nothing access); Cogitate 2025: onset ignition partial, offset absent |
| **FEP / predictive processing** (Friston) | F = E_q[−ln p(s,θ)] − H[q] = D_KL(q‖p(θ|s)) − ln p(s); precision weighting = gain control; Nat. Rev. Neurosci. 11:127–138, 2010 | Inference framework rigorous; *universal* FEP claim contested (unfalsifiability). **Precision weighting = Kalman gain** — directly implementable |
| **HOT** | Higher-order representation; meta-d′ (type-2 SDT) | Live; supplies the **metacognition/confidence/abstention** layer |

### 1.4 Cybernetic integration theory (ESTABLISHED)
- **Conant–Ashby good regulator** (1970): every good regulator is a (homomorphic) model of the regulated system → a fusion gate must internally model substrate joint statistics; **the PREDICT role is theorem-mandated, not optional**.
- **Ashby requisite variety**: V(regulator) ≥ V(disturbances) − V(buffer) — hard lower bound on gate capacity.
- **Kalman filter**: x̂ = x̂⁻ + K(z − Hx̂⁻), K = P⁻Hᵀ(HP⁻Hᵀ + R)⁻¹ — MMSE-optimal linear fusion; **K ∝ P⁻/R is exactly FEP precision weighting** (the rigorous bridge).
- **Internal Model Principle** (Francis–Wonham 1976): perfect tracking requires an embedded copy of the exosystem — the falsifiable counterpart of Conant–Ashby.

### 1.5 Fusion under correlated evidence (the honest core result)
| Method | Rule | Unknown correlations? |
|---|---|---|
| Information filter | Y = Y⁻ + HᵀR⁻¹H (pure summation) | No — double-counts |
| **Covariance Intersection** (Julier & Uhlmann 1997) | P⁻¹ = ωP_a⁻¹ + (1−ω)P_b⁻¹; ω optimized | **Yes — consistent for any unknown cross-correlation; the right tool for dependent substrates** |
| Track-to-track (Bar-Shalom–Campo) | explicit cross-covariance P_ab | Yes, if tracked |
| Dempster–Shafer | m(C) ∝ Σ_{A∩B=C} m₁(A)m₂(B); conflict K | Assumes independence; Zadeh paradox at high K — use K only as a conflict *signal* |
| Opinion pools | linear Σwᵢpᵢ (robust, never sharpens) vs logarithmic Πpᵢ^wᵢ (vetoes; double-counts shared evidence) | Linear pool safe under overlap |

**Key finding: naive Bayes, information filters, log pools, and Dempster's rule all assume independence and produce overconfident fusion — a false sense of integration health. With dependent substrates use CI, explicit cross-covariance, or linear pools.**

### 1.6 Learned integration
- MoE gating y = Σ gᵢ(x)Eᵢ(x): the router is a perceptron over reliability features; monitor **gate entropy** — collapse onto one expert = integration failure signal.
- Attention softmax(QKᵀ/√d)V: differentiable, data-dependent fusion kernel; attention weights ≈ adaptive Kalman gains.
- Neuro-symbolic: soft logic gates (Łukasiewicz/product t-norms) are generalized perceptrons; semantic loss L ∝ −ln Σ_{y⊨α} p(y).

### 1.7 Synthesis → calibration-gate design map
| Theory | Justifies | Early-warning signal |
|---|---|---|
| Novikoff/Cover/VC | capacity limits on one fusion layer | margin γ→0; claims > 2(d+1) |
| Minsky–Papert | hidden layer for XOR-contradiction detection | joint-contradiction blindness |
| Kalman / precision weighting | confidence-weighted fusion | innovation (z − Hx̂⁻) non-whiteness |
| Covariance Intersection | fusion of dependent substrates | overconfidence under independence assumption |
| D-S conflict mass K | explicit contradiction bookkeeping | rising K → deferral trigger |
| Requisite variety | minimum gate complexity | variety deficit → uncontrollable regime |
| Conant–Ashby / IMP | mandatory PREDICT role | model–plant residual growth |
| HOT | abstention/deferral layer | meta-d′ collapse (confidence–accuracy decoupling) |
| GNW [contested] | broadcast-bus architecture | ignition-threshold dysfunction |
| IIT [contested] | partition-perturbation probe only | partitionable pipelines pass data without integration |
| MoE/attention | learned routing | gate-entropy collapse |

---

## Part 2 — Language & stack strategy

### 2.1 Measured stdlib-Python envelope (benchmarks on reference hardware, ±3×)
| Operation | Time |
|---|---|
| matmul 64×64 / 128×128 / 256×256 | ~22 ms / ~170 ms / ~1.45 s |
| Truncated SVD (power iter, k=8) 128×128 / 256×256 | 0.5 s / 2.0 s |
| Linear CKA N=100,d=64 / N=300,d=64 | 0.12 s / ~1.1 s |
| Pure-Python multiply-accumulate | ~9 M MAC/s |
| 1 MNIST epoch, 784→64→10 MLP | **~20–25 min (practical ceiling)** |

**Feasible stdlib-only**: quantization *simulation* (INT8/INT4/ternary is integer bookkeeping — trivially stdlib), the full geometry battery at ≤256² (CKA, effective rank, SVD, Procrustes), small-MLP training (subset-MNIST), magnitude pruning/lottery-ticket on tiny nets, pure-Python distillation demos.
**Hard wall**: training any real ResNet/transformer (~10⁴–10⁵× MAC rate needed), GPU access, 100M-param inference (~100 s/token). **Stdlib covers algorithm prototyping + pedagogy + auditability, not evidence on realistic models. That distinction drives the tiering.**

### 2.2 Recommended tiered stack
| Tier | Contents | Deps |
|---|---|---|
| **0 — stdlib core (unchanged)** | ledgers, calibration, cascade detection, emitters, hypothesis-engine **plus**: quantization simulator, power-iteration SVD, CKA/erank ≤256², **.npy reader/writer** (~60 LOC via struct — spec is 1 page), SHA-256 content-addressed manifests | none |
| **1 — minimal-dep research** | numpy 2.x re-implementations validated bit-near against Tier 0; pytest | numpy, pytest |
| **2 — experiment** | torch for real compression (QAT, pruning, distillation, ResNet-scale); export ONNX/GGUF + manifest | torch, onnx |
| **3 — embedded** | MicroPython sensor nodes (RP2040/ESP32 — matches existing Scoppy work, stdlib-subset philosophy), TFLM/microTVM C firmware; Rust optional (WASM adapters, safe firmware); Julia optional sidecar only if manifold-valued geometry becomes central | per-project |

### 2.3 Interop contracts (the ecosystem's real asset: "core stays pure, adapters optional")
- **File-format-first**: NDJSON ledgers/streams, CSV calibration tables, **.npy v1.0 as tensor interchange** (readable by numpy/torch/JAX/Rust alike), GGUF only if local LLM inference enters scope, ONNX for non-LLM models.
- **Content-addressed artifacts**: every checkpoint/ledger named by SHA-256; manifest JSON hash→metadata. Falsifiability + cache-friendly Actions. (Directly extends the falsification_ledger hash chain to binary artifacts.)
- **Process pipes**: adapters are executables orchestrated via stdlib subprocess; failures isolated; core never imports adapters.
- **FFI only when pipes too slow**: tiny C-ABI .so via stdlib ctypes, graceful degradation.
- Reference structures: SQLite (amalgamation core + thin bindings), redis (protocol-first), ripgrep (core libs + thin CLI). **The contract (format/protocol) is what the core repo tests — not the implementations.**

### 2.4 Embedded reality check
- **RP2040 (264 KB SRAM, no FPU)**: int8 models ≲150–200 KB params, ≲100 KB activation arena; keyword-spotting CNNs (~20–60K params), tiny MLPs, anomaly detectors; CMSIS-NN is M4+-optimized → mostly plain C on RP2040; TFLM/microTVM target it. **A Tier-2-trained, distilled, int8/int4-quantized MLP genuinely fits — a real end-to-end demonstrable pipeline.**
- Rule of thumb: compression ratio needed = model_bytes / (0.5 × target_RAM). Pi-class: no compression for CNNs, int4 for 7B-class on 8 GB.
- Edge runtimes 2026: ExecuTorch (Meta-backed, growing), llama.cpp/GGUF de facto for local LLMs (Q4_K_M ≈ 4.8 bpw sweet spot).

### 2.5 2026 landscape facts (verified)
- Python 3.14: free-threading officially supported (PEP 779) but non-default, single-thread ~91% of standard; JIT experimental/neutral. Don't design around it.
- numpy 2.x stable, free-threading-compatible. Mojo 1.0.0b2 beta (June 2026) — watch, don't adopt.
- Decision: **first dependency ever added = numpy+pytest, in a separate Tier-1 package, never in core.**

### 2.6 Decision table
| Question | Answer |
|---|---|
| Core stays stdlib-only? | Yes — quant sim, CKA, SVD, tiny MLPs all fit |
| First dep? | numpy+pytest (Tier 1 only) |
| Real compression evidence? | torch in Tier 2, ONNX/GGUF + manifests out |
| RP2040? | int8/int4 MLPs/small CNNs ≤~150K params; MicroPython glue |
| Rust? | Optional: WASM demos, firmware |
| Julia? | Only if manifold geometry goes central |
| Mojo? | Watch at 1.0 |
