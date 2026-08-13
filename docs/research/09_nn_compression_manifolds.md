# 09 — Neural-Network Compression as Empirical Science + Manifold/Representation Geometry

Seed: user memo "NN compression as multi-objective constrained optimization." Research date: 2026-08-13.
Companion: notes/10 (integration theories & languages) · integration/INTEGRATION_POINTS.md.

---

## Part 1 — Compression science, verified

### 1.1 Verified backbone papers (exact IDs/venues/numbers)
| Method | Citation | Key verified facts |
|---|---|---|
| **GPTQ** | Frantar, Ashkboos, Hoefler, Alistarh, ICLR 2023, arXiv:2210.17323 | OBQ/OBS-based (Hassibi & Stork 1992 lineage); 175B → 3–4 bit in ~4 GPU-hours; damping λ ≈ 1% of mean Hessian diagonal; 3.25× speedup A100, 4.5× A6000. Layerwise objective argmin ‖WX − ŴX‖², H = 2XXᵀ; OBS update δ_q = −(w_q − quant(w_q))/[H⁻¹]_qq · H⁻¹_{:,q}; error E_q = (w_q − quant(w_q))²/(2[H⁻¹]_qq) |
| **AWQ** | Lin et al., **MLSys 2024** (memo said 2023-era), arXiv:2306.00978 | Salient weights = **0.1–1%** chosen by *activation* magnitude; 1% FP16 rescue: OPT-6.7B INT3-g128 PPL 43.2→13.0; equivalent scaling WX = (W·diag s)(diag(s)⁻¹X), s≈2 by 20-point grid |
| **SparseGPT** | Frantar & Alistarh, ICML 2023, arXiv:2301.00774 | One-shot 50–60% unstructured sparsity, no retraining; also 2:4 / 4:8 semi-structured; compounds with 4-bit GPTQ; larger models compress better at fixed sparsity |

### 1.2 Memo corrections (flagged by verification)
1. **Compression order**: "prune-first universally optimal" (Harma 2025) is **refuted** by arXiv:2603.18426 (2026) — use the **Progressive Intensity Hypothesis** (order by compression intensity) as the framing. Memo hypothesis #4 (noncommutativity) stands and is timely.
2. **"VGG16→pruning, ResNet→quantization"**: no single study states this; closest is arXiv:2509.04244 (SPQ simultaneous better on VGG-16, PPQ staged better on ResNets). Reword as "integration strategy is architecture-dependent."
3. **Low-rank rank-learning**: Idelbayev & Carreira-Perpiñán is **CVPR 2020, UC Merced** (not Berkeley); method = Learning-Compression framework with per-layer rank learning under constraint.
4. **Sheared-LLaMA** = arXiv:2310.06694.
5. **CKA** is invariant only to orthogonal transforms and *isotropic* scaling — rotation-based compression (QuaRot, SliceGPT) is **invisible to CKA**; geometry evaluation needs a multi-metric battery.
6. **8× compression ≠ 8× speedup**: LLM decode is memory-bandwidth-bound (GPTQ's own 3.25–4.5×); dequant overhead, group-wise scale traffic, codebook lookups (VPTQ/QuIP# decode can be slower than FP16 in naive kernels), unstructured sparsity ≈ 0 speedup without sparse kernels.

### 1.3 2024–2026 frontier (verified IDs)
- **Rotation methods**: QuIP (2307.13304), QuIP# (2402.04396, Hadamard incoherence + E8 lattice), QuaRot (2404.00456, W4A4KV4 via computational invariance), SpinQuant (2405.16406, learned Cayley rotations; shipped in quantized Llama 3.2). **Caveat**: rotation methods are model-fragile (collapse on Llama-3-70B W4A4; FlatQuant 2410.09426 more robust).
- **Vector quantization**: AQLM (2401.06118), VPTQ (2409.17066, SOTA 2-bit, 1.6–1.8× faster than AQLM); GPTQv2 (2404.16692).
- **KV-cache**: KIVI (2402.02750 — key per-channel, value per-token, 2-bit), KVQuant (2401.18079).
- **1-bit era**: BitNet b1.58 (2402.17764, ternary {−1,0,+1} matches FP16 at 3B+), bitnet.cpp (2410.16144, ~6× vs llama.cpp on CPU).
- **Pruning**: Wanda (2306.11695, S_ij = |W_ij|·‖X_j‖₂ per-output-row, no retraining), SliceGPT (2401.15024 — see §2.6), LLM-Pruner (2305.11627), depth-pruning (ShortGPT 2403.03853); 2:4 sparsity needs Ampere sparse tensor cores (2104.08378).

### 1.4 Evaluation methodology (verified)
- **Calibration**: prefer NLL/Brier over ECE (bin-sensitive).
- **Forgetting analysis**: Hooker et al., "What Do Compressed Deep Neural Networks Forget?" (1911.05248) — compressed-model errors concentrate on a long-tail subset (PIE); error-set overlap is the instrument.
- **Energy**: RAPL (CPU/DRAM only), NVML polling undersamples µs bursts, CodeCarbon is TDP×utilization estimation; ML.ENERGY benchmark; Luccioni et al. (2311.16863). Pitfalls: idle-power baseline subtraction, prefill (compute-bound) vs decode (memory-bound) separation, batch size dominates per-token energy.
- **Distillation ≠ geometry transfer**: Stanton et al., NeurIPS 2021 (2106.05945) — students match predictions, diverge in representations. Relational KD (RKD, 1904.05068) preserves more; no study shows full geometry preservation. Memo hypothesis #5 confirmed by literature.

---

## Part 2 — Manifold possibilities (representation geometry)

### 2.1 Manifold hypothesis, made falsifiable
- **Fefferman-Mitter-Narayanan, JAMS 29(4):983–1049 (2016)**: a *statistical test* — given samples, decide if data lie within ε of a d-manifold with bounded reach; sample-complexity guarantees. Template: geometry claims with error bars.
- **Empirical intrinsic dimension** (Pope et al., ICLR 2021, 2009.04059): ImageNet ID ≈ 26–43, CIFAR-10 ≈ 13, MNIST ≈ 13–15. Generalization scales with ID, not ambient dimension → **rank budgets should track manifold dimension, not parameter count**.
- Estimators: Levina–Bickel MLE d̂(x) = [(1/k)Σ log(T_k/T_j)]⁻¹; TWO-NN (Facco et al. 2017, Pareto ratio of 2nd/1st neighbor distances); GRIDE decimation fix. Pitfalls: all local estimators underestimate under anisotropic scaling — **fix metric convention (whiten/cosine) before comparing ID across compression arms, or the comparison is vacuous**.

### 2.2 Information geometry
- Fisher–Rao metric g_ij = E[∂ᵢ log p · ∂ⱼ log p]; natural gradient (Amari 1998); **OBS/GPTQ are quasi-Fisher corrections — the geometric reading: prune/quantize along low-curvature directions**.
- Information bottleneck L_IB = I(T;Y) − β⁻¹I(X;T): Saxe et al. 2019 (1805.12191) showed compression-phase claims are estimator/nonlinearity artifacts; Goldfeld et al. — geometric clustering, not MI compression. **Lesson: MI is estimator-fragile; subspaces and spectra are computable and auditable. Use IB as motivation, never as measurement.**

### 2.3 NTK / lazy training regime
- NTK (Jacot et al. 2018, 1806.07572): wide nets → frozen kernel; lazy training (Chizat et al. 2019) ‖θ−θ₀‖ = O(1/√n). **In the kernel regime, preserving the empirical kernel up to small spectral perturbation provably preserves function — CKA becomes exact.** Real LLMs are in the feature-learning regime; a cheap NTK-stability / parameter-distance pre-screen decides which theory applies per subnetwork.

### 2.4 Representation-geometry empirics
- **Manifold capacity** (Chung, Lee, Sompolinsky PRX 8:031003, 2018; Cohen et al. Nat. Commun. 11:746, 2020): per-class-manifold anchor radius R_M, dimension D_M; perceptron capacity α_M ≈ α_ball(R_M, D_M). D_M, R_M decrease layer-by-layer; theory matches AlexNet/VGG and macaque IT. **Compression destroys function when it inflates R_M²D_M beyond surviving width's capacity.** This is the strongest formalization of "the functional subspace."
- **Platonic representation hypothesis** (Huh et al. ICML 2024, 2405.07987): representations converge with scale (mutual-kNN alignment); null-calibration work (2602.14486) shows local structure converges more robustly than global geometry → compressed models have a well-defined target geometry; metrics must be null-calibrated.
- **LLM anisotropy/cone effect** (Ethayarajh 2019, 1909.00512; Godey et al. 2024): embeddings occupy a narrow cone → **whiten/mean-center before any ID/angle/CKA on LLM activations**.
- **Hunchback ID profile** (Ansuini et al. NeurIPS 2019, 1905.12784): ID rises early, peaks mid-network, collapses toward classifier; replicated in LLMs and stable across scale (Cheng et al. 2023) → **per-layer ID profile = per-layer rank budget**; tail layers tolerate most slicing.

### 2.5 Topology (honest limits)
- Naitzat et al. JMLR 2020 (2004.06093): Betti numbers of class clouds decay layer-by-layer with training. PH is a hypothesis generator, not a certificate — unstable under re-embedding, cubic scaling, small-scale results only.

### 2.6 Symmetry & SliceGPT as existence proof
- Geometric DL blueprint (Bronstein et al. 2021, 2104.13478): architectures = symmetry groups; compressing G-equivariant layers needs G-invariant masks / tangential quantization noise.
- **SliceGPT (ICLR 2024, 2401.15024)**: RMSNorm commutes with orthogonal transforms (computational invariance, their Thm 1); rotate each block's residual stream to its activation-PCA basis, delete minor components; ~25–30% parameter removal at ~99% accuracy on 70B models. **This is exactly "compress along the activation manifold's minor directions while preserving the functional subspace" — a constructive existence proof of the program's premise** (with linear/PCA geometry rather than curved manifolds).

### 2.7 The geometry battery (computable, falsifiable)
For pre/post-compression hidden states H, H′ (whitened, fixed probe set):
| Criterion | Formula | Cost | Pure-Python feasible |
|---|---|---|---|
| Principal angles / chordal distance | θ_i = arccos σ_i(U_kᵀU_k′); ‖sin Θ‖_F | one SVD each | ✓ (≤256²) |
| Linear CKA | HSIC(K,L)/√(HSIC(K,K)HSIC(L,L)) | O(n²d) | ✓ |
| Orthogonal Procrustes | min_{Q∈O(d)} ‖H − H′Q‖_F | SVD of HᵀH′ | ✓ |
| Mutual-kNN alignment | shared-k fraction | O(n²d) | ✓ |
| TWO-NN ID profile | §2.1 | O(n log n) | ✓ |
| Manifold capacity R_M, D_M, α_M | §2.4 | small QP | moderate |
| MI estimators | — | high bias | corroboration only |

### 2.8 Falsifiable research hypotheses (the program's core)
1. **Manifold-aware quantization**: project quantization error onto the activation tangent space (local PCA) vs random equal-norm directions; H₁: tangent-projected error degrades downstream CKA/perplexity less at matched weight-MSE.
2. **ID-matched rank allocation**: at fixed total rank budget, hunchback-aware per-layer allocation beats uniform on accuracy AND principal-angle preservation; budgets transfer across scale (Cheng et al.).
3. **Grassmannian distillation**: L = λ₁‖sinΘ(H_s,H_t)‖²_F + λ₂(1−CKA) + λ₃|D_M^s − D_M^t| beats logit-KD on few-shot transfer probes at matched logit accuracy.
4. **Umbrella hypothesis**: at matched scalar accuracy, methods differ systematically in Grassmann distance of preserved functional subspaces, and geometry-preserving methods dominate transfer/robustness probes — accuracy is a coarse projection of a finer geometric invariant.
