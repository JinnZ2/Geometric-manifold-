# Notes 04 — Neural Architecture (Diagnostics, Geometric Inference, Hidden Variables)

## A. Repo: Systems Diagnostic Suite (GAE / HND / FDM) + geometric inference

**GAE — Geometric Applicability Engine** (`modules/gae.py`). Topological fingerprint of a system graph: SystemMetrics(C, N, L, R):
$$C = \frac{|\text{edges in simple cycles}|}{|\text{edges}|},\quad N = \#\{deg > \overline{deg}\},\quad L = \frac{\#\{in\le 1 \wedge out\le 1\}}{n},\quad R = \min\!\big(1, \mathrm{Var}(depths)/100\big)$$
Geometry scores (clamped 0–100), recommendation = argmax, forbidden = argmin:
```
LINE        = L·80 − C·60 − |N−2|·10
TRIANGLE    = (50 if N==3 else −|N−3|·20) + L·20 + (1−R)·20
TETRAHEDRON = (60 if N∈{4,5} else −|N−4|·15) + C·30 + (1−R)·10
TORUS       = C·80 + (20 if N≥4 else −20) + (30 if R<0.4 else −10)
ICOSAHEDRON = (60 if N≥6 else −(6−N)·15) + (1−L)·30 + R·10
FRACTAL     = (80 if R<0.3 else −20) + C·20
```
Design rule: "Avoid the forbidden geometry — it will break feedback loops and reduce resilience."

**HND — Hidden Node Detector** (`modules/hnd.py`): scan only if $\overline{|residual|} \ge 0.1$. Three detectors:
1. *Residual gradient:* Pearson $r(var, residuals)$ for env series not in model; suggest if $|r|>0.5$.
2. *Phantom causality:* disconnected model-node pairs with $|r|>0.7$ ⇒ find common cause $c$: $|r(a,c)|>0.6 \wedge |r(b,c)|>0.6$.
3. *Hidden buffer:* if $\overline{observed - predicted} > 0.05$, non-model vars with $var[-1] > 1.1\,var[0]$ flagged (conf 0.7).
$$r = \frac{n\sum xy - \sum x\sum y}{\sqrt{(n\sum x^2-(\sum x)^2)(n\sum y^2-(\sum y)^2)}}$$

**FDM — Fractal Dependency Mapper** (`modules/fdm.py`): recursive root-tracing to primitive roots (SUNLIGHT, SOIL, WATER, MUSCLE, GRAVITY, SEEDS, LIVESTOCK); cycle → DEGRADED; node BROKEN if any child broken; depth guard 20; knowledge externalized to `data/*.json`.

**Pipeline** (`modules/main.py`): GAE → HND → FDM → "update model with new nodes, re-run GAE, re-run HND, iterate until residuals fall below threshold."

**Harmony field / geometric inference** (`harmony_field_engine.py`, `geometric_inference_engine.py`): relational state on 30 icosidodecahedron vertices (cyclic permutations of $(0,0,\pm\varphi)$, $(\pm\tfrac12, \pm\varphi/2, \pm(1{+}\varphi)/2)$, golden ratio φ):
$$disp = F(1 - stiffness_i);\quad x \mathrel{+}= disp;\quad x \mathrel{+}= (x_{rest}-x)\cdot stiffness\cdot 0.2;\quad x \mathrel{-}= (x - x_{prev})\cdot 0.8$$
Prediction principle: **next action locus = vertex with maximal pole displacement** ($argmax\ \|x - x_{prev}\|$). Success stiffens (trust = spring constant); failure injects noise.

## B. Literature (2024–2026)

**NAS:** supernet weight-sharing $W_A = \arg\min_W \mathbb E_a[\mathcal L_{val}(a,W)]$; zero-cost proxies (SynFlow $S=\sum_i|\theta_i \partial \mathcal L/\partial\theta_i|$, Spearman ≈0.90); 2024–26 trend: LLMs as mutation operators in evolutionary NAS.

**KAN** (arXiv:2404.19756): $\mathbf x_{l+1,j}=\sum_i \phi_{l,j,i}(x_{l,i})$, $\phi(x)=w_b\,\mathrm{SiLU}(x)+w_s\sum_i c_i B_i(x)$ — learnable univariate edge functions; interpretable; KAN-Dreamer applies to world models.

**MoE routing:** TopK + softmax gates; aux balance loss $\mathcal L=\alpha\sum_i f_i P_i$ vs **loss-free bias balancing** (sign updates; proven convergence). Fine-grained experts + shared expert = current standard.

**Conditional compute:** Mixture-of-Depths (arXiv:2404.02258): $\tilde x_t^{l+1}=\mathbb 1[r_t^l \in \text{top-}k]\,f_l(x_t^l)+x_t^l$ — tokens skip layers; ~50% faster sampling at matched quality.

**Hidden-variable / latent-structure discovery:** Slot Attention (GRU + attention competition); causal-graph learning with NOTEARS acyclicity; latent confounders (CEVAE $q_\phi(z|X,T,Y)$; CI-StoNet identification guarantees). **Residual→confounder detection** (HND's core idea) exists as a real but *scattered* literature: residual independence tests ($\hat\varepsilon \not\perp T$ flags omitted confounders), Rosenbaum sensitivity bounds, partial-ancestral-graph RCA (arXiv 2606.20912). No flagship framework — HND's three-detector heuristic is a legitimate engineering instantiation of an open problem.

**Graph-based failure tracing (RCA):** causal graphs over system metrics scored with personalized PageRank $r = \alpha P^\top r + (1-\alpha)v$ (KGroot, CIRCA, RUN) — the industrial analogue of FDM root-tracing.

**Neural ODE/SDE:** $\dot h = f_\theta(h,t)$, adjoint $\dot a = -a^\top \partial f/\partial h$; now embedded in latent/generative/causal models rather than standalone.

**Geometric deep learning:** equivariance $f(\rho_g x)=\rho'_g f(x)$ — the repo's Platonic-solid geometry scores are a hand-crafted version of symmetry-informed architecture selection.

## C. Principles

- P1: Topology is diagnostic: cycle density / critical nodes / linearity / depth variance predict which feedback geometry a system can sustain.
- P2: Residuals are evidence of hidden variables; scan only when mean|residual| exceeds threshold (don't chase noise).
- P3: Phantom correlation between disconnected nodes ⇒ common cause; this is HND's strongest idea and matches the causal-inference literature.
- P4: Trace every dependency to a primitive root (things you cannot manufacture); cycles and broken roots are degradation, not detail.
- P5: Trust can be a physical parameter (spring stiffness); prediction = locating maximal distortion — a geometric, non-gradient inference rule.
- P6: Iterate diagnosis: discover hidden node → add to model → re-score geometry → re-scan residuals.
