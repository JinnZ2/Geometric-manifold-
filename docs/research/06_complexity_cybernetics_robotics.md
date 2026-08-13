# Notes 06 — Complexity Engineering, Cybernetics, Advanced Robotics (2026-08-13)

Companion to Notes 01–05. Detailed repo-interaction mapping lives in PLAN_FORWARD.md.

## 1. Complexity Engineering — principles & equations

- **CAS stance (INCOSE):** "gardener, not watchmaker" — influence emergence via incentives/feedback, not up-front control; zoom across scales; balance rather than optimize.
- **Effective complexity (Gell-Mann & Lloyd):** $\mathcal{E}(s) = K(E)$ — information in the *regularities* only; $K(s) \approx \mathcal{E} + \text{randomness}$.
- **Statistical complexity / ε-machines (Crutchfield):** causal states $S=\epsilon(\overleftarrow{x})$; $C_\mu = H[\mathcal S]$; ε-machine is the provably minimal, unique, optimal predictor. Adjacent: excess entropy $\mathbf E = I[\overleftarrow X;\overrightarrow X]$, transfer entropy $T_{Y\to X}$.
- **Self-organized criticality (Bak):** avalanche sizes $P(s)\sim s^{-\tau}$, $1/f$ noise; stress accumulation + threshold release.
- **Scale-free networks (Barabási):** $P(k)\sim k^{-\gamma}$; robust to random failure, fragile to targeted hub attack → distribute hubs, modular redundancy.
- **Antifragility (Taleb):** convex stressor response — $E[f(X+\sigma\epsilon)] > E[f(X)]$ for convex $f$ (Jensen gap); via negativa, hormesis, redundancy over efficiency.
- **Structural complexity (Sinha & de Weck, MIT):** $C = C_1 + C_2 C_3$, $C_3 = E(A)/n$ (normalized graph energy of the DSM); dev cost scales ~$C^{1.69}$; "Conservation of Complexity."
- **Causal emergence (Hoel):** macro scale can have *more* causal power: $\mathrm{EI} = \text{determinism} - \text{degeneracy}$ — select the scale of description that maximizes EI.
- **Digital-twin maturity (2025):** Mirroring → Monitoring → Modeling → Federation → Autonomous; CoDT → adaptive (CADT) → socio-technical (CSTDT).

## 2. Cybernetics — principles & equations

- **Ashby's Law of Requisite Variety (1956):** $V(Z) \ge V(D) - V(R)$ — only variety absorbs variety; attenuate environment or amplify regulator.
- **Wiener feedback:** $e = r - y$, $u = K_p e + K_i\!\int e\,d\tau + K_d \dot e$; negative feedback stabilizes.
- **Conant–Ashby Good Regulator Theorem (1970):** every good regulator is a model of the system; optimal + simplest regulator is a homomorphism $h: S \to R$. Modern: Internal Model Principle; Richens & Everitt (2024, arXiv:2402.10877) — bounded-regret agents *must* learn causal world models; Virgo et al. (2025, arXiv:2508.06326) — regulation implies observer-imputable beliefs.
- **VSM (Beer):** S1 operations, S2 coordination, S3 control (+S3* audit), S4 intelligence, S5 policy; algedonic signal bypasses hierarchy S1→S5; recursion: every viable system contains viable systems of the same form.
- **Pask Conversation Theory:** concept learned ⇔ teachback — reconstruct the other's explanation; agreement = synchronized procedures.
- **Powers PCT:** behavior controls *perception*: $e = r - p$; hierarchies of loops; reorganization learns when intrinsic error persists.
- **Allostasis:** setpoints predictively adjusted in anticipation; chronic anticipation cost = allostatic load.
- **Active inference (Friston/Parr/Pezzulo):** $F = E_Q[\ln Q(s) - \ln P(o,s)]$; policies minimize $G(\pi) = \text{risk} + \text{ambiguity} - \text{epistemic value}$ — Bayesian restatement of feedback control; curiosity = epistemic value.
- **Alignment as cybernetics:** reward hacking = proxy sensor with less variety than the true objective (requisite-variety violation); corrigibility = keep the algedonic channel open.

## 3. Advanced Robotics — principles & equations

- **VLA foundation models:** RT-2 (arXiv:2307.15818) discretizes actions into 256 token bins, plain cross-entropy; RT-X (2310.08864): positive transfer across 22 embodiments.
- **π0 (2410.24164) flow matching:** noised action $A^\tau = \tau A + (1-\tau)\epsilon$; $\mathcal L_{FM} = E\|v_\theta(A^\tau,o) - (A^\tau - A)\|^2$; ~10-step Euler ODE decode at 50 Hz. GR00T N1 (2503.14734): VLM System-2 + DiT System-1 at 120 Hz. Gemini Robotics On-Device (2025) — local inference.
- **Diffusion Policy (2303.04137):** $\mathcal L_{DDPM} = E\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_k}a_0 + \sqrt{1-\bar\alpha_k}\epsilon, k, o)\|^2$, receding horizon.
- **Sim-to-real:** domain randomization $\pi^* = \arg\max_\pi E_{\xi\sim p(\xi)}[\sum \gamma^t r_t]$ over friction/mass/delays; parkour zero-shot transfer.
- **Humanoid MPC:** $\min_u \sum_k \|x_k - x^{ref}_k\|_Q^2 + \|u_k\|_R^2$ s.t. $x_{k+1}=f(x_k,u_k)$.
- **World-model planning (V-JEPA 2-AC, 2025):** $a^*_{t:t+H} = \arg\min_a \|P(\cdot,a) - E(o_{goal})\|^2$, CEM in latent space, zero-shot pick-and-place.
- **Safety:** CBF-QP runtime assurance: safe set $h(x)\ge0$, constraint $L_f h + L_g h\,u \ge -\alpha(h)$, minimal-intervention filter $u^*=\arg\min\|u-u_{nom}\|$; HJ reachability $\partial_t V + \min\{0,\min_u\max_d \nabla V\cdot f\}=0$.
- **Robot self-models (Lipson, Sci. Robotics 2022):** learned morphological self-model → damage detection + replanning after motor loss.
- **Event/neuromorphic sensing (EveTac, DVS):** asynchronous Δ-threshold spikes — the hardware realization of band-index encodings.
- **VLA runtime monitors (2026, arXiv:2605.30834):** mine latent failure signals from action trajectories — falsification agents for robots.

## 4. Unifying observation

Complexity engineering supplies the *metrics* (ε-machines, graph energy, requisite variety), cybernetics the *architecture* (VSM, algedonic channels, good-regulator constraint), robotics the *embodiment* (CBF safety, world-model planning, repurposed-hardware control). The repo already contains proto-versions of all three: HND ≈ causal-state analysis, claims+dependency trees ≈ the good regulator's homomorphic model, hardware stewardship ≈ runtime-assurance fallback catalogs. The forward plan formalizes each.
