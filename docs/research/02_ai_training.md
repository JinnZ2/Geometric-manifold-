# Notes 02 — AI Training

## A. Training/learning rules found in the repo

**Confidence propagation through dependency trees** (`claims.py::propagate_confidence`):
$$c_{claim} = \overline{c_i},\qquad c_{dep} = \overline{c_{deps}},\qquad c_{node} = \tfrac{1}{2}(c_{claim} + c_{dep})$$
(empty sets default 0.5; order-independent by design after REVIEW §3.19 fix).

**World-model delta rule** (`grounding/worlds/bumpy.py::WorldModel.update`, lr = 0.01 — the LMS/Widrow–Hoff rule):
$$\hat y = w_0 x + w_1 a + b,\quad e = y - \hat y,\quad w_i \mathrel{+}= \eta\, e\, x_i$$

**Self-model (meta-learning)** (`self_modeling_explorer.py`): predicts its own world-model error; target $|e_{world}|$, inputs $(x, a, e_{world}, e_{self,prev})$, same delta rule with η = 0.001. Curiosity = $e_{self,prev} - |e_{self}|$ — *meta-curiosity*: seek experiences that sharpen self-understanding.

**Curiosity reward (garden/unified):** $r_{cur} = |e_{prev}| - |e|$ (reduction in prediction error).

**Skill extraction & learning** (`unified_playground.py::SkillLab`): distill world-model weights into executable code `predict_position = w0·x + w1·force + b` plus auto-generated assert tests; confidence updated with the same ±0.1/−0.2 claim rule; refactor = counted escape hatch.

**Episodic memory retrieval** (`memory.py`):
$$\mathrm{score}(e) = |q \cap e| + 0.1\cdot\frac{1}{1 + N - i}$$
(keyword overlap + recency boost).

**Harmony-field trust learning** (`geometric_inference_engine.py`): success → `stiffness ← min(1, stiffness + 0.1)` and vertex pulled 30% toward rest; failure → perturbation force $\mathcal{N}(0, 0.05, 3)$.

**LLM training sketches** (`project/Organize.md`, PLAN stage):
- `train_contrastive_repair`: mask failed reasoning chain at `first_error_step_idx`, train corrected continuation (self-repair mid-reasoning).
- `train_meta_cognitive_signal`: auxiliary classifier on per-step hidden states predicting success/failure — an "internal risk sensor" (cf. belief probes literature).
- `train_with_preference_alignment`: step-level rewards → DPO/PPO steering away from low-reward trajectories.
- Contrastive pairs: (failed_chain, corrected_chain); `StyleMetaLearner`: k-NN over (prompt_embedding, style, outcome).

## B. Literature equations (2024–2026)

**GRPO** (DeepSeekMath, arXiv:2402.03300) — critic-free RL:
$$\hat A_{i,t}=\frac{r_i-\mathrm{mean}(\{r_j\})}{\mathrm{std}(\{r_j\})},\quad \mathcal J=\mathbb E\Big[\tfrac1G\sum_i\tfrac{1}{|o_i|}\sum_t\min(\rho_{i,t}\hat A_{i,t},\mathrm{clip}(\rho_{i,t},1\pm\epsilon)\hat A_{i,t})-\beta D_{KL}\Big]$$
Corrections: Dr. GRPO (removes length/std normalization bias, arXiv:2503.20783), DAPO (clip-higher, token-level loss, 2503.14476), GSPO (sequence-level ratio $s_i=(\pi_\theta/\pi_{old})^{1/|o_i|}$, 2507.18071).

**RLVR / R1** (arXiv:2501.12948): rule-based binary reward $r(o)=\mathbb 1[\text{correct}]$ in GRPO → emergent long CoT, self-verification. *The repo's machine-checkable `logical_form` claims are an RLVR-style verifiable reward substrate.*

**Test-time compute:** s1 budget forcing (arXiv:2501.19393): accuracy $\approx a + b\log N_{think}$ ("Wait" injection).

**Muon optimizer** (arXiv:2409.20325; Moonlight 2502.16982): $M_t=\mu M_{t-1}+G_t$; $W \leftarrow W - \eta\,\mathrm{NewtonSchulz}_5(M_t)$ — orthogonalized updates, ~2× compute efficiency; Newton–Muon (2026) adds $(ZZ^\top)^{-1}$ preconditioning. Used in Kimi K2 (MuonClip), GLM-4.5.

**Scaling laws:** Chinchilla $L(N,D)=E + A/N^\alpha + B/D^\beta$; data-constrained revision $\hat D = U_D + U_D R_D^*(1-e^{-R_D/R_D^*})$ (arXiv:2305.16264) — unique data is the binding constraint; >4 epochs of repetition sharply diminishing.

**Knowledge distillation:** $\mathcal L_{KD}=T^2 D_{KL}(\sigma(z_T/T)\|\sigma(z_S/T))$; reasoning distillation mostly plain SFT on teacher traces (R1-Distill).

**Continual learning:** EWC $\mathcal L = \mathcal L_{new} + \tfrac{\lambda}{2}\sum_i F_i(\theta_i-\theta_i^*)^2$; replay is the strongest defense (forgetting grows with scale).

**Self-play:** SPIN (arXiv:2401.01335) — DPO-style loss discriminating target data from own generations; fixed point = target distribution.

## C. Principles

- P1: Asymmetric update costs (fail −0.2 vs pass +0.1) implement loss aversion; the Beta posterior is the principled replacement — prefer counts over heuristics.
- P2: Verifiable, machine-checkable rewards (RLVR) and machine-checkable falsification conditions (repo) are the same idea from two directions: ground truth must be executable.
- P3: Meta-curiosity (self-model of the world-model) anticipates the literature's internal risk sensors / belief probes; training signal = improvement of the self-model.
- P4: Skill = distilled world model + test. Extraction of executable skills from learned weights parallels Voyager-style skill libraries and distillation practice.
- P5: Modern training stack (GRPO-family + Muon + data-constrained scaling + replay) is convergent with the repo's loop: test → update → propagate → quarantine the unfalsifiable.
