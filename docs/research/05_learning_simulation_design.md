# Notes 05 — Learning Simulation Design

## A. Repo simulations

**BumpyWorld** (`grounding/worlds/bumpy.py`) — 1-D physics sandbox:
$$\mathrm{terrain}(x)=0.5\sin x,\quad slope=0.5\cos x,\quad v \mathrel{+}= force - 0.1\cdot slope,\quad v \mathrel{*}= 0.9,\quad x \mathrel{+}= v$$

**Explore/exploit policy:** if $\overline{|error|} > 0.3$ → explore $a \sim U(-1,1)$; else exploit uphill $0.5\,\mathrm{sign}(-slope) + U(-0.3,0.3)$ — error-gated exploration (the agent explores when its model is bad, not on a schedule).

**Unified experiment loop** (`unified_playground.py`): action → predict → step → world-model update → curiosity reward → build machine-checkable claim `abs_diff_lt(actual_x, predicted, 0.3)` with scope + reference class → evaluate → attach to concept node → propagate confidence every 10 steps → big error / falsified claim → UnknownJournal → episodic memory → hardware wear.

**Dream mechanism:** sample ≤5 memory fragments → recombine ("and"-join) → store dream → wake with generated curiosity: "Could there be a link between '{frag0}' and '{frag1}'?" Sleep trigger: novelty saturation / prediction-error threshold; transformations: reversal, metaphor, merging.

**Skill Lab:** extract executable skill from learned weights → run code+test in scratch namespace → confidence updates → refactor = counted escape hatch.

**Hardware stewardship world:** $drift = severity\cdot U(0.5,1.5)$; $health \mathrel{-}= 0.1\,drift$; $v = base(1 + U(-0.2,0.2)(1-health))$; $temp = base + (1-health)\,U(20,80)$; failure modes by health bands; **repurpose philosophy** — failed components reclassified (shorted diode → conductor, drift → sensor, open circuit → antenna, effectiveness 0–9), never discarded.

**Meta-playground daily cycle:** morning exploration → midday relational weave + distributed-self trace → afternoon skills → evening stillness/unknowns → night: 2 dreams + `tree.recalibrate()` + summary. A circadian curriculum.

**Transition Simulator** (`modules/transition.py`): Line vs Torus farm over 20 yr — linear: $yield = \max(0.5,\ 3.5 + 0.1t - 0.005t^2)$, drought ×0.7; torus: regenerative build-up (SOM ≤ 8, yield ≤ 6), drought ×0.85; $resilience = 0.3\,water + 0.3\,biodiversity + 0.4\,(SOM/8)$. Used to falsify the hypothesis "Torus outperforms Line under drought."

## B. Literature (2024–2026)

**DreamerV3** (arXiv:2301.04104; Nature 640:647–653, 2025) — RSSM:
$$h_t=f_\phi(h_{t-1},z_{t-1},a_{t-1}),\quad z_t\sim q_\phi(z_t|h_t,x_t)$$
$$\mathcal L(\phi)=-\ln p_\phi(x_t|z_t,h_t)-\ln p_\phi(r_t|z_t,h_t)+\beta\,\mathrm{KL}[q_\phi\|p_\phi]$$
Actor-critic trained purely in latent imagination; one hyperparameter set, 150+ tasks. *The repo's dream = recombination of episodic fragments is a symbolic analogue of latent imagination rollouts.*

**Genie line** (Genie 1 arXiv:2402.15391; Genie 2, 2024; Genie 3, 2025): latent-action VQ-VAE $l_t = e(z_{t+1}|z_t)$, dynamics $p_\theta(z_{t+1}|z_{1:t},l_t)$ → action-controllable generative worlds from unlabeled video; Genie 3 real-time 720p interactive worlds explicitly positioned as agent training grounds. Sora-as-simulator claim (2024) downgraded by consensus: physics priors without action conditioning = "simulator of appearance, not dynamics."

**Curiosity:** ICM $r^i=\tfrac{\eta}{2}\|\hat\phi(s_{t+1})-\phi(s_{t+1})\|^2$; RND $r^i=\|\hat f(s)-f(s)\|^2$ vs frozen random target; 2024–26: CDE curiosity bonuses for RLVR on LLMs (+~3 AIME pts, arXiv:2509.09675), IB-shaped latent curiosity (~50% fewer steps-to-goal), Absolute Zero / R-Zero self-generated tasks. *The repo's curiosity = error reduction is the learning-progress variant — it intrinsically avoids the noisy-TV trap.*

**Unsupervised Environment Design** (PAIRED arXiv:2012.02096; ACCEL 2203.01302): teacher generates levels maximizing regret $\mathrm{REGRET}=V(\pi^A)-V(\pi^P)$; equilibrium = minimax-regret policy. **This is the formalization of falsification-driven simulation design**: environments that falsify the current policy while remaining solvable.

**LLM-agent self-evolution:** AgentGym/AgentEvol (arXiv:2406.04151): behavioral-cloning bootstrap then exploration → trajectory sampling → self-fine-tuning across 14 sims. Voyager (TMLR 2024): automatic curriculum + executable skill library + self-verification in Minecraft; SkillWeaver (2504.07079): propose–practice–verify–hone skill lifecycle — the scaled version of the repo's Skill Lab.

**JEPA surprise signal (2025–26):** temporal prediction error in latent space as zero-label anomaly/complexity signal (arXiv 2606.28383) — residual error as hidden-variable flag, unifying with HND (Notes 04).

## C. Principles

- P1: Explore when wrong, exploit when calibrated — gate exploration on model error, not ε-schedules.
- P2: Every simulation step should yield a *testable claim*, not just a transition (RLVR-compatible).
- P3: Sleep/dream phases are recombinative memory consolidation plus curiosity generation — budget them into the curriculum (circadian loop).
- P4: Skills must be executable and test-carrying; extraction from learned weights beats hand-authoring.
- P5: Design worlds with failure modes and repurposing paths — graceful degradation is a learnable skill, not an exception.
- P6: The best training environments adversarially falsify the current policy while staying solvable (UED/regret) — a simulation is a hypothesis-testing instrument.
- P7: Include genuinely unfalsifiable mysteries (UnknownJournal) so the agent learns the boundary of knowledge, not just its content.
