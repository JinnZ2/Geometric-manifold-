# Notes 03 — Transformer Design & Encodings

## A. Repo contribution: Gray-coded band-index bitstreams as discrete sensor embeddings

The repo has no attention mechanism; its design contribution is an **encoding layer**: physical sensor values → band index → Gray code → fixed-position bitstream.

**Canonical encoder** (`grounding/core/graycode.py`):
$$idx = \max\{i : value \ge bands[i]\},\qquad g = idx \oplus (idx \gg 1),\qquad bits = \mathrm{bin}_n(g)$$
Inverse: $idx = g;\ \text{while } g:\ g \gg= 1;\ idx \oplus= g$. Default 3 bits = 8 bands.

**Why Gray code (principle):** adjacent band values differ by exactly one bit — Hamming-smooth transitions, robust to threshold jitter; the discrete analogue of Lipschitz-continuous embeddings.

**Bitstream layouts (fixed-position "tokens"):**
- Gravitational plugin: 39 bits, 4 sections (strain-mag log bands 10⁻²²–10⁻¹⁸, chirp mass 1–100 M⊙, freq 10–1000 Hz, polarization, detector-align, delta channels, EM/magnetic coherence, arm length, noise floor).
- HardwareBridgeEncoder: 39 bits (failure mode, health, confidence, voltage/current/temp/noise bands, repurpose class, drill depth). **Thermal-runaway override:** temp-band ≥ 6 AND current-band ≥ 6 ⇒ forced "quarantine".
- Harmony field state: 6 bits (mean/max displacement bands, Gray-coded).
- Affective channels: 3b intensity + 3b delta per channel.

**Adaptive band learning** (`init_bands`): thresholds from sample percentiles [0, 12.5, …, 87.5]; delta bands = range × p/100 — agent self-calibrates to novel sensors (cf. learned tokenizers / VQ codebooks).

**Physics discovery loop** (`physics_discovery.py`): $novelty = \text{fraction of buffered values outside all known band ranges}$; novelty > 0.9 ⇒ propose percentile bands, auto-generate + hot-load a new encoder plugin (`meta_encoder.py`) — **vocabulary extension**: the lexicon itself grows. Analogue: tokenizer/codebook expansion, OOV handling.

**Octahedral canon** (`octahedral_canon.py`): verified involution between two 3-bit vertex-index conventions — $(index \oplus 0b111) \oplus ((index \oplus (index \gg 1)) \mathbin{\&} 1) \times 0b011$; exhaustively verified over 8 cases. Principle: scoped formal verification of index bijections.

**field_adapter EM sensors:** $ambient = \tfrac12(\overline{E_n}+\overline{B_n})$; $coherence = \|\overline{\hat E}\|$; $vigilance = \mathrm{clip}((\max E_n/\overline{E_n}-1)/9,0,1)$; $pressure = \overline{E_n^2 + B_n^2}$ (EM energy density proxy); situational awareness = fraction of points with $E_n > 0.1$.

**Reasoning-style prompting** (`Organize.md`): 10 styles (geometric, formal_logic, spiral_recursive, …) with per-style temperature (0.6 formal / 0.8 exploratory); **cross-style falsification**: re-reason in a different style to break a chain; tree-of-thoughts branching.

## B. Literature: transformer design 2024–2026

**RoPE & extensions:** $f(x_m,m)=R_{\Theta,m}Wx_m$, 2×2 rotations, $\theta_i = b^{-2i/d}$; relative property $\langle q_m,k_n\rangle$ depends on $m-n$. YaRN: $\theta'_i=(1-\gamma_i)\theta_i/s + \gamma_i\theta_i$ + logit temperature $\sqrt{1/t}\approx 0.1\ln s + 1$. LongRoPE, NTK scaling $b' = b\cdot s^{d/(d-2)}$ → million-token contexts.

**MLA (DeepSeek-V2, arXiv:2405.04434):** cache low-rank latent $c_t^{KV}=W^{DKV}h_t$; reconstruct $k_t=W^{UK}c_t^{KV}$; absorption precomputes $q^\top W^{UK}$. ~93% KV-cache reduction; standard in DeepSeek V3/R1, Kimi K2/K3.

**Mamba/selective SSMs (arXiv:2312.00752; SSD 2405.21060):** $h_t=\bar A_t h_{t-1}+\bar B_t x_t$, $y_t=C_t h_t$, input-dependent $\bar B_t,\bar C_t,\Delta_t$; SSD duality $S_t = a_t S_{t-1} + b_t x_t^\top$ unifies SSMs and linear attention.

**Gated delta rule / Kimi Linear (arXiv:2412.06464; 2510.26692):**
$$S_t=\mathrm{Diag}(\alpha_t)\big(S_{t-1}-\beta_t k_t(k_t^\top S_{t-1})\big)+\beta_t v_t k_t^\top,\quad o_t=S_t^\top q_t$$
Erase-then-write associative memory; 3:1 KDA:MLA hybrid beats full attention, −75% KV cache, 6× decode throughput at 1M ctx.

**MoE:** $h_t'=\sum_{i\le N_s}\mathrm{FFN}^{(s)}_i(u_t)+\sum_{i\in TopK}g_{i,t}\mathrm{FFN}^{(r)}_i(u_t)$, $g_{i,t}=\sigma(u_t^\top e_i)$-normalized; **loss-free balancing** (arXiv:2408.15664): route by $s_i + b_i$, gate by $s_i$, update $b_i \mathrel{+}= u\,\mathrm{sign}(\mathrm{error}_i)$.

**Norm-free:** DyT (arXiv:2503.10622): $\mathrm{DyT}(x)=\gamma\tanh(\alpha x)+\beta$ matches LayerNorm/RMSNorm across modalities.

**Sparse attention:** NSA (arXiv:2502.11089): gated mixture of compressed/selected/window blocks; DSA "lightning indexer" in production DeepSeek-V3.2.

**Ultra-low-bit:** BitNet 1.58-bit (arXiv:2402.17764): $W\in\{-1,0,+1\}$, absmean quantization $W_q=\mathrm{RoundClip}(W/(\gamma+\epsilon))$, STE gradients; Kimi K2 native INT4 QAT. **Negative result (verified):** no canonical 2024–2026 paper applies Gray-code structure to token embeddings — the repo's Gray-coded sensor bitstreams occupy a genuine open niche: Hamming-smooth discrete codes could stabilize STE gradients for ultra-low-bit tokens.

## C. Principles

- P1: Encoding design = embedding design. Band thresholds are a learned scalar quantization (percentile codebook); Gray ordering is the smoothness prior.
- P2: Delta channels + coherence bits = the sensor-fusion analogue of cross-attention; cross-modal agreement is encoded, not inferred later.
- P3: Hard override bits (thermal quarantine) are safety-critical logic placed *in the encoding*, not the policy — defense in depth.
- P4: Vocabulary must be extensible at runtime (physics-discovery loop) — a static tokenizer cannot represent novel physics.
- P5: Index conventions between subsystems must be formally verified bijections (octahedral canon), not conventions-by-comment.
