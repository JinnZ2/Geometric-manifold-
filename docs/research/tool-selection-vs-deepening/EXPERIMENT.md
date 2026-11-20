# Tool selection vs deepening: experiment

Status: PROPOSED · not run · co-authored (owner + chat Claude, 2026-11-20)

Filed 2026-11-20 from the owner's order in chat. The block below is the text as received; nothing
in it was edited. Filer's note: the sources are cited as relayed. The filing session did not
read the papers, the repository or the dataset named here, and did not check the figures
(6.31%, +8.41pp, 500 questions). No step was run.

```text
Question: does RL that rewards SELECTING a reasoning tool when uncertainty
rises keep the base model's breadth, while RL that rewards the answer
narrows it?

Sources (OBSERVED, cited):
- R1-Zero "aha" (Guo et al. 2025); base models already self-reflect
  (Sea AI Lab, Liu et al. 2025: DeepSeek-V3-Base, 500 MATH questions)
- arXiv 2601.00514v2 (Princeton, Apr 2026): spontaneous shifts ~6.31% of
  traces, don't rise with training, seldom help; entropy-triggered
  external reconsideration +8.41pp on MATH-500. Scope: Qwen2.5 + Llama,
  GRPO; crosswords, MATH-500, Rush Hour.

Arms (PROPOSED):
  A base model, no RL
  B answer-reward RL (standard)
  C selection-reward RL: reward = right tool chosen at an entropy spike
    (tool set declared up front, e.g. substitution / series /
    numeric step / re-parse), plus answer reward
Measures, per arm:
  breadth   = pass@k at k = 1 … 256 (does the curve keep A's ceiling?)
  selection = tool-choice accuracy at entropy spikes
  shift quality = share of shifts that change the answer and are correct
Prediction (PROPOSED): B rises at pass@1 and falls below A at high k;
  C keeps A's high-k ceiling and improves selection.
Refuted if: C narrows like B, or selection doesn't improve.

Step 1 (stdlib, runnable now): Princeton released traces + code
  (github humans-and-machines/Illusion-of-Reasoning; HF
  od2961/illusion-of-reasoning-main-traces). Re-check: shift rate,
  accuracy given a shift, entropy-gated gain. Different family from
  the authors = the first independent read.
Step 2+ (needs GPU; not available to this owner): arms A/B/C on a small
  model; open slots for whoever can run it.
Slots (VOID): step1_shift_rate, step1_gated_gain, pass_k_curves_ABC,
  selection_acc_C, shift_quality_ABC.
```

Observer slots: null.
