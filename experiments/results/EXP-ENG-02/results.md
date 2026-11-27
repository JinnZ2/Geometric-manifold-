# EXP-ENG-02 results: ablation of the corroboration heuristic (run 2026-11-27)

Design: [`experiments/engine-evidence-audit.md`](../../engine-evidence-audit.md#exp-eng-02--ablation-of-the-corroboration-heuristic).
Code: [`run.py`](run.py), which calls `experiments/sketches/eng_lexical_audit.py::run` unchanged. That function imports the engine's `corroboration()` read-only. Raw output: [`raw.json`](raw.json).

## Environment and provenance

- Commit e814887bed510ef0d4bd67c54820399f0ad088d6 (main, digest of 2026-11-26).
- Python 3.14.8, numpy 2.5.5. CPU: Intel Xeon @ 2.90 GHz, 4 cores.
- The machine was shared with other jobs (load average ~20). Wall time was 1,400 s for 10 seeds. Most of that time was spent re-scoring the deterministic conditions in every seed, and the figure is inflated by the load.
- Data: `data/claim_tree.json` (525 claims) and `data/findings_log.jsonl` (972 findings) as of that commit.
- Seeds: 0–9 for the derangement in condition C, as pre-registered. Conditions A, A1, B and D are deterministic.

## Deviations (recorded before results)

- The heuristic was re-run on the findings log as of e814887, not on the log as it stood at each historical engine run. The design lists this as a threat. Pass counts here are one pass over the current log. They are not the stored `passed` values.
- No other deviations.

## Results

Counts are over (claim, finding) pairs within a topic. "≥3" means the number of claims that would reach at least 3 passes in one pass over the findings.

| Topic | Claims | A pos / ≥3 | A1 pos / ≥3 | B pos / ≥3 | D pos / ≥3 | C ≥3 (seeds 0–9) | C pos-rate ÷ A pos-rate, median (range) |
|---|---|---|---|---|---|---|---|
| loss-landscape (LLG) | 281 | 773 / 113 | 655 / 92 | 1703 / 156 | 3924 / 281 | 0–15 | 0.42 (0.17–0.81) |
| Fisher (FIT) | 43 | 36 / 4 | 26 / 2 | 62 / 7 | 508 / 43 | 0–17 | 0.83 (0.69–0.87) |
| safety drift (SDM) | 147 | 272 / 36 | 251 / 34 | 625 / 62 | 3651 / 147 | 0–43 | 0.38 (0.11–0.73) |
| model collapse (MCR) | 34 | 187 / 34 | 187 / 34 | 152 / 26 | 600 / 34 | 0–8 | 0.03 (0.01–0.08) |
| hidden variable (HVD) | 20 | 5 / 0 | 5 / 0 | 12 / 1 | 39 / 0 | 0–1 | 0.00 (0.00–0.54) |

Ratio of B to A, for claims reaching ≥3 passes:

| Topic | Ratio |
|---|---|
| LLG | 1.38 |
| FIT | 1.75 |
| SDM | 1.72 |
| MCR | 0.76 |
| HVD | 1/0 |

Ratio of B to A, for the per-pair positive rate:

| Topic | Ratio |
|---|---|
| LLG | 2.20 |
| FIT | 1.72 |
| SDM | 2.30 |
| MCR | 0.81 |
| HVD | 2.40 |

In condition C, the derangement seed decides which topic donates the findings, so the range across seeds is wide. For example, FIT claims scored against LLG findings reach ≥3 passes for 17 claims (seeds 1 and 5), against 4 in A. Every seed's mapping is in `raw.json`.

## Verdict against the pre-registered prediction

| Part | Prediction | Outcome |
|---|---|---|
| (i) | D (prefix only) gives ≥3 passes for at least one claim in at least one topic | **SUPPORTED.** D gives ≥3 passes for every claim in 4 of 5 topics (LLG 281/281, FIT 43/43, SDM 147/147, MCR 34/34); HVD 0/20. |
| (ii) | B reduces the ≥3-pass share by ≥30% in ≥3 topics | **FALSIFIED.** The effect goes the opposite way. Removing the prefix raises the number of claims reaching ≥3 passes in LLG, FIT and SDM (+38% to +75%). It lowers it by 24% in MCR only (below the 30% threshold). |
| (iii) | A − A1 > 0 (cross-domain findings contribute passes) | **SUPPORTED** in 3 topics: LLG +118, FIT +10, SDM +21 positive pairs. MCR and HVD: 0. |
| H1 falsifier | B within 20% of A in all topics, AND D zero everywhere, AND C < 0.25×A | **Not met.** D is far from zero, and C ≥ 0.25×A at the median in 3 of 5 topics. |

On H0's own terms:
- H0 says the cross-topic null pass rate is below 0.25 times the own-topic rate. That holds only for MCR (0.03) and HVD (0.00 median). In FIT, findings from another topic pass at 0.83 times the own-topic rate (median).
- H0 says removing the prefix changes the pass rate by less than 20%. That fails in every topic, but in the direction opposite to the one H1 predicted.

Mechanism for (ii), a structural reading of the code that was checked against these counts but not tested separately: the overlap score is |A∩B| / min(|A|, |B|). Dropping the prefix shortens the claim's token set and so shrinks the denominator. Prefix tokens that are absent from a finding then stop diluting the score. The prediction assumed the prefix adds matches. In this scoring it mostly adds denominator.

Dated design note (2026-11-27): prediction (ii) was written without working through the min-denominator normalisation. A content-ablation condition that holds the claim length fixed would separate "the prefix adds matches" from "the prefix changes the normaliser". That condition is not part of this design.

## What this establishes and what it does not

- It establishes which inputs the heuristic responds to on the current data. The topic-name prefix alone is enough to reach ≥3 passes for every claim in 4 topics. Cross-domain findings add passes. Findings from an unrelated topic pass at 38–83% of the own-topic rate in LLG, FIT and SDM (medians).
- It does not establish whether any claim is relevant or true. That needs the ENG-01 labels, which have not been collected.
- It does not reproduce historical pass counts (see the deviation above).
