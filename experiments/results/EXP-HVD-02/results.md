# EXP-HVD-02 results: engine claim outcomes vs nuisance covariates (run 2026-11-27)

- Design: [`experiments/hidden-variable-detection-causal-discovery-from-residuals.md`](../../hidden-variable-detection-causal-discovery-from-residuals.md#exp-hvd-02--the-engines-own-residuals-against-measurable-nuisance-covariates).
- Code: [`run.py`](run.py). Raw output: [`raw.json`](raw.json).
- Post-hoc check, not pre-registered: [`posthoc.py`](posthoc.py), output [`posthoc.json`](posthoc.json).

## Environment and provenance

- Commit e814887bed510ef0d4bd67c54820399f0ad088d6.
- Python 3.14.8, numpy 2.5.5, pandas and statsmodels 0.15.0. statsmodels was installed with `pip --user` for this run; the versions are in `raw.json`.
- CPU: Intel Xeon @ 2.90 GHz, 4 cores, shared with other jobs (load average 20–30).
- Runtime: 215 s for the logistic permutations and 172 s for the count-model permutations.
- Data: `data/claim_tree.json` (525 claims) joined on id to `data/findings_log.jsonl` (972 findings). Every claim id occurs exactly once in the log. Nothing in `data/` was changed.
- Permutation seeds: 1 (logistic) and 2 (count model).

## Deviations (recorded before results; also in the `run.py` docstring)

1. **"Same-topic findings logged by the claim's first test run".** This is not stored per claim. It is proxied by the number of same-topic findings with `retrieved_at` ≤ the claim's own `retrieved_at`.
2. **Year-only publication dates.** 96 of 972 findings give only a year. These are imputed as 1 July of that year, and a sensitivity fit is run without them.
3. **Permutations.** Logistic: 2,000 instead of the pre-registered 10,000. Count model: 500. One fit took 0.3 s on the loaded machine. The smallest attainable p is 5×10⁻⁴, which still resolves the 0.05 threshold.
4. **Separation.** `no_abstract` (14 of 15 survived) and `src_crossref` (28 of 29 survived) separate the logistic fit; their coefficients diverge. Under a rule fixed before the final run (|coef| > 10 or se > 50), they are reported with their counts and dropped before CIs are read.
5. **Count model.** The NB2 dispersion α was estimated at about 0 and the Hessian was singular, because `passed` is under-dispersed (survived claims stop at 3). The equivalent Poisson GLM is reported instead. The count-model permutation therefore compares Poisson fits.
6. **Pre-provenance claims.** Identified as claims whose finding has no `lane` field (66 claims). This is a proxy.
7. **Lane.** Every claim comes from the primary lane or has no lane, so a cross-domain lane effect cannot be estimated.

## Results

Coefficients are per SD for the continuous covariates and on the log-odds (logistic) or log-rate (Poisson) scale. Topic is a fixed effect. Full tables, including the sensitivity fits, are in `raw.json`.

| Covariate | Logistic, survived (main), coef [95% CI] | Poisson, passed, coef [95% CI] |
|---|---|---|
| abstract tokens | −0.23 [−0.56, 0.10] | −0.05 [−0.10, 0.01] |
| topic-name tokens in abstract | −0.06 [−0.36, 0.25] | +0.08 [0.02, 0.13] |
| same-topic findings at retrieval | **−4.00 [−5.50, −2.50]** | **−0.21 [−0.36, −0.06]** |
| publication date | +0.21 [−0.31, 0.74] | −0.04 [−0.09, 0.02] |
| retrieved_at | +1.43 [0.76, 2.11] | +0.22 [0.08, 0.35] |
| source = semantic_scholar (vs arxiv) | +3.47 [1.35, 5.59] | +0.47 [0.28, 0.67] |
| no lane (pre-provenance) | −0.85 [−2.37, 0.67] | −0.04 [−0.23, 0.15] |

Permutation test of the full model against topic only (survived permuted within topic):

| Model | Observed LR | Permuted 95th / 99th percentile | p |
|---|---|---|---|
| Logistic | 90.2 | 14.3 / 18.5 | 5.0×10⁻⁴, the floor at B = 2,000; 0 of 2,000 permuted LRs reached it |
| Poisson | 65.8 | 14.7 / 21.3 | 0.002 (B = 500) |

Publication date on its own, with topic fixed effects only, has coefficient −0.85 [−1.36, −0.34]. With the other covariates included it is +0.21 [−0.31, 0.74].

### Sensitivity fits

- **Without pre-provenance claims (459 rows).** The same signs. Same-topic count −4.40 [−6.04, −2.76]. Publication date +0.50 [0.04, 0.97].
- **Without reformulated claims (442 rows).** Same-topic count −3.47 [−5.09, −1.86]. The logistic fit did not report convergence.
- **Without year-only dates (453 rows).** The logistic fit did not converge. Publication date and retrieved_at become collinear (Spearman 0.88) and their CIs blow up, to about ±11. Same-topic count −4.50 [−6.19, −2.80].

### Post-hoc check (not pre-registered)

Within a topic, the same-topic count is a monotone function of retrieved_at, so the two main-model coefficients are not separately interpretable. Fitted one at a time:

| Covariate alone | Logistic coef [95% CI] | Poisson coef [95% CI] |
|---|---|---|
| same-topic count | −1.62 [−2.23, −1.01] | 0.00 [−0.06, 0.07] |
| retrieved_at | −0.38 [−0.68, −0.07] | +0.05 [−0.01, 0.11] |

## Verdict against the pre-registered prediction

The prediction was: same-topic findings at test time and abstract token count each have positive coefficients with CIs excluding 0, and publication date has none once those are included.

**FALSIFIED as stated.**
- The same-topic count has a CI excluding 0, but the sign is negative, not positive, in every fit.
- Abstract token count has a CI that includes 0 in both models.
- The third part did hold: publication date has no effect once the other covariates are in (+0.21 [−0.31, 0.74]), though it does on its own.

The design's falsifier for H1 (no covariate with a CI excluding 0, and permutation p > 0.05) is **not met**.
- Several mechanical covariates have CIs excluding 0: same-topic count, retrieved_at, and source.
- The full model beats the within-topic permutation null (p ≤ 5×10⁻⁴).

So H0 (no covariate beats the null) is rejected, but not in the predicted way.

The likeliest reading is not tested here: it is claim age. Claims retrieved later, after more same-topic findings, have had fewer engine runs and so have not yet reached 3 passes. This is a time-at-risk confound, not a text effect. A survival model with exposure (number of engine runs since the claim entered) would test it. Exposure is not stored per claim.

The source effect stands either way. semantic_scholar claims survive more after adjustment, and crossref claims survive 28 of 29 times (a raw count; the term is separated and could not be estimated). This is consistent with source-dependent text shapes, such as short or missing abstracts, interacting with the overlap score. That reading is UNCHECKED.

## What this establishes and what it does not

- It establishes that, on the data at e814887, claim survival is predicted by bookkeeping variables (retrieval order within topic, source) well beyond a within-topic permutation null. These are the "nuisance" covariates the design asked about. It is a candidate series for stage 6, with the caveat that exposure is missing.
- It does not establish that abstract length drives passes, which was the predicted effect and is not found.
- It does not establish anything about the scientific topics.
- It does not establish causal effects. These are observational regressions on engine output, and two of the covariates are near-collinear.
