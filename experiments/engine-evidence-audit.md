# Engine evidence audit (cross-cutting, all five topics)

Status: PROPOSED. Date 2026-11-24. Co-authored (human + AI).
Source read: `main` at `02320b51b9138dec032f8f85839bce420502a9ec` (engine commit
"weekly research digest 2026-11-23", drafts regenerated 2026-11-23T16:38:05Z).
The engine rewrites `hypotheses/` and `data/` on Mon/Thu; counts below are for that commit only.

Every topic file in this directory depends on the question asked here: what does a
"surviving claim" in a hypothesis draft actually measure? This file collects the gaps that
are properties of the engine rather than of any one topic, so the topic files can point here
instead of repeating them.

Prior art in this repo: `docs/research/HYPOTHESIS_TRIAGE.md` (on the 2026-08-13 runs)
already concluded that claims are template sentences, "survived" means "absence of textual
contradiction", and the loss-landscape topic was ~86% off-topic. This file does not repeat
that argument; it adds what has changed since and turns the open parts into experiments.

## What the engine does (read from code, `scripts/hypothesis_engine.py`)

| stage | behaviour (OBSERVED in code at the commit above) |
|---|---|
| claim | text = `On topic {topic}, {title} reports: {first sentence of abstract}` (`stage_claim`) |
| test | for each claim not yet `survived`/`falsified`, every logged finding with the same `topic` (except itself) is scored by `corroboration()`; +1 adds a pass, −1 a fail (`stage_test`) |
| corroboration | token-set overlap `|A∩B| / min(|A|,|B|)` ≥ 0.25 → related; direction from counts of words in fixed NEGATION / POSITIVE lists |
| status | `survived` at ≥ 3 passes; survived claims are never re-tested; `beta = (1+passed)/(2+passed+failed)` |
| consolidate | statement = `surviving[0]`, i.e. the first surviving claim in tree insertion order, not the highest-confidence one; "established" = ≥ 3 surviving claims |

## Facts on record at this commit (descriptive counts, re-derivable from `data/`)

- `data/claim_tree.json`: 492 claims. `data/findings_log.jsonl`: 906 findings, of which 341
  carry `lane = cross_domain`.
- Surviving claims per draft header: Fisher 14, hidden-variable 4, loss-landscape 249,
  model-collapse 33, safety-drift 72.
- 14 surviving claims have the abstract sentence `(no abstract)` (7 loss-landscape,
  7 model-collapse): their text is the topic prefix plus a title.
- Surviving claims with no `config_hash` (staked before provenance stamping, i.e. under the
  pre-category-filter config): loss-landscape 31, model-collapse 17, safety-drift 4, Fisher 1.
  The tree has no retirement mechanism (the engine reports this count itself as
  `claims_predating_provenance`).
- Every claim id joins to a dated finding (492 / 492), so a date axis for the hidden-variable
  stage is available by a join on `id` (see the hidden-variable topic file).

## Gaps

| id | gap | recorded by engine? |
|---|---|---|
| ENG-G1 | **Retrieval relevance.** On 2026-11-24 the arXiv API echoed the engine-style query `(cat:cs.LG OR cat:cs.AI OR cat:cs.CL) AND all:safety alignment degradation fine-tuning` back as `... AND all:safety OR all:alignment OR all:degradation OR all:fine-tuning`, with `totalResults = 96269`; a phrase/AND form (`all:"safety alignment" AND all:fine-tuning`) returned 320 (OBSERVED, one call each via `curl`, 2026-11-24). With `sortBy=submittedDate` and `max_results=5`, the engine's arXiv lane receives the five newest papers matching any one query word. How the API resolves AND/OR precedence in the first form is UNCHECKED. The off-topic titles in the current drafts (satellite networks, MIMO detection, Hilbert-inequality constants, neutrino cross-sections) are consistent with this, but consistency is not a measurement of relevance. | No. The category filter was added (HYPOTHESIS_TRIAGE), the OR expansion is not mentioned. |
| ENG-G2 | **What `corroboration()` responds to.** Every claim text contains the topic name, and every finding was retrieved with queries built from the same vocabulary, so the overlap test may pass on topic words alone. 14 surviving claims have no abstract sentence at all. | Partly: design doc says quality is "bounded by keyword overlap". The prefix effect is not measured. |
| ENG-G3 | **Cross-domain lane is used as test evidence.** `main()` comments that cross-domain findings are "not evidence about loss landscapes" and routes them out of the claim stage, but `stage_test` reads the full `findings_log.jsonl` with no lane filter, so cross-domain findings do corroborate primary claims. | No. |
| ENG-G4 | **Survival vs relevance.** Nothing measures whether `beta_confidence` or `survived` tracks topical relevance or support; HYPOTHESIS_TRIAGE asked for a per-topic precision estimate. | Asked for, not built. |
| ENG-G5 | **Non-independent "tests".** One claim can collect all 3 passes in a single run from three papers returned by the same query on the same day. Passes are not replications, and they are not independent of each other. | Yes, in general terms ("corroboration is not replication"). |
| ENG-G6 | **Statement selection.** The `> statement` line of each draft is the first surviving claim by insertion order. In the Fisher draft it is the satellite-infrastructure paper (beta 0.80); the top claim is 0.86. This is a property of the code, settled by reading it; no experiment needed. | No. |

None of these says a cited paper is wrong. They bear only on what the drafts' "surviving"
and "established" labels can be taken to mean.

---

## EXP-ENG-01 — Retrieval relevance audit (precision by topic, lane and query form)

Gap: ENG-G1, ENG-G4.

- **Hypothesis.** H1: the fraction of logged findings that are topically relevant to their
  topic is below 0.5 for the arXiv primary lane and rises by ≥ 0.3 when the query is issued
  in phrase/AND form. H0: relevance does not differ between query forms (difference < 0.1).
- **Variables.** Unit: one finding. Outcome: relevance label (relevant / adjacent /
  unrelated) to the topic as stated in `configs/topics.json`, given title + abstract only.
  Factors: topic (5), source (arxiv / semantic_scholar / crossref), lane (primary /
  cross_domain), query form (current / phrase-AND / phrase-AND + `sortBy=relevance`).
- **Design.**
  1. Existing findings: stratified random sample of 60 per topic from `findings_log.jsonl`
     (300 total; smaller topics take all rows if fewer), seed recorded.
  2. New retrievals: for each of the 15 configured queries, fetch top-10 under each of the
     three query forms (450 rows) with a stand-alone script outside `scripts/` (the engine is
     not modified). Store raw API responses and their sha256.
  3. Two raters label blind to source, lane, query form and engine status (shuffle, strip
     those fields). A third rater resolves disagreements. Raters can be humans or model
     families declared per the explore-harness independence ladder
     (`docs/research/residual-tower/EXPLORE_HARNESS_SPEC.md`); record which.
- **Sample size.** 60 per stratum gives a 95% Wilson half-width ≤ 0.13 at p = 0.5; 300
  existing + 450 new rows is a few hours of labelling.
- **Pre-registered prediction.** Precision (relevant + adjacent) of the current arXiv primary
  lane < 0.5 in at least 3 of 5 topics; phrase/AND form raises it by ≥ 0.3 in those topics.
  **Falsified if** current-form precision ≥ 0.7 in ≥ 3 topics, or the query-form difference is
  < 0.1 with the CI excluding 0.3.
- **Analysis.** Per-stratum precision with Wilson CIs; Cohen's κ between raters (report it;
  if κ < 0.4 the labels are not usable and the experiment is reported as VOID rather than
  re-labelled until agreement looks good); logistic regression relevance ~ topic + source +
  lane + form.
- **Threats.** "Relevant" is a judgement; write the rubric first, with 5 anchor examples per
  label. API results drift over time (store raw responses). Raters may recognise
  engine-favoured titles; blinding covers fields, not memory.
- **Tier.** Laptop CPU + rater time; network needed for step 2 only.
- **Would establish:** how much of each topic's input stream is on-topic, and whether the
  query form is a cause. **Would not establish:** anything about the truth of any paper, or
  whether relevant papers support the repo's hypotheses.

## EXP-ENG-02 — Ablation of the corroboration heuristic

Gap: ENG-G2, ENG-G3.

- **Hypothesis.** H1: a large part of the passes come from the topic-name prefix and from
  cross-domain findings, not from the paper-specific content of a claim. H0: passes depend
  on claim content: removing the prefix changes the pass rate by < 20% (relative), and the
  cross-topic null pass rate is < 0.25× the own-topic rate.
- **Conditions** (all use the engine's own `corroboration()`, imported read-only):
  A own-topic as stored; A1 own-topic primary lane only; B prefix removed; C cross-topic
  null (findings from a different topic, seeded derangement); D the prefix alone
  (`On topic <topic>,`) as the claim text.
- **Measured.** Per topic and condition: +1 / −1 / 0 counts over (claim, finding) pairs; number
  of claims that would reach ≥ 3 passes in one pass over the findings.
- **Pre-registered prediction.** (i) D (prefix only) yields ≥ 3 passes for at least one claim
  in at least one topic; (ii) B reduces the share of claims reaching ≥ 3 passes by ≥ 30%
  relative to A in at least 3 topics; (iii) A − A1 > 0 (cross-domain findings contribute
  passes). **Falsified (for H1) if** B is within 20% of A in all topics, D gives zero passes
  everywhere, and C is < 0.25× A.
- **Analysis.** Deterministic; no sampling error, except the seed for the derangement in C
  (run seeds 0–9 and report the range). Report the counts; no significance test.
- **Threats.** The heuristic is re-run on the current findings log, not the log as it stood
  at each historical run, so it does not reproduce past pass counts exactly. Say so in the
  output.
- **Tier.** Laptop CPU, seconds to minutes.
- **Would establish:** which inputs the heuristic responds to. **Would not establish:**
  whether any claim is relevant or true (that needs ENG-01 labels).
- **Sketch:** [`sketches/eng_lexical_audit.py`](sketches/eng_lexical_audit.py)
  (`--selftest` passes; the full run on `data/` has not been done).

## EXP-ENG-03 — Does survival predict relevance?

Gap: ENG-G4, ENG-G5. Depends on ENG-01 labels.

- **Hypothesis.** H1: among claims, `beta_confidence` predicts the ENG-01 relevance label
  (AUC ≥ 0.65). H0: AUC ≤ 0.55.
- **Design.** Join ENG-01 labels for findings that are also claims (claim id = finding id).
  Covariates that the lexical heuristic should be sensitive to: abstract length, token count,
  number of topic-name tokens in the abstract, lane, source.
- **Pre-registered prediction.** AUC of `beta_confidence` for relevant-vs-unrelated is in
  [0.45, 0.6] (no better than chance); abstract token count predicts `passed` better than the
  relevance label does. **Falsified if** AUC ≥ 0.65 with the 95% bootstrap CI above 0.55.
- **Analysis.** AUC with 2,000-resample bootstrap CI; permutation test of labels (10,000);
  Poisson or negative-binomial regression of `passed` on relevance + covariates.
- **Threats.** Survived claims are frozen after 3 passes, so `passed` is truncated; use
  `survived` (binary) as a second outcome. Pre-provenance claims (no `config_hash`) come from
  an older config; analyse with and without them.
- **Tier.** Laptop CPU.
- **Would establish:** whether the drafts' confidence numbers carry information about
  relevance. **Would not establish:** whether they carry information about truth.

## Not an experiment: ENG-G6

The statement-selection rule is settled by the code (`stage_consolidate`, `surviving[0]`).
Recorded here so the topic files do not treat a draft's headline sentence as the engine's
best-supported claim.
