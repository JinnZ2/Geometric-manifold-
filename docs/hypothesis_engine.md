# Hypothesis Engine — Design Doc

An autonomous, **stdlib-only, deterministic** research pipeline for
`Geometric-manifold-`. It explores free scholarly APIs, stakes claims with
explicit falsification conditions, tests them by cross-source verification,
reformulates failures (with escape hatches), scans for hidden variables, and
consolidates surviving claims into hypothesis drafts. No LLM in the loop, so it
runs free on GitHub runners.

It is deliberately decoupled from the torch-based manifold pipeline: no repo
imports, no third-party dependencies. It feeds that pipeline with literature
grounding rather than calling into it. The design is ported from the sibling
`curly-octo-happiness` epistemics work (see `docs/research/`).

## Pipeline

```
                 configs/topics.json
                        |
                        v
   +--------------------------------------------+
   | 1. EXPLORE   arXiv | Semantic Scholar | Crossref
   |     (urllib, timeouts, log-and-continue)   |
   +--------------------------------------------+
                        v
   | 2. LOG     data/findings_log.jsonl (dedup by hash)
   |          + EpisodicMemory append
                        v
   | 3. CLAIM   distill -> Claim(text, falsification, scope, reference_class)
   |            classify_falsifiability:
   |              unfalsifiable -----> data/unknown_journal.jsonl
   |              else ------------> DependencyTree (stake)
                        v
   | 4. TEST    cross-source corroboration/contradiction heuristics
   |            pass -> conf +0.1, fail -> conf -0.2
   |            persist data/claim_tree.json (reload next run)
                        v
   | 5. MODIFY  failed claims -> reformulate() (narrowed scope)
   |            reformulation_count >= 3 -> ESCAPE HATCH -> unknown journal
                        v
   | 6. HIDDEN  residual = |beta_confidence - 0.5| per topic
   |            trigger: mean|residual| >= 0.1 AND |pearson r| > 0.5
   |            -> data/hidden_variables.jsonl (hidden_variable_suggestion)
                        v
   | 7. CONSOLIDATE  hypotheses/<topic-slug>.md (regenerated each run)
   |                 + data/engine_report.md (stdout too)
   +--------------------------------------------+
```

## Stage mapping to repo philosophy

| Stage | Concept |
|---|---|
| 3. claim | **Claim staking** — every finding becomes a `Claim` with an explicit falsification condition, scope, and reference class before entering the tree. Same posture as `docs/theoretical_notes/CLAIM_TABLE.fab.json`: assertions are worthless, staked claims are not. |
| 4. test | **Falsification-first testing** — with no world available, the engine uses cross-source verification as the test oracle: independent corroboration raises confidence, contradiction lowers it. |
| 5. modify | **Escape hatches** — failed claims are `reformulate()`d with narrower scope; at 3 reformulations the claim exits the tree into the unknown journal rather than being infinitely patched. |
| 3/5 | **Unknown journal** — unfalsifiable or escape-hatched content is preserved, flagged, never silently deleted. |
| 6. hidden | **Hidden-node detection** — residual series are correlated against exogenous candidate series; triggers on mean|residual| ≥ 0.1 and |r| > 0.5. This is the literature-side analogue of the drift monitors in `repair/monitors.py`. |
| 2. log | **Episodic memory** — findings are appended to a persistent memory index (`data/episodic_memory.json`). |

## Relationship to the manifold pipeline

The engine produces `hypotheses/*.md` and `data/claim_tree.json`; it does **not**
touch `manifolds/`, `simulation/`, or `repair/`. Surviving claims are inputs for
humans deciding what to test in the simulation — e.g. a surviving claim about
critical slowing down as an early-warning signal is a candidate metric for
`repair/monitors.py`. Nothing flows back automatically; that link is deliberate
manual work, not an autocommit path into the safety machinery.

## Config reference (`configs/topics.json`)

```json
{
  "topics": [
    {
      "name": "<human-readable topic name>",
      "queries": ["<query string 1>", "..."],
      "sources": ["arxiv", "semantic_scholar", "crossref"]
    }
  ]
}
```

- `name` — used for scoping claims, hypothesis file slugs, and hidden-variable grouping.
- `queries` — each is sent to every listed source.
- `sources` — subset of `arxiv`, `semantic_scholar`, `crossref`.

**Adding a topic:** append an entry and commit; the next scheduled run picks it up.

## CLI

```
python scripts/hypothesis_engine.py [--config configs/topics.json] [--dry-run] [--max-per-topic N]
```

- `--dry-run` — skips all network access and uses `scripts/sample_findings.json` (5 synthetic entries across 2 topics, carrying their own `topic` fields independent of `configs/topics.json`). Used by the offline tests in `tests/test_hypothesis_engine.py`.
- `--max-per-topic N` — caps results per query per source.

## Operational notes

- **Idempotency:** findings are deduplicated by a SHA-256 hash of
  `source|title|url`; re-running with the same findings changes nothing. The
  claim tree is persisted in `data/claim_tree.json` and reloaded each run.
- **Rate limits:** the engine sleeps 1s between API calls and caps results;
  Semantic Scholar is unauthenticated (100 req / 5 min shared). Failures are
  logged and the run continues.
- **Timeouts:** every network call goes through `_fetch()` with a 20s timeout.
- **Artifacts & commits:** the workflow uploads `data/` + `hypotheses/` as
  artifacts and commits them back with message
  `chore(engine): weekly research digest <date>`.
- **Issue on new hypotheses:** if `data/engine_report.md` contains the marker
  `NEW HYPOTHESIS` (≥3 surviving claims on a topic), the workflow opens an
  issue with the report body.

## Limitations

- **Heuristic claim extraction** — claims are template-distilled
  ("On topic {topic}, {title} reports: {first sentence of abstract}"), not
  semantically parsed. False positives are expected and handled by staking +
  testing rather than by better parsing.
- **No LLM in the loop** — fully deterministic; quality is bounded by keyword
  overlap, negation heuristics, and shallow numeric extraction.
- **Cross-source "testing" is weak evidence** — corroboration is not
  replication; hypothesis drafts are starting points for human review.
- Crossref/abstract availability varies; findings without abstracts produce
  thin claims that tend to route to the unknown journal.
