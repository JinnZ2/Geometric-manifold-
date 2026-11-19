# residual-audit

Measurement layer for the Residual Check Protocol. Stdlib-only Python 3.10+,
single file (`residual_audit.py`), phone-runnable. CC0.

Built to sit alongside three sibling repos:

| borrowed from | what |
|---|---|
| `formula-audit` | VOID instead of guesses; intervals over point scores; `UNCHECKED` recorded, never passed |
| `explorative-math` | append-only JSONL archive; `human_interest` / `ai_interest` / `divergence_note` stay null until a person fills them; misses archived, nothing selected |
| `Geometric-manifold-` | the "errors are data" test discipline — the selftest pins honest instrument behavior, including one place where the first test's expectation was wrong, not the instrument |

## What it does

Given a claim, an optional reason chain, and the responses some query target
(a model, a human, an ensemble — always external) gave to progressively
surface-destroyed versions of the claim:

```
residual = post-destruction agreement - reason-chain coverage
```

High residual + thin chain = the claim is traveling by a channel the
explicit reasons do not account for (latent geometry, lineage artifact,
steganographic encoding). The instrument measures the channel, never the
truth value of the claim.

## What it does NOT do (stated scope, not gaps)

- No model API. You paste rung texts into whatever you are auditing and
  paste the responses back. The tool never queries anything.
- Agreement scorers are surface proxies (token containment, char-3-gram
  cosine, token-LCS). They are not semantic judges. Three scorers are
  always reported separately with an interval; a wide interval is
  instrument disagreement and is flagged, never averaged away.
- Order statistics on very short texts are coarse (LCS over ~9 tokens
  floors near 0.44 by chance). Audit realistic-length claims.
- No cross-process write lock on the archive. One writer at a time.
- Destruction methods are deterministic text operations. Modality shifts,
  embedding round-trips, and basin displacement (protocol rungs D3/D5 in
  the sketch) need external machinery; this tool archives their outputs
  fine, it just cannot produce them.

## Use

```sh
python3 residual_audit.py selftest                       # known answers
python3 residual_audit.py destroy examples/owl_claim.json --out runs/demo
# -> runs/demo/ladder_outputs.json: paste each rung output into the target
python3 residual_audit.py score  examples/owl_claim.json --out runs/demo
# -> appends one record to runs/demo/records.jsonl
python3 residual_audit.py compare runs/demo --claim <claim_id>
# -> cross-lineage view; one lineage = kin_only UNCHECKED, never absent
```

Input JSON: `claim`, `reason_chain` (may be empty — RC = 0 recorded, never
filled), `lineage` (model-family tag, nullable), `seed`, optional `ladder`
override, and `responses` with `original` plus one `steps` entry per rung.

## The ladder

`scrub` (case/punct) -> `shuffle_windows` (order) -> `vocab_cut` (lexicon),
plus `char_noise`, `decimate`, `full_shuffle`. Every rung records retention
on three channels — characters, word multiset, word order — and a rung's
strength is capped by its strongest surviving channel. A `full_shuffle`
rung honestly reports `limiting_channel: token_multiset`: the bag of words
survives it. A ladder whose final rung still leaks a channel is flagged
`weak_destruction`, and any residual measured behind it is cheap.

## Reading a record

The demo (`examples/owl_claim.json`) produces: cumulative ladder strength
1.0 (no surface channel survives), S = [1.0, 1.0], P = [0.29, 0.42],
RC = 0 (empty chain), residual = [0.29, 0.42] — which straddles the 0.30
threshold, so the record flags `residual_undecided` instead of collapsing
to a verdict. That is the intended asymmetry: false negatives are easy,
a clean positive must be expensive.

## Where it plugs in

- **Candidate 1 (carrier localization):** swap ladder rungs as
  training-data interventions; archive per-rung transmission deltas.
- **Candidate 2 (kinship):** `compare` across `lineage` tags; pair with
  `Geometric-manifold-`'s spectral instruments on parameter snapshots.
- **Candidate 3 (calibration):** replay honest controls through the same
  ladder to build a null residual distribution before flagging anything.
- **Candidate 4 (receiver-side):** this tool scores behavior; activation
  probes stay external, but their outputs archive here like any response.
