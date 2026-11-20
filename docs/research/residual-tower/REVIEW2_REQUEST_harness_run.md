# REVIEW-2 request: run the explore harness on the pin

Status: request written 2026-11-20. Not sent by this file; nothing is ordered from
it. No reply is on record. Report only; co-authored (JinnZ2 + Claude).

## Who

A reader of a different model family from the one that built the harness and made
the first run (Claude family, Claude Code session), WITH code execution: the reader
must be able to run Python 3 (stdlib only) and `git` on a clone of this repository.

The DeepSeek phone app has no code execution, so it cannot take this request. A
reader that cannot run the commands returns nothing here; a description of what the
output would be is not a run.

## What is pinned

| item | value |
|---|---|
| `residual_tower.py` sha256 | `94fcd1e8679a6f65375f4034b00cd8a88f93c0bd14a39128eb47f6e85ece92d9` |
| canonical stack | `dyadic_stack(top_range=4, levels=5, codes_per_level=16)` |
| harness | `docs/research/residual-tower/explore_harness.py` at the commit you clone; state the commit and the file's sha256 |
| commitments | `docs/research/residual-tower/expected/commitments.json` (hashes only) |

If the sha256 of `residual_tower.py` in your clone differs from the pin, say so and
run anyway; the harness records those runs as VOID.

## What to run

From `docs/research/residual-tower/`, with two paths of your own outside the clone:

```sh
python3 explore_harness.py self-test
python3 explore_harness.py run --mode BLIND \
    --runner-family "<your family>" --runner-version "<your version, or void>" \
    --independence <L0-L2> --exposure-file /outside/exposure.json \
    --sealed-out /outside/full_records.jsonl
python3 explore_harness.py run --mode REVIEW \
    --runner-family "<your family>" --runner-version "<your version, or void>" \
    --independence <L0-L2> --exposure-file /outside/exposure.json
python3 explore_harness.py validate
```

- `exposure.json` is a JSON list of strings: everything about residual-tower you
  read or were shown before the run. Write it before the BLIND run.
- If `runner_version` is `void`, the harness writes its own fixed reason into
  `notes`; if your reason differs, say so in your reply.
- Do not run `commit-expected`: it would replace the commitments with your own
  outputs, and the comparison would then be against yourself.
- The runs append to `runs/records.jsonl`. Do not edit or remove earlier lines.

## What to return

1. `runs/records.jsonl` as it stands after your runs (the earlier 18 lines plus yours).
2. Your `exposure.json`.
3. The exit code of each of the four commands, and the commit and harness sha256
   you ran.
4. Your machine and environment: OS, Python version.

Keep `/outside/full_records.jsonl` to yourself; it holds plain outputs.

## What you are exposed to by doing this

State these in `exposure.json`; they do not disqualify the run.

- Cloning the repository gives you `runs/records.jsonl` (sealed hashes),
  `expected/commitments.json` (hashes), the spec and this request.
- The harness's BLIND mode refuses to open `expected/`; you, the reader, can still
  open it. Say whether you did.
- Branch `claude/explore-harness` @ 4d79e45 holds plain outputs. Say whether you
  read it.
- The probes are deterministic, so the seal blocks reading, not deriving. This run
  measures reproducibility across runner, machine and environment
  (EXPLORE_HARNESS_SPEC.md, second addition of 2026-11-20).

## What this request does not ask

- No verdict on residual-tower and no score.
- No changes to any file other than the appended lines in `runs/records.jsonl`.
- No AUDIT or SUGGEST output. Those are separate modes and separate requests.

Observer slots: null.
