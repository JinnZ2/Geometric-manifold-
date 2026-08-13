# Plan: Research + Consolidated Notes from curly-octo-happiness

## Stage 1 — Repo Exploration (explore subagent)
- Clone/browse https://github.com/JinnZ2/curly-octo-happiness
- Inventory files, extract all equations, hypotheses, principles, code-level design ideas
- Output: repo content brief

## Stage 2 — External Research (parallel explore subagents)
- A: Newest hypotheses (2024-2026) in AI/ML theory relevant to repo themes
- B: AI training methods & transformer design advances
- C: Neural architecture & learning simulation design
- Cross-validate findings

## Stage 3 — Consolidated Notes (writing)
- One consolidated notes file per domain: hypotheses, AI training, transformer design, neural architecture, learning simulation design
- Each: equations (LaTeX), principles, research notes tying repo content to literature
- Deliverable: markdown notes in /mnt/agents/output/

## Phase 2 — Complexity Engineering / Cybernetics / Advanced Robotics × repo → plan forward
Stage 1: 3 parallel research agents (complexity engineering; cybernetics; advanced robotics), each briefed with repo architecture summary, asked for equations/principles + concrete interaction points.
Stage 2: Orchestrator synthesizes into notes/06 and a PLAN_FORWARD.md (roadmap: concrete modules to build in the repo).

## Phase 3 — Autonomous Hypothesis Engine (GitHub Action)
Deliverables in /mnt/agents/output/hypothesis-engine/ (drop-in for the repo):
- .github/workflows/hypothesis-engine.yml — scheduled + workflow_dispatch, commits artifacts, opens issue on new hypotheses
- scripts/hypothesis_engine.py — explore→log→claim→test→modify claim→hidden-variable scan→consolidate, using grounding/ package
- config/topics.yml — search topics; docs/hypothesis_engine.md
Coder subagent implements; orchestrator validates spec coverage.
