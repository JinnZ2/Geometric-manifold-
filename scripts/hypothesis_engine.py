#!/usr/bin/env python3
"""Hypothesis Engine — autonomous epistemic research pipeline for Geometric-manifold-.

Stages (orchestrated by main()):
    1. explore            — query arXiv / Semantic Scholar / Crossref for new findings.
    2. log                — deduplicate and append to data/findings_log.jsonl + episodic memory.
    3. claim              — distill findings into Claim objects; route unfalsifiable to
                            data/unknown_journal.jsonl, stake the rest in a DependencyTree.
    4. test               — cross-source verification (corroboration/contradiction heuristics).
    5. modify claim       — reformulate failed claims; escape-hatch (>=3) -> unknown journal.
    6. hidden variables   — HND-style scan for exogenous series correlated with residuals.
    7. consolidate        — regenerate hypotheses/<topic>.md and data/engine_report.md.

Stdlib only (Python 3.11+). All network access is behind small functions with timeouts.
The epistemic primitives (Claim, DependencyTree, falsifiability classification) are
self-contained here so the engine and its tests run with no repo imports and no
third-party dependencies — it is deliberately decoupled from the torch-based
manifold pipeline it feeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------
# Epistemic primitives (stdlib-only, self-contained)
# --------------------------------------------------------------------------

def classify_falsifiability(text: str) -> str:
    """Classify a claim as machine-checkable / falsifiable / unfalsifiable."""
    lowered = text.lower()
    numeric = re.search(r"\d+(\.\d+)?\s*(%|percent|x|times)?", lowered)
    quantitative_words = ("accuracy", "correlation", "r =", "p <", "improve", "reduc",
                          "outperform", "increase", "decrease", "error", "score")
    if numeric and any(w in lowered for w in quantitative_words):
        return "machine-checkable"
    hedged = ("might", "could", "perhaps", "may suggest", "we believe", "possibly",
              "in some sense", "arguably")
    if any(h in lowered for h in hedged) and not numeric:
        return "unfalsifiable"
    return "falsifiable"


@dataclass
class Claim:
    """A staked claim with a falsification condition and a test record."""
    text: str
    falsification: str
    confidence: float = 0.5
    passed: int = 0
    failed: int = 0
    logical_form: str = ""
    scope: dict[str, Any] = field(default_factory=dict)
    reference_class: str = ""
    reformulation_count: int = 0
    id: str = ""
    source_url: str = ""
    status_override: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = hashlib.sha256(self.text.encode()).hexdigest()[:16]

    @property
    def beta_confidence(self) -> float:
        return (1 + self.passed) / (2 + self.passed + self.failed)

    @property
    def status(self) -> str:
        if self.status_override:
            return self.status_override
        if self.failed >= 3:
            return "falsified"
        if self.passed >= 3:
            return "survived"
        return "active"

    def evaluate(self, outcome: bool) -> None:
        if outcome:
            self.passed += 1
            self.confidence = min(1.0, self.confidence + 0.1)
        else:
            self.failed += 1
            self.confidence = max(0.0, self.confidence - 0.2)

    def reformulate(self, restricting_condition: str) -> None:
        self.passed = 0
        self.failed = 0
        self.confidence = 0.5
        self.reformulation_count += 1
        self.text = f"{self.text} [restricted: {restricting_condition}]"
        conds = self.scope.setdefault("restrictions", [])
        conds.append(restricting_condition)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["beta_confidence"] = self.beta_confidence
        d["status"] = self.status
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Claim":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**d)


class DependencyTree:
    """Container of staked claims with confidence propagation."""
    def __init__(self) -> None:
        self.claims: dict[str, Claim] = {}

    def add_claim(self, claim: Claim) -> Claim:
        self.claims[claim.id] = claim
        return claim

    def propagate_confidence(self) -> None:
        for c in self.claims.values():
            c.confidence = max(0.0, min(1.0, c.beta_confidence))

    def to_dict(self) -> dict[str, Any]:
        return {"claims": [c.to_dict() for c in self.claims.values()]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DependencyTree":
        tree = cls()
        for cd in d.get("claims", []):
            tree.add_claim(Claim.from_dict(cd))
        return tree


# --------------------------------------------------------------------------
# Finding model
# --------------------------------------------------------------------------

@dataclass
class Finding:
    id: str
    source: str
    title: str
    url: str
    date: str
    topic: str
    abstract: str
    retrieved_at: str
    hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def finding_hash(source: str, title: str, url: str) -> str:
    key = f"{source}|{title.strip().lower()}|{url.strip().lower()}"
    return hashlib.sha256(key.encode()).hexdigest()[:24]


# --------------------------------------------------------------------------
# Network helpers (all behind timeouts, graceful failure)
# --------------------------------------------------------------------------

TIMEOUT = 20
USER_AGENT = "hypothesis-engine/1.0 (github.com/JinnZ2/Geometric-manifold-)"


def _fetch(url: str) -> bytes | None:
    """GET a URL with timeout; return None on any failure (log and continue)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        log(f"  [explore] fetch failed {url[:100]}: {exc}")
        return None


def query_arxiv(query: str, max_results: int) -> list[dict[str, str]]:
    """Query the arXiv Atom API; return raw finding dicts."""
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}", "start": 0,
        "max_results": max_results, "sortBy": "submittedDate",
    })
    raw = _fetch(f"https://export.arxiv.org/api/query?{params}")
    if not raw:
        return []
    out: list[dict[str, str]] = []
    try:
        root = ET.fromstring(raw)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("a:entry", ns):
            title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
            abstract = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
            url = (entry.findtext("a:id", default="", namespaces=ns) or "").strip()
            date = (entry.findtext("a:published", default="", namespaces=ns) or "")[:10]
            arxiv_id = url.rsplit("/", 1)[-1]
            out.append({"source": "arxiv", "title": re.sub(r"\s+", " ", title),
                        "abstract": abstract, "url": url, "date": date, "ext_id": arxiv_id})
    except ET.ParseError as exc:
        log(f"  [explore] arXiv parse error: {exc}")
    return out


def query_semantic_scholar(query: str, max_results: int) -> list[dict[str, str]]:
    """Query the Semantic Scholar Graph API; return raw finding dicts."""
    params = urllib.parse.urlencode({
        "query": query, "limit": min(max_results, 100),
        "fields": "title,abstract,year,citationCount,externalIds",
    })
    raw = _fetch(f"https://api.semanticscholar.org/graph/v1/paper/search?{params}")
    if not raw:
        return []
    out: list[dict[str, str]] = []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        log(f"  [explore] S2 parse error: {exc}")
        return []
    for p in data.get("data", []) or []:
        doi = (p.get("externalIds") or {}).get("DOI", "")
        url = f"https://doi.org/{doi}" if doi else f"https://www.semanticscholar.org/paper/{p.get('paperId','')}"
        out.append({"source": "semantic_scholar",
                    "title": (p.get("title") or "").strip(),
                    "abstract": (p.get("abstract") or "").strip(),
                    "url": url, "date": str(p.get("year") or ""),
                    "citations": str(p.get("citationCount") or 0),
                    "ext_id": doi or p.get("paperId", "")})
    return out


def query_crossref(query: str, max_results: int) -> list[dict[str, str]]:
    """Query Crossref works API (recent items); return raw finding dicts."""
    params = urllib.parse.urlencode({
        "query": query, "rows": max_results,
        "filter": "from-pub-date:2020-01-01", "select": "DOI,title,abstract,published,URL",
    })
    raw = _fetch(f"https://api.crossref.org/works?{params}")
    if not raw:
        return []
    out: list[dict[str, str]] = []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        log(f"  [explore] crossref parse error: {exc}")
        return []
    for item in data.get("message", {}).get("items", []) or []:
        title = " ".join(item.get("title") or []).strip()
        if not title:
            continue
        published = item.get("published", {}).get("date-parts", [[None]])[0]
        out.append({"source": "crossref", "title": title,
                    "abstract": re.sub(r"<[^>]+>", "", item.get("abstract", "") or "").strip(),
                    "url": item.get("URL", ""), "date": str(published[0] or ""),
                    "ext_id": item.get("DOI", "")})
    return out


SOURCES = {"arxiv": query_arxiv, "semantic_scholar": query_semantic_scholar,
           "crossref": query_crossref}


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[hypothesis-engine] {msg}", flush=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def tokenize(text: str) -> set[str]:
    stop = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with",
            "we", "is", "are", "that", "this", "by", "as", "at", "from", "it"}
    return {t for t in re.findall(r"[a-z0-9]+", text.lower())
            if len(t) > 2 and t not in stop}


def first_sentence(text: str, max_len: int = 300) -> str:
    m = re.search(r"(?<=[.!?])\s", text.strip())
    s = text[:m.start() + 1] if m else text
    return s[:max_len].strip()


def pearson_r(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation; 0.0 if degenerate."""
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    xs, ys = xs[:n], ys[:n]
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_config(path: Path) -> list[dict[str, Any]]:
    """Load topics config (JSON: list of {name, queries[], sources[]})."""
    data = json.loads(path.read_text(encoding="utf-8"))
    topics = data["topics"] if isinstance(data, dict) else data
    for t in topics:
        t.setdefault("sources", ["arxiv", "semantic_scholar", "crossref"])
        t.setdefault("queries", [t["name"]])
    return topics


# --------------------------------------------------------------------------
# Stage 1: explore
# --------------------------------------------------------------------------

def stage_explore(topics: list[dict[str, Any]], max_per_topic: int,
                  dry_run: bool, sample_path: Path) -> list[Finding]:
    """Query scholarly APIs (or sample data in dry-run) for each topic."""
    findings: list[Finding] = []
    retrieved = now_iso()
    if dry_run:
        log("[explore] DRY-RUN: using bundled sample findings")
        for row in json.loads(sample_path.read_text(encoding="utf-8")):
            topic = row.get("topic", topics[0]["name"] if topics else "misc")
            findings.append(Finding(
                id="", source=row["source"], title=row["title"], url=row["url"],
                date=row.get("date", ""), topic=topic, abstract=row.get("abstract", ""),
                retrieved_at=retrieved, hash=""))
        return findings
    for topic in topics:
        name = topic["name"]
        for query in topic["queries"]:
            for source in topic["sources"]:
                fn = SOURCES.get(source)
                if not fn:
                    log(f"  [explore] unknown source {source!r}, skipping")
                    continue
                for raw in fn(query, max_per_topic):
                    findings.append(Finding(
                        id="", source=raw["source"], title=raw["title"], url=raw["url"],
                        date=raw.get("date", ""), topic=name,
                        abstract=raw.get("abstract", ""), retrieved_at=retrieved, hash=""))
                time.sleep(1.0)  # be polite to free APIs
    return findings


# --------------------------------------------------------------------------
# Stage 2: log (dedup + persist)
# --------------------------------------------------------------------------

def stage_log(findings: list[Finding], log_path: Path) -> tuple[list[Finding], int]:
    """Deduplicate by DOI/arxiv-id/title hash against the findings log.

    Returns (new_findings, skipped_count). Appends new findings to the log.
    """
    seen = {row.get("hash") for row in read_jsonl(log_path)}
    new: list[Finding] = []
    skipped = 0
    seen_now: set[str] = set()
    for f in findings:
        f.hash = finding_hash(f.source, f.title, f.url)
        f.id = f.hash
        if f.hash in seen or f.hash in seen_now or not f.title:
            skipped += 1
            continue
        seen_now.add(f.hash)
        new.append(f)
    if new:
        append_jsonl(log_path, (f.to_dict() for f in new))
        # Episodic memory: a flat, append-only index of everything ever seen.
        mem_path = log_path.parent / "episodic_memory.json"
        try:
            memory = read_jsonl(mem_path)
            memory.extend({"event": "finding_logged", "id": f.id, "topic": f.topic,
                           "text": f"{f.title} {f.abstract[:200]}",
                           "at": f.retrieved_at} for f in new)
            mem_path.write_text("\n".join(json.dumps(r, ensure_ascii=False)
                                          for r in memory) + "\n", encoding="utf-8")
        except OSError as exc:
            log(f"  [log] memory write failed: {exc}")
    return new, skipped


# --------------------------------------------------------------------------
# Stage 3: claim
# --------------------------------------------------------------------------

def derive_falsification(topic: str, abstract: str) -> str:
    """Heuristic falsification condition for a distilled claim.

    If the abstract contains quantitative effect language, require numeric
    replication; otherwise require independent replication/reanalysis.
    """
    if re.search(r"(\d+(\.\d+)?\s*%|p\s*[<=]\s*0?\.\d+|r\s*=\s*0?\.\d+)", abstract):
        return ("Numeric check: an independent replication or reanalysis fails to "
                "reproduce the reported quantitative effect within a factor of 2.")
    return ("Replication check: an independent replication or reanalysis of the "
            "reported result contradicts it on the same task/metric class.")


def stage_claim(new_findings: list[Finding], tree: DependencyTree,
                unknown_path: Path) -> tuple[list[Claim], int]:
    """Convert findings to claims; route unfalsifiable to the unknown journal."""
    made: list[Claim] = []
    unknowns: list[dict[str, Any]] = []
    for f in new_findings:
        text = f"On topic {f.topic}, {f.title} reports: {first_sentence(f.abstract) or '(no abstract)'}"
        falsification = derive_falsification(f.topic, f.abstract)
        cls = classify_falsifiability(f"{text} {falsification}")
        if cls == "unfalsifiable":
            unknowns.append({"id": f.id, "topic": f.topic, "text": text,
                             "reason": "classify_falsifiability -> unfalsifiable",
                             "at": now_iso(), "flag": "unfalsifiable"})
            continue
        claim = Claim(text=text, falsification=falsification,
                      logical_form="abs_diff_lt" if "Numeric check" in falsification else "",
                      scope={"topic": f.topic, "source": f.source},
                      reference_class=f"{f.topic} / {f.source} findings",
                      id=f.id, source_url=f.url)
        tree.add_claim(claim)
        made.append(claim)
    if unknowns:
        append_jsonl(unknown_path, unknowns)
    return made, len(unknowns)


# --------------------------------------------------------------------------
# Stage 4: test (cross-source verification)
# --------------------------------------------------------------------------

NEGATION = {"not", "no", "never", "fails", "fail", "contradict", "refute", "however",
            "unable", "cannot", "worse", "degrade", "underperform", "limitation"}
POSITIVE = {"confirm", "demonstrate", "show", "outperform", "improve", "achieve",
            "consistent", "corroborate", "validate", "reproduce"}


def corroboration(a: str, b: str) -> int:
    """Heuristic: +1 corroborate, -1 contradict, 0 unrelated.

    Keyword overlap establishes relatedness; negation vs. positive cue balance
    decides direction. Documented heuristic — no LLM in the loop.
    """
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0
    overlap = len(ta & tb) / max(1, min(len(ta), len(tb)))
    if overlap < 0.25:
        return 0
    na = len(tokenize(a) & NEGATION) - len(tokenize(a) & POSITIVE)
    nb = len(tokenize(b) & NEGATION) - len(tokenize(b) & POSITIVE)
    if (na > 0) != (nb > 0) and abs(na - nb) >= 1:
        return -1
    return 1


def stage_test(tree: DependencyTree, all_findings: list[dict[str, Any]]) -> dict[str, int]:
    """Test claims by cross-source verification against logged findings."""
    stats = {"pass": 0, "fail": 0, "untested": 0}
    by_topic: dict[str, list[dict[str, Any]]] = {}
    for row in all_findings:
        by_topic.setdefault(row.get("topic", ""), []).append(row)
    for claim in tree.claims.values():
        if claim.status in ("falsified", "survived"):
            continue
        tested = False
        for row in by_topic.get(claim.scope.get("topic", ""), []):
            if row.get("hash") == claim.id:
                continue  # skip self
            verdict = corroboration(
                claim.text, f"{row.get('title','')} {row.get('abstract','')}")
            if verdict > 0:
                claim.evaluate(True)
                stats["pass"] += 1
                tested = True
            elif verdict < 0:
                claim.evaluate(False)
                stats["fail"] += 1
                tested = True
        if not tested:
            stats["untested"] += 1
    tree.propagate_confidence()
    return stats


# --------------------------------------------------------------------------
# Stage 5: modify claim
# --------------------------------------------------------------------------

def stage_modify(tree: DependencyTree, unknown_path: Path,
                 reform_log: Path) -> dict[str, int]:
    """Reformulate failed claims; escape-hatch (>=3 reformulations) to unknown journal."""
    stats = {"reformulated": 0, "escape_hatched": 0}
    to_remove: list[str] = []
    unknowns: list[dict[str, Any]] = []
    reforms: list[dict[str, Any]] = []
    for claim in tree.claims.values():
        if claim.failed >= 2 or claim.status == "falsified":
            condition = f"narrower scope within {claim.scope.get('topic','?')} (run {now_iso()[:10]})"
            claim.reformulate(condition)
            reforms.append({"id": claim.id, "count": claim.reformulation_count,
                            "condition": condition, "at": now_iso()})
            stats["reformulated"] += 1
            if claim.reformulation_count >= 3:
                unknowns.append({"id": claim.id, "topic": claim.scope.get("topic", ""),
                                 "text": claim.text, "flag": "escape-hatch",
                                 "reason": "reformulation_count >= 3", "at": now_iso()})
                to_remove.append(claim.id)
                stats["escape_hatched"] += 1
    for cid in to_remove:
        del tree.claims[cid]
    if reforms:
        append_jsonl(reform_log, reforms)
    if unknowns:
        append_jsonl(unknown_path, unknowns)
    return stats


# --------------------------------------------------------------------------
# Stage 6: hidden variables (HND-style scan)
# --------------------------------------------------------------------------

# Candidates that are computed FROM the residual's own source and can never be evidence
# of a hidden variable. The residual is |beta_confidence - 0.5|, so:
#   - claim_outcomes  is (passed - failed), which determines beta_confidence
#   - confidence_trend IS beta_confidence, so whenever every claim sits on one side of 0.5
#     the residual is an exact affine function of it and Pearson r is 1.0 by construction
# The second was found on the engine's first live run, which reported r = 1.0 on two topics
# with min(beta_confidence) = 0.5 -- a self-correlation, not a discovery.
ENDOGENOUS_CANDIDATES = frozenset({
    "exogenous:claim_outcomes",
    "exogenous:confidence_trend",
})


def stage_hidden(tree: DependencyTree, findings: list[dict[str, Any]],
                 out_path: Path) -> list[dict[str, Any]]:
    """Correlate per-topic residual series against exogenous candidate series.

    residual = |beta_confidence - 0.5| per claim (ordered by log date).
    Trigger: mean|residual| >= 0.1 and |r| > 0.5 (mirrors modules/hnd.py).
    """
    suggestions: list[dict[str, Any]] = []
    by_topic: dict[str, list[Claim]] = {}
    for c in tree.claims.values():
        by_topic.setdefault(c.scope.get("topic", ""), []).append(c)
    # Candidate exogenous series from the findings log.
    dates = sorted({row.get("date", "")[:7] for row in findings if row.get("date")})
    findings_per_period = [sum(1 for r in findings if r.get("date", "")[:7] == d)
                           for d in dates]
    source_counts: dict[str, int] = {}
    for r in findings:
        source_counts[r.get("source", "?")] = source_counts.get(r.get("source", "?"), 0) + 1
    for topic, claims in by_topic.items():
        residuals = [abs(c.beta_confidence - 0.5) for c in claims]
        if len(residuals) < 3 or (sum(residuals) / len(residuals)) < 0.1:
            continue
        outcomes = [float(c.passed - c.failed) for c in claims]
        conf_series = [c.beta_confidence for c in claims]
        n = len(residuals)
        candidates = {
            "exogenous:findings_rate": [float(findings_per_period[i % len(findings_per_period)])
                                        for i in range(n)] if findings_per_period else [],
            "exogenous:claim_outcomes": outcomes,
            "exogenous:confidence_trend": conf_series,
            "exogenous:source_diversity": [float(len(source_counts))] * n,
        }
        for name, series in candidates.items():
            if name in ENDOGENOUS_CANDIDATES:
                continue
            if len(set(series)) < 2:
                continue  # constant series: Pearson r is undefined, not zero
            r = pearson_r(residuals, series)
            if abs(r) > 0.5:
                suggestions.append({
                    "topic": topic, "suggested_variable": f"hidden:{topic} vs {name}",
                    "pearson_r": round(r, 4),
                    "mean_abs_residual": round(sum(residuals) / len(residuals), 4),
                    "confidence": round(min(0.95, abs(r)), 4), "at": now_iso(),
                    "type": "hidden_variable_suggestion"})
    if suggestions:
        append_jsonl(out_path, suggestions)
    return suggestions


# --------------------------------------------------------------------------
# Stage 7: consolidate
# --------------------------------------------------------------------------

def stage_consolidate(tree: DependencyTree, topics: list[dict[str, Any]],
                      unknown_path: Path, hidden_path: Path,
                      hypotheses_dir: Path,
                      announced_path: Path | None = None) -> dict[str, Any]:
    """Regenerate hypotheses/<topic-slug>.md from the claim tree."""
    hypotheses_dir.mkdir(parents=True, exist_ok=True)
    unknowns = read_jsonl(unknown_path)
    hidden = read_jsonl(hidden_path)
    by_topic: dict[str, list[Claim]] = {}
    for c in tree.claims.values():
        by_topic.setdefault(c.scope.get("topic", ""), []).append(c)
    new_hypotheses: list[str] = []
    announced: set[str] = set()
    if announced_path and announced_path.exists():
        try:
            announced = set(json.loads(announced_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            announced = set()
    for topic in {t["name"] for t in topics} | set(by_topic):
        claims = by_topic.get(topic, [])
        surviving = [c for c in claims if c.status == "survived" or c.beta_confidence >= 0.7]
        refuted = [c for c in claims if c.status == "falsified"]
        suspects = [h for h in hidden if h.get("topic") == topic][-5:]
        open_unknowns = [u for u in unknowns if u.get("topic") == topic][-10:]
        if not surviving and not claims:
            continue
        slug = slugify(topic)
        lines = [f"# Hypothesis draft: {topic}", "",
                 f"_Regenerated {now_iso()} by hypothesis_engine.py — do not hand-edit._", ""]
        if surviving:
            lines += ["## Hypothesis statement", ""]
            lines.append("Across corroborated findings, the following claims survived "
                         "staked testing (beta_confidence >= 0.7 or status=survived):")
            lines.append("")
            lines.append("> " + surviving[0].text)
            lines.append("")
            # "New" means crossing the threshold on THIS run. Without the announced-set
            # check the condition stays true forever once met, so the workflow opens an
            # identical issue on every scheduled run -- observed as issues #8 and #10,
            # same title, same two topics, 11 minutes apart.
            established = len(surviving) >= 3
            first_time = established and topic not in announced
            if established:
                marker = "NEW HYPOTHESIS" if first_time else "established"
            else:
                marker = "candidate"
            if first_time:
                new_hypotheses.append(topic)
                announced.add(topic)
            lines.append(f"Status: **{marker}** ({len(surviving)} surviving claims)")
            lines.append("")
        lines += ["## Supporting claims", ""]
        for c in sorted(surviving, key=lambda c: -c.beta_confidence):
            lines.append(f"- ({c.beta_confidence:.2f}) {c.text} — [source]({c.source_url})")
        lines += ["", "## Contradicted/refuted claims", ""]
        for c in refuted:
            lines.append(f"- ({c.beta_confidence:.2f}) {c.text} — falsified {c.failed}x")
        if not refuted:
            lines.append("- (none)")
        lines += ["", "## Hidden-variable suspects", ""]
        for h in suspects:
            lines.append(f"- `{h.get('suggested_variable')}` r={h.get('pearson_r')} "
                         f"conf={h.get('confidence')}")
        if not suspects:
            lines.append("- (none)")
        lines += ["", "## Open unknowns", ""]
        for u in open_unknowns:
            lines.append(f"- [{u.get('flag','?')}] {u.get('text','')[:160]}")
        if not open_unknowns:
            lines.append("- (none)")
        (hypotheses_dir / f"{slug}.md").write_text("\n".join(lines) + "\n",
                                                   encoding="utf-8")
    if announced_path:
        announced_path.write_text(json.dumps(sorted(announced), indent=2) + "\n",
                                  encoding="utf-8")
    return {"hypothesis_files": len(list(hypotheses_dir.glob("*.md"))),
            "new_hypotheses": new_hypotheses}


# --------------------------------------------------------------------------
# Tree persistence + report
# --------------------------------------------------------------------------

def save_tree(tree: DependencyTree, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tree.to_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8")


def load_tree(path: Path) -> DependencyTree:
    if not path.exists():
        return DependencyTree()
    try:
        return DependencyTree.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        log(f"[tree] failed to load {path}: {exc}; starting fresh")
        return DependencyTree()


def write_report(path: Path, stats: dict[str, Any]) -> str:
    lines = ["# Hypothesis engine run report", "",
             f"Run at: {now_iso()}", "",
             "## Stage counts", ""]
    for k, v in stats.items():
        lines.append(f"- **{k}**: {v}")
    new = stats.get("new_hypotheses") or []
    lines += ["", "## New hypotheses", ""]
    if new:
        lines.insert(len(lines) - 1, "NEW HYPOTHESIS")  # marker for the workflow
        for t in new:
            lines.append(f"- {t}")
    else:
        lines.append("- (none this run)")
    suspects = stats.get("hidden_variable_suggestions") or []
    lines += ["", "## Top hidden-variable suspects", ""]
    if suspects:
        for s in sorted(suspects, key=lambda s: -abs(s.get("pearson_r", 0)))[:5]:
            lines.append(f"- {s.get('suggested_variable')} (r={s.get('pearson_r')})")
    else:
        lines.append("- (none)")
    body = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return body


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Autonomous hypothesis engine (stdlib-only).")
    parser.add_argument("--config", default="configs/topics.json", help="topics JSON config")
    parser.add_argument("--dry-run", action="store_true",
                        help="skip network; use scripts/sample_findings.json")
    parser.add_argument("--max-per-topic", type=int, default=5)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--hypotheses-dir", default="hypotheses")
    parser.add_argument("--sample", default="scripts/sample_findings.json")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    log_path = data_dir / "findings_log.jsonl"
    tree_path = data_dir / "claim_tree.json"
    unknown_path = data_dir / "unknown_journal.jsonl"
    hidden_path = data_dir / "hidden_variables.jsonl"
    reform_path = data_dir / "reformulations.jsonl"

    topics = load_config(Path(args.config))
    log(f"config: {len(topics)} topics")

    # 1. explore
    findings = stage_explore(topics, args.max_per_topic, args.dry_run, Path(args.sample))
    log(f"[explore] {len(findings)} raw findings")

    # 2. log
    new_findings, skipped = stage_log(findings, log_path)
    log(f"[log] {len(new_findings)} new, {skipped} duplicates skipped")

    # 3. claim
    tree = load_tree(tree_path)
    made, unknown_count = stage_claim(new_findings, tree, unknown_path)
    log(f"[claim] {len(made)} staked, {unknown_count} -> unknown journal")

    # 4. test
    all_findings = read_jsonl(log_path)
    test_stats = stage_test(tree, all_findings)
    log(f"[test] {test_stats}")

    # 5. modify claim
    mod_stats = stage_modify(tree, unknown_path, reform_path)
    log(f"[modify] {mod_stats}")

    save_tree(tree, tree_path)

    # 6. hidden variables
    suggestions = stage_hidden(tree, all_findings, hidden_path)
    log(f"[hidden] {len(suggestions)} suggestions")

    # 7. consolidate
    cons = stage_consolidate(tree, topics, unknown_path, hidden_path,
                             Path(args.hypotheses_dir),
                             data_dir / "announced_topics.json")
    log(f"[consolidate] {cons['hypothesis_files']} hypothesis files")

    stats: dict[str, Any] = {
        "topics": len(topics), "raw_findings": len(findings),
        "new_findings": len(new_findings), "duplicates_skipped": skipped,
        "claims_staked": len(made), "routed_to_unknown": unknown_count,
        **{f"test_{k}": v for k, v in test_stats.items()},
        **{f"modify_{k}": v for k, v in mod_stats.items()},
        "hidden_variable_suggestions": suggestions,
        "hypothesis_files": cons["hypothesis_files"],
        "new_hypotheses": cons["new_hypotheses"],
        "total_claims_in_tree": len(tree.claims),
    }
    report = write_report(data_dir / "engine_report.md", stats)
    print("\n" + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
