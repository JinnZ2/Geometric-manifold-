"""
Heuristic scanner for variance-suppression patterns.

Flags source locations where data may be artificially constrained —
clamping, normalization, thresholding — that could mask true signal
variance. Results are candidates for human review, not confirmed violations.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Broad by design: each hit is a review candidate, not a confirmed violation.
DEFAULT_PATTERNS: dict[str, str] = {
    "hard_clamp": r"\.clip\(",
    "threshold_min": r"\bmin\(",
    "threshold_max": r"\bmax\(",
    "normalization": r"\bnormalize\(",
    "signal_dampening": r"\bsmooth\(",
    "goal_seeking": r"\btarget_value\b",
    "nonlinear_squash": r"\bsigmoid\b",
    "hardcoded_modifier": r"\bconstant\b",
}

AUDIT_EXTENSIONS: frozenset[str] = frozenset({".py", ".cpp", ".js", ".c"})


@dataclass
class SmoothingHit:
    file: str
    line: int
    pattern_name: str
    pattern: str
    context: str

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def audit_path(
    root_dir: str,
    patterns: Optional[dict[str, str]] = None,
    extensions: Optional[frozenset[str]] = None,
) -> list[SmoothingHit]:
    """Scan files under root_dir and return all smoothing pattern hits."""
    active_patterns = patterns or DEFAULT_PATTERNS
    active_extensions = extensions or AUDIT_EXTENSIONS
    hits: list[SmoothingHit] = []

    for path in sorted(Path(root_dir).rglob("*")):
        if not path.is_file() or path.suffix not in active_extensions:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, start=1):
            for name, pat in active_patterns.items():
                if re.search(pat, line, re.IGNORECASE):
                    hits.append(
                        SmoothingHit(
                            file=str(path),
                            line=i,
                            pattern_name=name,
                            pattern=pat,
                            context=line.strip(),
                        )
                    )
    return hits


def to_claim_table(
    hits: list[SmoothingHit],
    source_id: str = "smoothing_audit",
    path: Optional[str] = None,
) -> dict:
    """Export audit results in the project's CLAIM_TABLE format."""
    path = path or f"CLAIM_TABLE.{source_id}.json"

    by_pattern: dict[str, list[dict]] = {}
    for hit in hits:
        by_pattern.setdefault(hit.pattern_name, []).append(hit.as_dict())

    claims = [
        {
            "claim_id": f"{source_id}.smoothing.{name}",
            "claim": f"Pattern '{name}' may suppress signal variance",
            "falsification_condition": (
                "Demonstrate each hit is a legitimate use, not variance suppression"
            ),
            "hits": len(hit_list),
            # cap location list to avoid unbounded output
            "locations": [f"{h['file']}:{h['line']}" for h in hit_list[:10]],
            "status": "CANDIDATE",
        }
        for name, hit_list in sorted(by_pattern.items())
    ]

    table = {
        "source_id": source_id,
        "total_hits": len(hits),
        "total_claims": len(claims),
        "claims": claims,
        "note": "Hits are heuristic candidates for human review, not confirmed violations.",
    }

    with open(path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"[audit] {len(hits)} hits across {len(claims)} patterns → {path}")
    return table
