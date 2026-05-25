"""
Smoothing pattern audit across the Basin Repair Framework codebase.

Run:
  python experiments/experiment_smoothing_audit.py
"""

import os

from repair.smoothing_auditor import audit_path, to_claim_table

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    hits = audit_path(REPO_ROOT)

    print(f"Scanned: {REPO_ROOT}")
    print(f"Total hits: {len(hits)}")
    print()

    by_pattern: dict[str, int] = {}
    for h in hits:
        by_pattern[h.pattern_name] = by_pattern.get(h.pattern_name, 0) + 1

    for name in sorted(by_pattern):
        print(f"  {name:<22} {by_pattern[name]:>4} hits")

    print()
    table = to_claim_table(
        hits, source_id="basin_repair_framework", path="CLAIM_TABLE.smoothing_audit.json"
    )
    print(f"\n{len(table['claims'])} pattern claims written.")
    print("Note: hits are review candidates — verify each is a legitimate use before acting.")
