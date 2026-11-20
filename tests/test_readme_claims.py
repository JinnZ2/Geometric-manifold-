"""README / CLAUDE.md claim register: concrete statements bound to behaviour.

Status: added 2026-11-19, co-authored (Perplexity, at the owner's request). A
register of sentences in README.md and CLAUDE.md that state a concrete,
checkable result, each paired with what the code does now. Errors are data:
a sentence the code contradicts is not deleted or rewritten here; it is
registered as KNOWN_MISMATCH and the test pins the contradicting behaviour.

The test fails when
  - a quoted sentence leaves its file (the prose changed: re-check, update), or
  - the observed behaviour changes, for MATCHES or KNOWN_MISMATCH alike (a
    KNOWN_MISMATCH that starts to match means the code or the prose was fixed:
    move it to MATCHES and leave a dated note).

Not established: that every concrete statement is registered. Run

    python tests/test_readme_claims.py --hunt

to list lines that look like concrete claims and no quote covers. Those lines
are UNCHECKED, not wrong; the selection rule is a heuristic (PROPOSED).
"""

import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from repair.geometric_confidence import GeometricConfidence  # noqa: E402
from simulation.controller import Controller  # noqa: E402
from simulation.environment import Environment  # noqa: E402

FILES = {"README": ROOT / "README.md", "CLAUDE": ROOT / "CLAUDE.md"}


def norm(text):
    """Collapse whitespace so wrapped lines match a one-line quote."""
    return re.sub(r"\s+", " ", text).strip()


def config(name):
    return yaml.safe_load((ROOT / "configs" / name).read_text())


def default_run(tmp_dir):
    """The Quick start default run, as written, with output sent to tmp_dir."""
    cfg = config("default.yaml")
    cfg["monitoring"]["output_dir"] = str(tmp_dir) + "/"
    env = Environment(cfg["simulation"])
    return Controller(env, cfg).run().records


def pulls_back(tmp_dir):
    """(distance to reference grew over the run, repair flagged at every step)."""
    recs = default_run(tmp_dir)
    return (
        recs[-1]["dist_to_ref"] > recs[0]["dist_to_ref"],
        all(r["repair_triggered"] for r in recs),
    )


def gitignored_results():
    lines = [line.strip() for line in (ROOT / ".gitignore").read_text().splitlines()]
    return "results/" in lines


def default_parts():
    cfg = config("default.yaml")
    return cfg, Environment(cfg["simulation"])


def majority_dropped():
    """(majority samples before, majority samples dropped) under default.yaml."""
    from manifolds.data_manifold import DataManifold

    cfg, env = default_parts()
    _, labels, _ = DataManifold(cfg["manifolds"]["data"]).rectify(env.features, env.labels)
    before = int((env.labels == 0).sum())
    return before, before - int((labels == 0).sum())


def trust_limited_steps():
    """Steps of the 100 default repair steps whose size equals trust_radius."""
    from manifolds.parameter_manifold import ParameterManifold

    cfg, env = default_parts()
    pcfg = cfg["manifolds"]["parameter"]
    layer, theta, fn = (
        ParameterManifold(env.theta_ref, pcfg),
        env.theta_drifted.clone(),
        env.get_model_fn(),
    )
    n = 0
    for _ in range(cfg["simulation"]["steps"]):
        new, _ = layer.repair_step(theta, fn, env.safety_inputs, env.task_inputs, env.task_labels)
        n += abs((new - theta).norm().item() - pcfg["trust_radius"]) < 1e-6
        theta = new
    return n


def lines_in(path):
    return len((ROOT / path).read_text().splitlines())


# (id, file, quote, state, observed(tmp_dir), expected). See sims/objective_sign/
# FINDING.md for the experiment behind the two KNOWN_MISMATCH "pulls back" rows.
CLAIMS = [
    (
        "quickstart-default-config",
        "README",
        "# Run the default simulation (drift=0.3, 100 steps)",
        "MATCHES",
        lambda d: (lambda s: (s["drift_strength"], s["steps"]))(
            config("default.yaml")["simulation"]
        ),
        (0.3, 100),
    ),
    (
        "quickstart-adversarial-config",
        "README",
        "# Run with adversarial settings (drift=0.8, 200 steps)",
        "MATCHES",
        lambda d: (lambda s: (s["drift_strength"], s["steps"]))(
            config("adversarial.yaml")["simulation"]
        ),
        (0.8, 200),
    ),
    (
        "core-idea-pulls-back",
        "README",
        "this framework detects the drift and pulls parameters back using curvature-aware, "
        "asymmetric repair",
        "MATCHES",
        # observed 2026-11-19 on the default run: dist_to_ref 17.08 -> 18.45, every
        # step flagged; the parameters move away from the reference
        # (registered then as KNOWN_MISMATCH, expected (True, True))
        # Note 2026-11-20: objective sign corrected (cherry-pick of 59031bc). Observed on
        # the default run: dist_to_ref 17.06 -> 16.14, every step flagged. Moved to MATCHES.
        pulls_back,
        (False, True),
    ),
    (
        "prior-work-pulls-back",
        "README",
        "applies curvature-regularized, trust-region-bounded updates that pull drifting "
        "parameters back toward the reference",
        "MATCHES",
        # Note 2026-11-20: was KNOWN_MISMATCH, expected (True, True); see the entry above.
        pulls_back,
        (False, True),
    ),
    (
        "minus-sign-in-code",
        "CLAUDE",
        "**Do not change the sign in `task_loss - λ * weighted_safety`** in "
        "`parameter_manifold.py`.",
        "KNOWN_MISMATCH",
        # the sign is still in the code; its documented rationale is refuted by
        # sims/objective_sign (see the dated note in CLAUDE.md)
        # (registered 2026-11-19 as MATCHES, expected True)
        # Note 2026-11-20: the sign in the code is now "+" (cherry-pick of 59031bc). The
        # sentence is kept in CLAUDE.md, struck through; the code no longer follows it.
        lambda d: (
            "total_loss = task_loss - self.lambda_asym * weighted_safety"
            in (ROOT / "manifolds" / "parameter_manifold.py").read_text()
        ),
        False,
    ),
    (
        "confidence-weights",
        "CLAUDE",
        "**Weights sum to 1**: Default `(0.2, 0.5, 0.3)` must sum to 1.0.",
        "MATCHES",
        lambda d: GeometricConfidence().combined(1.0, 1.0, 1.0),
        pytest.approx(1.0),
    ),
    (
        "results-not-gitignored",
        "CLAUDE",
        "Results are written to `results/` which is not gitignored.",
        "KNOWN_MISMATCH",
        lambda d: gitignored_results(),
        True,
    ),
    (
        "majority-drop-fraction",
        "CLAUDE",
        "Drops ~30% of majority samples.",
        "KNOWN_MISMATCH",
        # observed 2026-11-19: 9 of 160 majority samples dropped (5.6%)
        lambda d: majority_dropped(),
        (160, 9),
    ),
    (
        "trust-region-limited",
        "CLAUDE",
        "Combined with trust_radius=0.05, most steps are trust-region-limited, not lr-limited.",
        "MATCHES",
        # observed 2026-11-19: all 100 steps are at the trust radius
        # Note 2026-11-20: with the sign corrected, 53 of 100 steps are at the trust radius
        lambda d: trust_limited_steps(),
        53,
    ),
    (
        "stability-line-count",
        "CLAUDE",
        "`addon_thermodynamic_control/stability.py` is 552 lines",
        "KNOWN_MISMATCH",
        lambda d: lines_in("addon_thermodynamic_control/stability.py"),
        586,
    ),
]


@pytest.mark.parametrize("cid,file,quote", [(c[0], c[1], c[2]) for c in CLAIMS])
def test_quote_is_still_in_its_file(cid, file, quote):
    assert norm(quote) in norm(FILES[file].read_text()), (
        f"{cid}: quote no longer in {file}.md; the prose changed. Re-check and update."
    )


@pytest.mark.parametrize("claim", CLAIMS, ids=[c[0] for c in CLAIMS])
def test_behaviour_is_as_registered(claim, tmp_path):
    cid, file, quote, state, observed, expected = claim
    assert state in ("MATCHES", "KNOWN_MISMATCH")
    assert observed(tmp_path) == expected, (
        f"{cid} ({state}): {file}.md says {quote!r}; behaviour changed from the register. "
        "If a KNOWN_MISMATCH now matches, move it to MATCHES with a dated note."
    )


CLAIMLIKE = re.compile(
    r"\d.*\b(tests?|lines?|steps|drift|samples|%|x|weight)\b|\b(must|always|never|exactly)\b.*\d",
    re.IGNORECASE,
)


def hunt():
    """Lines in README.md and CLAUDE.md that look like concrete claims, uncovered."""
    quotes = [norm(c[2]) for c in CLAIMS]
    out = []
    for name, path in FILES.items():
        for n, line in enumerate(path.read_text().splitlines(), 1):
            body = norm(line)
            if len(body) < 12 or not CLAIMLIKE.search(body):
                continue
            if any(body in q or q in body for q in quotes):
                continue
            if any(
                len(body) > 20 and body[len(body) // 4 : len(body) * 3 // 4] in q for q in quotes
            ):
                continue
            out.append((f"{name}.md", n, line.strip()))
    return out


if __name__ == "__main__":
    if "--hunt" in sys.argv:
        rows = hunt()
        print(
            f"Claim hunt: {len(CLAIMS)} registered "
            f"({sum(c[3] == 'KNOWN_MISMATCH' for c in CLAIMS)} KNOWN_MISMATCH), "
            f"{len(rows)} lines UNCHECKED (heuristic selection, PROPOSED; unchecked is not wrong)"
        )
        for f, n, line in rows:
            print(f"  {f}:{n}: {line}")
    else:
        sys.exit(pytest.main([__file__, "-q"]))
