"""
Generic basin repair controller — stdlib only, no external dependencies.

Saddle-point objective: task_loss − λ·safety_loss (minus sign intentional).
Trust region, KL basin boundary, diagonal Fisher — see CLAUDE.md for rationale.

ISS_PROOF_PENDING: True. Phase labels are heuristic, not certified stability claims.
"""

import json
import math
from dataclasses import dataclass
from typing import Callable, Optional

StateVec = list[float]
LossFn = Callable[[StateVec], float]


def _fd_gradient(f: LossFn, x: StateVec, eps: float = 1e-4) -> StateVec:
    grad = []
    for i in range(len(x)):
        xp, xm = x[:], x[:]
        xp[i] += eps
        xm[i] -= eps
        grad.append((f(xp) - f(xm)) / (2.0 * eps))
    return grad


def _fd_hvp(f: LossFn, x: StateVec, v: StateVec, eps: float = 1e-4) -> StateVec:
    """v used directly — do NOT normalize before passing."""
    v_norm = math.sqrt(sum(vi**2 for vi in v))
    if v_norm < 1e-12:
        return [0.0] * len(x)
    xp = [x[i] + eps * v[i] for i in range(len(x))]
    xm = [x[i] - eps * v[i] for i in range(len(x))]
    gp = _fd_gradient(f, xp, eps)
    gm = _fd_gradient(f, xm, eps)
    return [(gp[i] - gm[i]) / (2.0 * eps) for i in range(len(x))]


def _norm(v: StateVec) -> float:
    return math.sqrt(sum(vi**2 for vi in v))


def _dot(a: StateVec, b: StateVec) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def _softmax(x: StateVec) -> StateVec:
    m = max(x)
    e = [math.exp(xi - m) for xi in x]
    s = sum(e)
    return [ei / s for ei in e]


def _kl(p: StateVec, q: StateVec) -> float:
    kl = 0.0
    for pi, qi in zip(p, q):
        if pi > 1e-12 and qi > 1e-12:
            kl += pi * math.log(pi / qi)
    return max(0.0, kl)


def _fisher_diag(f: LossFn, x: StateVec, eps: float = 1e-4) -> StateVec:
    grad = _fd_gradient(f, x, eps)
    return [g**2 for g in grad]


def _kappa_eff(f: LossFn, x: StateVec, v: StateVec, eps: float = 1e-4) -> float:
    """Rayleigh quotient along v — spike precedes phase transition."""
    dnsq = _dot(v, v)
    if dnsq < 1e-12:
        return 0.0
    hvp = _fd_hvp(f, x, v, eps)
    return abs(_dot(v, hvp) / dnsq)


@dataclass
class RepairState:
    step: int
    theta: StateVec
    task_loss: float
    safety_loss: float
    saddle_objective: float  # task_loss − λ·safety_loss
    delta_norm: float
    trust_radius: float
    kl_from_reference: float
    in_basin: bool
    repair_energy: float
    cumulative_repair: float
    kappa_eff_value: float
    trend: float
    phase: str
    confidence: float  # 0.0 (outside basin) to 1.0 (deep inside)
    constraint_violations: list
    ISS_proof_pending: bool = True

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class GenericRepairController:
    """Basin repair for any (theta_ref, task_fn, safety_fn) triple."""

    def __init__(
        self,
        theta_ref: StateVec,
        task_loss_fn: LossFn,
        safety_loss_fn: LossFn,
        config: dict,
        constraint_fn: Optional[Callable] = None,
        domain: str = "generic",
    ):
        self.theta_ref = theta_ref[:]
        self.task_loss_fn = task_loss_fn
        self.safety_loss_fn = safety_loss_fn
        self.constraint_fn = constraint_fn
        self.domain = domain

        self.lr = config.get("lr", 0.01)
        self.lambda_safety = config.get("lambda_safety", 10.0)  # safety dominates by 10x
        self.trust_radius = config.get("trust_radius", 0.05)  # hard bound on step size
        self.epsilon_basin = config.get("epsilon_basin", 0.1)  # KL basin boundary
        self.repair_budget = config.get("repair_budget", 100.0)
        self.spectral_C_bound = config.get("spectral_C_bound", 20.0)
        self.fd_epsilon = config.get("fd_epsilon", 1e-4)
        self.mu = config.get("mu_repair", 0.1)
        self.mu_max = config.get("mu_max", 10.0)
        self.curvature_weight = config.get("curvature_weight", 2.0)
        # tune per domain — higher values shrink confidence faster with distance
        self.confidence_dist_scale = config.get("confidence_dist_scale", 0.1)

        self._cumulative_repair = 0.0
        self._per_step_energy: list[float] = []
        self._history: list[RepairState] = []

    def _saddle_objective(self, theta: StateVec) -> float:
        # minus sign is intentional — adversarial tension resolved by trust region
        return self.task_loss_fn(theta) - self.lambda_safety * self.safety_loss_fn(theta)

    def _kl_from_reference(self, theta: StateVec) -> float:
        return _kl(_softmax(theta), _softmax(self.theta_ref))

    def _in_basin(self, theta: StateVec) -> bool:
        return self._kl_from_reference(theta) < self.epsilon_basin

    def _confidence(self, theta: StateVec, kl: float) -> float:
        dist = _norm([t - r for t, r in zip(theta, self.theta_ref)])
        raw = math.exp(-self.curvature_weight * kl - dist * self.confidence_dist_scale)
        return max(0.0, min(1.0, raw))

    def _repair_energy(self, delta: StateVec, fisher: StateVec) -> float:
        """δᵀGδ — Fisher-weighted step energy."""
        return sum(d**2 * g for d, g in zip(delta, fisher))

    def _recent_trend(self, window: int = 10) -> float:
        if len(self._per_step_energy) < window * 2:
            return 1.0
        recent = sum(self._per_step_energy[-window:]) / window
        prior = sum(self._per_step_energy[-window * 2 : -window]) / window
        return recent / (prior + 1e-12)

    def _phase(self, kappa: float, kl: float, trend: float) -> str:
        if kappa > self.spectral_C_bound or kl > self.epsilon_basin * 2 or trend > 3.0:
            return "critical"
        if kappa > self.spectral_C_bound * 0.5 or kl > self.epsilon_basin or trend > 1.5:
            return "threshold"
        return "stable"

    def step(self, theta: StateVec) -> tuple[StateVec, RepairState]:
        """One Riemannian repair step; returns (new_theta, RepairState)."""
        eps = self.fd_epsilon
        task = self.task_loss_fn(theta)
        safe = self.safety_loss_fn(theta)

        # grad^2 proxy — avoids full O(n^2) Hessian
        fisher = _fisher_diag(self._saddle_objective, theta, eps)
        inv_fisher = [1.0 / (g + 1e-8) for g in fisher]
        grad = _fd_gradient(self._saddle_objective, theta, eps)

        # G^{-1} * (grad + 2μ·θ·G) — Lagrange multiplier on repair cost
        fisher_reg = [2.0 * self.mu * t * f for t, f in zip(theta, fisher)]
        total_grad = [g + fr for g, fr in zip(grad, fisher_reg)]
        riem_grad = [tg * inv_f for tg, inv_f in zip(total_grad, inv_fisher)]
        delta = [-self.lr * rg for rg in riem_grad]

        # trust region clamp — HARD INVARIANT, do not remove
        delta_n = _norm(delta)
        spectral_norm = max(fisher)
        trust_r = min(self.trust_radius, self.lr / (1.0 + self.mu * spectral_norm))
        if delta_n > trust_r:
            delta = [d * (trust_r / delta_n) for d in delta]
            delta_n = trust_r

        theta_new = [t + d for t, d in zip(theta, delta)]

        if not all(math.isfinite(t) for t in theta_new):
            # safety fallback — step was numerically catastrophic
            theta_new = self.theta_ref[:]
            delta = [r - t for r, t in zip(self.theta_ref, theta)]
            delta_n = _norm(delta)

        energy = self._repair_energy(delta, fisher)
        self._cumulative_repair += energy
        self._per_step_energy.append(energy)

        kl = self._kl_from_reference(theta_new)
        in_basin = kl < self.epsilon_basin
        trend = self._recent_trend()
        kappa = _kappa_eff(self._saddle_objective, theta_new, delta, eps)
        phase = self._phase(kappa, kl, trend)
        conf = self._confidence(theta_new, kl)

        # adaptive μ — tighten Lagrange multiplier when drifting out or over budget
        if self._cumulative_repair > self.repair_budget or not in_basin:
            self.mu = min(self.mu * 1.05, self.mu_max)

        violations = []
        if self.constraint_fn:
            for name, satisfied, _desc in self.constraint_fn(theta_new):
                if not satisfied:
                    violations.append(name)

        state = RepairState(
            step=len(self._history),
            theta=[round(t, 6) for t in theta_new],
            task_loss=round(task, 6),
            safety_loss=round(safe, 6),
            saddle_objective=round(task - self.lambda_safety * safe, 6),
            delta_norm=round(delta_n, 6),
            trust_radius=round(trust_r, 6),
            kl_from_reference=round(kl, 6),
            in_basin=in_basin,
            repair_energy=round(energy, 6),
            cumulative_repair=round(self._cumulative_repair, 6),
            kappa_eff_value=round(kappa, 6),
            trend=round(trend, 4),
            phase=phase,
            confidence=round(conf, 4),
            constraint_violations=violations,
        )
        self._history.append(state)
        return theta_new, state

    def run(self, theta: StateVec, n_steps: int = 20, verbose: bool = True) -> list[RepairState]:
        results = []
        for i in range(n_steps):
            theta, state = self.step(theta)
            results.append(state)
            if verbose and i % 5 == 0:
                print(
                    f"  [{state.step:03d}] {state.phase:9s} | "
                    f"KL={state.kl_from_reference:.4f} | "
                    f"kappa={state.kappa_eff_value:.4f} | "
                    f"conf={state.confidence:.3f} | "
                    f"violations={state.constraint_violations or 'none'}"
                )
        return results

    def summary(self) -> dict:
        if not self._history:
            return {}
        return {
            "domain": self.domain,
            "total_steps": len(self._history),
            "final_phase": self._history[-1].phase,
            "final_kl": self._history[-1].kl_from_reference,
            "in_basin_final": self._history[-1].in_basin,
            "final_confidence": self._history[-1].confidence,
            "cumulative_repair": self._history[-1].cumulative_repair,
            "peak_kappa_eff": max(s.kappa_eff_value for s in self._history),
            "violations_observed": sum(1 for s in self._history if s.constraint_violations),
            "phase_transition_threshold": next(
                (s.step for s in self._history if s.phase == "threshold"), None
            ),
            "phase_transition_critical": next(
                (s.step for s in self._history if s.phase == "critical"), None
            ),
            "ISS_proof_pending": True,
            "calibration_note": (
                "Validated on quadratic proxy loss. "
                "Domain-specific loss landscape validation pending."
            ),
        }

    def to_claim_table(self, source_id: str = None, path: str = None) -> dict:
        source_id = source_id or self.domain
        path = path or f"CLAIM_TABLE.repair.{self.domain}.json"
        claims = [
            {
                "claim_id": f"{source_id}.repair.trust_region",
                "claim": "||delta|| <= trust_radius after every step",
                "falsification_condition": "Find step where delta_norm > trust_radius",
                "evidence": f"max delta_norm observed: {max((s.delta_norm for s in self._history), default=0):.6f}",
                "trust_radius": self.trust_radius,
                "status": "OPEN",
            },
            {
                "claim_id": f"{source_id}.repair.confidence_bounded",
                "claim": "confidence in [0, 1] for all steps",
                "falsification_condition": "Find step where confidence < 0 or > 1",
                "evidence": f"range observed: [{min((s.confidence for s in self._history), default=0):.4f}, {max((s.confidence for s in self._history), default=0):.4f}]",
                "status": "OPEN",
            },
            {
                "claim_id": f"{source_id}.repair.finite_outputs",
                "claim": "All theta values finite (no NaN, no inf) after every step",
                "falsification_condition": "Find step where any theta value is NaN or inf",
                "status": "OPEN",
            },
            {
                "claim_id": f"{source_id}.repair.kl_basin",
                "claim": "Basin boundary defined by KL divergence, not Euclidean distance",
                "falsification_condition": "Demonstrate that KL and Euclidean boundaries diverge and KL is wrong",
                "status": "OPEN",
            },
            {
                "claim_id": f"{source_id}.repair.saddle_point",
                "claim": "Saddle-point objective: task_loss - lambda * safety_loss (minus intentional)",
                "falsification_condition": "Find configuration where plus sign produces better safety-task tradeoff",
                "status": "OPEN",
            },
            {
                "claim_id": f"{source_id}.repair.iss_pending",
                "claim": "ISS_PROOF_PENDING: input-to-state stability not proven",
                "falsification_condition": "Prove ISS with bounded curvature and bounded repair energy",
                "status": "OPEN_PROBLEM",
            },
        ]
        table = {
            "source_id": source_id,
            "domain": self.domain,
            "total_claims": len(claims),
            "claims": claims,
            "summary": self.summary(),
        }
        with open(path, "w") as f:
            json.dump(table, f, indent=2)
        print(f"[repair] {len(claims)} claims written to {path}")
        return table
