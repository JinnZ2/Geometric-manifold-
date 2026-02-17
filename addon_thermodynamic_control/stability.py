"""
Coupled Dynamical System and Lyapunov Stability Analysis

State space: (theta, pi_theta, rho_x)
  theta    in M_theta  - parameter manifold
  pi_theta in M_pi     - induced policy
  rho_x                - data manifold state (online/streaming)

Flow equations:
  theta_dot  = -G(theta)^{-1} grad_theta (L_task + lambda_s*L_safety + lambda_p*J_proactive)
  pi_theta   = P(f_theta)                  [induced, not independent]
  rho_x_dot  = Phi(rho_x, pi_theta)       [data manifold evolution]

Lyapunov candidate:
  V(theta, pi) = L_safety(theta)
               + lambda_pi * R_risk(pi)
               + alpha * KL(pi_theta || pi_theta0)
               + mu * C_repair_cumulative

Stability condition:
  dV/dt <= 0  iff
    1. Hessian of L_safety is positive definite near basin floor
    2. G is positive definite (Fisher metric)
    3. Policy coupling terms are bounded
    4. sup lambda_max(Hessian L_safety) < C  [spectral bound]

Thermodynamic phase transition:
  If spectral bound fails -> repair cost diverges -> critical phase.

Early warning scalar:
  kappa_eff(t) = theta_dot^T H_safety theta_dot / theta_dot^T theta_dot
  Spike in kappa_eff precedes behavioral collapse.

Open problem:
  Prove input-to-state stability under adversarial perturbations
  given bounded curvature and bounded repair energy.
  That converts this from monitoring heuristic to robustness theorem.
"""

import torch
import torch.nn.functional as F
import torch.linalg as LA
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CoupledState:
    """Full state of the coupled (theta, pi, rho_x) dynamical system."""
    step:                   int
    # Primary state
    theta_norm:             float   # ||theta - theta_ref||
    basin_kl:               float   # KL(f_theta || f_theta0) - distributional drift
    policy_risk:            float   # R_risk(pi_theta)
    policy_kl:              float   # KL(pi_theta || pi_theta0)
    data_manifold_entropy:  float   # H(rho_x) - data manifold state proxy
    # Lyapunov value and derivative
    V:                      float   # V(theta, pi)
    dV_dt:                  float   # estimated time derivative of V
    V_dot_negative:         bool    # dV/dt <= 0 ?
    # Energy
    repair_energy_step:     float   # delta^T G delta this step
    cumulative_repair:      float
    kappa_eff:              float   # early warning scalar
    spectral_norm_hessian:  float   # lambda_max(Hessian L_safety)
    # Control
    mu:                     float
    phase:                  str     # stable | threshold | critical
    # Stability certificate
    spectral_bound_satisfied: bool  # lambda_max < C_bound ?


# ─────────────────────────────────────────────────────────────────────────────
# Data Manifold Evolution
# ─────────────────────────────────────────────────────────────────────────────

class DataManifoldDynamics:
    """
    rho_x_dot = Phi(rho_x, pi_theta)

    Models how the data distribution evolves under the current policy.
    In a static dataset this is trivial (rho_x_dot = 0).
    In online/streaming settings the policy shapes what data is seen.

    Here we track a sufficient statistic: entropy of the representation
    distribution H(rho_x), as a scalar proxy for manifold state.
    """

    def __init__(self, decay: float = 0.95):
        self.decay = decay          # exponential smoothing
        self.entropy_estimate = 0.0

    def update(
        self,
        features: torch.Tensor,     # batch of representations
        policy_probs: torch.Tensor  # pi_theta output distribution
    ) -> float:
        """
        Phi: entropy of features reweighted by policy distribution.
        High entropy = diverse, well-covered manifold.
        Low entropy  = manifold collapsing toward policy mode.
        """
        with torch.no_grad():
            # Policy-reweighted feature distribution
            weights = policy_probs.mean(dim=-1)                 # (batch,)
            weights = weights / (weights.sum() + 1e-8)

            # Entropy of weighted feature norms as scalar proxy
            norms = LA.norm(features, dim=-1)                   # (batch,)
            norms_norm = norms / (norms.sum() + 1e-8)
            weighted = weights * norms_norm
            h = -(weighted * (weighted + 1e-8).log()).sum().item()

        # Exponential smoothing
        self.entropy_estimate = (self.decay * self.entropy_estimate
                                 + (1 - self.decay) * h)
        return self.entropy_estimate


# ─────────────────────────────────────────────────────────────────────────────
# Lyapunov Function
# ─────────────────────────────────────────────────────────────────────────────

class LyapunovFunction:
    """
    V(theta, pi) = L_safety(theta)
                 + lambda_pi * R_risk(pi)
                 + alpha * KL(pi_theta || pi_theta0)
                 + mu * C_repair_cumulative

    dV/dt = grad_theta V . theta_dot
           + grad_pi V . pi_dot
           + mu * theta_dot^T G theta_dot

    Substituting Riemannian gradient flow:
      theta_dot = -G^{-1} grad_theta L

    dV/dt = -grad_theta V^T G^{-1} grad_theta L
           + mu * theta_dot^T G theta_dot
           + policy coupling terms

    Negative definiteness requires:
      grad_theta V^T G^{-1} grad_theta L > mu * ||theta_dot||^2_G + coupling

    We check this numerically each step and flag violations.
    """

    def __init__(
        self,
        model_fn:   Callable,
        theta_ref:  torch.Tensor,
        pi_ref_probs: torch.Tensor,         # reference policy output probs
        config:     dict
    ):
        self.model_fn     = model_fn
        self.theta_ref    = theta_ref.detach()
        self.pi_ref_probs = pi_ref_probs.detach()

        self.lambda_pi  = config.get('lambda_pi',   0.5)
        self.alpha      = config.get('alpha_kl',    1.0)
        self.mu         = config.get('mu_repair',   0.1)

        self._V_prev    = None

    def evaluate(
        self,
        theta:          torch.Tensor,
        task_inputs:    torch.Tensor,
        unsafe_inputs:  torch.Tensor,
        cumulative_repair: float
    ) -> dict:
        """Evaluate V and all component terms."""

        with torch.no_grad():
            # L_safety: KL from reference on unsafe inputs
            curr_log = F.log_softmax(self.model_fn(unsafe_inputs, theta), dim=-1)
            ref_prob = F.softmax(self.model_fn(unsafe_inputs, self.theta_ref), dim=-1)
            l_safety = F.kl_div(curr_log, ref_prob, reduction='batchmean').item()

            # Policy probs under current theta
            pi_curr = F.softmax(self.model_fn(task_inputs, theta), dim=-1)

            # R_risk: proxy = max probability of any single action (overconfidence)
            r_risk = pi_curr.max(dim=-1).values.mean().item()

            # KL(pi_theta || pi_theta0)
            pi_ref_exp = self.pi_ref_probs[:len(task_inputs)]
            pi_kl = (pi_curr * (pi_curr.log() - pi_ref_exp.log())).sum(dim=-1).mean().item()
            pi_kl = max(0.0, pi_kl)

        V = (l_safety
             + self.lambda_pi * r_risk
             + self.alpha * pi_kl
             + self.mu * cumulative_repair)

        return {
            'V':            V,
            'l_safety':     l_safety,
            'r_risk':       r_risk,
            'pi_kl':        pi_kl,
            'basin_kl':     l_safety,   # same quantity, named for clarity
        }

    def dV_dt_estimate(self, V_curr: float, V_prev: float, dt: float = 1.0) -> float:
        """Finite difference estimate of dV/dt."""
        if V_prev is None:
            return 0.0
        return (V_curr - V_prev) / dt

    def check_negative_definite(self, dV_dt: float, tolerance: float = 1e-4) -> bool:
        """dV/dt <= 0 is the stability condition."""
        return dV_dt <= tolerance


# ─────────────────────────────────────────────────────────────────────────────
# Spectral Stability Certificate
# ─────────────────────────────────────────────────────────────────────────────

class SpectralCertificate:
    """
    Stability reduces to a spectral constraint:
      sup_theta lambda_max(Hessian L_safety) < C_bound

    If this fails: repair cost diverges -> thermodynamic phase transition.

    We estimate lambda_max via power iteration on Hessian-vector products.
    Track sup over trajectory.
    """

    def __init__(self, C_bound: float = 20.0, n_power_iter: int = 8):
        self.C_bound        = C_bound
        self.n_power_iter   = n_power_iter
        self._sup_kappa     = 0.0
        self._history       = []

    def lambda_max_safety_hessian(
        self,
        model_fn:       Callable,
        theta:          torch.Tensor,
        theta_ref:      torch.Tensor,
        unsafe_inputs:  torch.Tensor
    ) -> float:
        """
        Power iteration for lambda_max(Hessian of L_safety).
        Hessian-vector product via double backprop.
        """
        t = theta.detach().requires_grad_(True)

        curr_log = F.log_softmax(model_fn(unsafe_inputs, t), dim=-1)
        ref_prob = F.softmax(model_fn(unsafe_inputs, theta_ref).detach(), dim=-1)
        l_safety = F.kl_div(curr_log, ref_prob, reduction='batchmean')

        grad = torch.autograd.grad(l_safety, t, create_graph=True)[0]

        v = torch.randn_like(t)
        v = v / (LA.norm(v) + 1e-8)

        eigenvalue = 0.0
        for _ in range(self.n_power_iter):
            hvp = torch.autograd.grad(
                grad, t, grad_outputs=v.detach(), retain_graph=True
            )[0].detach()
            new_eigenvalue = (v * hvp).sum().item()
            v = hvp / (LA.norm(hvp) + 1e-8)
            eigenvalue = new_eigenvalue

        return abs(eigenvalue)

    def effective_repair_curvature(
        self,
        model_fn:       Callable,
        theta:          torch.Tensor,
        theta_ref:      torch.Tensor,
        unsafe_inputs:  torch.Tensor,
        theta_dot:      torch.Tensor
    ) -> float:
        """
        kappa_eff(t) = theta_dot^T H_safety theta_dot / theta_dot^T theta_dot

        Rayleigh quotient of safety Hessian along current flow direction.
        Spike here = curvature steepening along trajectory = imminent phase transition.
        """
        dot_norm_sq = (theta_dot ** 2).sum().item()
        if dot_norm_sq < 1e-12:
            return 0.0

        t = theta.detach().requires_grad_(True)
        curr_log = F.log_softmax(model_fn(unsafe_inputs, t), dim=-1)
        ref_prob = F.softmax(model_fn(unsafe_inputs, theta_ref).detach(), dim=-1)
        l_safety = F.kl_div(curr_log, ref_prob, reduction='batchmean')

        grad = torch.autograd.grad(l_safety, t, create_graph=True)[0]
        hvp  = torch.autograd.grad(
            grad, t, grad_outputs=theta_dot.detach() / (dot_norm_sq ** 0.5),
            retain_graph=False
        )[0].detach()

        kappa = (theta_dot.detach() / (dot_norm_sq ** 0.5) * hvp).sum().item()
        return abs(kappa)

    def update(self, lambda_max: float) -> bool:
        """Returns True if spectral bound is satisfied."""
        self._sup_kappa = max(self._sup_kappa, lambda_max)
        self._history.append(lambda_max)
        return lambda_max < self.C_bound

    @property
    def bound_satisfied_globally(self) -> bool:
        return self._sup_kappa < self.C_bound

    @property
    def sup_observed(self) -> float:
        return self._sup_kappa


# ─────────────────────────────────────────────────────────────────────────────
# Coupled System Controller
# ─────────────────────────────────────────────────────────────────────────────

class CoupledDynamicalSystem:
    """
    Full coupled system: (theta, pi_theta, rho_x)

    Flow:
      theta_dot  = -G(theta)^{-1} grad_theta (L_task + lambda_s*L_safety + lambda_p*J_proactive)
      pi_theta   = P(f_theta)
      rho_x_dot  = Phi(rho_x, pi_theta)

    Lyapunov stability monitored at each step.
    Spectral certificate tracked over trajectory.
    Early warning scalar kappa_eff computed each step.

    Adaptive mu: increases when repair budget exceeded or dV/dt > 0.
    This tightens the Lagrange constraint and forces geometric self-stabilization.
    """

    def __init__(
        self,
        model_fn:   Callable,
        theta_ref:  torch.Tensor,
        task_inputs: torch.Tensor,  # needed to initialize pi_ref
        config:     dict
    ):
        self.model_fn   = model_fn
        self.theta_ref  = theta_ref.detach()

        self.lambda_s   = config.get('lambda_safety',       1.0)
        self.lambda_p   = config.get('lambda_proactive',    0.5)
        self.lambda_pi  = config.get('lambda_policy',       0.3)
        self.mu         = config.get('mu_repair',           0.1)
        self.mu_max     = config.get('mu_max',              10.0)
        self.C_bound    = config.get('spectral_C_bound',    20.0)
        self.kappa_bgt  = config.get('repair_budget',       100.0)
        self.epsilon_s  = config.get('epsilon_basin',       0.1)
        self.lr         = config.get('lr',                  0.01)
        self.sigma_d    = config.get('sigma_drift',         0.05)
        self.alpha_kl   = config.get('alpha_kl',            1.0)

        # Reference policy probs (frozen)
        with torch.no_grad():
            pi_ref = F.softmax(model_fn(task_inputs, theta_ref), dim=-1)

        self.lyapunov   = LyapunovFunction(model_fn, theta_ref, pi_ref, {
            'lambda_pi': self.lambda_pi,
            'alpha_kl':  self.alpha_kl,
            'mu_repair': self.mu,
        })
        self.certificate    = SpectralCertificate(C_bound=self.C_bound)
        self.data_dynamics  = DataManifoldDynamics()

        self._cumulative_repair = 0.0
        self._history: list[CoupledState] = []
        self._V_prev = None

    def _fisher_diag(self, theta: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        grads_sq = torch.zeros_like(theta)
        n = min(8, len(inputs))
        for i in range(n):
            t = theta.detach().requires_grad_(True)
            lp = F.log_softmax(self.model_fn(inputs[i:i+1], t), dim=-1)
            s  = torch.multinomial(lp.exp(), 1).squeeze()
            g  = torch.autograd.grad(-lp[0, s], t)[0]
            grads_sq = grads_sq + g.detach() ** 2
        return grads_sq / n

    def _proactive(self, theta: torch.Tensor, unsafe_inputs: torch.Tensor,
                   fisher_diag: torch.Tensor) -> torch.Tensor:
        inv_f = 1.0 / (fisher_diag.detach() + 1e-8)
        total = torch.tensor(0.0)
        n = 6
        for _ in range(n):
            delta = self.sigma_d * torch.randn_like(theta.detach())
            t2 = (theta.detach() + delta).requires_grad_(True)
            curr_log = F.log_softmax(self.model_fn(unsafe_inputs, t2), dim=-1)
            ref_prob = F.softmax(self.model_fn(unsafe_inputs, self.theta_ref).detach(), dim=-1)
            kl = F.kl_div(curr_log, ref_prob, reduction='batchmean')
            grad = torch.autograd.grad(kl, t2)[0].detach()
            total = total + (grad ** 2 * inv_f).sum() / n
        return total

    def _phase(self, kappa: float, basin_kl: float, dV_dt: float) -> str:
        if kappa > self.C_bound or basin_kl > self.epsilon_s * 2 or dV_dt > 0.1:
            return "critical"
        if kappa > self.C_bound * 0.5 or basin_kl > self.epsilon_s or dV_dt > 0.0:
            return "threshold"
        return "stable"

    def step(
        self,
        theta:          torch.Tensor,
        unsafe_inputs:  torch.Tensor,
        task_inputs:    torch.Tensor,
        task_labels:    torch.Tensor
    ) -> tuple[torch.Tensor, CoupledState]:

        theta_prev = theta.detach().clone()

        # ── Fisher metric ────────────────────────────────────────────────────
        fisher = self._fisher_diag(theta.detach(), unsafe_inputs)
        inv_f  = 1.0 / (fisher + 1e-8)

        # ── Gradient of full objective ───────────────────────────────────────
        t = theta.detach().requires_grad_(True)

        l_task  = F.cross_entropy(self.model_fn(task_inputs, t), task_labels)

        curr_log = F.log_softmax(self.model_fn(unsafe_inputs, t), dim=-1)
        ref_prob = F.softmax(self.model_fn(unsafe_inputs, self.theta_ref).detach(), dim=-1)
        l_safe   = F.kl_div(curr_log, ref_prob, reduction='batchmean')

        j_pro    = self._proactive(t, unsafe_inputs, fisher)
        fish_reg = (t ** 2 * fisher.detach()).sum()

        total = (l_task
                 + self.lambda_s * l_safe
                 + self.lambda_p * j_pro
                 + self.mu * fish_reg)
        total.backward()

        # ── Riemannian gradient flow ─────────────────────────────────────────
        with torch.no_grad():
            riem_grad   = t.grad * inv_f           # G^{-1} grad
            delta       = -self.lr * riem_grad
            trust_r     = self.lr / (1.0 + self.mu * fisher.max().item())
            norm        = LA.norm(delta)
            if norm > trust_r:
                delta = delta * (trust_r / norm)
            theta_new   = t + delta

        theta_dot = delta.detach()                 # discrete approximation of theta_dot

        # ── Energy accounting ────────────────────────────────────────────────
        step_energy = (theta_dot ** 2 * fisher).sum().item()
        self._cumulative_repair += step_energy

        # ── Spectral certificate ─────────────────────────────────────────────
        lambda_max = self.certificate.lambda_max_safety_hessian(
            self.model_fn, theta_new.detach(), self.theta_ref, unsafe_inputs
        )
        spectral_ok = self.certificate.update(lambda_max)

        kappa_eff = self.certificate.effective_repair_curvature(
            self.model_fn, theta_new.detach(), self.theta_ref, unsafe_inputs, theta_dot
        )

        # ── Lyapunov evaluation ──────────────────────────────────────────────
        self.lyapunov.mu = self.mu
        lv = self.lyapunov.evaluate(
            theta_new.detach(), task_inputs, unsafe_inputs, self._cumulative_repair
        )
        V_curr  = lv['V']
        dV_dt   = self.lyapunov.dV_dt_estimate(V_curr, self._V_prev)
        V_neg   = self.lyapunov.check_negative_definite(dV_dt)
        self._V_prev = V_curr

        # ── Data manifold evolution ──────────────────────────────────────────
        with torch.no_grad():
            pi_curr = F.softmax(self.model_fn(task_inputs, theta_new.detach()), dim=-1)
        data_h = self.data_dynamics.update(task_inputs, pi_curr)

        # ── Adaptive mu ──────────────────────────────────────────────────────
        if self._cumulative_repair > self.kappa_bgt or not V_neg:
            self.mu = min(self.mu * 1.05, self.mu_max)

        phase = self._phase(kappa_eff, lv['basin_kl'], dV_dt)

        state = CoupledState(
            step=                   len(self._history),
            theta_norm=             LA.norm(theta_new.detach() - self.theta_ref).item(),
            basin_kl=               lv['basin_kl'],
            policy_risk=            lv['r_risk'],
            policy_kl=              lv['pi_kl'],
            data_manifold_entropy=  data_h,
            V=                      V_curr,
            dV_dt=                  dV_dt,
            V_dot_negative=         V_neg,
            repair_energy_step=     step_energy,
            cumulative_repair=      self._cumulative_repair,
            kappa_eff=              kappa_eff,
            spectral_norm_hessian=  lambda_max,
            mu=                     self.mu,
            phase=                  phase,
            spectral_bound_satisfied= spectral_ok,
        )
        self._history.append(state)

        if len(self._history) % 10 == 0:
            print(
                f"  {state.step:4d} | {phase:9s} | "
                f"V={V_curr:.4f} dV/dt={dV_dt:+.4f} {'✓' if V_neg else '✗'} | "
                f"kappa={kappa_eff:.3f} | KL={lv['basin_kl']:.4f} | "
                f"spec={'ok' if spectral_ok else 'FAIL'} | mu={self.mu:.3f}"
            )

        return theta_new.detach(), state

    def lyapunov_certificate_summary(self) -> dict:
        """
        Summary of stability certificate over full trajectory.

        Key check: was dV/dt <= 0 maintained?
        If yes and spectral bound held: basin is empirically stable.
        Toward a proof: need this to hold for all admissible perturbations,
        not just the observed trajectory.
        """
        if not self._history:
            return {}
        V_violations    = [s for s in self._history if not s.V_dot_negative]
        spec_violations = [s for s in self._history if not s.spectral_bound_satisfied]

        return {
            'total_steps':              len(self._history),
            'V_dot_violations':         len(V_violations),
            'first_V_dot_violation':    V_violations[0].step if V_violations else None,
            'spectral_violations':      len(spec_violations),
            'first_spectral_violation': spec_violations[0].step if spec_violations else None,
            'sup_kappa_eff':            max(s.kappa_eff for s in self._history),
            'sup_lambda_max_hessian':   self.certificate.sup_observed,
            'C_bound':                  self.C_bound,
            'spectral_bound_held':      self.certificate.bound_satisfied_globally,
            'cumulative_repair':        self._cumulative_repair,
            'final_V':                  self._history[-1].V,
            'V_trajectory_decreasing':  len(V_violations) == 0,
            'empirical_stability':      len(V_violations) == 0 and self.certificate.bound_satisfied_globally,
            # Open problem flag
            'ISS_proof_pending':        True,
        }
