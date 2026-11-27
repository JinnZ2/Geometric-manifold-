"""Recursive 2-D GMM refitting, vectorised across seeds (EXP-MCR-01 / EXP-MCR-02, 2026-11-27).

Real distribution (fixed before any run; not specified in the design, see results.md):
K = 5 components, means on a circle of radius 1.5 at angles 0, 72, ..., 288 degrees,
isotropic component sd 1.0, weights (0.40, 0.25, 0.15, 0.12, 0.08).
sigma0 = sqrt(mean per-coordinate variance of the real distribution).

Generation t: fit a K = 5 full-covariance GMM by EM (max 50 iterations, stop when the
mean log-likelihood of every chain changes by < 1e-5; warm start from generation t-1's fit;
generation 0 starts from the true parameters), then sample n points at the given
temperature (component covariances scaled by temperature^2). replace: next training set
is the n samples. accumulate: next training set is the real data plus all samples so far.
A component whose total responsibility falls below 1e-9 is frozen with weight 0 (mode
lost; EM cannot revive it).
"""
from __future__ import annotations

import numpy as np

K = 5
ANG = np.deg2rad(np.arange(K) * 72.0)
TRUE_MU = 1.5 * np.stack([np.cos(ANG), np.sin(ANG)], axis=1)
TRUE_W = np.array([0.40, 0.25, 0.15, 0.12, 0.08])
TRUE_SD = 1.0
SIGMA0 = float(np.sqrt(TRUE_SD ** 2 + (TRUE_W[:, None] * TRUE_MU ** 2).sum(0).mean()
                       - ((TRUE_W[:, None] * TRUE_MU).sum(0) ** 2).mean()))
REG = 1e-6


def sample_gmm(rng, w, mu, cov, n, temperature=1.0):
    """w (S,K), mu (S,K,2), cov (S,K,2,2) -> x (S,n,2)."""
    S = w.shape[0]
    u = rng.random((S, n))
    cw = np.cumsum(w, axis=1)
    cw[:, -1] = 1.0 + 1e-12
    comp = (u[:, :, None] > cw[:, None, :]).sum(-1)
    comp = np.minimum(comp, K - 1)
    L = np.linalg.cholesky(cov * temperature ** 2 + REG * np.eye(2))
    z = rng.standard_normal((S, n, 2))
    idx = np.arange(S)[:, None]
    Lc = L[idx, comp]
    return mu[idx, comp] + np.einsum("snij,snj->sni", Lc, z)


def log_gauss(x, mu, cov):
    """x (S,N,2), mu (S,K,2), cov (S,K,2,2) -> (S,N,K). Explicit 2x2 algebra."""
    a = cov[..., 0, 0] + REG
    b = cov[..., 0, 1]
    c = cov[..., 1, 1] + REG
    det = a * c - b * b
    d0 = x[:, :, None, 0] - mu[:, None, :, 0]
    d1 = x[:, :, None, 1] - mu[:, None, :, 1]
    m = (c[:, None] * d0 * d0 - 2 * b[:, None] * d0 * d1 + a[:, None] * d1 * d1) / det[:, None]
    return -0.5 * m - 0.5 * np.log(det)[:, None, :] - np.log(2 * np.pi)


def em(x, w, mu, cov, iters=50, tol=1e-5):
    S, N, _ = x.shape
    alive = w > 0
    prev = np.full(S, -np.inf)
    for _ in range(iters):
        lw = np.where(alive, np.log(np.where(alive, w, 1.0)), -np.inf)
        lp = log_gauss(x, mu, cov) + lw[:, None, :]
        mx = lp.max(-1, keepdims=True)
        ll_n = mx[..., 0] + np.log(np.exp(lp - mx).sum(-1))
        r = np.exp(lp - ll_n[..., None])
        nk = r.sum(1)
        alive = alive & (nk > 1e-9)
        nk_safe = np.where(alive, nk, 1.0)
        w = np.where(alive, nk / N, 0.0)
        w = w / w.sum(1, keepdims=True)
        x0, x1 = x[..., 0], x[..., 1]
        m0 = np.matmul(x0[:, None, :], r)[:, 0] / nk_safe
        m1 = np.matmul(x1[:, None, :], r)[:, 0] / nk_safe
        mu_new = np.stack([m0, m1], axis=-1)
        s00 = np.matmul((x0 * x0)[:, None, :], r)[:, 0] / nk_safe - m0 * m0
        s01 = np.matmul((x0 * x1)[:, None, :], r)[:, 0] / nk_safe - m0 * m1
        s11 = np.matmul((x1 * x1)[:, None, :], r)[:, 0] / nk_safe - m1 * m1
        cov_new = np.stack([np.stack([s00, s01], -1), np.stack([s01, s11], -1)], -2)
        mu = np.where(alive[..., None], mu_new, mu)
        cov = np.where(alive[..., None, None], cov_new, cov)
        ll = ll_n.mean(1)
        if np.all(np.abs(ll - prev) < tol):
            break
        prev = ll
    return w, mu, cov


def tail_mass(rng, w, mu, cov, m=20000):
    x = sample_gmm(rng, w, mu, cov, m)
    return (np.linalg.norm(x, axis=-1) > 3 * SIGMA0).mean(1)


def overall_sd(w, mu, cov):
    m = (w[..., None] * mu).sum(1)
    second = (w[..., None, None] * (cov + np.einsum("ski,skj->skij", mu, mu))).sum(1)
    c = second - np.einsum("si,sj->sij", m, m)
    return np.sqrt((c[:, 0, 0] + c[:, 1, 1]) / 2)


def simulate(n, T, seeds, mode, temperature=1.0, tail_at=(), seed_offset=0, tail_m=20000):
    """Run len(seeds) chains in parallel. Returns dict of per-generation arrays (S, T+1)."""
    S = len(seeds)
    rng = np.random.default_rng([seed_offset, n, int(temperature * 1000),
                                 0 if mode == "replace" else 1, seeds[0], S])
    w0 = np.tile(TRUE_W, (S, 1))
    mu0 = np.tile(TRUE_MU, (S, 1, 1))
    cov0 = np.tile(TRUE_SD ** 2 * np.eye(2), (S, K, 1, 1))
    real = sample_gmm(rng, w0, mu0, cov0, n)
    data = real
    w, mu, cov = w0.copy(), mu0.copy(), cov0.copy()
    out = {"min_w": np.zeros((S, T + 1)), "n_alive": np.zeros((S, T + 1), int),
           "n_above_floor": np.zeros((S, T + 1), int), "sd": np.zeros((S, T + 1)),
           "w": np.zeros((S, T + 1, K)), "tail": {}}
    trng = np.random.default_rng([seed_offset, 99, n, seeds[0], S])
    for t in range(T + 1):
        w, mu, cov = em(data, w, mu, cov)
        out["min_w"][:, t] = w.min(1)
        out["n_alive"][:, t] = (w > 0).sum(1)
        out["n_above_floor"][:, t] = (w > 1 / (10 * K)).sum(1)
        out["sd"][:, t] = overall_sd(w, mu, cov)
        out["w"][:, t] = w
        if t in tail_at:
            out["tail"][t] = tail_mass(trng, w, mu, cov, tail_m)
        synth = sample_gmm(rng, w, mu, cov, n, temperature)
        if mode == "replace":
            data = synth
        else:
            data = np.concatenate([data, synth], axis=1)
    return out


def true_tail_mass(m=2_000_000, seed=0):
    rng = np.random.default_rng(seed)
    w0, mu0 = TRUE_W[None], TRUE_MU[None]
    cov0 = (TRUE_SD ** 2 * np.eye(2))[None, None].repeat(K, 1)
    return float(tail_mass(rng, w0, mu0, cov0, m)[0])
