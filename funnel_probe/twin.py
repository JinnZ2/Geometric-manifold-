"""
twin.py -- numpy twin of the repo's ToyLLM + ParameterManifold.repair_step.

Same architecture (32-64-16, ReLU), same losses (CE on task; KL(ref || current)
on safety with batchmean; curvature proxy = torch.var(softmax, dim=-1).mean()
with torch's unbiased variance), same saddle objective
`task - lambda * kl * (1 + lambda_c * curv)`, same trust-region clamp.
Gradient is analytic (hand backprop) and checked against torch autograd on an
identical theta when torch is importable.  Init follows nn.Linear's
U(-1/sqrt(fan_in), 1/sqrt(fan_in)) under numpy seed 42 -- different bytes from
torch.manual_seed(42), stated in PREREGISTRATION.md.
"""
import numpy as np

I, H, O = 32, 64, 16
S1, S2, S3 = I * H, I * H + H, I * H + H + H * O


def unpack(theta):
    w1 = theta[:S1].reshape(H, I); b1 = theta[S1:S2]
    w2 = theta[S2:S3].reshape(O, H); b2 = theta[S3:]
    return w1, b1, w2, b2


def forward(X, theta):
    w1, b1, w2, b2 = unpack(theta)
    z1 = X @ w1.T + b1
    h = np.maximum(z1, 0.0)
    return z1, h, h @ w2.T + b2


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def backprop(X, z1, h, dlogits, theta):
    w1, b1, w2, b2 = unpack(theta)
    dw2 = dlogits.T @ h
    db2 = dlogits.sum(0)
    dh = dlogits @ w2
    dz1 = dh * (z1 > 0)
    dw1 = dz1.T @ X
    db1 = dz1.sum(0)
    return np.concatenate([dw1.ravel(), db1, dw2.ravel(), db2])


def kl_batchmean(p_ref, logits):
    """F.kl_div(log_softmax(logits), p_ref, 'batchmean') = mean_s sum_j q (log q - log p)."""
    logp = logits - logits.max(axis=1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(axis=1, keepdims=True))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p_ref > 0, p_ref * (np.log(p_ref) - logp), 0.0)
    return t.sum(axis=1).mean()


def objective_and_grad(theta, theta_ref, Xs, Xt, yt, lam, lamc):
    n_t, n_s = len(Xt), len(Xs)
    # task CE
    z1t, ht, lt = forward(Xt, theta)
    pt = softmax(lt)
    ce = -np.log(pt[np.arange(n_t), yt] + 1e-300).mean()
    dlt = pt.copy(); dlt[np.arange(n_t), yt] -= 1.0; dlt /= n_t
    g_task = backprop(Xt, z1t, ht, dlt, theta)
    # safety KL + curvature
    _, _, lref = forward(Xs, theta_ref)
    q = softmax(lref)
    z1s, hs, ls = forward(Xs, theta)
    p = softmax(ls)
    kl = kl_batchmean(q, ls)
    K = ls.shape[1]
    curv = ((p ** 2).sum(axis=1) - 1.0 / K).mean() / (K - 1)   # torch.var unbiased
    weighted = kl * (1.0 + lamc * curv)
    dkl = (p - q) / n_s
    dcurv = (2.0 / ((K - 1) * n_s)) * (p ** 2 - p * (p ** 2).sum(axis=1, keepdims=True))
    dls = dkl * (1.0 + lamc * curv) + kl * lamc * dcurv
    g_safe = backprop(Xs, z1s, hs, dls, theta)
    total = ce - lam * weighted
    grad = g_task - lam * g_safe
    return total, grad, {"task_loss": ce, "kl": kl, "curv": curv}


def repair_step(theta, theta_ref, Xs, Xt, yt, cfg):
    lr, trust = cfg["lr"], cfg["trust_radius"]
    lam, lamc = cfg["asymmetry_lambda"], cfg["curvature_weight"]
    _, grad, m = objective_and_grad(theta, theta_ref, Xs, Xt, yt, lam, lamc)
    delta = -lr * grad
    nrm = np.linalg.norm(delta)
    if nrm > trust:
        delta = delta * (trust / nrm)
    return theta + delta, m


def init(seed):
    rng = np.random.default_rng(seed)
    def lin(fan_in, n):
        return rng.uniform(-1 / np.sqrt(fan_in), 1 / np.sqrt(fan_in), n)
    return np.concatenate([lin(I, I * H), lin(I, H), lin(H, H * O), lin(H, O)])


def make_env(cfg, seed=42):
    rng = np.random.default_rng(seed)
    theta_ref = init(seed)
    drifted = theta_ref + cfg["simulation"]["drift_strength"] * rng.standard_normal(theta_ref.size)
    Xs = rng.standard_normal((32, I)); Xt = rng.standard_normal((32, I))
    yt = rng.integers(0, O, 32)
    return theta_ref, drifted, Xs, Xt, yt


def run(cfg, steps):
    theta_ref, theta, Xs, Xt, yt = make_env(cfg)
    pc = cfg["manifolds"]["parameter"]
    disp0 = float(np.linalg.norm(theta - theta_ref))
    rows = []
    for step in range(steps):
        kl0 = kl_batchmean(softmax(forward(Xs, theta_ref)[2]), forward(Xs, theta)[2])
        new, m = repair_step(theta, theta_ref, Xs, Xt, yt, pc)
        rows.append({"step": step, "delta": float(np.linalg.norm(new - theta)), "kl_before": float(kl0),
                     "kl_after": float(kl_batchmean(softmax(forward(Xs, theta_ref)[2]), forward(Xs, new)[2])),
                     "dist": float(np.linalg.norm(new - theta_ref)), "task_loss": float(m["task_loss"])})
        theta = new
    return {"disp0": disp0, "rows": rows}


def check_against_torch(cfg):
    """Same theta, same inputs: twin analytic gradient vs torch autograd of the repo's objective."""
    import torch
    import torch.nn.functional as F
    from simulation.environment import model_fn
    theta_ref, theta, Xs, Xt, yt = make_env(cfg, seed=7)
    pc = cfg["manifolds"]["parameter"]
    lam, lamc = pc["asymmetry_lambda"], pc["curvature_weight"]
    _, g_np, _ = objective_and_grad(theta, theta_ref, Xs, Xt, yt, lam, lamc)
    t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    tr = torch.tensor(theta_ref, dtype=torch.float64)
    TXs, TXt = torch.tensor(Xs, dtype=torch.float64), torch.tensor(Xt, dtype=torch.float64)
    Ty = torch.tensor(yt)
    task = F.cross_entropy(model_fn(TXt, t), Ty)
    with torch.no_grad():
        ref_out = model_fn(TXs, tr)
    so = model_fn(TXs, t)
    kl = F.kl_div(F.log_softmax(so, dim=-1), F.softmax(ref_out, dim=-1), reduction="batchmean")
    curv = torch.var(F.softmax(so, dim=-1), dim=-1).mean()
    total = task - lam * kl * (1.0 + lamc * curv)
    total.backward()
    g_t = t.grad.numpy()
    rel = float(np.linalg.norm(g_np - g_t) / np.linalg.norm(g_t))
    return {"rel_err": rel, "grad_norm_torch": float(np.linalg.norm(g_t))}
