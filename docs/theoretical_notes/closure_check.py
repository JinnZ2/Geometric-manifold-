# closure_check.py
import numpy as np
from scipy.stats import skew, kurtosis

def closure_diagnostics(Z, p):
    """Returns per-dim skew, excess kurt; Gaussian → 0, 0."""
    active = Z[:, :p]
    sk = skew(active, axis=0)
    ek = kurtosis(active, axis=0, fisher=True)  # excess
    return sk, ek

def closure_trajectory(model, p, T_max, log_every=100):
    """Run model, log moments + diversity over time."""
    out = {'t': [], 'D': [], 'max_skew': [], 'max_exkurt': []}
    steps = int(T_max / model.dt)
    for s in range(steps):
        model.step()
        model.t += model.dt
        if s % log_every == 0:
            sk, ek = closure_diagnostics(model.Z, p)
            out['t'].append(model.t)
            out['D'].append(model.diversity())
            out['max_skew'].append(np.max(np.abs(sk)))
            out['max_exkurt'].append(np.max(np.abs(ek)))
    return out
