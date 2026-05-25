# lambda_measure.py
import numpy as np

def measure_lambda_plus_sim(model_cls, p, c, a, sigma, D_mut,
                            N=200, n_reps=20, t_growth=2.0, dt=0.01,
                            init_perturb=1e-4, seed_base=0):
    """
    Start at homogeneous fixed point + tiny perturbation toward saddle.
    Measure exponential growth rate of |Z - mean| along unstable dir.
    """
    a_sqrt = np.sqrt(a)
    rates = []
    for r in range(n_reps):
        rng = np.random.default_rng(seed_base + r)
        Z = np.full((N, p+2), a_sqrt)             # homogeneous
        Z += rng.normal(0, init_perturb, Z.shape)  # small kick
        # evolve deterministic + small noise
        n_steps = int(t_growth / dt)
        norms = []
        for step in range(n_steps):
            mean_z = Z.mean(axis=0)
            drift = np.zeros_like(Z)
            for k in range(p):
                zk = Z[:, k]
                drift[:, k] = -(zk**3 - a*zk)
            drift -= c*(Z - mean_z)
            noise = np.sqrt(2*sigma**2*dt) * rng.normal(size=Z.shape)
            Z += drift*dt + noise
            # track deviation in active dims
            dev = Z[:, :p] - Z[:, :p].mean(axis=0)
            norms.append(np.linalg.norm(dev))
        norms = np.array(norms)
        # fit log(norm) vs t in early window before saturation
        ts = np.arange(n_steps)*dt
        mask = (norms > 0) & (ts < t_growth*0.3)
        if mask.sum() > 10:
            slope = np.polyfit(ts[mask], np.log(norms[mask]), 1)[0]
            rates.append(slope)
    return np.mean(rates), np.std(rates)/np.sqrt(len(rates))
