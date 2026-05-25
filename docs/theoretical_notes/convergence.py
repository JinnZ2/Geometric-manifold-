# convergence.py
def dt_sweep(model_cls, p, c, kwargs, dt_values=(0.02, 0.01, 0.005, 0.0025),
             n_trials=30, T_max=2000):
    results = {}
    for dt in dt_values:
        ts = []
        for r in range(n_trials):
            m = model_cls(p=p, d=p+2, c=c, dt=dt, seed=10_000+r, **kwargs)
            t, _ = m.simulate(T_max=T_max)
            ts.append(t)
        results[dt] = (np.mean(ts), np.std(ts)/np.sqrt(n_trials))
    return results

def N_sweep(model_cls, p, c, kwargs, N_values=(50, 200, 800),
            n_trials=30, T_max=2000):
    results = {}
    for N in N_values:
        ts = []
        for r in range(n_trials):
            m = model_cls(p=p, d=p+2, c=c, N=N, seed=20_000+r, **kwargs)
            t, _ = m.simulate(T_max=T_max)
            ts.append(t)
        results[N] = (np.mean(ts), np.std(ts)/np.sqrt(n_trials))
    return results
