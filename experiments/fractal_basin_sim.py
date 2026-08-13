# Fractal basin boundary sim (vectorized): double- vs triple-well damped oscillator.
#
# Imported from the ecosystem research pass; see docs/research/17_fractals_bio_cosmo_trig.md
# section 1, whose reported numbers this reproduces exactly (alpha=0.688 / 0.392, Wada 8.0%).
#
# Harness status: does NOT conform to docs/research/HARNESS.md. It is item 4 in that
# standard's retrofit queue, whose stated deficiency is "alpha measured at single damping
# — sweep gamma mandatory". gamma is a parameter of basin_grid() but only ever called at
# the default 0.25, so alpha is a single-damping measurement, not a swept one.
#
# Only change from the original: the output directory was hardcoded to an agent workspace
# (/mnt/agents/output/figures/) that does not exist here.
import os

import numpy as np

OUTPUT = os.environ.get("FRACTAL_BASIN_OUT", "results/fractal_basin")

def basin_grid(centers, N=200, xr=(0.4, 3.6), vr=(-1.5, 1.5), dt=0.05, T=120.0, gamma=0.25):
    # integrate ALL initial conditions at once; potential built from centers
    def F(x):  # force = -dE/dx, E = prod (x-c)^2 ; use numerical derivative
        h = 1e-5

        def E(z):
            return np.prod([(z-c)**2 for c in centers], axis=0)
        return -(E(x+h)-E(x-h))/(2*h)
    xs = np.linspace(*xr, N)
    vs = np.linspace(*vr, N)
    X, V = np.meshgrid(xs, vs)
    for _ in range(int(T/dt)):
        V += dt*(F(X) - gamma*V)
        X += dt*V
    G = np.argmin(np.abs(X[..., None] - np.array(centers)), axis=-1)
    return G.astype(int), xs, vs

def uncertainty_exponent(G, xs, n_probe=4000, rng=None):
    N = G.shape[0]
    dx = xs[1]-xs[0]
    epss = dx * 2.0**np.arange(0, 8)
    fs = []
    for eps in epss:
        dj = max(1, int(round(eps/dx)))
        i = rng.integers(2, N-2, n_probe)
        j = rng.integers(2, N-2-dj, n_probe)
        fs.append(np.mean(G[i, j] != G[i, j+dj]))
    epss = np.array(epss)
    fs = np.array(fs)
    m = fs > 0
    alpha = np.polyfit(np.log(epss[m]), np.log(fs[m]), 1)[0]
    return alpha

def wada_fraction(G, rad=2):
    N = G.shape[0]
    tot = wada = 0
    for i in range(rad, N-rad):
        row = G[i-rad:i+rad+1]
        for j in range(rad, N-rad):
            u = np.unique(row[:, j-rad:j+rad+1])
            if len(u) > 1:
                tot += 1
                wada += (len(u) == 3)
    return wada/max(tot, 1), tot

os.makedirs(OUTPUT, exist_ok=True)
rng = np.random.default_rng(0)
G2, xs2, _ = basin_grid([1.2, 1.8], N=200, xr=(0.6, 2.4), vr=(-1.2, 1.2))
a2 = uncertainty_exponent(G2, xs2, rng=rng)
print(f"DOUBLE well: alpha={a2:.3f} -> D_boundary={2-a2:.3f} (alpha=1 => smooth boundary)")
np.save(os.path.join(OUTPUT, 'basins_double.npy'), G2)

G3, xs3, _ = basin_grid([1.0, 2.0, 3.0], N=200, xr=(0.4, 3.6), vr=(-1.5, 1.5))
a3 = uncertainty_exponent(G3, xs3, rng=rng)
wf, tot = wada_fraction(G3)
print(f"TRIPLE well: alpha={a3:.3f} -> D_boundary={2-a3:.3f}")
print(f"Wada: {wf*100:.1f}% of {tot} boundary cells touch all THREE basins")
np.save(os.path.join(OUTPUT, 'basins_triple.npy'), G3)
print("alpha reading: certainty gain per 2x measurement improvement = 2^alpha")
