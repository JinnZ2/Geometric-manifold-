#!/usr/bin/env python3
"""Tier 1 substrate: a tiny MLP in pure Python, a skewed-frequency classification task
with a representation bottleneck, and the three candidate axes that give parameter
drift a sample dimension.

THE GAP THIS FILE CLOSES EXPLICITLY (work order section 1)
    Li et al. take a covariance over N input samples. Parameter space has one theta per
    checkpoint and no sample dimension. Each axis below manufactures one, and each is a
    different substitution:

    A1/TIME          rows = the last W per-step deltas of theta        (W from the repo, see INVENTORY)
    A2/UNIT-L1       rows = the per-step delta of each hidden unit's incoming weights + bias
    A2/UNIT-L2       rows = the per-step delta of each output unit's incoming weights + bias
    A3/SEED-raw      rows = (theta_t - theta_0) for each seed at matched step, raw coordinates
    A3/SEED-aligned  same, after hidden units are permuted to match seed 0 (see align_hidden)

    REP              rows = hidden activations over a fixed probe set  (the reference channel)

    Drift axes use the UNCENTERED second moment [CHOICE 1]: the question "how many
    independent directions is theta moving in" includes the persistent direction, and
    centering across the window would subtract exactly that. REP is centered, as the
    reference instrument is a covariance.

Nothing here is a claim about a real network. Stdlib only.
"""

from __future__ import annotations

import itertools
import math
import random

from spectrum import Spectrum, spectrum_of_rows

# ------------------------------------------------------------------ data


def make_prototypes(rng: random.Random, n_classes: int, n_in: int):
    return [[rng.gauss(0.0, 1.0) for _ in range(n_in)] for _ in range(n_classes)]


def class_frequencies(n_classes: int, skew: float):
    """p_k proportional to (k+1)^-skew. skew = 0 is the uniform control."""
    w = [(k + 1) ** (-skew) for k in range(n_classes)]
    z = sum(w)
    return [v / z for v in w]


def sample_batch(rng, protos, freqs, batch, noise):
    xs, ys = [], []
    cum, acc = [], 0.0
    for p in freqs:
        acc += p
        cum.append(acc)
    for _ in range(batch):
        u = rng.random()
        k = 0
        while k < len(cum) - 1 and u > cum[k]:
            k += 1
        xs.append([v + rng.gauss(0.0, noise) for v in protos[k]])
        ys.append(k)
    return xs, ys


def probe_set(rng, protos, per_class, noise):
    """Fixed probe set, UNIFORM over classes [CHOICE 2]: the reference channel asks how
    many classes are represented distinctly, so every class gets the same number of
    probes regardless of its training frequency."""
    xs, ys = [], []
    for k, p in enumerate(protos):
        for _ in range(per_class):
            xs.append([v + rng.gauss(0.0, noise) for v in p])
            ys.append(k)
    return xs, ys


# ------------------------------------------------------------------ model


class MLP:
    """n_in -> d (tanh) -> n_classes, softmax cross-entropy, plain SGD with L2.

    theta layout (flat): W1 (d rows of n_in) | b1 (d) | W2 (n_classes rows of d) | b2.
    """

    def __init__(self, rng: random.Random, n_in: int, d: int, n_classes: int, init_scale: float):
        self.n_in, self.d, self.n_classes = n_in, d, n_classes
        s1 = init_scale / math.sqrt(n_in)
        s2 = init_scale / math.sqrt(d)
        self.W1 = [[rng.gauss(0.0, s1) for _ in range(n_in)] for _ in range(d)]
        self.b1 = [0.0] * d
        self.W2 = [[rng.gauss(0.0, s2) for _ in range(d)] for _ in range(n_classes)]
        self.b2 = [0.0] * n_classes

    # flat parameter vector ------------------------------------------------
    def theta(self):
        out = []
        for r in self.W1:
            out.extend(r)
        out.extend(self.b1)
        for r in self.W2:
            out.extend(r)
        out.extend(self.b2)
        return out

    def n_params(self):
        return self.d * self.n_in + self.d + self.n_classes * self.d + self.n_classes

    def hidden(self, x):
        return [
            math.tanh(sum(w * v for w, v in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)
        ]

    def logits_from_hidden(self, h):
        return [sum(w * v for w, v in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]

    def forward(self, x):
        h = self.hidden(x)
        return h, self.logits_from_hidden(h)

    @staticmethod
    def softmax(z):
        m = max(z)
        e = [math.exp(v - m) for v in z]
        s = sum(e)
        return [v / s for v in e]

    def loss_and_grads(self, xs, ys):
        d, n_in, n_cls = self.d, self.n_in, self.n_classes
        gW1 = [[0.0] * n_in for _ in range(d)]
        gb1 = [0.0] * d
        gW2 = [[0.0] * d for _ in range(n_cls)]
        gb2 = [0.0] * n_cls
        loss = 0.0
        B = len(xs)
        for x, y in zip(xs, ys):
            h, z = self.forward(x)
            p = self.softmax(z)
            loss -= math.log(max(p[y], 1e-300))
            dz = p[:]
            dz[y] -= 1.0
            dh = [0.0] * d
            for c in range(n_cls):
                dzc = dz[c]
                gb2[c] += dzc
                row = gW2[c]
                w2c = self.W2[c]
                for j in range(d):
                    row[j] += dzc * h[j]
                    dh[j] += dzc * w2c[j]
            for j in range(d):
                da = dh[j] * (1.0 - h[j] * h[j])
                gb1[j] += da
                row = gW1[j]
                for i in range(n_in):
                    row[i] += da * x[i]
        inv = 1.0 / B
        return (
            loss * inv,
            [[v * inv for v in r] for r in gW1],
            [v * inv for v in gb1],
            [[v * inv for v in r] for r in gW2],
            [v * inv for v in gb2],
        )

    def sgd_step(self, xs, ys, lr: float, weight_decay: float):
        loss, gW1, gb1, gW2, gb2 = self.loss_and_grads(xs, ys)
        for j in range(self.d):
            rw, rg = self.W1[j], gW1[j]
            for i in range(self.n_in):
                rw[i] -= lr * (rg[i] + weight_decay * rw[i])
            self.b1[j] -= lr * gb1[j]
        for c in range(self.n_classes):
            rw, rg = self.W2[c], gW2[c]
            for j in range(self.d):
                rw[j] -= lr * (rg[j] + weight_decay * rw[j])
            self.b2[c] -= lr * gb2[c]
        return loss

    def accuracy(self, xs, ys):
        hit = 0
        for x, y in zip(xs, ys):
            _, z = self.forward(x)
            hit += int(max(range(len(z)), key=z.__getitem__) == y)
        return hit / len(xs)


# ------------------------------------------------------------------ axes


def rep_spectrum(model: MLP, probe_xs) -> Spectrum:
    """Reference channel: centered covariance of hidden activations over the probe set."""
    rows = [model.hidden(x) for x in probe_xs]
    return spectrum_of_rows(
        "REP", rows, centered=True, note="hidden activations, uniform probe set"
    )


def unit_rows(delta, model: MLP):
    """Split a flat per-step delta into the two families of unit rows."""
    d, n_in, n_cls = model.d, model.n_in, model.n_classes
    o = 0
    w1 = [delta[o + j * n_in : o + (j + 1) * n_in] for j in range(d)]
    o += d * n_in
    b1 = delta[o : o + d]
    o += d
    w2 = [delta[o + c * d : o + (c + 1) * d] for c in range(n_cls)]
    o += n_cls * d
    b2 = delta[o : o + n_cls]
    rows_l1 = [w1[j] + [b1[j]] for j in range(d)]
    rows_l2 = [w2[c] + [b2[c]] for c in range(n_cls)]
    return rows_l1, rows_l2


def a1_time(window_deltas) -> Spectrum:
    return spectrum_of_rows(
        "A1/TIME",
        window_deltas,
        centered=False,
        note=f"last {len(window_deltas)} per-step deltas, uncentered",
    )


def a2_unit(delta, model: MLP):
    l1, l2 = unit_rows(delta, model)
    return (
        spectrum_of_rows(
            "A2/UNIT-L1", l1, centered=False, note="hidden-unit rows of one per-step delta"
        ),
        spectrum_of_rows(
            "A2/UNIT-L2", l2, centered=False, note="output-unit rows of one per-step delta"
        ),
    )


def align_hidden(model: MLP, ref: MLP):
    """Permutation of this model's hidden units that best matches `ref`, by brute force
    over d! permutations of W1 rows (d is small here by construction) minimising the
    squared distance of W1 rows and W2 columns jointly. Returns the permutation.

    Why: two seeds that learn the same function up to a relabelling of hidden units have
    near-orthogonal delta-theta in raw coordinates, so A3/SEED-raw cannot tell "different
    solution" from "same solution, units renamed". A3/SEED-aligned removes the second.
    """
    d = model.d
    best, best_cost = None, None
    w2_cols = [[model.W2[c][j] for c in range(model.n_classes)] for j in range(d)]
    w2_cols_ref = [[ref.W2[c][j] for c in range(ref.n_classes)] for j in range(d)]
    for perm in itertools.permutations(range(d)):
        cost = 0.0
        for j, pj in enumerate(perm):
            cost += sum((a - b) ** 2 for a, b in zip(model.W1[pj], ref.W1[j]))
            cost += (model.b1[pj] - ref.b1[j]) ** 2
            cost += sum((a - b) ** 2 for a, b in zip(w2_cols[pj], w2_cols_ref[j]))
            if best_cost is not None and cost >= best_cost:
                break
        else:
            if best_cost is None or cost < best_cost:
                best, best_cost = perm, cost
    return list(best)


def permute_theta(theta, perm, d: int, n_in: int, n_cls: int):
    """Apply a hidden-unit permutation to a flat theta (or flat delta)."""
    o1 = d * n_in
    o2 = o1 + d
    o3 = o2 + n_cls * d
    W1 = [theta[j * n_in : (j + 1) * n_in] for j in range(d)]
    b1 = theta[o1:o2]
    W2 = [theta[o2 + c * d : o2 + (c + 1) * d] for c in range(n_cls)]
    b2 = theta[o3:]
    out = []
    for j in range(d):
        out.extend(W1[perm[j]])
    out.extend(b1[perm[j]] for j in range(d))
    for c in range(n_cls):
        out.extend(W2[c][perm[j]] for j in range(d))
    out.extend(b2)
    return out


def a3_seed(deltas_from_init, aligned: bool) -> Spectrum:
    axis = "A3/SEED-aligned" if aligned else "A3/SEED-raw"
    return spectrum_of_rows(
        axis,
        deltas_from_init,
        centered=False,
        note=f"theta_t - theta_0 across {len(deltas_from_init)} seeds at matched step",
    )


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def norm(a):
    return math.sqrt(sum(x * x for x in a))
