#!/usr/bin/env python3
"""EXP-HVD-02 run (2026-11-27): engine claim outcomes vs measurable nuisance covariates.

Data: data/claim_tree.json joined to data/findings_log.jsonl on id (every claim id occurs
exactly once in the findings log at the commit used). Read-only; nothing in data/ is changed.
Covariates (continuous ones standardised to mean 0, sd 1 so coefficients are per SD):
  abs_tokens      whitespace token count of the abstract
  no_abstract     abstract empty or '(no abstract)'
  topic_tokens    abstract tokens (lowercased, alphanumeric, length >= 4) in the topic name
  same_topic_at   findings of the same topic with retrieved_at <= this claim's retrieved_at
                  (proxy for "same-topic findings logged by the claim's first test run": the
                  test-run log is not stored per claim; deviation)
  pub_days        publication date, days since 2026-01-01 (96 of 972 findings give only a
                  year; imputed as 1 July, deviation; sensitivity fit without those claims)
  retr_days       retrieved_at, days since 2026-01-01
  source          categorical (arxiv reference)
  lane            categorical: primary (reference), cross_domain, none (no lane field)
  topic           fixed effect
Models: logistic regression of survived (status == 'survived'), main outcome; negative
binomial (NB2, alpha estimated) regression of passed. Pre-provenance claims = claims whose
finding has no lane field (lane recorded from the provenance change onward; proxy, recorded
here). Sensitivity: without pre-provenance claims; without reformulated claims.
Permutation test of the full model (logistic, main outcome): likelihood-ratio statistic of all
non-topic covariates vs topic-only model; survived permuted within topic. 2,000 permutations
(deviation from the pre-registered 10,000: one fit took 0.3 s on the loaded machine; the
smallest attainable p is 1/2001 = 5e-4, enough to resolve the 0.05 threshold).
Count model permutation: 500 permutations (deviation, same reason).
Separation: a binary term whose ML coefficient diverges (|coef| > 10 or se > 50) is reported
with its counts and dropped from the model before CIs are read (rule fixed before the final
run, after a first fit showed separation for no_abstract and src_crossref).
NB2 fallback: where alpha is estimated at ~0 (passed is under-dispersed because survived claims
stop at 3) the NB Hessian is singular; the Poisson GLM is used and recorded.
Output: raw.json.
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import REPO, Timer, environment, write_json  # noqa: E402

warnings.filterwarnings("ignore")
EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)
CONT = ["abs_tokens", "topic_tokens", "same_topic_at", "pub_days", "retr_days"]
BIN = ["no_abstract", "src_semantic_scholar", "src_crossref", "lane_none"]
RHS = " + ".join(CONT + BIN + ["C(topic)"])


def days(s):
    if len(s) == 4:  # year-only publication date: imputed as 1 July of that year
        s = s + "-07-01"
    d = datetime.fromisoformat(s) if "T" in s else datetime.fromisoformat(s + "T00:00:00+00:00")
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return (d - EPOCH).total_seconds() / 86400


def load():
    F = [json.loads(line) for line in open(REPO / "data/findings_log.jsonl")]
    claims = json.load(open(REPO / "data/claim_tree.json"))["claims"]
    fid = {f["id"]: f for f in F}
    retr = {t: sorted(days(f["retrieved_at"]) for f in F if f["topic"] == t) for t in {f["topic"] for f in F}}
    rows = []
    for c in claims:
        f = fid[c["id"]]
        ab = f["abstract"].strip()
        toks = re.findall(r"[a-z0-9]+", ab.lower())
        ttoks = {w for w in re.findall(r"[a-z0-9]+", f["topic"].lower()) if len(w) >= 4}
        rd = days(f["retrieved_at"])
        rows.append({"id": c["id"], "topic": c["scope"]["topic"], "survived": int(c["status"] == "survived"),
                     "passed": int(c["passed"]), "failed": int(c["failed"]),
                     "reformulated": int(c.get("reformulation_count", 0) > 0),
                     "abs_tokens": len(ab.split()), "no_abstract": int(ab in ("", "(no abstract)")),
                     "topic_tokens": sum(w in ttoks for w in toks),
                     "same_topic_at": int(np.searchsorted(retr[f["topic"]], rd, side="right")),
                     "pub_days": days(f["date"]), "retr_days": rd,
                     "source": f["source"], "lane": f.get("lane") or "none",
                     "year_only_date": int(len(f["date"]) == 4)})
    for r in rows:
        r["src_semantic_scholar"] = int(r["source"] == "semantic_scholar")
        r["src_crossref"] = int(r["source"] == "crossref")
        r["lane_none"] = int(r["lane"] == "none")
    return pd.DataFrame(rows)


def standardise(d):
    d = d.copy()
    for k in CONT:
        d[k] = (d[k] - d[k].mean()) / d[k].std(ddof=0)
    return d


def coef_table(fit):
    ci = fit.conf_int()
    return {k: {"coef": float(fit.params[k]), "se": float(fit.bse[k]), "ci95": [float(ci.loc[k, 0]), float(ci.loc[k, 1])],
                "p": float(fit.pvalues[k])} for k in fit.params.index}


def fit_logit(d, rhs=RHS):
    try:
        return smf.logit("survived ~ " + rhs, d).fit(disp=0, maxiter=200)
    except Exception:  # perfect separation etc.
        return None


def fit_nb(d, rhs=RHS):
    """NB2; if alpha is estimated at ~0 (no overdispersion) the NB Hessian is singular, and
    the equivalent Poisson GLM is fitted instead (recorded as fallback)."""
    try:
        f = smf.negativebinomial("passed ~ " + rhs, d).fit(disp=0, maxiter=500)
        if f.params["alpha"] > 1e-3 and np.all(np.isfinite(f.bse)):
            f.fallback = None
            return f
    except Exception:
        pass
    try:
        f = smf.glm("passed ~ " + rhs, d, family=sm.families.Poisson()).fit()
        f.fallback = "poisson (NB alpha ~ 0 or singular Hessian)"
        return f
    except Exception:
        return None


def separated(fit):
    """Terms with quasi-complete separation: |coef| > 10 or se > 50 (ML estimate diverging)."""
    return [k for k in fit.params.index if k != "Intercept" and not k.startswith("C(topic")
            and (abs(fit.params[k]) > 10 or fit.bse[k] > 50)]


def models(d, label):
    out = {"label": label, "rows": int(len(d)), "survived": int(d["survived"].sum())}
    d = standardise(d)
    terms = [t for t in CONT + BIN if t in CONT or d[t].nunique() > 1]
    rhs = " + ".join(terms + ["C(topic)"])
    lg = fit_logit(d, rhs)
    out["separated_terms_dropped"] = []
    if lg is not None and separated(lg):
        sep = separated(lg)
        out["separated_terms_dropped"] = sep
        out["logit_with_separated_terms"] = coef_table(lg)
        cnt = {t: {"rows": int(d[t].sum()), "survived": int(d.loc[d[t] == 1, "survived"].sum())} for t in sep if t in BIN}
        out["separated_term_counts"] = cnt
        rhs = " + ".join([t for t in terms if t not in sep] + ["C(topic)"])
        lg = fit_logit(d, rhs)
    nb = fit_nb(d, rhs)
    out["nb_fallback"] = getattr(nb, "fallback", None) if nb is not None else None
    out["rhs"] = rhs
    out["logit"] = coef_table(lg) if lg is not None else "fit failed"
    out["logit_converged"] = bool(lg.mle_retvals["converged"]) if lg is not None else None
    out["nb"] = coef_table(nb) if nb is not None else "fit failed"
    # date-only model (prediction: date effect vanishes once the mechanical covariates are in)
    lg_date = fit_logit(d, "pub_days + C(topic)")
    out["logit_pub_days_only"] = coef_table(lg_date)["pub_days"] if lg_date is not None else "fit failed"
    return out, lg, nb, rhs


def perm_test(d, rhs, B, kind, seed):
    d = standardise(d).reset_index(drop=True)
    y = "survived" if kind == "logit" else "passed"
    full = fit_logit(d, rhs) if kind == "logit" else fit_nb(d, rhs)
    base = fit_logit(d, "C(topic)") if kind == "logit" else fit_nb(d, "C(topic)")
    obs = 2 * (full.llf - base.llf)
    rng = np.random.default_rng(seed)
    groups = [np.where(d["topic"].values == t)[0] for t in d["topic"].unique()]
    X = d.copy()
    stats, fails = [], 0
    for _ in range(B):
        yy = d[y].values.copy()
        for gi in groups:
            yy[gi] = yy[rng.permutation(gi)]
        X[y] = yy
        f = fit_logit(X, rhs) if kind == "logit" else fit_nb(X, rhs)
        if f is None or not np.isfinite(f.llf):
            fails += 1
            continue
        stats.append(2 * (f.llf - base.llf))  # base llf unchanged by within-topic permutation
    stats = np.array(stats)
    return {"kind": kind, "B": B, "failed_fits": fails, "observed_LR": float(obs),
            "perm_p": float((1 + (stats >= obs).sum()) / (1 + len(stats))),
            "perm_LR_quantiles": np.percentile(stats, [50, 95, 99]).tolist()}


def main():
    res = {"experiment": "EXP-HVD-02", "environment": environment()}
    import statsmodels
    res["environment"]["statsmodels"] = statsmodels.__version__
    res["environment"]["pandas"] = pd.__version__
    d = load()
    res["descriptives"] = {"n": int(len(d)), "by_topic": d.groupby("topic")["survived"].agg(["count", "sum"]).to_dict(),
                           "lane_counts": d["lane"].value_counts().to_dict(), "source_counts": d["source"].value_counts().to_dict(),
                           "means": d[CONT + ["no_abstract", "passed", "survived"]].mean().to_dict(),
                           "corr_spearman": d[CONT + ["passed", "survived"]].corr("spearman").round(3).to_dict()}
    res["fits"] = []
    main_out, lg, nb, rhs = models(d, "all")
    res["fits"].append(main_out)
    res["fits"].append(models(d[d["lane"] != "none"], "without pre-provenance (no lane)")[0])
    res["fits"].append(models(d[d["reformulated"] == 0], "without reformulated")[0])
    res["fits"].append(models(d[d["year_only_date"] == 0], "without year-only publication dates")[0])
    with Timer() as t:
        res["perm_logit"] = perm_test(d, rhs, 2000, "logit", 1)
    res["perm_logit"]["seconds"] = t.seconds
    with Timer() as t:
        res["perm_nb"] = perm_test(d, rhs, 500, "nb", 2)
    res["perm_nb"]["seconds"] = t.seconds
    write_json(HERE / "raw.json", res)
    for f in res["fits"]:
        print(f["label"], f["rows"])
        for m in ("logit", "nb"):
            if isinstance(f[m], dict):
                print(" ", m, {k: (round(v["coef"], 3), [round(x, 3) for x in v["ci95"]]) for k, v in f[m].items()
                               if not k.startswith("C(topic") and k != "Intercept"})
    print(res["perm_logit"], res["perm_nb"])


if __name__ == "__main__":
    main()
