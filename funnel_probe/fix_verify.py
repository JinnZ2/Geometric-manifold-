#!/usr/bin/env python3
"""
fix_verify.py -- V1..V3 of PREREGISTRATION_FIX.md on the corrected objective.

    V1  KL(theta_k || theta_ref) over 100 repair steps from the delivered drifted
        start (timescale_check.torch_run, reference FIXED per check C1):
        final KL < 0.1 x initial and no step raises KL by more than 1e-3
    V2  check C2 re-run (divergence_checks.c2_fd_step): dKL < 0 cap on and off,
        grad(KL).delta < 0, cos(delta, grad KL) < -0.9
    V3  task loss at step 100 beside KL (no threshold); `pytest tests/ -q`
        count reported, any failing test listed, nothing edited here.
Gate for step 2: V1 and V2.
"""
import contextlib
import io
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import timescale_check as TC        # noqa: E402
import divergence_checks as DC      # noqa: E402


def main():
    env, layer, disp0, rows = TC.torch_run(100)
    kl_first, kl_last = rows[0]["kl_before"], rows[-1]["kl_after"]
    rises = [r["kl_after"] - r["kl_before"] for r in rows]
    max_rise = max(rises)
    v1 = (kl_last < 0.1 * kl_first) and (max_rise <= 1e-3)
    print("V1  KL on the FIXED reference over 100 steps (default.yaml, seed 42)")
    print("    KL %.4f -> %.4f   (ratio %.4f; gate < 0.1)   max single-step rise %+.2e (gate <= 1e-3)"
          % (kl_first, kl_last, kl_last / kl_first, max_rise))
    print("    KL every 20 steps:", ["%.3f" % rows[i]["kl_before"] for i in range(0, 100, 20)] + ["%.3f" % kl_last])
    print("    dist to ref %.3f -> %.3f ; ||delta|| mean %.4f (cap 0.05) ; steps at cap %d/100"
          % (disp0, rows[-1]["dist"], sum(r["delta"] for r in rows) / 100, sum(1 for r in rows if r["delta"] > 0.0499)))
    print("    task loss step 0 %.4f -> step 100 %.4f" % (rows[0]["task_loss"], rows[-1]["task_loss"]))
    print("    V1", "HELD" if v1 else "FAILED")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        DC.c2_fd_step()
    txt = buf.getvalue()
    print("\nV2  check C2 re-run on the corrected objective")
    print("\n".join("    " + ln for ln in txt.strip().splitlines()))
    dkl = [float(m) for m in re.findall(r"dKL ([+-][0-9.]+(?:e[+-]?\d+)?)", txt)]
    gd = [float(m) for m in re.findall(r"grad\(KL\)\.delta = ([+-][0-9.]+(?:e[+-]?\d+)?)", txt)]
    cos = [float(m) for m in re.findall(r"cos\(delta, grad KL\) = ([+-][0-9.]+)", txt)]
    v2 = len(dkl) >= 2 and all(v < 0 for v in dkl) and all(v < 0 for v in gd) and all(c < -0.9 for c in cos)
    print("    parsed dKL %s ; grad.delta %s ; cos %s" % (dkl, gd, cos))
    print("    V2", "HELD" if v2 else "FAILED")

    print("\nV3  repo suite")
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider"], cwd=ROOT,
                       capture_output=True, text=True, timeout=1500)
    lines = r.stdout.strip().splitlines()
    summary = lines[-1] if lines else r.stderr.strip().splitlines()[-1:]
    print("    pytest exit %d: %s" % (r.returncode, summary))
    fails = [ln for ln in lines if ln.startswith("FAILED")]
    for ln in fails:
        print("    " + ln)
    gate = v1 and v2
    print("\nGATE for step 2: %s  (V1 %s, V2 %s)" % ("OPEN" if gate else "CLOSED", v1, v2))
    out = {"kl_first": kl_first, "kl_last": kl_last, "max_rise": max_rise, "rows": rows, "disp0": disp0,
           "V1": v1, "V2": v2, "c2_text": txt, "pytest_summary": summary, "pytest_failed": fails,
           "pytest_exit": r.returncode, "gate_open": gate}
    json.dump(out, open("fix_verify.json", "w"), indent=1)
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
