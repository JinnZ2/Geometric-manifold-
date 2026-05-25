# run_collapse_experiment.py
# Sequence: P0 → P3 → main sweep. Writes results to fab artifacts.

import json, time, numpy as np
from mean_field import (find_fixed_points, label_fps, classify,
                        lambda_plus, mf_potential_line,
                        find_c_for_target_barrier)
# from collapse_sim import (HighDimDiversityModel,
#                           LockedHighDimDiversityModel)  # your existing module
# from lambda_measure import measure_lambda_plus_sim
# from closure_check  import closure_trajectory
# from convergence    import dt_sweep, N_sweep

def stage_0_mean_field(a, sigma_n, D_mut, p_values, dphi_target):
    out = {}
    for p in p_values:
        c_star, dphi = find_c_for_target_barrier(
            a, sigma_n, D_mut, target_dphi=dphi_target)
        fps = find_fixed_points(a, c_star, sigma_n, D_mut)
        lab = label_fps(fps, a, c_star, sigma_n, D_mut)
        saddle = min(lab["saddle"], key=lambda fp: abs(fp[0])) \
                 if lab["saddle"] else None
        lp = lambda_plus(saddle, a, c_star, sigma_n, D_mut) \
             if saddle else None
        out[p] = {"c_star": c_star, "dphi": dphi,
                  "saddle": saddle, "lambda_plus_mf": lp}
    return out

def stage_1_lambda_cross_check(mf_out, a, sigma_n, D_mut, p_values):
    # for each p, measure λ+ in sim, compare to MF
    out = {}
    for p in p_values:
        c_star = mf_out[p]["c_star"]
        # mean, sem = measure_lambda_plus_sim(
        #     HighDimDiversityModel, p, c_star, a, sigma_n, D_mut)
        # out[p] = {"lambda_plus_sim_mean": mean,
        #           "lambda_plus_sim_sem": sem,
        #           "lambda_plus_mf": mf_out[p]["lambda_plus_mf"],
        #           "ratio": mean / mf_out[p]["lambda_plus_mf"]}
        pass
    return out

def stage_2_closure(mf_out, a, sigma_n, D_mut, p_test=2):
    c_star = mf_out[p_test]["c_star"]
    # m = HighDimDiversityModel(p=p_test, d=p_test+2, c=c_star,
    #                           a=a, sigma=sigma_n, D_mut=D_mut)
    # traj = closure_trajectory(m, p_test, T_max=200)
    # return traj
    pass

def stage_3_convergence(mf_out, a, sigma_n, D_mut, p_test=2):
    c_star = mf_out[p_test]["c_star"]
    kwargs = dict(a=a, sigma=sigma_n, D_mut=D_mut)
    # dt_res = dt_sweep(HighDimDiversityModel, p_test, c_star, kwargs)
    # N_res  = N_sweep (HighDimDiversityModel, p_test, c_star, kwargs)
    # return {"dt": dt_res, "N": N_res}
    pass

def stage_4_main_sweep(mf_out, a, sigma_n, D_mut, p_values,
                       n_trials=50, T_max_collapse=5000, T_max_repair=500):
    coll = {"independent": {}, "locked": {}}
    rep  = {"independent": {}, "locked": {}}
    for p in p_values:
        c_star = mf_out[p]["c_star"]
        # ... run independent and locked, collapse + repair
        pass
    return coll, rep

if __name__ == "__main__":
    A, SIGMA_N, D_MUT = 1.0, 0.15, 0.02
    P_VALUES = [1, 2, 3, 4]
    DPHI_TARGET = 0.5  # tune from stage_0 output for a single p first

    t0 = time.time()
    print("[stage 0] mean-field calibration")
    mf = stage_0_mean_field(A, SIGMA_N, D_MUT, P_VALUES, DPHI_TARGET)
    print(json.dumps(mf, indent=2, default=str))

    print("[stage 1] λ+ cross-check (sim vs MF)")
    # lam = stage_1_lambda_cross_check(mf, A, SIGMA_N, D_MUT, P_VALUES)

    print("[stage 2] moment closure")
    # clo = stage_2_closure(mf, A, SIGMA_N, D_MUT)

    print("[stage 3] dt + N convergence")
    # conv = stage_3_convergence(mf, A, SIGMA_N, D_MUT)

    print("[stage 4] main p-sweep (independent + locked)")
    # coll, rep = stage_4_main_sweep(mf, A, SIGMA_N, D_MUT, P_VALUES)

    print(f"elapsed: {time.time()-t0:.1f}s")
