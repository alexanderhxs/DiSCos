import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import logging
from ..solvers import disco_weights_reg, disco_mixture
from ..models import PermutResult
from ..utils import myQuant, getGrid

def run_permutation_test(disco_instance, peridx=None):
    """
    Run the permutation test for DiSCo models.
    
    Parameters:
        disco_instance: DiSCo class instance after `.fit()`
        peridx: specific control indices to use (defaults to all controls)
    """
    if peridx is None:
        peridx = np.arange(len(disco_instance.controls_id))
        
    c_df = {}
    t_df = {}
    
    for t in disco_instance.periods:
        res_t = disco_instance.results_by_period[t]
        
        c_df[t] = res_t.controls.data
        t_df[t] = res_t.target.data
        
    # Standardize dist_t
    distt = np.zeros(len(disco_instance.periods))
    
    for i, t in enumerate(disco_instance.periods):
        res_t = disco_instance.results_by_period[t]
        distt[i] = disco_instance.solver.compute_distance(
            target=res_t.target.data,
            controls=res_t.controls.data,
            weights=disco_instance.weights_opt,
            grid_ord=res_t.target.grid,
        )
            
    # Parallel permutation
    total_perms = len(peridx)
    logging.info(f"Starting {total_perms} permutation tests...")
    distp = []
    distp = Parallel(n_jobs=disco_instance.num_cores)(
        delayed(_disco_per_iter)(
            idx, c_df, t_df, 
            disco_instance.T0_idx, peridx, 
            disco_instance.results_by_period,
            disco_instance.M, disco_instance.simplex, disco_instance.solver
        ) for idx in peridx
    )

    distp_matrix = np.array(distp)

    # Calculate ranks and p-values
    p_val = _disco_per_rank(distt, distp_matrix, disco_instance.T0_idx)
    
    return PermutResult(
        distp=distp_matrix,
        distt=distt,
        p_overall=p_val,
        J_1=len(distp_matrix),
        q_min=disco_instance.q_min,
        q_max=disco_instance.q_max,
        plot=None
    )

def _disco_per_iter(idx, c_df, t_df, T0, peridx, results_by_period, M, simplex, solver):
    """
    Run one iteration of the permutation test.
    idx is the index from peridx of the new "target" unit.
    """
    periods = list(c_df.keys())
    
    perc = {t: [] for t in periods}
    pert = {t: None for t in periods}
    
    keepcon = [p for p in peridx if p != idx]
    
    for t in periods:
        perc[t].append(t_df[t])
        
        for j in keepcon:
            perc[t].append(c_df[t][j])
            
        pert[t] = c_df[t][idx]
        
    # Calculate lambda for t <= T0
    lambda_tp = []

    for t in periods:
        if t > T0:
            continue
        
        target_data = np.asarray(pert[t])
        controls_data = [np.asarray(c) for c in perc[t][1:]]
        
        if len(target_data) > 0 and len(controls_data) > 0:
            G_grid = len(results_by_period[t].target.grid) - 1
            grid_min, grid_max, grid_ord = getGrid(target_data, controls_data, G_grid)

            res = solver.fit_weights(
                target=target_data,
                controls=controls_data,
                grid_min=grid_min,
                grid_max=grid_max,
                grid_ord=grid_ord,
                M=M,
                simplex=simplex
            )
            lambda_tp.append(res)
        else:
            print(f"Skipping period {t} for permutation {idx} due to insufficient data or post-treatment period.")
            lambda_tp.append(np.ones(len(controls_data)) / len(controls_data))

    # Average optimal lambda
    lambda_opt = np.mean(lambda_tp, axis=0)
    dist = np.zeros(len(periods))
    # Calculate counterfactual and distance for post treatment periods
    for idx, t in enumerate(periods):
        target_data = np.asarray(pert[t])
        controls_data = [np.asarray(c) for c in perc[t][1:]]
        
        if len(target_data) > 0 and len(controls_data) > 0:
            G_grid = len(results_by_period[t].target.grid) - 1
            _, _, grid_ord_perm = getGrid(target_data, controls_data, G_grid)
            
            dist[idx] = solver.compute_distance(
                target=target_data,
                controls=controls_data,
                weights=lambda_opt,
                grid_ord=grid_ord_perm,
            )
        else:
            dist[idx] = np.nan
        
    return dist

def _disco_per_rank(dist_t, dist_p_matrix, T0_idx):
    """
    Rank Wasserstein distances
    """
    J_1 = len(dist_p_matrix)
    
    # MSR = mean target score / mean pre-treatment score
    def get_ratio(dist_arr):
        idx_post = np.where(np.arange(len(dist_arr)) >= T0_idx)[0]
        idx_pre = np.where(np.arange(len(dist_arr)) < T0_idx)[0]
        return np.sqrt(np.mean(dist_arr[idx_post])) / np.sqrt(np.mean(dist_arr[idx_pre]))
    
    R_t = get_ratio(dist_t)
    R_p = np.array([get_ratio(row) for row in dist_p_matrix])
    
    all_ratios = np.append(R_p, R_t)
    # p-value: rank from top (number of permutation ratios >= target ratio)
    rank = np.sum(all_ratios >= R_t)
    p_val = rank / len(all_ratios)
    
    return p_val
