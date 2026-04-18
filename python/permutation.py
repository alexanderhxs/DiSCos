import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import logging
from .solvers import disco_weights_reg, disco_mixture
from .models import PermutResult
from .utils import myQuant, getGrid

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
    c_df_q = {}
    
    for t in disco_instance.periods:
        res_t = disco_instance.results_by_period[t]
        
        c_df[t] = res_t.controls.data
        c_df_q[t] = res_t.controls.quantiles
        t_df[t] = res_t.target.data
        
    # Standardize dist_t
    distt = np.zeros(len(disco_instance.periods))
    
    for i, t in enumerate(disco_instance.periods):
        res_t = disco_instance.results_by_period[t]
        if not disco_instance.mixture:
            bc_t = res_t.DiSCo.quantile
            true_q = res_t.target.quantiles
            distt[i] = np.mean((bc_t - true_q)**2)
        else:
            cdf_t = res_t.DiSCo.cdf
            true_cdf = res_t.target.cdf
            distt[i] = np.mean((cdf_t - true_cdf)**2)
            
    # Parallel permutation
    total_perms = len(peridx)
    logging.info(f"Starting {total_perms} permutation tests...")
    
    distp = Parallel(n_jobs=disco_instance.num_cores)(
        delayed(_disco_per_iter)(
            idx, c_df, c_df_q, t_df, 
            disco_instance.T0_idx, peridx, disco_instance.evgrid,
            disco_instance.results_by_period,
            disco_instance.M, disco_instance.simplex, disco_instance.mixture
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

def _disco_per_iter(idx, c_df, c_df_q, t_df, T0, peridx, evgrid, results_by_period, M, simplex, mixture):
    """
    Run one iteration of the permutation test.
    idx is the index from peridx of the new "target" unit.
    """
    periods = list(c_df.keys())
    
    perc = {t: [] for t in periods}
    perc_q = {t: [] for t in periods}
    pert = {t: None for t in periods}
    
    keepcon = [p for p in peridx if p != idx]
    
    for t in periods:
        perc[t].append(t_df[t])
        perc_q[t].append(results_by_period[t].target.quantiles)
        
        for j in keepcon:
            perc[t].append(c_df[t][j])
            perc_q[t].append(c_df_q[t][:, j])
            
        pert[t] = c_df[t][idx]
        perc_q[t] = np.column_stack(perc_q[t])
        
    # Calculate lambda for t <= T0
    lambda_tp = []
    perc_cdf = {}
    
    for t_idx, t in enumerate(periods):
        if not mixture:
            if t <= T0:
                w = disco_weights_reg(perc[t], pert[t], M=M, simplex=simplex)
                lambda_tp.append(w)
        else:
            # We need the CDF matrix for all t to calculate distillation distances
            G_grid = len(results_by_period[t].target.grid) - 1
            grid_min, grid_max, grid_rand, grid_ord = getGrid(pert[t], perc[t], G_grid)
            
            num_controls = len(perc[t])
            cdf_matrix = np.zeros((len(grid_rand), num_controls + 1))
            
            target_sorted = np.sort(pert[t])
            cdf_matrix[:, 0] = np.searchsorted(target_sorted, grid_rand, side='right') / len(pert[t])
            
            for k, ctrl in enumerate(perc[t]):
                ctrl_sorted = np.sort(ctrl)
                cdf_matrix[:, k+1] = np.searchsorted(ctrl_sorted, grid_rand, side='right') / len(ctrl)
                
            perc_cdf[t] = cdf_matrix
            
            if t <= T0:
                res = disco_mixture(perc[t], pert[t], grid_min, grid_max, grid_ord, M, simplex)
                lambda_tp.append(res['weights_opt'])
                
    # Average optimal lambda
    lambda_opt = np.mean(lambda_tp, axis=0)
    
    dist = np.zeros(len(periods))
    
    # Eval dist
    for i, t in enumerate(periods):
        if not mixture:
            bc_t = perc_q[t] @ lambda_opt
            target_q = myQuant(pert[t], evgrid)
            dist[i] = np.mean((bc_t - target_q)**2)
        else:
            bc_t = perc_cdf[t][:, 1:] @ lambda_opt
            target_q = perc_cdf[t][:, 0]
            dist[i] = np.mean((bc_t - target_q)**2)
            
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
