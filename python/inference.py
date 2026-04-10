import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from solvers import disco_weights_reg
from utils import myQuant

def disco_per_iter(c_df, c_df_q, t_df, T0, peridx, evgrid, idx, grid_df, M=1000, 
                   ww=0, q_min=0, q_max=1, simplex=False, mixture=False):
    """
    One iteration of the permutation test.
    """
    pert = []
    perc = []
    perc_q = []
    
    num_periods = len(c_df)
    
    for i in range(num_periods):
        perc.append([t_df[i]])
        
    keepcon = [p for p in peridx if p != idx]
    
    for i in range(num_periods):
        perc_q_matrices = []
        perc_q_matrices.append(myQuant(t_df[i], evgrid).reshape(-1, 1))
        for j in keepcon:
            perc[i].append(c_df[i][j])
            perc_q_matrices.append(c_df_q[i][:, j].reshape(-1, 1))
        perc_q.append(np.hstack(perc_q_matrices))
        
    for i in range(num_periods):
        pert.append(c_df[i][idx])
        
    # calculate lambda_t for t < T0
    lambda_tp = []
    if not mixture:
        for t in range(T0):
            w = disco_weights_reg(perc[t], pert[t], M=M, simplex=simplex, q_min=q_min, q_max=q_max)
            lambda_tp.append(w)
    else:
        from .mixture import disco_mixture
        raise NotImplementedError("Mixture approach not implemented.")
        
    # calculate average optimal weights
    if isinstance(ww, (int, float)) and ww == 0:
        w_t = np.ones(T0) / T0
        lambda_opt = np.average(lambda_tp, axis=0, weights=w_t)
    else:
        lambda_opt = np.average(lambda_tp, axis=0, weights=ww)
        
    bc_t = []
    target_q = []
    
    if not mixture:
        for t in range(num_periods):
            bc_t.append(perc_q[t] @ lambda_opt)
            target_q.append(myQuant(pert[t], evgrid))
            
    dist = []
    for t in range(num_periods):
        dist.append(np.mean((bc_t[t] - target_q[t])**2))
        
    return dist

def permutation_test(results_periods, T0, ww=0, peridx=None, evgrid=None, 
                     num_cores=1, q_min=0, q_max=1, M=1000, simplex=False, mixture=False, verbose=True):
    """
    Function to implement permutation test for Distributional Synthetic Controls.
    """
    if evgrid is None:
        evgrid = np.linspace(0, 1, 101)
        
    c_df = [res['controls']['data'] for res in results_periods]
    t_df = [res['target']['data'] for res in results_periods]
    
    controls_q = [res['controls']['quantiles'] for res in results_periods]
    target_q = [res['target']['quantiles'] for res in results_periods]
    
    grid_df = [res['target']['grid'] for res in results_periods]
    
    num_periods = len(c_df)
    
    bc_t = [res['DiSCo']['quantile'] for res in results_periods]
    
    distt = np.zeros(num_periods)
    if not mixture:
        for t in range(num_periods):
            distt[t] = np.mean((bc_t[t] - target_q[t])**2)
    else:
        raise NotImplementedError("Mixture approach not implemented.")
        
    if peridx is None:
        peridx = list(range(len(c_df[0])))
        
    if verbose:
        print(f"Calculating {len(peridx)} permutations using {num_cores} jobs...")
        
    distp = Parallel(n_jobs=num_cores)(
        delayed(disco_per_iter)(
            c_df, controls_q, t_df, T0, peridx, evgrid, idx, 
            grid_df, M, ww, q_min, q_max, simplex, mixture
        ) for idx in peridx
    )
    
    distp = np.array(distp)
    
    # rank the squared Wasserstein distances
    distall = np.vstack([distp, distt])
    J_1 = distall.shape[0]
    
    # R uses mean of periods from T0..end over mean of periods 0..T0-1
    R = np.sqrt(np.mean(distall[:, T0:], axis=1)) / np.sqrt(np.mean(distall[:, :T0], axis=1))
    
    # p-value: proportion of R values >= the target unit's R value
    p_val = np.sum(R >= R[-1]) / J_1
    
    if verbose:
        print("All placebo permutations finished successfully!")
        
    return {
        'distp': distp.tolist(),
        'distt': distt.tolist(),
        'p_overall': float(p_val),
        'J_1': J_1,
        'q_min': q_min,
        'q_max': q_max,
        'R': R.tolist()
    }


def disco_ci_iter(t, controls_t, target_t, grid, T0, M=1000, evgrid=None, 
                  mixture=False, simplex=False, replace=True):
    """
    Compute confidence intervals in the DiSCo method for a single period.
    """
    if evgrid is None:
        evgrid = np.linspace(0, 1, 101)
        
    t_len = len(target_t)
    mytar = np.random.choice(target_t, size=t_len, replace=replace)
    mytar_q = myQuant(mytar, evgrid)
    mytar_cdf = np.mean(mytar[:, None] <= grid[None, :], axis=0) # wait, broadcast might be heavy, but okay
    
    mycon_list = []
    mycon_q = np.zeros((len(evgrid), len(controls_t)))
    mycon_cdf = np.zeros((len(grid), len(controls_t)))
    
    for ii, ctrl in enumerate(controls_t):
        c_len = len(ctrl)
        mycon = np.random.choice(ctrl, size=c_len, replace=replace)
        mycon_list.append(mycon)
        mycon_q[:, ii] = myQuant(mycon, evgrid)
        mycon_cdf[:, ii] = np.mean(mycon[:, None] <= grid[None, :], axis=0)
        
    if t <= T0:
        if not mixture:
            lambda_opt = disco_weights_reg(mycon_list, mytar, M=M, simplex=simplex)
        else:
            raise NotImplementedError("Mixture approach not implemented.")
    else:
        lambda_opt = None

    return {
        'weights': lambda_opt,
        'target': {'quantile': mytar_q, 'cdf': mytar_cdf},
        'controls': {'quantile': mycon_q, 'cdf': mycon_cdf}
    }


def boot_counterfactuals(result_t, t, mixture, weights, evgrid, grid):
    """
    Calculate bootstrapped counterfactuals in the DiSCo method.
    """
    if mixture:
        raise NotImplementedError("Mixture approach not implemented.")
    else:
        q_t = result_t['controls']['quantile'] @ weights
        cdf_t = np.mean(q_t[:, None] <= grid[None, :], axis=0)
        
    cdf_diff = result_t['target']['cdf'] - cdf_t
    q_diff = result_t['target']['quantile'] - q_t
    
    return {
        'cdf': cdf_t,
        'quantile': q_t,
        'quantile_diff': q_diff,
        'cdf_diff': cdf_diff
    }


def confidence_interval(redraw, controls, target, T_max, T0, grid, num_cores=1,
                        evgrid=None, M=1000, mixture=False, simplex=False, replace=True):
    """
    Confidence intervals in the DiSCo method using the bootstrap approach.
    """
    if evgrid is None:
        evgrid = np.linspace(0, 1, 101)
        
    boots_periods = Parallel(n_jobs=num_cores)(
        delayed(disco_ci_iter)(
            t, controls[t], target[t], grid[t], T0, M, evgrid, mixture, simplex, replace
        ) for t in range(1, T_max+1)
    )
    
    weights_sum = np.zeros_like(boots_periods[0]['weights'])
    
    for t in range(T0):
        # DEBUG
        if boots_periods[t]['weights'] is None:
            print(f"Warning: weights is None for t={t}")
            continue
        try:
            weights_sum += boots_periods[t]['weights']
        except TypeError as e:
            print(f"TypeError on t={t}. Type of weights: {type(boots_periods[t]['weights'])}")
            print(boots_periods[t]['weights'])
            raise e

    weights_avg = weights_sum / T0
    
    disco_boot = []
    for t in range(T_max):
        grid_key = t + 1  # grid comes from 1-based dictionary
        disco_boot.append(
            boot_counterfactuals(boots_periods[t], t + 1, mixture, weights_avg, evgrid, grid[grid_key])
        )

    return {'weights': weights_avg, 'disco_boot': disco_boot}


def get_cis(btmat, cl, og, uniform):
    """
    Helper function to calculate confidence bounds.
    btmat: shape (B, G, T_max)
    og: shape (G, T_max)
    """
    bt_diff = btmat - og
    
    if uniform:
        bt_diff_max = np.max(np.abs(bt_diff), axis=(1, 2))
        t_all = np.quantile(bt_diff_max, q=cl)
        upper = og + t_all
        lower = og - t_all
    else:
        t_upper = np.quantile(bt_diff, q=(1 - cl) / 2, axis=0)
        t_lower = np.quantile(bt_diff, q=cl + (1 - cl) / 2, axis=0)
        upper = og - t_upper
        lower = og - t_lower
        
    se = np.std(btmat, axis=0)
    return {'lower': lower, 'upper': upper, 'se': se}


def parse_boots(ci_temp, cl, q_disco, cdf_disco, q_obs, cdf_obs, uniform=True):
    """
    Parse bootstrapped counterfactuals to extract confidence intervals.
    """
    q_d = np.column_stack(q_disco)
    cdf_d = np.column_stack(cdf_disco)
    q_diff_d = np.column_stack(q_obs) - q_d
    cdf_diff_d = np.column_stack(cdf_obs) - cdf_d
    
    B = len(ci_temp)
    T = cdf_d.shape[1]
    G_q = q_d.shape[0]
    G_cdf = cdf_d.shape[0]
    
    q_boot = np.zeros((B, G_q, T))
    cdf_boot = np.zeros((B, G_cdf, T))
    q_diff_boot = np.zeros((B, G_q, T))
    cdf_diff_boot = np.zeros((B, G_cdf, T))
    weights_boot = []
    
    for b in range(B):
        weights_boot.append(ci_temp[b]['weights'])
        for t in range(T):
            boot_t = ci_temp[b]['disco_boot'][t]
            q_boot[b, :, t] = boot_t['quantile']
            cdf_boot[b, :, t] = boot_t['cdf']
            q_diff_boot[b, :, t] = boot_t['quantile_diff']
            cdf_diff_boot[b, :, t] = boot_t['cdf_diff']
            
    q_ci = get_cis(q_boot, cl, q_d, uniform)
    cdf_ci = get_cis(cdf_boot, cl, cdf_d, uniform)
    q_diff_ci = get_cis(q_diff_boot, cl, q_diff_d, uniform)
    cdf_diff_ci = get_cis(cdf_diff_boot, cl, cdf_diff_d, uniform)
    
    weights_boot = np.array(weights_boot)
    weights_ci = {
        'upper': np.quantile(weights_boot, q=cl + (1 - cl) / 2, axis=0),
        'lower': np.quantile(weights_boot, q=(1 - cl) / 2, axis=0)
    }
    
    return {
        'quantile': q_ci,
        'cdf': cdf_ci,
        'quantile_diff': q_diff_ci,
        'cdf_diff': cdf_diff_ci,
        'weights': weights_ci,
        'bootmat': {
            'quantile': q_boot,
            'cdf': cdf_boot,
            'quantile_diff': q_diff_boot,
            'cdf_diff': cdf_diff_boot
        }
    }