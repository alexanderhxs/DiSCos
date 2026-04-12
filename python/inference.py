import numpy as np
from joblib import Parallel, delayed
from .solvers import disco_weights_reg, disco_mixture
from .utils import getGrid, myQuant
from .models import CIResult, CIBand, CIWeights, CIBootmat

def disco_ci_iter(t, controls_t, target_t, grid_t, T0, M=1000,
                  evgrid=np.linspace(0, 1, 1001), mixture=False, simplex=False, replace=True):
    """
    Performs a single bootstrap redraw for a single time period `t`.
    
    Resamples the target and control distributions with replacement, recomputes 
    empirical CDFs and quantiles on the evaluation grid, and if in a pre-treatment 
    period (t <= T0), solves for the optimal weights.
    """
    # --- RESAMPLE TARGET STATS ---
    t_len = len(target_t)
    mytar = target_t[np.random.choice(t_len, size=t_len, replace=replace)]
    
    # Recompute empirical quantiles and CDFs based on the resampled target array
    mytar_q = myQuant(mytar, evgrid)
    mytar_sorted = np.sort(mytar)
    mytar_cdf = np.searchsorted(mytar_sorted, grid_t, side='right') / t_len
    
    # --- RESAMPLE CONTROLS STATS ---
    mycon_list = []
    mycon_q = np.zeros((len(evgrid), len(controls_t)))
    mycon_cdf = np.zeros((len(grid_t), len(controls_t)))
    
    for ii, controls_t_i in enumerate(controls_t):
        c_len = len(controls_t_i)
        
        # Resample each control unit drawing `c_len` observations with replacement
        mycon = controls_t_i[np.random.choice(c_len, size=c_len, replace=replace)]
        mycon_list.append(mycon)
        
        # Compute counterfactual column slice for this control unit in the resampled period
        mycon_q[:, ii] = myQuant(mycon, evgrid)
        
        # Build strict CDF vectors by counting how many elements fall below each grid point
        mycon_sorted = np.sort(mycon)
        mycon_cdf[:, ii] = np.searchsorted(mycon_sorted, grid_t, side='right') / c_len
        
    # --- SOLVE FOR WEIGHTS ---
    # We only recalculate weights for pre-treatment periods (t <= T0)
    if t <= T0:
        if not mixture:
            lambda_weights = disco_weights_reg(mycon_list, mytar, M=M, simplex=simplex)
        else:
            grid_min, grid_max, grid_rand, grid_ord = getGrid(mytar, mycon_list, len(grid_t))
            mixt = disco_mixture(mycon_list, mytar, grid_min, grid_max, grid_ord, M, simplex)
            lambda_weights = mixt['weights_opt']
    else:
        lambda_weights = None
        
    return {
        "weights": lambda_weights,
        "target": {"quantile": mytar_q, "cdf": mytar_cdf},
        "controls": {"quantile": mycon_q, "cdf": mycon_cdf}
    }

def boot_counterfactuals(result_t, t, mixture, weights, evgrid, grid_t):
    """
    Computes bootstrapped counterfactual distributions for period `t` given 
    the averaged weights from the pre-treatment bootstrapped periods.
    
    Differences between the target and synthetic (counterfactual) distributions 
    are calculated for both quantiles and CDFs.
    """
    controls_cdf = result_t["controls"]["cdf"]
    controls_q = result_t["controls"]["quantile"]
    target_q = result_t["target"]["quantile"]
    target_cdf = result_t["target"]["cdf"]
    
    if mixture:
        # MIXTURE APPROACH: Interpolate the synthetic CDF 
        # Calculate cross-sectional synthetic CDF as the dot product between the controls' CDF 
        # matrix and the averaged weights across the entire evaluation grid.
        cdf_t = controls_cdf @ weights
        
        # Estimate the synthetic quantile function by inverting the synthetic CDF. 
        # For each quantile bin in 'evgrid', we locate the first grid index where 
        # the synthetic CDF accumulates mass >= y.
        q_t = np.array([grid_t[np.argmax(cdf_t >= (y - 1e-5))] for y in evgrid])
    else:
        # BARYCENTER/QUANTILE APPROACH: Interpolate the synthetic Quantile function
        # Compute the counterfactual quantiles as the dot product of the 
        # control quantiles matrix and the averaged weights 
        q_t = controls_q @ weights
        
        # Estimate the synthetic CDF from the synthetic quantile outputs
        # by searching where the evaluated grid points fall inside the sorted counterfactual quantiles.
        q_t_sorted = np.sort(q_t)
        cdf_t = np.searchsorted(q_t_sorted, grid_t, side='right') / len(q_t)
        
    # --- DISTRIBUTIONAL DIFFERENCES ---
    # Difference = Actual Distribution - Synthetic Distribution (Counterfactual)
    cdf_diff = target_cdf - cdf_t
    q_diff = target_q - q_t
    
    return {
        "cdf": cdf_t,
        "quantile": q_t,
        "quantile_diff": q_diff,
        "cdf_diff": cdf_diff
    }

def disco_ci(redraw, controls, target, T_max, T0, grids, evgrid,
             mixture=False, simplex=False, M=1000, replace=True, seed=None):
    """
    Executes a single complete bootstrap draw across all time periods (T=1 to T_max).
    
    Averages pre-treatment weights (t <= T0) and generates counterfactuals for all periods using 
    the averaged weights. `redraw` tracks the current bootstrap iteration and is optionally used for deterministic seeding.
    """
    if seed:
        np.random.seed(seed + redraw)

    boots_periods = []
    for t in range(T_max):
        result_t = disco_ci_iter(
            t + 1, controls[t], target[t], grids[t], T0,
            M=M, evgrid=evgrid, mixture=mixture, simplex=simplex, replace=replace
        )
        boots_periods.append(result_t)
        
    # extract and average weights (only from t <= T0)
    pre_weights = [bp["weights"] for bp in boots_periods[:int(T0)] if bp["weights"] is not None]
    weights = sum(pre_weights) / T0
    
    disco_boot = []
    for t in range(T_max):
        cf = boot_counterfactuals(boots_periods[t], t + 1, mixture, weights, evgrid, grids[t])
        disco_boot.append(cf)
        
    return {
        "weights": weights,
        "disco_boot": disco_boot
    }

def parse_boots(CI_temp, cl, q_disco, cdf_disco, q_obs, cdf_obs, uniform=True):
    """
    Parses the full array of bootstrap results across all draws and periods.
    
    Computes Standard Errors (SE) and (1-cl) Confidence Intervals (CIs) for:
    - Counterfactual Quantiles & CDFs
    - Difference between observed and counterfactual Quantiles & CDFs
    - Model weights
    
    If `uniform` is True, calculates uniform confidence bands based on the maximum 
    absolute deviation across the grid. Otherwise, point-wise exact quantiles.
    """
    n_boots = len(CI_temp)
    T_max = len(q_disco)
    G_q = q_disco[0].shape[0]
    G_cdf = cdf_disco[0].shape[0]
    
    # --- PREPARE 3D MATRICES ---
    # Dimensions: (Grid size, Time Periods, Number of Bootstraps)
    q_boot = np.zeros((G_q, T_max, n_boots))
    cdf_boot = np.zeros((G_cdf, T_max, n_boots))
    q_diff = np.zeros((G_q, T_max, n_boots))
    cdf_diff = np.zeros((G_cdf, T_max, n_boots))
    
    # 1) Reformat multidimensional lists from the CI_temp object into 3D Numpy Arrays
    # to efficiently compute array-level quantiles across the `n_boots` dimension.
    for b in range(n_boots):
        for t in range(T_max):
            boot_t = CI_temp[b]["disco_boot"][t]
            q_boot[:, t, b] = boot_t["quantile"]
            cdf_boot[:, t, b] = boot_t["cdf"]
            q_diff[:, t, b] = boot_t["quantile_diff"]
            cdf_diff[:, t, b] = boot_t["cdf_diff"]
            
    # 2) Flatten the main model's periods into 2D Arrays (Grid Size x Time Periods)
    q_d = np.column_stack(q_disco)
    cdf_d = np.column_stack(cdf_disco)
    q_diff_d = np.column_stack([q_obs[t] - q_disco[t] for t in range(T_max)])
    cdf_diff_d = np.column_stack([cdf_obs[t] - cdf_disco[t] for t in range(T_max)])
    
    def get_CIs(btmat, cl, og, uniform):
        """
        Calculates (1 - `cl`) error bands. 
        `btmat` is the [Grid x T-max x Boots] bootstrapped matrix.
        `og` is the [Grid x T-max] actual observed matrix from the model. 
        """
        # Calculate element-wise difference between the boots and the overall estimate.
        # np.newaxis aligns the OG 2D array along the third `boot` dimension 
        # for proper broadcasting across the Boot dimension map.
        bt_diff = btmat - og[:, :, np.newaxis]
        
        if uniform:
            # For uniform bounds: Compute maximum absolute deviation across the entire Grid distribution 
            # for each time period, producing a matrix of (T x Boots) size.
            bt_diff_abs_max = np.max(np.abs(bt_diff), axis=0) # T x B
            
            # Find the critical threshold bounding `cl` % of the deviations for each individual Time period `t`
            t_crit = np.quantile(bt_diff_abs_max, cl, axis=1) # T
            
            # Form symmetric error bands around the observed estimate block
            upper = og + t_crit[np.newaxis, :]
            lower = og - t_crit[np.newaxis, :]
        else:
            # For Point-wise bounds: Identify exact asymmetric quantiles 
            # per individual grid-point evaluating the entire `boots` space individually.
            # Example: with cl=0.95 -> calculates bounds isolating the middle 95% 
            # by evaluating the (2.5% and 97.5%) tails along axis 2 (Boots dimension)
            t_lower = np.quantile(bt_diff, (1 - cl) / 2, axis=2)
            t_upper = np.quantile(bt_diff, cl + (1 - cl) / 2, axis=2)
            
            # Shift actual estimates using computed tail thresholds constraints
            upper = og - t_lower
            lower = og - t_upper
            
        # Compute pure Standard Error as standard deviation across the Boots axis
        se = np.std(btmat, axis=2, ddof=1)
        return CIBand(lower=lower, upper=upper, se=se)
        
    # --- ASSEMBLE RESULTS ---
    # Retrieve confidence matrices for primary quantities
    q_CI = get_CIs(q_boot, cl, q_d, uniform)
    cdf_CI = get_CIs(cdf_boot, cl, cdf_d, uniform)
    q_diff_CI = get_CIs(q_diff, cl, q_diff_d, uniform)
    cdf_diff_CI = get_CIs(cdf_diff, cl, cdf_diff_d, uniform)
    
    # Calculate confidence bands separately mapped across the scalar weights 
    weights_boot = np.column_stack([CI_temp[b]["weights"] for b in range(n_boots)])
    weights_CI = CIWeights(
        lower=np.quantile(weights_boot, (1 - cl) / 2, axis=1),
        upper=np.quantile(weights_boot, cl + (1 - cl) / 2, axis=1)
    )
    
    return CIResult(
        quantile=q_CI,
        cdf=cdf_CI,
        quantile_diff=q_diff_CI,
        cdf_diff=cdf_diff_CI,
        weights=weights_CI,
        bootmat=CIBootmat(
            quantile=q_boot,
            cdf=cdf_boot,
            quantile_diff=q_diff,
            cdf_diff=cdf_diff
        )
    )

def run_bootstrap_ci(model, uniform=True, replace=True):
    """
    Orchestrates the parallel computation of bootstrap confidence intervals 
    for the entire `DiSCo` model.
    
    Reads data and parameters directly from the solved `model` instance, 
    distributes `B` bootstrap jobs recursively across periods, and parses the 
    results into a structured `CIResult` object.
    """
    T_max = len(model.periods)
    T0 = model.T0_idx
    B = model.B
    cl = model.cl
    seed = model.seed
    uniform = model.uniform
    
    controls = []
    target = []
    grids = []
    q_disco = []
    cdf_disco = []
    q_obs = []
    cdf_obs = []
    
    # Needs to be extracted iteratively over the sorted periods to maintain order
    for t in model.periods:
        res = model.results_by_period[t]
        controls.append(res.controls.data)
        target.append(res.target.data)
        grids.append(res.target.grid)
        
        q_disco.append(res.DiSCo.quantile)
        cdf_disco.append(res.DiSCo.cdf)
        q_obs.append(res.target.quantiles)
        cdf_obs.append(res.target.cdf)
        
    CI_bootmat = Parallel(n_jobs=model.num_cores)(
        delayed(disco_ci)(
            b, controls, target, T_max, T0, grids, model.evgrid,
            mixture=model.mixture, simplex=model.simplex, M=model.M, replace=replace, seed=seed
        )
        for b in range(B)
    )
    
    return parse_boots(CI_bootmat, cl, q_disco, cdf_disco, q_obs, cdf_obs, uniform)
