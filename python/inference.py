import numpy as np
from joblib import Parallel, delayed
from .solvers import disco_weights_reg, disco_mixture
from .utils import getGrid, myQuant
from .models import CIResult, CIBand, CIWeights, CIBootmat

def disco_ci_iter(t, controls_t, target_t, grid_t, T0, solver, M=1000,
                  evgrid=np.linspace(0, 1, 1001), simplex=False, replace=True):
    """
    Performs a single bootstrap redraw for a single time period `t`.
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
    if t <= T0:
        G_grid = len(grid_t) - 1
        grid_min, grid_max, grid_rand, grid_ord = getGrid(mytar, mycon_list, G_grid)

        lambda_weights = solver.fit_weights(
            target=mytar, controls=mycon_list, M=M, simplex=simplex,
            q_min=0, q_max=1, grid_min=grid_min, grid_max=grid_max, grid_rand=grid_rand, grid_ord=grid_ord
        )
    else:
        lambda_weights = None

    return {
        "weights": lambda_weights,
        "target": {"data": mytar, "quantile": mytar_q, "cdf": mytar_cdf, "grid_t": grid_t},
        "controls": {"data": mycon_list, "quantile": mycon_q, "cdf": mycon_cdf}
    }

def boot_counterfactuals(result_t, t, solver, weights, evgrid):        
    """
    Computes bootstrapped counterfactual distributions for period `t` given     
    the averaged weights from the pre-treatment bootstrapped periods.
    """
    controls_cdf = result_t["controls"]["cdf"]
    controls_q = result_t["controls"]["quantile"]
    target_q = result_t["target"]["quantile"]
    target_cdf = result_t["target"]["cdf"]
    grid_t = result_t["target"]["grid_t"]
    target_data = result_t["target"]["data"]
    controls_data = result_t["controls"]["data"]

    eval_res = solver.evaluate_counterfactual(
        target=target_data,
        controls=controls_data,
        weights=weights,
        grid_ord=grid_t,
        evgrid=evgrid,
        controls_cdf=controls_cdf,
        controls_q=controls_q,
        target_q=target_q
    )
    
    cdf_t = eval_res["disco_cdf"]
    q_t = eval_res["disco_quantile"]

    # Difference = Actual Distribution - Synthetic Distribution (Counterfactual)
    if target_cdf is not None and cdf_t is not None:
        cdf_diff = target_cdf - cdf_t
    else:
        cdf_diff = None

    if target_q is not None and q_t is not None:
        q_diff = target_q - q_t
    else:
        q_diff = None

    return {
        "cdf": cdf_t,
        "quantile": q_t,
        "quantile_diff": q_diff,
        "cdf_diff": cdf_diff
    }

def disco_ci(redraw, controls, target, T_max, T0, grids, evgrid,
             solver, simplex=False, M=1000, replace=True, seed=None):    
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
            t + 1, controls[t], target[t], grids[t], T0, solver=solver,
            M=M, evgrid=evgrid, simplex=simplex, replace=replace
        )
        boots_periods.append(result_t)

    # extract and average weights (only from t <= T0)
    pre_weights = [bp["weights"] for bp in boots_periods[:int(T0)] if bp["weights"] is not None]
    weights = sum(pre_weights) / T0 if len(pre_weights) > 0 else np.zeros(len(controls[0]))

    disco_boot = []
    for t in range(T_max):
        cf = boot_counterfactuals(boots_periods[t], t + 1, solver, weights, evgrid)
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
        
        # Fallback vectors with NaNs handling multidimensional missing 1D stats 
        q_d = res.DiSCo.quantile if res.DiSCo.quantile is not None else np.full_like(model.evgrid, np.nan)
        cdf_d = res.DiSCo.cdf if res.DiSCo.cdf is not None else np.full_like(model.evgrid, np.nan)
        q_o = res.target.quantiles if res.target.quantiles is not None else np.full_like(model.evgrid, np.nan)
        cdf_o = res.target.cdf if res.target.cdf is not None else np.full_like(model.evgrid, np.nan)

        q_disco.append(q_d)
        cdf_disco.append(cdf_d)
        q_obs.append(q_o)
        cdf_obs.append(cdf_o)

    CI_bootmat = Parallel(n_jobs=model.num_cores)(
        delayed(disco_ci)(
            b, controls, target, T_max, T0, grids, model.evgrid,
            solver=model.solver, simplex=model.simplex, M=model.M, replace=replace, seed=seed
        ) for b in range(B)
    )

    return parse_boots(CI_bootmat, cl, q_disco, cdf_disco, q_obs, cdf_obs, uniform)
