"""
Treatment Effect Analysis (TEA) module for DiSCo.

This module contains functions for computing Wasserstein barycenters
and aggregating treatment effects from DiSCo analysis results.
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


def disco_bc(controls_q: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute the Wasserstein barycenter in the DiSCo method.
    
    The barycenter is computed as the weighted average of quantile functions,
    as described in Definition 1, Step 4 of Gunsilius (2023).
    
    Parameters:
    -----------
    controls_q : np.ndarray
        Matrix of control quantile functions (shape: n_grid_points × n_controls)
    weights : np.ndarray
        Vector of optimal synthetic control weights (shape: n_controls,)
        
    Returns:
    --------
    np.ndarray
        The quantile function of the barycenter evaluated at the grid points
    """
    # Obtaining the Wasserstein barycenter as the weighted average of quantile functions
    thebc = controls_q @ weights
    
    return thebc


@dataclass
class DiSCoT:
    """
    S3-like class for storing aggregated treatment effects from DiSCo analysis.
    
    Analogous to the R S3 class `DiSCoT`, this class stores treatment effects,
    confidence intervals, and summary statistics.
    
    Attributes:
    -----------
    agg : str
        Aggregation method used ("quantileDiff", "quantile", "cdfDiff", or "cdf")
    treats : dict
        Dictionary of treatment effects by time period
    grid : np.ndarray
        Grid points at which effects are evaluated
    ses : dict or None
        Standard errors by time period (if CI available)
    ci_lower : dict or None
        Lower confidence interval bounds by time period
    ci_upper : dict or None
        Upper confidence interval bounds by time period
    t0 : int
        Treatment start time
    cl : float
        Confidence level (e.g., 0.95)
    N : int
        Total number of observations
    J : int
        Number of control units
    agg_df : pd.DataFrame or None
        Aggregated treatment effects and CIs (if agg is quantileDiff or cdfDiff)
    perm : dict or None
        Permutation test results (if available)
    plot : dict or None
        Plot data (matplotlib figure or dict with plot info)
    """
    agg: str
    treats: Dict
    grid: np.ndarray
    ses: Optional[Dict] = None
    ci_lower: Optional[Dict] = None
    ci_upper: Optional[Dict] = None
    t0: Optional[int] = None
    cl: Optional[float] = None
    N: Optional[int] = None
    J: Optional[int] = None
    agg_df: Optional[pd.DataFrame] = None
    perm: Optional[Dict] = None
    plot: Optional[Dict] = None
    
    def summary(self):
        """
        Print summary of aggregated treatment effects.
        
        Returns summary statistics only for quantileDiff and cdfDiff aggregations.
        """
        if self.agg_df is None:
            print(f"No treatment effects to summarize for agg='{self.agg}'.")
            print("Summary is only available for agg='quantileDiff' or agg='cdfDiff'.")
            return
        
        print(f"\nAggregation Method: {self.agg}")
        print(f"Confidence Level: {self.cl:.1%}" if self.cl else "")
        print(f"Sample Size: N={self.N}, J={self.J}")
        print("\n" + "="*80)
        print(self.agg_df.to_string(index=False))
        print("="*80)
        
        if self.perm is not None:
            print("\n\nPermutation Test Results:")
            print(self.perm)
    
    def __repr__(self):
        return f"DiSCoT(agg='{self.agg}', N={self.N}, J={self.J})"


def DiSCoTEA(disco_obj, agg: str = "quantileDiff", graph: bool = True, 
             t_plot: Optional[List] = None, savePlots: bool = False,
             xlim: Optional[Tuple] = None, ylim: Optional[Tuple] = None,
             samples: Optional[List] = None) -> DiSCoT:
    """
    Aggregate treatment effects from DiSCo analysis results.
    
    This function takes the output of a DiSCo analysis and computes aggregate
    treatment effects using a user-specified aggregation statistic. Optionally
    computes confidence intervals from bootstrap replications.
    
    Parameters:
    -----------
    disco_obj : DiSCo or dict
        Output from DiSCo analysis containing results by period
    agg : str, default "quantileDiff"
        Aggregation statistic to use. Options:
        - "quantileDiff": Difference in quantiles (target - counterfactual)
        - "quantile": Both observed and counterfactual quantiles
        - "cdfDiff": Difference in CDFs
        - "cdf": Both observed and counterfactual CDFs
    graph : bool, default True
        Whether to create plots
    t_plot : list, optional
        Time periods to include in plots. If None, includes all periods.
    savePlots : bool, default False
        Whether to save plots to disk
    xlim : tuple, optional
        X-axis limits for plots (min, max)
    ylim : tuple, optional
        Y-axis limits for plots (min, max)
    samples : list, optional
        Quantile samples for aggregation summary (default: [0.25, 0.5, 0.75])
        
    Returns:
    --------
    DiSCoT
        Object containing aggregated treatment effects and metadata
    """
    if samples is None:
        samples = [0.25, 0.5, 0.75]
    
    # Extract parameters from disco object
    # Support both dict and class-like access patterns
    if hasattr(disco_obj, 'params'):
        params = disco_obj.params
        results_periods = disco_obj.results_periods
    elif isinstance(disco_obj, dict):
        params = disco_obj.get('params', {})
        results_periods = disco_obj.get('results_periods', {})
    else:
        raise TypeError("disco_obj must have params and results_periods attributes/keys")
    
    # Reconstruct parameters
    df = params.get('df')
    t_max = df['time_col'].max() if df is not None else max(results_periods.keys())
    t_min = df['time_col'].min() if df is not None else min(results_periods.keys())
    t0 = params.get('t0')
    T0 = t0 - t_min
    T_max = t_max - t_min + 1
    CI = params.get('CI', False)
    cl = params.get('cl', 0.95)
    G = params.get('G', 100)
    q_min = params.get('q_min', 0.0)
    q_max = params.get('q_max', 1.0)
    
    # Create evaluation grid
    evgrid = np.linspace(0, 1, G + 1)
    
    t_start = t_min
    T_start = 1
    
    if t_plot is None:
        t_plot = list(range(int(t_start), int(t_max) + 1))
    
    # Calculate treatment effects based on aggregation method
    treats = {}
    treats_boot = {}
    treats_boot_q_diff = {}
    treats_boot_cdf_diff = {}
    cdfs_target = {}
    cdfs_disco = {}
    target_qtiles = {}
    disco_qtiles = {}
    grids = {}
    
    # Extract results by time period
    for i, (t_idx, period_result) in enumerate(results_periods.items()):
        # Map back to absolute time (years) using t_start (which matches the 1-based results_periods convention)
        t_period = int(t_idx) + int(t_start) - 1
        if 'target' in period_result and 'quantiles' in period_result['target']:
            target_qtiles[t_period] = period_result['target']['quantiles']       
        # Get DiSCo quantiles
        if 'DiSCo' in period_result and 'quantile' in period_result['DiSCo']:
            disco_qtiles[t_period] = period_result['DiSCo']['quantile']
        
        # Get grid for this period
        if 'target' in period_result and 'grid' in period_result['target']:
            grids[t_period] = period_result['target']['grid']
        
        # Compute CDFs if needed
        if agg in ["cdf", "cdfDiff"] and t_period in target_qtiles:
            if t_period in grids:
                grid = grids[t_period]
            else:
                # Use a default grid if not specified
                all_values = np.concatenate([
                    target_qtiles[t_period].flatten() if hasattr(target_qtiles[t_period], 'flatten') else np.atleast_1d(target_qtiles[t_period]),
                    disco_qtiles[t_period].flatten() if hasattr(disco_qtiles[t_period], 'flatten') else np.atleast_1d(disco_qtiles[t_period])
                ])
                grid = np.linspace(all_values.min(), all_values.max(), len(evgrid))
            
            grids[t_period] = grid
            cdfs_target[t_period] = _empirical_cdf(target_qtiles[t_period], grid)
            cdfs_disco[t_period] = _empirical_cdf(disco_qtiles[t_period], grid)
        
        # Handle confidence intervals from bootstrap
        if CI and 'DiSCo' in period_result and 'CI' in period_result['DiSCo']:
            ci_data = period_result['DiSCo']['CI']
            if 'bootmat' in ci_data:
                treats_boot[t_period] = ci_data['bootmat']
            if 'bootmat_q_diff' in ci_data:
                treats_boot_q_diff[t_period] = ci_data['bootmat_q_diff']
            if 'bootmat_cdf_diff' in ci_data:
                treats_boot_cdf_diff[t_period] = ci_data['bootmat_cdf_diff']
                
    # Compute treatment effects based on aggregation method
    agg_df = None
    sds = {}
    ci_lower_dict = {}
    ci_upper_dict = {}
    if agg == "quantileDiff":
        # Quantile differences: target - DiSCo
        for t_period in target_qtiles:
            if t_period in disco_qtiles:
                treats[t_period] = target_qtiles[t_period] - disco_qtiles[t_period]
        
        grid = evgrid
        
        # Compute CIs from bootstrap if available
        if CI:
            for t_period in treats_boot:
                if t_period in treats_boot_q_diff:
                    # Use accurate bootstrap quantile differences
                    boot_diffs = treats_boot_q_diff[t_period]
                else:
                    # Fallback if bootmat_q_diff wasn't saved
                    boot_diffs = target_qtiles[t_period][:, None] - treats_boot[t_period]
                    
                sds[t_period] = np.std(boot_diffs, axis=1, ddof=1) if boot_diffs.ndim > 1 else np.std(boot_diffs, ddof=1)
                
                # Center CIs using the point estimate directly: og +/- quantiles_of_deviation
                # The bootstrap empirical distribution of the difference handles the variance
                # but might be biased. Classic bootstrap percentile uses just quantiles.
                # Since the original plots might center perfectly, we can use the quantiles directly.
                ci_lower_dict[t_period] = np.quantile(boot_diffs, (1 - cl) / 2, axis=1 if boot_diffs.ndim > 1 else -1)
                ci_upper_dict[t_period] = np.quantile(boot_diffs, cl + (1 - cl) / 2, axis=1 if boot_diffs.ndim > 1 else -1)

        
        # Create aggregation dataframe
        agg_df = _create_aggregation_df(treats, grid, t0, samples, CI, 
                                       ci_lower_dict, ci_upper_dict, sds, cl,
                                       agg_type="quantileDiff", q_min=q_min, q_max=q_max)
    
    elif agg == "quantile":
        # Both counterfactual and observed quantiles
        for t_period in target_qtiles:
            if t_period in disco_qtiles:
                treats[t_period] = disco_qtiles[t_period]
                target_qtiles[t_period] = target_qtiles[t_period]
        
        grid = evgrid
        
    elif agg == "cdfDiff":
        # CDF differences
        for t_period in cdfs_target:
            if t_period in cdfs_disco:
                treats[t_period] = cdfs_target[t_period] - cdfs_disco[t_period]
        
        if grids:
            grid = list(grids.values())[0]  # Use first grid
        else:
            grid = evgrid
        
        # Compute CIs from bootstrap if available
        if CI:
            for t_period in treats_boot:
                if t_period in treats_boot_cdf_diff:
                    boot_diffs = treats_boot_cdf_diff[t_period]
                else:
                    boot_mat = treats_boot[t_period] # shape (1001, B)
                    if boot_mat.ndim > 1:
                        # Calculate CDF for each bootstrap sample
                        boot_cdfs = np.zeros((len(grid), boot_mat.shape[1]))        
                        for b in range(boot_mat.shape[1]):
                            boot_cdfs[:, b] = _empirical_cdf(boot_mat[:, b], grid)  
                        boot_diffs = cdfs_target[t_period][:, None] - boot_cdfs     
                    else:
                        boot_cdfs = _empirical_cdf(boot_mat, grid)
                        boot_diffs = cdfs_target[t_period] - boot_cdfs

                sds[t_period] = np.std(boot_diffs, axis=1, ddof=1) if boot_diffs.ndim > 1 else np.std(boot_diffs, ddof=1)
                
                # To ensure valid confidence bounds, if the mean of bootstrap difference 
                # is biased relative to the actual point estimate, standard pivot method centers it.
                # However, percentile method direct usage might be what R does if bias is low.
                # Let's apply basic CIs directly from standard quantiles of the diffs
                ci_lower_dict[t_period] = np.quantile(boot_diffs, (1 - cl) / 2, axis=1 if boot_diffs.ndim > 1 else -1)
                ci_upper_dict[t_period] = np.quantile(boot_diffs, cl + (1 - cl) / 2, axis=1 if boot_diffs.ndim > 1 else -1)

        # Create aggregation dataframe
        agg_df = _create_aggregation_df(treats, grid, t0, samples, CI,
                                       ci_lower_dict, ci_upper_dict, sds, cl,
                                       agg_type="cdfDiff", q_min=q_min, q_max=q_max)
    
    elif agg == "cdf":
        # Both counterfactual and observed CDFs
        for t_period in cdfs_disco:
            treats[t_period] = cdfs_disco[t_period]
        
        if grids:
            grid = list(grids.values())[0]
        else:
            grid = evgrid
    
    # Prepare result metadata
    N = len(df) if df is not None else None
    J = len([t for t in treats]) if treats else None
    
    # Create plot data if requested
    plot_data = None
    if graph:
        plot_data = _prepare_plot_data(treats, grid, cdfs_target if agg in ["cdf"] else None,
                                      t_start, t_max, CI, ci_lower_dict, ci_upper_dict,
                                      agg, xlim, ylim, t_plot)
    
    # Return DiSCoT object
    return DiSCoT(
        agg=agg,
        treats=treats,
        grid=grid,
        ses=sds if sds else None,
        ci_lower=ci_lower_dict if ci_lower_dict else None,
        ci_upper=ci_upper_dict if ci_upper_dict else None,
        t0=t0,
        cl=cl,
        N=N,
        J=J,
        agg_df=agg_df,
        perm=None,  # Placeholder for permutation results
        plot=plot_data
    )


def _empirical_cdf(data: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Compute empirical CDF on a given grid.
    
    Parameters:
    -----------
    data : np.ndarray
        Data samples (1D array)
    grid : np.ndarray
        Grid points at which to evaluate CDF
        
    Returns:
    --------
    np.ndarray
        CDF values at grid points
    """
    if data.ndim > 1:
        data = data.flatten()
    
    sorted_data = np.sort(data)
    cdf_values = np.searchsorted(sorted_data, grid, side='right') / len(data)
    return cdf_values


def _create_aggregation_df(treats: Dict, grid: np.ndarray, t0: int, samples: List,
                          CI: bool, ci_lower: Dict, ci_upper: Dict, sds: Dict,
                          cl: float, agg_type: str = "quantileDiff",
                          q_min: float = 0.0, q_max: float = 1.0) -> pd.DataFrame:
    """
    Create aggregated treatment effects dataframe.
    
    Parameters:
    -----------
    treats : dict
        Treatment effects by time period
    grid : np.ndarray
        Grid points
    t0 : int
        Treatment start time
    samples : list
        Quantile samples for aggregation
    CI : bool
        Whether CIs are available
    ci_lower : dict
        Lower CI bounds by period
    ci_upper : dict
        Upper CI bounds by period
    sds : dict
        Standard deviations by period
    cl : float
        Confidence level
    agg_type : str
        Type of aggregation ("quantileDiff" or "cdfDiff")
    q_min : float
        Minimum quantile
    q_max : float
        Maximum quantile
        
    Returns:
    --------
    pd.DataFrame
        Aggregated treatment effects
    """
    # Ensure samples are in [0, 1]
    samples = sorted(set([0.0] + list(samples) + [1.0]))
    
    # If quantile range is restricted, only use the endpoints
    if q_min > 0 or q_max < 1:
        samples = [0.0, 1.0]
    
    rows = []
    grid_indices = np.linspace(0, len(grid) - 1, len(grid)).astype(int)
    
    for t_period in sorted(treats.keys()):
        if t_period >= t0:  # Only post-treatment periods
            effect = treats[t_period]
            
            # Iterate over quantile ranges
            for i in range(len(samples) - 1):
                f_idx = int(samples[i] * (len(grid) - 1))
                t_idx = int(samples[i + 1] * (len(grid) - 1))
                
                # Average treatment effect in this range
                effect_agg = np.mean(effect[f_idx:t_idx + 1])
                
                if CI and t_period in ci_lower:
                    boot_effect = effect[f_idx:t_idx + 1]
                    sd_agg = np.std(boot_effect, ddof=1)
                    # Compute CI from bootstrap distribution
                    ci_l_agg = np.percentile(boot_effect, (1 - cl) / 2 * 100)
                    ci_u_agg = np.percentile(boot_effect, (cl + (1 - cl) / 2) * 100)
                else:
                    sd_agg = np.nan
                    ci_l_agg = np.nan
                    ci_u_agg = np.nan
                
                rows.append({
                    'Time': t_period,
                    'X_from': grid[f_idx] if not (q_min > 0 or q_max < 1) else q_min,
                    'X_to': grid[t_idx] if not (q_min > 0 or q_max < 1) else q_max,
                    'Effect': effect_agg,
                    'Std. Error': sd_agg,
                    f'CI_Lower_{int(cl*100)}%': ci_l_agg,
                    f'CI_Upper_{int(cl*100)}%': ci_u_agg,
                })
    
    if rows:
        df = pd.DataFrame(rows)
        return df.round(4)
    else:
        return None


def _prepare_plot_data(treats: Dict, grid: np.ndarray, 
                       target_cdf: Optional[Dict] = None,
                       t_start: Optional[int] = None, t_max: Optional[int] = None,
                       CI: bool = False, ci_lower: Optional[Dict] = None,
                       ci_upper: Optional[Dict] = None,
                       agg: str = "quantileDiff", xlim: Optional[Tuple] = None,
                       ylim: Optional[Tuple] = None, t_plot: Optional[List] = None) -> Dict:
    """
    Prepare data for plotting treatment effects over time.
    
    Parameters:
    -----------
    treats : dict
        Treatment effects by time period
    grid : np.ndarray
        Grid points
    target_cdf : dict, optional
        Target CDFs by time period (for comparison)
    t_start, t_max : int
        Time range
    CI : bool
        Whether to include CIs
    ci_lower, ci_upper : dict
        CI bounds
    agg : str
        Aggregation method
    xlim, ylim : tuple
        Axis limits
    t_plot : list
        Time periods to include
        
    Returns:
    --------
    dict
        Plot data for visualization
    """
    plot_data = {
        'treats': treats,
        'grid': grid,
        'agg': agg,
        'xlim': xlim,
        'ylim': ylim,
        't_plot': t_plot,
        'CI': CI,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'target_cdf': target_cdf,
    }
    
    # Compute default axis limits if not provided
    if xlim is None and grid is not None:
        xlim = (np.percentile(grid, 1), np.percentile(grid, 99))
        plot_data['xlim'] = xlim
    
    if ylim is None and treats:
        all_treats = np.concatenate([v.flatten() if hasattr(v, 'flatten') else np.atleast_1d(v) 
                                    for v in treats.values()])
        ylim = (np.percentile(all_treats, 1), np.percentile(all_treats, 99))
        plot_data['ylim'] = ylim
    
    return plot_data