import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Optional, Union
from .models import DiSCoResult, DiSCoTEAResult

class BaseTEA(ABC):
    def __init__(self, disco: DiSCoResult, agg: str, graph: bool, t_plot: Optional[List[int]], xlim, ylim, samples):
        self.disco = disco
        self.agg = agg
        self.graph = graph
        self.t_plot = t_plot
        self.xlim = xlim
        self.ylim = ylim
        self.samples = samples
        
        self.df = disco.params.df
        self.t0 = disco.params.t0

        # Create time mapper
        self.periods = sorted(list(self.disco.results_periods.keys()))
        self._raw_mapper = {t: t for t in self.periods}
        if isinstance(self.df, pd.DataFrame):
            for col in self.df.columns:
                if col != 't_col' and self.t0 in self.df[col].values:
                    self._raw_mapper = dict(zip(self.df['t_col'], self.df[col]))
                    break
        self.t_mapper = {k: int(v) if isinstance(v, float) and v.is_integer() else v for k, v in self._raw_mapper.items()}
        self.t_start = min(self.periods)
        self.t_max = max(self._raw_mapper.values())
        if self.t_plot is None:
            self.t_plot = [self.t_mapper[p] for p in self.periods]

        self.CI = disco.params.CI
        self.q_min = disco.params.q_min
        self.q_max = disco.params.q_max

    @abstractmethod
    def evaluate(self) -> DiSCoTEAResult:
        pass

class ClassicTEA(BaseTEA):
    def evaluate(self) -> DiSCoTEAResult:
        treats = {}
        treats_boot = {}
        target_vals = {}
        sds = {}
        ci_lower = {}
        ci_upper = {}

        grid = self.disco.evgrid

def plot_dist_over_time(
    cdf_centered: dict, 
    grid_cdf: np.ndarray, 
    t_start: int, 
    t_max: int, 
    CI: bool, 
    ci_lower: dict, 
    ci_upper: dict, 
    ylim=None, 
    xlim=None, 
    cdf=True, 
    xlab="Distribution Difference", 
    ylab="CDF", 
    obs_line=None, 
    t_plot=None
):
    if t_plot is None:
        t_plot = list(cdf_centered.keys())
    else:
        t_plot = [t for t in t_plot if t in cdf_centered]
        
    n_plots = len(t_plot)
    if n_plots == 0:
        return None
        
    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 3 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
        
    fig.suptitle("Distribution of Treatment Effects Over Time")
    
    for ax, t in zip(axes, t_plot):
        y = cdf_centered[t]
        x = grid_cdf
        
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
            
        ax.plot(x, y, color="black", linestyle="-")
        
        if obs_line is not None and t in obs_line:
            ax.plot(x, obs_line[t], color="blue", linestyle="--", label="Observed")
            
        ax.axhline(0, color="grey", linestyle="-")
        if cdf:
            ax.axhline(1, color="grey", linestyle="-")
            
        if CI and ci_lower is not None and ci_upper is not None:
            ax.plot(x, ci_lower[t], color="grey", linestyle="--")
            ax.plot(x, ci_upper[t], color="grey", linestyle="--")
            
        ax.set_title(f"Time: {t}")
        ax.set_ylabel(ylab)
        ax.grid(True)
        
    axes[-1].set_xlabel(xlab)
    plt.tight_layout()
    
    return fig

class MultivariateTEA(BaseTEA):
    def evaluate(self) -> DiSCoTEAResult:
        raise NotImplementedError("Multivariate Treatment Effects (e.g. Sliced Wasserstein Distance over time) not yet implemented.")

def disco_tea(
    disco: DiSCoResult,
    agg: str = "quantileDiff",
    graph: bool = True,
    t_plot: Optional[List[int]] = None,
    xlim=None,
    ylim=None,
    samples=[0.25, 0.5, 0.75]
) -> DiSCoTEAResult:
    if agg in ["wasserstein_dist", "sinkhorn_div"]:
        strategy = MultivariateTEA(disco, agg, graph, t_plot, xlim, ylim, samples)
        return strategy.evaluate()

    
    df = disco.params.df
    t0 = disco.params.t0
    # Einfaches Mapping: Finde die Spalte, die den Startzeitpunkt t0 enthält
    _raw_mapper = {t: t for t in disco.results_periods.keys()}
    if isinstance(df, pd.DataFrame):
        for col in df.columns:
            if col != 't_col' and t0 in df[col].values:
                _raw_mapper = dict(zip(df['t_col'], df[col]))
                break

    t_max = max(_raw_mapper.values())
    t_mapper = {k: int(v) if isinstance(v, float) and v.is_integer() else v for k, v in _raw_mapper.items()}
    
    # periods is a sorted list of integer periods 1, 2, 3...
    periods = sorted(list(disco.results_periods.keys()))
    t_start = min(periods)
    t_max = max(periods)
    
    CI = disco.params.CI
    evgrid = disco.evgrid
    q_min = disco.params.q_min
    q_max = disco.params.q_max
    
    if t_plot is None:
        t_plot = [t_mapper[p] for p in periods]

    treats = {}
    treats_boot = {}
    target_vals = {}
    
    sds = {}
    ci_lower = {}
    ci_upper = {}

    grid = evgrid
    
    if agg == "cdfDiff":
        # Diff of cdfs: t_cdf - c_cdf
        # Wait, for mixture we evaluate ECDFs on the grid...
        # In Python discopy, target_grid is specific to each period but evgrid is consistent
        for t in periods:
            p_res = disco.results_periods[t]
            t_cdf_eval = p_res.target.cdf  # length evgrid? No, length of grid
            # Let's align exactly with R's `stats::ecdf` over `target$grid`.
            # In Python, we have grid_ord stored in target.grid.
            # But the simplest way is to evaluate ECDF on that grid_ord.
            # Or use the stored ones: DiSCo CDF and Target CDF are evaluated on target.grid!
            # Python models store them exactly like that.
            
            c_cdf = p_res.DiSCo.cdf
            t_cdf = p_res.target.cdf
            treats[t_mapper[t]] = t_cdf - c_cdf
            grid = p_res.target.grid # grid can vary in mixture ?
            
            if CI:
                # bootmat.cdf_diff is array of shape (G, T, B)
                # python stores `['cdf'][:, t_idx, b]`
                # We need to map `t` (normalized time) to index
                t_idx = periods.index(t)
                treats_boot[t_mapper[t]] = disco.CI.bootmat.cdf_diff[:, t_idx, :]
                
        if CI:
            agg_nam = "cdf_diff"
            for t in periods:
                t_idx = periods.index(t)
                sds[t_mapper[t]] = disco.CI.cdf_diff.se[:, t_idx]
                ci_lower[t_mapper[t]] = disco.CI.cdf_diff.lower[:, t_idx]
                ci_upper[t_mapper[t]] = disco.CI.cdf_diff.upper[:, t_idx]
                
        if graph:
            if ylim is None:
                all_vals = np.concatenate(list(treats.values()))
                ylim = (np.percentile(all_vals, 1), np.percentile(all_vals, 99))
            if xlim is None:
                all_grids = grid
                xlim = (np.percentile(all_grids, 1), np.percentile(all_grids, 99))
                
            fig = plot_dist_over_time(
                treats, grid, t_start, t_max, CI, ci_lower, ci_upper, 
                ylim=ylim, xlim=xlim, xlab="Y", ylab="CDF Change", 
                t_plot=t_plot
            )
            
    elif agg == "cdf":
        for t in periods:
            p_res = disco.results_periods[t]
            treats[t_mapper[t]] = p_res.DiSCo.cdf
            target_vals[t_mapper[t]] = p_res.target.cdf
            grid = p_res.target.grid
            
            if CI:
                t_idx = periods.index(t)
                treats_boot[t_mapper[t]] = disco.CI.bootmat.cdf[:, t_idx, :]
                
        if CI:
            for t in periods:
                t_idx = periods.index(t)
                sds[t_mapper[t]] = disco.CI.cdf.se[:, t_idx]
                ci_lower[t_mapper[t]] = disco.CI.cdf.lower[:, t_idx]
                ci_upper[t_mapper[t]] = disco.CI.cdf.upper[:, t_idx]
                
        if graph:
            if xlim is None:
                xlim = (np.min(grid), np.max(grid))
                
            fig = plot_dist_over_time(
                treats, grid, t_start, t_max, CI, ci_lower, ci_upper, 
                ylim=ylim, xlim=xlim, xlab="Y", ylab="CDF", 
                obs_line=target_vals, t_plot=t_plot
            )
            
    elif agg == "quantileDiff":
        grid = evgrid
        for t in periods:
            p_res = disco.results_periods[t]
            c_qtile = p_res.DiSCo.quantile
            t_qtile = p_res.target.quantiles
            treats[t_mapper[t]] = t_qtile - c_qtile
            
            if CI:
                t_idx = periods.index(t)
                treats_boot[t_mapper[t]] = disco.CI.bootmat.quantile_diff[:, t_idx, :]
                
        if CI:
            for t in periods:
                t_idx = periods.index(t)
                sds[t_mapper[t]] = disco.CI.quantile_diff.se[:, t_idx]
                ci_lower[t_mapper[t]] = disco.CI.quantile_diff.lower[:, t_idx]
                ci_upper[t_mapper[t]] = disco.CI.quantile_diff.upper[:, t_idx]
                
        if graph:
            if ylim is None:
                all_vals = np.concatenate(list(treats.values()))
                ylim = (np.percentile(all_vals, 1), np.percentile(all_vals, 99))
            if xlim is None:
                xlim = (np.min(grid), np.max(grid))
                
            fig = plot_dist_over_time(
                treats, grid, t_start, t_max, CI, ci_lower, ci_upper, 
                ylim=ylim, xlim=xlim, xlab="Quantile", ylab="Treatment Effect", 
                cdf=False, t_plot=t_plot
            )
            
    elif agg == "quantile":
        grid = evgrid
        for t in periods:
            p_res = disco.results_periods[t]
            treats[t_mapper[t]] = p_res.DiSCo.quantile
            target_vals[t_mapper[t]] = p_res.target.quantiles
            
            if CI:
                t_idx = periods.index(t)
                treats_boot[t_mapper[t]] = disco.CI.bootmat.quantile[:, t_idx, :]
                
        if CI:
            for t in periods:
                t_idx = periods.index(t)
                sds[t_mapper[t]] = disco.CI.quantile.se[:, t_idx]
                ci_lower[t_mapper[t]] = disco.CI.quantile.lower[:, t_idx]
                ci_upper[t_mapper[t]] = disco.CI.quantile.upper[:, t_idx]

        if graph:
            if ylim is None:
                all_vals = np.concatenate(list(treats.values()))
                ylim = (np.percentile(all_vals, 1), np.percentile(all_vals, 99))
            if xlim is None:
                xlim = (np.min(grid), np.max(grid))
                
            fig = plot_dist_over_time(
                treats, grid, t_start, t_max, CI, ci_lower, ci_upper, 
                ylim=ylim, xlim=xlim, xlab="Quantile", ylab="Treatment Effect", 
                cdf=False, obs_line=target_vals, t_plot=t_plot
            )
    else:
        raise ValueError(f"Unknown aggregation method: {agg}")

    # Build Summary DataFrame
    agg_df = None
    
    if agg in ["cdfDiff", "quantileDiff"]:
        samples = np.array(samples)
        if np.min(samples) > 0:
            samples = np.insert(samples, 0, 0.0)
        if np.max(samples) < 1:
            samples = np.append(samples, 1.0)
            
        if q_min != 0 or q_max != 1:
            samples = np.array([0.0, 1.0])
            
        post_treat_periods = [t for t in periods if t_mapper[t] >= t0]
        
        grid_q = np.round(samples * (len(grid) - 1)).astype(int)
        
        rows = []
        for t in post_treat_periods:
            for j in range(len(grid_q) - 1):
                f = grid_q[j]
                to = grid_q[j+1]
                
                t_mapped = t_mapper[t]
                
                # R does treats[f:to], inclusive indexing. Python slice is f:to+1
                slice_arr = treats[t_mapped][f:to+1]
                treats_agg = np.mean(slice_arr)
                
                if CI:
                    boot_slice = treats_boot[t_mapped][f:to+1, :]
                    treats_boot_agg = np.mean(boot_slice, axis=0) # Mean over grid -> array of shape (B,)
                    sd_agg = np.std(treats_boot_agg, ddof=1)
                    ci_lower_agg = np.quantile(treats_boot_agg, (1 - disco.params.cl) / 2)
                    ci_upper_agg = np.quantile(treats_boot_agg, disco.params.cl + (1 - disco.params.cl) / 2)
                else:
                    sd_agg = np.nan
                    ci_lower_agg = np.nan
                    ci_upper_agg = np.nan
                    
                rows.append({
                    "Time": t_mapped, # Map back to actual year from data
                    "X_from": grid[f],
                    "X_to": grid[to],
                    "Treats": treats_agg,
                    "Std. Error": sd_agg,
                    "CI_Lower": ci_lower_agg,
                    "CI_Upper": ci_upper_agg
                })
                
        out = pd.DataFrame(rows)
        if not out.empty:
            if q_min != 0 or q_max != 1:
                out["X_from"] = q_min
                out["X_to"] = q_max
                
            sig_text = []
            for _, row in out.iterrows():
                if CI and not np.isnan(row["CI_Lower"]) and not np.isnan(row["CI_Upper"]):
                    if row["CI_Lower"] > 0 or row["CI_Upper"] < 0:
                        sig_text.append("*")
                    else:
                        sig_text.append("")
                else:
                    sig_text.append("")
            out["Sig"] = sig_text
            
            ooi = "CDF Delta" if agg == "cdfDiff" else "Quantile Delta"
            cband_text = f"[{100 * disco.params.cl:.0f}% Conf. Band]"
            
            # Format and rename columns exactly like R
            out = out.round(4)
            out.columns = ["Time", "X_from", "X_to", ooi, "Std. Error", f"[{100 * disco.params.cl:.0f}% ", "Conf. Band]", "Sig"]
            agg_df = out

    N_obs = len(disco.params.df)
    J_controls = len(disco.control_ids)

    return DiSCoTEAResult(
        agg=agg,
        treats=treats,
        grid=grid,
        ses=sds if CI else None,
        ci_lower=ci_lower if CI else None,
        ci_upper=ci_upper if CI else None,
        t0=t0,
        cl=disco.params.cl if CI else 0.95,
        N=N_obs,
        J=J_controls,
        agg_df=agg_df,
        perm=disco.perm,
        plot=fig if graph else None
    )