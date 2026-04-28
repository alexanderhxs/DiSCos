import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict
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

class DensityRatioTEA(BaseTEA):
    def evaluate(self) -> DiSCoTEAResult:
        raise NotImplementedError("Density Ratio TEA is not yet implemented.")

class TransportMapTEA(BaseTEA):
    def evaluate(self) -> DiSCoTEAResult:
        raise NotImplementedError("Transport Map TEA (Optimal Transport) is not yet implemented.")

class MarginalTEA(BaseTEA):
    def evaluate(self) -> Union[DiSCoTEAResult, Dict[int, DiSCoTEAResult]]:
        import copy
        first_period = self.periods[0]
        target_data = self.disco.results_periods[first_period].target.data
        
        is_multi = getattr(target_data, 'ndim', 1) > 1 and target_data.shape[1] > 1
        
        if not is_multi:
            # Falls wider Erwarten 1D, falle auf Single ClassicTEA zurück
            strategy = ClassicTEA(self.disco, "quantileDiff", self.graph, self.t_plot, self.xlim, self.ylim, self.samples)
            return {0: strategy.evaluate()}

        num_dims = target_data.shape[1]
        marginal_results = {}
        
        for d in range(num_dims):
            # Shallow Copy des Hauptobjekts
            disco_d = copy.copy(self.disco)
            # Deep Copy der Period-Results um Arrays gefahrlos zu slicen
            disco_d.results_periods = copy.deepcopy(self.disco.results_periods)
            
            for t in self.periods:
                p_res = disco_d.results_periods[t]
                
                # Ersetze die Matrizen in den Kopien durch ihren 1D-Sektor je Dimension
                if p_res.target.data is not None and p_res.target.data.ndim > 1:
                    p_res.target.data = p_res.target.data[:, d]
                if p_res.target.quantiles is not None and p_res.target.quantiles.ndim > 1:
                    p_res.target.quantiles = p_res.target.quantiles[:, d]
                if p_res.DiSCo.quantile is not None and p_res.DiSCo.quantile.ndim > 1:
                    p_res.DiSCo.quantile = p_res.DiSCo.quantile[:, d]
                if p_res.target.cdf is not None and p_res.target.cdf.ndim > 1:
                    p_res.target.cdf = p_res.target.cdf[:, d]
                if p_res.DiSCo.cdf is not None and p_res.DiSCo.cdf.ndim > 1:
                    p_res.DiSCo.cdf = p_res.DiSCo.cdf[:, d]
                    
            # Nutze ClassicTEA (mit quantileDiff Logik) um die 1D-Marginalverteilung zu berechnen
            strategy = ClassicTEA(disco_d, "quantileDiff", self.graph, self.t_plot, self.xlim, self.ylim, self.samples)
            marginal_results[d] = strategy.evaluate()
            
        return marginal_results

class ClassicTEA(BaseTEA):
    def evaluate(self) -> DiSCoTEAResult:
        treats = {}
        treats_boot = {}
        target_vals = {}
        sds = {}
        ci_lower = {}
        ci_upper = {}

        grid = self.disco.evgrid
        
        if self.agg == "cdfDiff":
            for t in self.periods:
                p_res = self.disco.results_periods[t]
                c_cdf = p_res.DiSCo.cdf
                t_cdf = p_res.target.cdf
                treats[self.t_mapper[t]] = t_cdf - c_cdf
                grid = p_res.target.grid
                
                if self.CI:
                    t_idx = self.periods.index(t)
                    treats_boot[self.t_mapper[t]] = self.disco.CI.bootmat.cdf_diff[:, t_idx, :]
                    
            if self.CI:
                for t in self.periods:
                    t_idx = self.periods.index(t)
                    sds[self.t_mapper[t]] = self.disco.CI.cdf_diff.se[:, t_idx]
                    ci_lower[self.t_mapper[t]] = self.disco.CI.cdf_diff.lower[:, t_idx]
                    ci_upper[self.t_mapper[t]] = self.disco.CI.cdf_diff.upper[:, t_idx]
                    
            if self.graph:
                ylim = self.ylim
                if ylim is None:
                    all_vals = np.concatenate(list(treats.values()))
                    ylim = (np.percentile(all_vals, 1), np.percentile(all_vals, 99))
                xlim = self.xlim
                if xlim is None:
                    xlim = (np.percentile(grid, 1), np.percentile(grid, 99))
                    
                fig = plot_dist_over_time(
                    treats, grid, self.t_start, self.t_max, self.CI, ci_lower, ci_upper, 
                    ylim=ylim, xlim=xlim, xlab="Y", ylab="CDF Change", 
                    t_plot=self.t_plot
                )
                
        elif self.agg == "cdf":
            for t in self.periods:
                p_res = self.disco.results_periods[t]
                treats[self.t_mapper[t]] = p_res.DiSCo.cdf
                target_vals[self.t_mapper[t]] = p_res.target.cdf
                grid = p_res.target.grid
                
                if self.CI:
                    t_idx = self.periods.index(t)
                    treats_boot[self.t_mapper[t]] = self.disco.CI.bootmat.cdf[:, t_idx, :]
                    
            if self.CI:
                for t in self.periods:
                    t_idx = self.periods.index(t)
                    sds[self.t_mapper[t]] = self.disco.CI.cdf.se[:, t_idx]
                    ci_lower[self.t_mapper[t]] = self.disco.CI.cdf.lower[:, t_idx]
                    ci_upper[self.t_mapper[t]] = self.disco.CI.cdf.upper[:, t_idx]
                    
            if self.graph:
                xlim = self.xlim
                if xlim is None:
                    xlim = (np.min(grid), np.max(grid))
                    
                fig = plot_dist_over_time(
                    treats, grid, self.t_start, self.t_max, self.CI, ci_lower, ci_upper, 
                    ylim=self.ylim, xlim=xlim, xlab="Y", ylab="CDF", 
                    obs_line=target_vals, t_plot=self.t_plot
                )
                
        elif self.agg == "quantileDiff":
            grid = self.disco.evgrid
            for t in self.periods:
                p_res = self.disco.results_periods[t]
                c_qtile = p_res.DiSCo.quantile
                t_qtile = p_res.target.quantiles
                treats[self.t_mapper[t]] = t_qtile - c_qtile
                
                if self.CI:
                    t_idx = self.periods.index(t)
                    treats_boot[self.t_mapper[t]] = self.disco.CI.bootmat.quantile_diff[:, t_idx, :]
                    
            if self.CI:
                for t in self.periods:
                    t_idx = self.periods.index(t)
                    sds[self.t_mapper[t]] = self.disco.CI.quantile_diff.se[:, t_idx]
                    ci_lower[self.t_mapper[t]] = self.disco.CI.quantile_diff.lower[:, t_idx]
                    ci_upper[self.t_mapper[t]] = self.disco.CI.quantile_diff.upper[:, t_idx]
                    
            if self.graph:
                ylim = self.ylim
                if ylim is None:
                    all_vals = np.concatenate(list(treats.values()))
                    ylim = (np.percentile(all_vals, 1), np.percentile(all_vals, 99))
                xlim = self.xlim
                if xlim is None:
                    xlim = (np.min(grid), np.max(grid))
                    
                fig = plot_dist_over_time(
                    treats, grid, self.t_start, self.t_max, self.CI, ci_lower, ci_upper, 
                    ylim=ylim, xlim=xlim, xlab="Quantile", ylab="Treatment Effect", 
                    cdf=False, t_plot=self.t_plot
                )
                
        elif self.agg == "quantile":
            grid = self.disco.evgrid
            for t in self.periods:
                p_res = self.disco.results_periods[t]
                treats[self.t_mapper[t]] = p_res.DiSCo.quantile
                target_vals[self.t_mapper[t]] = p_res.target.quantiles
                
                if self.CI:
                    t_idx = self.periods.index(t)
                    treats_boot[self.t_mapper[t]] = self.disco.CI.bootmat.quantile[:, t_idx, :]
                    
            if self.CI:
                for t in self.periods:
                    t_idx = self.periods.index(t)
                    sds[self.t_mapper[t]] = self.disco.CI.quantile.se[:, t_idx]
                    ci_lower[self.t_mapper[t]] = self.disco.CI.quantile.lower[:, t_idx]
                    ci_upper[self.t_mapper[t]] = self.disco.CI.quantile.upper[:, t_idx]

            if self.graph:
                ylim = self.ylim
                if ylim is None:
                    all_vals = np.concatenate(list(treats.values()))
                    ylim = (np.percentile(all_vals, 1), np.percentile(all_vals, 99))
                xlim = self.xlim
                if xlim is None:
                    xlim = (np.min(grid), np.max(grid))
                    
                fig = plot_dist_over_time(
                    treats, grid, self.t_start, self.t_max, self.CI, ci_lower, ci_upper, 
                    ylim=ylim, xlim=xlim, xlab="Quantile", ylab="Treatment Effect", 
                    cdf=False, obs_line=target_vals, t_plot=self.t_plot
                )
        else:
            raise ValueError(f"Unknown aggregation method for ClassicTEA: {self.agg}")

        # Build Summary DataFrame
        agg_df = None
        
        if self.agg in ["cdfDiff", "quantileDiff"]:
            samples = np.array(self.samples)
            if np.min(samples) > 0:
                samples = np.insert(samples, 0, 0.0)
            if np.max(samples) < 1:
                samples = np.append(samples, 1.0)
                
            if self.q_min != 0 or self.q_max != 1:
                samples = np.array([0.0, 1.0])
                
            post_treat_periods = [t for t in self.periods if self.t_mapper[t] >= self.t0]
            
            grid_q = np.round(samples * (len(grid) - 1)).astype(int)
            
            rows = []
            for t in post_treat_periods:
                for j in range(len(grid_q) - 1):
                    f = grid_q[j]
                    to = grid_q[j+1]
                    
                    t_mapped = self.t_mapper[t]
                    
                    slice_arr = treats[t_mapped][f:to+1]
                    treats_agg = np.mean(slice_arr)
                    
                    if self.CI:
                        boot_slice = treats_boot[t_mapped][f:to+1, :]
                        treats_boot_agg = np.mean(boot_slice, axis=0) # Mean over grid -> array of shape (B,)
                        sd_agg = np.std(treats_boot_agg, ddof=1)
                        ci_lower_agg = np.quantile(treats_boot_agg, (1 - self.disco.params.cl) / 2)
                        ci_upper_agg = np.quantile(treats_boot_agg, self.disco.params.cl + (1 - self.disco.params.cl) / 2)
                    else:
                        sd_agg = np.nan
                        ci_lower_agg = np.nan
                        ci_upper_agg = np.nan
                        
                    rows.append({
                        "Time": t_mapped, 
                        "X_from": grid[f],
                        "X_to": grid[to],
                        "Treats": treats_agg,
                        "Std. Error": sd_agg,
                        "CI_Lower": ci_lower_agg,
                        "CI_Upper": ci_upper_agg
                    })
                    
            out = pd.DataFrame(rows)
            if not out.empty:
                if self.q_min != 0 or self.q_max != 1:
                    out["X_from"] = self.q_min
                    out["X_to"] = self.q_max
                    
                sig_text = []
                for _, row in out.iterrows():
                    if self.CI and not np.isnan(row["CI_Lower"]) and not np.isnan(row["CI_Upper"]):
                        if row["CI_Lower"] > 0 or row["CI_Upper"] < 0:
                            sig_text.append("*")
                        else:
                            sig_text.append("")
                    else:
                        sig_text.append("")
                out["Sig"] = sig_text
                
                ooi = "CDF Delta" if self.agg == "cdfDiff" else "Quantile Delta"
                
                out = out.round(4)
                out.columns = ["Time", "X_from", "X_to", ooi, "Std. Error", f"[{100 * self.disco.params.cl:.0f}% ", "Conf. Band]", "Sig"]
                agg_df = out

        N_obs = len(self.disco.params.df)
        J_controls = len(self.disco.control_ids)

        return DiSCoTEAResult(
            agg=self.agg,
            treats=treats,
            grid=grid,
            ses=sds if self.CI else None,
            ci_lower=ci_lower if self.CI else None,
            ci_upper=ci_upper if self.CI else None,
            t0=self.t0,
            cl=self.disco.params.cl if self.CI else 0.95,
            N=N_obs,
            J=J_controls,
            agg_df=agg_df,
            perm=self.disco.perm,
            plot=fig if self.graph and 'fig' in locals() else None
        )

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
) -> Union[DiSCoTEAResult, Dict[int, DiSCoTEAResult]]:
    # Dispatcher Logic based on Strategy Pattern
    if agg in ["wasserstein_dist", "sinkhorn_div"]:
        strategy = MultivariateTEA(disco, agg, graph, t_plot, xlim, ylim, samples)
    elif agg in ["cdfDiff", "cdf", "quantileDiff", "quantile"]:
        strategy = ClassicTEA(disco, agg, graph, t_plot, xlim, ylim, samples)
    elif agg == "density_ratio":
        strategy = DensityRatioTEA(disco, agg, graph, t_plot, xlim, ylim, samples)
    elif agg == "transport_map":
        strategy = TransportMapTEA(disco, agg, graph, t_plot, xlim, ylim, samples)
    elif agg == "marginals":
        strategy = MarginalTEA(disco, agg, graph, t_plot, xlim, ylim, samples)
    else:
        raise ValueError(f"Unknown aggregation method: {agg}")
    
    return strategy.evaluate()