import numpy as np
import pandas as pd
from ..models import DiSCoTEAResult
from .base import BaseTEA, plot_dist_over_time

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
                    
                fig = plot_dist_over_time(treats, grid, self.t_start, self.t_max, self.CI, ci_lower, ci_upper, ylim=ylim, xlim=xlim, xlab="Y", ylab="CDF Change", t_plot=self.t_plot)
                
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
                    
                fig = plot_dist_over_time(treats, grid, self.t_start, self.t_max, self.CI, ci_lower, ci_upper, ylim=self.ylim, xlim=xlim, xlab="Y", ylab="CDF", obs_line=target_vals, t_plot=self.t_plot)
                
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
                    
                fig = plot_dist_over_time(treats, grid, self.t_start, self.t_max, self.CI, ci_lower, ci_upper, ylim=ylim, xlim=xlim, xlab="Quantile", ylab="Treatment Effect", cdf=False, t_plot=self.t_plot)
                
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
                    
                fig = plot_dist_over_time(treats, grid, self.t_start, self.t_max, self.CI, ci_lower, ci_upper, ylim=ylim, xlim=xlim, xlab="Quantile", ylab="Treatment Effect", cdf=False, obs_line=target_vals, t_plot=self.t_plot)
        else:
            raise ValueError(f"Unknown aggregation method for ClassicTEA: {self.agg}")

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
                        treats_boot_agg = np.mean(boot_slice, axis=0) 
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
            agg=self.agg, treats=treats, grid=grid,
            ses=sds if self.CI else None, ci_lower=ci_lower if self.CI else None, ci_upper=ci_upper if self.CI else None,
            t0=self.t0, cl=self.disco.params.cl if self.CI else 0.95, N=N_obs, J=J_controls,
            agg_df=agg_df, perm=self.disco.perm, plot=fig if self.graph and 'fig' in locals() else None
        )
