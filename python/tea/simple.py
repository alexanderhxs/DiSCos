import numpy as np
import pandas as pd
from ..models import DiSCoTEAResult
from .base import BaseTEA

class SimpleTEA(BaseTEA):
    def evaluate(self) -> DiSCoTEAResult:
        treats = {}
        rows = []
        
        for t in self.periods:
            p_res = self.disco.results_periods[t]
            
            target_data = p_res.target.data
            controls_data = p_res.controls.data
            weights = self.disco.weights
            
            if target_data is not None and controls_data is not None and weights is not None and len(target_data) > 0 and len(controls_data) > 0:
                controls_all = np.concatenate(controls_data)
                
                weights_cf = np.array([weights[i] / len(controls_data[i]) for i in range(len(controls_data)) for _ in range(len(controls_data[i]))], dtype=np.float64)
                
                # Adjust for numerical instabilities
                negative_weigths = np.where(weights_cf < 0.0)
                weights_cf[negative_weigths] = 0

                check = np.sum(weights_cf)
                if check > 0:
                    weights_cf = weights_cf / check
                else:
                    weights_cf = np.ones(len(weights_cf), dtype=np.float64) / len(weights_cf)
                
                target_means = np.mean(target_data, axis=0)
                disco_means = np.average(controls_all, axis=0, weights=weights_cf)
                mean_diffs = target_means - disco_means
                
                if self.disco.params.is_multivariate:
                    cov_t = np.cov(target_data, rowvar=False)
                    cov_d = np.cov(controls_all, aweights=weights_cf, rowvar=False)
                    cov_diff = cov_t - cov_d
                else:
                    cov_t = float(np.var(target_data))
                    c_flat = controls_all.flatten()
                    w_mean = np.average(c_flat, weights=weights_cf)
                    cov_d = float(np.average((c_flat - w_mean)**2, weights=weights_cf))
                    cov_diff = np.array([[cov_t - cov_d]])

                treats[self.t_mapper[t]] = {
                    "mean_diff": mean_diffs,
                    "cov_diff": cov_diff
                }
                
                rows.append({
                    "Time": self.t_mapper[t],
                    "Mean Diff (Norm)": float(np.linalg.norm(mean_diffs)),
                    "Cov Diff (Frobenius)": float(np.linalg.norm(cov_diff, ord='fro'))
                })
            else:
                treats[self.t_mapper[t]] = {
                    "mean_diff": np.nan,
                    "cov_diff": np.nan
                }
                rows.append({
                    "Time": self.t_mapper[t],
                    "Mean Diff (Norm)": np.nan,
                    "Cov Diff (Frobenius)": np.nan
                })

        agg_df = pd.DataFrame(rows).round(4)
        
        N_obs = len(self.disco.params.df)
        J_controls = len(self.disco.control_ids)
        
        return DiSCoTEAResult(
            agg=self.agg, treats=treats, grid=np.array([]),
            ses=None, ci_lower=None, ci_upper=None,
            t0=self.t0, cl=self.disco.params.cl if self.CI else 0.95, N=N_obs, J=J_controls,
            agg_df=agg_df, perm=self.disco.perm, plot=None
        )
