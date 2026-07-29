import numpy as np
import pandas as pd
from ..models import DiSCoTEAResult
from .base import BaseTEA

class ProbMassTEA(BaseTEA):
    def __init__(self, disco, agg, graph, t_plot, xlim, ylim, samples, bounds=None, quantiles=None, **kwargs):
        super().__init__(disco, agg, graph, t_plot, xlim, ylim, samples)
        self.bounds = bounds
        self.quantiles = quantiles

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
                
                # Ensure data is at least 2D for consistency
                if target_data.ndim == 1:
                    target_data = target_data[:, np.newaxis]
                if controls_all.ndim == 1:
                    controls_all = controls_all[:, np.newaxis]
                
                D = target_data.shape[1]
                
                # Determine bounds
                active_bounds = []
                if self.bounds is not None:
                    active_bounds = self.bounds
                elif self.quantiles is not None:
                    # Calculate quantiles on the synthetic control distribution
                    for d in range(D):
                        # Sort data to compute quantiles
                        dim_data = controls_all[:, d]
                        sort_idx = np.argsort(dim_data)
                        sorted_data = dim_data[sort_idx]
                        sorted_weights = weights_cf[sort_idx]
                        cum_weights = np.cumsum(sorted_weights)
                        
                        q_min_val = self.quantiles[d][0]
                        q_max_val = self.quantiles[d][1]
                        
                        # Find values corresponding to cumulative probabilities
                        idx_min = np.searchsorted(cum_weights, q_min_val)
                        idx_max = np.searchsorted(cum_weights, q_max_val)
                        
                        # Handle edge cases where idx might be out of bounds
                        idx_min = min(idx_min, len(sorted_data) - 1)
                        idx_max = min(idx_max, len(sorted_data) - 1)
                        
                        b_min = sorted_data[idx_min]
                        b_max = sorted_data[idx_max]
                        active_bounds.append((b_min, b_max))
                else:
                    # If neither provided, default to full range (mass = 1.0 everywhere)
                    active_bounds = [(-np.inf, np.inf) for _ in range(D)]
                
                # Verify bounds structure
                if len(active_bounds) < D:
                    # Extend active bounds with inf for missing dimensions
                    active_bounds.extend([(-np.inf, np.inf) for _ in range(D - len(active_bounds))])
                
                # Filter target data
                in_bounds_target = np.ones(len(target_data), dtype=bool)
                for d in range(D):
                    b_min, b_max = active_bounds[d]
                    in_bounds_target &= (target_data[:, d] >= b_min) & (target_data[:, d] <= b_max)
                
                target_mass = np.mean(in_bounds_target)
                
                # Filter control data
                in_bounds_control = np.ones(len(controls_all), dtype=bool)
                for d in range(D):
                    b_min, b_max = active_bounds[d]
                    in_bounds_control &= (controls_all[:, d] >= b_min) & (controls_all[:, d] <= b_max)
                
                control_mass = float(np.sum(weights_cf[in_bounds_control]))
                
                mass_diff = target_mass - control_mass

                treats[self.t_mapper[t]] = {
                    "mass_diff": mass_diff,
                    "target_mass": target_mass,
                    "control_mass": control_mass,
                    "active_bounds": active_bounds
                }
                
                rows.append({
                    "Time": self.t_mapper[t],
                    "Target Mass": float(target_mass),
                    "Control Mass": float(control_mass),
                    "Mass Diff": float(mass_diff)
                })
            else:
                treats[self.t_mapper[t]] = {
                    "mass_diff": np.nan,
                    "target_mass": np.nan,
                    "control_mass": np.nan,
                    "active_bounds": []
                }
                rows.append({
                    "Time": self.t_mapper[t],
                    "Target Mass": np.nan,
                    "Control Mass": np.nan,
                    "Mass Diff": np.nan
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
