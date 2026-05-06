import numpy as np
import pandas as pd
import ot
import itertools
from ..models import DiSCoTEAResult
from .base import BaseTEA

class TransportMapTEA(BaseTEA):
    def evaluate(self) -> DiSCoTEAResult:
        treats = {}
        quant_list = sorted(list(set([0.0] + [s for s in self.samples if 0 < s < 1] + [1.0])))
        q_labels = [f"Q{i+1}" for i in range(len(quant_list) - 1)]

        for t in self.periods:
            p_res = self.disco.results_periods[t]
            target_dist = np.asarray(p_res.target.data)
            if target_dist.ndim == 1:
                target_dist = target_dist.reshape(-1, 1)
            N = len(target_dist)

            controls_data = p_res.controls.data
            weights = self.disco.weights if self.disco.weights is not None else np.ones(len(controls_data))/len(controls_data)
            
            pool_data = []
            pool_w = []
            for c_data, w in zip(controls_data, weights):
                c_data_arr = np.asarray(c_data)
                if c_data_arr.ndim == 1:
                    c_data_arr = c_data_arr.reshape(-1, 1)
                if len(c_data_arr) > 0 and w > 1e-5:
                    pool_data.append(c_data_arr)
                    pool_w.extend([w / len(c_data_arr)] * len(c_data_arr))
            
            if len(pool_data) > 0:
                pool_data = np.vstack(pool_data)
                pool_w = np.array(pool_w)
                pool_w_norm = pool_w / np.sum(pool_w)
                np.random.seed(42 + self.t_mapper[t]) 
                sample_idx = np.random.choice(len(pool_data), size=N, p=pool_w_norm)
                counterfactual_dist = pool_data[sample_idx]
            else:
                counterfactual_dist = target_dist.copy()

            weights_target = np.ones(N) / N
            weights_cf = np.ones(N) / N

            all_data = np.vstack((target_dist, counterfactual_dist))
            mean_val = np.mean(all_data, axis=0)
            std_val = np.std(all_data, axis=0)
            std_val[std_val == 0] = 1.0 

            target_scaled = (target_dist - mean_val) / std_val
            cf_scaled = (counterfactual_dist - mean_val) / std_val

            cost_matrix = ot.dist(target_scaled, cf_scaled, metric='euclidean')
            T_samples = ot.emd(weights_target, weights_cf, cost_matrix)

            df_target = pd.DataFrame(target_dist)
            df_cf = pd.DataFrame(counterfactual_dist)

            def assign_quantiles(series):
                return pd.qcut(series, q=quant_list, labels=q_labels, duplicates='drop')

            num_dims = target_dist.shape[1]
            bin_cols_target, bin_cols_cf = [], []
            
            for d in range(num_dims):
                df_target[f'd{d}_bin'] = assign_quantiles(df_target[d])
                df_cf[f'd{d}_bin'] = assign_quantiles(df_cf[d])
                bin_cols_target.append(f'd{d}_bin')
                bin_cols_cf.append(f'd{d}_bin')

            df_target['Combined_Bin'] = df_target[bin_cols_target].astype(str).agg('_'.join, axis=1)
            df_cf['Combined_Bin'] = df_cf[bin_cols_cf].astype(str).agg('_'.join, axis=1)

            all_bins = ["_".join(comb) for comb in itertools.product(q_labels, repeat=num_dims)]

            H_target = np.column_stack([df_target['Combined_Bin'] == b for b in all_bins]).astype(float)
            H_cf = np.column_stack([df_cf['Combined_Bin'] == b for b in all_bins]).astype(float)

            T_aggregated = H_target.T @ T_samples @ H_cf
            df_T_agg = pd.DataFrame(np.round(T_aggregated * 100, 2), index=all_bins, columns=all_bins)
            
            treats[self.t_mapper[t]] = df_T_agg

        return DiSCoTEAResult(
            agg=self.agg, treats=treats, grid=self.disco.evgrid,
            ses=None, ci_lower=None, ci_upper=None,
            t0=self.t0, cl=self.disco.params.cl if self.CI else 0.95,
            N=len(self.df), J=len(self.disco.control_ids),
            agg_df=None, perm=self.disco.perm, plot=None
        )
