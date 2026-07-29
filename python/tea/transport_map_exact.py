import numpy as np
import pandas as pd
import ot
import itertools
from ..models import DiSCoTEAResult
from .base import BaseTEA

class TransportMapTEA2(BaseTEA):
    
    def _calculate_transport_map(self, weights, quant_list, q_labels):

        treats = {}
        for t in self.periods:
            p_res = self.disco.results_periods[t]
            target_dist = np.asarray(p_res.target.data)
            if target_dist.ndim == 1:
                target_dist = target_dist.reshape(-1, 1)
            N = len(target_dist)

            controls_data = p_res.controls.data
            
            controls_all = np.concatenate(controls_data) if controls_data is not None and len(controls_data) > 0 else np.empty((0, target_dist.shape[1]))

            weights_cf = np.array([weights[i] / len(controls_data[i]) for i in range(len(controls_data)) for _ in range(len(controls_data[i]))], dtype=np.float64)
            
            # Adjust for numerical instabilities or negative weights from non-simplex solvers
            negative_weights = np.where(weights_cf < 0.0)
            weights_cf[negative_weights] = 0
            check = np.sum(weights_cf)
            if check > 0:
                weights_cf = weights_cf / check
            else:
                weights_cf = np.ones(len(weights_cf), dtype=np.float64) / len(weights_cf)
                
            weights_target = np.ones(N, dtype=np.float64) / N

            #all_data = np.vstack((target_dist, controls_all))
            #mean_val = np.mean(all_data, axis=0)
            #std_val = np.std(all_data, axis=0)
            #std_val[std_val == 0] = 1.0 

            target_std = np.std(target_dist, axis=0)
            target_std[target_std == 0] = 1.0

            #target_scaled = (target_dist  / target_std)
            #cf_scaled = (controls_all / target_std)

            cost_matrix = ot.dist(target_dist, controls_all, metric='euclidean')
            T_samples = ot.emd(weights_target, weights_cf, cost_matrix, numItermax=1e7)

            df_target = pd.DataFrame(target_dist)
            df_cf = pd.DataFrame(controls_all)

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
            df_T_agg = pd.DataFrame(np.round(T_aggregated * 100, 4), index=all_bins, columns=all_bins)
            
            treats[self.t_mapper[t]] = df_T_agg

        return treats

    def evaluate(self) -> DiSCoTEAResult:
        
        treats = {}
        quant_list = sorted(list(set([0.0] + [s for s in self.samples if 0 < s < 1] + [1.0])))
        q_labels = [f"Q{i+1}" for i in range(len(quant_list) - 1)]

        treats['Estimate'] = self._calculate_transport_map(self.disco.weights, quant_list, q_labels)

        if self.CI:
            weights_lower = self.disco.CI.weights.lower
            weights_upper = self.disco.CI.weights.upper

            treats['Upper'] = self._calculate_transport_map(weights_upper, quant_list, q_labels)
            treats['Lower'] = self._calculate_transport_map(weights_lower, quant_list, q_labels)

        treats_H0 = {}
        for t, df_t in treats['Estimate'].items():
            
            # 1. Reine NumPy Null-Matrix in der richtigen Größe erstellen
            arr_h0 = np.zeros((len(df_t.index), len(df_t.columns)))

            # 2. Marginals berechnen
            target_marginals = df_t.sum(axis=1)

            # 3. Diagonale in der ungeschützten NumPy-Matrix füllen
            np.fill_diagonal(arr_h0, target_marginals)

            # 4. Erst jetzt das Pandas DataFrame daraus bauen
            df_h0 = pd.DataFrame(arr_h0, index=df_t.index, columns=df_t.columns)

            treats_H0[t] = df_h0
        treats['H0'] = treats_H0
        



        return DiSCoTEAResult(
            agg=self.agg, treats=treats, grid=self.disco.evgrid,
            ses=None, ci_lower=None, ci_upper=None,
            t0=self.t0, cl=self.disco.params.cl if self.CI else 0.95,
            N=len(self.df), J=len(self.disco.control_ids),
            agg_df=None, perm=self.disco.perm, plot=None
        )
