import numpy as np
import pandas as pd
import itertools
from ..models import  PreTreatmentFitMetrics, DiSCoResult
from scipy.stats import ks_2samp
# pyrefly: ignore [missing-import]
import ot
# pyrefly: ignore [missing-import]
import scoringrules as sr

def calculate_pretreatment_fit(disco_res: DiSCoResult, eval_size: int = 1000) -> PreTreatmentFitMetrics:
    """
    Calculates goodness-of-fit metrics for all pre-treatment periods.
    For 1D, calculates deterministically on the grid without sampling,
    supporting negative weights (simplex=False).
    For Multi-D, falls back to pooling with non-negative weights if needed.
    """
    results_by_period = disco_res.results_periods
    periods = sorted(list(results_by_period.keys()))
    
    # Use the pre-calculated T0 index
    t0_idx = disco_res.params.t0_idx
            
    per_period_metrics = {}
    
    for t in periods: # All periods 
        
            
        p_res = results_by_period[t]
        
        target_data = np.asarray(p_res.target.data)
        controls_data = p_res.controls.data
        weights = disco_res.weights
        
        if len(target_data) > 0 and len(controls_data) > 0 and weights is not None:
            
            # Energy Distance Calculation
            J = len(controls_data)
            A = np.zeros(J)
            D = np.zeros((J, J))

            for j in range(J):
                c = np.asarray(controls_data[j])
                if c.size == 0 or weights[j] == 0:
                    continue
                diff = c[:, None, :] - target_data[None, :, :]
                A[j] = np.mean(np.linalg.norm(diff, axis=-1))
                for k in range(j, J):
                    d = np.asarray(controls_data[k])
                    if d.size == 0 or weights[k] == 0:
                        continue
                    diff = c[:, None, :] - d[None, :, :]
                    D[j, k] = np.mean(np.linalg.norm(diff, axis=-1))
                    D[k, j] = D[j, k]
            
            target_diff = target_data[:, None, :] - target_data[None, :, :]
            target_spread = np.mean(np.linalg.norm(target_diff, axis=-1))
            
            energy_divergence = (A @ weights) - 0.5 * (weights.T @ D @ weights) - 0.5 * target_spread

            # Wasserstein-1 Distance 
            controls_all = np.concatenate(controls_data)
            weights_cf = np.array([weights[i] / len(controls_data[i]) for i in range(len(controls_data)) for _ in range(len(controls_data[i]))], dtype=np.float64)
            weights_target = np.ones(len(target_data), dtype=np.float64) / len(target_data)

            # Adjust for numerical instabilities
            negative_weigths = np.where(weights_cf < 0.0)
            weights_cf[negative_weigths] = 0

            check = np.sum(weights_cf)
            if check > 0:
                weights_cf = weights_cf / check
            else:
                weights_cf = np.ones(len(weights_cf), dtype=np.float64) / len(weights_cf)
                
            weights_target = weights_target / np.sum(weights_target)


            target_std = np.std(target_data, axis=0)
            target_std[target_std == 0] = 1.0
            target_scaled = target_data / target_std
            cf_scaled = controls_all / target_std

            cost_matrix = ot.dist(target_data, controls_all, metric='euclidean')
            w1 = float(ot.emd2(weights_target, weights_cf, cost_matrix, numItermax=int(1e6)))
            
            cost_matrix_w2 = ot.dist(target_data, controls_all, metric='sqeuclidean')
            w2 = float(ot.emd2(weights_target, weights_cf, cost_matrix_w2, numItermax=int(1e6)))

            
            from ..utils import sample_counterfactual_distribution
            disco_dist = sample_counterfactual_distribution(controls_data, weights, grid=p_res.target.grid, num_samples=eval_size)
            target_dist = target_data[np.random.choice(len(target_data), size=eval_size)]

            if disco_res.params.is_multivariate:
                def _compute_1d_metrics(t_1d, d_1d):
                    ks_val, _ = ks_2samp(t_1d, d_1d)
                    return ks_val
                    
                t_2d = target_dist.reshape(eval_size, -1)
                d_2d = disco_dist.reshape(eval_size, -1)
                
                marginals = [_compute_1d_metrics(t_2d[:, dim], d_2d[:, dim]) for dim in range(t_2d.shape[1])]
                ks_stats = [m for m in marginals]

                target_means = np.mean(target_data, axis=0)
                disco_means = np.average(controls_all, axis=0, weights=weights_cf)

                mean_diffs = [float(target_means[dim] - disco_means[dim]) for dim in range(len(target_means))]

                cov_t = np.cov(target_data, rowvar=False)
                cov_d = np.cov(controls_all, aweights=weights_cf, rowvar=False)
                cov_error = float(np.linalg.norm(cov_t - cov_d, ord='fro'))
                
            else:
                # 1D Deterministic metrics (Supports simplex=False for these)
                target_q = p_res.target.quantiles
                disco_q = p_res.DiSCo.quantile
                
                w1 = float(np.mean(np.abs(target_q - disco_q)))
                w2 = float(np.mean((target_q - disco_q)**2))
                ks_stats = [float(np.max(np.abs(p_res.target.cdf - p_res.DiSCo.cdf)))]
                mean_diffs = [float(np.abs(np.mean(target_q) - np.mean(disco_q)))]
                
                # 1D approximate metrics from samples
                cov_t = np.var(target_data)
                cov_d = np.var(controls_all, axis=0, aweights=weights_cf)
                cov_error = float(np.abs(cov_t - cov_d))
        else:
            w1, w2, energy_divergence, cov_error = np.nan, np.nan, np.nan, np.nan
            ks_stats, mean_diffs = [np.nan], [np.nan]
            
        per_period_metrics[t] = {
            "w1": float(w1), "w2": float(w2), "energy_divergence": float(energy_divergence), 
            "ks_stat": ks_stats, "mean_diff": mean_diffs, 
            "cov_error": float(cov_error)
        }

    num_dims = len(list(per_period_metrics.values())[0]["ks_stat"]) if per_period_metrics else 0

    return PreTreatmentFitMetrics(
        w1=per_period_metrics.get(t0_idx, {}).get("w1", np.nan),
        w2=per_period_metrics.get(t0_idx, {}).get("w2", np.nan),
        energy_divergence=per_period_metrics.get(t0_idx, {}).get("energy_divergence", np.nan),
        cov_error=per_period_metrics.get(t0_idx, {}).get("cov_error", np.nan),
        marginal_ks=[float(np.mean([per_period_metrics[t]["ks_stat"][dim] for t in per_period_metrics])) for dim in range(num_dims)],
        marginal_mean_diff=[float(np.mean([per_period_metrics[t]["mean_diff"][dim] for t in per_period_metrics])) for dim in range(num_dims)],
        metrics_per_period=per_period_metrics
    )

def calculate_ground_truth_effect(df: pd.DataFrame, target_id: str = '0', time_col: str = 'time_col', y_cols: list = None, 
                                    cf_cols: list = None, quant_list: list = [0.0, 0.25, 0.5, 0.75, 1.0],
                                     q_labels: list = ['Q1', 'Q2', 'Q3', 'Q4'], calc_wasserstein: bool = True, calc_transport_map: bool = False) -> dict:
    """
    Berechnet den wahren Treatment-Effekt (Ground Truth) für die Target-Unit.
    Gibt die W1-Distanz und den Durchschnittseffekt pro Periode zurück.
    """
    if y_cols is None or cf_cols is None:
        raise ValueError("y_cols und cf_cols müssen angegeben werden.")
    
    # Sicherstellen, dass id_col als String behandelt wird, falls target_id ein String ist
    target_df = df[df['id_col'].astype(str) == str(target_id)]
    periods = sorted(target_df[time_col].unique())
    
    gt_effects = {}
    
    for t in periods:
        data_t = target_df[target_df[time_col] == t]
        
        y_treated = np.asarray(data_t[y_cols])
        y_cf = np.asarray(data_t[cf_cols])
        
        if len(y_treated) == 0:
            continue
        
        num_dims = y_treated.shape[1]
        
        if calc_wasserstein or calc_transport_map:
            N = len(y_treated)
            weights = np.ones(N) / N
            
            # W1 Distanz
            cost_matrix = ot.dist(y_treated, y_cf, metric='euclidean')
            w1 = float(ot.emd2(weights, weights, cost_matrix))
            
            # W2 Distanz
            cost_matrix_w2 = ot.dist(y_treated, y_cf, metric='sqeuclidean')
            w2 = float(ot.emd2(weights, weights, cost_matrix_w2))
            
            # Energy Divergence
            diff_cross = y_treated[:, None, :] - y_cf[None, :, :]
            dist_cross = np.mean(np.linalg.norm(diff_cross, axis=-1))
            
            diff_treated = y_treated[:, None, :] - y_treated[None, :, :]
            dist_treated = np.mean(np.linalg.norm(diff_treated, axis=-1))
            
            diff_cf = y_cf[:, None, :] - y_cf[None, :, :]
            dist_cf = np.mean(np.linalg.norm(diff_cf, axis=-1))
            
            energy_div = float(dist_cross - 0.5 * dist_treated - 0.5 * dist_cf)
        else:
            w1, w2, energy_div = np.nan, np.nan, np.nan

        if calc_transport_map:
            # Transport Map (Exact)
            T_samples = ot.emd(weights, weights, cost_matrix, numItermax=int(1e7))
            
            df_target = pd.DataFrame(y_treated)
            df_cf = pd.DataFrame(y_cf)

            def assign_quantiles(series):
                return pd.qcut(series, q=quant_list, labels=q_labels, duplicates='drop')

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
        else:
            df_T_agg = None
        
        target_means = np.mean(y_treated, axis=0)
        disco_means = np.mean(y_cf, axis=0)
        mean_diffs = target_means - disco_means
        
        if num_dims > 1:
            cov_t = np.cov(y_treated, rowvar=False)
            cov_d = np.cov(y_cf, rowvar=False)
            cov_diff = cov_t - cov_d
        else:
            cov_t = float(np.var(y_treated))
            cov_d = float(np.var(y_cf))
            cov_diff = np.array([[cov_t - cov_d]])

        gt_effects[t] = {
            'w1': w1,
            'w2': w2,
            'energy_divergence': energy_div,
            'mean_diff': mean_diffs.tolist(),
            'cov_diff': cov_diff.tolist(),
            'transport_map': df_T_agg,
            'Mean Diff (Norm)': float(np.linalg.norm(mean_diffs)),
            'Cov Diff (Frobenius)': float(np.linalg.norm(cov_diff, ord='fro'))
        }
        
    return gt_effects
