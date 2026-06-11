import numpy as np
from ..models import  PreTreatmentFitMetrics, DiSCoResult
from scipy.stats import ks_2samp
import ot
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
                    if k == j:
                        D[j, k] = 0
                    else:
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
            N = len(target_data)
            weights_target = np.ones(N) / N
            weights_cf = np.array([weights[i] / len(controls_data[i]) for i in range(len(controls_data)) 
                                   for _ in range(len(controls_data[i]))])

            # Adjust for numerical instabilities
            negative_weigths = np.where(weights_cf < 0.0)
            weights_cf[negative_weigths] = 0

            check = np.sum(weights_cf)
            if np.abs(1.0 - check) > 1e-6:
                weights_cf = weights_cf * 1.0/ check


            target_std = np.std(target_data, axis=0)
            target_std[target_std == 0] = 1.0
            target_scaled = target_data / target_std
            cf_scaled = controls_all / target_std

            cost_matrix = ot.dist(target_data, controls_all, metric='euclidean')
            w1 = float(ot.emd2(weights_target, weights_cf, cost_matrix))

            
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
                ks_stats = [float(np.max(np.abs(p_res.target.cdf - p_res.DiSCo.cdf)))]
                mean_diffs = [float(np.abs(np.mean(target_q) - np.mean(disco_q)))]
                
                # 1D approximate metrics from samples
                cov_t = np.var(target_data)
                cov_d = np.var(controls_all, axis=0, aweights=weights_cf)
                cov_error = float(np.abs(cov_t - cov_d))
        else:
            w1, energy_divergence, cov_error = np.nan, np.nan, np.nan
            ks_stats, mean_diffs = [np.nan], [np.nan]
            
        per_period_metrics[t] = {
            "w1": float(w1), "energy_divergence": float(energy_divergence), 
            "ks_stat": ks_stats, "mean_diff": mean_diffs, 
            "cov_error": float(cov_error)
        }

    num_dims = len(list(per_period_metrics.values())[0]["ks_stat"]) if per_period_metrics else 0

    return PreTreatmentFitMetrics(
        w1=per_period_metrics.get(t0_idx, {}).get("w1", np.nan),
        energy_divergence=per_period_metrics.get(t0_idx, {}).get("energy_divergence", np.nan),
        cov_error=per_period_metrics.get(t0_idx, {}).get("cov_error", np.nan),
        marginal_ks=[float(np.mean([per_period_metrics[t]["ks_stat"][dim] for t in per_period_metrics])) for dim in range(num_dims)],
        marginal_mean_diff=[float(np.mean([per_period_metrics[t]["mean_diff"][dim] for t in per_period_metrics])) for dim in range(num_dims)],
        metrics_per_period=per_period_metrics
    )
