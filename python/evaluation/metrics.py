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
    
    for t in periods: # only pre-treatment
        if t > t0_idx or t not in results_by_period:
            continue
            
        p_res = results_by_period[t]
        
        target_data = np.asarray(p_res.target.data)
        controls_data = p_res.controls.data
        weights = disco_res.weights
        
        if len(target_data) > 0 and len(controls_data) > 0 and weights is not None:
            
            # Sampling for both 1D and Multi-D to calculate metrics like cov_error and energy_divergence
            target_dist = target_data[np.random.choice(len(target_data), size=eval_size)]
            
            from ..utils import sample_counterfactual_distribution
            disco_dist = sample_counterfactual_distribution(controls_data, weights, grid=p_res.target.grid, num_samples=eval_size)
            
            if disco_res.params.is_multivariate:
                def _compute_1d_metrics(t_1d, d_1d):
                    ks_val, _ = ks_2samp(t_1d, d_1d)
                    mean_df = np.abs(np.mean(t_1d) - np.mean(d_1d))
                    var_err = np.abs(np.var(t_1d) - np.var(d_1d))
                    return ks_val, mean_df, var_err
                    
                t_2d = target_dist.reshape(eval_size, -1)
                d_2d = disco_dist.reshape(eval_size, -1)
                
                marginals = [_compute_1d_metrics(t_2d[:, dim], d_2d[:, dim]) for dim in range(t_2d.shape[1])]
                ks_stats = [m[0] for m in marginals]
                mean_diffs = [m[1] for m in marginals]
                
                cov_t = np.cov(target_dist, rowvar=False)
                cov_d = np.cov(disco_dist, rowvar=False)
                cov_error = float(np.linalg.norm(cov_t - cov_d, ord='fro'))
                
                a, b = np.ones(eval_size) / eval_size, np.ones(eval_size) / eval_size
                M = ot.dist(target_dist, disco_dist, metric='euclidean')
                w1 = float(ot.emd2(a, b, M))
                energy_divergence = float(np.mean(sr.energy_score(target_dist, disco_dist[:,None,:])))
                
            else:
                # 1D Deterministic metrics (Supports simplex=False for these)
                target_q = p_res.target.quantiles
                disco_q = p_res.DiSCo.quantile
                
                w1 = float(np.mean(np.abs(target_q - disco_q)))
                ks_stats = [float(np.max(np.abs(p_res.target.cdf - p_res.DiSCo.cdf)))]
                mean_diffs = [float(np.abs(np.mean(target_q) - np.mean(disco_q)))]
                
                # 1D approximate metrics from samples
                cov_t = np.var(target_dist)
                cov_d = np.var(disco_dist)
                cov_error = float(np.abs(cov_t - cov_d))
                energy_divergence = float(np.mean(sr.energy_score(target_dist[:,None], disco_dist[:, None, None])))
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
