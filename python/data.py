import numpy as np
import pandas as pd

import math
from scipy.stats import norm

#creation of contoll variables for testing
def get_contunious_data(sample_size, num_controls, target_offset=0, dist_control=3, dist_target=4, dim=1, base_corr=0.8, corr_drift=0.05):
    data = []
    
    # 1. Systematische Basis-Korrelationsmatrix
    base_corr_mat = np.full((dim, dim), base_corr)
    np.fill_diagonal(base_corr_mat, 1.0)
    
    def draw_mix(n, n_comp, m, covs):
        c = np.random.choice(n_comp, size=n)
        out = np.zeros((n, dim))
        for comp in range(n_comp):
            mask = (c == comp)
            n_samples = mask.sum()
            if n_samples > 0:
                out[mask] = np.random.multivariate_normal(m[comp], covs[comp], size=n_samples)
        return out
        
    def generate_params(n_comp, is_target=False):
        offset = target_offset if is_target else 0
        
        # Basis-Zentrum der Unit uniform ziehen
        unit_base = np.random.uniform(-5 + offset, 5 + offset, dim)
        
        # 1. Means der Sub-Komponenten generieren
        if dim > 1:
            z = np.random.multivariate_normal(np.zeros(dim), base_corr_mat, size=n_comp)
            u = norm.cdf(z)  # Uniform zwischen 0 und 1
            # Sub-Komponenten gruppieren sich engmaschig um das Basis-Zentrum
            means = unit_base + (-2.0 + 4.0 * u)
        else:
            means = unit_base + np.random.uniform(-2, 2, (n_comp, dim))
        
        covs = np.zeros((n_comp, dim, dim))
        for i in range(n_comp):
            if dim > 1:
                # Leicht verbogene Korrelationsmatrix für Varianz innerhalb der Regionen
                noise = np.random.uniform(-corr_drift, corr_drift, (dim, dim))
                noise = (noise + noise.T) / 2
                corr_mat = base_corr_mat + noise
                np.fill_diagonal(corr_mat, 1.0)
                
                # Sichern, dass Matrix positiv semi-definit ist
                vals, vecs = np.linalg.eigh(corr_mat)
                vals = np.maximum(vals, 1e-4)
                corr_mat = vecs @ np.diag(vals) @ vecs.T
                
                # Diagonale wieder auf 1 normieren
                d = np.sqrt(np.diag(corr_mat))
                corr_mat = corr_mat / np.outer(d, d)
            else:
                corr_mat = np.array([[1.0]])
            
            # Varianzen hier deutlich breiter machen, damit die Punktewolke zusammenhängt
            # und die Korrelation nicht von der zufälligen Lage der 3 Means zerschossen wird
            stds = np.random.uniform(2.0, 4.5, dim)
            S = np.diag(stds)
            
            covs[i] = S @ corr_mat @ S 
        return means, covs

    def append_data(unit_id, time_val, samples):
        for y in samples:
            row = {'id_col': unit_id, 'time_col': time_val}
            if dim == 1:
                row['y_col'] = y[0]
            else:
                for d in range(dim):
                    row[f'y_col_{d+1}'] = y[d]
            data.append(row)
            
    # Target 
    means_t, covs_t = generate_params(dist_target, is_target=True)
    target_pre = draw_mix(sample_size, dist_target, means_t, covs_t)
    target_post = draw_mix(sample_size, dist_target, means_t, covs_t)

    append_data('0', 9998, target_pre)
    append_data('0', 9999, target_post)
    
    # Controls
    for i in range(num_controls):
        means_c, covs_c = generate_params(dist_control, is_target=False)
        c_pre = draw_mix(sample_size, dist_control, means_c, covs_c)
        c_post = draw_mix(sample_size, dist_control, means_c, covs_c)
        
        append_data(str(i+1), 9998, c_pre)
        append_data(str(i+1), 9999, c_post)

    return pd.DataFrame(data)


def get_discrete_data(sample_size, num_controls, dist_control=3, dist_target=4):
    data = []
    
    def draw_mix_binom(n, n_comp, n_trials, p_probs):
        c = np.random.choice(n_comp, size=n)
        return np.random.binomial(n=n_trials[c], p=p_probs[c])
        
    # Target (Mix aus dist_target Binomialverteilungen)
    # Statt Means und Variances nehmen wir Anzahlen (n) und Wahrscheinlichkeiten (p)
    n_trials_t = np.random.randint(1, 20, dist_target)
    p_probs_t = np.random.uniform(0.1, 0.9, dist_target)
    
    target_pre = draw_mix_binom(sample_size, dist_target, n_trials_t, p_probs_t)
    target_post = draw_mix_binom(sample_size, dist_target, n_trials_t, p_probs_t)

    data.extend([{'id_col': '0', 'time_col': 9998, 'y_col': y} for y in target_pre])
    data.extend([{'id_col': '0', 'time_col': 9999, 'y_col': y} for y in target_post]) # post-treatment
    
    # Controls (jede Unit hat ihren eigenen Binomial-Mix)
    for i in range(num_controls):
        n_trials_c = np.random.randint(1, 20, dist_control)
        p_probs_c = np.random.uniform(0.1, 0.9, dist_control)
        
        c_pre = draw_mix_binom(sample_size, dist_control, n_trials_c, p_probs_c)
        c_post = draw_mix_binom(sample_size, dist_control, n_trials_c, p_probs_c)
        
        data.extend([{'id_col': str(i+1), 'time_col': 9998, 'y_col': y} for y in c_pre])
        data.extend([{'id_col': str(i+1), 'time_col': 9999, 'y_col': y} for y in c_post])

    return pd.DataFrame(data)