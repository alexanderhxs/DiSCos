import matplotlib.pyplot as plt
import numpy as np

def _get_pooled_data(fit_synth, period):
    periods_list = period if isinstance(period, list) else [period]
    
    first_p = periods_list[0]
    first_res = fit_synth.results_periods[first_p]
    weights = first_res.DiSCo.weights if first_res.DiSCo.weights is not None else fit_synth.weights
    
    pooled_target = []
    pooled_controls = [[] for _ in range(len(weights))]
    pooled_samples = []
    
    for p in periods_list:
        p_res = fit_synth.results_periods[p]
        
        t_data = p_res.target.data
        if t_data.ndim == 1:
            t_data = t_data.reshape(-1, 1)
        pooled_target.append(t_data)
        
        for i, c_data in enumerate(p_res.controls.data):
            if c_data.ndim == 1:
                c_data = c_data.reshape(-1, 1)
            pooled_controls[i].append(c_data)
            
        if getattr(p_res.DiSCo, 'samples', None) is not None:
            s_data = p_res.DiSCo.samples
            if s_data.ndim == 1:
                s_data = s_data.reshape(-1, 1)
            pooled_samples.append(s_data)
            
    final_target = np.vstack(pooled_target)
    final_controls = [np.vstack(c_list) if len(c_list) > 0 and sum(len(c) for c in c_list) > 0 else np.array([]) for c_list in pooled_controls]
    final_samples = np.vstack(pooled_samples) if len(pooled_samples) > 0 else None
    
    return final_target, final_controls, weights, final_samples


def plot_fit_quantiles(fit_synth, show_controls=False, period=None):    
    periods = sorted(list(fit_synth.results_periods.keys()))
    if period is None:
        period = periods[-1]
        
    target_data, controls_data, weights, final_samples = _get_pooled_data(fit_synth, period)
    is_multi = fit_synth.params.is_multivariate
    
    if is_multi or isinstance(period, list):
        dim = target_data.shape[1]
        fig, axes = plt.subplots(1, dim, figsize=(6 * dim, 5))
        if dim == 1: axes = [axes]
            
        for d, ax in enumerate(axes):
            t_data = target_data[:, d]
            c_data_list = [c[:, d] for c in controls_data]
            
            grid_min = min(t_data.min(), np.min([c.min() for c in c_data_list if len(c) > 0]))
            grid_max = max(t_data.max(), np.max([c.max() for c in c_data_list if len(c) > 0]))
            grid = np.linspace(grid_min, grid_max, 2000)
            
            w_sum = np.sum(weights)
            norm_weights = weights / w_sum if w_sum > 1e-8 else weights
            
            if final_samples is not None:
                synth_cdf = np.mean(final_samples[:, d][:, None] <= grid, axis=0)
            else:
                synth_cdf = np.zeros_like(grid)
                for c_data, w in zip(c_data_list, norm_weights):
                    if w > 1e-5 and len(c_data) > 0:
                        synth_cdf += w * np.mean(c_data[:, None] <= grid, axis=0)
                    
            q_grid = np.linspace(0, 1, 200)
            target_quantiles = np.quantile(t_data, q_grid)
            
            synth_quantiles = np.array([grid[np.searchsorted(synth_cdf, q)] if q <= synth_cdf[-1] else grid[-1] for q in q_grid])
            
            ax.plot(q_grid, target_quantiles, color='black', linewidth=3, label='Target')
            ax.plot(q_grid, synth_quantiles, color='red', linewidth=3, label='DSC')
            
            if hasattr(fit_synth, 'CI') and fit_synth.CI is not None and getattr(fit_synth.CI.bootmat, 'weights', None) is not None:
                n_boots = fit_synth.CI.bootmat.weights.shape[1]
                boot_synth_quantiles = np.zeros((n_boots, len(q_grid)))
                
                for b in range(n_boots):
                    b_weights = fit_synth.CI.bootmat.weights[:, b]
                    b_w_sum = np.sum(b_weights)
                    b_norm_weights = b_weights / b_w_sum if b_w_sum > 1e-8 else b_weights
                    
                    b_synth_cdf = np.zeros_like(grid)
                    for c_data, w in zip(c_data_list, b_norm_weights):
                        if w > 1e-5 and len(c_data) > 0:
                            b_synth_cdf += w * np.mean(c_data[:, None] <= grid, axis=0)
                            
                    boot_synth_quantiles[b, :] = np.array([grid[np.searchsorted(b_synth_cdf, q)] if q <= b_synth_cdf[-1] else grid[-1] for q in q_grid])
                
                alpha = 1 - fit_synth.params.cl
                lower = np.quantile(boot_synth_quantiles, alpha / 2, axis=0)
                upper = np.quantile(boot_synth_quantiles, 1 - (alpha / 2), axis=0)
                ax.plot(q_grid, lower, color='red', linewidth=1, linestyle='--', label='CI (DSC)')
                ax.plot(q_grid, upper, color='red', linewidth=1, linestyle='--')

            if show_controls:
                for i, c_data in enumerate(c_data_list):
                    if len(c_data) > 0:
                        c_quant = np.quantile(c_data, q_grid)
                        ax.plot(q_grid, c_quant, color='grey', linewidth=1, linestyle='--', label=f'Controls' if i == 0 else None)
                        
            ax.set_xlim(-0.02, 1.02)
            ax.set_xlabel('x', fontsize=14)
            ax.set_ylabel('$F^{-1}(x)$', fontsize=14)
            ax.set_title(f"Marginal Quantiles: Dim {d+1}")
            ax.legend(loc='lower right', frameon=True, edgecolor='black', framealpha=1, borderpad=1, fontsize=12)
            
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.grid(linestyle='--', alpha=0.5)
            
        period_label = f"Perioden {period}" if isinstance(period, list) else f"Periode {period}"
        plt.suptitle(f'Quantiles Plot ({period_label})', fontsize=16)
        plt.tight_layout()
        plt.show()
        return

    # 1D fallback
    period_res = fit_synth.results_periods[period]
    x_grid = fit_synth.evgrid
    sort_idx = np.argsort(x_grid)
    x_grid = x_grid[sort_idx]
    
    target_quantiles = period_res.target.quantiles[sort_idx]
    disco_quantiles = period_res.DiSCo.quantile[sort_idx]

    plt.figure(figsize=(6, 5))
    plt.plot(x_grid, target_quantiles, color='black', linewidth=3, label='Target')
    plt.plot(x_grid, disco_quantiles, color='red', linewidth=3, label='DSC')
    
    if hasattr(fit_synth, 'CI') and fit_synth.CI is not None and hasattr(fit_synth.CI, 'quantile') and fit_synth.CI.quantile is not None:
        period_idx = periods.index(period)
        lower = fit_synth.CI.quantile.lower[:, period_idx][sort_idx]
        upper = fit_synth.CI.quantile.upper[:, period_idx][sort_idx]
        plt.plot(x_grid, lower, color='red', linewidth=1, linestyle='--', label='CI (DSC)')
        plt.plot(x_grid, upper, color='red', linewidth=1, linestyle='--')

    if show_controls:
        for i in range(period_res.controls.quantiles.shape[1]):
            control_quantiles = period_res.controls.quantiles[:,i][sort_idx]
            plt.plot(x_grid, control_quantiles, color='grey', linewidth=1,linestyle = '--' , label=f'Controls' if i == 0 else None)

    plt.xlim(-0.02, 1.02)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('$F^{-1}(x)$', fontsize=14)
    plt.legend(loc='lower right', frameon=True, edgecolor='black', framealpha=1, borderpad=1, fontsize=12)

    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.grid(linestyle='--', alpha=0.5)

    plt.show()

def plot_fit_cdf(fit_synth, show_controls=False, period=None):    
    periods = sorted(list(fit_synth.results_periods.keys()))
    if period is None:
        period = periods[-1]
        
    target_data, controls_data, weights, final_samples = _get_pooled_data(fit_synth, period)
    is_multi = fit_synth.params.is_multivariate
    
    if is_multi or isinstance(period, list):
        dim = target_data.shape[1]
        fig, axes = plt.subplots(1, dim, figsize=(6 * dim, 5))
        if dim == 1: axes = [axes]
            
        for d, ax in enumerate(axes):
            t_data = target_data[:, d]
            c_data_list = [c[:, d] for c in controls_data]
            
            grid_min = min(t_data.min(), np.min([c.min() for c in c_data_list if len(c) > 0]))
            grid_max = max(t_data.max(), np.max([c.max() for c in c_data_list if len(c) > 0]))
            grid = np.linspace(grid_min, grid_max, 2000)
            
            w_sum = np.sum(weights)
            norm_weights = weights / w_sum if w_sum > 1e-8 else weights
            
            target_cdf = np.mean(t_data[:, None] <= grid, axis=0)
            if final_samples is not None:
                synth_cdf = np.mean(final_samples[:, d][:, None] <= grid, axis=0)
            else:
                synth_cdf = np.zeros_like(grid)
                for c_data, w in zip(c_data_list, norm_weights):
                    if w > 1e-5 and len(c_data) > 0:
                        synth_cdf += w * np.mean(c_data[:, None] <= grid, axis=0)
                    
            ax.plot(grid, target_cdf, label="Target", color="black", linewidth=3)
            ax.plot(grid, synth_cdf, label="DSC", color="red", linewidth=3)
            
            if hasattr(fit_synth, 'CI') and fit_synth.CI is not None and getattr(fit_synth.CI.bootmat, 'weights', None) is not None:
                n_boots = fit_synth.CI.bootmat.weights.shape[1]
                boot_synth_cdf = np.zeros((n_boots, len(grid)))
                
                for b in range(n_boots):
                    b_weights = fit_synth.CI.bootmat.weights[:, b]
                    b_w_sum = np.sum(b_weights)
                    b_norm_weights = b_weights / b_w_sum if b_w_sum > 1e-8 else b_weights
                    
                    b_cdf = np.zeros_like(grid)
                    for c_data, w in zip(c_data_list, b_norm_weights):
                        if w > 1e-5 and len(c_data) > 0:
                            b_cdf += w * np.mean(c_data[:, None] <= grid, axis=0)
                    boot_synth_cdf[b, :] = b_cdf
                
                alpha = 1 - fit_synth.params.cl
                lower = np.quantile(boot_synth_cdf, alpha / 2, axis=0)
                upper = np.quantile(boot_synth_cdf, 1 - (alpha / 2), axis=0)
                ax.plot(grid, lower, color='red', linewidth=1, linestyle='--', label='CI (DSC)')
                ax.plot(grid, upper, color='red', linewidth=1, linestyle='--')

            if show_controls:
                for i, c_data in enumerate(c_data_list):
                    if len(c_data) > 0:
                        c_cdf = np.mean(c_data[:, None] <= grid, axis=0)
                        ax.plot(grid, c_cdf, color='grey', linewidth=1, linestyle='--', label=f'Controls' if i == 0 else None)
                        
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel('$y$', fontsize=14)
            ax.set_ylabel('$F(y)$', fontsize=14)
            ax.set_title(f"Marginal CDF: Dim {d+1}")
            ax.legend(loc='lower right', frameon=True, edgecolor='black', framealpha=1, borderpad=1, fontsize=12)
            
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.grid(linestyle='--', alpha=0.5)
            
        period_label = f"Perioden {period}" if isinstance(period, list) else f"Periode {period}"
        plt.suptitle(f'CDF Plot ({period_label})', fontsize=16)
        plt.tight_layout()
        plt.show()
        return
        
    period_res = fit_synth.results_periods[period]
    x_grid = period_res.target.grid
    sort_idx = np.argsort(x_grid)
    x_grid = x_grid[sort_idx]

    target_cdf = period_res.target.cdf[sort_idx]
    disco_cdf = period_res.DiSCo.cdf[sort_idx]

    plt.figure(figsize=(6, 5))
    plt.plot(x_grid, target_cdf, color='black', linewidth=3, label='Target')
    plt.plot(x_grid, disco_cdf, color='red', linewidth=3, label='DSC')
    
    if hasattr(fit_synth, 'CI') and fit_synth.CI is not None and hasattr(fit_synth.CI, 'cdf') and fit_synth.CI.cdf is not None:
        period_idx = periods.index(period)
        lower = fit_synth.CI.cdf.lower[:, period_idx][sort_idx]
        upper = fit_synth.CI.cdf.upper[:, period_idx][sort_idx]
        plt.plot(x_grid, lower, color='red', linewidth=1, linestyle='--', label='CI (DSC)')
        plt.plot(x_grid, upper, color='red', linewidth=1, linestyle='--')

    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.grid(linestyle='--', alpha=0.5)

    plt.show()

def plot_fit_copula(fit_synth, period=None):
    """
    Erstellt einen 2D Copula Scatter-Plot zwischen Target und DSC.
    (Geht von 2 Dimensionen aus). Zeigt die reine, isolierte Abhängigkeitsstruktur (Ränge).
    """
    from scipy.stats import rankdata
    
    periods = sorted(list(fit_synth.results_periods.keys()))
    if period is None:
        period = periods[-1]
        
    target_data, controls_data, weights, final_samples = _get_pooled_data(fit_synth, period)
    is_multi = fit_synth.params.is_multivariate
    
    if target_data.shape[1] != 2:
        print("Joint Plot wird nur für 2D Daten unterstützt.")
        return
        
    if len(controls_data) == 0:
        return
    
    N = len(target_data)
    u_target = rankdata(target_data[:, 0]) / N
    v_target = rankdata(target_data[:, 1]) / N
    
    from ..utils import sample_counterfactual_distribution
    first_p = period[0] if isinstance(period, list) else period
    grid = fit_synth.results_periods[first_p].target.grid
    
    if final_samples is not None:
        disco_dist = final_samples
    else:
        w_sum = np.sum(weights)
        norm_weights = weights / w_sum if w_sum > 1e-8 else weights
        disco_dist = sample_counterfactual_distribution(controls_data, norm_weights, grid=grid, num_samples=N)
    
    if disco_dist is None or len(disco_dist) == 0:
        return
        
    u_dsc = np.empty(len(disco_dist))
    v_dsc = np.empty(len(disco_dist))
    
    sort_idx_x = np.argsort(disco_dist[:, 0])
    u_dsc[sort_idx_x] = (np.arange(len(disco_dist)) + 1) / len(disco_dist)
    
    sort_idx_y = np.argsort(disco_dist[:, 1])
    v_dsc[sort_idx_y] = (np.arange(len(disco_dist)) + 1) / len(disco_dist)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].scatter(u_target, v_target, alpha=0.5, s=20, c='black')
    axes[0].set_title('Target Copula', fontsize=14)
    axes[0].set_xlabel('Rank Dim 1 ($F_1$)', fontsize=12)
    axes[0].set_ylabel('Rank Dim 2 ($F_2$)', fontsize=12)
    
    axes[1].scatter(u_dsc, v_dsc, alpha=0.5, s=20, c='red')
    axes[1].set_title('DSC Copula (Sampled)', fontsize=14)
    axes[1].set_xlabel('Rank Dim 1 ($F_1$)', fontsize=12)
    
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.grid(linestyle='--', alpha=0.3)
        
    period_label = f"Perioden {period}" if isinstance(period, list) else f"Periode {period}"
    plt.suptitle(f'Copula Comparison ({period_label})', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_fit_joint_contour(fit_synth, period=None):
    """
    Erstellt einen 2D Contour Overlay Plot der Joint Density (KDE) zwischen Target und DSC.
    (Geht von 2 Dimensionen aus). Target und DSC werden übereinander gelegt.
    """
    from scipy.stats import gaussian_kde
    import matplotlib.lines as mlines
    
    periods = sorted(list(fit_synth.results_periods.keys()))
    if period is None:
        period = periods[-1]
        
    target_data, controls_data, weights, final_samples = _get_pooled_data(fit_synth, period)
    is_multi = fit_synth.params.is_multivariate
    
    if target_data.shape[1] != 2:
        print("Joint Contour Plot wird nur für 2D Daten unterstützt.")
        return
        
    if len(controls_data) == 0:
        return
    
    x_t = target_data[:, 0]
    y_t = target_data[:, 1]
    
    if final_samples is not None:
        pool_x = final_samples[:, 0]
        pool_y = final_samples[:, 1]
        pool_w = np.ones(len(final_samples)) / len(final_samples)
    else:
        pool_x = []
        pool_y = []
        pool_w = []
        
        w_sum = np.sum(weights)
        norm_weights = weights / w_sum if w_sum > 1e-8 else weights
        
        for c_data, w in zip(controls_data, norm_weights):
            if len(c_data) > 0 and w > 1e-6:
                pool_x.extend(c_data[:, 0])
                pool_y.extend(c_data[:, 1])
                pool_w.extend([w / len(c_data)] * len(c_data))
                
        pool_x = np.array(pool_x)
        pool_y = np.array(pool_y)
        pool_w = np.array(pool_w)
    
    x_min = min(x_t.min(), pool_x.min())
    x_max = max(x_t.max(), pool_x.max())
    y_min = min(y_t.min(), pool_y.min())
    y_max = max(y_t.max(), pool_y.max())
    
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    
    X, Y = np.mgrid[x_min-x_pad:x_max+x_pad:100j, y_min-y_pad:y_max+y_pad:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    try:
        kde_target = gaussian_kde(np.vstack([x_t, y_t]))
        Z_target = np.reshape(kde_target(positions).T, X.shape)
    except np.linalg.LinAlgError:
        print("LinAlgError beim Berechnen der Target KDE (evtl. Datenpunkte zu dicht).")
        return
        
    try:
        kde_dsc = gaussian_kde(np.vstack([pool_x, pool_y]), weights=pool_w)
        Z_dsc = np.reshape(kde_dsc(positions).T, X.shape)
    except np.linalg.LinAlgError:
        print("LinAlgError beim Berechnen der DSC KDE (evtl. Punkte zu dicht).")
        return
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    contour_t = ax.contour(X, Y, Z_target, levels=5, colors='black', linewidths=2, alpha=0.8)
    contour_d = ax.contour(X, Y, Z_dsc, levels=5, colors='red', linewidths=2, alpha=0.8)
    
    legend_target = mlines.Line2D([], [], color='black', linewidth=2, label='Target')
    legend_dsc = mlines.Line2D([], [], color='red', linewidth=2, label='DSC')
    ax.legend(handles=[legend_target, legend_dsc], loc='best', frameon=True, edgecolor='black', fontsize=12)
    
    period_label = f"Perioden {period}" if isinstance(period, list) else f"Periode {period}"
    ax.set_title(f'Joint Density Contour ({period_label})', fontsize=14)
    ax.set_xlabel('Dim 1', fontsize=12)
    ax.set_ylabel('Dim 2', fontsize=12)
    ax.grid(linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    plt.tight_layout()
    plt.show()

def plot_fit_density(fit_synth, period=None, var_names = None):
    """
    Erstellt ein überlappendes Histogramm / Dichteplot von Target und DSC.
    Unterstützt jetzt auch das Plotten mehrerer Perioden gleichzeitig als gemeinsame Verteilung.
    """
    periods = sorted(list(fit_synth.results_periods.keys()))
    if period is None:
        period = periods[-1]
        
    target_data, controls_data, weights, final_samples = _get_pooled_data(fit_synth, period)
    is_multi = fit_synth.params.is_multivariate
    
    if is_multi or isinstance(period, list):
        dim = target_data.shape[1]
        fig, axes = plt.subplots(1, dim, figsize=(6 * dim, 5))
        if dim == 1: axes = [axes]
            
        for d, ax in enumerate(axes):
            t_data = target_data[:, d]
            c_data_list = [c[:, d] for c in controls_data]
            
            if final_samples is not None:
                flat_controls = final_samples[:, d]
                flat_weights = np.ones(len(flat_controls)) / len(flat_controls)
            else:
                flat_controls = []
                flat_weights = []
                for w, c in zip(weights, c_data_list):
                    if len(c) > 0 and w > 1e-6:
                        flat_controls.extend(c)
                        flat_weights.extend([w / len(c)] * len(c))
                    
            ax.hist(t_data, bins=30, density=True, alpha=0.5, color='black', label='Target')
            if len(flat_controls) > 0:
                ax.hist(flat_controls, bins=30, density=True, weights=flat_weights, alpha=0.5, color='red', label='DSC')
            
            if var_names is not None:
                ax.set_title(f'{var_names[d]}')
            else:
                ax.set_title(f'Dimension {d+1}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend(loc='upper right', frameon=True, edgecolor='black', framealpha=1)
            
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.grid(linestyle='--', alpha=0.5)
            
        period_label = f"Perioden {period}" if isinstance(period, list) else f"Periode {period}"
        plt.suptitle(f'Density Plot ({period_label})', fontsize=16)
        plt.tight_layout()
        plt.show()
    else:
        period_res = fit_synth.results_periods[period]
        target_data = period_res.target.data
        controls_data = period_res.controls.data
        
        if final_samples is not None:
            flat_controls = final_samples[:, 0] if final_samples.ndim > 1 else final_samples
            flat_weights = np.ones(len(flat_controls)) / len(flat_controls)
        else:
            flat_controls = []
            flat_weights = []
            for w, c in zip(weights, controls_data):
                if len(c) > 0 and w > 1e-6:
                    flat_controls.extend(c)
                    flat_weights.extend([w / len(c)] * len(c))
                
        plt.figure(figsize=(8, 6))
        plt.hist(target_data, bins=30, density=True, alpha=0.5, color='black', label='Target')
        if len(flat_controls) > 0:
            plt.hist(flat_controls, bins=30, density=True, weights=flat_weights, alpha=0.5, color='red', label='DSC')
            
        plt.title(f'Density Plot (Period {period})', fontsize=16)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend(loc='upper right', frameon=True, edgecolor='black', framealpha=1)
        
        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['right'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(True)
        plt.gca().spines['left'].set_visible(True)
        plt.grid(linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()

def plot_fit_scatter2d(fit_synth, period=None):
    """
    Erstellt einen klassischen 2D Scatterplot der Originaldaten.
    Target-Datenpunkte werden schwarz gezeichnet.
    Die Punkte der DSC (gemischte Controls) werden rot gezeichnet und
    ihre Deckkraft/Größe kann durch das jeweilige Gewicht bestimmt werden.
    (Geht von 2 Dimensionen aus).
    """
    import matplotlib.lines as mlines
    
    periods = sorted(list(fit_synth.results_periods.keys()))
    if period is None:
        period = periods[-1]
        
    target_data, controls_data, weights, final_samples = _get_pooled_data(fit_synth, period)
    is_multi = fit_synth.params.is_multivariate
    
    if target_data.shape[1] != 2:
        print("2D Scatterplot wird nur für 2D Daten unterstützt.")
        return
        
    if len(controls_data) == 0:
        return
        
    x_t = target_data[:, 0]
    y_t = target_data[:, 1]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    N = len(target_data)
    
    if final_samples is not None:
        disco_dist = final_samples
    else:
        from ..utils import sample_counterfactual_distribution
        first_p = period[0] if isinstance(period, list) else period
        grid = fit_synth.results_periods[first_p].target.grid
        
        # Use normalized weights
        w_sum = np.sum(weights)
        norm_weights = weights / w_sum if w_sum > 1e-8 else weights
        
        disco_dist = sample_counterfactual_distribution(controls_data, norm_weights, grid=grid, num_samples=N)
    
    if disco_dist is not None and len(disco_dist) > 0:
        ax.scatter(disco_dist[:, 0], disco_dist[:, 1], alpha=0.6, s=30, color='red', label='DSC (Sampled)')
        
    ax.scatter(x_t, y_t, alpha=0.7, s=30, color='black', label='Target')
    
    period_label = f"Perioden {period}" if isinstance(period, list) else f"Periode {period}"
    ax.set_title(f'2D Scatterplot ({period_label})', fontsize=14)
    ax.set_xlabel('Dim 1', fontsize=12)
    ax.set_ylabel('Dim 2', fontsize=12)
    
    ax.legend(loc='best', frameon=True, edgecolor='black', fontsize=12)
    ax.grid(linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    plt.tight_layout()
    plt.show()

def plot_transport_comparison(gt_effect, tea_result, period=5, save_path=None):
    """
    Plots a side-by-side comparison of the Ground Truth (GT) transport map and 
    the estimated transport map, along with their difference and a scatter comparison.
    
    Parameters:
    -----------
    gt_effect : dict
        The result of calculate_ground_truth_effect(...).
    tea_result : DiSCoTEAResult
        The TEA result object containing estimated transport maps.
    period : int
        The period for which to plot the comparison (default is 5).
    save_path : str, optional
        Path to save the resulting figure.
    """
    import seaborn as sns

    if period not in gt_effect:
        raise ValueError(f"Period {period} not found in gt_effect keys: {list(gt_effect.keys())}")
    
    if period not in tea_result.treats['Estimate']:
        raise ValueError(f"Period {period} not found in tea_result.treats['Estimate'] keys: {list(tea_result.treats['Estimate'].keys())}")
        
    gt_map = gt_effect[period]['transport_map']
    est_map = tea_result.treats['Estimate'][period]
    
    # Create grid of subplots: 2 top (GT, Estimate), 2 bottom (Difference, Scatter)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Ground Truth Heatmap
    max_val = max(gt_map.max().max(), est_map.max().max())
    sns.heatmap(gt_map, ax=axes[0, 0], cmap='Blues', vmin=0, vmax=max_val, 
                cbar_kws={'label': 'Transport Mass (%)'}, square=True)
    axes[0, 0].set_title(f'Ground Truth Transport Map (Period {period})', fontsize=14, fontweight='bold', pad=10)
    axes[0, 0].set_xlabel('Target Bins', fontsize=12)
    axes[0, 0].set_ylabel('Source Bins', fontsize=12)
    
    # 2. Estimate Heatmap
    sns.heatmap(est_map, ax=axes[0, 1], cmap='Blues', vmin=0, vmax=max_val, 
                cbar_kws={'label': 'Transport Mass (%)'}, square=True)
    axes[0, 1].set_title(f'Estimated Transport Map (Period {period})', fontsize=14, fontweight='bold', pad=10)
    axes[0, 1].set_xlabel('Target Bins', fontsize=12)
    axes[0, 1].set_ylabel('Source Bins', fontsize=12)
    
    # 3. Difference Heatmap
    diff_map = est_map - gt_map
    max_diff = np.max(np.abs(diff_map.values))
    vmin_diff, vmax_diff = (-max_diff, max_diff) if max_diff > 1e-8 else (-1, 1)
    
    sns.heatmap(diff_map, ax=axes[1, 0], cmap='RdBu_r', center=0, vmin=vmin_diff, vmax=vmax_diff, 
                cbar_kws={'label': 'Difference (Estimate - GT) %'}, square=True)
    axes[1, 0].set_title('Difference Map (Estimate - Ground Truth)', fontsize=14, fontweight='bold', pad=10)
    axes[1, 0].set_xlabel('Target Bins', fontsize=12)
    axes[1, 0].set_ylabel('Source Bins', fontsize=12)
    
    # 4. Scatter Plot of coupling values
    gt_flat = gt_map.values.flatten()
    est_flat = est_map.values.flatten()
    
    axes[1, 1].scatter(gt_flat, est_flat, alpha=0.6, color='#1f77b4', edgecolors='k', s=50, label='Coupling Cells')
    
    max_limit = max(gt_flat.max(), est_flat.max())
    axes[1, 1].plot([0, max_limit], [0, max_limit], 'r--', linewidth=2, label='Perfect Agreement (y = x)')
    
    # Calculate performance metrics
    mae = np.mean(np.abs(gt_flat - est_flat))
    rmse = np.sqrt(np.mean((gt_flat - est_flat)**2))
    corr = np.corrcoef(gt_flat, est_flat)[0, 1] if len(np.unique(gt_flat)) > 1 and len(np.unique(est_flat)) > 1 else np.nan
    
    axes[1, 1].text(0.05 * max_limit, 0.85 * max_limit, 
                    f'MAE: {mae:.3f}%\nRMSE: {rmse:.3f}%\nCorr: {corr:.4f}', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8), fontsize=12)
    
    axes[1, 1].set_title('Scatter Comparison of Coupling Values', fontsize=14, fontweight='bold', pad=10)
    axes[1, 1].set_xlabel('Ground Truth Transport Mass (%)', fontsize=12)
    axes[1, 1].set_ylabel('Estimated Transport Mass (%)', fontsize=12)
    axes[1, 1].set_xlim(-0.5, max_limit + 1)
    axes[1, 1].set_ylim(-0.5, max_limit + 1)
    axes[1, 1].legend(loc='lower right', fontsize=12)
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)
    
    # Remove top/right borders for a cleaner look
    axes[1, 1].spines['top'].set_visible(False)
    axes[1, 1].spines['right'].set_visible(False)
        
    plt.suptitle(f'Visual Transport Coupling Comparison (Period {period})', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        
    plt.show()

def plot_fit_image(fit_synth, period=None, resolution=50):
    periods = sorted(list(fit_synth.results_periods.keys()))
    if period is None:
        period = periods[-1]
        
    target_data, controls_data, weights, final_samples = _get_pooled_data(fit_synth, period)
    
    if final_samples is None:
        print("No counterfactual samples available to plot as image.")
        return
        
    if target_data.shape[1] != 2:
        print("Image plotting is only supported for 2D spatial point clouds.")
        return
        
    # Create figure with 1x2 subplots (Target vs Counterfactual)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Target image
    H_t, xedges, yedges = np.histogram2d(target_data[:, 0], target_data[:, 1], bins=resolution, range=[[0, 1], [0, 1]])
    axes[0].imshow(H_t.T, origin='lower', extent=[0, 1, 0, 1], cmap='Greys')
    axes[0].set_title('Target')
    axes[0].axis('off')
    
    # Counterfactual image
    H_c, _, _ = np.histogram2d(final_samples[:, 0], final_samples[:, 1], bins=resolution, range=[[0, 1], [0, 1]])
    axes[1].imshow(H_c.T, origin='lower', extent=[0, 1, 0, 1], cmap='Greys')
    axes[1].set_title('Counterfactual Barycenter')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
