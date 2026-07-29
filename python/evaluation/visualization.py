import matplotlib.pyplot as plt
import numpy as np

def plot_fit_quantiles(fit_synth, show_controls=False, period=None):    
    periods = sorted(list(fit_synth.results_periods.keys()))
    if period is None:
        period = periods[-1]
        
    period_res = fit_synth.results_periods[period]
    # Check if target data is multidimensional
    is_multi = fit_synth.params.is_multivariate
    
    if is_multi:
        dim = period_res.target.data.shape[1]
        weights = period_res.DiSCo.weights if period_res.DiSCo.weights is not None else fit_synth.weights
        
        fig, axes = plt.subplots(1, dim, figsize=(6 * dim, 5))
        if dim == 1: axes = [axes]
            
        for d, ax in enumerate(axes):
            # Lade Dimension d aus test-daten (nicht aus pd.DataFrame by name!)
            target_data = period_res.target.data[:, d]
            controls_data = [c[:, d] for c in period_res.controls.data]
            
            grid_min = min(target_data.min(), np.min([c.min() for c in controls_data if len(c) > 0]))
            grid_max = max(target_data.max(), np.max([c.max() for c in controls_data if len(c) > 0]))
            grid = np.linspace(grid_min, grid_max, 200)
            
            target_cdf = np.mean(target_data[:, None] <= grid, axis=0)
            synth_cdf = np.zeros_like(grid)
            for c_data, w in zip(controls_data, weights):
                if w > 1e-5 and len(c_data) > 0:
                    synth_cdf += w * np.mean(c_data[:, None] <= grid, axis=0)
                    
            # 1. Wir brauchen ein auf [0, 1] verteiltes Standard-Quantilgrid
            q_grid = np.linspace(0, 1, 200)
            
            # 2. Target Quantil via np.quantile (äquivalent zu R's quantile)
            target_quantiles = np.quantile(target_data, q_grid)
            
            # 3. Synth Quantil numerisch berechnen (Inverse ECDF)
            # np.interp erwartet, dass die X-Werte (hier synth_cdf) monoton wachsend sind
            # Bei ECDFs kann es aber plateaus geben (gleiche CDF-Werte). 
            # Besserer Ansatz für Inverse CDF: Für jedes q den kleinsten grid-Wert suchen, 
            # wo die CDF >= q ist.
            synth_quantiles = np.array([grid[np.searchsorted(synth_cdf, q)] if q <= synth_cdf[-1] else grid[-1] for q in q_grid])
            
            ax.plot(q_grid, target_quantiles, color='black', linewidth=3, label='Target')
            ax.plot(q_grid, synth_quantiles, color='red', linewidth=3, label='DSC')
            
            if hasattr(fit_synth, 'CI') and fit_synth.CI is not None and getattr(fit_synth.CI.bootmat, 'weights', None) is not None:
                n_boots = fit_synth.CI.bootmat.weights.shape[1]
                boot_synth_quantiles = np.zeros((n_boots, len(q_grid)))
                
                for b in range(n_boots):
                    b_weights = fit_synth.CI.bootmat.weights[:, b]
                    b_synth_cdf = np.zeros_like(grid)
                    for c_data, w in zip(controls_data, b_weights):
                        if w > 1e-5 and len(c_data) > 0:
                            b_synth_cdf += w * np.mean(c_data[:, None] <= grid, axis=0)
                            
                    boot_synth_quantiles[b, :] = np.array([grid[np.searchsorted(b_synth_cdf, q)] if q <= b_synth_cdf[-1] else grid[-1] for q in q_grid])
                
                alpha = 1 - fit_synth.params.cl
                lower = np.quantile(boot_synth_quantiles, alpha / 2, axis=0)
                upper = np.quantile(boot_synth_quantiles, 1 - (alpha / 2), axis=0)
                ax.plot(q_grid, lower, color='red', linewidth=1, linestyle='--', label='CI (DSC)')
                ax.plot(q_grid, upper, color='red', linewidth=1, linestyle='--')

            if show_controls:
                for i, c_data in enumerate(controls_data):
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
            
        plt.tight_layout()
        plt.show()
        return

    # Fallback auf reines 1D Verhalten
    x_grid = fit_synth.evgrid
    
    # Sicherstellen, dass das grid fürs Plotten monoton steigend sortiert ist
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
        # x_grid hinzufügen für korrekte Ausrichtung auf der X-Achse
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

    # Adding the border around the axis as seen in the R plot
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
        
    period_res = fit_synth.results_periods[period]
    is_multi = fit_synth.params.is_multivariate
    
    if is_multi:
        dim = period_res.target.data.shape[1]
        weights = period_res.DiSCo.weights if period_res.DiSCo.weights is not None else fit_synth.weights
        
        fig, axes = plt.subplots(1, dim, figsize=(6 * dim, 5))
        if dim == 1: axes = [axes]
            
        for d, ax in enumerate(axes):
            target_data = period_res.target.data[:, d]
            controls_data = [c[:, d] for c in period_res.controls.data]
            
            grid_min = min(target_data.min(), np.min([c.min() for c in controls_data if len(c) > 0]))
            grid_max = max(target_data.max(), np.max([c.max() for c in controls_data if len(c) > 0]))
            grid = np.linspace(grid_min, grid_max, 200)
            
            target_cdf = np.mean(target_data[:, None] <= grid, axis=0)
            synth_cdf = np.zeros_like(grid)
            for c_data, w in zip(controls_data, weights):
                if w > 1e-5 and len(c_data) > 0:
                    synth_cdf += w * np.mean(c_data[:, None] <= grid, axis=0)
                    
            ax.plot(grid, target_cdf, label="Target", color="black", linewidth=3)
            ax.plot(grid, synth_cdf, label="DSC", color="red", linewidth=3)
            
            if hasattr(fit_synth, 'CI') and fit_synth.CI is not None and getattr(fit_synth.CI.bootmat, 'weights', None) is not None:
                n_boots = fit_synth.CI.bootmat.weights.shape[1]
                boot_synth_cdf = np.zeros((n_boots, len(grid)))
                
                for b in range(n_boots):
                    b_weights = fit_synth.CI.bootmat.weights[:, b]
                    b_cdf = np.zeros_like(grid)
                    for c_data, w in zip(controls_data, b_weights):
                        if w > 1e-5 and len(c_data) > 0:
                            b_cdf += w * np.mean(c_data[:, None] <= grid, axis=0)
                    boot_synth_cdf[b, :] = b_cdf
                
                alpha = 1 - fit_synth.params.cl
                lower = np.quantile(boot_synth_cdf, alpha / 2, axis=0)
                upper = np.quantile(boot_synth_cdf, 1 - (alpha / 2), axis=0)
                ax.plot(grid, lower, color='red', linewidth=1, linestyle='--', label='CI (DSC)')
                ax.plot(grid, upper, color='red', linewidth=1, linestyle='--')

            if show_controls:
                for i, c_data in enumerate(controls_data):
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
            
        plt.tight_layout()
        plt.show()
        return
        
    # In the CDF space, the x-axis is comprised of the evaluation grid of values (Y)
    x_grid = period_res.target.grid
    
    # Sicherstellen, dass das grid fürs Plotten monoton steigend sortiert ist
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
        
    period_res = fit_synth.results_periods[period]
    target_data = period_res.target.data
    is_multi = fit_synth.params.is_multivariate
    
    if not is_multi:
        print("Joint Plot wird nur für 2D Daten unterstützt.")
        return
        
    weights = period_res.DiSCo.weights if period_res.DiSCo.weights is not None else fit_synth.weights
    controls_data = [c for w, c in zip(weights, period_res.controls.data) if w > 1e-5 and len(c) > 0]
    filtered_weights = [w for w in weights if w > 1e-5]
    
    if len(controls_data) == 0:
        return
    
    # Target Ranks (u, v) in [0,1]
    N = len(target_data)
    u_target = rankdata(target_data[:, 0]) / N
    v_target = rankdata(target_data[:, 1]) / N
    
    from ..utils import sample_counterfactual_distribution
    grid = period_res.target.grid
    disco_dist = sample_counterfactual_distribution(period_res.controls.data, weights, grid=grid, num_samples=N)
    
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
    axes[0].set_title('Target Copula (Empirical Ranks)', fontsize=14)
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
        
    period_res = fit_synth.results_periods[period]
    target_data = period_res.target.data
    is_multi = fit_synth.params.is_multivariate
    
    if not is_multi:
        print("Joint Contour Plot wird nur für 2D Daten unterstützt.")
        return
        
    weights = period_res.DiSCo.weights if period_res.DiSCo.weights is not None else fit_synth.weights
    controls_data = [c for w, c in zip(weights, period_res.controls.data) if w > 1e-5 and len(c) > 0]
    filtered_weights = [w for w in weights if w > 1e-5]
    
    if len(controls_data) == 0:
        return
    
    # Target Data
    x_t = target_data[:, 0]
    y_t = target_data[:, 1]
    
    # DSC Mixture Pooling
    pool_x = []
    pool_y = []
    pool_w = []
    
    for c_data, w in zip(controls_data, filtered_weights):
        pool_x.extend(c_data[:, 0])
        pool_y.extend(c_data[:, 1])
        pool_w.extend([w / len(c_data)] * len(c_data))
            
    pool_x = np.array(pool_x)
    pool_y = np.array(pool_y)
    pool_w = np.array(pool_w)
    
    # 2D Grid für die Evaluierung erstellen
    x_min = min(x_t.min(), pool_x.min())
    x_max = max(x_t.max(), pool_x.max())
    y_min = min(y_t.min(), pool_y.min())
    y_max = max(y_t.max(), pool_y.max())
    
    # Padding
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    
    X, Y = np.mgrid[x_min-x_pad:x_max+x_pad:100j, y_min-y_pad:y_max+y_pad:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    # KDE Target
    try:
        kde_target = gaussian_kde(np.vstack([x_t, y_t]))
        Z_target = np.reshape(kde_target(positions).T, X.shape)
    except np.linalg.LinAlgError:
        print("LinAlgError beim Berechnen der Target KDE (evtl. Datenpunkte zu dicht).")
        return
        
    # KDE DSC
    try:
        kde_dsc = gaussian_kde(np.vstack([pool_x, pool_y]), weights=pool_w)
        Z_dsc = np.reshape(kde_dsc(positions).T, X.shape)
    except np.linalg.LinAlgError:
        print("LinAlgError beim Berechnen der DSC KDE (evtl. Punkte zu dicht).")
        return
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Contour plots
    contour_t = ax.contour(X, Y, Z_target, levels=5, colors='black', linewidths=2, alpha=0.8)
    contour_d = ax.contour(X, Y, Z_dsc, levels=5, colors='red', linewidths=2, alpha=0.8)
    
    # Custom legend
    legend_target = mlines.Line2D([], [], color='black', linewidth=2, label='Target')
    legend_dsc = mlines.Line2D([], [], color='red', linewidth=2, label='DSC')
    ax.legend(handles=[legend_target, legend_dsc], loc='best', frameon=True, edgecolor='black', fontsize=12)
    
    ax.set_title('Joint Density (KDE Contour Overlay)', fontsize=14)
    ax.set_xlabel('Dim 1', fontsize=12)
    ax.set_ylabel('Dim 2', fontsize=12)
    ax.grid(linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
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
        
    period_res = fit_synth.results_periods[period]
    target_data = period_res.target.data
    is_multi = fit_synth.params.is_multivariate
    
    if not is_multi:
        print("2D Scatterplot wird nur für 2D Daten unterstützt.")
        return
        
    weights = period_res.DiSCo.weights if period_res.DiSCo.weights is not None else fit_synth.weights
    controls_data = [c for w, c in zip(weights, period_res.controls.data) if w > 1e-5 and len(c) > 0]
    filtered_weights = [w for w in weights if w > 1e-5]
    
    if len(controls_data) == 0:
        return
        
    # Target Data
    x_t = target_data[:, 0]
    y_t = target_data[:, 1]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    N = len(target_data)
    from ..utils import sample_counterfactual_distribution
    grid = period_res.target.grid
    disco_dist = sample_counterfactual_distribution(period_res.controls.data, weights, grid=grid, num_samples=N)
    
    if disco_dist is not None and len(disco_dist) > 0:
        ax.scatter(disco_dist[:, 0], disco_dist[:, 1], alpha=0.6, s=30, color='red', label='DSC (Sampled)')
        
    ax.scatter(x_t, y_t, alpha=0.7, s=30, color='black', label='Target')
    
    ax.set_title('2D Scatterplot (Bivariate Distribution)', fontsize=14)
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