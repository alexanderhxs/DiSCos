import matplotlib.pyplot as plt
import numpy as np

def plot_fit_quantiles(fit_synth, show_controls=False, period=None):    
    periods = sorted(list(fit_synth.results_periods.keys()))
    if period is None:
        period = periods[-1]
        
    period_res = fit_synth.results_periods[period]
    # Check if target data is multidimensional
    is_multi = len(period_res.target.data.shape) > 1 and period_res.target.data.shape[1] > 1
    
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
    is_multi = len(period_res.target.data.shape) > 1 and period_res.target.data.shape[1] > 1
    
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
    is_multi = len(target_data.shape) > 1 and target_data.shape[1] == 2
    
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
    
    # Marginale EDFs für Mischung bilden => F_dsc(x)
    sort_idx_x = np.argsort(pool_x)
    ecdf_x_vals = np.cumsum(pool_w[sort_idx_x])
    u_dsc = np.empty_like(pool_x)
    u_dsc[sort_idx_x] = ecdf_x_vals

    sort_idx_y = np.argsort(pool_y)
    ecdf_y_vals = np.cumsum(pool_w[sort_idx_y])
    v_dsc = np.empty_like(pool_y)
    v_dsc[sort_idx_y] = ecdf_y_vals
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].scatter(u_target, v_target, alpha=0.5, s=20, c='black')
    axes[0].set_title('Target Copula (Empirical Ranks)', fontsize=14)
    axes[0].set_xlabel('Rank Dim 1 ($F_1$)', fontsize=12)
    axes[0].set_ylabel('Rank Dim 2 ($F_2$)', fontsize=12)
    
    # Sample N points prop to weights to keep density scatter visually comparable
    pool_w_norm = pool_w / np.sum(pool_w)
    sample_idx = np.random.choice(len(pool_x), size=N, p=pool_w_norm)
    
    axes[1].scatter(u_dsc[sample_idx], v_dsc[sample_idx], alpha=0.5, s=20, c='red')
    axes[1].set_title('DSC Copula (Weighted Mixture)', fontsize=14)
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
    is_multi = len(target_data.shape) > 1 and target_data.shape[1] == 2
    
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