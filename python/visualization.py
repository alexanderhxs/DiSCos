import matplotlib.pyplot as plt

def plot_fit_quantiles(fit_synth, show_controls=False, period=None):    
    periods = sorted(list(fit_synth.results_periods.keys()))
    if period is None:
        # Default to the last period (usually post-treatment)
        period = periods[-1]
        
    period_res = fit_synth.results_periods[period]
    x_grid = fit_synth.evgrid
    target_quantiles = period_res.target.quantiles
    disco_quantiles = period_res.DiSCo.quantile

    plt.figure(figsize=(6, 5))
    plt.plot(x_grid, target_quantiles, color='black', linewidth=3, label='Target')
    plt.plot(x_grid, disco_quantiles, color='red', linewidth=3, label='DSC')
    
    if hasattr(fit_synth, 'CI') and fit_synth.CI is not None and hasattr(fit_synth.CI, 'quantile') and fit_synth.CI.quantile is not None:
        period_idx = periods.index(period)
        lower = fit_synth.CI.quantile.lower[:, period_idx]
        upper = fit_synth.CI.quantile.upper[:, period_idx]
        # x_grid hinzufügen für korrekte Ausrichtung auf der X-Achse
        plt.plot(x_grid, lower, color='red', linewidth=1, linestyle='--', label='CI (DSC)')
        plt.plot(x_grid, upper, color='red', linewidth=1, linestyle='--')

    if show_controls:
        for i in range(period_res.controls.quantiles.shape[1]):
            control_quantiles = period_res.controls.quantiles[:,i]
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
        # Default to the last period (usually post-treatment)
        period = periods[-1]
        
    period_res = fit_synth.results_periods[period]
    
    # In the CDF space, the x-axis is comprised of the evaluation grid of values (Y)
    x_grid = period_res.target.grid
    target_cdf = period_res.target.cdf
    disco_cdf = period_res.DiSCo.cdf

    plt.figure(figsize=(6, 5))
    plt.plot(x_grid, target_cdf, color='black', linewidth=3, label='Target')
    plt.plot(x_grid, disco_cdf, color='red', linewidth=3, label='DSC')
    
    if hasattr(fit_synth, 'CI') and fit_synth.CI is not None and hasattr(fit_synth.CI, 'cdf') and fit_synth.CI.cdf is not None:
        period_idx = periods.index(period)
        lower = fit_synth.CI.cdf.lower[:, period_idx]
        upper = fit_synth.CI.cdf.upper[:, period_idx]
        plt.plot(x_grid, lower, color='red', linewidth=1, linestyle='--', label='CI (DSC)')
        plt.plot(x_grid, upper, color='red', linewidth=1, linestyle='--')

    if show_controls and hasattr(period_res.controls, 'cdf') and period_res.controls.cdf is not None:
        for i in range(period_res.controls.cdf.shape[1]):
            control_cdf = period_res.controls.cdf[:, i]
            plt.plot(x_grid, control_cdf, color='grey', linewidth=1, linestyle='--', label='Controls' if i == 0 else None)

    plt.ylim(-0.02, 1.02)
    plt.xlabel('$y$', fontsize=14)
    plt.ylabel('$F(y)$', fontsize=14)
    plt.legend(loc='lower right', frameon=True, edgecolor='black', framealpha=1, borderpad=1, fontsize=12)

    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.grid(linestyle='--', alpha=0.5)

    plt.show()