import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import joblib
import matplotlib.pyplot as plt
import numpy as np

# Mapping of method names to display names and positions in 2x2 grid
METHODS = ['energy', 'mixture', 'tangential', 'swasserstein']
DISPLAY_METHODS = {
    'energy': 'Energy',
    'mixture': 'Mixture',
    'tangential': 'Tangential',
    'swasserstein': 'S-Wasserstein'
}

def load_mc_data(method):
    metrics_file = f"C:\\Dokumente\\Studium\\1. Master Thesis\\DiSCos\\python\\results\\metrics\\mc_metrics_{method}.pkl"
    teas_file = f"C:\\Dokumente\\Studium\\1. Master Thesis\\DiSCos\\python\\results\\effects\\mc_teas_{method}.pkl"
    
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"Ergebnisdatei nicht gefunden: {metrics_file}. Bitte zuerst mc_metrics.py ausführen.")
        
    data = joblib.load(metrics_file)
    if os.path.exists(teas_file):
        teas_data = joblib.load(teas_file)
        data['simple_teas'] = teas_data.get('simple_teas', [])
    return data

def extract_metric_arrays(gt_effects_list, metric_list, periods, metric_key):
    n_mc = len(gt_effects_list)
    n_t = len(periods)
    
    gt_arr = np.full((n_mc, n_t), np.nan)
    arr = np.full((n_mc, n_t), np.nan)
    
    for i in range(n_mc):
        gt_dict = gt_effects_list[i]
        m_dict = metric_list[i] if i < len(metric_list) else None
        
        for t_idx, t in enumerate(periods):
            if t in gt_dict and metric_key in gt_dict[t]:
                val = gt_dict[t][metric_key]
                if val is not None and not np.isnan(val):
                    gt_arr[i, t_idx] = float(val)
                    
            if m_dict is not None and t in m_dict and metric_key in m_dict[t]:
                val = m_dict[t][metric_key]
                if val is not None and not np.isnan(val):
                    arr[i, t_idx] = float(val)
                    
    return gt_arr, arr

def extract_tea_arrays(gt_effects_list, simple_teas, periods, dim):
    n_mc = len(gt_effects_list)
    n_t = len(periods)
    
    gt_arr = np.full((n_mc, n_t), np.nan)
    est_arr = np.full((n_mc, n_t), np.nan)
    
    for i in range(n_mc):
        gt_dict = gt_effects_list[i]
        tea_obj = simple_teas[i] if i < len(simple_teas) else None
        
        for t_idx, t in enumerate(periods):
            if t in gt_dict and 'mean_diff' in gt_dict[t]:
                md_gt = gt_dict[t]['mean_diff']
                if md_gt is not None and len(md_gt) > dim:
                    gt_arr[i, t_idx] = float(md_gt[dim])
                    
            if tea_obj is not None and hasattr(tea_obj, 'treats') and t in tea_obj.treats:
                md_est = tea_obj.treats[t].get('mean_diff', np.nan)
                if isinstance(md_est, (list, np.ndarray)) and len(md_est) > dim:
                    est_arr[i, t_idx] = float(md_est[dim])
                    
    return gt_arr, est_arr

def plot_distance_metric(metric_key, display_name):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, method in enumerate(METHODS):
        ax = axes[idx]
        try:
            data = load_mc_data(method)
        except Exception as e:
            ax.text(0.5, 0.5, f"Data missing\n{method}", ha='center', va='center')
            continue
            
        gt_effects_list = data.get('gt_effects', [])
        est_metrics_list = data.get('est_metrics_list', [])
        naive_metrics_list = data.get('naive_metrics_list', [])
        
        if not gt_effects_list or not est_metrics_list or not naive_metrics_list:
            ax.text(0.5, 0.5, f"Missing metrics\n{method}", ha='center', va='center')
            continue
            
        periods = np.array(data['periods'])
        t0 = data['t0']
        
        _, est_arr = extract_metric_arrays(gt_effects_list, est_metrics_list, periods, metric_key)
        _, naive_arr = extract_metric_arrays(gt_effects_list, naive_metrics_list, periods, metric_key)
        
        # Normalize: Dist_Synth / Dist_Naive
        with np.errstate(divide='ignore', invalid='ignore'):
            norm_arr = np.where(naive_arr > 1e-12, est_arr / naive_arr, np.nan)
            
        for i in range(len(norm_arr)):
            ax.plot(periods, norm_arr[i], color='tab:blue', alpha=0.3, linewidth=1)
            
        mean_norm = np.nanmean(norm_arr, axis=0)
        ax.plot(periods, mean_norm, color='darkblue', linewidth=2.5, linestyle='-', marker='o')
        
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5)
        ax.axvline(x=t0 + 1, color='red', linestyle=':', linewidth=1.5)
        
        ax.set_title(DISPLAY_METHODS[method], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if idx >= 2:
            ax.set_xlabel('Time (Periods)', fontsize=10)
        if idx % 2 == 0:
            ax.set_ylabel('Relative Distance (Synth vs. Naive)', fontsize=10)
            
    plt.ylim(0, 1.5)
    
    fig.suptitle(f'Estimation Error - {display_name}', fontsize=16, fontweight='bold')
    
    import matplotlib.lines as mlines
    blue_line = mlines.Line2D([], [], color='tab:blue', alpha=0.3, label='MC Iteration')
    dark_line = mlines.Line2D([], [], color='darkblue', marker='o', linewidth=2.5, label='Mean Relative Error')
    black_line = mlines.Line2D([], [], color='black', linestyle='--', label='Naive Baseline (1.0)')
    red_line = mlines.Line2D([], [], color='red', linestyle=':', label='Treatment Time')
    
    fig.legend(handles=[blue_line, dark_line, black_line, red_line], loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02))
    
    if metric_key == 'w2':
        fig.text(0.5, 0.08, "Note: For Wasserstein-2, some MC iterations may diverge strongly and are clipped for visual clarity.", ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.1, 1, 0.96])
    
    out_dir = os.path.dirname(f"C:\\Dokumente\\Studium\\1. Master Thesis\\DiSCos\\python\\results\\")
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"mc_metrics_distance_{metric_key}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()

def plot_mean_effect(dim, display_name):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, method in enumerate(METHODS):
        ax = axes[idx]
        try:
            data = load_mc_data(method)
        except Exception as e:
            ax.text(0.5, 0.5, f"Data missing\n{method}", ha='center', va='center')
            continue
            
        gt_effects_list = data.get('gt_effects', [])
        simple_teas = data.get('simple_teas', [])
        
        if not gt_effects_list or not simple_teas:
            ax.text(0.5, 0.5, f"Missing metrics\n{method}", ha='center', va='center')
            continue
            
        periods = np.array(data['periods'])
        t0 = data['t0']
        
        gt_arr, est_arr = extract_tea_arrays(gt_effects_list, simple_teas, periods, dim=dim)
        
        # Raw Error
        raw_error = est_arr - gt_arr
        
        # Diff-in-Diff adjustment: center around pre-treatment average error
        pre_indices = np.where(periods <= t0)[0]
        if len(pre_indices) > 0:
            pre_avg = np.nanmean(raw_error[:, pre_indices], axis=1, keepdims=True)
        else:
            pre_avg = 0
            
        adj_error = raw_error - pre_avg
        
        for i in range(len(adj_error)):
            ax.plot(periods, adj_error[i], color='tab:blue', alpha=0.3, linewidth=1)
            
        mean_adj = np.nanmean(adj_error, axis=0)
        ax.plot(periods, mean_adj, color='darkblue', linewidth=2.5, linestyle='-', marker='o')
        
        ax.axhline(y=0.0, color='black', linestyle='--', linewidth=1.5)
        ax.axvline(x=t0 + 1, color='red', linestyle=':', linewidth=1.5)
        
        ax.set_title(DISPLAY_METHODS[method], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if idx >= 2:
            ax.set_xlabel('Time (Periods)', fontsize=10)
        if idx % 2 == 0:
            ax.set_ylabel('Adjusted Bias (Est - GT)', fontsize=10)
            
    fig.suptitle(f'Mean Treatment Effect - {display_name}', fontsize=16, fontweight='bold')
    
    import matplotlib.lines as mlines
    blue_line = mlines.Line2D([], [], color='tab:blue', alpha=0.3, label='MC Iteration')
    dark_line = mlines.Line2D([], [], color='darkblue', marker='o', linewidth=2.5, label='Mean Adjusted Bias')
    black_line = mlines.Line2D([], [], color='black', linestyle='--', label='True Effect (0.0)')
    red_line = mlines.Line2D([], [], color='red', linestyle=':', label='Treatment Time')
    
    fig.legend(handles=[blue_line, dark_line, black_line, red_line], loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    
    out_dir = os.path.dirname(f"C:\\Dokumente\\Studium\\1. Master Thesis\\DiSCos\\python\\results\\")
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"mc_metrics_mean_effect_dim{dim}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()

def main():
    print("Generiere Plots...")
    plot_distance_metric('w2', 'Wasserstein-2')
    plot_distance_metric('energy_divergence', 'Energy Divergence')
    plot_mean_effect(0, 'Dimension 1')
    plot_mean_effect(1, 'Dimension 2')
    print("Alle Plots wurden generiert.")

if __name__ == "__main__":
    main()
