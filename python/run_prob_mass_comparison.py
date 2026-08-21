import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import python.models
sys.modules['models'] = python.models

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from python.tea.base import disco_tea

METHODS = ['energy', 'mixture', 'tangential', 'swasserstein']
DISPLAY_METHODS = {
    'energy': 'Energy',
    'mixture': 'Mixture',
    'tangential': 'Tangential',
    'swasserstein': 'S-Wasserstein'
}

def calc_gt_mass_diff(disco_res, bounds):
    """
    Berechnet die Ground Truth Wahrscheinlichkeitsmassendifferenz direkt
    aus den Rohdaten des DataFrames (ohne DiSCo-Schätzung).
    """
    df = disco_res.params.df
    target_id = disco_res.params.id_col_target
    time_col = disco_res.params.time_col
    
    target_df = df[df['id_col'].astype(str) == str(target_id)]
    periods = sorted(target_df[time_col].unique())
    
    y_cols = [c for c in df.columns if c.startswith('y_col') and not c.endswith('_cf')]
    cf_cols = [c + '_cf' for c in y_cols]
    
    if not all(c in df.columns for c in cf_cols):
        return None
        
    D = len(y_cols)
    gt_diffs = []
    
    for t in periods:
        data_t = target_df[target_df[time_col] == t]
        
        y_treated = np.asarray(data_t[y_cols])
        y_cf = np.asarray(data_t[cf_cols])
        
        if len(y_treated) == 0:
            continue
            
        in_bounds_target = np.ones(len(y_treated), dtype=bool)
        for d in range(D):
            b_min, b_max = bounds[d]
            in_bounds_target &= (y_treated[:, d] >= b_min) & (y_treated[:, d] <= b_max)
        target_mass = np.mean(in_bounds_target)
        
        in_bounds_cf = np.ones(len(y_cf), dtype=bool)
        for d in range(D):
            b_min, b_max = bounds[d]
            in_bounds_cf &= (y_cf[:, d] >= b_min) & (y_cf[:, d] <= b_max)
        cf_mass = np.mean(in_bounds_cf)
        
        gt_diffs.append({
            "Time": t,
            "GT Target Mass": target_mass,
            "GT CF Mass": cf_mass,
            "GT Mass Diff": target_mass - cf_mass
        })
        
    return pd.DataFrame(gt_diffs)

def load_and_calculate_data(method, bounds, n_mc=10):
    mc_gt_mass_diff = []
    mc_est_mass_diff = []
    valid_periods = None
    t0 = None
    
    print(f"Lade und berechne Daten für Methode: {method}...")
    for i in range(n_mc):
        pkl_path = os.path.join(project_root, "python", "results", "fits", f"disco_{method}_te_mc{i}.pkl")
        if not os.path.exists(pkl_path):
            continue
            
        disco_res = joblib.load(pkl_path)
        if t0 is None:
            t0 = disco_res.params.t0
            
        tea_res = disco_tea(disco_res, agg="prob_mass", graph=False, bounds=bounds)
        est_df = tea_res.agg_df
        gt_df = calc_gt_mass_diff(disco_res, bounds=bounds)
        
        if gt_df is not None:
            comparison = pd.merge(gt_df, est_df, on="Time")
            if valid_periods is None:
                valid_periods = comparison['Time'].values
                
            mc_gt_mass_diff.append(comparison['GT Mass Diff'].values)
            mc_est_mass_diff.append(comparison['Mass Diff'].values)
            
    if len(mc_gt_mass_diff) == 0:
        return None, None, None, None
        
    gt_arr = np.array(mc_gt_mass_diff)
    est_arr = np.array(mc_est_mass_diff)
    
    return valid_periods, t0, gt_arr, est_arr

def plot_prob_mass_effect(bounds):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    print(f"\n--- MAE for Probability Mass Comparison ---")
    
    for idx, method in enumerate(METHODS):
        ax = axes[idx]
        periods, t0, gt_arr, est_arr = load_and_calculate_data(method, bounds)
        
        if periods is None:
            ax.text(0.5, 0.5, f"Data missing\n{method}", ha='center', va='center')
            print(f"{DISPLAY_METHODS[method]:15s} | Data missing")
            continue
            
        # Raw Error
        raw_error = est_arr - gt_arr
        
        # Calculate MAE
        pre_idx = np.where(periods <= t0)[0]
        post_idx = np.where(periods > t0)[0]
        
        mae_pre = np.nanmean(np.abs(raw_error[:, pre_idx]))
        mae_post = np.nanmean(np.abs(raw_error[:, post_idx]))
        print(f"{DISPLAY_METHODS[method]:15s} | Pre-Treat MAE: {mae_pre:8.4f} | Post-Treat MAE: {mae_post:8.4f}")
        
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
            
    bounds_str = f"Dim1: [{bounds[0][0]}, {bounds[0][1]}], Dim2: [{bounds[1][0]}, {bounds[1][1]}]"
    fig.suptitle(f'Treatment Effect - Probability Mass\nRegion: {bounds_str}', fontsize=16, fontweight='bold')
    
    import matplotlib.lines as mlines
    blue_line = mlines.Line2D([], [], color='tab:blue', alpha=0.3, label='MC Iteration')
    dark_line = mlines.Line2D([], [], color='darkblue', marker='o', linewidth=2.5, label='Mean Adjusted Bias')
    black_line = mlines.Line2D([], [], color='black', linestyle='--', label='True Effect (0.0)')
    red_line = mlines.Line2D([], [], color='red', linestyle=':', label='Treatment Time')
    
    fig.legend(handles=[blue_line, dark_line, black_line, red_line], loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    
    out_dir = os.path.dirname(f"C:\\Dokumente\\Studium\\1. Master Thesis\\DiSCos\\python\\results\\")
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"mc_metrics_prob_mass.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    test_bounds = [(-np.inf, -0.2), (-np.inf, 0.2)]
    print("Generiere Plots für Probability Mass...")
    plot_prob_mass_effect(test_bounds)
    print("Alle Plots wurden generiert.")

if __name__ == "__main__":
    main()
