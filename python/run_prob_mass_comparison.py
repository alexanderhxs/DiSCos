from IPython.core import magic_arguments
from scipy.integrate._ivp.ivp import METHODS
import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sicherstellen, dass das Projekt-Root im Pfad ist
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import python.models
sys.modules['models'] = python.models

from python.tea.base import disco_tea

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

def main():
    try:
        N_MC = 10
        METHOD = 'swasserstein' # Nur eine Methode, wie gewünscht
        
        test_bounds = [(-np.inf, -0.2), (-np.inf, 0.2)] 
        print(f"Starte MC Studie für Probability Mass (Region: {test_bounds})...\n")
        
        all_results = []
        
        print(f"Auswertung für Methode: {METHOD}")
        
        mc_deviations = []
        mc_gt_target_mass = []
        mc_gt_mass_diff = []
        
        valid_periods = None
        t0 = None
        
        for i in range(N_MC):
            pkl_path = os.path.join(os.path.dirname(__file__), "results", "fits", f"disco_{METHOD}_te_mc{i}.pkl")
            if not os.path.exists(pkl_path):
                continue
                
            disco_res = joblib.load(pkl_path)
            
            if t0 is None:
                t0 = disco_res.params.t0
            
            tea_res = disco_tea(disco_res, agg="prob_mass", graph=False, bounds=test_bounds)
            est_df = tea_res.agg_df
            
            gt_df = calc_gt_mass_diff(disco_res, bounds=test_bounds)
            
            if gt_df is not None:
                comparison = pd.merge(gt_df, est_df, on="Time")
                
                if valid_periods is None:
                    valid_periods = comparison['Time'].values
                    
                deviation = comparison['Mass Diff'].values - comparison['GT Mass Diff'].values
                mc_deviations.append(deviation)
                mc_gt_target_mass.append(comparison['GT Target Mass'].values)
                mc_gt_mass_diff.append(comparison['GT Mass Diff'].values)
        
        if len(mc_deviations) > 0:
            mc_deviations_arr = np.array(mc_deviations)
            mean_sq_dev = np.mean(mc_deviations_arr**2, axis=0)
            max_dev = np.max(np.abs(mc_deviations), axis=0)
            
            gt_target_arr = np.array(mc_gt_target_mass)
            mean_gt_target = np.mean(gt_target_arr, axis=0)
            
            gt_diff_arr = np.array(mc_gt_mass_diff)
            mean_gt_diff = np.mean(gt_diff_arr, axis=0)
            post_treat_means = mean_sq_dev[t0:]
            
            print(f"  Gefundene MC-Iterationen: {len(mc_deviations)}")
            print(f"  Mean Squared Error über post treatment Perioden: {np.mean(post_treat_means):.4f}")
            
            # Gebe die gewünschten Average-Werte pro Periode aus
            print("\n  Durchschnittliche Werte pro Periode (über alle MC Iterationen):")
            print("  Time | Avg GT Target Mass | Avg GT Mass Diff | Mean Sq. Error | Max Error (Bias)")
            print("  " + "-"*75)
            for idx, t in enumerate(valid_periods):
                print(f"  {t:4} | {mean_gt_target[idx]:18.4f} | {mean_gt_diff[idx]:16.4f} | {mean_sq_dev[idx]:20.4f} | {max_dev[idx]:20.4f}")
            
            all_results.append({
                'method': METHOD,
                'periods': valid_periods,
                'mean_dev': mean_sq_dev,
                't0': t0,
                'mc_deviations': mc_deviations_arr
            })
        else:
            print(f"  Keine Daten für Methode {METHOD} gefunden.\n")

        # Plotting (mit Einzel-Trajektorien)
        if len(all_results) > 0:
            plt.figure(figsize=(10, 6))
            
            res = all_results[0]
            periods = res['periods']
            mean_sq_dev = res['mean_dev']
            t0 = res['t0']
            mc_deviations_arr = res['mc_deviations']
            
            # Zeichne jede MC-Iteration einzeln ein (transparent)
            for i in range(mc_deviations_arr.shape[0]):
                plt.plot(periods, mc_deviations_arr[i], color='gray', alpha=0.3, linewidth=1, 
                         label='Einzelne MC-Iteration' if i == 0 else "")
            
            # Zeichne den Durchschnitt
            #plt.plot(periods, mean_sq_dev, label=f"{res['method']} (Mean Sq. Error)", color='blue', linewidth=2.5, marker='o')
                
            plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
            if t0 is not None:
                plt.axvline(t0 + 1, color='red', linestyle=':', label='Treatment', linewidth=1.5)
                
            plt.title(f'Absolute Abweichung (Est - GT) der Probability Mass\n Method: {METHOD}', fontsize=14, fontweight='bold')
            plt.xlabel('Zeit (Perioden)', fontsize=12)
            plt.ylabel('Abweichung vom GT (Probability Mass)', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(os.path.dirname(__file__), "results", f"prob_mass_mc_study_{METHOD}.png")
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot mit Einzel-Trajektorien wurde gespeichert unter: {plot_path}")

    except Exception as e:
        print(f"Fehler: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
