import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import seaborn as sns
import sys
import os
import logging
import copy
from joblib import Parallel, delayed

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Füge Projekt-Root zum Pfad hinzu, damit `python.evaluation` funktioniert
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from python.evaluation.metrics import calculate_ground_truth_effect, calculate_pretreatment_fit
from python.tea import disco_tea

# =============================================================================
# KONFIGURATION (Hier kannst du einfach alles einstellen)
# =============================================================================
METHODS_TO_RUN = ['mixture', 'swasserstein', 'energy', 'tangential']
N_MC = 1
NUM_WORKERS = 1

# Steuerung, was genau berechnet / nachberechnet werden soll
RECALC_GT = False            # True: Ground Truth Effekte (w1, w2, energy) (neu) berechnen (Achtung: Nur für simulierte Daten mit _cf Spalten!)
RECALC_FIT = True            # True: Pretreatment Fit Metriken & Simple TEA (langsam) neu berechnen
RECALC_NAIVE = True          # True: Nur Naive Baseline für W2/Energy nachberechnen
CALC_TRANSPORT_MAP = False   # True: Langsame Quantile/Transport Map im GT berechnen (meist nicht nötig)

# Dateiname-Prefix (z.B. für die Medicaid-Daten)
BASE_FILENAME = 'medicaid_30_2017'
# =============================================================================

def process_mc_iteration(f, recalc_gt, recalc_fit, recalc_naive, calc_transport_map):
    """
    Lädt das Ergebnis, berechnet je nach Konfiguration die Metriken und gibt sie zurück.
    Gibt (gt_effect, simple_tea, est_metrics, naive_metrics) zurück. Alles was nicht berechnet wird, ist None.
    """
    logger.info(f"Verarbeite Datei: {os.path.basename(f)}...")
    r = joblib.load(f)
    
    gt_effect = None
    if recalc_gt:
        y_cols = [c for c in r.params.df.columns if c.startswith('y_col') and not c.endswith('_cf')]
        cf_cols = [c + '_cf' for c in y_cols]
        
        gt_effect = calculate_ground_truth_effect(
            df=r.params.df, 
            target_id=r.params.id_col_target, 
            time_col=r.params.time_col, 
            y_cols=y_cols, 
            cf_cols=cf_cols,
            calc_wasserstein=True,
            calc_transport_map=calc_transport_map
        )
    
    simple_tea = None
    est_metrics = None
    if recalc_fit:
        # Simple TEA
        simple_tea = disco_tea(r, agg='simple')
        
        # Distanzen (Pretreatment Fit / DiSCo Distances)
        fit_metrics = calculate_pretreatment_fit(r)
        est_metrics = fit_metrics.metrics_per_period
        
    naive_metrics = None
    if recalc_naive:
        r_naive = copy.copy(r)

        J = len(r_naive.weights)
        r_naive.weights = np.ones(J) / J

        # Berechne Naive Fit Metriken (ohne Downsampling)
        naive_fit_metrics = calculate_pretreatment_fit(r_naive)
        naive_metrics = naive_fit_metrics.metrics_per_period

    logger.info(f"Abgeschlossen: {os.path.basename(f)}")
    return gt_effect, simple_tea, est_metrics, naive_metrics

if __name__ == "__main__":
    for method in METHODS_TO_RUN:
        logger.info(f"Starte Auswertung für Methode: {method} (GT={RECALC_GT}, FIT={RECALC_FIT}, NAIVE={RECALC_NAIVE})")

        if N_MC > 1:
            results_files = [os.path.join(project_root, f"python/results/fits/{BASE_FILENAME}_{method}_mc{i}.pkl") for i in range(N_MC)]    
        else:
            results_files = [os.path.join(project_root, f"python/results/fits/{BASE_FILENAME}_{method}.pkl")]
        
        if not os.path.exists(results_files[0]):
            logger.warning(f"Fit-Dateien für {method} nicht gefunden ({results_files[0]}). Überspringe...")
            continue
            
        # Führe die Berechnung parallel durch
        results_metrics = Parallel(n_jobs=NUM_WORKERS)(
            delayed(process_mc_iteration)(f, RECALC_GT, RECALC_FIT, RECALC_NAIVE, CALC_TRANSPORT_MAP) for f in results_files
        )
        
        gt_effects_new = [res[0] for res in results_metrics]
        simple_teas_new = [res[1] for res in results_metrics]
        est_metrics_list_new = [res[2] for res in results_metrics]
        naive_metrics_list_new = [res[3] for res in results_metrics]
        
        # Pfade
        if N_MC > 1:
            metrics_file = os.path.join(project_root, f"python/results/metrics/{BASE_FILENAME}_{method}_mc_metrics.pkl")
            teas_file = os.path.join(project_root, f"python/results/effects/{BASE_FILENAME}_{method}_mc_teas.pkl")
        else:
            metrics_file = os.path.join(project_root, f"python/results/metrics/{BASE_FILENAME}_{method}_metrics.pkl")
            teas_file = os.path.join(project_root, f"python/results/effects/{BASE_FILENAME}_{method}_teas.pkl")
        
        # 1. Bestehende Daten laden, um sie nur partiell zu überschreiben
        if os.path.exists(metrics_file):
            metrics_data = joblib.load(metrics_file)
        else:
            metrics_data = {'gt_effects': None, 'est_metrics_list': None, 'naive_metrics_list': None, 'periods': None, 't0': None}
            
        if os.path.exists(teas_file):
            teas_data = joblib.load(teas_file)
        else:
            teas_data = {'simple_teas': None}

        # 2. Aktualisieren der Daten basierend auf Config
        if RECALC_GT:
            metrics_data['gt_effects'] = gt_effects_new
            
        if RECALC_FIT:
            metrics_data['est_metrics_list'] = est_metrics_list_new
            if est_metrics_list_new[0] is not None:
                metrics_data['periods'] = sorted(list(est_metrics_list_new[0].keys()))
            if simple_teas_new[0] is not None:
                metrics_data['t0'] = simple_teas_new[0].t0
            teas_data['simple_teas'] = simple_teas_new
            
        if RECALC_NAIVE:
            metrics_data['naive_metrics_list'] = naive_metrics_list_new

        # 3. Abspeichern der zusammengeführten/aktualisierten Ergebnisse
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        joblib.dump(metrics_data, metrics_file)
        
        if RECALC_FIT:
            os.makedirs(os.path.dirname(teas_file), exist_ok=True)
            joblib.dump(teas_data, teas_file)
            
        logger.info(f"Metriken für {method} erfolgreich aktualisiert und gespeichert.\n")
