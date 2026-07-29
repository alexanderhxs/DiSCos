import joblib
from disco import DiSCo
from temp import get_continuous_data,  generate_dynamic_panel_data, create_mdsc_panel_data, generate_multivariate_panel_dgp
import numpy as np
import argparse
import joblib
import logging
import pandas as pd

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# from your_module import DiSCo  # Hier dein Modell importieren

def parse_args():
    parser = argparse.ArgumentParser(description="Run DiSCo Model Experiment")

    # --- Daten & Spalten ---
    parser.add_argument('--data_path', type=str, default=None, 
                        help='Pfad zum Datensatz (z.B. data.csv)')
    parser.add_argument('--id_col', type=str, default='id_col', 
                        help='Name der ID-Spalte')
    parser.add_argument('--time_col', type=str, default='time_col', 
                        help='Name der Zeit-Spalte')
    parser.add_argument('--id_col_target', type=str, default='0', 
                        help='ID der Ziel-Einheit')
    
    # nargs='+' erlaubt die Übergabe mehrerer Argumente: --y_col y_col_1 y_col_2
    parser.add_argument('--y_col', type=str, nargs='+', default=['y_col_1', 'y_col_2'], 
                        help='Liste der Zielvariablen')

    # --- Numerische Parameter ---
    parser.add_argument('--t0', type=int, default=5, help='Zeitpunkt des Treatments (T0)')
    parser.add_argument('--M', type=int, default=1000, help='Anzahl Iterationen M')
    parser.add_argument('--G', type=int, default=10, help='Grid-Größe G')
    parser.add_argument('--B', type=int, default=10, help='Anzahl der Bootstraps B')

    # --- MC Parameter ---
    parser.add_argument('--n_mc', type=int, default=1, help='Anzahl der Monte Carlo Iterationen')
    parser.add_argument('--base_seed', type=int, default=16, help='Basis Seed für MC Simulationen')

    # --- Boolean Flags ---
    # In Python 3.9+ ist BooleanOptionalAction perfekt. Es generiert automatisch 
    # --simplex und --no-simplex für die Konsole.
    parser.add_argument('--simplex', action=argparse.BooleanOptionalAction, default=True, 
                        help='Simplex-Bedingung aktivieren/deaktivieren')
    parser.add_argument('--joint_opt', action=argparse.BooleanOptionalAction, default=False, 
                        help='Joint Optimization aktivieren/deaktivieren')
    
    # Für Standard-False-Werte reicht 'store_true' (wird True, wenn in Konsole angegeben)
    parser.add_argument('--perm', action='store_true', help='Führe Permutationstest durch')
    parser.add_argument('--ci', action='store_true', help='Berechne Confidence Intervals')

    # --- Modell & Speichern ---
    parser.add_argument('--method', type=str, default='tangential', help='DiSCo Methode')
    parser.add_argument('--out', type=str, default='python/disco_results.pkl', help='Ausgabepfad')
    parser.add_argument('--num_workers', type=int, default=1, help='Anzahl der Worker für Parallelisierung')

    return parser.parse_args()

if __name__ == "__main__":
    # 1. Argumente parsen
    args = parse_args()

    import pathlib

    for mc_i in range(args.n_mc):
        current_seed = args.base_seed + mc_i
        np.random.seed(current_seed)
        
        logger.info(f"=== Starte MC Iteration {mc_i + 1}/{args.n_mc} (Seed: {current_seed}) ===")

        # 2. Daten laden
        logger.info(f"Lade Daten von {args.data_path}...")
        if args.data_path is None:
            #df = create_mdsc_panel_data(sample_size=1000, num_controls=25, num_periods=10, dim=2, ar_coef=np.array([[0.5, 0.3], [0.4, 0.6]]), t_treat=5, apply_treatment=True)
            #loaded_model = joblib.load('python/disco_results.pkl')
            #df = loaded_model.params.df
            df = get_continuous_data(sample_size=1000, num_controls=25, num_periods=10, dim=2, t_treat=6, seed=current_seed) # t_treat > num_periods for no treatment
        else:
            df = pd.read_csv(args.data_path)  # Oder read_parquet, read_excel etc.

        # 3. Dictionary für das Modell aufbauen
        arguments = {
            'df': df,
            'id_col': args.id_col,
            'time_col': args.time_col,
            'id_col_target': args.id_col_target,
            'y_col': args.y_col,
            't0': args.t0,
            'M': args.M,
            'G': args.G,
            'B': args.B,
            'simplex': args.simplex,
            'perm': args.perm,
            'CI': args.ci,
            'num_cores': args.num_workers,
        }

        if mc_i == 0:
            print(df.head())  # Optional: Zeige die ersten Zeilen des DataFrames, um sicherzustellen, dass er korrekt geladen wurde
            # print full config
            logger.info("Konfiguration:")
            for key, value in arguments.items():
                if key == 'df':
                    pass
                else:
                    print(f"  {key}: {value}")

        # 4. Modell initialisieren
        logger.info(f"Initialisiere DiSCo (Method: {args.method}, Joint Opt: {args.joint_opt})...")
        disco = DiSCo(
            method=args.method,
            joint_opt=args.joint_opt,
            **arguments
        )
        
        # 5. Fitting und Speichern
        logger.info("Starte Fitting...")
        disco_results = disco.fit()

        out_path = pathlib.Path(args.out)
        final_out = out_path.parent / f"{out_path.stem}_mc{mc_i}{out_path.suffix}"

        logger.info(f"Speichere Ergebnisse in {final_out}...")
        joblib.dump(disco_results, final_out)
        logger.info(f"Ergebnisse für Iteration {mc_i + 1}/{args.n_mc} gespeichert.")

