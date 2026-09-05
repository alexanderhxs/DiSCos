from data import get_medicaid_data, get_hybrid_data, get_image_data, get_mnist_data, get_cps_data
import joblib
from disco import DiSCo
from data.data import get_continuous_data,  generate_dynamic_panel_data, create_mdsc_panel_data, generate_multivariate_panel_dgp
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
    parser.add_argument('--base_seed', type=int, default=0, help='Basis Seed für MC Simulationen')

    # --- Daten-Optionen ---
    parser.add_argument('--downsample', type=int, default=None, 
                        help='Maximale Anzahl an Datenpunkten pro ID und Zeitpunkt (zufälliges Sample)')

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
        
        elif args.data_path == 'medicaid':
            df = get_medicaid_data(
                outcome_cols=['INCWAGE', 'UHRSWORK','EMPSTAT','HINSCAID'],
                pooled=False,
                weighted=True)
            args.id_col = 'STATEFIP'
            args.time_col = 'YEAR'
            args.y_col = ['INCWAGE', 'UHRSWORK','EMPSTAT','HINSCAID']

        elif args.data_path == 'hybrid':
            df, true_weights = get_hybrid_data('python/data/datasets/medicaid.csv', seed = mc_i)
            args.id_col = 'STATEFIP'
            args.time_col = 'YEAR'
            args.y_col = ['INCWAGE', 'UHRSWORK','EMPSTAT','HINSCAID']
            args.id_col_target = 'synthetic_target'

        elif args.data_path == 'image':
            df = get_image_data(num_samples=20000, seed=current_seed)
            args.id_col = 'ID'
            args.time_col = 'TIME'
            args.y_col = ['X', 'Y']
            if args.id_col_target == '0':
                args.id_col_target = '0001.png'
        
        elif args.data_path == 'mnist':
            # Pass string arguments for digit and corruption
            df = get_mnist_data(digit='4', n_controls=10, corruption='occlusion', num_samples=5000, seed=current_seed)
            args.id_col = 'ID'
            args.time_col = 'TIME'
            args.y_col = ['X', 'Y']
            args.id_col_target = 'target'
        
        elif args.data_path == 'cps':
            df = get_cps_data(num_samples=2000, random_state=current_seed)
            args.id_col = 'state'
            args.time_col = 'year'
            args.y_col = ['earnhre', 'uhourse']
            # Target is state 22 by default
            if args.id_col_target == '0':
                args.id_col_target = 22
            else:
                args.id_col_target = int(args.id_col_target)
        else:
            df = pd.read_csv(args.data_path)  # Oder read_parquet, read_excel etc.

        # Sicherstellen, dass die Zielvariablen float32 sind
        df[args.y_col] = df[args.y_col].astype(np.float32)

        if args.downsample is not None:
            logger.info(f"Downsampling auf max {args.downsample} Beobachtungen pro {args.id_col} und {args.time_col}...")
            sampled_indices = df.groupby([args.id_col, args.time_col]).apply(
                lambda x: x.sample(n=min(len(x), args.downsample), random_state=current_seed).index
            ).explode().values
            df = df.loc[sampled_indices].reset_index(drop=True)

        # 3. Datentypen dynamisch an die DataFrame-Spalten anpassen
        id_type = df[args.id_col].dtype.type
        id_target = id_type(args.id_col_target)

        time_type = df[args.time_col].dtype.type
        t0 = time_type(args.t0)

        arguments = {
            'df': df,
            'id_col': args.id_col,
            'time_col': args.time_col,
            'id_col_target': id_target,
            'y_col': args.y_col,
            't0': t0,
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
        if args.n_mc > 1:
            final_out = out_path.parent / f"{out_path.stem}_{args.method}_mc{mc_i}{out_path.suffix}"
        else:
            final_out = out_path.parent / f"{out_path.stem}_{args.method}{out_path.suffix}"
        
        logger.info(f"Speichere Ergebnisse in {final_out}...")
        joblib.dump(disco_results, final_out)
        logger.info(f"Ergebnisse für Iteration {mc_i + 1}/{args.n_mc} gespeichert.")

