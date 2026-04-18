import os
import pandas as pd

folder_path = './dube_full_data'
all_dfs = []

# Spalten, die extrahiert werden sollen (können in den Stata-Files variieren, typischerweise 'year', 'state_fips'/'fips', 'adj0contpov')
columns_to_extract = ['year', 'state_fips', 'adj0contpov', 'age']

for file in os.listdir(folder_path):
    if file.endswith('.dta'):
        file_path = os.path.join(folder_path, file)
        
        try:
            # Lade die Stata-Datei
            df = pd.read_stata(file_path, convert_categoricals=False)
            
            # Stelle sicher, dass die gesuchten Spalten existieren (Fallback für 'fips' falls 'state_fips' nicht existiert
            
            # Extrahiere nur die existierenden relevanten Spalten
            df_subset = df.copy()
            
            # Standardisiere den Namen auf 'fips' für das finale DataFrame
            if 'state_fips' in df_subset.columns:
                df_subset.rename(columns={'state_fips': 'fips'}, inplace=True)

            all_dfs.append(df_subset)
            print(f"Loaded {file} with shape {df_subset.shape}")
            
        except Exception as e:
            print(f"Fehler beim Laden von {file}: {e}")

if all_dfs:
    # Füge alle zusammen
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Entferne eventuelle Zeilen, die komplett NaN sind oder wo die ID/Time fehlen
    combined_df.dropna(subset=['year', 'fips'], inplace=True)
    
    print(f"\nErfolgreich zusammengeführt! Gesamte Shape: {combined_df.shape}")
    
    # Schreibe in den gleichen Ordner
    out_path = os.path.join(folder_path, 'dube_combined.csv')
    combined_df.to_csv(out_path, index=False)
    print(f"Gespeichert in: {out_path}")
else:
    print("Es wurden keine .dta Dateien mit den entsprechenden Spalten gefunden.")


