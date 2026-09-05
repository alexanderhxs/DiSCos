import numpy as np
import pandas as pd
import os

import pandas as pd
import numpy as np
import os

def get_cps_data(data_path=None,
                 num_samples=2000,
                 random_state=42,
                 donors=[
                     # --- TIER 1: Deine bisherigen "Golden Donors" ---
                     # (Große Stichproben, harte $7.25 Grenze, keine starken lokalen MWs)
                     23,  # Pennsylvania
                     32,  # Indiana
                     35,  # Wisconsin
                     47,  # Kansas
                     54,  # Virginia
                     56,  # North Carolina
                     74,  # Texas
                     87,  # Utah
                     
                     
                     ## --- TIER 2: Solide weitere Staaten (Etwas kleinere Stichproben) ---
                     #11,  # Maine (Hinweis: Konstanter Mindestlohn lag hier bei $7.50, nicht $7.25)
                     #12,  # New Hampshire
                     #42,  # Iowa
                     #44,  # North Dakota
                     #73,  # Oklahoma
                     #82,  # Idaho
                     
                     # --- TIER 3: Die Südstaaten (Kein eigenes MW-Gesetz) ---
                     # (Hier gilt faktisch das $7.25 Bundesminimum, aber mit kleinen Ausnahmen für Kleinstbetriebe)
                     57,  # South Carolina
                     58,  # Georgia
                     61,  # Kentucky
                     62,  # Tennessee
                     63,  # Alabama
                     64,  # Mississippi
                     72,  # Louisiana
                 ],
                 target=22): # New Jersey
    """
    Fetches CPS data and returns a DataFrame suitable for DiSCo.
    """
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), 'datasets', 'data3', 'data', 'matchedCPS_1979_2016.dta')

    df_cps = pd.read_stata(data_path)

    # 1. Strikter Filter: Keine imputierten Löhne oder Stunden zulassen!
    # (Passe die Spaltennamen an, falls sie in deinem df leicht abweichen, z.B. imputed_1, wageimputed_1 etc.)
    df_cps = df_cps[(df_cps['wageimputed'] == 0) & (df_cps['wageimputed_1'] == 0)]
    df_cps = df_cps[(df_cps['hoursimputed'] == 0) & (df_cps['hoursimputed_1'] == 0)]
    
    # Optional, aber empfohlen für scharfe Spikes: Nur tatsächliche Stundenlöhner
    # df_cps = df_cps[(df_cps['paidhre'] == 1) & (df_cps['paidhre_1'] == 1)]

    # 2. Aufspalten nach Jahren
    df_2013 = df_cps[df_cps['year_1'] == 2013].copy()
    df_2014 = df_cps[df_cps['year'] == 2014].copy()

    states = donors + [target]

    # 3. KORREKTUR: Für 2013 zwingend den state_1 Filter nutzen!
    df_2013s = df_2013[df_2013['state'].isin(states)].copy()
    df_2014s = df_2014[df_2014['state'].isin(states)].copy()

    df_2013s = df_2013s[df_2013s['age_1'].between(18, 65)]
    df_2014s = df_2014s[df_2014s['age'].between(18, 65)]

    # Spaltenauswahl (bei 2013 state_1 beibehalten)
    df_2013s = df_2013s[['state', 'year_1', 'earnwt_1', 'earnhre_1', 'uhourse_1']]
    df_2014s = df_2014s[['state', 'year', 'earnwt', 'earnhre', 'uhourse']]
        
    # 4. Sampling mit Gewichten
    # ACHTUNG: Bei df_2013s muss jetzt nach state_1 gruppiert werden!
    df_2013_sampled = df_2013s.groupby('state', group_keys=False).sample(
        n=num_samples, replace=True, weights='earnwt_1', random_state=random_state
    )
    df_2014_sampled = df_2014s.groupby('state', group_keys=False).sample(
        n=num_samples, replace=True, weights='earnwt', random_state=random_state
    )

    # 5. Vereinheitlichung der Spaltennamen für Concat
    df_2013_sampled.rename(columns={
        'state_1': 'state',
        'year_1': 'year',
        'earnwt_1': 'earnwt',
        'earnhre_1': 'earnhre',
        'uhourse_1': 'uhourse'
    }, inplace=True)
    
    df_final = pd.concat([df_2013_sampled, df_2014_sampled], ignore_index=True)
    
    # 6. Drop NAs VOR der Log-Transformation (sonst crasht np.log bei <= 0)
    df_final.dropna(subset=['earnhre', 'uhourse'], inplace=True)
    df_final = df_final[df_final['earnhre'] > 0]
    
    # Transformationen
    df_final['earnhre'] = np.log(df_final['earnhre'])
    #df_final['uhourse'] += np.random.uniform(-0.5, 0.5, size=len(df_final))
    
    return df_final