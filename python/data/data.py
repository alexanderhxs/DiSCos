import numpy as np
import pandas as pd

import math
from scipy.stats import norm, t

#creation of contoll variables for testing
def get_continuous_data(sample_size, num_controls, num_periods=10, t_treat=6, treatment_effect=2.0, 
                        target_offset=0, dist_control=3, dist_target=4, dim=1, base_corr=0.8, corr_drift=0.05,
                        seed=None): # <-- Neuer Parameter hier
    
    # Lokaler Zufallsgenerator für perfekte Reproduzierbarkeit, 
    # ohne globale Seiteneffekte in deinem restlichen Code zu verursachen.
    rng = np.random.default_rng(seed)
    
    data = []
    
    # 1. Systematische Basis-Korrelationsmatrix
    base_corr_mat = np.full((dim, dim), base_corr)
    np.fill_diagonal(base_corr_mat, 1.0)
    
    def draw_mix(n, n_comp, m, covs):
        # np.random ersetzt durch rng
        c = rng.choice(n_comp, size=n)
        out = np.zeros((n, dim))
        for comp in range(n_comp):
            mask = (c == comp)
            n_samples = mask.sum()
            if n_samples > 0:
                out[mask] = rng.multivariate_normal(m[comp], covs[comp], size=n_samples)
        return out
        
    def generate_params(n_comp, is_target=False):
        offset = target_offset if is_target else 0
        
        # Basis-Zentrum der Unit uniform ziehen
        unit_base = rng.uniform(-5 + offset, 5 + offset, dim)
        
        # 1. Means der Sub-Komponenten generieren
        if dim > 1:
            z = rng.multivariate_normal(np.zeros(dim), base_corr_mat, size=n_comp)
            u = norm.cdf(z)  # Uniform zwischen 0 und 1
            # Sub-Komponenten gruppieren sich engmaschig um das Basis-Zentrum
            means = unit_base + (-8.0 + 16.0 * u)
        else:
            means = unit_base + rng.uniform(-8.0, 8.0, (n_comp, dim))
        
        covs = np.zeros((n_comp, dim, dim))
        for i in range(n_comp):
            if dim > 1:
                # Leicht verbogene Korrelationsmatrix für Varianz innerhalb der Regionen
                noise = rng.uniform(-corr_drift, corr_drift, (dim, dim))
                noise = (noise + noise.T) / 2
                corr_mat = base_corr_mat + noise
                np.fill_diagonal(corr_mat, 1.0)
                
                # Sichern, dass Matrix positiv semi-definit ist
                vals, vecs = np.linalg.eigh(corr_mat)
                vals = np.maximum(vals, 1e-4)
                corr_mat = vecs @ np.diag(vals) @ vecs.T
                
                # Diagonale wieder auf 1 normieren
                d = np.sqrt(np.diag(corr_mat))
                corr_mat = corr_mat / np.outer(d, d)
            else:
                corr_mat = np.array([[1.0]])
            
            # Varianzen hier deutlich breiter machen, damit die Punktewolke zusammenhängt
            stds = rng.uniform(2.0, 4.5, dim)
            S = np.diag(stds)
            
            covs[i] = S @ corr_mat @ S 
        return means, covs

    def append_data(unit_id, time_val, samples, is_treated=False):
        for y in samples:
            row = {'id_col': unit_id, 'time_col': time_val}
            
            # Realistischerer Treatment-Effekt (Kombination aus Skalierung und Verschiebung)
            if is_treated:
                if isinstance(treatment_effect, (int, float)):
                    if dim == 2:
                        # Dimension 1: positive Skalierung + Shift
                        # Dimension 2: Dämpfung/Stauchung + kleinerer Shift
                        scales = np.array([1.15, 0.85])
                        shifts = np.array([treatment_effect, treatment_effect * 0.5])
                        y_adjusted = y * scales + shifts
                    else:
                        y_adjusted = y * 1.1 + treatment_effect
                else:
                    # Falls ein Array/eine Liste übergeben wurde
                    y_adjusted = y + treatment_effect
            else:
                y_adjusted = y
            
            # Die Counterfactual-Verteilung ist der unbehandelte Zustand (y)
            y_cf = y
            
            if dim == 1:
                row['y_col'] = y_adjusted[0]
                row['y_col_cf'] = y_cf[0]
            else:
                for d in range(dim):
                    row[f'y_col_{d+1}'] = y_adjusted[d]
                    row[f'y_col_{d+1}_cf'] = y_cf[d]
            data.append(row)
            
    # Target 
    means_t, covs_t = generate_params(dist_target, is_target=True)
    for t in range(1, num_periods + 1):
        target_data = draw_mix(sample_size, dist_target, means_t, covs_t)
        # Treatment wirkt ab t_treat (inklusive)
        is_treated = (t >= t_treat)
        append_data('0', t, target_data, is_treated=is_treated)
    
    # Controls
    for i in range(num_controls):
        means_c, covs_c = generate_params(dist_control, is_target=False)
        for t in range(1, num_periods + 1):
            c_data = draw_mix(sample_size, dist_control, means_c, covs_c)
            # Controls erhalten nie ein Treatment
            append_data(str(i+1), t, c_data, is_treated=False)

    df = pd.DataFrame(data)
    
    # Reines Counterfactual-DataFrame generieren (ohne Treatment-Effekt)
    cf_df = df.copy()
    if dim == 1:
        cf_df['y_col'] = cf_df['y_col_cf']
    else:
        for d in range(dim):
            cf_df[f'y_col_{d+1}'] = cf_df[f'y_col_{d+1}_cf']
            
    df.attrs['counterfactual'] = cf_df
    
    return df

def generate_dynamic_panel_data(
    num_periods=10, 
    sample_size=1000, 
    num_controls=25, 
    dim=2, 
    num_components=3, 
    ar_coef=0.7, 
    seed=None
):
    """
    Generiert multivariate Panel-Daten basierend auf einem interaktiven Faktormodell.
    Das Target-Profil wird unabhängig von den Kontrollen generiert.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # ---------------------------------------------------------
    # 1. Globale Zeitdynamik (AR(1) Prozess für latente Faktoren)
    # ---------------------------------------------------------
    # f_t hat die Form (num_periods, dim)
    f_t = np.zeros((num_periods, dim))
    f_t[0] = np.random.normal(0, 1, dim)
    for t in range(1, num_periods):
        f_t[t] = ar_coef * f_t[t-1] + np.random.normal(0, 1, dim)
        
    # ---------------------------------------------------------
    # 2. Profile generieren (Controls UND Target unabhängig)
    # ---------------------------------------------------------
    # Basis-Zentren für jede Kontrolle, jede Komponente, jede Dimension
    mu_c = np.random.uniform(-5, 5, (num_controls, num_components, dim))
    # Loadings (Sensibilität auf Schocks) für jede Kontrolle
    lambda_c = np.random.uniform(-2, 2, (num_controls, dim))
    
    # Target-Profil völlig unabhängig ziehen
    mu_t = np.random.uniform(-5, 5, (num_components, dim))
    lambda_t = np.random.uniform(-2, 2, dim)
    
    # Kovarianzmatrizen generieren (geteilt über alle Units pro Komponente)
    covs = np.zeros((num_components, dim, dim))
    for k in range(num_components):
        # Generiere eine strikt positiv definite Matrix
        A = np.random.randn(dim, dim)
        covs[k] = A @ A.T + np.eye(dim) * 0.5 
        
    # ---------------------------------------------------------
    # 3. Hilfsfunktion zum Ziehen der Samples (GMM)
    # ---------------------------------------------------------
    def draw_gmm_samples(n, means, cov_matrices):
        # Gleichverteilte Zuweisung zu den Mixture-Komponenten
        comp_choices = np.random.choice(num_components, size=n)
        out = np.zeros((n, dim))
        for k in range(num_components):
            mask = (comp_choices == k)
            n_k = mask.sum()
            if n_k > 0:
                out[mask] = np.random.multivariate_normal(means[k], cov_matrices[k], size=n_k)
        return out

    # ---------------------------------------------------------
    # 4. Simulation Loop über Zeit und Einheiten
    # ---------------------------------------------------------
    data = []
    
    for t in range(num_periods):
        current_factor = f_t[t]
        
        # Target in Periode t
        # mu_{target, t} = mu_{target} + lambda_{target} * f_t
        mu_t_current = mu_t + (lambda_t * current_factor)[None, :]
        samples_t = draw_gmm_samples(sample_size, mu_t_current, covs)
        
        for y in samples_t:
            row = {'id_col': '0', 'time_col': t}
            if dim == 1:
                row['y_col'] = y[0]
            else:
                for d in range(dim):
                    row[f'y_col_{d+1}'] = y[d]
            data.append(row)
            
        # Controls in Periode t
        for j in range(num_controls):
            mu_c_current = mu_c[j] + (lambda_c[j] * current_factor)[None, :]
            samples_c = draw_gmm_samples(sample_size, mu_c_current, covs)
            
            for y in samples_c:
                row = {'id_col': str(j+1), 'time_col': t}
                if dim == 1:
                    row['y_col'] = y[0]
                else:
                    for d in range(dim):
                        row[f'y_col_{d+1}'] = y[d]
                data.append(row)

    return pd.DataFrame(data)


def create_mdsc_panel_data(sample_size, num_controls, num_periods, dim, ar_coef, 
                           t_treat, apply_treatment=True):
    """
    Simuliert Paneldaten für Distributional Synthetic Controls.
    
    - sample_size: Anzahl der Samples pro (Einheit, Periode)-Knoten.
    - num_controls: Anzahl der Donors in der Donor-Pool.
    - num_periods: Anzahl der Zeitpunkte (T).
    - dim: Dimensionalität der Verteilung (1D, 2D, ...).
    - ar_coef: AR-Matrix (dim x dim) für die Dynamik der Makro-Zustände.
    - t_treat: Zeitpunkt des Treatments.
    - apply_treatment: Wenn False, wird kein Treatment-Effekt simuliert (Placebo-Modus).
    """
    
    # 1 Treatment-Einheit + num_controls Donors
    num_entities = num_controls + 1 
    treated_id = 0  # Wir definieren ID 0 als die behandelte Einheit
    
    ar_coef = np.array(ar_coef)
    if ar_coef.shape != (dim, dim):
        raise ValueError(f"'ar_coef' muss eine quadratische ({dim}x{dim}) Matrix sein.")

    # --- GMM-Parameter (Mischverteilung) ---
    num_components = 2
    weights = [0.7, 0.3] 
    
    # Basis-Mittelwerte und Varianzen der Komponenten (D-dimensional)
    gmm_means = [np.full(dim, -1.0), np.full(dim, 1.5)]
    gmm_covs = [np.eye(dim) * 0.5, np.eye(dim) * 1.2]

    all_data = []
    
    # Namen für die Dimensions-Spalten generieren (y_col_1, y_col_2, ...)
    dim_cols = [f'y_col_{d+1}' for d in range(dim)]

    for entity_id in range(num_entities):
        # 1. Makro-VAR-Prozess für die Dimensionen dieser Einheit
        macro_states = np.zeros((num_periods, dim))
        macro_states[0] = np.random.multivariate_normal(np.zeros(dim), np.eye(dim))
        
        for t in range(1, num_periods):
            macro_shock = np.random.multivariate_normal(np.zeros(dim), np.eye(dim) * 0.2)
            macro_states[t] = ar_coef @ macro_states[t-1] + macro_shock
            
        # 2. Mikro-Samples für jede Periode ziehen
        for t in range(num_periods):
            is_treated_post = (entity_id == treated_id) and (t >= t_treat)
            
            # Ordne die sample_size vielen Beobachtungen den GMM-Komponenten zu
            components = np.random.choice(num_components, size=sample_size, p=weights)
            micro_samples = np.zeros((sample_size, dim))
            
            for comp_idx in range(num_components):
                mask = (components == comp_idx)
                n_comp = np.sum(mask)
                
                if n_comp > 0:
                    current_mean = macro_states[t] + gmm_means[comp_idx].copy()
                    current_cov = gmm_covs[comp_idx].copy()
                    
                    # DISTRIBUTIONAL TREATMENT EFFECT
                    # Wird nur angewendet, wenn apply_treatment=True ist
                    if apply_treatment and is_treated_post and comp_idx == 0:
                        # Verschiebt den Mittelwert stark und erhöht die Varianz (in Dim 1)
                        current_mean[0] += 0.5
                        
                        
                        # Falls wir mind. 2 Dimensionen haben, spillover auf Dim 2 simulieren
                        if dim > 1:
                            current_mean[1] += 0.5
                            current_cov[1, 1] += 0.5
                            current_cov[0, 1] += 0.3
                            current_cov[1, 0] += 0.3
                            current_cov[0, 0] -= 0.2 if current_cov[0, 0] > 0.3 else 0.1
                            
                        
                    micro_samples[mask] = np.random.multivariate_normal(current_mean, current_cov, size=n_comp)
            
            # 3. In DataFrame packen (Jede Dimension bekommt ihre eigene Spalte)
            df_node = pd.DataFrame(micro_samples, columns=dim_cols)
            df_node['time_col'] = t
            df_node['id_col'] = str(entity_id)
            df_node['is_treated_unit'] = 1 if entity_id == treated_id else 0
            df_node['post_treatment'] = 1 if t >= t_treat else 0
            
            all_data.append(df_node)

    # 4. Alles kombinieren
    final_df = pd.concat(all_data, ignore_index=True)
    
    # 5. Spalten in eine schöne Reihenfolge bringen
    final_cols = ['time_col', 'id_col'] + dim_cols + ['is_treated_unit', 'post_treatment']
    
    return final_df[final_cols]





import numpy as np
import pandas as pd
from scipy.stats import norm, t

def generate_multivariate_panel_dgp(N=30, T=10, M=1000, rho=0.5, sigma_alpha=1.0, sigma_gamma=0.5, nu=4, random_state=None):
    """
    Generiert ein synthetisches bivariates Panel-Datenset auf Mikroebene.
    """
    # 1. Lokalen Generator erstellen (akzeptiert Seed-Int oder bestehenden Generator)
    rng = np.random.default_rng(random_state)
    
    total_obs = N * T * M
    
    # Index-Gitter erstellen
    units = np.repeat(np.arange(1, N + 1), T * M)
    times = np.tile(np.repeat(np.arange(1, T + 1), M), N)
    inds = np.tile(np.arange(1, M + 1), N * T)
    
    # 2. Makro-Schocks generieren (jetzt über das 'rng'-Objekt!)
    alpha_1_base = rng.normal(0, sigma_alpha, N)
    alpha_2_base = rng.normal(0, sigma_alpha, N)
    kappa_1_base = rng.uniform(0.5, 1.5, N)
    kappa_2_base = rng.uniform(0.5, 1.5, N)
    
    alpha_1 = alpha_1_base[units - 1]
    alpha_2 = alpha_2_base[units - 1]
    kappa_1 = kappa_1_base[units - 1]
    kappa_2 = kappa_2_base[units - 1]
    
    gamma_1_base = rng.normal(0, sigma_gamma, T)
    gamma_2_base = rng.normal(0, sigma_gamma, T)
    
    gamma_1 = gamma_1_base[times - 1]
    gamma_2 = gamma_2_base[times - 1]
    
    # 3. Mikro-Störterme (Copula-Ansatz)
    cov_matrix = np.array([[1.0, rho], [rho, 1.0]])
    
    # rng.multivariate_normal statt np.random.multivariate_normal
    Z = rng.multivariate_normal(mean=[0, 0], cov=cov_matrix, size=total_obs)
    Z1, Z2 = Z[:, 0], Z[:, 1]
    
    # Transformation in uniforme Ränge
    U1 = norm.cdf(Z1)
    U2 = norm.cdf(Z2)
    
    # WICHTIG: Den Generator an SciPy übergeben (via random_state),
    # falls du dort Zufallszahlen ziehst. Da du hier cdf/ppf nutzt, ist es mathematisch 
    # deterministisch, aber für die Zukunftssicherheit bei t.rvs etc. extrem wichtig:
    eps1 = t.ppf(U1, df=nu)
    eps2 = t.ppf(U2, df=nu)
    
    # 4. Basis-Outcomes berechnen
    Y1_0 = alpha_1 + gamma_1 + kappa_1 * eps1
    Y2_0 = alpha_2 + gamma_2 + kappa_2 * eps2
    
    Y1 = Y1_0.copy()
    Y2 = Y2_0.copy()
    
    # 5. DataFrame erstellen
    df = pd.DataFrame({
        'id_col': units.astype(str),
        'time_col': times,
        'micro_id': inds,
        'y_col_1': Y1,
        'y_col_2': Y2
    })
    
    return df