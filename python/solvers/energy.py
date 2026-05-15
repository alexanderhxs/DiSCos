import numpy as np
import cvxpy as cp
import scoringrules as sr
from .base import BaseSolver

class EnergySolver(BaseSolver):
    def fit_weights(self, target, controls, **kwargs):
        """
        target: numpy array der Form (n, d) - n Samples, d Dimensionen
        controls: Liste von arrays oder array der Form (J, n, d) - J Modelle
        """
        # Sicherstellen, dass controls ein 3D-Array der Form (J, n, d) ist
        if isinstance(controls, list):
            controls = np.array(controls)
            
        J, n, d = controls.shape
        
        # Um target von (n, d) von jedem Modell in (J, n, d) abziehen zu können,
        # fügen wir eine Dimension für Broadcasting hinzu -> (1, n, d)
        target_expanded = np.expand_dims(target, axis=0)
        
        # ---------------------------------------------------------
        # 1. Lineare Distanz zur Wahrheit (Vektor A) berechnen
        # ---------------------------------------------------------
        # Euklidische Distanz über die Feature-Dimension (axis=-1)
        err_norm = np.linalg.norm(controls - target_expanded, axis=-1)  # Shape: (J, n)
        # Durchschnitt über alle n Samples (Zeitpunkte/Beobachtungen)
        A = np.mean(err_norm, axis=-1)  # Shape: (J,)
        
        # ---------------------------------------------------------
        # 2. Streuung der Modelle untereinander (Matrix D) berechnen
        # ---------------------------------------------------------
        # Cross-Differences durch Broadcasting erzeugen -> Shape: (J, J, n, d)
        spread_diff = np.expand_dims(controls, axis=1) - np.expand_dims(controls, axis=0)
        spread_norm = np.linalg.norm(spread_diff, axis=-1)  # Shape: (J, J, n)
        # Durchschnitt über alle n Samples
        D = np.mean(spread_norm, axis=-1)  # Shape: (J, J)
        
        # ---------------------------------------------------------
        # 3. Der CVXPY PSD-Trick (Positive Semi-Definite)
        # ---------------------------------------------------------
        # Die Energy Divergence zieht die Streuung ab: Loss = w^T A - 0.5 * w^T D w
        # CVXPY benötigt für cp.quad_form eine positiv semi-definite Matrix.
        H = -0.5 * D
        
        # Wir berechnen die Eigenwerte, um H zu shiften
        eigenvalues = np.linalg.eigvalsh(H)
        min_eig = np.min(eigenvalues)
        
        if min_eig < 0:
            # Da sum(w) = 1 gilt, können wir eine Konstante auf alle Felder addieren,
            # ohne das Minimum auf dem Simplex zu verschieben.
            gamma = -min_eig + 1e-6 
            H_psd = H + gamma * np.ones((J, J))
        else:
            H_psd = H
        
        scaling_factor = np.max(np.abs(H_psd)) if np.max(np.abs(H_psd)) > 0 else 1.0
        H_psd /= scaling_factor
        A /= scaling_factor
        

        # ---------------------------------------------------------
        # 4. CVXPY Optimierung
        # ---------------------------------------------------------
        simplex = kwargs.get("simplex", True) 
        w = cp.Variable(J, nonneg=simplex)  
        
        # Zielfunktion: Linearer Term + Quadratischer Term
        objective = cp.Minimize(A @ w + cp.quad_form(w, H_psd))
        
        # Nebenbedingungen (Constraints)    
        constraints = [cp.sum(w) == 1]

        prob = cp.Problem(objective, constraints)
        
        # OSQP ist für QP-Probleme meist schneller und robuster als SCS
        prob.solve(solver=cp.OSQP, max_iter=10000)
        
        # Fallback auf SCS, falls OSQP (aus numerischen Gründen) scheitert
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            prob.solve(solver=cp.SCS, max_iters=10000, eps=1e-5)
            
        weights_opt = w.value
        
        # ---------------------------------------------------------
        # 5. Fallbacks und Bereinigung (Clean-up)
        # ---------------------------------------------------------
        if weights_opt is None:
            # Wenn alles fehlschlägt, Gleichverteilung zurückgeben
            weights_opt = np.ones(J) / J
            
        if simplex:
            # Numerische Ungenauigkeiten vom Solver bereinigen (z.B. -1e-18 -> 0)
            weights_opt = np.clip(weights_opt, 0, 1)
            # Exakt auf 1 normalisieren
            weights_opt /= np.sum(weights_opt) 
            
        return weights_opt
    
    def evaluate_counterfactual(self, controls, weights, **kwargs):

        grid_ord = kwargs.get("grid_ord")
        evgrid = kwargs.get("evgrid")

        from ..utils import sample_counterfactual_distribution
        counterfactual = sample_counterfactual_distribution(controls, weights, grid_ord)
        
        if counterfactual is not None and grid_ord is not None:
            cf_sq = np.squeeze(counterfactual)
            
            if cf_sq.ndim == 1:
                cf_sorted = np.sort(cf_sq)
                disco_cdf = np.searchsorted(cf_sorted, grid_ord, side='right') / len(cf_sq)
            else:
                disco_cdf = np.mean(np.all(counterfactual[None, :, :] <= grid_ord[:, None, :], axis=2), axis=1)
                
            if cf_sq.ndim == 1 or (cf_sq.ndim == 2 and cf_sq.shape[1] == 1):
                from ..utils import myQuant
                disco_quantile = myQuant(cf_sq, evgrid) if evgrid is not None else None
            else:
                disco_quantile = None
        else:
            disco_cdf = None
            disco_quantile = None

        return {
            "disco_quantile": disco_quantile,
            "disco_cdf": disco_cdf
        }
        
    def compute_distance(self, target, controls, weights, **kwargs):
        if weights is None or len(weights) == 0:
            return np.nan
            
        if isinstance(controls, list):
            # Filtern von leeren Controls
            valid_controls = [c for c in controls if len(c) > 0]
            if len(valid_controls) == 0:
                return np.nan
            valid_idx = [i for i, c in enumerate(controls) if len(c) > 0]
            if len(valid_idx) < len(weights):
                weights = weights[valid_idx] / np.sum(weights[valid_idx])
            controls = np.array(valid_controls)
            
        if target is None or len(target) == 0:
            return np.nan
            
        target = np.asarray(target)
        if target.ndim == 1:
            target = target[:, None]
        if controls.ndim == 2:
            controls = controls[:, :, None]

        # ---------------------------------------------------------
        # Energy Distance mittels der Vektoren und Matrizen aus fit_weights interpolieren
        # ED = E|X_c - X_t| - 0.5 * E|X_c - X_c'| - 0.5 * E|X_t - X_t'|
        # ---------------------------------------------------------
        target_expanded = np.expand_dims(target, axis=0)
        err_norm = np.linalg.norm(controls - target_expanded, axis=-1)
        A = np.mean(err_norm, axis=-1)
        
        spread_diff = np.expand_dims(controls, axis=1) - np.expand_dims(controls, axis=0)
        spread_norm = np.linalg.norm(spread_diff, axis=-1)
        D = np.mean(spread_norm, axis=-1)
        
        target_diff = np.expand_dims(target, axis=1) - np.expand_dims(target, axis=0)
        target_norm = np.linalg.norm(target_diff, axis=-1)
        target_spread = np.mean(target_norm)
        
        energy_dist = (A @ weights) - 0.5 * (weights.T @ D @ weights) - 0.5 * target_spread
        return float(energy_dist)