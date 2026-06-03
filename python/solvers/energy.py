import numpy as np
import cvxpy as cp
import scoringrules as sr
from .base import BaseSolver

class EnergySolver(BaseSolver):
    def fit_weights(self, target, controls, **kwargs):
        """
        target: numpy array der Form (n_t, d) - n_t Samples, d Dimensionen pro Periode
        controls: Liste der Länge J, wobei Element controls[j] ein Array der Form (n_j, d) ist
        """
        J = len(controls) # J: Anzahl der Kontroll-Einheiten
        target = np.asarray(target)
        if target.size > 0 and target.ndim == 1:
            target = target[:, None] # target wird zu Shape (n_t, 1) falls 1D
        n_t = target.shape[0] if target.size > 0 else 0 # n_t: Anzahl Observations im Target
        d = target.shape[1] if target.size > 0 else 1   # d: Dimension des Features

        A = np.zeros(J) # Shape: (J,) - Energy dist Target -> jede Control
        D = np.zeros((J, J)) # Shape: (J, J) - Energy dist Controls untereinander
        valid_mask = np.ones(J, dtype=bool) # Shape: (J,)

        proc_controls = [] # Liste (Länge J) von validen Control-Arrays (Shape: (n_j, d))
        for j in range(J):
            c = np.asarray(controls[j])
            if c.size == 0:
                valid_mask[j] = False
                proc_controls.append(np.empty((0, d)))
            else:
                if c.ndim == 1:
                    c = c[:, None]
                proc_controls.append(c)

        for j in range(J):
            if not valid_mask[j]:
                continue
            c_j = proc_controls[j] # Shape: (n_j, d)
            if n_t > 0:
                # Broadcasting von c_j (n_j, 1, d) und target (1, n_t, d) -> diff Shape: (n_j, n_t, d)
                diff = c_j[:, None, :] - target[None, :, :]
                A[j] = np.mean(np.linalg.norm(diff, axis=-1)) # A[j] ist ein Skalar

            for k in range(j, J):
                if not valid_mask[k]:
                    continue
                c_k = proc_controls[k] # Shape: (n_k, d)
                # Broadcasting von c_j (n_j, 1, d) und c_k (1, n_k, d) -> diff Shape: (n_j, n_k, d)
                diff = c_j[:, None, :] - c_k[None, :, :]
                val = np.mean(np.linalg.norm(diff, axis=-1)) # val ist ein Skalar
                D[j, k] = val
                D[k, j] = val
        
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
        # Verwende cp.psd_wrap, um numerische Ungenauigkeiten bei der PSD-Prüfung von CVXPY zu umgehen
        objective = cp.Minimize(A @ w + cp.quad_form(w, cp.psd_wrap(H_psd)))
        
        # Nebenbedingungen (Constraints)    
        constraints = [cp.sum(w) == 1]
        for j in range(J):
            if not valid_mask[j]:
                constraints.append(w[j] == 0)

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
        # counterfactual: Array mit resampelten/gepoolten Werten, typisches Shape (N_pool, d)
        counterfactual = sample_counterfactual_distribution(controls, weights, grid_ord)
        
        if counterfactual is not None and grid_ord is not None:
            cf_sq = np.squeeze(counterfactual) # Squeezed Array, z.B. Shape (N_pool,) falls d=1 oder (N_pool, d) falls d>1
            
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
        """
        target: array mit Shape (n_t, d)
        controls: Liste der Länge J, Elemente haben Shape (n_j, d)
        weights: array mit Shape (J,)
        """
        if weights is None or len(weights) == 0:
            return np.nan
        
        target = np.asarray(target)
        if target.size == 0:
            return np.nan
        if target.ndim == 1:
            target = target[:, None] # target Shape: (n_t, 1)
        n_t = target.shape[0] # n_t: Anzahl Target Observations
        d = target.shape[1]   # d: Feature-Dimensionen
        
        J = len(controls) # J: Anzahl Kontrollgruppen
        valid_mask = np.ones(J, dtype=bool) # Shape: (J,)
        proc_controls = [] # Liste (Länge J), Elemente Shape: (n_j, d)
        
        for j in range(J):
            c = np.asarray(controls[j])
            if c.size == 0:
                valid_mask[j] = False
                proc_controls.append(np.empty((0, d)))
            else:
                if c.ndim == 1:
                    c = c[:, None]
                proc_controls.append(c)

        if not np.any(valid_mask):
            return np.nan

        # Adjust weights for missing out-of-sample controls
        weights = np.array(weights, dtype=float)
        if not np.all(valid_mask):
            valid_w_sum = np.sum(weights[valid_mask])
            if valid_w_sum > 0:
                weights[~valid_mask] = 0.0
                weights /= valid_w_sum
            else:
                return np.nan

        A = np.zeros(J)
        D = np.zeros((J, J))

        for j in range(J):
            if not valid_mask[j] or weights[j] == 0:
                continue
            c_j = proc_controls[j]
            diff = c_j[:, None, :] - target[None, :, :]
            A[j] = np.mean(np.linalg.norm(diff, axis=-1))

            for k in range(j, J):
                if not valid_mask[k] or weights[k] == 0:
                    continue
                c_k = proc_controls[k]
                diff = c_j[:, None, :] - c_k[None, :, :]
                val = np.mean(np.linalg.norm(diff, axis=-1))
                D[j, k] = val
                D[k, j] = val
        
        target_diff = target[:, None, :] - target[None, :, :]
        target_spread = np.mean(np.linalg.norm(target_diff, axis=-1))
        
        energy_dist = (A @ weights) - 0.5 * (weights.T @ D @ weights) - 0.5 * target_spread
        return float(energy_dist)