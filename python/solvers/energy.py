import numpy as np
import cvxpy as cp
#import scoringrules as sr
from scipy.spatial.distance import cdist
from .base import BaseSolver

class EnergySolver(BaseSolver):

    # ---------------------------------------------------------
    # Hilfsfunktionen
    # ---------------------------------------------------------
    @staticmethod
    def _preprocess_controls(controls, d):
        """Preprocesse Controls: 1D→2D konvertieren, leere markieren.
        
        Returns:
            proc_controls: Liste (Länge J) von Arrays mit Shape (n_j, d)
            valid_mask: boolean Array mit Shape (J,)
        """
        J = len(controls)
        valid_mask = np.ones(J, dtype=bool)
        proc_controls = []
        for j in range(J):
            c = np.asarray(controls[j])
            if c.size == 0:
                valid_mask[j] = False
                proc_controls.append(np.empty((0, d)))
            else:
                if c.ndim == 1:
                    c = c[:, None]
                proc_controls.append(c)
        return proc_controls, valid_mask

    @staticmethod
    def _compute_energy_matrices(target, proc_controls, valid_mask):
        """Berechne die Energy-Distance-Matrizen A und D.
        
        Verwendet scipy.spatial.distance.cdist statt 3D-Broadcasting,
        um den Speicherverbrauch von O(n_j * n_k * d) auf O(n_j * n_k) zu reduzieren.
        
        Args:
            target: Array mit Shape (n_t, d) oder None
            proc_controls: Liste von Arrays mit Shape (n_j, d)
            valid_mask: boolean Array mit Shape (J,)
            
        Returns:
            A: Array mit Shape (J,) — mean pairwise distance Control j ↔ Target
            D: Array mit Shape (J, J) — mean pairwise distance Control j ↔ Control k
        """
        J = len(proc_controls)
        A = np.zeros(J)
        D = np.zeros((J, J))
        
        n_t = target.shape[0] if target is not None and target.size > 0 else 0
        
        for j in range(J):
            if not valid_mask[j]:
                continue
            c_j = proc_controls[j]  # Shape: (n_j, d)
            
            if n_t > 0:
                # cdist berechnet die (n_j, n_t) Distanzmatrix direkt,
                # ohne den (n_j, n_t, d) Zwischentensor zu allokieren
                A[j] = np.mean(cdist(c_j, target, metric='euclidean'))
            
            for k in range(j, J):
                if not valid_mask[k]:
                    continue
                c_k = proc_controls[k]  # Shape: (n_k, d)
                val = np.mean(cdist(c_j, c_k, metric='euclidean'))
                D[j, k] = val
                D[k, j] = val
        
        return A, D

    @staticmethod
    def _make_psd_and_scale(H, A):
        """Mache H positiv semi-definit (PSD-Trick) und skaliere H und A gemeinsam.
        
        Da sum(w) = 1 auf dem Simplex, verschiebt das Addieren einer Konstanten
        auf alle Einträge von H das Minimum nicht.
        
        Args:
            H: Quadratische Matrix mit Shape (J, J), wird modifiziert zu PSD
            A: Linearer Koeffizientenvektor mit Shape (J,)
            
        Returns:
            H_psd: PSD-Matrix mit Shape (J, J), skaliert
            A_scaled: Skalierter linearer Koeffizientenvektor mit Shape (J,)
        """
        J = H.shape[0]
        eigenvalues = np.linalg.eigvalsh(H)
        min_eig = np.min(eigenvalues)
        
        if min_eig < 0:
            gamma = -min_eig + 1e-6
            H_psd = H + gamma * np.ones((J, J))
        else:
            H_psd = H.copy()
        
        scaling_factor = np.max(np.abs(H_psd)) if np.max(np.abs(H_psd)) > 0 else 1.0
        H_psd /= scaling_factor
        A_scaled = A / scaling_factor
        
        return H_psd, A_scaled

    @staticmethod
    def _solve_qp(A, H_psd, J, valid_mask, simplex=True):
        """Löse das quadratische Programm: min A@w + w^T H_psd w, s.t. sum(w)=1.
        
        Returns:
            weights_opt: Array mit Shape (J,)
        """
        w = cp.Variable(J, nonneg=simplex)
        
        objective = cp.Minimize(A @ w + cp.quad_form(w, cp.psd_wrap(H_psd)))
        
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
        
        if weights_opt is None:
            weights_opt = np.ones(J) / J
        
        if simplex:
            weights_opt = np.clip(weights_opt, 0, 1)
            weights_opt /= np.sum(weights_opt)
        
        return weights_opt

    # ---------------------------------------------------------
    # Hauptmethoden
    # ---------------------------------------------------------
    def fit_weights(self, target, controls, **kwargs):
        """
        target: numpy array der Form (n_t, d) - n_t Samples, d Dimensionen pro Periode
        controls: Liste der Länge J, wobei Element controls[j] ein Array der Form (n_j, d) ist
        """
        J = len(controls)
        target = np.asarray(target)
        if target.size > 0 and target.ndim == 1:
            target = target[:, None]
        d = target.shape[1] if target.size > 0 else 1

        proc_controls, valid_mask = self._preprocess_controls(controls, d)
        A, D = self._compute_energy_matrices(target, proc_controls, valid_mask)
        
        # ---------------------------------------------------------
        # PSD-Trick und Optimierung
        # ---------------------------------------------------------
        # Die Energy Divergence: Loss = w^T A - 0.5 * w^T D w
        H = -0.5 * D
        simplex = kwargs.get("simplex", True)
        H_psd, A_scaled = self._make_psd_and_scale(H, A)
        weights_opt = self._solve_qp(A_scaled, H_psd, J, valid_mask, simplex)
            
        return weights_opt
    
    def evaluate_counterfactual(self, controls, weights, **kwargs):

        grid_ord = kwargs.get("grid_ord")
        evgrid = kwargs.get("evgrid")

        from utils import sample_counterfactual_distribution
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
            target = target[:, None]
        
        J = len(controls)
        d = target.shape[1]
        proc_controls, valid_mask = self._preprocess_controls(controls, d)

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

        # Verwende gemeinsame Hilfsfunktion mit cdist
        A, D = self._compute_energy_matrices(target, proc_controls, valid_mask)
        
        # Target-Spread: E[||Y - Y'||] — verwende cdist für Konsistenz
        target_spread = np.mean(cdist(target, target, metric='euclidean'))
        
        energy_dist = (A @ weights) - 0.5 * (weights.T @ D @ weights) - 0.5 * target_spread
        return float(energy_dist)
    

    def _compute_joint_energy_matrices(self, targets_list, controls_list):
        """Berechne aggregierte Energy-Matrizen über mehrere Zeitperioden.
        
        Gemeinsame Logik für fit_weights_joint und second_level_fit.
        
        Args:
            targets_list: Liste von Target-Arrays, Länge T0
            controls_list: Liste von Control-Listen, Länge J, jeweils Subliste Länge T0
            
        Returns:
            A_total: Aggregierter linearer Term, Shape (J,)
            D_total: Aggregierte Distanzmatrix, Shape (J, J)
        """
        T0 = len(targets_list)
        J = len(controls_list)
        
        A_total = np.zeros(J)
        D_total = np.zeros((J, J))
        
        for t in range(T0):
            target_t = np.asarray(targets_list[t])
            controls_t = [controls_list[j][t] for j in range(J)]
            
            if target_t.size > 0 and target_t.ndim == 1:
                target_t = target_t[:, None]
            
            d = target_t.shape[1] if target_t.size > 0 else 1
            
            proc_controls, valid_mask = self._preprocess_controls(controls_t, d)
            A_t, D_t = self._compute_energy_matrices(target_t, proc_controls, valid_mask)
            
            A_total += A_t
            D_total += D_t
        
        return A_total, D_total

    def fit_weights_joint(self, targets_list, controls_list, **kwargs):
        T0 = len(targets_list)
        J = len(controls_list)
        
        A_total, D_total = self._compute_joint_energy_matrices(targets_list, controls_list)
            
        # ---------------------------------------------------------
        # PSD-Trick mit aggregierten Matrizen
        # ---------------------------------------------------------
        H = -0.5 * D_total
        simplex = kwargs.get("simplex", True)
        
        # Maske: Einheiten die über alle Perioden komplett fehlen
        all_zero_mask = np.ones(J, dtype=bool)
        for j in range(J):
            if np.all(D_total[j, :] == 0) and A_total[j] == 0:
                all_zero_mask[j] = False
        
        H_psd, A_scaled = self._make_psd_and_scale(H, A_total)
        weights_opt = self._solve_qp(A_scaled, H_psd, J, all_zero_mask, simplex)
            
        return weights_opt
    
    def second_level_fit(self, weights, targets_list, controls_list):
        
        T0 = len(targets_list)
        J = len(controls_list)
        
        A_total, D_total = self._compute_joint_energy_matrices(targets_list, controls_list)
            
        # ---------------------------------------------------------
        # PSD-Trick mit aggregierten Matrizen
        # ---------------------------------------------------------
        H = -0.5 * D_total
        H_psd, A_scaled = self._make_psd_and_scale(H, A_total)
        
        # ---------------------------------------------------------
        # CVXPY Optimierung: Konvexkombination der Perioden-Gewichte
        # ---------------------------------------------------------
        weigths = np.column_stack(weights)  # Shape: (J, T0)

        lamb = cp.Variable(T0, nonneg=True)  
        
        objective = cp.Minimize(A_scaled @ (weigths @ lamb) + cp.quad_form(weigths @ lamb, cp.psd_wrap(H_psd)))
        
        constraints = [cp.sum(lamb) == 1]
        
        # Prüfe ob bestimmte Perioden keine Daten haben (alle Controls + Target leer)
        # Hinweis: lamb hat Dimension T0, nicht J!
        for t in range(T0):
            target_t = np.asarray(targets_list[t])
            if target_t.size == 0:
                constraints.append(lamb[t] == 0)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, max_iter=10000)
        
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            prob.solve(solver=cp.SCS, max_iters=10000, eps=1e-5)
            
        weights_opt = lamb.value
        
        # ---------------------------------------------------------
        # Clean-up: Fallback hat Dimension T0 (nicht J!)
        # ---------------------------------------------------------
        if weights_opt is None:
            weights_opt = np.ones(T0) / T0
            
        weights_opt = np.clip(weights_opt, 0, 1)
        weights_opt /= np.sum(weights_opt) 
            
        return weights_opt
