import numpy as np
import torch
from utils import myQuant
from utils.swasserstein import radon_transform
from .quantile1d import Quantile1DSolver, disco_weights_reg

def fast_quantile(X, q_vec):
    """Highly optimized vectorized quantile calculation for sorted 1D projections."""
    N = X.shape[0]
    if N == 0:
        return np.zeros((len(q_vec), X.shape[1]))
    X_s = np.sort(X, axis=0)
    idx = q_vec * (N - 1)
    i0 = np.floor(idx).astype(int)
    i1 = np.ceil(idx).astype(int)
    w = (idx - i0)[:, None]
    return X_s[i0, :] * (1 - w) + X_s[i1, :] * w

class SlicedWassersteinSolver(Quantile1DSolver):
    def __init__(self, n_slices=1000):
        super().__init__()
        self.n_slices = n_slices

    def fit_weights(self, target, controls, **kwargs):
        num_controls = len(controls)
        M = kwargs.get("M", 500)
        simplex = kwargs.get("simplex", True)
        
        radon_result = radon_transform(target, controls, n_slices=self.n_slices, sort_output=False)
        projected_data = radon_result['projected_data']
        

        
        q_min = kwargs.get("q_min", 0.0)
        q_max = kwargs.get("q_max", 1.0)
        m_vec = np.linspace(q_min, q_max, M)
        
        n_target = len(target)
        
        # Vektorisierte Quantilberechnung über alle Slices gleichzeitig
        # statt Schleife über n_slices mit einzelnen myQuant-Aufrufen
        target_projected = projected_data[:n_target, :]  # Shape: (n_target, L)
        if n_target > 0:
            target_s_2d = fast_quantile(target_projected, m_vec)  # Shape: (M, L)
            target_s = target_s_2d.T.flatten().reshape((-1, 1))  # Shape: (M*L, 1)
        else:
            target_s = np.zeros((self.n_slices * M, 1))

        controls_s = np.zeros((self.n_slices * M, num_controls))
        for i, ctrl in enumerate(controls):
            n_c = len(ctrl)
            offset = n_target + sum(len(c) for c in controls[:i])
            if n_c > 0:
                ctrl_projected = projected_data[offset : offset + n_c, :]  # Shape: (n_c, L)
                # Vektorisiert: Quantile über alle Slices gleichzeitig
                ctrl_q_2d = fast_quantile(ctrl_projected, m_vec)  # Shape: (M, L)
                controls_s[:, i] = ctrl_q_2d.T.flatten()
            # else: bleibt Null (default)

        import cvxpy as cp
        sc = np.max(np.abs(controls_s))
        if sc == 0:
            sc = 1.0

        C = controls_s / sc
        d_vec = (target_s / sc).flatten()

        H = C.T @ C
        H = 0.5 * (H + H.T)
        H += np.eye(num_controls) * 1e-8 # Ensure strict PSD
        f = -C.T @ d_vec

        w = cp.Variable(num_controls, nonneg = simplex)
        
        try:
            objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(H)) + 2 * f.T @ w)
        except AttributeError:
            objective = cp.Minimize(cp.quad_form(w, H) + 2 * f.T @ w)
            
        constraints = [cp.sum(w) == 1] 
        
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.SCS)
        except:
            prob.solve()

        return w.value 
    
    def evaluate_counterfactual(self, controls, weights, **kwargs):
        grid_ord = kwargs.get('grid_ord')

        if weights is not None:
            dim = controls[0].shape[1]
            M = kwargs.get('M', 500)
            n_iters = 500
            n_slices = kwargs.get('n_slices', self.n_slices)
            
            q_vec = np.linspace(0, 1, M)
            Y_init = np.zeros((M, dim))
            for j, ctrl in enumerate(controls):
                if len(ctrl) > 0:
                    for d in range(dim):
                        Y_init[:, d] += weights[j] * myQuant(ctrl[:, d], q_vec)

            valid_controls = [c for c in controls if len(c) > 0]
            if not valid_controls:
                return {"disco_quantile": None, "disco_cdf": None}

            # Fix: Alle valid Controls zusammen projizieren, nicht die erste als Target verwenden
            # Wir erstellen einen Dummy-Target (erste Control) und packen den Rest als Controls
            # So werden alle Daten korrekt in projected_data zusammengefasst
            all_ctrl_data = np.concatenate(valid_controls, axis=0)
            radon_result = radon_transform(all_ctrl_data, [], n_slices=n_slices, sort_output=False)
            projected_data = radon_result['projected_data']
            directions = radon_result['directions']
            
            # Gewichtete Quantile pro Slice berechnen
            controls_projections = np.zeros((M, n_slices))
            offset = 0
            
            ctrl_idx = 0
            for j, ctrl in enumerate(controls):
                n_c = len(ctrl)
                if n_c > 0:
                    c_block = projected_data[offset : offset + n_c, :]  # Shape: (n_c, n_slices)
                    # Vektorisierte Quantilberechnung
                    c_quantiles = fast_quantile(c_block, q_vec)  # Shape: (M, n_slices)
                    controls_projections += weights[j] * c_quantiles
                    offset += n_c
                
            controls_proj_t = torch.tensor(controls_projections, dtype=torch.float32)
            proj_t = torch.tensor(directions, dtype=torch.float32)
            
            Y = torch.tensor(Y_init, dtype=torch.float32, requires_grad=True)
            optimizer = torch.optim.Adam([Y], lr=0.05)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
            
            prev_loss = float('inf')
            for iteration in range(n_iters):
                optimizer.zero_grad()
                y_proj = Y @ proj_t 
                y_proj_sorted, _ = torch.sort(y_proj, dim=0)
                loss = torch.mean((y_proj_sorted - controls_proj_t)**2)
                loss.backward()
                optimizer.step()
                
                # Early Stopping: wenn Loss-Änderung < 1e-6
                current_loss = loss.item()
                scheduler.step(current_loss)
                if abs(prev_loss - current_loss) < 1e-6:
                    break
                prev_loss = current_loss
                
            Y_opt = Y.detach().numpy()
            
            if grid_ord is not None and len(grid_ord) > 0:
                disco_cdf = np.mean(np.all(Y_opt[None, :, :] <= grid_ord[:, None, :], axis=2), axis=1)
            else:
                disco_cdf = None
        else:
            disco_cdf = None

        return {
            "disco_quantile": None,
            "disco_cdf": disco_cdf
        }
    
    def compute_distance(self, target, controls, weights, **kwargs):
        # Stateless: Frische Directions generieren. Der SWD-Schätzer konvergiert
        # mit O(1/sqrt(L)) — bei L=1000 ist die Varianz durch verschiedene
        # Directions-Sets vernachlässigbar gegenüber der Stichprobenvarianz.
        radon_result = radon_transform(target, controls, n_slices=self.n_slices, 
                                        sort_output=False)
        projected_data = radon_result['projected_data']

        M = kwargs.get("M", 500)
        q_min = kwargs.get("q_min", 0)
        q_max = kwargs.get("q_max", 1)
        m_vec = np.linspace(q_min, q_max, M)

        n_target = len(target)
        
        # Vektorisierte Quantilberechnung für Target
        target_projected = projected_data[:n_target, :]  # Shape: (n_target, L)
        if n_target > 0:
            target_q_2d = fast_quantile(target_projected, m_vec)  # Shape: (M, L)
        else:
            target_q_2d = np.zeros((M, self.n_slices))

        # Vektorisierte Quantilberechnung für Controls + gewichtete Kombination
        bc_q_2d = np.zeros((M, self.n_slices))
        offset = n_target
        for i, ctrl in enumerate(controls):
            n_c = len(ctrl)
            if n_c > 0:
                ctrl_projected = projected_data[offset : offset + n_c, :]
                ctrl_q_2d = fast_quantile(ctrl_projected, m_vec)  # Shape: (M, L)
                bc_q_2d += weights[i] * ctrl_q_2d
            offset += n_c

        # Mittlere quadrierte Differenz über alle Quantile und Slices
        dist = np.mean((bc_q_2d - target_q_2d)**2)
        return dist
