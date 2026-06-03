import numpy as np
import torch
from ..utils import myQuant
from ..utils.swasserstein import radon_transform
from .quantile1d import Quantile1DSolver, disco_weights_reg

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
        target_s_list = []
        for l in range(self.n_slices):
            target_slice = projected_data[:n_target, l]
            if n_target > 0:
                target_s_list.append(myQuant(target_slice, m_vec))
            else:
                target_s_list.append(np.zeros(M))
        target_s = np.concatenate(target_s_list).reshape((self.n_slices * M, 1))

        controls_s = np.zeros((self.n_slices * M, num_controls))
        for i, ctrl in enumerate(controls):
            n_c = len(ctrl)
            c_s_list = []
            offset = n_target + sum(len(c) for c in controls[:i])
            for l in range(self.n_slices):
                if n_c > 0:
                    c_slice = projected_data[offset : offset + n_c, l]
                    c_s_list.append(myQuant(c_slice, m_vec))
                else:
                    c_s_list.append(np.zeros(M))
            controls_s[:, i] = np.concatenate(c_s_list)

        import cvxpy as cp
        sc = np.linalg.norm(controls_s, ord=2)
        if sc == 0:
            sc = 1.0

        C = controls_s / sc
        d_vec = (target_s / sc).flatten()

        w = cp.Variable(num_controls, nonneg = simplex)
        objective = cp.Minimize(cp.sum_squares(C @ w - d_vec))
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

            radon_result = radon_transform(valid_controls[0], valid_controls[1:], n_slices=n_slices, sort_output=False)
            projected_data = radon_result['projected_data']
            directions = radon_result['directions']
            
            controls_projections = np.zeros((M, n_slices))
            offset = 0
            
            for j, ctrl in enumerate(controls):
                n_c = len(ctrl)
                if n_c > 0:
                    c_block = projected_data[offset : offset + n_c, :]
                    controls_projections += weights[j] * myQuant(c_block, q_vec)
                    offset += n_c
                
            controls_proj_t= torch.tensor(controls_projections, dtype=torch.float32)
            proj_t = torch.tensor(directions, dtype=torch.float32)
            
            Y = torch.tensor(Y_init, dtype=torch.float32, requires_grad=True)
            optimizer = torch.optim.Adam([Y], lr=0.05)
            
            for _ in range(n_iters):
                optimizer.zero_grad()
                y_proj = Y @ proj_t 
                y_proj_sorted, _ = torch.sort(y_proj, dim=0)
                loss = torch.mean((y_proj_sorted - controls_proj_t)**2)
                loss.backward()
                optimizer.step()
                
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
        radon_result = radon_transform(target, controls, n_slices=self.n_slices, sort_output=False)
        projected_data = radon_result['projected_data']

        M = kwargs.get("M", 500)
        q_min = kwargs.get("q_min", 0)
        q_max = kwargs.get("q_max", 1)
        m_vec = np.linspace(q_min, q_max, M)

        dist = 0.0
        n_target = len(target)
        
        for l in range(self.n_slices):
            target_slice = projected_data[:n_target, l]
            if n_target > 0:
                target_q = myQuant(target_slice, m_vec)
            else:
                target_q = np.zeros(M)

            ctrl_q_list = []
            offset = n_target
            for ctrl in controls:
                n_c = len(ctrl)
                if n_c > 0:
                    c_slice = projected_data[offset : offset + n_c, l]
                    ctrl_q_list.append(myQuant(c_slice, m_vec))
                else:
                    ctrl_q_list.append(np.zeros(M))
                offset += n_c
            
            controls_q_stacked = np.column_stack(ctrl_q_list)
            bc_q = controls_q_stacked @ weights
            
            dist += np.mean((bc_q - target_q)**2)

        return dist / self.n_slices
