import numpy as np
import cvxpy as cp
from ..utils.utils import myQuant
from .base import BaseSolver

def disco_weights_reg(controls, target, M=500, simplex=False, q_min=0, q_max=1):
    num_controls = len(controls)
    
    m_vec = np.random.uniform(q_min, q_max, M)

    if isinstance(controls, list) or controls.ndim == 2:
        controls_s = np.zeros((M, num_controls))
        for i, ctrl in enumerate(controls):
            controls_s[:, i] = myQuant(ctrl, m_vec)
        target_s = myQuant(target, m_vec).reshape((M, 1))
    else:
        controls_s = np.zeros((M, controls.shape[-1], num_controls))
        for i, ctrl in enumerate(controls):
            controls_s[:, :, i] = myQuant(ctrl, m_vec)
        target_s = myQuant(target, m_vec).reshape((M, controls_s.shape[1], 1))

    if controls_s.ndim == 3:
        C = controls_s.reshape(-1, num_controls)
    else:
        C = controls_s
        
    sc = np.linalg.norm(C, ord=2) 
    if np.abs(sc) < 1e-9:
        sc = 1.0  
        
    C = C / sc
    d_vec = target_s.flatten() / sc
    
    w = cp.Variable(num_controls)
    objective = cp.Minimize(cp.sum_squares(C @ w - d_vec))
    
    constraints = [cp.sum(w) == 1]
    if simplex:
        constraints.append(w >= 0)
        
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, max_iters=10000, eps=1e-5)
    
    return w.value

class Quantile1DSolver(BaseSolver):
    def fit_weights(self, target, controls, M=500, simplex=False, q_min=0, q_max=1, **kwargs):
        return disco_weights_reg(controls, target, M, simplex, q_min, q_max)    
        
    def evaluate_counterfactual(self,  controls, weights, **kwargs):
        grid_ord = kwargs.get("grid_ord")
        controls_q = kwargs.get("controls_q", np.array([]))

        if len(controls_q) > 0 and weights is not None:
            disco_quantile = controls_q @ weights
            bc_sorted = np.sort(disco_quantile)
            disco_cdf = np.searchsorted(bc_sorted, grid_ord, side='right') / len(bc_sorted)
        else:
            disco_quantile = None
            disco_cdf = None

        return {
            "disco_quantile": disco_quantile,
            "disco_cdf": disco_cdf
        }

    def compute_distance(self, target, controls, weights, **kwargs):
        dist = 0
        target_q = kwargs.get("target_q")
        if target_q is not None and weights is not None:
            controls_q = kwargs.get("controls_q")
            bc_q = controls_q @ weights
            dist = np.mean((bc_q - target_q)**2)
        return dist
