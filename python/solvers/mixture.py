import numpy as np
import cvxpy as cp
from .base import BaseSolver

def disco_mixture(controls, target, grid_min, grid_max, grid_rand, M, simplex):
    num_controls = len(controls)
    cdf_matrix = np.zeros((len(grid_rand), num_controls + 1))
    target = np.squeeze(target)

    if target.ndim == 1:
        target_sorted = np.sort(target)
        cdf_matrix[:, 0] = np.searchsorted(target_sorted, grid_rand, side='right') / len(target)

        for i, ctrl in enumerate(controls):
            ctrl = np.squeeze(ctrl)
            ctrl_sorted = np.sort(ctrl)
            cdf_matrix[:, i+1] = np.searchsorted(ctrl_sorted, grid_rand, side='right') / len(ctrl)
    else:
        cdf_matrix[:, 0] = np.mean(np.all(target[None, :, :] <= grid_rand[:, None, :], axis=2), axis=1)

        for i, ctrl in enumerate(controls):
            ctrl = np.atleast_2d(ctrl)
            cdf_matrix[:, i+1] = np.mean(np.all(ctrl[None, :, :] <= grid_rand[:, None, :], axis=2), axis=1)
    
    w = cp.Variable(num_controls)
    obj = cp.Minimize(cp.norm1(cdf_matrix[:, 1:] @ w - cdf_matrix[:, 0]))
    
    if simplex:
        constraints = [w >= 0, cp.sum(w) == 1]
    else:
        constraints = [cp.sum(w) == 1]
        
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, max_iters=10000, eps=1e-5)
    
    weights_opt = w.value
    if weights_opt is None:
        weights_opt = np.ones(num_controls) / num_controls
        
    distance_opt = prob.value * (1/M) * (grid_max - grid_min)
    mean_val = cdf_matrix[:, 1:] @ weights_opt
    
    order_idx = np.argsort(grid_rand)
    mean_order = mean_val[order_idx]
    target_order = cdf_matrix[:, 0][order_idx]
    
    return {
        "weights_opt": weights_opt,
        "distance_opt": distance_opt,
        "mean": mean_order,
        "target_order": target_order,
        "cdf": cdf_matrix
    }

class MixtureSolver(BaseSolver):
    def fit_weights(self, target, controls, grid_min=None, grid_max=None, grid_rand=None, M=500, simplex=False, **kwargs):
        res = disco_mixture(controls, target, grid_min, grid_max, grid_rand, M, simplex)
        return res["weights_opt"]
        
    def evaluate_counterfactual(self, controls, weights, **kwargs):     
        grid_ord = kwargs.get("grid_ord")
        evgrid = kwargs.get("evgrid")
        controls_cdf = kwargs.get("controls_cdf", np.array([]))
        
        if len(controls_cdf) > 0 and weights is not None:
            disco_cdf = controls_cdf @ weights
            disco_quantile = np.array([grid_ord[np.argmax(disco_cdf >= (y - 1e-5))] for y in evgrid])
        else:
            disco_cdf = None
            disco_quantile = None

        return {
            "disco_quantile": disco_quantile,
            "disco_cdf": disco_cdf
        }

    def compute_distance(self, target, controls, weights, **kwargs):
        dist = 0
        target_cdf = kwargs.get("target_cdf")
        if target_cdf is not None and weights is not None:
            controls_cdf = kwargs.get("controls_cdf")
            bc_cdf = controls_cdf[:, 1:] @ weights if controls_cdf.shape[1] > len(weights) else controls_cdf @ weights
            dist = np.mean((bc_cdf - target_cdf)**2)
        return dist
