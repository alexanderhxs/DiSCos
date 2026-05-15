import numpy as np
import cvxpy as cp
from .base import BaseSolver

def disco_mixture(controls, target, grid_min, grid_max, grid_ord, M, simplex):
    num_controls = len(controls)
    cdf_matrix = np.zeros((len(grid_ord), num_controls + 1))
    target = np.squeeze(target)

    if target.ndim == 1:
        target_sorted = np.sort(target)
        cdf_matrix[:, 0] = np.searchsorted(target_sorted, grid_ord, side='right') / len(target)

        for i, ctrl in enumerate(controls):
            ctrl = np.squeeze(ctrl)
            ctrl_sorted = np.sort(ctrl)
            cdf_matrix[:, i+1] = np.searchsorted(ctrl_sorted, grid_ord, side='right') / len(ctrl)
    else:
        cdf_matrix[:, 0] = np.mean(np.all(target[None, :, :] <= grid_ord[:, None, :], axis=2), axis=1)

        for i, ctrl in enumerate(controls):
            ctrl = np.atleast_2d(ctrl)
            cdf_matrix[:, i+1] = np.mean(np.all(ctrl[None, :, :] <= grid_ord[:, None, :], axis=2), axis=1)
    
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
    
    # Since grid_ord is typically sorted, mean_order essentially is mean_val
    order_idx = np.argsort(grid_ord)
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
    def fit_weights(self, target, controls, grid_min=None, grid_max=None, grid_ord=None, M=500, simplex=False, **kwargs):
        res = disco_mixture(controls, target, grid_min, grid_max, grid_ord, M, simplex)
        return res["weights_opt"]
        
    def evaluate_counterfactual(self, controls, weights, **kwargs):     
        grid_ord = kwargs.get("grid_ord")
        evgrid = kwargs.get("evgrid")
        controls_cdf = kwargs.get("controls_cdf", None)
        
        if controls_cdf is None:
            num_controls = len(controls)
            controls_cdf = np.zeros((len(grid_ord), num_controls))
            
            sample_ctrl = np.array(controls[0])
            if sample_ctrl.ndim > 1 or (sample_ctrl.ndim == 1 and sample_ctrl.shape[0] > 0 and isinstance(sample_ctrl[0], (list, np.ndarray))):
                for i, ctrl in enumerate(controls):
                    ctrl = np.atleast_2d(ctrl)
                    controls_cdf[:, i] = np.mean(np.all(ctrl[None, :, :] <= grid_ord[:, None, :], axis=2), axis=1)
            else:
                for i, ctrl in enumerate(controls):
                    ctrl_sq = np.squeeze(ctrl)
                    ctrl_sorted = np.sort(ctrl_sq)
                    controls_cdf[:, i] = np.searchsorted(ctrl_sorted, grid_ord, side='right') / len(ctrl_sq)

        if len(controls_cdf) > 0 and weights is not None:
            disco_cdf = controls_cdf @ weights
            disco_quantile = None
            if grid_ord.ndim == 1 or (grid_ord.ndim == 2 and grid_ord.shape[1] == 1): 
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
        target_cdf = kwargs.get("target_cdf", None)
        controls_cdf = kwargs.get("controls_cdf", None)
        grid = kwargs.get("grid_ord", None) 

        if target_cdf is None or controls_cdf is None:
            if grid is None:
                raise ValueError("grid_ord must be provided to compute CDFs.")
                
            target_sq = np.squeeze(target)
            if target_sq.ndim == 1:
                target_sorted = np.sort(target_sq)
                target_cdf = np.searchsorted(target_sorted, grid, side='right') / len(target_sq)

                num_controls = len(controls)
                controls_cdf = np.zeros((len(grid), num_controls))
                for i, ctrl in enumerate(controls):
                    ctrl_sq = np.squeeze(ctrl)
                    ctrl_sorted = np.sort(ctrl_sq)
                    controls_cdf[:, i] = np.searchsorted(ctrl_sorted, grid, side='right') / len(ctrl_sq)
            else:
                target_cdf = np.mean(np.all(target[None, :, :] <= grid[:, None, :], axis=2), axis=1)
                
                num_controls = len(controls)
                controls_cdf = np.zeros((len(grid), num_controls))
                for i, ctrl in enumerate(controls):
                    ctrl = np.atleast_2d(ctrl)
                    controls_cdf[:, i] = np.mean(np.all(ctrl[None, :, :] <= grid[:, None, :], axis=2), axis=1)

        if target_cdf is not None and weights is not None:
            bc_cdf = controls_cdf[:, 1:] @ weights if controls_cdf.shape[1] > len(weights) else controls_cdf @ weights
            dist = np.mean((bc_cdf - target_cdf)**2)
        return dist
