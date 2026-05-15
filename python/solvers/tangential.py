import numpy as np
import cvxpy as cp
import ot
from .base import BaseSolver

class TangentialWassersteinSolver(BaseSolver):
    def __init__(self, method='emd'):
        super().__init__()
        self.method = method
        self.G_list = []

    def baryc_proj(self, source, target):
        n1, p = source.shape
        n2 = target.shape[0]   
        a_ones, b_ones = np.ones((n1,)) / n1, np.ones((n2,)) / n2
        
        M = ot.dist(source, target)
        M = M.astype('float64')
        if M.max() > 0:
            M /= M.max()
        
        if self.method == 'emd':
            OTplan = ot.emd(a_ones, b_ones, M, numItermax=int(1e7))
        elif self.method == 'entropic':
            OTplan = ot.bregman.sinkhorn_stabilized(a_ones, b_ones, M, reg=5e-3)
        else:
            raise ValueError("Method must be 'emd' or 'entropic'")
        
        row_sums = OTplan.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-16 
        OTplan_normalized = OTplan / row_sums
        
        OTmap = OTplan_normalized @ target
        return OTmap.astype('float32')

    def fit_weights(self, target, controls, **kwargs):
        n, d = target.shape
        J = len(controls)
        
        self.G_list = []
        proj_list = []
        for i in range(J):
            temp = self.baryc_proj(target, controls[i])
            self.G_list.append(temp)
            proj_list.append(temp - target)
            
        S = np.mean(target) * n * d * J if np.mean(target) != 0 else 1.0 
        S = np.abs(S) 
        mylambda = cp.Variable(J)

        proj_flat = np.array([p.flatten() for p in proj_list]).T  
        objective = cp.Minimize(cp.sum_squares(proj_flat @ mylambda) / S)
        
        simplex = kwargs.get("simplex", True)
        if simplex:
            constraints = [mylambda >= 0, cp.sum(mylambda) == 1]
        else:
            constraints = [cp.sum(mylambda) == 1]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, max_iters=10000, eps=1e-5)
        
        return mylambda.value
        
    def evaluate_counterfactual(self, controls, weights, **kwargs):
        target = kwargs.get("target")
        grid_ord = kwargs.get("grid_ord")
        
        if weights is not None and target is not None:
            counterfactual_points = np.zeros_like(target, dtype='float32')
            for i, w in enumerate(weights):
                counterfactual_points += w * self.G_list[i]
                
            if grid_ord is not None and len(grid_ord) > 0:
                disco_cdf = np.mean(np.all(counterfactual_points[None, :, :] <= grid_ord[:, None, :], axis=2), axis=1)
            else:
                disco_cdf = None
        else:
            disco_cdf = None

        return {
            "disco_quantile": None,
            "disco_cdf": disco_cdf
        }

    def compute_distance(self, target, controls, weights, **kwargs):
        if weights is None: return 0.0
        
        counterfactual_points = np.zeros_like(target, dtype='float32')
        for i, w in enumerate(weights):
            counterfactual_points += w * self.G_list[i]
            
        dist = np.mean((counterfactual_points - target)**2)
        return dist
