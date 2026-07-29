import numpy as np
import cvxpy as cp
import ot

from utils import getGrid
from .base import BaseSolver

class TangentialWassersteinSolver(BaseSolver):
    def __init__(self, method='emd'):
        super().__init__()
        self.method = method

    def baryc_proj(self, source, target):
        n1, p = source.shape
        n2 = target.shape[0]   
        a_ones, b_ones = np.ones((n1,)) / n1, np.ones((n2,)) / n2
        
        M = ot.dist(source, target)
        M = M.astype('float64')
        
        if self.method == 'emd':
            # EMD ist skaleninvariant bzgl. der Kostmatrix — Normalisierung optional
            if M.max() > 0:
                M /= M.max()
            OTplan = ot.emd(a_ones, b_ones, M, numItermax=int(1e7))
        elif self.method == 'entropic':
            # Bei Sinkhorn: reg muss proportional zur Kostskala sein.
            # Normalisiere M auf [0,1] und skaliere reg entsprechend,
            # sodass reg relativ zur normalisierten Skala korrekt ist.
            m_max = M.max()
            if m_max > 0:
                # reg_eff = 5e-3 relativ zu unnormalisierten Kosten
                # → nach Normalisierung: reg_normalized = 5e-3 * m_max / m_max = 5e-3
                # Wir wollen reg relativ zur Original-Skala, also reg = 5e-3 * m_max
                # auf der normalisierten Skala (M/m_max) ist das reg/m_max = 5e-3
                reg = 5e-3 * m_max
                OTplan = ot.bregman.sinkhorn_stabilized(a_ones, b_ones, M, reg=reg)
            else:
                OTplan = np.outer(a_ones, b_ones)
        else:
            raise ValueError("Method must be 'emd' or 'entropic'")
        
        row_sums = OTplan.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-16 
        OTplan_normalized = OTplan / row_sums
        
        OTmap = OTplan_normalized @ target
        return OTmap  # float64 beibehalten für CVXPY-Konsistenz

    def fit_weights(self, target, controls, **kwargs):
        n, d = target.shape
        J = len(controls)
        
        proj_list = []
        for i in range(J):
            temp = self.baryc_proj(target, controls[i])
            proj_list.append(temp - target)
            
        proj_flat = np.array([p.flatten() for p in proj_list]).T  # Shape: (n*d, J)
        
        # Skalierungsfaktor basierend auf der tatsächlichen Größenordnung der Zielfunktion,
        # nicht auf np.mean(target) — stabil bei zentrierten oder kleinen Daten
        S = max(np.sum(proj_flat**2) / proj_flat.size, 1e-8)
        
        mylambda = cp.Variable(J)

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
        if grid_ord is None:
            _, _, grid_ord = getGrid(target, controls, kwargs.get("G") )
        
        if weights is not None and target is not None:
            counterfactual_points = np.zeros_like(target, dtype='float64')
            for i, w in enumerate(weights):
                counterfactual_points += w * self.baryc_proj(target, controls[i])
                
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
        
        counterfactual_points = np.zeros_like(target, dtype='float64')
        for i, w in enumerate(weights):
            counterfactual_points += w * self.baryc_proj(target, controls[i])
        
        # Wasserstein-2-artige Distanz: mittlere quadrierte Verschiebung Target → Counterfactual
        # ||T(x) - x||^2 gemittelt über alle Samples
        displacement = counterfactual_points - target
        dist = np.mean(np.sum(displacement**2, axis=1))
        return float(dist)
