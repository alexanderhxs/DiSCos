import numpy as np
import cvxpy as cp
from scipy.optimize import lsq_linear, minimize
from .swasserstein import radon_transform
from .utils import myQuant
from abc import ABC, abstractmethod

class BaseSolver(ABC):
    @abstractmethod
    def fit_weights(self, target, controls, **kwargs):
        """Fit empirical distributions to find optimal synthetic weights."""   
        pass
        
    @abstractmethod
    def evaluate_counterfactual(self, target, controls, weights, **kwargs):     
        """Returns the specific geometry data for synthetic control matching 
        (e.g., CDF/Quantiles on fixed grids, or pure 2D samples)."""
        pass
        
    @abstractmethod
    def compute_distance(self, target, controls, weights, **kwargs):
        """Compute the distance metric used for inference/permutation."""
        pass

class Quantile1DSolver(BaseSolver):
    def fit_weights(self, target, controls, M=500, simplex=False, q_min=0, q_max=1, **kwargs):
        return disco_weights_reg(controls, target, M, simplex, q_min, q_max)    
        
    def evaluate_counterfactual(self, target, controls, weights, **kwargs):
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
        # 1D Quantile method uses squared differences in quantiles evaluated on evgrid
        dist = 0
        target_q = kwargs.get("target_q")
        if target_q is not None and weights is not None:
            controls_q = kwargs.get("controls_q")
            bc_q = controls_q @ weights
            dist = np.mean((bc_q - target_q)**2)
        return dist

class MixtureSolver(BaseSolver):
    def fit_weights(self, target, controls, grid_min=None, grid_max=None, grid_rand=None, M=500, simplex=False, **kwargs):
        res = disco_mixture(controls, target, grid_min, grid_max, grid_rand, M, simplex)
        return res["weights_opt"]
        
    def evaluate_counterfactual(self, target, controls, weights, **kwargs):     
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
        # Mixture method uses squared differences in CDF evaluated on the grid 
        dist = 0
        target_cdf = kwargs.get("target_cdf")
        if target_cdf is not None and weights is not None:
            controls_cdf = kwargs.get("controls_cdf")
            bc_cdf = controls_cdf[:, 1:] @ weights if controls_cdf.shape[1] > len(weights) else controls_cdf @ weights
            dist = np.mean((bc_cdf - target_cdf)**2)
        return dist


class SlicedWassersteinSolver(Quantile1DSolver):
    def __init__(self, n_slices=10000):
        super().__init__()
        self.n_slices = n_slices

    def fit_weights(self, target, controls, **kwargs):
        N, num_controls = target.shape[0], len(controls)
        M = kwargs.get("M", 500)
        simplex = kwargs.get("simplex", True)
        # 1. Radon-Transformation: Projizieren auf zufällige 1D-Slices
        radon_result = radon_transform(target, controls, n_slices=self.n_slices, sort_output=False)
        projected_data = radon_result['projected_data'].reshape(num_controls+1, N, self.n_slices) 

        # 2. Quantile Regression auf den projizierten Daten
        weights = disco_weights_reg(projected_data[1:, :,:], projected_data[0, :, :], M=M, simplex=simplex)
        
        return weights 
    
    def evaluate_counterfactual(self, target, controls, weights, **kwargs):     
        grid_ord = kwargs.get("grid_ord")
        evgrid = kwargs.get("evgrid")
        controls_cdf = kwargs.get("controls_cdf", np.array([]))
        
        #TODO: Directional/Sliced Wasserstein CDF/Quantile Berechnung hier implementieren
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
        # 1. Radon-Transformation verwenden (ohne gemeinsames Sortieren, da Ziel/Controls separat bleiben muessen)
        radon_result = radon_transform(target, controls, n_slices=self.n_slices, sort_output=False)
        projected_data = radon_result['projected_data']

        M = kwargs.get("M", 500)
        q_min = kwargs.get("q_min", 0)
        q_max = kwargs.get("q_max", 1)
        m_vec = np.linspace(q_min, q_max, M)

        dist = 0.0
        n_target = len(target)
        
        # Calculate 1D Wasserstein distance on each slice and average
        for l in range(self.n_slices):
            # Target aus dem gepoolten Array extrahieren (erste n_target Zeilen)
            target_slice = projected_data[:n_target, l]
            target_q = myQuant(target_slice, m_vec)

            ctrl_q_list = []
            offset = n_target
            # Controls aus den restlichen Zeilen extrahieren
            for ctrl in controls:
                n_c = len(ctrl)
                c_slice = projected_data[offset : offset + n_c, l]
                ctrl_q_list.append(myQuant(c_slice, m_vec))
                offset += n_c
            
            controls_q_stacked = np.column_stack(ctrl_q_list)
            bc_q = controls_q_stacked @ weights
            
            dist += np.mean((bc_q - target_q)**2)

        return dist / self.n_slices


def disco_weights_reg(controls, target, M=500, simplex=False, q_min=0, q_max=1):
    """
    Function for obtaining the weights in the DiSCo method via quantile regression.
    
    Parameters:
    controls (list of np.ndarray): List of control distributions
    target (np.ndarray): Target distribution
    M (int): Number of probabilistic draws
    simplex (bool): If True, weights are bounded by [0,1]. Otherwise, unconstrained (just sum to 1).
    q_min (float): Minimum quantile
    q_max (float): Maximum quantile
    
    Returns:
    np.ndarray: Optimal synthetic control weights
    """
    num_controls = len(controls)
    
    # M draws from uniform
    m_vec = np.random.uniform(q_min, q_max, M)

    # Quantiles for controls
    # If controls are list, compute one by one (could be different size)
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

    # Flatten the dimensions (e.g. projections d) to 2D for cvxpy if necessary
    if controls_s.ndim == 3:
        C = controls_s.reshape(-1, num_controls)
    else:
        C = controls_s
        
    # Scale matrix norm to avoid overflow
    sc = np.linalg.norm(C, ord=2) 
    if np.abs(sc) < 1e-9:
        sc = 1.0  # Avoid division by zero if all controls are zero
        
    C = C / sc
    d_vec = target_s.flatten() / sc
    
    # Solve using cvxpy for exact quadratic programming matching R's quadprog
    w = cp.Variable(num_controls)
    # Add a tiny Ridge penalty to explicitly stabilize flat minima
    # This forces a unique solution when control units are highly collinear
    objective = cp.Minimize(cp.sum_squares(C @ w - d_vec))
    
    # R's pracma::lsqlincon implicitly bounds weights to <= 1 even outside simplex
    constraints = [cp.sum(w) == 1]
    if simplex:
        constraints.append(w >= 0)
        
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, max_iters=10000, eps=1e-5)
    
    return w.value


def disco_mixture(controls, target, grid_min, grid_max, grid_rand, M, simplex):
    """
    The alternative mixture of distributions approach using L1 distance of CDFs.
    
    Parameters:
    controls (list of np.ndarray): List of controls
    target (np.ndarray): The target unit
    grid_min, grid_max, grid_rand, M, simplex: Parameters from getGrid / args
    
    Returns:
    dict: Results containing optimal weights, etc.
    """
    num_controls = len(controls)
    
    # Empirical CDF evaluations
    # Using np.searchsorted to quickly evaluate ECDF on grid_rand
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
        # Multivariate eCDF durch Broadcasting
        # grid_rand hat Shape (G, dim) -> erweitern auf (G, 1, dim)
        # target hat Shape (N, dim)    -> erweitern auf (1, N, dim)
        # Dadurch entsteht durch <= eine Matrix der Form (G, N, dim)
        
        # Target auswerten
        # 1. np.all(..., axis=2) prüft, ob BEIDE Bedingungen (Alter <= g UND Einkommen <= g) wahr sind.
        # 2. np.mean(..., axis=1) berechnet den Anteil der True-Werte (das ist die eCDF!)
        cdf_matrix[:, 0] = np.mean(np.all(target[None, :, :] <= grid_rand[:, None, :], axis=2), axis=1)

        # Controls auswerten
        for i, ctrl in enumerate(controls):
            ctrl = np.atleast_2d(ctrl) # Zur Sicherheit, falls eine Unit nur 1 Beobachtung hat
            cdf_matrix[:, i+1] = np.mean(np.all(ctrl[None, :, :] <= grid_rand[:, None, :], axis=2), axis=1)
    
    
    # CVXPY optimization
    w = cp.Variable(num_controls)
    
    # Minimize L2 norm: || C * w - target_cdf ||_1
    obj = cp.Minimize(cp.norm1(cdf_matrix[:, 1:] @ w - cdf_matrix[:, 0]))
    
    if simplex:
        constraints = [w >= 0, cp.sum(w) == 1]
    else:
        constraints = [cp.sum(w) == 1]
        
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, max_iters=10000, eps=1e-5)
    
    weights_opt = w.value
    if weights_opt is None:
        # Fallback if solver fails
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
