import numpy as np
import cvxpy as cp
from scipy.optimize import lsq_linear, minimize
from utils import myQuant

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

    #debuggging version
    #m_vec = np.linspace(q_min, q_max, M)
    
    # Quantiles for controls
    # If controls are list, compute one by one (could be different size)
    controls_s = np.zeros((M, num_controls))
    for i, ctrl in enumerate(controls):
        controls_s[:, i] = myQuant(ctrl, m_vec)
        
    target_s = myQuant(target, m_vec).reshape((M, 1))
    
    # Scale matrix norm to avoid overflow
    sc = np.linalg.norm(controls_s, ord=2) 
    if sc == 0:
        sc = 1.0
        
    C = controls_s / sc
    d = (target_s / sc).flatten()
    
    # Solve using cvxpy for exact quadratic programming matching R's quadprog
    w = cp.Variable(num_controls)
    # Add a tiny Ridge penalty to explicitly stabilize flat minima
    # This forces a unique solution when control units are highly collinear
    objective = cp.Minimize(cp.sum_squares(C @ w - d) + 1e-8 * cp.sum_squares(w))
    
    # R's pracma::lsqlincon implicitly bounds weights to <= 1 even outside simplex
    constraints = [cp.sum(w) == 1, w <= 1]
    if simplex:
        constraints.append(w >= 0)
        
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    return w.value

