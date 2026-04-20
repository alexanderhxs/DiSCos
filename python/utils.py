import numpy as np

def myQuant(X, q):
    """
    Compute the empirical quantile function.
    
    Parameters:
    X (np.ndarray): 1D array of data (or 2D where axis=1 for batching over controls)
    q (np.ndarray): 1D array of probabilities (0 to 1)
    
    Returns:
    np.ndarray: Evaluated quantiles
    """
    # Equivalent to R's quantile type 7
    return np.quantile(X, q, axis=0, method='linear')

def getGrid(target, controls, G):
    """
    Set up a grid for the estimation of the quantile functions and CDFs.
    Handles both 1D and multi-dimensional (N, dim) inputs seamlessly.
    """
    is_multi = getattr(target, 'ndim', 1) > 1
    
    if is_multi:
        axis = 0
    else:
        axis = None  # Verhält sich wie bisher (flattens)
        
    if isinstance(controls, list):
        control_min = np.min([np.min(c, axis=axis) for c in controls if len(c) > 0], axis=0)
        control_max = np.max([np.max(c, axis=axis) for c in controls if len(c) > 0], axis=0)
    else:
        control_min = np.min(controls, axis=axis)
        control_max = np.max(controls, axis=axis)
        
    grid_min = np.minimum(np.min(target, axis=axis), control_min)
    grid_max = np.maximum(np.max(target, axis=axis), control_max)
    
    # round like R (works on scalars and arrays)
    grid_min = np.floor(grid_min * 10) / 10
    grid_max = np.ceil(grid_max * 10) / 10
    

    # sampling uniformly on the grid
    if is_multi:
        dim = target.shape[1]
    else:
        dim = 1
    grid_rand = np.random.uniform(grid_min, grid_max, (G**dim, dim)) #TODO: Latin Hypercube Sampling
    if dim ==1:
        grid_rand = grid_rand.squeeze(axis=1) 
        grid_ord = np.sort(grid_rand)
    else:
        grid_ord = grid_rand
    
    return grid_min, grid_max, grid_rand, grid_ord
