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
    return np.quantile(X, q, axis=-1, method='linear')

def getGrid(target, controls, G):
    """
    Set up a grid for the estimation of the quantile functions and CDFs.
    
    Parameters:
    target (np.ndarray): 1D array of target data
    controls (list of np.ndarray or 2D array): control data
    G (int): number of grid points
    
    Returns:
    tuple: (grid_min, grid_max, grid_rand, grid_ord)
    """
    if isinstance(controls, list):
        control_min = min(np.min(c) for c in controls if len(c) > 0)
        control_max = max(np.max(c) for c in controls if len(c) > 0)
    else:
        control_min = np.min(controls)
        control_max = np.max(controls)
        
    grid_min = min(np.min(target), control_min)
    grid_max = max(np.max(target), control_max)
    
    # round like R
    grid_min = np.floor(grid_min * 10) / 10
    grid_max = np.ceil(grid_max * 10) / 10
    
    # sampling uniformly on the grid
    grid_rand = np.random.uniform(grid_min - 0.25, grid_max + 0.25, G)
    grid_ord = np.sort(grid_rand)
    
    return grid_min, grid_max, grid_rand, grid_ord
