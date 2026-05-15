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
    #TODO: Bei 2D X, shape (N,d) soll (N,d) zurückkommen, wobei die Quantiele pro Spalte berechnet werden
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
    
    return grid_min, grid_max, grid_ord

def sample_counterfactual_distribution(controls, weights, grid, num_samples=None):

    if len(controls) > 0 and weights is not None:
        w = np.clip(weights, 0, None) # clip for sampling
        w_sum = w.sum()
        w = w / w_sum if w_sum > 0 else np.ones(len(w)) / len(w)
        
        positive_idx = np.where(w > 0)[0]
        chosen_controls = [controls[i] for i in positive_idx]
        choosen_conrols_weights = w[positive_idx]

        if num_samples is None:
            capacities = np.array([len(c) for c in chosen_controls]) / choosen_conrols_weights
            num_samples = int(np.floor(np.min(capacities)))

        strata_sizes = num_samples * choosen_conrols_weights
        strata_sizes = np.round(strata_sizes).astype(int)

        if strata_sizes.sum() != num_samples:
            idx = strata_sizes.argmax() # Korrigieren der größten Strata, um die Gesamtzahl der Samples zu erreichen
            strata_sizes[idx] += int(num_samples - strata_sizes.sum())   

        samples = []
        for i, ctrl in enumerate(chosen_controls):
            k = strata_sizes[i]
            if k <= 0:
                continue
                
            ctrl_arr = np.asarray(ctrl)
            
            # ECDF calculation on grid (1D and Multi-D)
            if ctrl_arr.ndim == 1 or (ctrl_arr.ndim == 2 and ctrl_arr.shape[1] == 1):

                ctrl_sq = np.squeeze(ctrl_arr)
                ctrl_sorted = np.sort(ctrl_sq)
                grid_sq = np.squeeze(grid)
                cdf = np.searchsorted(ctrl_sorted, grid_sq, side='right') / len(ctrl_sq)

                u = (np.arange(k) + 0.5) / k
                # Find the first index where cdf >= u
                mask = cdf[None, :] >= u[:, None]
                idx = np.argmax(mask, axis=1)
            
                # Handle cases where `u` might be strictly larger than the max available CDF value
                missing = ~np.any(mask, axis=1)
                idx[missing] = len(grid) - 1
            
                samples.append(grid[idx])
            else:
                indices = np.linspace(0, len(ctrl_arr) - 1, k).astype(int)
                samples.append(ctrl_arr[indices])

            

        if len(samples) > 0:
            disco_dist = np.concatenate(samples, axis=0)
            # Zufälliges Mischen der stratifizerten Samples um Cluster-Bildung aufzuheben
            np.random.shuffle(disco_dist)
            return disco_dist
        else:
            return np.array([])
    else:
        return None