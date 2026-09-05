import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

def get_mnist_data(digit='8', n_controls=8, corruption='occlusion', num_samples=2000, seed=42):
    """
    Fetches MNIST, selects a digit, creates 1 target and n_controls control images.
    Applies corruption (occlusion or noise) as described in the DiSCo paper.
    Returns a DataFrame suitable for DiSCo.
    """
    np.random.seed(seed)
    
    print(f"Fetching MNIST data from OpenML (this may take a minute if not cached)...")
    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False, parser='auto')
    
    # Filter for the chosen digit
    # target labels are strings in openml mnist
    X = mnist.data[mnist.target == str(digit)]
    
    if len(X) < 1 + n_controls:
        raise ValueError(f"Not enough samples for digit {digit}")
        
    indices = np.random.choice(X.shape[0], size=1 + n_controls, replace=False)
    images = X[indices].reshape(-1, 28, 28)
    
    alpha = 0.5
    dfs = []
    
    for i in range(1 + n_controls):
        img = images[i].copy()
        
        # Normalize original image to max 1 
        if img.max() > 0:
            img = img / img.max()
            
        is_target = (i == 0)
        img_id = 'target' if is_target else f'control_{i}'
        
        if corruption == 'occlusion':
            # Target: central 8x8 square removed
            # Controls: random 8x8 square removed
            if not is_target:
                start_x = np.random.randint(0, 15)
                start_y = np.random.randint(0, 15)
                
                img[start_y:start_y+14, start_x:start_x+14] = 0.0
            
        elif corruption == 'noise':
            # Target: (1-alpha)*mu0 + alpha*zeta
            # Controls: (1-alpha)*mui + alpha*E[zeta]
            if is_target:
                zeta = np.random.rand(28, 28)
                img = (1 - alpha) * img + alpha * zeta
            else:
                img = (1 - alpha) * img + alpha * 0.5
                
        # Renormalize image to sum to 1
        img_sum = img.sum()
        if img_sum > 0:
            prob = img.flatten() / img_sum
        else:
            prob = np.ones_like(img.flatten()) / len(img.flatten())
            
        # Sample points based on the probability distribution
        sampled_indices = np.random.choice(len(prob), size=num_samples, p=prob, replace=True)
        y_coords, x_coords = np.unravel_index(sampled_indices, img.shape)
        
        # Normalize to [0, 1]
        x_coords = x_coords / img.shape[1]
        y_coords = (img.shape[0] - 1 - y_coords) / img.shape[0] # Invert Y so it plots upright
        
        # Add small continuous noise to prevent exact overlapping
        x_coords += np.random.uniform(-0.5/img.shape[1], 0.5/img.shape[1], size=num_samples)
        y_coords += np.random.uniform(-0.5/img.shape[0], 0.5/img.shape[0], size=num_samples)
        
        x_coords = np.clip(x_coords, 0, 1)
        y_coords = np.clip(y_coords, 0, 1)
        
        df = pd.DataFrame({
            'ID': img_id,
            'TIME': 0,
            'X': x_coords,
            'Y': y_coords
        })
        dfs.append(df)
        
    return pd.concat(dfs, ignore_index=True)
