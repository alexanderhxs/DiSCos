import os
import pandas as pd
import numpy as np
import cv2

def get_image_data(data_path: str = None, num_samples: int = 10000, shape = (24,24),seed: int = 42):
    """
    Loads images and converts them into 2D empirical point clouds.
    Each image's pixel intensities are treated as a probability measure on the 2D grid,
    and we sample `num_samples` points from this distribution.
    """
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), 'datasets', 'Lego')
    
    pictures = sorted([f for f in os.listdir(data_path) if f.endswith('.png')])
    
    np.random.seed(seed)
    
    dfs = []
    for pic in pictures:
        img_path = os.path.join(data_path, pic)
        
        # Read as grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        # Invert if the background is white and the object is dark. 
        # Typically for Lego we want the object to have the mass.
        # Let's check the corners to see if background is white or black.
        bg_color = img[0, 0]
        if bg_color > 127:
            img = 255 - img
            
        # Resize image for appropriate image downsampling (e.g. to 50x50)
        img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
        
        img_flat = img.flatten()
        img_sum = img_flat.sum()
        
        if img_sum == 0:
            prob = np.ones_like(img_flat) / len(img_flat)
        else:
            prob = img_flat / img_sum
            
        sampled_indices = np.random.choice(len(img_flat), size=num_samples, p=prob, replace=True)
        
        y_coords, x_coords = np.unravel_index(sampled_indices, img.shape)
        
        # Normalize to [0, 1]
        x_coords = x_coords / img.shape[1]
        y_coords = 1.0 - (y_coords / img.shape[0])  # invert y so upright
        
        # We can also add a small noise to coordinates so they are continuous rather than strictly on a grid
        x_coords += np.random.uniform(-0.5/img.shape[1], 0.5/img.shape[1], size=num_samples)
        y_coords += np.random.uniform(-0.5/img.shape[0], 0.5/img.shape[0], size=num_samples)
        
        x_coords = np.clip(x_coords, 0, 1)
        y_coords = np.clip(y_coords, 0, 1)
        
        df = pd.DataFrame({
            'ID': pic,
            'TIME': 0,
            'X': x_coords,
            'Y': y_coords
        })
        dfs.append(df)
        
    return pd.concat(dfs, ignore_index=True)
