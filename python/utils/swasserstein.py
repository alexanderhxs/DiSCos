import pandas as pd
import numpy as np

def radon_transform(target: np.array, 
                    controls: list, 
                    n_slices: int = 100, 
                    directions: np.ndarray = None, 
                    sort_output: bool = True):
    """
    Projiziert hochdimensionale Daten auf zufällige 1D-Slices und sortiert diese.
    Dies ist der Kernschritt zur Berechnung der Sliced Wasserstein Distance.
    
    Args:
        target (np.ndarray): Target dataset observations.
        controls (list of np.ndarray): List of control dataset observations.
        n_slices (int): Die Anzahl der zufälligen 1D-Projektionen (L).
        directions (np.ndarray, optional): Vorberechnete Richtungsvektoren (d, L).
        sort_output (bool): Ob die projizierten Werte aufsteigend sortiert werden sollen.
        
    Returns:
        dict: Enthält 'projected_data' (N, L) und 'directions' (d, L).
    """
    # 1. Daten extrahieren und aneinander hängen
    if hasattr(target, "shape") and len(target) > 0:
        d = target.shape[1]
    elif len(controls) > 0 and hasattr(controls[0], "shape") and len(controls[0]) > 0:
        d = controls[0].shape[1]
    else:
        raise ValueError("Target and controls are empty or malformed.")

    valid_controls = [c for c in controls if len(c) > 0]
    data_list = [target] + valid_controls if len(target) > 0 else valid_controls
    
    if len(data_list) == 0:
        return {'projected_data': np.array([]), 'directions': np.array([])}
        
    data_np = np.concatenate(data_list, axis=0)  # Shape: (N, d)
    N = data_np.shape[0]
    L = n_slices
    
    # 2. Richtungen bestimmen
    if directions is None:
        # Gaussian Trick: Werte aus Standardnormalverteilung ziehen
        directions = np.random.randn(d, L)
        # Normalisieren, sodass L2-Norm = 1. Epsilon (1e-8) verhindert Division durch 0.
        norms = np.linalg.norm(directions, ord=2, axis=0, keepdims=True)
        directions = directions / (norms + 1e-8)
    else:
        assert directions.shape == (d, L), f"Erwartetes Shape für directions ist {(d, L)}, aber {directions.shape} erhalten."
        
    # 3. Projizieren (Matrixmultiplikation numpy dot)
    # X: (N, d), directions: (d, L) -> projected_data: (N, L)
    projected_data = np.dot(data_np, directions)
    
    # 4. Sortieren (optional, aber wichtig für CDF)
    if sort_output:
        projected_data = np.sort(projected_data, axis=0)
        
    return {
        'projected_data': projected_data,
        'directions': directions
    }
