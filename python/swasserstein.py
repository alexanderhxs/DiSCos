import torch
import pandas as pd
import numpy as np

def radon_transform(target: np.array, 
                    controls: list, 
                    n_slices: int = 100, 
                    directions: torch.Tensor = None, 
                    sort_output: bool = True):
    """
    Projiziert hochdimensionale Daten auf zufällige 1D-Slices und sortiert diese.
    Dies ist der Kernschritt zur Berechnung der Sliced Wasserstein Distance.
    
    Args:
        df (pandas.DataFrame): Der Datensatz mit den Beobachtungen.
        features (list of strings): Die Spaltennamen (Dimension d).
        n_slices (int): Die Anzahl der zufälligen 1D-Projektionen (L).
        directions (torch.Tensor, optional): Vorberechnete Richtungsvektoren (d, L).
        sort_output (bool): Ob die projizierten Werte aufsteigend sortiert werden sollen.
        
    Returns:
        dict: Enthält 'projected_data' (N, L) und 'directions' (d, L).
    """
    # 1. Daten extrahieren und in PyTorch Tensor konvertieren
    data_np = np.concatenate([target] + controls, axis=0)  # Shape: (N, d)
    X = torch.tensor(data_np, dtype=torch.float32)  # Shape: (N, d)
    
    N, d = X.shape
    L = n_slices
    
    # 2. Richtungen bestimmen
    if directions is None:
        # Gaussian Trick: Werte aus Standardnormalverteilung ziehen
        directions = torch.randn(d, L)
        # Normalisieren, sodass L2-Norm = 1. Epsilon (1e-8) verhindert Division durch 0.
        norms = torch.norm(directions, p=2, dim=0, keepdim=True)
        directions = directions / (norms + 1e-8)
    else:
        assert directions.shape == (d, L), f"Erwartetes Shape für directions ist {(d, L)}, aber {directions.shape} erhalten."
        
    # 3. Projizieren (Matrixmultiplikation)
    # X: (N, d), directions: (d, L) -> projected_data: (N, L)
    projected_data = torch.matmul(X, directions)
    
    # 4. Sortieren (optional, aber wichtig für CDF)
    if sort_output:
        projected_data, _ = torch.sort(projected_data, dim=0)
        
    return {
        'projected_data': projected_data,
        'directions': directions
    }
