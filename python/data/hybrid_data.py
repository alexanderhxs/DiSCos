import numpy as np
import pandas as pd

def get_hybrid_data(
    data_path: str, 
    seed: int = 42,
    y_cols: list = ['INCWAGE', 'UHRSWORK'],
    unit_col: str = 'STATEFIP',
    time_col: str = 'YEAR',
    n_donors: int = 5,
    max_size: int = 2000
) -> tuple[pd.DataFrame, dict]:
    """
    Takes real world microdata and creates a synthetic target population based on ground truth weights.
    Weights sum up to 1, and we only use 4-6 non-zero weights to create sparsity.
    
    The synthetic target is a mixture distribution formed by sampling individuals from the chosen donors.
    
    Args:
        data_path (str): Path to the data file.
        seed (int): Random seed.
        variables (list): List of variables to use for the synthetic data.
        t0 (int): The year of the control group.
        y (str): The name of the treatment variable.
        max_sample_size (int): The total sample size (number of individuals) per time period for the synthetic target.
        unit_col (str): Column name for the unit identifier.
        time_col (str): Column name for the time variable.
        n_donors (int): Number of donors to use for the synthetic target.

    Returns:
        pd.DataFrame: Hybrid data containing original units and the 'synthetic_target'.
        dict: Dictionary mapping donor unit IDs to their non-zero ground truth weights.
    """
    np.random.seed(seed)
    df_data = pd.read_csv(data_path)

    cols = [unit_col, time_col] + y_cols
    df_data = df_data[cols]
    
    if max_size is not None:
        df_data = (df_data.groupby([unit_col, time_col])
                   .apply(lambda x: x.sample(n=min(len(x), max_size), replace=False, random_state=seed))
                   .reset_index(drop=False))
    
    unit_ids = df_data[unit_col].unique()
    time_periods = df_data[time_col].unique()
    
    if len(unit_ids) <= n_donors:
        raise ValueError(f"Not enough units ({len(unit_ids)}) to select {n_donors} donors from.")

    # 1. Randomly select donors and generate their weights
    chosen_donors = np.random.choice(unit_ids, size=n_donors, replace=False)
    
    # Generate weights from a flat Dirichlet distribution
    weights = np.random.dirichlet(np.ones(n_donors))
    donor_weights = dict(zip(chosen_donors, weights))
    
    weights_dict = {u: donor_weights.get(u, np.float64(0.0)) for u in sorted(unit_ids)}
    # 2. Build the synthetic target distribution by sampling
    synthetic_rows = []
    
    for t in time_periods:
        for donor in chosen_donors:
            weight = weights_dict[donor]
            # Number of individuals to sample from this donor for this time period
            n_samples = int(np.round(weight * max_size))
            
            if n_samples == 0:
                continue
                
            donor_data = df_data[(df_data[unit_col] == donor) & (df_data[time_col] == t)]
            sampled = donor_data.sample(n=n_samples, replace=True, random_state=seed)
            sampled[unit_col] = 'synthetic_target'
            synthetic_rows.append(sampled)
                
    if not synthetic_rows:
        raise RuntimeError("Failed to generate any synthetic data rows. Check the dataset structure.")
        
    df_synth = pd.concat(synthetic_rows, ignore_index=True)
    
    # Append the synthetic target back to the original dataset
    df_hybrid = pd.concat([df_data, df_synth], ignore_index=True)
    
    # Ensure consistent sorting
    df_hybrid = df_hybrid.sort_values(by=[unit_col, time_col]).reset_index(drop=True)
    
    return df_hybrid, weights_dict
