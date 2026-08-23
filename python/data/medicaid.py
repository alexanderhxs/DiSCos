from cvxpy.atoms import quad_form
import os
import pandas as pd


def get_medicaid_data(
    data_path: str = None,
    unit_col: str = 'STATEFIP',
    time_col: str = 'YEAR',
    outcome_cols: list[str] = ['INCWAGE', 'UHRSWORK'],
    max_size: int = 2000,
    random_state: int = 42,
    weighted: bool = False,
    pooled: bool = False,
):
    """
    Loads the Medicaid dataset in right format.

    Args:
        data_path (str): Path to the data file.
        unit_col (str): Column name for the unit identifier.
        time_col (str): Column name for the time variable.
        outcome_col (str): Column name for the outcome variable.

    Returns:
        pd.DataFrame: Medicaid dataset in right format.
    """
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), 'datasets', 'medicaid.csv')
        
    df_medicaid = pd.read_csv(data_path)
    
    cols = [unit_col, time_col] + outcome_cols
    if weighted and'PERWT' not in cols:
        cols.append('PERWT')
    
    df_medicaid = df_medicaid[cols]
    
    if max_size is not None:
        def _get_sampled_indices(x):
            n = min(len(x), max_size)
            if weighted:
                weights = x['PERWT'].fillna(0)
                if weights.sum() <= 0:
                    raise ValueError("Invalid weights: weights sum to zero")
            else:
                weights = None
            return x.sample(n=n, replace=True, random_state=random_state, weights=weights)

        if not pooled:
            df_medicaid = (df_medicaid.groupby([unit_col, time_col])
                            .apply(_get_sampled_indices)
                            .reset_index(drop=False))
        else:
            
            df_medicaid_pre = (df_medicaid[df_medicaid[time_col] <=2016].groupby([unit_col])
                            .apply(_get_sampled_indices)
                            .reset_index(drop=False))
            df_medicaid_pre[time_col] = 0
            df_medicaid_post = (df_medicaid[df_medicaid[time_col] >2016].groupby([unit_col])
                            .apply(_get_sampled_indices)
                            .reset_index(drop=False))
            df_medicaid_post[time_col] = 1
            df_medicaid = pd.concat([df_medicaid_pre, df_medicaid_post])
    
    return df_medicaid