import os
import pandas as pd


def get_medicaid_data(
    data_path: str = None,
    unit_col: str = 'STATEFIP',
    time_col: str = 'YEAR',
    outcome_cols: list[str] = ['INCWAGE', 'UHRSWORK'],
    max_size: int = 2000,
    random_state: int = 42,

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
    df_medicaid = df_medicaid[cols]
    
    if max_size is not None:
        df_medicaid = (df_medicaid.groupby([unit_col, time_col])
                       .apply(lambda x: x.sample(n=min(len(x), max_size), replace=False, random_state=random_state))
                       .reset_index(drop=False))
    
    return df_medicaid
    