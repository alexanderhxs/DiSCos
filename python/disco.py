import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .solvers import disco_weights_reg, disco_mixture
from .utils import getGrid

class DiSCo:
    def __init__(self, df, id_col, time_col, y_col, id_col_target, t0, 
                 M=1000, G=100, num_cores=-1, q_min=0.0, q_max=1.0, 
                 mixture=False, simplex=False):
        """
        Distributional Synthetic Controls
        
        Parameters:
        df (pd.DataFrame): Data containing unit ids, time periods, and outcomes
        id_col (str): Column name for unit IDs
        time_col (str): Column name for time periods
        y_col (str): Column name for outcome variable
        id_col_target (int/str): The ID of the primary treated unit
        t0 (int/float): The time period where treatment begins
        M (int): Number of Monte Carlo draws for quantile approach
        G (int): Grid size for the distribution mixtures
        num_cores (int): Number of cores for joblib. -1 uses all available.
        q_min, q_max (float): Bounds for quantile evaluation (0 to 1)
        mixture (bool): If True, use the CDF mixture approach instead of quantile regression
        simplex (bool): If True, constrain weights to be non-negative and sum to 1
        """
        self.df = df.copy()
        self.id_col = id_col
        self.time_col = time_col
        self.y_col = y_col
        self.id_col_target = id_col_target
        self.t0 = t0
        self.M = M
        self.G = G
        self.num_cores = num_cores
        self.q_min = q_min
        self.q_max = q_max
        self.mixture = mixture
        self.simplex = simplex
        
        self._preprocess()
        
    def _preprocess(self):
        # Time normalization (1-indexed)
        min_time = self.df[self.time_col].min()
        self.df['t_col'] = self.df[self.time_col] - min_time + 1
        self.T0_idx = self.t0 - min_time + 1
        
        self.periods = sorted(self.df['t_col'].unique())
        
        all_ids = self.df[self.id_col].unique()
        self.controls_id = [uid for uid in all_ids if uid != self.id_col_target]
        
        # Determine quantile constraints if necessary
        if self.q_min > 0 or self.q_max < 1:
            # Drop data outside the quantile bounds per unit/time
            # Replicating R's data.table::frank(y_col, ties.method = "average") / .N
            quantiles = self.df.groupby([self.id_col, 't_col'])[self.y_col].rank(method='average', pct=True)
            self.df = self.df[(quantiles >= self.q_min) & (quantiles <= self.q_max)]

        self.evgrid = np.linspace(0, 1, self.G + 1)
        
    def _iter_period(self, t):
        """
        Process a single time period (analogous to DiSCo_iter)
        """
        df_t = self.df[self.df['t_col'] == t]
        
        target_data = df_t[df_t[self.id_col] == self.id_col_target][self.y_col].values
        
        if len(target_data) == 0:
            return None
            
        controls_data = []
        for cid in self.controls_id:
            c_data = df_t[df_t[self.id_col] == cid][self.y_col].values
            if len(c_data) > 0:
                controls_data.append(c_data)
                
        if len(controls_data) == 0:
            return None
            
        if self.mixture:
            grid_min, grid_max, grid_rand, grid_ord = getGrid(target_data, controls_data, self.G)
            res = disco_mixture(controls_data, target_data, grid_min, grid_max, grid_rand, self.M, self.simplex)
            weights = res['weights_opt']
        else:
            weights = disco_weights_reg(controls_data, target_data, M=self.M, simplex=self.simplex, 
                                        q_min=self.q_min, q_max=self.q_max)
            
        return {'t': t, 'weights': weights}

    def fit(self):
        """
        Run the complete DiSCo estimation across all periods in parallel.
        """
        results = Parallel(n_jobs=self.num_cores)(
            delayed(self._iter_period)(t) for t in self.periods
        )
        
        results = [r for r in results if r is not None]
        
        if not results:
            raise ValueError("No valid periods data found.")
            
        self.weights_by_period = {r['t']: r['weights'] for r in results}
        
        # Average pre-treatment weights
        pre_treat_weights = []
        for t in self.periods:
            if t < self.T0_idx and t in self.weights_by_period:
                pre_treat_weights.append(self.weights_by_period[t])
                
        if not pre_treat_weights:
            raise ValueError("No pre-treatment periods found or calculated.")
            
        self.weights_opt = np.mean(pre_treat_weights, axis=0)
        
        return self.weights_opt
