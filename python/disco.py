import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .solvers import disco_weights_reg, disco_mixture
from .inference import run_bootstrap_ci
from .permutation import run_permutation_test
from .utils import getGrid, myQuant
from .models import DiSCoResult, DiSCoParams, PeriodResult, DiSCoMethodResult, MixtureMethodResult, TargetData, ControlsData, PermutResult

class DiSCo:
    def __init__(self, df, id_col, time_col, y_col, id_col_target, t0, 
                 M=1000, G=100, num_cores=-1, q_min=0.0, q_max=1.0, CI= False, uniform=False, perm=False, cl=0.95, B=100,
                 mixture=False, simplex=False, seed=None):
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
        self.CI = CI
        self.cl = cl
        self.B = B
        self.perm = perm
        self.seed = seed
        self.uniform = uniform
        
        self._preprocess()
        
    def _preprocess(self):
        np.random.seed(self.seed)
        # Time normalization (1-indexed)
        min_time = self.df[self.time_col].min()
        self.df['t_col'] = self.df[self.time_col] - min_time + 1
        
        # FIX: T0_idx determines the number of PRE-treatment periods.
        # Strict mapping to the exact t_col corresponding to t0, minus 1 just like in R's DiSCo.R
        t0_mapped = self.df[self.df[self.time_col] == self.t0]['t_col'].unique()
        if len(t0_mapped) == 0:
            raise ValueError(f"Behandlungsjahr t0={self.t0} nicht im DataFrame gefunden!")
        self.T0_idx = t0_mapped[0] - 1
        
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

        # Evaluating the quantile functions on the grid "evgrid"
        controls_q = np.zeros((len(self.evgrid), len(controls_data)))
        for jj, ctrl in enumerate(controls_data):
            controls_q[:, jj] = myQuant(ctrl, self.evgrid)

        # Sample grid
        grid_min, grid_max, grid_rand, grid_ord = getGrid(target_data, controls_data, self.G)
            
        if self.mixture:
            res = disco_mixture(controls_data, target_data, grid_min, grid_max, grid_rand, self.M, self.simplex)
            weights = res['weights_opt']
            
            mixture_res = MixtureMethodResult(
                weights=weights,
                distance=res['distance_opt'],
                mean=res['mean']
            )
            disco_res = DiSCoMethodResult(weights=None)
            cdf_t = res['target_order']
            controls_cdf = res['cdf'][:, 1:] # Python solvers returns target at 0, controls at 1:
        else:
            weights = disco_weights_reg(controls_data, target_data, M=self.M, simplex=self.simplex, 
                                        q_min=0, q_max=1)
            mixture_res = None
            disco_res = DiSCoMethodResult(weights=weights)
            
            target_sorted = np.sort(target_data)
            cdf_t = np.searchsorted(target_sorted, grid_ord, side='right') / len(target_data)
            controls_cdf = None

        target_q = myQuant(target_data, self.evgrid)

        target_obj = TargetData(
            cdf=cdf_t,
            grid=grid_ord,
            data=target_data,
            quantiles=target_q
        )

        controls_obj = ControlsData(
            cdf=controls_cdf,
            data=controls_data,
            quantiles=controls_q
        )

        period_result = PeriodResult(
            DiSCo=disco_res,
            mixture=mixture_res,
            target=target_obj,
            controls=controls_obj
        )

        return {'t': t, 'period_result': period_result}

    def fit(self) -> DiSCoResult:
        """
        Run the complete DiSCo estimation across all periods in parallel.
        """
        results = Parallel(n_jobs=self.num_cores)(
            delayed(self._iter_period)(t) for t in self.periods
        )
        
        results = [r for r in results if r is not None]
        
        if not results:
            raise ValueError("No valid periods data found.")
            
        self.results_by_period = {r['t']: r['period_result'] for r in results}
        
        # Average pre-treatment weights
        pre_treat_weights = []
        for t in self.periods:
            if t <= self.T0_idx and t in self.results_by_period:
                if self.mixture:
                    w = self.results_by_period[t].mixture.weights
                else:
                    w = self.results_by_period[t].DiSCo.weights
                pre_treat_weights.append(w)
                
        if not pre_treat_weights:
            raise ValueError("No pre-treatment periods found or calculated.")
            
        self.weights_opt = np.mean(pre_treat_weights, axis=0)

        # calculating the counterfactual target quantiles and CDF
        # R code: bc <- lapply(seq(1:T_max), function(x) DiSCo_bc(controls.q=results.periods[[x]]$controls$quantiles, Weights_DiSCo_avg, evgrid))
        # R code for mixture: cdf <- lapply(seq(1:T_max), function(x) results.periods[[x]]$controls$cdf[,-1] %*% as.vector(Weights_mixture_avg) )
        for t in self.periods:
            if t in self.results_by_period:
                p_res = self.results_by_period[t]
                if self.mixture:
                    cdf_x = p_res.controls.cdf @ self.weights_opt
                    p_res.DiSCo.cdf = cdf_x
                    
                    # Compute quantile from CDF conceptually as inversion
                    # bc_x <- sapply(evgrid, function(y) grid_ord[which(cdf_x >= (y-(1e-5)))[1]])
                    grid_ord = p_res.target.grid
                    bc_x = np.array([grid_ord[np.argmax(cdf_x >= (y - 1e-5))] for y in self.evgrid])
                    p_res.DiSCo.quantile = bc_x
                else:
                    # DiSCo_bc is basically cross-sectional matrix multiplication with evgrid interpolation
                    # In python: controls.q @ weights_opt
                    bc_x = p_res.controls.quantiles @ self.weights_opt
                    p_res.DiSCo.quantile = bc_x
                    
                    grid_ord = p_res.target.grid
                    # Evaluate ECDF of bc_x on grid_ord
                    bc_sorted = np.sort(bc_x)
                    disco_cdf = np.searchsorted(bc_sorted, grid_ord, side='right') / len(bc_sorted)
                    p_res.DiSCo.cdf = disco_cdf

        ci_out = None
        if self.CI:
            ci_out = run_bootstrap_ci(self, replace=True)

        perm_out = None
        if self.perm:
            perm_out = run_permutation_test(self)

        params = DiSCoParams(
            df=self.df,
            id_col_target=self.id_col_target,
            t0=self.t0,
            M=self.M,
            G=self.G,
            CI=self.CI,
            cl=self.cl,
            qmethod=None,
            boot=self.B,
            q_min=self.q_min,
            q_max=self.q_max
        )
        
        return DiSCoResult(
            results_periods=self.results_by_period,
            weights=self.weights_opt,
            CI=ci_out,
            control_ids=self.controls_id,
            perm=perm_out,
            evgrid=self.evgrid,
            params=params
        )
