import copy
import numpy as np
from ..utils.utils import myQuant
from .base import BaseTEA

class MarginalTEA(BaseTEA):
    def evaluate(self):
        first_period = self.periods[0]
        
        if not self.disco.params.is_multivariate:
            from .classic import ClassicTEA
            strategy = ClassicTEA(self.disco, "quantileDiff", self.graph, self.t_plot, self.xlim, self.ylim, self.samples)
            return {0: strategy.evaluate()}

        target_data = self.disco.results_periods[first_period].target.data
        num_dims = target_data.shape[1]
        marginal_results = {}
        
        from .classic import ClassicTEA
        
        for d in range(num_dims):
            disco_d = copy.copy(self.disco)
            disco_d.results_periods = copy.deepcopy(self.disco.results_periods)
            
            for t in self.periods:
                p_res = disco_d.results_periods[t]
                
                if p_res.target.data is not None and p_res.target.data.ndim > 1:
                    p_res.target.data = p_res.target.data[:, d]
                
                if p_res.target.data is not None:
                    p_res.target.quantiles = myQuant(p_res.target.data, self.disco.evgrid)
                    
                if p_res.controls.data is not None and self.disco.weights is not None:
                    controls_q = np.zeros((len(self.disco.evgrid), len(p_res.controls.data)))
                    for jj, ctrl in enumerate(p_res.controls.data):
                        ctrl_d = ctrl[:, d] if ctrl.ndim > 1 else ctrl
                        controls_q[:, jj] = myQuant(ctrl_d, self.disco.evgrid)
                    p_res.DiSCo.quantile = controls_q @ self.disco.weights

                if p_res.target.cdf is not None and p_res.target.cdf.ndim > 1:
                    p_res.target.cdf = p_res.target.cdf[:, d]
                if p_res.DiSCo.cdf is not None and p_res.DiSCo.cdf.ndim > 1:
                    p_res.DiSCo.cdf = p_res.DiSCo.cdf[:, d]
                    
            strategy = ClassicTEA(disco_d, "quantileDiff", self.graph, self.t_plot, self.xlim, self.ylim, self.samples)
            marginal_results[d] = strategy.evaluate()
            
        return marginal_results
