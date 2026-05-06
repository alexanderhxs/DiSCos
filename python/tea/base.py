import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict
from ..models import DiSCoResult, DiSCoTEAResult

class BaseTEA(ABC):
    def __init__(self, disco: DiSCoResult, agg: str, graph: bool, t_plot: Optional[List[int]], xlim, ylim, samples):
        self.disco = disco
        self.agg = agg
        self.graph = graph
        self.t_plot = t_plot
        self.xlim = xlim
        self.ylim = ylim
        self.samples = samples
        
        self.df = disco.params.df
        self.t0 = disco.params.t0

        self.periods = sorted(list(self.disco.results_periods.keys()))
        self._raw_mapper = {t: t for t in self.periods}
        if isinstance(self.df, pd.DataFrame):
            for col in self.df.columns:
                if col != 't_col' and self.t0 in self.df[col].values:
                    self._raw_mapper = dict(zip(self.df['t_col'], self.df[col]))
                    break
        self.t_mapper = {k: int(v) if isinstance(v, float) and v.is_integer() else v for k, v in self._raw_mapper.items()}
        self.t_start = min(self.periods)
        self.t_max = max(self._raw_mapper.values())
        if self.t_plot is None:
            self.t_plot = [self.t_mapper[p] for p in self.periods]

        self.CI = disco.params.CI
        self.q_min = disco.params.q_min
        self.q_max = disco.params.q_max

    @abstractmethod
    def evaluate(self) -> DiSCoTEAResult:
        pass


def plot_dist_over_time(
    cdf_centered: dict, 
    grid_cdf: np.ndarray, 
    t_start: int, 
    t_max: int, 
    CI: bool, 
    ci_lower: dict, 
    ci_upper: dict, 
    ylim=None, 
    xlim=None, 
    cdf=True, 
    xlab="Distribution Difference", 
    ylab="CDF", 
    obs_line=None, 
    t_plot=None
):
    if t_plot is None:
        t_plot = list(cdf_centered.keys())
    else:
        t_plot = [t for t in t_plot if t in cdf_centered]
        
    n_plots = len(t_plot)
    if n_plots == 0:
        return None
        
    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 3 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
        
    fig.suptitle("Distribution of Treatment Effects Over Time")
    
    for ax, t in zip(axes, t_plot):
        y = cdf_centered[t]
        x = grid_cdf
        
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
            
        ax.plot(x, y, color="black", linestyle="-")
        
        if obs_line is not None and t in obs_line:
            ax.plot(x, obs_line[t], color="blue", linestyle="--", label="Observed")
            
        ax.axhline(0, color="grey", linestyle="-")
        if cdf:
            ax.axhline(1, color="grey", linestyle="-")
            
        if CI and ci_lower is not None and ci_upper is not None:
            ax.plot(x, ci_lower[t], color="grey", linestyle="--")
            ax.plot(x, ci_upper[t], color="grey", linestyle="--")
            
        ax.set_title(f"Time: {t}")
        ax.set_ylabel(ylab)
        ax.grid(True)
        
    axes[-1].set_xlabel(xlab)
    plt.tight_layout()
    
    return fig


def disco_tea(
    disco: DiSCoResult,
    agg: str = "quantileDiff",
    graph: bool = True,
    t_plot: Optional[List[int]] = None,
    xlim=None,
    ylim=None,
    samples=[0.25, 0.5, 0.75]
):
    from .classic import ClassicTEA
    from .marginals import MarginalTEA
    from .transport_map import TransportMapTEA
    from .density_ratio import DensityRatioTEA
    from .multivariate import MultivariateTEA

    if agg in ["wasserstein_dist"]:
        strategy = MultivariateTEA(disco, agg, graph, t_plot, xlim, ylim, samples)
    elif agg in ["cdfDiff", "cdf", "quantileDiff", "quantile"]:
        strategy = ClassicTEA(disco, agg, graph, t_plot, xlim, ylim, samples)
    elif agg == "density_ratio":
        strategy = DensityRatioTEA(disco, agg, graph, t_plot, xlim, ylim, samples)
    elif agg == "transport_map":
        strategy = TransportMapTEA(disco, agg, graph, t_plot, xlim, ylim, samples)
    elif agg == "marginals":
        strategy = MarginalTEA(disco, agg, graph, t_plot, xlim, ylim, samples)
    else:
        raise ValueError(f"Unknown aggregation method: {agg}")
    
    return strategy.evaluate()
