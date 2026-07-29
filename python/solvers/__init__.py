from .base import BaseSolver
from .quantile1d import Quantile1DSolver
from .mixture import MixtureSolver
from .sliced import SlicedWassersteinSolver
from .tangential import TangentialWassersteinSolver
from .energy import EnergySolver

__all__ = [
    'BaseSolver',
    'Quantile1DSolver',
    'MixtureSolver',
    'SlicedWassersteinSolver',
    'TangentialWassersteinSolver',
    'EnergySolver'
]
