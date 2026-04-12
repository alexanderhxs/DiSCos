from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

@dataclass
class PermutResult:
    distp: "np.ndarray"
    distt: "np.ndarray"
    p_overall: float
    J_1: int
    q_min: float
    q_max: float
    plot: Any = None

# ---------------------------------------------------------
# Period-spezifische Dataclasses (In R: results.periods)
# ---------------------------------------------------------

@dataclass
class TargetData:
    cdf: Optional[np.ndarray]
    grid: np.ndarray
    data: np.ndarray
    quantiles: np.ndarray

@dataclass
class ControlsData:
    cdf: Optional[np.ndarray]
    data: List[np.ndarray]
    quantiles: np.ndarray

@dataclass
class DiSCoMethodResult:
    weights: Optional[np.ndarray]
    quantile: Optional[np.ndarray] = None
    cdf: Optional[np.ndarray] = None

@dataclass
class MixtureMethodResult:
    weights: Optional[np.ndarray]
    distance: Optional[float] = None
    mean: Optional[np.ndarray] = None

@dataclass
class PeriodResult:
    DiSCo: DiSCoMethodResult
    mixture: Optional[MixtureMethodResult]
    target: TargetData
    controls: ControlsData

# ---------------------------------------------------------
# Confidence Interval Dataclasses (In R: parseBoots Output)
# ---------------------------------------------------------

@dataclass
class CIBand:
    lower: np.ndarray
    upper: np.ndarray
    se: np.ndarray

@dataclass
class CIWeights:
    lower: np.ndarray
    upper: np.ndarray

@dataclass
class CIBootmat:
    quantile: np.ndarray
    cdf: np.ndarray
    quantile_diff: np.ndarray
    cdf_diff: np.ndarray

@dataclass
class CIResult:
    quantile: CIBand
    cdf: CIBand
    quantile_diff: CIBand
    cdf_diff: CIBand
    weights: CIWeights
    bootmat: CIBootmat

# ---------------------------------------------------------
# Permutation Test Dataclasses (In R: permut Object)
# ---------------------------------------------------------

@dataclass
class PermutResult:
    distp: List[float]       # Liste der quadratischen Wasserstein-Distanzen für Permutationen
    distt: np.ndarray        # Vektor der Distanzen für die tatsächliche Target-Unit
    p_overall: float         # Berechneter p-Value
    J_1: int                 # Anzahl der Control Units (Anzahl Permutationen)
    q_min: float
    q_max: float
    plot: Any                # Optional: Matplotlib/Seaborn Figure (entspricht ggplot)

# ---------------------------------------------------------
# Haupt-Output und Parameter Dataclasses
# ---------------------------------------------------------

@dataclass
class DiSCoParams:
    df: pd.DataFrame
    id_col_target: Any
    t0: Any
    M: int
    G: int
    CI: bool
    cl: float
    qmethod: Optional[str]
    boot: int
    q_min: float
    q_max: float

@dataclass
class DiSCoResult:
    results_periods: Dict[int, PeriodResult]
    weights: np.ndarray
    CI: Optional[CIResult]
    control_ids: List[Any]
    perm: Optional[PermutResult]
    evgrid: np.ndarray
    params: DiSCoParams

# ---------------------------------------------------------
# TEA Output
# ---------------------------------------------------------

@dataclass
class DiSCoTEAResult:
    agg: str
    treats: Dict[int, np.ndarray]
    grid: np.ndarray
    ses: Optional[Dict[int, np.ndarray]]
    ci_lower: Optional[Dict[int, np.ndarray]]
    ci_upper: Optional[Dict[int, np.ndarray]]
    t0: Any
    cl: float
    N: int
    J: int
    agg_df: Optional[pd.DataFrame]
    perm: Optional[PermutResult]
    plot: Any

    def __str__(self):
        out = "\nCall:\nDiSCoTEA\n\n"
        
        if self.agg in ["quantile", "cdf"]:
            out += "No treatment effects to summarize, set graph=True in function call or specify a treatment effect option in `agg`.\n"
            return out
            
        if self.agg_df is not None:
            out += "--- Treatment Effects ---\n"
            out += self.agg_df.to_string(index=False) + "\n\n"
            
        if self.perm is not None:
            out += "--- Permutation Test ---\n"
            out += f"p-value: {self.perm.p_overall:.4f}\n"
            
        return out
        
    def summary(self):
        print(self.__str__())
