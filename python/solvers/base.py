from abc import ABC, abstractmethod

class BaseSolver(ABC):
    @abstractmethod
    def fit_weights(self, target, controls, **kwargs):
        """Fit empirical distributions to find optimal synthetic weights."""   
        pass
        
    @abstractmethod
    def evaluate_counterfactual(self, controls, weights, **kwargs):     
        """Returns the specific geometry data for synthetic control matching 
        (e.g., CDF/Quantiles on fixed grids, or pure 2D samples)."""
        pass
        
    @abstractmethod
    def compute_distance(self, target, controls, weights, **kwargs):
        """Compute the distance metric used for inference/permutation."""
        pass
