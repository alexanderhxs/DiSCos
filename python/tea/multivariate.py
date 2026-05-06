from .base import BaseTEA

class MultivariateTEA(BaseTEA):
    def evaluate(self):
        raise NotImplementedError("Multivariate Treatment Effects (e.g. Sliced Wasserstein Distance over time) not yet implemented.")
