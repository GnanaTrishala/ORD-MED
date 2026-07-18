from .predictor import Predictor

# Alias for backwards compatibility
InferenceEngine = Predictor

__all__ = [
    "Predictor",
    "InferenceEngine"
]
