from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

from tradsl.functions import Function


class MLFunction(Function):
    """Abstract base class for ML-powered functions."""
    
    def __init__(
        self,
        model=None,
        warmup: int = 50,
        **kwargs
    ):
        self.model = model
        self.warmup = warmup
        self._ticks = 0
        self._is_ready = False
        self._model_params = kwargs
    
    def apply(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Apply function after warmup period."""
        self._ticks += 1
        if self._ticks < self.warmup:
            return None
        if not self._is_ready:
            self._is_ready = True
        return self._predict(data)
    
    @abstractmethod
    def _predict(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Subclasses implement their prediction logic."""
        pass


class Regressor(MLFunction):
    """ML function for regression tasks."""
    
    def _predict(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.model is None:
            return None
        X = self._extract_features(data)
        if X is None:
            return None
        try:
            pred = self.model.predict(X)
            return pd.DataFrame({self.__class__.__name__.lower(): pred})
        except Exception:
            return None
    
    def _extract_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract features from input DataFrame. Override for custom extraction."""
        if data.empty or len(data) < 1:
            return None
        return data


class Classifier(MLFunction):
    """ML function for classification tasks."""
    
    def _predict(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.model is None:
            return None
        X = self._extract_features(data)
        if X is None:
            return None
        try:
            pred = self.model.predict(X)
            return pd.DataFrame({self.__class__.__name__.lower(): pred})
        except Exception:
            return None
    
    def _extract_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract features from input DataFrame. Override for custom extraction."""
        if data.empty or len(data) < 1:
            return None
        return data


class Agent(MLFunction):
    """ML function for reinforcement learning / agent tasks.
    
    Agents output a DataFrame with columns:
    - action: int (0=buy, 1=hold, 2=sell)
    - confidence: float (0.0 to 1.0)
    - asset: str (ticker symbol)
    """
    
    def _predict(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.model is None:
            return self._default_predict(data)
        state = self._extract_state(data)
        if state is None:
            return None
        try:
            action = self.model.predict(state)
            return self._format_output(action, 1.0, "UNKNOWN")
        except Exception:
            return self._default_predict(data)
    
    def _default_predict(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Fallback prediction when no model is available. Override for custom behavior."""
        return None
    
    def _extract_state(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract state from input DataFrame. Override for custom state."""
        if data.empty or len(data) < 1:
            return None
        return data
    
    def _format_output(self, action, confidence, asset) -> pd.DataFrame:
        """Format agent output as DataFrame."""
        return pd.DataFrame({
            "action": [int(action)],
            "confidence": [float(confidence)],
            "asset": [str(asset)],
        })
