import pandas as pd
from typing import Optional

from tradsl.functions import Function


class EMA(Function):
    """Exponential Moving Average."""
    
    def __init__(self, window: int = 20):
        self.window = window
        self.alpha = 2 / (window + 1)
        self._ema = None
    
    def apply(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        if len(data) < self.window:
            return None
        values = data.iloc[:, 0].values
        if self._ema is None:
            self._ema = values[-1]
        else:
            self._ema = self.alpha * values[-1] + (1 - self.alpha) * self._ema
        return pd.DataFrame({data.columns[0]: [self._ema]})


class PairwiseCorrelation(Function):
    """Pearson correlation between two series over a rolling window."""
    
    def __init__(self, window: int = 20):
        self.window = window
    
    def apply(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        if len(data) < self.window:
            return None
        col1 = data.iloc[:, 0]
        col2 = data.iloc[:, 1]
        corr = col1.rolling(self.window).corr(col2).iloc[-1]
        return pd.DataFrame({f"{data.columns[0]}_{data.columns[1]}_corr": [corr]})
