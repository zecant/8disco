"""
Feature Functions for TradSL

Pure functions that compute features from circular buffer arrays.
Each function conforms to Section 7.4 contract:
- Input: np.ndarray of shape (window,), oldest to newest, arr[-1] = current bar
- Output: float or dict for multi-output
- Must return None for insufficient/invalid data
"""
from typing import Optional, Dict, Any, Callable, List
import numpy as np
from dataclasses import dataclass
from enum import Enum


class FeatureCategory(Enum):
    """Feature classification for date-blind mode (Section 16)."""
    SAFE = "safe"
    CYCLICAL_STRUCTURAL = "cyclical_structural"
    DATE_DERIVED = "date_derived"


@dataclass
class FunctionSpec:
    """Specification for a registered feature function."""
    func: Callable
    category: FeatureCategory
    description: str
    min_lookback: int
    output_type: type = float


def log_returns(arr: np.ndarray, period: int = 1, **params) -> Optional[float]:
    """
    Compute log returns over specified period.
    
    Args:
        arr: Price array, oldest to newest
        period: Number of bars to look back (default 1)
    
    Returns:
        Log return: ln(price_t / price_{t-period})
        None if insufficient data or invalid
    """
    if arr is None or len(arr) < period + 1:
        return None
    
    arr = arr[-period-1:]
    if np.any(np.isnan(arr)):
        return None
    
    current = arr[-1]
    previous = arr[0]
    
    if previous <= 0 or current <= 0:
        return None
    
    return np.log(current / previous)


def sma(arr: np.ndarray, period: int = 20, **params) -> Optional[float]:
    """
    Simple moving average.
    
    Args:
        arr: Price array, oldest to newest
        period: Window size
    
    Returns:
        SMA value or None if insufficient data
    """
    if arr is None or len(arr) < period:
        return None
    
    window = arr[-period:]
    if np.any(np.isnan(window)):
        return None
    
    return float(np.mean(window))


def ema(arr: np.ndarray, period: int = 20, **params) -> Optional[float]:
    """
    Exponential moving average.
    
    Uses alpha = 2 / (period + 1) for EMA calculation.
    
    Args:
        arr: Price array, oldest to newest
        period: Window size (affects alpha)
    
    Returns:
        EMA value or None if insufficient data
    """
    if arr is None or len(arr) < period:
        return None
    
    arr = arr[-period:]
    if np.any(np.isnan(arr)):
        return None
    
    alpha = 2.0 / (period + 1)
    
    ema_val = arr[0]
    for i in range(1, len(arr)):
        ema_val = alpha * arr[i] + (1 - alpha) * ema_val
    
    return float(ema_val)


def rsi(arr: np.ndarray, period: int = 14, **params) -> Optional[float]:
    """
    Relative Strength Index.
    
    Args:
        arr: Price array, oldest to newest
        period: Window size for calculation
    
    Returns:
        RSI value in [0, 100] or None if insufficient data
    """
    if arr is None or len(arr) < period + 1:
        return None
    
    prices = arr[-period-1:]
    if np.any(np.isnan(prices)):
        return None
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return float(rsi)


def rolling_std(arr: np.ndarray, period: int = 20, **params) -> Optional[float]:
    """
    Rolling standard deviation.
    
    Args:
        arr: Price array, oldest to newest
        period: Window size
    
    Returns:
        Standard deviation or None if insufficient data
    """
    if arr is None or len(arr) < period:
        return None
    
    window = arr[-period:]
    if np.any(np.isnan(window)):
        return None
    
    return float(np.std(window, ddof=0))


def volatility(arr: np.ndarray, period: int = 20, **params) -> Optional[float]:
    """
    Realized volatility (annualized).
    
    Args:
        arr: Price array (log returns or prices)
        period: Window size
    
    Returns:
        Annualized volatility (assumes 252 trading days)
        None if insufficient data
    """
    if arr is None or len(arr) < period:
        return None
    
    window = arr[-period:]
    if np.any(np.isnan(window)):
        return None
    
    std = np.std(window, ddof=0)
    
    return float(std * np.sqrt(252))


def roc(arr: np.ndarray, period: int = 10, **params) -> Optional[float]:
    """
    Rate of change: (current - past) / past * 100
    
    Args:
        arr: Price array, oldest to newest
        period: Bars to look back
    
    Returns:
        ROC percentage or None if insufficient data
    """
    if arr is None or len(arr) < period + 1:
        return None
    
    current = arr[-1]
    past = arr[0]
    
    if np.isnan(current) or np.isnan(past) or past == 0:
        return None
    
    return float((current - past) / past * 100)


def zscore(arr: np.ndarray, period: int = 20, **params) -> Optional[float]:
    """
    Rolling z-score: (value - mean) / std
    
    Args:
        arr: Price array, oldest to newest
        period: Window size
    
    Returns:
        Z-score or None if insufficient data
    """
    if arr is None or len(arr) < period:
        return None
    
    window = arr[-period:]
    if np.any(np.isnan(window)):
        return None
    
    mean = np.mean(window)
    std = np.std(window, ddof=0)
    
    if std == 0:
        return 0.0
    
    current = arr[-1]
    return float((current - mean) / std)


# Registry of all implemented functions
FUNCTION_REGISTRY: Dict[str, FunctionSpec] = {
    'log_returns': FunctionSpec(
        func=log_returns,
        category=FeatureCategory.SAFE,
        description="Log returns over N periods",
        min_lookback=2
    ),
    'sma': FunctionSpec(
        func=sma,
        category=FeatureCategory.SAFE,
        description="Simple moving average",
        min_lookback=2
    ),
    'ema': FunctionSpec(
        func=ema,
        category=FeatureCategory.SAFE,
        description="Exponential moving average",
        min_lookback=2
    ),
    'rsi': FunctionSpec(
        func=rsi,
        category=FeatureCategory.SAFE,
        description="Relative Strength Index",
        min_lookback=15
    ),
    'rolling_std': FunctionSpec(
        func=rolling_std,
        category=FeatureCategory.SAFE,
        description="Rolling standard deviation",
        min_lookback=2
    ),
    'volatility': FunctionSpec(
        func=volatility,
        category=FeatureCategory.SAFE,
        description="Annualized realized volatility",
        min_lookback=2
    ),
    'roc': FunctionSpec(
        func=roc,
        category=FeatureCategory.SAFE,
        description="Rate of change percentage",
        min_lookback=2
    ),
    'zscore': FunctionSpec(
        func=zscore,
        category=FeatureCategory.SAFE,
        description="Rolling z-score",
        min_lookback=2
    ),
}


def get_function(name: str) -> Optional[FunctionSpec]:
    """Get function specification by name."""
    return FUNCTION_REGISTRY.get(name)


def compute_function(name: str, arr: np.ndarray, **params) -> Optional[float]:
    """
    Compute a function by name.
    
    Args:
        name: Function name
        arr: Input array
        **params: Function parameters
    
    Returns:
        Computed value or None
    """
    spec = FUNCTION_REGISTRY.get(name)
    if spec is None:
        raise ValueError(f"Unknown function: {name}")
    
    return spec.func(arr, **params)
