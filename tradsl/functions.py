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


def spread(arr1: np.ndarray, arr2: np.ndarray, **params) -> Optional[float]:
    """
    Compute spread: arr1 - arr2 (current values).
    
    Args:
        arr1: First series (current value used)
        arr2: Second series (current value used)
    
    Returns:
        Spread value (arr1 - arr2)
    """
    if arr1 is None or arr2 is None:
        return None
    if len(arr1) < 1 or len(arr2) < 1:
        return None
    v1 = arr1[-1]
    v2 = arr2[-1]
    if np.isnan(v1) or np.isnan(v2):
        return None
    return float(v1 - v2)


def ratio(arr1: np.ndarray, arr2: np.ndarray, **params) -> Optional[float]:
    """
    Compute ratio: arr1 / arr2 (current values).
    
    Args:
        arr1: First series (numerator)
        arr2: Second series (denominator)
    
    Returns:
        Ratio value (arr1 / arr2)
    """
    if arr1 is None or arr2 is None:
        return None
    if len(arr1) < 1 or len(arr2) < 1:
        return None
    v1 = arr1[-1]
    v2 = arr2[-1]
    if np.isnan(v1) or np.isnan(v2) or v2 == 0:
        return None
    return float(v1 / v2)


def rolling_correlation(arr1: np.ndarray, arr2: np.ndarray, period: int = 20, **params) -> Optional[float]:
    """
    Compute rolling Pearson correlation between two series.
    
    Args:
        arr1: First series (full array)
        arr2: Second series (full array)
        period: Rolling window size
    
    Returns:
        Correlation coefficient in [-1, 1]
    """
    if arr1 is None or arr2 is None:
        return None
    if len(arr1) < period or len(arr2) < period:
        return None
    
    a1 = arr1[-period:]
    a2 = arr2[-period:]
    
    if np.any(np.isnan(a1)) or np.any(np.isnan(a2)):
        return None
    
    cov = np.cov(a1, a2, ddof=0)[0, 1]
    std1 = np.std(a1, ddof=0)
    std2 = np.std(a2, ddof=0)
    
    if std1 == 0 or std2 == 0:
        return None
    
    return float(cov / (std1 * std2))


def beta(asset_arr: np.ndarray, benchmark_arr: np.ndarray, period: int = 60, **params) -> Optional[float]:
    """
    Compute rolling beta: covariance(asset, benchmark) / variance(benchmark).
    
    Args:
        asset_arr: Asset returns/volatility series (full array)
        benchmark_arr: Benchmark returns/volatility series (full array)
        period: Rolling window size
    
    Returns:
        Beta coefficient
    """
    if asset_arr is None or benchmark_arr is None:
        return None
    if len(asset_arr) < period or len(benchmark_arr) < period:
        return None
    
    a = asset_arr[-period:]
    b = benchmark_arr[-period:]
    
    if np.any(np.isnan(a)) or np.any(np.isnan(b)):
        return None
    
    var_b = np.var(b, ddof=0)
    
    if var_b == 0:
        return None
    
    cov = np.cov(a, b, ddof=0)[0, 1]
    
    return float(cov / var_b)


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
    'spread': FunctionSpec(
        func=spread,
        category=FeatureCategory.SAFE,
        description="Spread between two series: arr1 - arr2",
        min_lookback=1
    ),
    'ratio': FunctionSpec(
        func=ratio,
        category=FeatureCategory.SAFE,
        description="Ratio of two series: arr1 / arr2",
        min_lookback=1
    ),
    'rolling_correlation': FunctionSpec(
        func=rolling_correlation,
        category=FeatureCategory.SAFE,
        description="Rolling Pearson correlation between two series",
        min_lookback=20
    ),
    'beta': FunctionSpec(
        func=beta,
        category=FeatureCategory.SAFE,
        description="Rolling beta: covariance(asset, benchmark) / variance(benchmark)",
        min_lookback=60
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
