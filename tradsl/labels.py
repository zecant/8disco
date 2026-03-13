"""
Label Functions for TradSL

Label functions generate supervised training targets for trainable models.
Section 10.5 contract:
- Input: features array, prices array, bar_index, **params
- Output: np.ndarray of labels aligned with features
- Uses FUTURE data (acceptable for labels, not for features)
"""
from typing import Optional, Dict, Any, Callable
import numpy as np
from dataclasses import dataclass


@dataclass
class LabelSpec:
    """Specification for a registered label function."""
    func: Callable
    description: str
    requires_future_bars: int


def forward_return_sign(
    features: np.ndarray,
    prices: np.ndarray,
    bar_index: int,
    period: int = 1,
    **params
) -> np.ndarray:
    """
    Label: sign of forward return over N periods.
    
    +1 if return > 0
    -1 if return < 0
    0 if return == 0
    
    Args:
        features: Unused (kept for interface compatibility)
        prices: Close prices, oldest to newest
        bar_index: Current bar index
        period: Forward period to look (default 1)
    
    Returns:
        Array of labels: +1, -1, or 0
        NaN for last 'period' bars (outcome not yet known)
    """
    n = len(prices)
    labels = np.full(n, np.nan)
    
    for i in range(n - period):
        current_price = prices[i]
        future_price = prices[i + period]
        
        if current_price <= 0 or future_price <= 0:
            labels[i] = np.nan
            continue
        
        ret = (future_price - current_price) / current_price
        
        if ret > 0:
            labels[i] = 1.0
        elif ret < 0:
            labels[i] = -1.0
        else:
            labels[i] = 0.0
    
    return labels


def forward_return(
    features: np.ndarray,
    prices: np.ndarray,
    bar_index: int,
    period: int = 1,
    log: bool = False,
    **params
) -> np.ndarray:
    """
    Label: forward return over N periods (continuous).
    
    Args:
        features: Unused
        prices: Close prices
        bar_index: Current bar index  
        period: Forward period
        log: If True, return log return
    
    Returns:
        Array of returns
        NaN for last 'period' bars
    """
    n = len(prices)
    labels = np.full(n, np.nan)
    
    for i in range(n - period):
        current_price = prices[i]
        future_price = prices[i + period]
        
        if current_price <= 0 or future_price <= 0:
            labels[i] = np.nan
            continue
        
        if log:
            labels[i] = np.log(future_price / current_price)
        else:
            labels[i] = (future_price - current_price) / current_price
    
    return labels


def triple_barrier(
    features: np.ndarray,
    prices: np.ndarray,
    bar_index: int,
    upper_barrier: float = 0.02,
    lower_barrier: float = 0.01,
    time_barrier: int = 20,
    **params
) -> np.ndarray:
    """
    Triple barrier labels: which barrier is hit first?
    
    +1: upper barrier hit first (take profit)
    -1: lower barrier hit first (stop loss)  
    0: time barrier (no directional resolution)
    
    Args:
        features: Feature array for reference
        prices: Close prices
        bar_index: Current bar index
        upper_barrier: Upper barrier as fraction (e.g., 0.02 = 2%)
        lower_barrier: Lower barrier as fraction
        time_barrier: Max bars to hold
    
    Returns:
        Array of labels: +1, -1, 0, or NaN
    """
    n = len(prices)
    labels = np.full(n, np.nan)
    
    entry_price = prices[0]
    
    for i in range(n):
        hit = False
        
        for j in range(i + 1, min(i + time_barrier + 1, n)):
            if j >= n:
                break
                
            current_price = prices[j]
            
            upper_hit = (current_price - entry_price) / entry_price >= upper_barrier
            lower_hit = (entry_price - current_price) / entry_price >= lower_barrier
            
            if upper_hit:
                labels[i] = 1.0
                hit = True
                break
            elif lower_hit:
                labels[i] = -1.0
                hit = True
                break
        
        if not hit:
            labels[i] = 0.0
        
        if i + 1 < n:
            entry_price = prices[i + 1]
    
    return labels


LABEL_REGISTRY: Dict[str, LabelSpec] = {
    'forward_return_sign': LabelSpec(
        func=forward_return_sign,
        description="Sign of forward return: +1, -1, or 0",
        requires_future_bars=1
    ),
    'forward_return': LabelSpec(
        func=forward_return,
        description="Continuous forward return",
        requires_future_bars=1
    ),
    'triple_barrier': LabelSpec(
        func=triple_barrier,
        description="Triple barrier labels: +1 (upper), -1 (lower), 0 (time)",
        requires_future_bars=20
    ),
}


def get_label_function(name: str) -> Optional[LabelSpec]:
    """Get label function specification by name."""
    return LABEL_REGISTRY.get(name)


def compute_labels(
    name: str,
    features: np.ndarray,
    prices: np.ndarray,
    bar_index: int,
    **params
) -> np.ndarray:
    """
    Compute labels using a registered label function.
    
    Args:
        name: Label function name
        features: Feature array
        prices: Price array
        bar_index: Current bar index
        **params: Additional parameters
    
    Returns:
        Array of labels
    """
    spec = LABEL_REGISTRY.get(name)
    if spec is None:
        raise ValueError(f"Unknown label function: {name}")
    
    return spec.func(features, prices, bar_index, **params)
