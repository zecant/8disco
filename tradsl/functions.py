"""
Built-in timeseries functions for tradsl DSL.

These functions are automatically available in the DSL context
without requiring manual injection.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional


def sma(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Simple Moving Average.
    
    Args:
        series: Input price series
        window: Lookback window for calculation
        
    Returns:
        Series with SMA values
    """
    return series.rolling(window=window).mean()


def ema(series: pd.Series, span: int = 20) -> pd.Series:
    """
    Exponential Moving Average.
    
    Args:
        series: Input price series
        span: Span for EMA (equivalent to window for SMA)
        
    Returns:
        Series with EMA values
    """
    return series.ewm(span=span, adjust=False).mean()


def rolling_std(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling Standard Deviation.
    
    Args:
        series: Input price series
        window: Lookback window for calculation
        
    Returns:
        Series with rolling std values
    """
    return series.rolling(window=window).std()


def z_score(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Z-Score: (value - SMA) / rolling_std
    
    Measures how many standard deviations the current value is
    from the moving average. Useful for mean reversion strategies.
    
    Args:
        series: Input price series
        window: Lookback window for SMA and std calculation
        
    Returns:
        Series with z-score values
        
    Example:
        z < -2.0 -> oversold (buy signal)
        z > 2.0 -> overbought (sell signal)
    """
    sma_val = sma(series, window)
    std_val = rolling_std(series, window)
    
    # Avoid division by zero
    std_val = std_val.replace(0, np.nan)
    
    return (series - sma_val) / std_val


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.
    
    Momentum oscillator that measures the magnitude of recent price
    changes to evaluate overbought or oversold conditions.
    
    Args:
        series: Input price series
        period: Lookback period for RSI calculation
        
    Returns:
        Series with RSI values (0-100)
        
    Interpretation:
        RSI > 70 -> overbought (potential sell)
        RSI < 30 -> oversold (potential buy)
    """
    delta = series.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))
    
    return rsi_val


def bollinger_bands(
    series: pd.Series, 
    window: int = 20, 
    num_std: float = 2.0
) -> pd.DataFrame:
    """
    Bollinger Bands.
    
    Creates upper and lower bands around a simple moving average.
    
    Args:
        series: Input price series
        window: Lookback window for SMA
        num_std: Number of standard deviations for bands
        
    Returns:
        DataFrame with columns: [middle, upper, lower]
    """
    middle = sma(series, window)
    std = rolling_std(series, window)
    
    return pd.DataFrame({
        'middle': middle,
        'upper': middle + (std * num_std),
        'lower': middle - (std * num_std)
    }, index=series.index)


def returns(series: pd.Series, periods: int = 1) -> pd.Series:
    """
    Simple returns: (price_t - price_t-1) / price_t-1
    
    Args:
        series: Input price series
        periods: Number of periods for return calculation
        
    Returns:
        Series with returns
    """
    return series.pct_change(periods=periods)


def log_returns(series: pd.Series, periods: int = 1) -> pd.Series:
    """
    Log returns: log(price_t / price_t-1)
    
    Args:
        series: Input price series
        periods: Number of periods for return calculation
        
    Returns:
        Series with log returns
    """
    return np.log(series / series.shift(periods))


def momentum(series: pd.Series, window: int = 10) -> pd.Series:
    """
    Momentum: price_t - price_t-window
    
    Args:
        series: Input price series
        window: Lookback window
        
    Returns:
        Series with momentum values
    """
    return series - series.shift(window)


def rate_of_change(series: pd.Series, window: int = 10) -> pd.Series:
    """
    Rate of Change: (price_t / price_t-window) - 1
    
    Args:
        series: Input price series
        window: Lookback window
        
    Returns:
        Series with ROC values (as percentage)
    """
    return ((series / series.shift(window)) - 1) * 100


def macd(
    series: pd.Series, 
    fast_span: int = 12, 
    slow_span: int = 26,
    signal_span: int = 9
) -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence).
    
    Args:
        series: Input price series
        fast_span: Fast EMA span
        slow_span: Slow EMA span
        signal_span: Signal line span
        
    Returns:
        DataFrame with columns: [macd, signal, histogram]
    """
    fast_ema = ema(series, fast_span)
    slow_ema = ema(series, slow_span)
    
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_span)
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }, index=series.index)


def atr(
    high: pd.Series, 
    low: pd.Series, 
    close: pd.Series, 
    period: int = 14
) -> pd.Series:
    """
    Average True Range.
    
    Volatility indicator measuring price movement.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period
        
    Returns:
        Series with ATR values
    """
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    return tr.rolling(window=period).mean()


def generate_labels_from_zscore(
    close: pd.Series,
    window: int = 20,
    buy_threshold: float = -2.0,
    sell_threshold: float = 2.0
) -> pd.Series:
    """
    Generate trading labels from z-score for mean reversion.
    
    Args:
        close: Close price series
        window: Window for z-score calculation
        buy_threshold: Z-score threshold for buy signal
        sell_threshold: Z-score threshold for sell signal
        
    Returns:
        Series with labels: 0=sell, 1=hold, 2=buy
    """
    z = z_score(close, window)
    
    labels = pd.Series(np.ones(len(close), dtype=int), index=close.index)
    labels[z < buy_threshold] = 2   # buy
    labels[z > sell_threshold] = 0  # sell
    
    return labels


def generate_labels_from_rsi(
    close: pd.Series,
    period: int = 14,
    buy_threshold: float = 30,
    sell_threshold: float = 70
) -> pd.Series:
    """
    Generate trading labels from RSI.
    
    Args:
        close: Close price series
        period: RSI period
        buy_threshold: RSI threshold for buy signal
        sell_threshold: RSI threshold for sell signal
        
    Returns:
        Series with labels: 0=sell, 1=hold, 2=buy
    """
    rsi_val = rsi(close, period)
    
    labels = pd.Series(np.ones(len(close), dtype=int), index=close.index)
    labels[rsi_val < buy_threshold] = 2   # buy
    labels[rsi_val > sell_threshold] = 0  # sell
    
    return labels


BUILTIN_FUNCTIONS = {
    'sma': sma,
    'ema': ema,
    'rolling_std': rolling_std,
    'z_score': z_score,
    'zscore': z_score,
    'rsi': rsi,
    'bollinger_bands': bollinger_bands,
    'returns': returns,
    'log_returns': log_returns,
    'momentum': momentum,
    'rate_of_change': rate_of_change,
    'macd': macd,
    'atr': atr,
    'generate_labels_from_zscore': generate_labels_from_zscore,
    'generate_labels_from_rsi': generate_labels_from_rsi,
}
