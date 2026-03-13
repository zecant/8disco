"""
TestData Adapter for synthetic data generation.

Used for E2E testing without requiring network access or real market data.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List

from tradsl.adapters import BaseAdapter
from tradsl.exceptions import AdapterError


class TestDataAdapter(BaseAdapter):
    """
    Synthetic data adapter for testing.
    
    Generates configurable OHLCV data with controllable:
    - Trend (up/down/sideways)
    - Volatility
    - Noise level
    - Number of bars
    
    Useful for E2E tests and benchmarking.
    """
    
    SUPPORTED_FREQUENCIES = {'1min', '5min', '15min', '30min', '1h', '4h', '1d', '1wk'}
    
    def __init__(
        self,
        seed: int = 42,
        trend: float = 0.0,
        volatility: float = 0.02,
        noise: float = 0.5
    ):
        """
        Initialize test data adapter.
        
        Args:
            seed: Random seed for reproducibility
            trend: Drift per bar (0.0 = sideways, positive = uptrend)
            volatility: Volatility factor (0.02 = 2% std per bar)
            noise: Noise multiplier (0.0 = pure GBM, 1.0 = noisy)
        """
        self.seed = seed
        self.trend = trend
        self.volatility = volatility
        self.noise = noise
        self._rng = np.random.default_rng(seed)
    
    def load_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: str
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data.
        
        Args:
            symbol: Symbol identifier (used for index)
            start: Start datetime
            end: End datetime
            frequency: Pandas offset alias
            
        Returns:
            DataFrame with columns: open, high, low, close, volume
            
        Raises:
            AdapterError: If frequency not supported
        """
        if not self.supports_frequency(frequency):
            raise AdapterError(
                f"Unsupported frequency: {frequency}",
                symbol=symbol,
                requested_start=start,
                requested_end=end
            )
        
        bars = self._generate_bars(start, end, frequency)
        
        df = pd.DataFrame(bars, columns=['open', 'high', 'low', 'close', 'volume'])
        df.index = pd.DatetimeIndex([b['timestamp'] for b in bars], tz='UTC')
        
        return df
    
    def supports_frequency(self, frequency: str) -> bool:
        """Check if frequency is supported."""
        return frequency in self.SUPPORTED_FREQUENCIES
    
    def max_lookback(self, frequency: str) -> Optional[int]:
        """Return maximum bars (unlimited for synthetic)."""
        return None
    
    def validate_symbol(self, symbol: str) -> bool:
        """Always valid for synthetic data."""
        return True
    
    def _generate_bars(
        self,
        start: datetime,
        end: datetime,
        frequency: str
    ) -> List[dict]:
        """Generate synthetic OHLCV bars using geometric Brownian motion."""
        
        n_bars = self._calculate_bar_count(start, end, frequency)
        
        if n_bars <= 0:
            raise AdapterError(
                f"Invalid date range: start={start}, end={end}",
                symbol="TEST",
                requested_start=start,
                requested_end=end
            )
        
        dt = self._frequency_to_delta(frequency)
        timestamps = [start + i * dt for i in range(n_bars)]
        
        log_returns = self._rng.normal(
            loc=self.trend * dt.total_seconds() / 86400,
            scale=self.volatility * np.sqrt(dt.total_seconds() / 86400),
            size=n_bars
        )
        
        if self.noise > 0:
            noise = self._rng.normal(0, self.noise * self.volatility, size=n_bars)
            log_returns = log_returns + noise
        
        prices = 100 * np.exp(np.cumsum(log_returns))
        
        bars = []
        for i, (ts, close) in enumerate(zip(timestamps, prices)):
            high_factor = 1.0 + abs(self._rng.normal(0, self.volatility / 2))
            low_factor = 1.0 - abs(self._rng.normal(0, self.volatility / 2))
            open_price = close * (1 + self._rng.normal(0, self.volatility / 4))
            
            high = max(open_price, close) * high_factor
            low = min(open_price, close) * low_factor
            
            volume = int(self._rng.lognormal(10, 1) * 1000)
            
            bars.append({
                'timestamp': ts,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        return bars
    
    def _calculate_bar_count(self, start: datetime, end: datetime, frequency: str) -> int:
        """Calculate number of bars for given date range and frequency."""
        delta = end - start
        
        freq_to_hours = {
            '1min': 1/60,
            '5min': 5/60,
            '15min': 15/60,
            '30min': 30/60,
            '1h': 1,
            '4h': 4,
            '1d': 24,
            '1wk': 24 * 7,
        }
        
        hours_per_bar = freq_to_hours.get(frequency, 24)
        total_hours = delta.total_seconds() / 3600
        
        return max(1, int(total_hours / hours_per_bar))
    
    def _frequency_to_delta(self, frequency: str) -> timedelta:
        """Convert frequency string to timedelta."""
        freq_to_delta = {
            '1min': timedelta(minutes=1),
            '5min': timedelta(minutes=5),
            '15min': timedelta(minutes=15),
            '30min': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '1d': timedelta(days=1),
            '1wk': timedelta(weeks=1),
        }
        return freq_to_delta.get(frequency, timedelta(days=1))


def create_test_adapter(
    trend: float = 0.0,
    volatility: float = 0.02,
    seed: int = 42
) -> TestDataAdapter:
    """
    Factory function to create a configured test adapter.
    
    Args:
        trend: Drift per bar (0.0 = sideways)
        volatility: Volatility factor (0.02 = 2%)
        seed: Random seed
        
    Returns:
        Configured TestDataAdapter
    """
    return TestDataAdapter(trend=trend, volatility=volatility, seed=seed)
