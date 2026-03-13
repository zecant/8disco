"""
Data Adapter Interface for TradSL

Section 8: BaseAdapter interface for loading historical data.
"""
from abc import ABC, abstractmethod
from typing import Optional, List
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger("tradsl.adapters")


class AdapterError(Exception):
    """Base exception for adapter errors."""
    pass


class SymbolNotFound(AdapterError):
    """Requested symbol not found."""
    pass


class DateRangeTruncated(AdapterError):
    """Requested date range partially unavailable."""
    pass


class BaseAdapter(ABC):
    """
    Abstract base class for data adapters.
    
    Section 8.1 contract:
    - load_historical returns DataFrame with OHLCV columns
    - Index: DatetimeIndex, UTC, sorted ascending
    - Never silently truncate date range
    """
    
    @abstractmethod
    def load_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: str
    ) -> pd.DataFrame:
        """
        Load historical OHLCV data.
        
        Args:
            symbol: Instrument symbol (e.g., 'SPY', 'AAPL')
            start: Start datetime
            end: End datetime  
            frequency: Pandas offset alias ('1d', '1h', '1min', etc.)
        
        Returns:
            DataFrame with columns: open, high, low, close, volume (float64)
            Index: DatetimeIndex (UTC), sorted ascending, no duplicates
        
        Raises:
            SymbolNotFound: Symbol not available
            DateRangeTruncated: Requested range partially unavailable
            AdapterError: Other failures
        """
        pass
    
    @abstractmethod
    def supports_frequency(self, frequency: str) -> bool:
        """Check if adapter supports this frequency."""
        pass
    
    @abstractmethod
    def max_lookback(self, frequency: str) -> Optional[int]:
        """Maximum available bars at frequency, or None if unlimited."""
        pass
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate symbol exists. Override in subclasses."""
        return True
    
    def validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validate DataFrame structure and quality.
        
        Raises AdapterError if:
        - Missing required columns
        - NaN in close column
        - Zero/negative close
        - Negative volume (zero is warning)
        - Duplicate timestamps
        
        Warns if:
        - Zero volume
        - DataFrame is empty (no data for period)
        """
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise AdapterError(f"Missing columns: {missing}")
        
        if len(df) == 0:
            logger.warning("Empty DataFrame - no data for the requested period")
            return
        
        if df['close'].isna().any():
            raise AdapterError("NaN values in close column")
        
        if bool((df['close'] <= 0).any()):
            raise AdapterError("Zero or negative close prices")
        
        if bool((df['volume'] < 0).any()):
            raise AdapterError("Negative volume is not allowed")
        
        zero_vol_count = int((df['volume'] == 0).sum())
        if zero_vol_count > 0:
            logger.warning(f"Zero volume detected in {zero_vol_count} bars")
        
        if df.index.duplicated().any():
            raise AdapterError("Duplicate timestamps in data")
        
        if not df.index.is_monotonic_increasing:
            raise AdapterError("Timestamps not sorted ascending - adapter should sort before returning")


# Valid frequency strings (pandas offset aliases)
VALID_FREQUENCIES = {
    '1min', '5min', '15min', '30min',
    '1h', '4h', '1d', '1wk', '1mo'
}


def validate_frequency(frequency: str) -> bool:
    """Check if frequency string is valid."""
    return frequency in VALID_FREQUENCIES
