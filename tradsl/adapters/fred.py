from datetime import datetime
from typing import Optional
import pandas as pd

from .base import BaseAdapter

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False


class FREDAdapter(BaseAdapter):
    """
    Adapter for FRED (Federal Reserve Economic Data) via fredapi.
    
    Supports:
    - Historical series data (load_historical)
    - Series search
    - First release data (as observed initially)
    
    Note: Only forward fills missing values. NO backfill - this prevents
    lookahead bias in backtesting by not using future information.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize FRED adapter.
        
        Args:
            api_key: FRED API key (get from https://fred.stlouisfed.org/)
            
        Raises:
            ImportError: If fredapi is not installed
        """
        if not FRED_AVAILABLE:
            raise ImportError(
                "fredapi is required. Install with: pip install fredapi"
            )
        self.api_key = api_key
        self._fred = Fred(api_key=api_key)
    
    def load_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load historical data from FRED.
        
        Args:
            symbol: FRED series identifier (e.g., 'GDP', 'UNRATE', 'SP500')
            start: Start datetime
            end: End datetime
            frequency: Optional frequency (not used, included for API compatibility)
            
        Returns:
            DataFrame with single column: ['value']
            Index: DatetimeIndex (timezone-aware UTC)
            
        Note:
            - Only forward fills missing values (NO backfill)
            - Uses future data is avoided to prevent lookahead bias
        """
        series_data = self._fred.get_series(
            symbol,
            observation_start=start.strftime('%Y-%m-%d'),
            observation_end=end.strftime('%Y-%m-%d')
        )
        
        if series_data is None or series_data.empty:
            raise ValueError(f"No data available for FRED series '{symbol}'")
        
        df = pd.DataFrame({'value': series_data})
        df.index = df.index.tz_localize('UTC')
        
        df = df[(df.index >= start) & (df.index <= end)]
        
        df['value'] = df['value'].ffill()
        
        df = df.dropna(how='all')
        
        return df.sort_index()
    
    def search_series(self, query: str, limit: int = 10) -> list:
        """
        Search FRED database for series.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of series dictionaries
        """
        results = self._fred.search(
            query,
            limit=limit,
            order_by="search_rank",
            sort_order="desc"
        )
        
        if results is None or results.empty:
            return []
        
        return list(results.T.to_dict().values())
    
    def get_first_release(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get first release data for a series.
        
        Gets the initial release of data (before revisions).
        This is useful for backtesting with realistic latency.
        
        Args:
            symbol: FRED series identifier
            start: Optional start datetime
            end: Optional end datetime
            
        Returns:
            DataFrame with single column: ['value']
        """
        series_data = self._fred.get_series_first_release(symbol)
        
        if series_data is None or series_data.empty:
            raise ValueError(f"No first release data for '{symbol}'")
        
        df = pd.DataFrame({'value': series_data})
        df.index = df.index.tz_localize('UTC')
        
        if start:
            df = df[df.index >= start]
        if end:
            df = df[df.index <= end]
        
        df['value'] = df['value'].ffill()
        df = df.dropna(how='all')
        
        return df.sort_index()
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if FRED series exists.
        
        Args:
            symbol: FRED series identifier
            
        Returns:
            True if series exists
        """
        try:
            self._fred.get_series(symbol, limit=1)
            return True
        except Exception:
            return False
    
    def supports_historical(self) -> bool:
        """FRED adapter supports historical data."""
        return True
    
    def get_series_info(self, symbol: str) -> dict:
        """
        Get metadata about a FRED series.
        
        Args:
            symbol: FRED series identifier
            
        Returns:
            Dictionary with series metadata
        """
        try:
            series = self._fred.get_series_info(symbol)
            return {
                'id': series.get('id'),
                'realtime_start': series.get('realtime_start'),
                'realtime_end': series.get('realtime_end'),
                'title': series.get('title'),
                'observation_start': series.get('observation_start'),
                'observation_end': series.get('observation_end'),
                'frequency': series.get('frequency'),
                'units': series.get('units'),
                'seasonal_adjustment': series.get('seasonal_adjustment'),
            }
        except Exception as e:
            raise ValueError(f"Failed to get info for '{symbol}': {e}")


class FREDDataAdapter(FREDAdapter):
    """Alias for FREDAdapter for backwards compatibility."""
    pass
