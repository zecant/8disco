from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Any
import pandas as pd


class BaseAdapter(ABC):
    """
    Abstract base class for data adapters.
    
    Adapters serve as the bridge between external data sources and 
    tradsl's internal data structures. Each adapter handles:
    
    1. Historical data loading for backtesting
    2. Additional data fetching (fundamentals, options, etc.)
    3. Data format normalization
    
    The adapter pattern allows the system to work with any data source
    (APIs, files, databases, live feeds) without changing core logic.
    """
    
    @abstractmethod
    def load_historical(
        self, 
        symbol: str, 
        start: datetime, 
        end: datetime,
        frequency: str = '1d'
    ) -> pd.DataFrame:
        """
        Load historical OHLCV data for backtesting.
        
        Args:
            symbol: Instrument identifier (e.g., "AAPL", "BTCUSD")
            start: Start datetime for historical data
            end: End datetime for historical data
            frequency: Data frequency (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)
            
        Returns:
            DataFrame with columns: [open, high, low, close, volume]
            Index: DatetimeIndex representing bar timestamps
            
        Note:
            - Must return data in chronological order
            - Must handle missing data appropriately
            - Timestamps should be timezone-aware (UTC preferred)
        """
        pass
    
    def supports_historical(self) -> bool:
        """Indicate if adapter supports historical data loading."""
        return True
    
    def get_fundamentals(
        self, 
        ticker: str, 
        statement_type: str, 
        period: str = 'annual'
    ) -> dict:
        """
        Get fundamental data (income statement, balance sheet, cashflow).
        
        Args:
            ticker: Stock symbol
            statement_type: 'income_statement', 'balance_sflow'
            period: 'annual' or 'quarterlyheet', 'cash'
            
        Returns:
            Dictionary with fundamental data
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support fundamentals")
    
    def get_option_chain(self, ticker: str, expiry: str) -> dict:
        """
        Get option chain for a ticker and expiry.
        
        Args:
            ticker: Stock symbol
            expiry: Expiry date string (YYYY-MM-DD)
            
        Returns:
            Dictionary with 'calls' and 'puts' keys
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support options")
    
    def get_price_targets(self, ticker: str) -> dict:
        """
        Get analyst price targets and recommendations.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary with target price, recommendation, history
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support price targets")
    
    def get_key_metrics_timeseries(
        self, 
        ticker: str, 
        period: str = 'quarterly'
    ) -> dict:
        """
        Get common financial metrics over time.
        
        Args:
            ticker: Stock symbol
            period: 'quarterly' or 'annual'
            
        Returns:
            Dictionary with metric timeseries
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support key metrics")
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if symbol is available from this data source.
        
        Args:
            symbol: Instrument identifier
            
        Returns:
            True if symbol is valid and available
        """
        return True
