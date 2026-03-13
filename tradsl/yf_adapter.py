"""
YFinance Adapter for TradSL

Section 8: Data adapter using yfinance library.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from tradsl.adapters import BaseAdapter, AdapterError, SymbolNotFound, DateRangeTruncated, validate_frequency


class YFinanceAdapter(BaseAdapter):
    """
    Data adapter using Yahoo Finance.
    
    Loads historical OHLCV data via yfinance library.
    Properly handles date range requests without silent truncation.
    """
    
    FREQUENCY_MAP = {
        '1min': '1m',
        '5min': '5m',
        '15min': '15m',
        '30min': '30m',
        '1h': '1h',
        '4h': '4h',
        '1d': '1d',
        '1wk': '1wk',
        '1mo': '1mo'
    }
    
    VALID_FREQUENCIES = {'1min', '5min', '15min', '30min', '1h', '4h', '1d', '1wk', '1mo'}
    
    def __init__(
        self,
        interval: str = '1d',
        max_workers: int = 5,
        timeout: int = 30,
        retry_count: int = 3
    ):
        """
        Initialize YFinance adapter.
        
        Args:
            interval: Default data frequency
            max_workers: Max concurrent download threads
            timeout: Request timeout in seconds
            retry_count: Number of retries on failure
        """
        self.interval = interval
        self.max_workers = max_workers
        self.timeout = timeout
        self.retry_count = retry_count
        
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def load_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: str
    ) -> pd.DataFrame:
        """
        Load historical OHLCV data from Yahoo Finance.
        
        Args:
            symbol: Instrument symbol (e.g., 'SPY', 'AAPL', '^VIX')
            start: Start datetime
            end: End datetime
            frequency: Pandas offset alias
        
        Returns:
            DataFrame with columns: open, high, low, close, volume (float64)
            Index: DatetimeIndex (UTC), sorted ascending
        
        Raises:
            SymbolNotFound: Symbol not available
            DateRangeTruncated: Requested range partially unavailable
            AdapterError: Other failures
        """
        if not validate_frequency(frequency):
            raise AdapterError(f"Invalid frequency: {frequency}")
        
        yf_freq = self.FREQUENCY_MAP.get(frequency, frequency)
        
        try:
            import yfinance as yf
        except ImportError:
            raise AdapterError("yfinance not installed: pip install yfinance")
        
        ticker = yf.Ticker(symbol)
        
        df = self._download_with_retry(ticker, start, end, yf_freq)
        
        if df is None or df.empty:
            raise SymbolNotFound(
                f"Symbol '{symbol}' not found or no data available"
            )
        
        df = self._process_dataframe(df, symbol)
        
        self._validate_date_range(df, start, end, symbol)
        
        self.validate_dataframe(df)
        
        return df
    
    def _download_with_retry(
        self,
        ticker,
        start: datetime,
        end: datetime,
        frequency: str,
        attempt: int = 0
    ) -> Optional[pd.DataFrame]:
        """Download data with retry logic."""
        try:
            import yfinance as yf
            
            df = ticker.history(
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                interval=frequency,
                auto_adjust=False,
                repair=False
            )
            
            if df is not None and not df.empty:
                return df
            
            if attempt < self.retry_count:
                return self._download_with_retry(
                    ticker, start, end, frequency, attempt + 1
                )
            
            return None
            
        except Exception as e:
            if attempt < self.retry_count:
                import time
                time.sleep(0.5 * (attempt + 1))
                return self._download_with_retry(
                    ticker, start, end, frequency, attempt + 1
                )
            raise AdapterError(f"Failed to download {ticker.ticker}: {str(e)}")
    
    def _process_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """
        Process raw yfinance DataFrame to standard format.
        
        Args:
            df: Raw yfinance DataFrame
            symbol: Symbol for logging
        
        Returns:
            Processed DataFrame with standard columns
        """
        result = pd.DataFrame()
        
        if 'Open' in df.columns:
            result['open'] = df['Open'].astype(np.float64)
        elif 'open' in df.columns:
            result['open'] = df['open'].astype(np.float64)
        else:
            raise AdapterError(f"Missing Open column for {symbol}")
        
        if 'High' in df.columns:
            result['high'] = df['High'].astype(np.float64)
        elif 'high' in df.columns:
            result['high'] = df['high'].astype(np.float64)
        else:
            raise AdapterError(f"Missing High column for {symbol}")
        
        if 'Low' in df.columns:
            result['low'] = df['Low'].astype(np.float64)
        elif 'low' in df.columns:
            result['low'] = df['low'].astype(np.float64)
        else:
            raise AdapterError(f"Missing Low column for {symbol}")
        
        if 'Close' in df.columns:
            result['close'] = df['Close'].astype(np.float64)
        elif 'close' in df.columns:
            result['close'] = df['close'].astype(np.float64)
        else:
            raise AdapterError(f"Missing Close column for {symbol}")
        
        if 'Volume' in df.columns:
            result['volume'] = df['Volume'].astype(np.float64)
        elif 'volume' in df.columns:
            result['volume'] = df['volume'].astype(np.float64)
        else:
            result['volume'] = 0.0
        
        if isinstance(df.index, pd.DatetimeIndex):
            result.index = df.index.tz_convert('UTC')
        else:
            result.index = pd.to_datetime(df.index).tz_localize('UTC')
        
        result.index.name = 'timestamp'
        
        result = result[~result.index.duplicated(keep='first')]
        
        if not result.index.is_monotonic_increasing:
            result = result.sort_index()
            import logging
            logging.getLogger("tradsl.adapters").warning(
                f"{symbol}: Timestamps were not sorted, sorted automatically"
            )
        
        return result
    
    def _validate_date_range(
        self,
        df: pd.DataFrame,
        requested_start: datetime,
        requested_end: datetime,
        symbol: str
    ) -> None:
        """
        Validate that the full requested date range is available.
        
        Raises DateRangeTruncated if range is incomplete.
        """
        if df.empty:
            raise DateRangeTruncated(
                f"No data returned for {symbol} in range "
                f"[{requested_start.date()}, {requested_end.date()})"
            )
        
        actual_start = df.index.min()
        actual_end = df.index.max()
        
        if actual_start.tzinfo is not None:
            requested_start = requested_start.replace(tzinfo=actual_start.tzinfo)
            requested_end = requested_end.replace(tzinfo=actual_end.tzinfo)
        
        start_buffer = timedelta(days=3)
        end_buffer = timedelta(days=3)
        
        if actual_start > requested_start + start_buffer:
            raise DateRangeTruncated(
                f"Symbol {symbol}: requested start {requested_start.date()} "
                f"but data begins {actual_start.date()}"
            )
        
        if actual_end < requested_end - end_buffer:
            raise DateRangeTruncated(
                f"Symbol {symbol}: requested end {requested_end.date()} "
                f"but data ends {actual_end.date()}"
            )
    
    def supports_frequency(self, frequency: str) -> bool:
        """Check if adapter supports this frequency."""
        return frequency in self.VALID_FREQUENCIES
    
    def max_lookback(self, frequency: str) -> Optional[int]:
        """
        Maximum available bars at this frequency.
        
        Yahoo Finance typically provides:
        - 1min: ~7 days
        - 5min/15min/30min/1h: ~60 days
        - 1d: max
        - 1wk: max
        - 1mo: max
        """
        limits = {
            '1min': 7 * 24 * 60,
            '5min': 60 * 24 * 12,
            '15min': 60 * 24 * 4,
            '30min': 60 * 24 * 2,
            '1h': 60 * 24,
            '4h': None,
            '1d': None,
            '1wk': None,
            '1mo': None
        }
        return limits.get(frequency)
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate that symbol exists.
        
        Args:
            symbol: Symbol to validate
        
        Returns:
            True if symbol is valid
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info is not None
        except Exception:
            return False
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get additional symbol information from Yahoo Finance.
        
        Args:
            symbol: Symbol to query
        
        Returns:
            Dict with symbol metadata
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('shortName', info.get('longName', symbol)),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'UNKNOWN'),
                'market_cap': info.get('marketCap'),
                'sector': info.get('sector'),
                'industry': info.get('industry')
            }
        except Exception:
            return {'symbol': symbol, 'name': symbol}
    
    def get_multiple(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        frequency: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple symbols efficiently.
        
        Args:
            symbols: List of symbols to load
            start: Start datetime
            end: End datetime
            frequency: Data frequency
        
        Returns:
            Dict mapping symbol to DataFrame
        """
        results = {}
        
        for symbol in symbols:
            try:
                df = self.load_historical(symbol, start, end, frequency)
                results[symbol] = df
            except AdapterError as e:
                results[symbol] = None
        
        return results


def create_yf_adapter(
    interval: str = '1d',
    **kwargs
) -> YFinanceAdapter:
    """
    Factory function to create YFinance adapter.
    
    Args:
        interval: Default frequency
        **kwargs: Additional adapter parameters
    
    Returns:
        Configured YFinanceAdapter instance
    """
    return YFinanceAdapter(interval=interval, **kwargs)
