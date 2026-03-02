from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
import pandas as pd

from .base import BaseAdapter


DEFAULT_COLUMN_MAPPING = {
    'open': 'open',
    'Open': 'open',
    'OPEN': 'open',
    'high': 'high',
    'High': 'high',
    'HIGH': 'high',
    'low': 'low',
    'Low': 'low',
    'LOW': 'low',
    'close': 'close',
    'Close': 'close',
    'CLOSE': 'close',
    'adj close': 'adj_close',
    'Adj Close': 'adj_close',
    'adj_close': 'adj_close',
    'Adj_Close': 'adj_close',
    'volume': 'volume',
    'Volume': 'volume',
    'VOLUME': 'volume',
}


class CSVAdapter(BaseAdapter):
    """
    Adapter for CSV files with OHLCV data.
    
    Supports:
    - Flexible column naming (close, Close, CLOSE all map to 'close')
    - Custom column mappings
    - Multiple date formats
    - Custom timestamp columns
    """
    
    def __init__(
        self,
        data_dir: str,
        column_mapping: Optional[Dict[str, str]] = None,
        timestamp_col: str = 'timestamp',
        date_format: Optional[str] = None,
        required_columns: Optional[list] = None
    ):
        """
        Initialize CSV adapter.
        
        Args:
            data_dir: Directory containing CSV files (named like SYMBOL.csv)
            column_mapping: Custom column mapping {original_name: normalized_name}
                           If None, uses default flexible mapping
            timestamp_col: Name of the timestamp column in CSV
            date_format: Optional datetime format string (e.g., '%Y-%m-%d')
                         If None, attempts auto-detection
            required_columns: List of required column names after mapping
                              Defaults to ['open', 'high', 'low', 'close', 'volume']
        """
        self.data_dir = Path(data_dir)
        self.column_mapping = column_mapping or DEFAULT_COLUMN_MAPPING
        self.timestamp_col = timestamp_col
        self.date_format = date_format
        self.required_columns = required_columns or ['open', 'high', 'low', 'close', 'volume']
    
    def load_historical(
        self, 
        symbol: str, 
        start: datetime, 
        end: datetime,
        frequency: str = None
    ) -> pd.DataFrame:
        """
        Load historical data from CSV file.
        
        Args:
            symbol: Stock symbol (used for filename, e.g., 'AAPL.csv')
            start: Start datetime (inclusive filter)
            end: End datetime (inclusive filter)
            frequency: Ignored for CSV (included for API compatibility)
            
        Returns:
            DataFrame with columns: [open, high, low, close, volume]
            Index: DatetimeIndex (timezone-aware UTC)
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If required columns are missing
        """
        file_path = self.data_dir / f"{symbol}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        df = self._normalize_columns(df)
        
        df = self._parse_timestamp(df)
        
        if df.empty:
            raise ValueError(f"CSV file is empty: {file_path}")
        
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        
        df = df[(df.index >= start) & (df.index <= end)]
        
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns {missing} in {file_path}")
        
        return df[self.required_columns].sort_index()
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names to lowercase.
        
        Applies custom mapping first, then falls back to flexible defaults.
        """
        rename_map = {}
        
        for col in df.columns:
            if col in self.column_mapping:
                rename_map[col] = self.column_mapping[col]
            elif col.lower() in DEFAULT_COLUMN_MAPPING.values():
                rename_map[col] = col.lower()
            else:
                rename_map[col] = col.lower()
        
        df = df.rename(columns=rename_map)
        
        return df
    
    def _parse_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse timestamp column to datetime index."""
        if self.timestamp_col not in df.columns:
            if 'date' in df.columns:
                self.timestamp_col = 'date'
            elif 'datetime' in df.columns:
                self.timestamp_col = 'datetime'
            else:
                raise ValueError(
                    f"Timestamp column '{self.timestamp_col}' not found. "
                    f"Available columns: {list(df.columns)}"
                )
        
        ts_col = self.timestamp_col
        
        if self.date_format:
            df[ts_col] = pd.to_datetime(df[ts_col], format=self.date_format)
        else:
            df[ts_col] = pd.to_datetime(df[ts_col])
        
        df = df.set_index(ts_col)
        df = df.sort_index()
        
        return df
    
    def supports_historical(self) -> bool:
        """CSV adapter supports historical data."""
        return True
    
    def validate_symbol(self, symbol: str) -> bool:
        """Check if CSV file exists for symbol."""
        file_path = self.data_dir / f"{symbol}.csv"
        return file_path.exists()
    
    def list_available_symbols(self) -> list:
        """List all available symbols (CSV files without extension)."""
        if not self.data_dir.exists():
            return []
        return [f.stem for f in self.data_dir.glob('*.csv')]


class CSVDataAdapter(CSVAdapter):
    """Alias for CSVAdapter for backwards compatibility."""
    pass
