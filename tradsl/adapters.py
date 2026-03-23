from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import yfinance


class Adapter(ABC):
    def __init__(self, dag=None, **kwargs):
        self.dag = dag
    
    @abstractmethod
    def set_start(self, start_time: datetime) -> None:
        """Set the start time for the adapter and initialize internal cursor."""
        pass
    
    @abstractmethod
    def tick(self) -> Optional[pd.DataFrame]:
        """Return DataFrame with timestamp index for current tick."""
        pass


class YFinanceAdapter(Adapter):
    """Yahoo Finance data adapter that loads historical data and streams it tick by tick."""
    
    MAX_DAYS_PER_INTERVAL = {
        "1m": 8,
        "2m": 8,
        "5m": 8,
        "15m": 8,
        "30m": 8,
        "60m": 730,
        "1h": 730,
        "1d": 3650,
    }
    
    def __init__(self, symbol: str = "AAPL", interval: str = "1m", dag=None):
        super().__init__(dag=dag)
        self.symbol = symbol
        self.interval = interval
        self.ticker = yfinance.Ticker(symbol)
        self.data = None
        self.idx = 0
        self._start_time = None
        self._end_time = None
    
    def set_start(self, start_time: datetime) -> None:
        """Load historical data starting from start_time."""
        self._start_time = start_time
        max_days = self.MAX_DAYS_PER_INTERVAL.get(self.interval, 8)
        self._end_time = start_time + timedelta(days=max_days)
        self.data = self.ticker.history(
            start=start_time,
            end=self._end_time,
            interval=self.interval
        )
        self.idx = 0
    
    def tick(self) -> Optional[pd.DataFrame]:
        """Return OHLCV row as DataFrame with timestamp index."""
        if self.data is None:
            return None
        if self.idx >= len(self.data):
            return None
        row = self.data.iloc[self.idx]
        self.idx += 1
        df = pd.DataFrame({
            "open": [row["Open"]],
            "high": [row["High"]],
            "low": [row["Low"]],
            "close": [row["Close"]],
            "volume": [row["Volume"]],
        })
        df.index = pd.DatetimeIndex([row.name])
        return df
