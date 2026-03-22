from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Union
import pandas as pd
import yfinance


class Adapter(ABC):
    @abstractmethod
    def set_start(self, start_time: datetime) -> None:
        """Set the start time for the adapter and initialize internal cursor."""
        pass
    
    @abstractmethod
    def tick(self) -> Optional[Union[list, pd.DataFrame]]:
        """Return values for current tick: list for single-row, DataFrame for multi-column."""
        pass


class YFinanceAdapter(Adapter):
    """Yahoo Finance data adapter that loads historical data and streams it tick by tick."""
    
    # Maximum days per request varies by interval
    # 1m/2m/5m/15m/30m: 8 days, 1h: 730 days, 1d: unlimited
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
    
    def __init__(self, symbol: str = "AAPL", interval: str = "1m"):
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
    
    def tick(self) -> Optional[list]:
        """Return next OHLCV row as list [open, high, low, close, volume]."""
        if self.data is None:
            return None
        if self.idx >= len(self.data):
            return None
        row = self.data.iloc[self.idx]
        self.idx += 1
        return [row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]]
