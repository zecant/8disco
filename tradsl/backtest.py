from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
import pandas as pd

from tradsl.dag import DAG


class TickSpeed(Enum):
    """Tick speed enum for backtesting."""
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    ONE_HOUR = "1h"
    ONE_DAY = "1d"


class Backtester:
    def __init__(
        self,
        dag: DAG,
        start_date: datetime,
        end_date: datetime,
        tick_speed: TickSpeed = TickSpeed.ONE_MINUTE,
    ):
        self.dag = dag
        self.start_date = start_date
        self.end_date = end_date
        self.tick_speed = tick_speed
        self.current_date = start_date
        self.results: list[dict] = []
        self._initialized = False
    
    def run(self) -> pd.DataFrame:
        """Run backtest from start_date to end_date."""
        if not self._initialized:
            self._initialize()
        
        while self.current_date <= self.end_date:
            self.dag._current_timestamp = self.current_date
            self.dag.step()
            self._record_snapshot()
            self._advance_tick()
        
        return self.to_dataframe()
    
    def _initialize(self) -> None:
        """Initialize all adapters with start time."""
        for name, node in self.dag.nodes.items():
            if node.type == "timeseries":
                adapter = self.dag._adapter_registry.get(name)
                if adapter is not None and hasattr(adapter, "set_start"):
                    adapter.set_start(self.start_date)
        self._initialized = True
    
    def _advance_tick(self) -> None:
        """Increment current_date by tick_speed interval."""
        # Tick speed interval mapping in minutes
        tick_intervals = {
            TickSpeed.ONE_MINUTE: 1,
            TickSpeed.FIVE_MINUTES: 5,
            TickSpeed.FIFTEEN_MINUTES: 15,
            TickSpeed.ONE_HOUR: 60,
            TickSpeed.ONE_DAY: 1440,
        }
        interval_minutes = tick_intervals.get(self.tick_speed, 1)
        self.current_date += timedelta(minutes=interval_minutes)
    
    def _record_snapshot(self) -> None:
        """Record current state with timestamp."""
        snapshot = {"timestamp": self.current_date}
        for name, value in self.dag.values():
            snapshot[name] = value
        self.results.append(snapshot)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Return results as DataFrame."""
        return pd.DataFrame(self.results).set_index("timestamp")
