"""
Python execution layer for TradSL functions.

Functions in this module execute in Python using pyarrow/polars
instead of ClickHouse SQL.
"""
import uuid
from abc import ABC, abstractmethod

if False:
    import polars as pl

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tradsl.storage.connection import ClickHouseConnection


class ArrowFunction:
    """
    Base class for functions executing in Python via pyarrow/polars.
    
    These functions:
    1. Query data from ClickHouse as Arrow
    2. Process with polars
    3. Write results back to ClickHouse
    4. Return output table name
    
    Example:
        class MyIndicator(ArrowFunction):
            def __init__(self, column: str = "close", window: int = 20, **kwargs):
                super().__init__(columns=column, window=window, **kwargs)
                self.column = column
                self.window = window
            
            def transform(self, df):
                return df.with_columns(
                    pl.col(self.column).rolling_mean(self.window).alias(f"indicator_{self.window}")
                )
    """
    
    def __init__(self, columns: str | list[str], **kwargs):
        """
        Initialize the function.
        
        Args:
            columns: Single column name or list of column names to operate on
            **kwargs: Additional parameters (e.g., window=20, periods=5)
        """
        self.columns = [columns] if isinstance(columns, str) else columns
        self.params = kwargs
    
    @abstractmethod
    def transform(self, df: "pl.DataFrame") -> "pl.DataFrame":
        """
        Transform input DataFrame to output DataFrame.
        
        Args:
            df: Polars DataFrame with input data
            
        Returns:
            Polars DataFrame with transformed data
        """
        pass
    
    def apply(self, conn: "ClickHouseConnection", input_table: str) -> str:
        """
        Execute function in Python layer.
        
        Args:
            conn: ClickHouseConnection instance
            input_table: Name of input table in ClickHouse
            
        Returns:
            Name of output table in ClickHouse
        """
        import polars as pl
        
        cols = self.columns.copy()
        if "timestamp" not in cols:
            cols.insert(0, "timestamp")
        if "symbol" not in cols:
            cols.append("symbol")
        
        col_select = ', '.join(set(cols))
        
        query = f"SELECT {col_select} FROM {input_table} ORDER BY timestamp"
        
        df = conn.query_polars(query)
        
        result = self.transform(df)
        
        output_table = self._generate_output_table_name("arrow")
        
        conn.insert_polars(output_table, result, create=True)
        
        return output_table
    
    def _generate_output_table_name(self, prefix: str = "arrow") -> str:
        """Generate a unique output table name."""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
    
    @property
    def output_columns(self) -> list[tuple[str, str]]:
        """
        Return output column definitions.
        
        Override in subclass to define output schema.
        """
        return []


# Example ArrowFunction implementations

def SMA(column: str = "close", window: int = 20, **kwargs):
    """
    Simple Moving Average using polars instead of SQL.
    """
    import polars as pl
    
    class PythonSMA(ArrowFunction):
        def __init__(self, column=column, window=window, **kwargs):
            super().__init__(columns=column, window=window, **kwargs)
            self.column = column
            self.window = window
        
        def transform(self, df):
            return df.with_columns(
                pl.col(self.column).rolling_mean(self.window).alias(f"sma_{self.window}")
            )
        
        @property
        def output_columns(self):
            return [(f"sma_{self.window}", "Float64")]
    
    return PythonSMA(column=column, window=window, **kwargs)


def EMA(column: str = "close", window: int = 20, **kwargs):
    """
    Exponential Moving Average using polars.
    """
    import polars as pl
    
    class PythonEMA(ArrowFunction):
        def __init__(self, column=column, window=window, **kwargs):
            super().__init__(columns=column, window=window, **kwargs)
            self.column = column
            self.window = window
        
        def transform(self, df):
            alpha = 2.0 / (self.window + 1)
            return df.with_columns(
                pl.col(self.column).ewm_mean(alpha=alpha).alias(f"ema_{self.window}")
            )
        
        @property
        def output_columns(self):
            return [(f"ema_{self.window}", "Float64")]
    
    return PythonEMA(column=column, window=window, **kwargs)


def Returns(column: str = "close", periods: int = 1, **kwargs):
    """
    Returns (percentage change) using polars.
    """
    import polars as pl
    
    class PythonReturns(ArrowFunction):
        def __init__(self, column=column, periods=periods, **kwargs):
            super().__init__(columns=column, periods=periods, **kwargs)
            self.column = column
            self.periods = periods
        
        def transform(self, df):
            return df.with_columns(
                (pl.col(self.column).pct_change(self.periods) * 100).alias(f"returns_{self.periods}")
            )
        
        @property
        def output_columns(self):
            return [(f"returns_{self.periods}", "Float64")]
    
    return PythonReturns(column=column, periods=periods, **kwargs)