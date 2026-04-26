"""
Function abstractions for TradSL.

Functions operate on ClickHouse tables, not DataFrames.
Each function takes table names as input and returns a table name as output.
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
import uuid

if TYPE_CHECKING:
    from tradsl.storage.connection import ClickHouseConnection


class SignalType:
    """
    Standard signal column names for one-hot encoded order types.
    
    One-hot encoding means exactly one column is 1, all others are 0.
    Example (buy signal):
        buy_signal: 1
        sell_signal: 0
        hold_signal: 0
    """
    BUY = "buy_signal"
    SELL = "sell_signal"
    HOLD = "hold_signal"
    
    # Extended for more complex order types
    MARKET_ORDER = "market_order"
    LIMIT_ORDER = "limit_order"
    CANCEL_ORDER = "cancel_order"
    NO_ORDER_BUY = "no_order_buy"
    NO_ORDER_SELL = "no_order_sell"
    
    # Standard set for simple backtesting
    STANDARD = [BUY, SELL, HOLD]


class TimeSeriesFunction(ABC):
    """
    Abstract base class for timeseries functions.
    
    All functions push computation to ClickHouse - never pull data out.
    The implementation can be pure SQL, external Python scripts, or any other
    mechanism that runs inside ClickHouse.
    
    When a function has multiple inputs, the DAG automatically joins them
    using ASOF UNION ALL + GROUP BY before passing to apply().
    
    Example:
        class MyIndicator(TimeSeriesFunction):
            def __init__(self, column: str = "close", window: int = 20, **kwargs):
                super().__init__(columns=[column], window=window, **kwargs)
                self.column = column
                self.window = window
            
            @property
            def output_columns(self) -> list[tuple[str, str]]:
                return [(f"my_indicator_{self.window}", "Float64")]
            
            def apply(self, conn: "ClickHouseConnection", input_table: str) -> str:
                # Implementation here...
                pass
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
        self._output_table = ""
    
    @abstractmethod
    def apply(self, conn: "ClickHouseConnection", input_table: str) -> str:
        """
        Push computation to ClickHouse.
        
        Args:
            conn: ClickHouseConnection instance
            input_table: Name of the input table in ClickHouse (already joined if multiple inputs)
            
        Returns:
            Name of the output table in ClickHouse containing the results
        """
        pass
    
    def _join_tables(self, conn: "ClickHouseConnection", input_tables: dict[str, str], columns: list[str] | None = None) -> str:
        """
        Join multiple input tables using UNION ALL + GROUP BY.
        
        This is the ClickHouse-preferred way to horizontally concatenate tables.
        Uses NULL for missing values (join_use_nulls = 1).
        
        Args:
            conn: ClickHouseConnection instance
            input_tables: Dict mapping input names to table names
            columns: List of column names to include with input prefix (default: self.columns)
            
        Returns:
            Name of the joined table
        """
        if len(input_tables) == 1:
            return list(input_tables.values())[0]
        
        conn.execute("SET join_use_nulls = 1")
        
        cols_to_use = columns or self.columns
        all_prefixes = list(input_tables.keys())
        
        # Build column list ONCE in consistent order for ALL subqueries
        # Format: prefix_colname for each prefix, each column
        all_prefixes_clean = [
            p.replace(" ", "_").replace("-", "_").replace(".", "_") 
            for p in all_prefixes
        ]
        
        table_subqueries = []
        for input_name, table_name in input_tables.items():
            input_prefix_clean = input_name.replace(" ", "_").replace("-", "_").replace(".", "_")
            
            col_selects = ["timestamp"]
            # For each prefix in consistent order, either get actual value or 0
            for prefix_clean in all_prefixes_clean:
                for col in cols_to_use:
                    if prefix_clean == input_prefix_clean:
                        # This input's actual value - cast to Float64 for consistency
                        col_selects.append(f"toFloat64({col}) as {prefix_clean}_{col}")
                    else:
                        # Other inputs get 0
                        col_selects.append(f"toFloat64(0) as {prefix_clean}_{col}")
            
            table_subqueries.append(f"SELECT {', '.join(col_selects)} FROM {table_name}")
        
        union_sql = " UNION ALL ".join(table_subqueries)
        
        joined_name = self._generate_output_table_name("joined")
        
        any_selects = []
        for input_name in input_tables.keys():
            prefix = input_name.replace(" ", "_").replace("-", "_").replace(".", "_")
            for col in cols_to_use:
                any_selects.append(f"max({prefix}_{col}) as {prefix}_{col}")
        
        create_sql = f"CREATE TABLE {joined_name} AS SELECT timestamp, {', '.join(any_selects)} FROM ({union_sql}) AS t GROUP BY timestamp ORDER BY timestamp"
        
        conn.execute(f"DROP TABLE IF EXISTS {joined_name}")
        conn.execute(create_sql)
        
        return joined_name
    
    @property
    @abstractmethod
    def output_columns(self) -> list[tuple[str, str]]:
        """
        Return the output column specifications.
        
        Returns:
            List of (name, ClickHouse_type) tuples for output columns.
            Example: [("sma_20", "Float64"), ("close_lag1", "Nullable(Float64)")]
        """
        pass
    
    @property
    def output_table(self) -> str:
        """Return the output table name after apply() is called."""
        return self._output_table
    
    @output_table.setter
    def output_table(self, value: str):
        self._output_table = value
    
    def _generate_output_table_name(self, prefix: str = "fn") -> str:
        """Generate a unique output table name."""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
    
    def _create_and_insert(self, conn: "ClickHouseConnection", output_table: str, columns_sql: str, select_sql: str) -> str:
        """Create table and insert data using separate statements."""
        conn.execute(f"DROP TABLE IF EXISTS {output_table}")
        
        create_sql = f"""
            CREATE TABLE {output_table} (
                {columns_sql}
            ) ENGINE = MergeTree()
            ORDER BY timestamp
        """
        conn.execute(create_sql)
        
        conn.execute(f"INSERT INTO {output_table} {select_sql}")
        return output_table


class Lag(TimeSeriesFunction):
    """
    Lag function - shifts a column by n periods.
    
    Example:
        price_lag5:
            type=function
            function=functions.lag
            inputs=[price]
            periods=5
    """
    
    def __init__(self, periods: int = 1, column: str = "close", **kwargs):
        super().__init__(columns=column, periods=periods, **kwargs)
        self.periods = periods
        self.column = column
    
    @property
    def output_columns(self) -> list[tuple[str, str]]:
        return [(f"{self.column}_lag{self.periods}", "Nullable(Float64)")]
    
    def apply(self, conn: "ClickHouseConnection", input_table: str) -> str:
        output_table = self._generate_output_table_name("lag")
        
        lag_col_name = f"{self.column}_lag{self.periods}"
        columns = f"symbol String, timestamp DateTime64, {self.column} Float64, {lag_col_name} Nullable(Float64)"
        select = f"""
            SELECT symbol, timestamp, {self.column},
                   LAG({self.column}, {self.periods}) OVER (ORDER BY timestamp) as {lag_col_name}
            FROM {input_table}
        """
        
        return self._create_and_insert(conn, output_table, columns, select)


class EMA(TimeSeriesFunction):
    """
    Exponential Moving Average function.
    
    Uses ClickHouse's native exponentialMovingAverage as a window function.
    The smoothing factor alpha = 2 / (window + 1) for trading-style EMA.
    
    Example:
        ema_20:
            type:function
            function=functions.ema
            inputs:[price]
            window:20
            column:close
    """
    
    def __init__(self, window: int = 20, column: str = "close", **kwargs):
        super().__init__(columns=column, window=window, **kwargs)
        self.window = window
        self.column = column
    
    @property
    def output_columns(self) -> list[tuple[str, str]]:
        return [(f"ema_{self.window}", "Nullable(Float64)")]
    
    def apply(self, conn: "ClickHouseConnection", input_table: str) -> str:
        output_table = self._generate_output_table_name("ema")
        
        ema_col_name = f"ema_{self.window}"
        
        alpha = 2.0 / (self.window + 1)
        
        columns = f"symbol String, timestamp DateTime64, {self.column} Float64, {ema_col_name} Nullable(Float64)"
        select = f"""
            SELECT symbol, timestamp, {self.column},
                   exponentialMovingAverage({alpha})({self.column}, toUnixTimestamp(timestamp)) OVER (ORDER BY timestamp ASC) as {ema_col_name}
            FROM {input_table}
        """
        
        return self._create_and_insert(conn, output_table, columns, select)


class SMA(TimeSeriesFunction):
    """
    Simple Moving Average function.
    
    Uses ClickHouse's native window function.
    
    Example:
        sma_20:
            type=function
            function=functions.sma
            inputs=[price]
            window=20
            column=close
    """
    
    def __init__(self, window: int = 20, column: str = "close", **kwargs):
        super().__init__(columns=column, window=window, **kwargs)
        self.window = window
        self.column = column
    
    @property
    def output_columns(self) -> list[tuple[str, str]]:
        return [(f"sma_{self.window}", "Nullable(Float64)")]
    
    def apply(self, conn: "ClickHouseConnection", input_table: str) -> str:
        output_table = self._generate_output_table_name("sma")
        
        sma_col_name = f"sma_{self.window}"
        window_size = self.window - 1
        columns = f"symbol String, timestamp DateTime64, {self.column} Float64, {sma_col_name} Nullable(Float64)"
        select = f"""
            SELECT symbol, timestamp, {self.column},
                   avg({self.column}) OVER (ORDER BY timestamp ROWS BETWEEN {window_size} PRECEDING AND CURRENT ROW) as {sma_col_name}
            FROM {input_table}
        """
        
        return self._create_and_insert(conn, output_table, columns, select)


class Returns(TimeSeriesFunction):
    """
    Calculate returns (percentage change).
    
    Example:
        returns:
            type=function
            function=functions.returns
            inputs=[price]
            periods=1
            column=close
    """
    
    def __init__(self, periods: int = 1, column: str = "close", **kwargs):
        super().__init__(columns=column, periods=periods, **kwargs)
        self.periods = periods
        self.column = column
    
    @property
    def output_columns(self) -> list[tuple[str, str]]:
        return [(f"returns_{self.periods}", "Nullable(Float64)")]
    
    def apply(self, conn: "ClickHouseConnection", input_table: str) -> str:
        output_table = self._generate_output_table_name("returns")
        
        returns_col_name = f"returns_{self.periods}"
        columns = f"symbol String, timestamp DateTime64, {self.column} Float64, {returns_col_name} Nullable(Float64)"
        select = f"""
            SELECT symbol, timestamp, {self.column},
                   (({self.column} - LAG({self.column}, {self.periods}) OVER (ORDER BY timestamp)) / 
                   LAG({self.column}, {self.periods}) OVER (ORDER BY timestamp)) * 100 as {returns_col_name}
            FROM {input_table}
        """
        
        conn.execute(f"DROP TABLE IF EXISTS {output_table}")
        
        create_sql = f"""
            CREATE TABLE {output_table} (
                {columns}
            ) ENGINE = MergeTree()
            ORDER BY timestamp
        """
        conn.execute(create_sql)
        conn.execute(f"INSERT INTO {output_table} {select}")
        return output_table


class LogReturn(TimeSeriesFunction):
    """
    Calculate log returns (natural log of price ratio).
    
    Example:
        log_return:
            type=function
            function=functions.logreturn
            inputs=[price]
            periods=1
            column=close
    """
    
    def __init__(self, periods: int = 1, column: str = "close", **kwargs):
        super().__init__(columns=column, periods=periods, **kwargs)
        self.periods = periods
        self.column = column
    
    @property
    def output_columns(self) -> list[tuple[str, str]]:
        return [(f"log_return_{self.periods}", "Nullable(Float64)")]
    
    def apply(self, conn: "ClickHouseConnection", input_table: str) -> str:
        output_table = self._generate_output_table_name("logreturn")
        
        col_name = f"log_return_{self.periods}"
        columns = f"symbol String, timestamp DateTime64, {self.column} Float64, {col_name} Nullable(Float64)"
        select = f"""
            SELECT symbol, timestamp, {self.column},
                   LN({self.column} / LAG({self.column}, {self.periods}) OVER (ORDER BY timestamp)) as {col_name}
            FROM {input_table}
        """
        
        conn.execute(f"DROP TABLE IF EXISTS {output_table}")
        
        create_sql = f"""
            CREATE TABLE {output_table} (
                {columns}
            ) ENGINE = MergeTree()
            ORDER BY timestamp
        """
        conn.execute(create_sql)
        conn.execute(f"INSERT INTO {output_table} {select}")
        return output_table


class Mean(TimeSeriesFunction):
    """
    Calculate simple moving average (mean) over a window.
    
    Example:
        mean:
            type=function
            function=functions.mean
            inputs=[price]
            window=20
            column=close
    """
    
    def __init__(self, window: int = 20, column: str = "close", **kwargs):
        super().__init__(columns=column, window=window, **kwargs)
        self.window = window
        self.column = column
    
    @property
    def output_columns(self) -> list[tuple[str, str]]:
        return [(f"mean_{self.window}", "Nullable(Float64)")]
    
    def apply(self, conn: "ClickHouseConnection", input_table: str) -> str:
        output_table = self._generate_output_table_name("mean")
        
        col_name = f"mean_{self.window}"
        columns = f"symbol String, timestamp DateTime64, {self.column} Float64, {col_name} Nullable(Float64)"
        select = f"""
            SELECT symbol, timestamp, {self.column},
                   AVG({self.column}) OVER (ORDER BY timestamp ROWS BETWEEN {self.window-1} PRECEDING AND CURRENT ROW) as {col_name}
            FROM {input_table}
        """
        
        conn.execute(f"DROP TABLE IF EXISTS {output_table}")
        
        create_sql = f"""
            CREATE TABLE {output_table} (
                {columns}
            ) ENGINE = MergeTree()
            ORDER BY timestamp
        """
        conn.execute(create_sql)
        conn.execute(f"INSERT INTO {output_table} {select}")
        return output_table


class Add(TimeSeriesFunction):
    """
    Add two columns or a column and a scalar.
    
    Example:
        price_plus_fee:
            type:function
            function:functions.add
            inputs:[price, fee]
            left:price
            right:fee
        # Or with scalar:
        price_plus_ten:
            type:function
            function:functions.add
            inputs:[price]
            left:price
            right:10
    """
    
    def __init__(self, left: str = "col1", right: str = "col2", **kwargs):
        super().__init__(columns=[left, right], left=left, right=right, **kwargs)
        self.left = left
        self.right = right
    
    @property
    def output_columns(self) -> list[tuple[str, str]]:
        return [(f"{self.left}_plus_{self.right}", "Float64")]
    
    def apply(self, conn: "ClickHouseConnection", input_table: str) -> str:
        output_table = self._generate_output_table_name("add")
        
        try:
            scalar = float(self.right)
            op = f"{self.left} + {scalar}"
            columns = f"symbol String, timestamp DateTime64, {self.left} Float64, {self.left}_plus_{self.right} Float64"
            select = f"""
                SELECT symbol, timestamp, {self.left}, {op}
                FROM {input_table}
            """
        except ValueError:
            op = f"{self.left} + {self.right}"
            columns = f"symbol String, timestamp DateTime64, {self.left} Float64, {self.right} Float64, {self.left}_plus_{self.right} Float64"
            select = f"""
                SELECT symbol, timestamp, {self.left}, {self.right}, {op}
                FROM {input_table}
            """
        
        return self._create_and_insert(conn, output_table, columns, select)


class Subtract(TimeSeriesFunction):
    """
    Subtract two columns or a column and a scalar.
    
    Example:
        price_minus_fee:
            type:function
            function:functions.subtract
            inputs:[price, fee]
            left:price
            right:fee
    """
    
    def __init__(self, left: str = "col1", right: str = "col2", **kwargs):
        super().__init__(columns=[left, right], left=left, right=right, **kwargs)
        self.left = left
        self.right = right
    
    @property
    def output_columns(self) -> list[tuple[str, str]]:
        return [(f"{self.left}_minus_{self.right}", "Float64")]
    
    def apply(self, conn: "ClickHouseConnection", input_table: str) -> str:
        output_table = self._generate_output_table_name("subtract")
        
        try:
            scalar = float(self.right)
            op = f"{self.left} - {scalar}"
            columns = f"symbol String, timestamp DateTime64, {self.left} Float64, {self.left}_minus_{self.right} Float64"
            select = f"""
                SELECT symbol, timestamp, {self.left}, {op}
                FROM {input_table}
            """
        except ValueError:
            op = f"{self.left} - {self.right}"
            columns = f"symbol String, timestamp DateTime64, {self.left} Float64, {self.right} Float64, {self.left}_minus_{self.right} Float64"
            select = f"""
                SELECT symbol, timestamp, {self.left}, {self.right}, {op}
                FROM {input_table}
            """
        
        return self._create_and_insert(conn, output_table, columns, select)


class Multiply(TimeSeriesFunction):
    """
    Multiply two columns or a column and a scalar.
    
    Example:
        price_times_quantity:
            type:function
            function:functions.multiply
            inputs:[price, quantity]
            left:price
            right:quantity
    """
    
    def __init__(self, left: str = "col1", right: str = "col2", **kwargs):
        super().__init__(columns=[left, right], left=left, right=right, **kwargs)
        self.left = left
        self.right = right
    
    @property
    def output_columns(self) -> list[tuple[str, str]]:
        return [(f"{self.left}_times_{self.right}", "Float64")]
    
    def apply(self, conn: "ClickHouseConnection", input_table: str) -> str:
        output_table = self._generate_output_table_name("multiply")
        
        try:
            scalar = float(self.right)
            op = f"{self.left} * {scalar}"
            columns = f"symbol String, timestamp DateTime64, {self.left} Float64, {self.left}_times_{self.right} Float64"
            select = f"""
                SELECT symbol, timestamp, {self.left}, {op}
                FROM {input_table}
            """
        except ValueError:
            op = f"{self.left} * {self.right}"
            columns = f"symbol String, timestamp DateTime64, {self.left} Float64, {self.right} Float64, {self.left}_times_{self.right} Float64"
            select = f"""
                SELECT symbol, timestamp, {self.left}, {self.right}, {op}
                FROM {input_table}
            """
        
        return self._create_and_insert(conn, output_table, columns, select)


class Divide(TimeSeriesFunction):
    """
    Divide two columns or a column by a scalar.
    
    Example:
        price_divided_by_shares:
            type:function
            function:functions.divide
            inputs:[price, shares]
            left:price
            right:shares
    """
    
    def __init__(self, left: str = "col1", right: str = "col2", **kwargs):
        super().__init__(columns=[left, right], left=left, right=right, **kwargs)
        self.left = left
        self.right = right
    
    @property
    def output_columns(self) -> list[tuple[str, str]]:
        return [(f"{self.left}_divided_by_{self.right}", "Nullable(Float64)")]
    
    def apply(self, conn: "ClickHouseConnection", input_table: str) -> str:
        output_table = self._generate_output_table_name("divide")
        
        try:
            scalar = float(self.right)
            op = f"{self.left} / {scalar}"
            columns = f"symbol String, timestamp DateTime64, {self.left} Float64, {self.left}_divided_by_{self.right} Nullable(Float64)"
            select = f"""
                SELECT symbol, timestamp, {self.left}, {op}
                FROM {input_table}
            """
        except ValueError:
            op = f"{self.left} / {self.right}"
            columns = f"symbol String, timestamp DateTime64, {self.left} Float64, {self.right} Float64, {self.left}_divided_by_{self.right} Nullable(Float64)"
            select = f"""
                SELECT symbol, timestamp, {self.left}, {self.right}, {op}
                FROM {input_table}
            """
        
        return self._create_and_insert(conn, output_table, columns, select)
