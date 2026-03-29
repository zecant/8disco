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


class Function(ABC):
    """
    Abstract base class for all transformation functions.
    
    Functions operate on ClickHouse tables. They receive table names as input,
    perform SQL operations, and return a new table name.
    """
    
    def __init__(self, **kwargs):
        self.params = kwargs
        self._output_table = ""
    
    @abstractmethod
    def apply(self, conn: "ClickHouseConnection", input_tables: dict[str, str]) -> str:
        """
        Apply function to input tables in ClickHouse.
        
        Args:
            conn: ClickHouseConnection instance
            input_tables: Dict mapping node names to their table names in ClickHouse
            
        Returns:
            Table name in ClickHouse containing the output
        """
        pass
    
    @property
    def output_table(self) -> str:
        """Return the output table name."""
        return self._output_table
    
    @output_table.setter
    def output_table(self, value: str):
        self._output_table = value
    
    def _generate_output_table_name(self, prefix: str = "fn") -> str:
        """Generate a unique output table name."""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
    
    def _create_and_insert(self, conn, output_table: str, columns_sql: str, select_sql: str) -> str:
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
    
    def _create_and_insert(self, conn, output_table: str, columns_sql: str, select_sql: str) -> str:
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


class EMA(TimeSeriesFunction):
    """
    Exponential Moving Average function.
    
    Note: ClickHouse's exponentialMovingAverage is time-based (different from 
    trading EMA). This implementation uses pandas for compatibility with 
    standard trading EMA calculations.
    
    Example:
        ema_20:
            type=function
            function=functions.ema
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
        return [(f"ema_{self.window}", "Nullable(Float64)")]
    
    def apply(self, conn: "ClickHouseConnection", input_table: str) -> str:
        output_table = self._generate_output_table_name("ema")
        
        df = conn.query(f"SELECT symbol, timestamp, {self.column} FROM {input_table} ORDER BY timestamp")
        
        import pandas as pd
        df[f"ema_{self.window}"] = df[self.column].ewm(span=self.window).mean()
        
        columns = f"symbol String, timestamp DateTime64, {self.column} Float64, ema_{self.window} Nullable(Float64)"
        
        conn.execute(f"DROP TABLE IF EXISTS {output_table}")
        
        create_sql = f"""
            CREATE TABLE {output_table} (
                {columns}
            ) ENGINE = MergeTree()
            ORDER BY timestamp
        """
        conn.execute(create_sql)
        
        import io
        output = io.StringIO()
        df.to_csv(output, sep='\t', header=False, index=False, na_rep='\\N')
        output.seek(0)
        
        conn.execute(f"INSERT INTO {output_table} FORMAT TSV", data=output.getvalue())
        
        self._output_table = output_table
        return output_table


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
    
    def _create_and_insert(self, conn, output_table: str, columns_sql: str, select_sql: str) -> str:
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
        
        return self._create_and_insert(conn, output_table, columns, select)
    
    def _create_and_insert(self, conn, output_table: str, columns_sql: str, select_sql: str) -> str:
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


class ExternalFunction(TimeSeriesFunction):
    """
    Run arbitrary Python scripts in ClickHouse using the Executable table engine.
    
    The script receives data from ClickHouse via stdin and outputs results to stdout.
    Data is passed in TabSeparated format.
    
    Example:
        # Define a simple script that doubles the input
        script = '''
        #!/usr/bin/python3
        import sys
        for line in sys.stdin:
            value = float(line.strip())
            print(value * 2)
        '''
        
        # Create the function
        fn = ExternalFunction(
            script=script,
            script_name='double.py',
            columns=['close'],
            output_columns=[('close_doubled', 'Float64')]
        )
        
        # Use it
        output_table = fn.apply(conn, input_table)
    
    Requirements:
        - The users_scripts directory must be mounted/accessible
        - In docker: -v /path/to/scripts:/var/lib/clickhouse/user_scripts
        - Or set CLICKHOUSE_USER_SCRIPTS environment variable
    """
    
    def __init__(
        self,
        script: str | None = None,
        script_name: str | None = None,
        script_path: str | None = None,
        columns: str | list[str] = "close",
        output_columns: list[tuple[str, str]] | None = None,
        **kwargs
    ):
        super().__init__(columns=columns, **kwargs)
        
        if script is not None:
            self.script_name = script_name or f"tradsl_{uuid.uuid4().hex[:8]}.py"
        else:
            self.script_name = script_name
        
        self.script = script
        self.script_path = script_path
        self._output_columns = output_columns or [("result", "Float64")]
    
    @property
    def output_columns(self) -> list[tuple[str, str]]:
        return self._output_columns
    
    def apply(self, conn: "ClickHouseConnection", input_table: str) -> str:
        output_table = self._generate_output_table_name("ext")
        
        script_name = self.script_name
        if self.script is not None:
            script_name = script_name or f"tradsl_{uuid.uuid4().hex[:8]}.py"
            conn.upload_script(script_name, self.script)
        elif self.script_path is not None:
            script_name = self.script_path.split('/')[-1]
        
        if script_name is None:
            raise ValueError("Either script, script_name, or script_path must be provided")
        
        input_cols = ', '.join(self.columns)
        input_query = f"SELECT {input_cols}, timestamp, symbol FROM {input_table}"
        
        conn.create_executable_table(
            table_name=output_table,
            script_name=script_name,
            output_columns=self.output_columns,
            input_query=input_query
        )
        
        return output_table
