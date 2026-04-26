"""
ClickHouse connection wrapper for TradSL.

Provides a simple interface for executing queries and loading data.
"""
import io
import os
from io import BytesIO
from typing import Optional, Any

import pandas as pd

try:
    import pyarrow as pa
    import polars as pl
    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False
    pa = None
    pl = None


class ClickHouseConnection:
    """
    Simple wrapper around ClickHouse HTTP API.
    
    Uses the HTTP endpoint on port 8123 by default.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8123,
        database: str = "default",
        user: str = "default",
        password: str = "",
        timeout: int = 30,
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.timeout = timeout
    
    @property
    def _url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    def _params(self) -> dict:
        return {
            "database": self.database,
            "user": self.user,
            "password": self.password,
        }
    
    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute a SQL query and return results as DataFrame.
        
        Args:
            sql: SQL query to execute
            
        Returns:
            DataFrame with query results
        """
        import requests
        
        sql_with_format = sql + " FORMAT TabSeparatedWithNames"
        
        response = requests.post(
            self._url,
            params=self._params(),
            data=sql_with_format,
            timeout=self.timeout,
        )
        response.raise_for_status()
        
        if not response.text or response.text.strip() == "":
            return pd.DataFrame()
        
        lines = response.text.strip().split('\n')
        if len(lines) < 2:
            return pd.DataFrame()
        
        header = lines[0].split('\t')
        data = '\n'.join(lines[1:])
        
        return pd.read_csv(io.StringIO(data), sep='\t', header=None, names=header)
    
    def execute(self, sql: str, data: str | None = None) -> None:
        """
        Execute a SQL statement (INSERT, CREATE, etc.).
        
        Args:
            sql: SQL statement to execute
            data: Optional data for INSERT statements (e.g., TSV format)
        """
        import requests
        
        if data is not None:
            combined = sql + "\n" + data
            response = requests.post(
                self._url,
                params=self._params(),
                data=combined,
                timeout=self.timeout,
            )
        else:
            response = requests.post(
                self._url,
                params=self._params(),
                data=sql,
                timeout=self.timeout,
            )
        response.raise_for_status()
    
    def load_parquet(self, path: str, table_name: str) -> None:
        """
        Load a parquet file into a ClickHouse table.
        
        Args:
            path: Path to parquet file
            table_name: Target table name in ClickHouse
        """
        self.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} AS
            SELECT * FROM file('{path}', Parquet)
        """)
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in ClickHouse.
        
        Args:
            table_name: Table name to check
            
        Returns:
            True if table exists
        """
        result = self.query(f"EXISTS TABLE {table_name}")
        if result.empty:
            return False
        return result.iloc[0]['result'] == 1
    
    def drop_table(self, table_name: str) -> None:
        """
        Drop a table from ClickHouse.
        
        Args:
            table_name: Table to drop
        """
        self.execute(f"DROP TABLE IF EXISTS {table_name}")
    
    def query_arrow(self, query: str) -> "pa.Table":
        """
        Execute a SQL query and return results as PyArrow Table.
        
        Args:
            query: SQL query to execute
            
        Returns:
            PyArrow Table with query results
        """
        import requests
        
        query_with_format = query + " FORMAT ArrowStream"
        
        response = requests.post(
            self._url,
            params=self._params(),
            data=query_with_format,
            timeout=self.timeout,
        )
        response.raise_for_status()
        
        if not response.content:
            raise ValueError("Empty response from ClickHouse")
        
        return pa.ipc.open_stream(BytesIO(response.content)).read_all()
    
    def query_polars(self, query: str) -> "pl.DataFrame":
        """
        Execute a SQL query and return results as Polars DataFrame.
        
        Args:
            query: SQL query to execute
            
        Returns:
            Polars DataFrame with query results
        """
        arrow_table = self.query_arrow(query)
        return pl.from_arrow(arrow_table)
    
    def insert_arrow(self, table_name: str, table: "pa.Table", create: bool = True) -> None:
        """
        Insert a PyArrow Table into ClickHouse.
        
        Args:
            table_name: Target table name
            table: PyArrow Table to insert
            create: If True, CREATE TABLE first; if False, INSERT only
        """
        import requests
        import polars as pl
        
        if create:
            # Map pyarrow types to ClickHouse types
            type_mapping = {
                'string': 'String',
                'int64': 'Int64',
                'int32': 'Int32',
                'float64': 'Float64',
                'float32': 'Float32',
                'timestamp[ms]': 'DateTime64',
                'timestamp[us]': 'DateTime64',
                'bool': 'UInt8',
            }
            columns = []
            for field in table.schema:
                ch_type = type_mapping.get(str(field.type), 'String')
                columns.append(f"{field.name} {ch_type}")
            
            columns_sql = ', '.join(columns)
            self.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    {columns_sql}
                ) ENGINE = MergeTree()
                ORDER BY tuple()
            """)
        
        df = pl.from_arrow(table)
        
        buf = BytesIO()
        df.write_csv(buf, separator='\t', include_header=False)
        buf.seek(0)
        
        # Use INSERT with FORMAT
        insert_sql = f"INSERT INTO {table_name} FORMAT TabSeparated"
        response = requests.post(
            self._url,
            params=self._params(),
            data=insert_sql + "\n" + buf.read().decode('utf-8'),
            timeout=self.timeout,
        )
        response.raise_for_status()
    
    def insert_polars(self, table_name: str, df: "pl.DataFrame", create: bool = True) -> None:
        """
        Insert a Polars DataFrame into ClickHouse.
        
        Args:
            table_name: Target table name
            df: Polars DataFrame to insert
            create: If True, CREATE TABLE first; if False, INSERT only
        """
        arrow_table = df.to_arrow()
        self.insert_arrow(table_name, arrow_table, create=create)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
