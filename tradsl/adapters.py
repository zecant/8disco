"""
Adapters for TradSL.

Adapters load data from external sources into ClickHouse tables.
Each adapter implements the load() method which returns the table name
that can be queried in ClickHouse.
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import uuid
import pandas as pd
import io

if TYPE_CHECKING:
    from tradsl.storage.connection import ClickHouseConnection


class Adapter(ABC):
    """
    Abstract base class for data adapters.
    
    Adapters are responsible for loading data from external sources
    into ClickHouse tables. Each adapter must implement load() which
    returns the table name in ClickHouse.
    """
    
    def __init__(self, dag=None, **kwargs):
        self.dag = dag
        self._table_name: str = ""
    
    @abstractmethod
    def load(self, conn: "ClickHouseConnection") -> str:
        """
        Load data into ClickHouse and return the table name.
        
        Args:
            conn: ClickHouseConnection instance
            
        Returns:
            Table name in ClickHouse that contains the loaded data
        """
        pass
    
    @property
    def table_name(self) -> str:
        """Return the table name after loading."""
        return self._table_name
    
    def _convert_dtype_to_ch(self, dtype) -> str:
        """Convert pandas dtype to ClickHouse type."""
        if dtype == 'object':
            return 'String'
        elif dtype == 'int64':
            return 'Int64'
        elif dtype == 'int32':
            return 'Int32'
        elif dtype == 'float64':
            return 'Float64'
        elif dtype == 'float32':
            return 'Float32'
        elif 'datetime' in str(dtype):
            return 'DateTime64'
        elif 'date' in str(dtype):
            return 'Date'
        else:
            return 'String'


class ParquetAdapter(Adapter):
    """
    Adapter that loads data from parquet files into ClickHouse.
    
    For dockerized ClickHouse, this adapter:
    1. Reads the parquet file with pandas
    2. Inserts the data into ClickHouse via SQL
    
    Example DSL usage:
        price:
            type=timeseries
            adapter=parquet
            path=/data/prices/aapl.parquet
            symbol=AAPL
    """
    
    def __init__(
        self,
        path: str,
        symbol: str | None = None,
        table_name: str | None = None,
        dag=None,
    ):
        super().__init__(dag=dag)
        self.path = path
        self.symbol = symbol
        self._table_name = table_name or f"node_{symbol or uuid.uuid4().hex[:8]}"
    
    def load(self, conn: "ClickHouseConnection") -> str:
        """
        Load parquet file into ClickHouse table.
        
        For docker environments where ClickHouse can't access local files,
        we read with pandas and insert via SQL.
        
        Args:
            conn: ClickHouseConnection instance
            
        Returns:
            Table name in ClickHouse
        """
        df = pd.read_parquet(self.path)
        
        self._create_table_from_df(conn, df)
        
        return self._table_name
    
    def _create_table_from_df(self, conn, df: "pd.DataFrame") -> None:
        """Create table and insert data from DataFrame."""
        columns = []
        for col, dtype in df.dtypes.items():
            ch_type = self._convert_dtype_to_ch(dtype)
            columns.append(f"{col} {ch_type}")
        
        conn.execute(f"DROP TABLE IF EXISTS {self._table_name}")
        
        create_sql = f"CREATE TABLE {self._table_name} ({', '.join(columns)}) ENGINE = MergeTree() ORDER BY timestamp"
        
        conn.execute(create_sql)
        
        output = io.StringIO()
        df.to_csv(output, sep='\t', header=False, index=False, na_rep='\\N')
        output.seek(0)
        
        conn.execute(f"INSERT INTO {self._table_name} FORMAT TSV", data=output.getvalue())


class CSVAdapter(Adapter):
    """
    Adapter that loads data from CSV files into ClickHouse.
    
    Example DSL usage:
        price:
            type=timeseries
            adapter=csv
            path=/data/prices/aapl.csv
            symbol=AAPL
    """
    
    def __init__(
        self,
        path: str,
        symbol: str | None = None,
        table_name: str | None = None,
        header: bool = True,
        delimiter: str = ",",
        dag=None,
    ):
        super().__init__(dag=dag)
        self.path = path
        self.symbol = symbol
        self._table_name = table_name or f"node_{symbol or uuid.uuid4().hex[:8]}"
        self.header = header
        self.delimiter = delimiter
    
    def load(self, conn: "ClickHouseConnection") -> str:
        """
        Load CSV file into ClickHouse table.
        
        Args:
            conn: ClickHouseConnection instance
            
        Returns:
            Table name in ClickHouse
        """
        df = pd.read_csv(self.path, sep=self.delimiter)
        
        self._create_table_from_df(conn, df)
        
        return self._table_name
    
    def _create_table_from_df(self, conn, df: "pd.DataFrame") -> None:
        """Create table and insert data from DataFrame."""
        columns = []
        for col, dtype in df.dtypes.items():
            ch_type = self._convert_dtype_to_ch(dtype)
            columns.append(f"{col} {ch_type}")
        
        conn.execute(f"DROP TABLE IF EXISTS {self._table_name}")
        
        create_sql = f"CREATE TABLE {self._table_name} ({', '.join(columns)}) ENGINE = MergeTree() ORDER BY timestamp"
        
        conn.execute(create_sql)
        
        output = io.StringIO()
        df.to_csv(output, sep='\t', header=False, index=False, na_rep='\\N')
        output.seek(0)
        
        conn.execute(f"INSERT INTO {self._table_name} FORMAT TSV", data=output.getvalue())
