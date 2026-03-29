"""
ClickHouse connection wrapper for TradSL.

Provides a simple interface for executing queries and loading data.
"""
import io
import pandas as pd
from typing import Optional, Any


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
    
    def get_users_scripts_path(self) -> str:
        """
        Get the users_scripts path from ClickHouse config.
        
        Returns:
            Path to users_scripts directory
        """
        result = self.query("SELECT * FROM system.errors WHERE name = ''")
        try:
            result = self.query("SELECT * FROM system.build_options")
            for _, row in result.iterrows():
                if row['name'] == 'CLICKHOUSE_BUILD_GCC_SUFFIX':
                    break
        except:
            pass
        return '/var/lib/clickhouse/user_scripts'
    
    def upload_script(self, script_name: str, script_content: str) -> str:
        """
        Upload a Python script to ClickHouse's users_scripts directory.
        
        Note: This requires the users_scripts directory to be mounted/accessible.
        In docker, mount: -v /path/to/scripts:/var/lib/clickhouse/user_scripts
        
        Args:
            script_name: Name of the script file (e.g., 'my_func.py')
            script_content: Python script content
            
        Returns:
            Path where script was uploaded
        """
        import os
        
        scripts_path = os.environ.get('CLICKHOUSE_USER_SCRIPTS', '/var/lib/clickhouse/user_scripts')
        script_path = os.path.join(scripts_path, script_name)
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        
        return script_path
    
    def create_executable_table(
        self,
        table_name: str,
        script_name: str,
        output_columns: list[tuple[str, str]],
        input_query: str,
    ) -> str:
        """
        Create an Executable table that runs a Python script.
        
        Args:
            table_name: Name for the output table
            script_name: Name of script in users_scripts folder
            output_columns: List of (name, type) tuples for output
            input_query: SQL query whose results are passed to the script
            
        Returns:
            Table name
        """
        columns_sql = ', '.join(f"{name} {dtype}" for name, dtype in output_columns)
        
        sql = f"""
            CREATE TABLE {table_name} (
                {columns_sql}
            ) ENGINE = Executable(
                '{script_name}',
                'TabSeparated',
                ({input_query})
            )
        """
        self.execute(sql)
        return table_name
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
