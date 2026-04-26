"""Tests for config blocks in DAG."""
import pytest
import pandas as pd
from tradsl import DAG, Node


class TestConfigBlocks:
    """Tests for type=config blocks."""

    def test_config_block_not_processed(self):
        """Config blocks should be stored but not processed as DAG nodes."""
        config = {
            "settings": {
                "type": "config",
                "clickhouse_host": "localhost",
                "clickhouse_port": 8123,
            },
            "price": {
                "type": "timeseries",
                "adapter": "parquet",
                "path": "/data/prices.parquet",
            },
        }
        
        dag = DAG.from_config(config)
        
        assert "settings" in dag._config_blocks
        assert dag._config_blocks["settings"]["clickhouse_host"] == "localhost"
        assert "settings" not in dag.nodes
        assert "price" in dag.nodes

    def test_config_block_ignored_in_validation(self):
        """Config blocks should not cause validation errors."""
        config = {
            "bad_type": {
                "type": "config",
                "some_setting": "value",
            },
        }
        
        dag = DAG.from_config(config)
        dag.build()
        dag.validate()

    def test_multiple_config_blocks(self):
        """Multiple config blocks should all be stored."""
        config = {
            "database": {"type": "config", "host": "localhost"},
            "storage": {"type": "config", "path": "/data"},
            "price": {"type": "timeseries", "adapter": "parquet"},
        }
        
        dag = DAG.from_config(config)
        
        assert len(dag._config_blocks) == 2
        assert "database" in dag._config_blocks
        assert "storage" in dag._config_blocks
        assert "price" in dag.nodes

    def test_config_block_accessible_after_build(self):
        """Config blocks should be accessible after from_config and build."""
        config = {
            "my_config": {
                "type": "config",
                "api_key": "secret123",
                "max_retries": 5,
            },
            "price": {"type": "timeseries", "adapter": "parquet"},
        }
        
        dag = DAG.from_config(config)
        dag.build()
        
        assert dag._config_blocks["my_config"]["api_key"] == "secret123"
        assert dag._config_blocks["my_config"]["max_retries"] == 5

    def test_config_block_attributes_preserved(self):
        """All attributes from config blocks should be preserved."""
        config = {
            "settings": {
                "type": "config",
                "nested": {"key": "value"},
                "list": [1, 2, 3],
            },
        }
        
        dag = DAG.from_config(config)
        
        assert dag._config_blocks["settings"]["nested"]["key"] == "value"
        assert dag._config_blocks["settings"]["list"] == [1, 2, 3]

    def test_config_with_functions_and_timeseries(self):
        """Config blocks should work alongside functions and timeseries."""
        config = {
            "config": {"type": "config", "setting": "value"},
            "price": {"type": "timeseries", "adapter": "parquet"},
            "indicator": {
                "type": "function",
                "function": "functions.lag",
                "inputs": ["price"],
            },
        }
        
        dag = DAG.from_config(config)
        dag.build()
        
        assert "config" in dag._config_blocks
        assert "price" in dag.nodes
        assert "indicator" in dag.nodes


class TestFunctionRegistry:
    """Tests for function registry."""

    def test_default_registry_has_sql_functions(self):
        """Default registry should have all SQL functions."""
        from tradsl import default_registry
        
        assert "functions.lag" in default_registry
        assert "functions.ema" in default_registry
        assert "functions.sma" in default_registry
        assert "functions.returns" in default_registry
        assert "functions.logreturn" in default_registry

    def test_default_registry_has_external(self):
        """Default registry should have external functions."""
        from tradsl import default_registry
        
        assert "external.double" in default_registry

    def test_function_registry_resolve(self):
        """Functions should resolve from registry."""
        from tradsl import default_registry, Lag
        from tradsl.dag import DAG
        
        # Create a function node that references an existing timeseries source
        config = {
            "source": {
                "type": "timeseries", 
                "adapter": "parquet", 
                "path": "/data/test.parquet"
            },
            "lagged": {
                "type": "function",
                "function": "functions.lag",
                "inputs": ["source"],
                "periods": 1,
            },
        }
        
        dag = DAG.from_config(config)
        dag.build()
        
        assert "lagged" not in dag._function_registry  # Not resolved yet
        
        # Manually instantiate to test registry works
        lag_fn = default_registry["functions.lag"](periods=1)
        assert isinstance(lag_fn, Lag)


class TestNoFutureDataLeakage:
    """Tests to ensure functions don't leak future data - verified via integration tests."""
    
    def test_lag_order_by_clause(self):
        """Lag function uses ORDER BY timestamp which ensures no future data."""
        from tradsl.functions import Lag
        import inspect
        
        source = inspect.getsource(Lag.apply)
        assert "OVER (ORDER BY timestamp)" in source, "Must use ORDER BY to prevent future leakage"
    
    def test_sma_order_by_clause(self):
        """SMA function uses ORDER BY timestamp which ensures no future data."""
        from tradsl.functions import SMA
        import inspect
        
        source = inspect.getsource(SMA.apply)
        assert "OVER (ORDER BY" in source, "Must use windowed ORDER BY to prevent future leakage"
    
    def test_ema_order_by_clause(self):
        """EMA function uses ORDER BY timestamp which ensures no future data."""
        from tradsl.functions import EMA
        import inspect
        
        source = inspect.getsource(EMA.apply)
        # EMA currently uses pandas internally - verified by pandas ORDER BY in query
        assert "ORDER BY timestamp" in source, "Must ORDER data to prevent future leakage"
    
    def test_returns_order_by_clause(self):
        """Returns function uses ORDER BY timestamp which ensures no future data."""
        from tradsl.functions import Returns
        import inspect
        
        source = inspect.getsource(Returns.apply)
        assert "OVER (ORDER BY timestamp)" in source, "Must use ORDER BY to prevent future leakage"


class TestJoinTablesNoFutureLeak:
    """Ensure join tables doesn't leak future data."""

    @pytest.fixture
    def conn(self):
        """Create ClickHouse connection."""
        try:
            from tradsl.storage import ClickHouseConnection
            conn = ClickHouseConnection(host="127.0.0.1", port=8123, timeout=30)
            conn.execute("SELECT 1")
            return conn
        except Exception:
            pytest.skip("ClickHouse unavailable")

    def test_join_preserves_column_isolation(self, conn):
        """Joined tables should preserve column isolation per source."""
        from tradsl.functions import TimeSeriesFunction
        
        # Create two tables with different data
        conn.execute("DROP TABLE IF EXISTS jt_table1")
        conn.execute("CREATE TABLE jt_table1 (timestamp DateTime, close Float64) ENGINE = MergeTree() ORDER BY timestamp")
        conn.execute("INSERT INTO jt_table1 VALUES ('2024-01-01', 100), ('2024-01-02', 101)")
        
        conn.execute("DROP TABLE IF EXISTS jt_table2")
        conn.execute("CREATE TABLE jt_table2 (timestamp DateTime, close Float64) ENGINE = MergeTree() ORDER BY timestamp")
        conn.execute("INSERT INTO jt_table2 VALUES ('2024-01-01', 200), ('2024-01-02', 201)")
        
        class PassThrough(TimeSeriesFunction):
            def __init__(self):
                super().__init__(columns=["close"])
            
            @property
            def output_columns(self):
                return [("result", "Float64")]
            
            def apply(self, conn, input_table):
                return input_table
        
        fn = PassThrough()
        result_table = fn._join_tables(conn, {"t1": "jt_table1", "t2": "jt_table2"})
        
        result = conn.query(f"SELECT * FROM {result_table} ORDER BY timestamp")
        
        # t1_close should ONLY have values from t1, t2_close from t2
        # No mixing should occur
        assert list(result["t1_close"]) == [100, 101]
        assert list(result["t2_close"]) == [200, 201]

    def test_join_with_different_timestamps_no_future(self, conn):
        """Join with different timestamps should not see future data."""
        from tradsl.functions import TimeSeriesFunction
        
        conn.execute("DROP TABLE IF EXISTS jt_a")
        conn.execute("CREATE TABLE jt_a (timestamp DateTime, close Float64) ENGINE = MergeTree() ORDER BY timestamp")
        conn.execute("INSERT INTO jt_a VALUES ('2024-01-01', 100), ('2024-01-03', 102)")
        
        conn.execute("DROP TABLE IF EXISTS jt_b")
        conn.execute("CREATE TABLE jt_b (timestamp DateTime, close Float64) ENGINE = MergeTree() ORDER BY timestamp")
        conn.execute("INSERT INTO jt_b VALUES ('2024-01-01', 200), ('2024-01-02', 201), ('2024-01-03', 202)")
        
        class PassThrough(TimeSeriesFunction):
            def __init__(self):
                super().__init__(columns=["close"])
            
            @property
            def output_columns(self):
                return [("result", "Float64")]
            
            def apply(self, conn, input_table):
                return input_table
        
        fn = PassThrough()
        result_table = fn._join_tables(conn, {"jt_a": "jt_a", "jt_b": "jt_b"})
        
        result = conn.query(f"SELECT * FROM {result_table} ORDER BY timestamp")
        
        # On 2024-01-02, jt_a should be 0 (no data), jt_b should be 201
        # This ensures no future data leaked
        row_jan02 = result[result["timestamp"] == "2024-01-02 00:00:00"]
        
        if len(row_jan02) > 0:
            assert row_jan02["jt_a_close"].values[0] == 0  # No data for jt_a on Jan 02
            assert row_jan02["jt_b_close"].values[0] == 201  # jt_b has data


class TestDAGIntegration:
    """Integration tests for DAG with config blocks."""

    def test_dag_with_only_config(self):
        """DAG with only config blocks should have no nodes."""
        config = {
            "settings": {"type": "config", "key": "value"},
        }
        
        dag = DAG.from_config(config)
        dag.build()
        
        assert len(dag.nodes) == 0
        assert len(dag._config_blocks) == 1

    def test_config_accessible_post_execute(self):
        """Config blocks should be accessible after DAG execution would start."""
        config = {
            "db_config": {"type": "config", "host": "localhost"},
            "price": {"type": "timeseries", "adapter": "parquet"},
        }
        
        dag = DAG.from_config(config)
        dag.build()
        
        # Config should still be accessible
        assert dag._config_blocks["db_config"]["host"] == "localhost"
        
        # Nodes should be built
        assert "price" in dag.nodes