"""
Comprehensive tests for TimeSeriesFunction._join_tables()

Tests cover:
- Multiple input tables (2, 3, 4+)
- Multiple columns per table
- Different column types (Float64, Int64, DateTime)
- Special characters in table/input names (spaces, dots, hyphens)
- Single column vs multiple columns
- Proper value preservation
"""
import pytest
from datetime import datetime, timedelta

from tradsl.functions import TimeSeriesFunction
from tradsl.storage import ClickHouseConnection


class TestJoinTables:
    """Tests for _join_tables method."""
    
    @pytest.fixture
    def conn(self):
        """Create ClickHouse connection. Skip if not available."""
        try:
            conn = ClickHouseConnection(host="127.0.0.1", port=8123, timeout=30)
            conn.execute("SELECT 1")
            return conn
        except Exception as e:
            pytest.skip(f"ClickHouse not available: {e}")
    
    @pytest.fixture
    def pass_through(self):
        """Create a pass-through function for testing."""
        class PassThrough(TimeSeriesFunction):
            def __init__(self, columns=None):
                super().__init__(columns=columns or ["close"])
            
            @property
            def output_columns(self):
                return [("result", "Float64")]
            
            def apply(self, conn, input_table):
                return input_table
        
        return PassThrough()
    
    def create_test_table(self, conn, name, columns_data):
        """Helper to create a test table with data.
        
        Args:
            conn: ClickHouse connection
            name: Table name
            columns_data: Dict mapping column name to list of values
        """
        conn.execute(f"DROP TABLE IF EXISTS {name}")
        
        # Infer types from data
        col_defs = ["timestamp DateTime"]
        data = {}
        for col_name, values in columns_data.items():
            data[col_name] = list(values)  # Ensure it's a list
            if values and all(isinstance(v, int) for v in values):
                ch_type = "Float64"  # Use Float64 for compatibility with _join_tables
            elif values and all(isinstance(v, float) for v in values):
                ch_type = "Float64"
            else:
                ch_type = "String"
            col_defs.append(f"{col_name} {ch_type}")
        
        num_rows = len(list(data.values())[0]) if data else 5
        
        conn.execute(f"CREATE TABLE {name} ({', '.join(col_defs)}) ENGINE = MergeTree() ORDER BY timestamp")
        
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(num_rows)]
        
        rows = []
        for i, ts in enumerate(timestamps):
            row_vals = [f"'{ts}'"]
            for col_name in data:
                val = data[col_name][i] if i < len(data[col_name]) else 0
                row_vals.append(str(val))
            rows.append(f"({', '.join(row_vals)})")
        
        conn.execute(f"INSERT INTO {name} (timestamp, {', '.join(data.keys())}) VALUES {', '.join(rows)}")
        
        return data
    
    def get_all_values(self, conn, table, prefix, cols):
        """Get all values for a prefix/column combination."""
        result = conn.query(f"SELECT * FROM {table} ORDER BY timestamp")
        values = {}
        for col in cols:
            col_name = f"{prefix}_{col}"
            if col_name in result.columns:
                values[col] = list(result[col_name].dropna().values)
            else:
                values[col] = []
        return values
    
    def test_two_tables_single_column(self, conn, pass_through):
        """Test joining two tables with a single column each."""
        self.create_test_table(conn, "test_t1", {"close": [100, 101, 102, 103, 104]})
        self.create_test_table(conn, "test_t2", {"close": [200, 201, 202, 203, 204]})
        
        result_table = pass_through._join_tables(conn, {"t1": "test_t1", "t2": "test_t2"})
        
        result = conn.query(f"SELECT * FROM {result_table} ORDER BY timestamp")
        
        assert list(result["t1_close"]) == [100, 101, 102, 103, 104]
        assert list(result["t2_close"]) == [200, 201, 202, 203, 204]
    
    def test_two_tables_multiple_columns(self, conn, pass_through):
        """Test joining two tables with multiple columns each."""
        self.create_test_table(conn, "test_t1", {
            "close": [100, 101, 102, 103, 104],
            "volume": [1000, 1100, 1200, 1300, 1400]
        })
        self.create_test_table(conn, "test_t2", {
            "close": [200, 201, 202, 203, 204],
            "volume": [2000, 2100, 2200, 2300, 2400]
        })
        
        result_table = pass_through._join_tables(conn, {"t1": "test_t1", "t2": "test_t2"}, columns=["close", "volume"])
        
        result = conn.query(f"SELECT * FROM {result_table} ORDER BY timestamp")
        
        assert list(result["t1_close"]) == [100, 101, 102, 103, 104]
        assert list(result["t1_volume"]) == [1000, 1100, 1200, 1300, 1400]
        assert list(result["t2_close"]) == [200, 201, 202, 203, 204]
        assert list(result["t2_volume"]) == [2000, 2100, 2200, 2300, 2400]
    
    def test_three_tables(self, conn, pass_through):
        """Test joining three tables."""
        self.create_test_table(conn, "test_a", {"close": [10, 11, 12]})
        self.create_test_table(conn, "test_b", {"close": [20, 21, 22]})
        self.create_test_table(conn, "test_c", {"close": [30, 31, 32]})
        
        result_table = pass_through._join_tables(conn, {
            "a": "test_a", 
            "b": "test_b", 
            "c": "test_c"
        })
        
        result = conn.query(f"SELECT * FROM {result_table} ORDER BY timestamp")
        
        assert list(result["a_close"]) == [10, 11, 12]
        assert list(result["b_close"]) == [20, 21, 22]
        assert list(result["c_close"]) == [30, 31, 32]
    
    def test_four_tables(self, conn, pass_through):
        """Test joining four tables."""
        self.create_test_table(conn, "test_w", {"close": [1, 2, 3]})
        self.create_test_table(conn, "test_x", {"close": [10, 20, 30]})
        self.create_test_table(conn, "test_y", {"close": [100, 200, 300]})
        self.create_test_table(conn, "test_z", {"close": [1000, 2000, 3000]})
        
        result_table = pass_through._join_tables(conn, {
            "w": "test_w",
            "x": "test_x", 
            "y": "test_y",
            "z": "test_z"
        })
        
        result = conn.query(f"SELECT * FROM {result_table} ORDER BY timestamp")
        
        assert list(result["w_close"]) == [1, 2, 3]
        assert list(result["x_close"]) == [10, 20, 30]
        assert list(result["y_close"]) == [100, 200, 300]
        assert list(result["z_close"]) == [1000, 2000, 3000]
    
    def test_table_with_space_in_name(self, conn, pass_through):
        """Test table with space in input name."""
        self.create_test_table(conn, "test_apple", {"close": [100, 101]})
        self.create_test_table(conn, "test_google", {"close": [200, 201]})
        
        result_table = pass_through._join_tables(conn, {
            "my table": "test_apple",
            "other table": "test_google"
        })
        
        result = conn.query(f"SELECT * FROM {result_table} ORDER BY timestamp")
        
        assert list(result["my_table_close"]) == [100, 101]
        assert list(result["other_table_close"]) == [200, 201]
    
    def test_table_with_dot_in_name(self, conn, pass_through):
        """Test table with dot in input name."""
        self.create_test_table(conn, "test_a1", {"close": [100, 101]})
        self.create_test_table(conn, "test_b2", {"close": [200, 201]})
        
        result_table = pass_through._join_tables(conn, {
            "a.b": "test_a1",
            "c.d": "test_b2"
        })
        
        result = conn.query(f"SELECT * FROM {result_table} ORDER BY timestamp")
        
        assert list(result["a_b_close"]) == [100, 101]
        assert list(result["c_d_close"]) == [200, 201]
    
    def test_table_with_hyphen_in_name(self, conn, pass_through):
        """Test table with hyphen in input name."""
        self.create_test_table(conn, "test_x", {"close": [100, 101]})
        self.create_test_table(conn, "test_y", {"close": [200, 201]})
        
        result_table = pass_through._join_tables(conn, {
            "us-stock": "test_x",
            "eu-stock": "test_y"
        })
        
        result = conn.query(f"SELECT * FROM {result_table} ORDER BY timestamp")
        
        assert list(result["us_stock_close"]) == [100, 101]
        assert list(result["eu_stock_close"]) == [200, 201]
    
    def test_custom_columns(self, conn):
        """Test _join_tables with custom columns parameter."""
        class CustomFn(TimeSeriesFunction):
            def __init__(self):
                super().__init__(columns=["close"])
            
            @property
            def output_columns(self):
                return [("result", "Float64")]
            
            def apply(self, conn, input_table):
                return input_table
        
        fn = CustomFn()
        
        self.create_test_table(conn, "test_c1", {"close": [100, 101], "high": [110, 111], "low": [90, 91]})
        self.create_test_table(conn, "test_c2", {"close": [200, 201], "high": [210, 211], "low": [190, 191]})
        
        result_table = fn._join_tables(conn, {"c1": "test_c1", "c2": "test_c2"}, columns=["close", "high"])
        
        result = conn.query(f"SELECT * FROM {result_table} ORDER BY timestamp")
        
        assert "c1_close" in result.columns
        assert "c1_high" in result.columns
        assert "c2_close" in result.columns
        assert "c2_high" in result.columns
        assert "c1_low" not in result.columns
        assert "c2_low" not in result.columns
    
    def test_single_input_returns_original(self, conn, pass_through):
        """Test that single input returns the original table."""
        self.create_test_table(conn, "test_single", {"close": [100, 101, 102]})
        
        result_table = pass_through._join_tables(conn, {"single": "test_single"})
        
        assert result_table == "test_single"
    
    def test_mixed_column_types(self, conn):
        """Test with different column types (Int64, Float64) - same column name."""
        # Both tables have "value" column but different types
        self.create_test_table(conn, "test_int_tbl", {"value": [10, 20]})  # Infers Float64
        self.create_test_table(conn, "test_float_tbl", {"value": [100.5, 200.5]})
        
        class IntFloatFn(TimeSeriesFunction):
            def __init__(self):
                super().__init__(columns=["value"])
            
            @property
            def output_columns(self):
                return [("result", "Float64")]
            
            def apply(self, conn, input_table):
                return input_table
        
        fn = IntFloatFn()
        result_table = fn._join_tables(conn, {
            "int_tbl": "test_int_tbl",
            "float_tbl": "test_float_tbl"
        })
        
        result = conn.query(f"SELECT * FROM {result_table} ORDER BY timestamp")
        
        assert list(result["int_tbl_value"]) == [10, 20]
        assert list(result["float_tbl_value"]) == [100.5, 200.5]
    
    def test_preserves_float_precision(self, conn, pass_through):
        """Test that float precision is preserved."""
        self.create_test_table(conn, "test_prec1", {"close": [100.123456, 101.654321]})
        self.create_test_table(conn, "test_prec2", {"close": [200.999999, 201.888888]})
        
        result_table = pass_through._join_tables(conn, {"prec1": "test_prec1", "prec2": "test_prec2"})
        
        result = conn.query(f"SELECT * FROM {result_table} ORDER BY timestamp")
        
        assert abs(result["prec1_close"].values[0] - 100.123456) < 0.0001
        assert abs(result["prec2_close"].values[0] - 200.999999) < 0.0001
    
    def test_timestamps_aligned(self, conn, pass_through):
        """Test that timestamps are properly aligned."""
        conn.execute("DROP TABLE IF EXISTS test_tsa1")
        conn.execute("""
            CREATE TABLE test_tsa1 (timestamp DateTime, close Float64) 
            ENGINE = MergeTree() ORDER BY timestamp
        """)
        conn.execute("""
            INSERT INTO test_tsa1 VALUES 
            ('2024-01-01 00:00:00', 100),
            ('2024-01-03 00:00:00', 102)
        """)
        
        conn.execute("DROP TABLE IF EXISTS test_tsa2")
        conn.execute("""
            CREATE TABLE test_tsa2 (timestamp DateTime, close Float64) 
            ENGINE = MergeTree() ORDER BY timestamp
        """)
        conn.execute("""
            INSERT INTO test_tsa2 VALUES 
            ('2024-01-01 00:00:00', 200),
            ('2024-01-02 00:00:00', 201),
            ('2024-01-03 00:00:00', 202)
        """)
        
        result_table = pass_through._join_tables(conn, {"tsa1": "test_tsa1", "tsa2": "test_tsa2"})
        
        result = conn.query(f"SELECT * FROM {result_table} ORDER BY timestamp")
        
        assert len(result) == 3
        # ClickHouse returns timestamps as strings
        ts_1 = result[result["timestamp"] == "2024-01-01 00:00:00"]["tsa1_close"].values[0]
        ts_2 = result[result["timestamp"] == "2024-01-02 00:00:00"]["tsa1_close"].values[0]
        ts_3 = result[result["timestamp"] == "2024-01-03 00:00:00"]["tsa1_close"].values[0]
        
        assert ts_1 == 100
        assert ts_2 == 0  # No data for tsa1 on 2024-01-02, should be 0
        assert ts_3 == 102
    
    def test_large_number_of_columns(self, conn, pass_through):
        """Test with many columns (stress test)."""
        cols = {f"col{i}": [i * 10 + j for j in range(3)] for i in range(10)}
        
        self.create_test_table(conn, "test_lg1", cols)
        self.create_test_table(conn, "test_lg2", {f"col{i}": [i * 100 + j for j in range(3)] for i in range(10)})
        
        col_list = [f"col{i}" for i in range(10)]
        result_table = pass_through._join_tables(conn, {"lg1": "test_lg1", "lg2": "test_lg2"}, columns=col_list)
        
        result = conn.query(f"SELECT * FROM {result_table} ORDER BY timestamp")
        
        for i in range(10):
            assert list(result[f"lg1_col{i}"]) == [i * 10 + j for j in range(3)]
            assert list(result[f"lg2_col{i}"]) == [i * 100 + j for j in range(3)]
    
    def test_different_row_counts(self, conn, pass_through):
        """Test tables with different numbers of rows."""
        self.create_test_table(conn, "test_short", {"close": [100, 101]})
        self.create_test_table(conn, "test_long", {"close": [200, 201, 202, 203, 204]})
        
        result_table = pass_through._join_tables(conn, {"short": "test_short", "long": "test_long"})
        
        result = conn.query(f"SELECT * FROM {result_table} ORDER BY timestamp")
        
        assert len(result) == 5
        assert list(result["short_close"]) == [100, 101, 0, 0, 0]
        assert list(result["long_close"]) == [200, 201, 202, 203, 204]
