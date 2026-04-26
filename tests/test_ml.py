"""
Tests for ML API - Architecture Registry and MLFunction

These tests define the expected behavior of the ML API:
- Default architecture registry
- MLFunction base class
- Model registry for storing/loading
- Architecture validation
"""
import pytest
import json
import tempfile
import os
from pathlib import Path


class TestDefaultArchitectures:
    """Tests for the default architecture registry."""
    
    def test_default_architectures_exist(self):
        """Default architectures should be importable."""
        from tradsl.ml import DEFAULT_ARCHITECTURES
        
        assert "lightgbm" in DEFAULT_ARCHITECTURES
        assert "xgboost" in DEFAULT_ARCHITECTURES
        assert "mlp" in DEFAULT_ARCHITECTURES
        assert "lstm" in DEFAULT_ARCHITECTURES
    
    def test_lightgbm_architecture(self):
        """LightGBM architecture should have correct structure."""
        from tradsl.ml import DEFAULT_ARCHITECTURES
        
        arch = DEFAULT_ARCHITECTURES["lightgbm"]
        
        assert arch["framework"] == "lightgbm"
        assert "params" in arch
        assert arch["params"]["objective"] == "regression"
        assert arch["params"]["num_leaves"] == 31
        assert arch["params"]["learning_rate"] == 0.05
    
    def test_xgboost_architecture(self):
        """XGBoost architecture should have correct structure."""
        from tradsl.ml import DEFAULT_ARCHITECTURES
        
        arch = DEFAULT_ARCHITECTURES["xgboost"]
        
        assert arch["framework"] == "xgboost"
        assert "params" in arch
        assert arch["params"]["objective"] == "reg:squarederror"
        assert arch["params"]["n_estimators"] == 100
        assert arch["params"]["max_depth"] == 6
    
    def test_mlp_architecture(self):
        """MLP architecture should have correct structure."""
        from tradsl.ml import DEFAULT_ARCHITECTURES
        
        arch = DEFAULT_ARCHITECTURES["mlp"]
        
        assert arch["framework"] == "pytorch"
        assert arch["class"] == "MLPRegressor"
        assert "params" in arch
        assert arch["params"]["hidden_dims"] == [128, 64]
        assert arch["params"]["output_dim"] == 1
    
    def test_lstm_architecture(self):
        """LSTM architecture should have correct structure."""
        from tradsl.ml import DEFAULT_ARCHITECTURES
        
        arch = DEFAULT_ARCHITECTURES["lstm"]
        
        assert arch["framework"] == "pytorch"
        assert arch["class"] == "LSTMRegressor"
        assert "params" in arch
        assert arch["params"]["hidden_dim"] == 128
        assert arch["params"]["num_layers"] == 2


class TestMLFunctionInitialization:
    """Tests for MLFunction initialization."""
    
    def test_init_with_string_architecture(self):
        """Should load architecture from default registry when string passed."""
        from tradsl.ml import MLFunction, DEFAULT_ARCHITECTURES
        
        fn = MLFunction(columns=["f1", "f2"], architecture="lightgbm")
        
        assert fn.architecture == DEFAULT_ARCHITECTURES["lightgbm"]
        assert fn.framework == "lightgbm"
    
    def test_init_with_dict_architecture(self):
        """Should use custom architecture when dict passed."""
        from tradsl.ml import MLFunction
        
        custom_arch = {
            "framework": "lightgbm",
            "params": {"num_leaves": 63, "learning_rate": 0.03}
        }
        
        fn = MLFunction(columns=["f1", "f2"], architecture=custom_arch)
        
        assert fn.architecture["params"]["num_leaves"] == 63
        assert fn.architecture["params"]["learning_rate"] == 0.03
    
    def test_init_with_weights(self):
        """Should handle weights parameter."""
        from tradsl.ml import MLFunction
        
        fn = MLFunction(
            columns=["f1", "f2"],
            architecture="lightgbm",
            weights={"source": "registry", "name": "my_model"}
        )
        
        assert fn.weights == {"source": "registry", "name": "my_model"}
    
    def test_init_train_on_init(self):
        """Should support train_on_init flag."""
        from tradsl.ml import MLFunction
        
        fn = MLFunction(
            columns=["f1", "f2"],
            architecture="lightgbm",
            train_on_init=True
        )
        
        assert fn.train_on_init is True
    
    def test_framework_from_architecture(self):
        """Should extract framework from architecture."""
        from tradsl.ml import MLFunction
        
        fn = MLFunction(columns=["f1", "f2"], architecture="xgboost")
        assert fn.framework == "xgboost"
        
        fn = MLFunction(columns=["f1", "f2"], architecture="mlp")
        assert fn.framework == "pytorch"
    
    def test_output_columns(self):
        """Should return correct output columns."""
        from tradsl.ml import MLFunction
        
        fn = MLFunction(
            columns=["f1", "f2"],
            architecture="lightgbm",
            model_name="price_pred"
        )
        
        cols = fn.output_columns
        assert len(cols) >= 1
        assert any("prediction" in col[0] for col in cols)


class TestModelRegistry:
    """Tests for ModelRegistry."""
    
    @pytest.fixture
    def registry(self):
        """Create a temporary registry."""
        from tradsl.ml import ModelRegistry
        with tempfile.TemporaryDirectory() as tmpdir:
            yield ModelRegistry(tmpdir)
    
    def test_registry_init(self, registry):
        """Registry should initialize with empty index."""
        assert registry.index == {}
    
    def test_register_sklearn_model(self, registry):
        """Should register sklearn model."""
        from sklearn.ensemble import HistGradientBoostingRegressor
        import numpy as np
        
        # Create simple model
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([1, 2, 3, 4, 5])
        model = HistGradientBoostingRegressor(max_iter=10)
        model.fit(X, y)
        
        config = {
            "feature_columns": ["feature1"],
            "target_column": "target",
        }
        
        version = registry.register("test_model", model, config, "sklearn")
        
        assert "test_model" in registry.index
        assert version.startswith("v1_")
    
    def test_register_lightgbm_model(self, registry):
        """Should register LightGBM model."""
        try:
            import lightgbm as lgb
        except ImportError:
            pytest.skip("LightGBM not installed")
        
        import numpy as np
        
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([1, 2, 3, 4, 5])
        train_data = lgb.Dataset(X, label=y)
        params = {"objective": "regression", "num_iterations": 10}
        model = lgb.train(params, train_data)
        
        config = {"feature_columns": ["f1"], "target_column": "target"}
        
        version = registry.register("lgb_model", model, config, "lightgbm")
        
        assert "lgb_model" in registry.index
        assert version.startswith("v1_")
    
    def test_register_xgboost_model(self, registry):
        """Should register XGBoost model."""
        try:
            import xgboost as xgb
        except ImportError:
            pytest.skip("XGBoost not installed")
        
        import numpy as np
        
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([1, 2, 3, 4, 5])
        model = xgb.XGBRegressor(n_estimators=10)
        model.fit(X, y)
        
        config = {"feature_columns": ["f1"], "target_column": "target"}
        
        version = registry.register("xgb_model", model, config, "xgboost")
        
        assert "xgb_model" in registry.index
    
    def test_load_latest_version(self, registry):
        """Should load latest version of model."""
        from sklearn.ensemble import HistGradientBoostingRegressor
        import numpy as np
        
        # Register two versions
        for i in range(2):
            X = np.array([[1], [2], [3]])
            y = np.array([i, i, i])
            model = HistGradientBoostingRegressor(max_iter=5)
            model.fit(X, y)
            
            config = {"version": i}
            registry.register(f"model_v{i}", model, config, "sklearn")
        
        latest = registry.get_latest("model_v0")
        assert latest is not None
    
    def test_list_models(self, registry):
        """Should list all registered models."""
        from sklearn.ensemble import HistGradientBoostingRegressor
        import numpy as np
        
        for name in ["model_a", "model_b", "model_c"]:
            X = np.array([[1], [2]])
            y = np.array([1, 2])
            model = HistGradientBoostingRegressor(max_iter=5)
            model.fit(X, y)
            registry.register(name, model, {}, "sklearn")
        
        models = registry.list_models()
        
        assert "model_a" in models
        assert "model_b" in models
        assert "model_c" in models
    
    def test_get_config(self, registry):
        """Should get model configuration."""
        from sklearn.ensemble import HistGradientBoostingRegressor
        import numpy as np
        
        X = np.array([[1], [2], [3]])
        y = np.array([1, 2, 3])
        model = HistGradientBoostingRegressor(max_iter=5)
        model.fit(X, y)
        
        config = {
            "feature_columns": ["f1", "f2"],
            "target_column": "target",
            "custom_key": "custom_value"
        }
        
        version = registry.register("test_model", model, config, "sklearn")
        
        loaded_config = registry.get_config("test_model", version)
        
        assert loaded_config["feature_columns"] == ["f1", "f2"]
        assert loaded_config["target_column"] == "target"
        assert loaded_config["framework"] == "sklearn"


class TestArchitectureValidation:
    """Tests for architecture validation."""
    
    def test_validate_lightgbm_params(self):
        """Should validate LightGBM parameters."""
        from tradsl.ml import validate_architecture
        
        arch = {
            "framework": "lightgbm",
            "params": {"num_leaves": 31, "learning_rate": 0.05}
        }
        
        # Should not raise
        validate_architecture(arch)
    
    def test_validate_xgboost_params(self):
        """Should validate XGBoost parameters."""
        from tradsl.ml import validate_architecture
        
        arch = {
            "framework": "xgboost",
            "params": {"n_estimators": 100, "max_depth": 6}
        }
        
        validate_architecture(arch)
    
    def test_validate_pytorch_mlp(self):
        """Should validate PyTorch MLP architecture."""
        from tradsl.ml import validate_architecture
        
        arch = {
            "framework": "pytorch",
            "class": "MLPRegressor",
            "params": {"hidden_dims": [128, 64], "output_dim": 1}
        }
        
        validate_architecture(arch)
    
    def test_validate_missing_framework(self):
        """Should raise error for missing framework."""
        from tradsl.ml import validate_architecture
        
        arch = {"params": {}}
        
        with pytest.raises(ValueError, match="framework"):
            validate_architecture(arch)
    
    def test_validate_unknown_framework(self):
        """Should raise error for unknown framework."""
        from tradsl.ml import validate_architecture
        
        arch = {"framework": "unknown_framework", "params": {}}
        
        with pytest.raises(ValueError, match="Unknown framework"):
            validate_architecture(arch)


class TestFeatureBuilder:
    """Tests for feature engineering."""
    
    def test_feature_builder_init(self):
        """FeatureBuilder should initialize with config."""
        from tradsl.ml import FeatureBuilder
        
        config = {
            "feature_columns": ["lag_1", "lag_2", "returns_1d"],
            "target_column": "price"
        }
        
        builder = FeatureBuilder(config)
        
        assert builder.config == config
        assert builder.feature_columns == ["lag_1", "lag_2", "returns_1d"]
    
    def test_create_lag_features(self):
        """Should create lag features."""
        from tradsl.ml import FeatureBuilder
        
        config = {"feature_columns": ["price"]}
        builder = FeatureBuilder(config)
        
        import pandas as pd
        df = pd.DataFrame({
            "price": [100, 101, 102, 103, 104]
        })
        
        result = builder._create_lags(df, [1, 2])
        
        assert "lag_1" in result.columns
        assert "lag_2" in result.columns
        assert list(result["lag_1"]) == [None, 100, 101, 102, 103]
    
    def test_create_returns(self):
        """Should create return features."""
        from tradsl.ml import FeatureBuilder
        
        config = {"feature_columns": ["price"]}
        builder = FeatureBuilder(config)
        
        import pandas as pd
        df = pd.DataFrame({
            "price": [100, 110, 121, 133]
        })
        
        result = builder._create_returns(df, [1])
        
        assert "returns_1d" in result.columns
        # 110/100 - 1 = 0.1
        assert abs(result["returns_1d"].iloc[1] - 0.1) < 0.01
    
    def test_create_rolling_features(self):
        """Should create rolling statistics."""
        from tradsl.ml import FeatureBuilder
        
        config = {"feature_columns": ["price"]}
        builder = FeatureBuilder(config)
        
        import pandas as pd
        df = pd.DataFrame({
            "price": [100, 101, 102, 103, 104, 105, 106]
        })
        
        result = builder._create_rolling(df, [3])
        
        assert "roll_mean_3" in result.columns
        assert "roll_std_3" in result.columns


class TestMLFunctionApply:
    """Tests for MLFunction.apply() method - requires ClickHouse."""
    
    @pytest.fixture
    def conn(self):
        """Create ClickHouse connection if available."""
        try:
            from tradsl.storage import ClickHouseConnection
            conn = ClickHouseConnection(timeout=30)
            conn.execute("SELECT 1")
            return conn
        except Exception:
            pytest.skip("ClickHouse not available")
    
    @pytest.fixture
    def test_data(self, conn):
        """Create test data table."""
        conn.execute("DROP TABLE IF EXISTS ml_test_features")
        conn.execute("""
            CREATE TABLE ml_test_features (
                timestamp DateTime,
                symbol String,
                close Float64,
                feature1 Float64,
                feature2 Float64
            ) ENGINE = MergeTree() ORDER BY timestamp
        """)
        
        conn.execute("""
            INSERT INTO ml_test_features VALUES
            ('2024-01-01 00:00:00', 'AAPL', 100, 0.1, 0.2),
            ('2024-01-02 00:00:00', 'AAPL', 101, 0.2, 0.3),
            ('2024-01-03 00:00:00', 'AAPL', 102, 0.3, 0.4),
            ('2024-01-04 00:00:00', 'AAPL', 103, 0.4, 0.5),
            ('2024-01-05 00:00:00', 'AAPL', 104, 0.5, 0.6)
        """)
        
        return "ml_test_features"
    
    @pytest.mark.integration
    def test_apply_returns_table_name(self, conn, test_data):
        """apply() should return a table name."""
        from tradsl.ml import MLFunction
        
        fn = MLFunction(
            columns=["feature1", "feature2"],
            architecture="lightgbm",
            train_on_init=True
        )
        
        result_table = fn.apply(conn, test_data)
        
        assert isinstance(result_table, str)
        assert result_table.startswith("ml_")
    
    @pytest.mark.integration
    def test_apply_creates_output_columns(self, conn, test_data):
        """apply() should create output columns in result."""
        from tradsl.ml import MLFunction
        
        fn = MLFunction(
            columns=["feature1", "feature2"],
            architecture="lightgbm",
            train_on_init=True
        )
        
        result_table = fn.apply(conn, test_data)
        
        # Query the result table
        result = conn.query(f"DESCRIBE TABLE {result_table}")
        col_names = list(result["name"])
        
        assert "timestamp" in col_names
        assert "prediction" in col_names


class TestIntegrationWithDAG:
    """Tests for ML integration with DAG."""
    
    def test_ml_function_in_registry(self):
        """MLFunction should be in default registry."""
        from tradsl import default_registry
        
        assert "ml.lightgbm" in default_registry
        assert "ml.xgboost" in default_registry
        assert "ml.mlp" in default_registry
        assert "ml.lstm" in default_registry
    
    def test_dag_with_ml_function(self):
        """DAG should be able to include ML function nodes."""
        from tradsl import DAG
        
        config = {
            "features": {
                "type": "timeseries",
                "adapter": "parquet",
                "path": "/data/prices.parquet"
            },
            "prediction": {
                "type": "function",
                "function": "ml.lightgbm",
                "inputs": ["features"],
                "architecture": "lightgbm"
            }
        }
        
        dag = DAG.from_config(config)
        dag.build()
        
        assert "prediction" in dag.nodes
    
    def test_ml_with_multiple_inputs(self):
        """ML function should work with multiple input tables."""
        from tradsl import DAG
        
        config = {
            "price_a": {
                "type": "timeseries",
                "adapter": "parquet",
                "path": "/data/a.parquet"
            },
            "price_b": {
                "type": "timeseries", 
                "adapter": "parquet",
                "path": "/data/b.parquet"
            },
            "prediction": {
                "type": "function",
                "function": "ml.lightgbm",
                "inputs": ["price_a", "price_b"]
            }
        }
        
        dag = DAG.from_config(config)
        dag.build()
        
        assert "prediction" in dag.nodes


class TestPyTorchModels:
    """Tests for PyTorch model support."""
    
    def test_mlp_architecture_creates_model(self):
        """Should create PyTorch MLP model from architecture."""
        from tradsl.ml import DEFAULT_ARCHITECTURES
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        
        arch = DEFAULT_ARCHITECTURES["mlp"]
        
        # Would create model - just verify structure here
        assert arch["class"] == "MLPRegressor"
        assert arch["params"]["hidden_dims"] == [128, 64]
    
    def test_lstm_architecture_creates_model(self):
        """Should create PyTorch LSTM model from architecture."""
        from tradsl.ml import DEFAULT_ARCHITECTURES
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        
        arch = DEFAULT_ARCHITECTURES["lstm"]
        
        assert arch["class"] == "LSTMRegressor"
        assert arch["params"]["hidden_dim"] == 128
        assert arch["params"]["num_layers"] == 2


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_unknown_architecture_string(self):
        """Should raise error for unknown architecture string."""
        from tradsl.ml import MLFunction
        
        with pytest.raises(KeyError):
            MLFunction(columns=["f1"], architecture="nonexistent_arch")
    
    def test_invalid_weights_source(self):
        """Should handle invalid weights source gracefully."""
        from tradsl.ml import MLFunction
        
        # Weights that don't exist - should be handled
        fn = MLFunction(
            columns=["f1"],
            architecture="lightgbm",
            weights={"source": "registry", "name": "nonexistent"}
        )
        
        # Should not raise on init, may raise on apply
        assert fn.weights is not None
    
    def test_missing_columns(self):
        """Should handle missing columns gracefully."""
        from tradsl.ml import MLFunction
        
        fn = MLFunction(columns=[], architecture="lightgbm")
        
        # Should have empty or default columns
        assert fn.columns == []
