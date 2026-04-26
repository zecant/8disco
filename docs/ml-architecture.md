# ML Model Architecture for TradSL

## Executive Summary

This document outlines the architecture for ML model integration in TradSL. We design a framework-agnostic approach that supports:

1. **Parameterized architectures**: Pass model parameters directly (e.g., LightGBM params)
2. **Custom architectures**: Pass PyTorch/keras neural network definitions
3. **Pre-trained weights**: Load from registry (file-based or model hub)
4. **Default architecture registry**: Built-in model templates

The key is: **architecture (parameters or code) + weights = model**

---

## 1. Design Principles

### 1.1 Core Requirements
- **Framework Agnostic**: Support sklearn, LightGBM, XGBoost, PyTorch, ONNX
- **Lightweight**: Minimal dependencies at inference time
- **Secure**: No arbitrary code execution (avoid pickle from untrusted sources)
- **Versioned**: Model versioning with timestamps and lineage
- **Composable**: Works with existing DAG and multi-input joining

### 1.2 Three Model Loading Modes

| Mode | Example | Use Case |
|------|---------|----------|
| **Parameters only** | `{"n_estimators": 100, "learning_rate": 0.05}` | Quick prototyping, simple models |
| **Architecture + Weights** | PyTorch model code + .safetensors | Custom neural networks |
| **Registry load** | `"model_name": "price_predictor"` | Production, versioned models |

### 1.3 Design Philosophy
```
┌────────────────────────────────────────────────────────────────┐
│                    MODEL = ARCHITECTURE + WEIGHTS              │
├────────────────────────────────────────────────────────────────┤
│  ARCHITECTURE (how)          │    WEIGHTS (what)              │
│  ─────────────────────        │    ─────────────               │
│  • Parameters dict (LightGBM) │ • Trained .safetensors         │
│  • Model code (PyTorch)       │ • Trained .joblib              │
│  • Config JSON                │ • Registry version              │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. Class Architecture

### 2.1 Architecture Definition

Our architecture is defined by dictionary parameters:

```python
# Example: Parameterized LightGBM architecture
ARCH_LGBM = {
    "framework": "lightgbm",
    "params": {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
    }
}

# Example: PyTorch architecture (code + weights)
ARCH_PYTORCH = {
    "framework": "pytorch",
    "class": "PricePredictor",  # Class name in model_code
    "params": {
        "input_dim": 10,
        "hidden_dim": 128,
        "output_dim": 1
    },
    "weights": "model_weights.safetensors"
}

### 2.3 Default Architecture Registry

Initial set of pre-built architectures:

```python
DEFAULT_ARCHITECTURES = {
    # ===== Gradient Boosting Models =====
    "lightgbm": {
        "framework": "lightgbm",
        "params": {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
        }
    },
    
    "xgboost": {
        "framework": "xgboost",
        "params": {
            "objective": "reg:squarederror",
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
    },
    
    # ===== Neural Networks (PyTorch) =====
    "mlp": {
        "framework": "pytorch",
        "class": "MLPRegressor",
        "params": {
            "input_dim": "auto",  # Set from data at init
            "hidden_dims": [128, 64],
            "output_dim": 1,
            "activation": "relu",
            "dropout": 0.2,
        }
    },
    
    "lstm": {
        "framework": "pytorch",
        "class": "LSTMRegressor",
        "params": {
            "input_dim": "auto",
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout": 0.2,
            "output_dim": 1,
        }
    },
}
```

### How to Use Default Architectures

```python
# Method 1: Use string from registry (recommended for DSL)
ml_fn = MLFunction(
    columns=["feature1", "feature2"],
    architecture="lightgbm"  # Loads from DEFAULT_ARCHITECTURES
)

# Method 2: Pass custom parameters (overrides defaults)
ml_fn = MLFunction(
    columns=["feature1", "feature2"],
    architecture={
        "framework": "xgboost",
        "params": {
            "max_depth": 8,  # Override default
            "learning_rate": 0.03,
        }
    }
)

# Method 3: Load weights from registry
ml_fn = MLFunction(
    columns=["feature1", "feature2"],
    architecture="lightgbm",
    weights={
        "source": "registry",
        "name": "price_model_v1"
    }
)

# Method 4: Custom PyTorch architecture
ml_fn = MLFunction(
    columns=["feature1", "feature2"],
    architecture={
        "framework": "pytorch",
        "class": "CustomPricePredictor",
        "model_code": "...",  # or import from module
        "params": {"hidden_dim": 256}
    },
    weights="/path/to/weights.safetensors"
)
```

### 2.2 Base Class: `MLFunction`

```python
class MLFunction(TimeSeriesFunction):
    """
    Base class for ML models.
    
    Three ways to specify the model:
    1. architecture dict (parameters only - trains on load)
    2. architecture + registry weights
    3. custom model code (PyTorch/Custom class)
    
    Example:
        price_ml:
            type=function
            function=ml.lightgbm
            inputs:[features]
            architecture:
                framework: lightgbm
                params:
                    num_leaves: 31
                    learning_rate: 0.05
            weights:  # optional, trains if not provided
                source: registry
                name: price_predictor
                version: latest
    """
    
    def __init__(
        self,
        columns: str | list[str],
        architecture: dict | str,  # Can pass dict directly or string from registry
        weights: dict | str = None,  # Optional weights source
        train_on_init: bool = False,  # If True, train when no weights provided
        **kwargs
    ):
        super().__init__(columns=columns, **kwargs)
        
        # Handle architecture specification
        if isinstance(architecture, str):
            # Load from default registry
            self.architecture = DEFAULT_ARCHITECTURES[architecture]
        else:
            self.architecture = architecture
        
        if isinstance(weights, str):
            # Load from registry
            self.weights = registry.load(weights)
        else:
            self.weights = weights
        
        self.train_on_init = train_on_init
        self._model = None
        self._config = self.architecture.get('params', {})
        self.framework = self.architecture.get('framework', 'sklearn')
    
    @property
    def output_columns(self) -> list[tuple[str, str]]:
        return [
            (f"{self.model_name}_prediction", "Float64"),
            (f"{self.model_name}_confidence", "Float64"),  # if available
        ]
    
    def _load_model(self):
        """Load model from registry or path."""
        # Implementation in Section 3
        pass
    
    def _get_features(self, conn, input_table: str) -> pd.DataFrame:
        """Extract features from input table."""
        # Implementation in Section 4
        pass
    
    def apply(self, conn: "ClickHouseConnection", input_table: str) -> str:
        """Run inference in database or Python."""
        # Implementation in Section 5
        pass
```

### 2.2 Why Separate Architecture from Weights?

| Aspect | Combined (pickle) | Separated (our approach) |
|--------|-------------------|-------------------------|
| Security | Unsafe (code execution) | Safe (only weights/ONNX) |
| Size | Large | Small (just weights) |
| Versioning | Full model | Just weights/hash |
| Framework swap | Requires retrain | Swap weights |
| Debugging | Self-contained | More complex |

---

## 3. Model Registry

### 3.1 Registry Structure

```
./models/
├── registry.json           # Index of all models
├── price_predictor/
│   ├── v1_20240115/
│   │   ├── model.sklearn    # sklearn weights
│   │   └── config.json      # Architecture config
│   ├── v2_20240201/
│   │   ├── model.onnx       # ONNX format
│   │   └── config.json
│   └── latest -> v2         # Symlink
└── feature_config/
    └── price_features.json
```

### 3.2 Registry Implementation

```python
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional
import joblib
import numpy as np

class ModelRegistry:
    """
    Framework-agnostic model registry.
    
    Storage format:
    - sklearn: joblib
    - lightgbm: .lgb (native)
    - xgboost: .json (native)
    - onnx: .onnx
    - pytorch: safetensors (state_dict)
    """
    
    FRAMEWORK_EXTENSIONS = {
        'sklearn': ['.joblib', '.pkl'],
        'lightgbm': ['.lgb'],
        'xgboost': ['.json'],
        'onnx': ['.onnx'],
        'pytorch': ['.pt', '.safetensors'],
    }
    
    def __init__(self, base_path: str = "./models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._load_index()
    
    def _load_index(self):
        index_path = self.base_path / "registry.json"
        if index_path.exists():
            with open(index_path) as f:
                self.index = json.load(f)
        else:
            self.index = {}
    
    def _save_index(self):
        with open(self.base_path / "registry.json", 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def register(
        self,
        model_name: str,
        model,
        config: dict,
        framework: str = "sklearn"
    ) -> str:
        """
        Register a new model version.
        
        Args:
            model_name: Name of the model (e.g., "price_predictor")
            model: Fitted model object or path
            config: Architecture configuration
            framework: One of sklearn/lightgbm/xgboost/onnx/pytorch
            
        Returns:
            Version string (e.g., "v1_20240115")
        """
        timestamp = datetime.now().strftime("%Y%m%d")
        
        # Determine version number
        versions = self.index.get(model_name, {})
        version_num = len(versions) + 1
        version = f"v{version_num}_{timestamp}"
        
        # Save model based on framework
        model_dir = self.base_path / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = self._save_model(model, model_dir, framework)
        
        # Compute integrity hash
        with open(model_path, 'rb') as f:
            artifact_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        
        # Save config
        config_path = model_dir / "config.json"
        config['framework'] = framework
        config['artifact_hash'] = artifact_hash
        config['created_at'] = timestamp
        config['feature_columns'] = config.get('feature_columns', [])
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Update index
        if model_name not in self.index:
            self.index[model_name] = {}
        
        self.index[model_name][version] = {
            'version': version_num,
            'timestamp': timestamp,
            'artifact_path': str(model_path),
            'artifact_hash': artifact_hash,
            'framework': framework,
            'config_path': str(config_path),
            'status': 'staging'
        }
        
        self._save_index()
        
        # Update latest symlink
        latest_link = self.base_path / model_name / "latest"
        if latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(version)
        
        return version
    
    def _save_model(self, model, model_dir: Path, framework: str) -> Path:
        """Save model in framework-appropriate format."""
        ext = self.FRAMEWORK_EXTENSIONS[framework][0]
        model_path = model_dir / f"model{ext}"
        
        if framework == 'sklearn':
            joblib.dump(model, model_path)
        elif framework == 'onnx':
            # ONNX already saved externally
            pass
        elif framework == 'lightgbm':
            model.save_model(str(model_path))
        elif framework == 'xgboost':
            model.save_model(str(model_path))
        elif framework == 'pytorch':
            import torch
            torch.save(model.state_dict(), model_path)
        
        return model_path
    
    def load(self, model_name: str, version: str = "latest"):
        """Load model by name and version."""
        if version == "latest":
            version = self.get_latest(model_name)
        
        entry = self.index[model_name][version]
        model_path = entry['artifact_path']
        framework = entry['framework']
        
        # Load based on framework
        if framework == 'sklearn':
            return joblib.load(model_path)
        elif framework == 'onnx':
            import onnxruntime as ort
            return ort.InferenceSession(model_path)
        elif framework == 'lightgbm':
            import lightgbm as lgb
            return lgb.Booster(model_file=model_path)
        elif framework == 'xgboost':
            import xgboost as xgb
            return xgb.XGBRegressor()
            model.load_model(model_path)
        elif framework == 'pytorch':
            import torch
            # Requires architecture reconstruction
            return torch.load(model_path)
    
    def get_latest(self, model_name: str) -> str:
        """Get latest version for model."""
        latest_link = self.base_path / model_name / "latest"
        if latest_link.is_symlink():
            return latest_link.readlink()
        # Fallback to highest version number
        versions = self.index.get(model_name, {})
        return max(versions.keys(), key=lambda v: versions[v]['version'])
    
    def get_config(self, model_name: str, version: str = "latest") -> dict:
        """Get model configuration."""
        if version == "latest":
            version = self.get_latest(model_name)
        
        entry = self.index[model_name][version]
        with open(entry['config_path']) as f:
            return json.load(f)
    
    def list_models(self) -> dict:
        """List all registered models."""
        return self.index.copy()
```

---

## 4. Feature Engineering

### 4.1 Feature Config Schema

```json
{
  "model_name": "price_predictor",
  "framework": "lightgbm",
  "feature_columns": [
    "lag_1", "lag_2", "lag_7",
    "returns_1d", "returns_7d",
    "roll_mean_7", "roll_std_7",
    "rsi_14", "macd", "macd_signal"
  ],
  "target_column": "price",
  "feature_groups": {
    "lags": ["lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_30"],
    "returns": ["returns_1d", "returns_7d", "returns_30d"],
    "rolling": ["roll_mean_7", "roll_std_7", "roll_mean_30", "roll_max_30"],
    "technical": ["rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower"]
  },
  "preprocessing": {
    "fill_na_method": "ffill",
    "scale": false
  }
}
```

### 4.2 Feature Builder

```python
class FeatureBuilder:
    """Build features from raw market data."""
    
    # Class-level feature definitions
    FEATURE_GROUPS = {
        'lags': lambda df, params: self._create_lags(df, params.get('lags', [1,2,3,5,7,14,30])),
        'returns': lambda df, params: self._create_returns(df, params.get('periods', [1,7,30])),
        'rolling': lambda df, params: self._create_rolling(df, params.get('windows', [7,14,30,60])),
        'technical': lambda df, params: self._create_technical(df, params.get('indicators', ['rsi', 'macd', 'bb'])),
    }
    
    def __init__(self, config: dict):
        self.config = config
        self.feature_columns = config.get('feature_columns', [])
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build all features from input DataFrame."""
        result = pd.DataFrame()
        
        for group_name, group_fn in self.FEATURE_GROUPS.items():
            if any(col.startswith(group_name) for col in self.feature_columns):
                group_df = group_fn(df, self.config.get(group_name, {}))
                result = pd.concat([result, group_df], axis=1)
        
        return result
    
    def _create_lags(self, df: pd.DataFrame, lags: list) -> pd.DataFrame:
        """Create lag features."""
        result = pd.DataFrame()
        for col in df.columns:
            for lag in lags:
                result[f'lag_{lag}'] = df[col].shift(lag)
        return result
    
    def _create_returns(self, df: pd.DataFrame, periods: list) -> pd.DataFrame:
        """Create return features."""
        result = pd.DataFrame()
        for col in df.columns:
            for period in periods:
                result[f'returns_{period}d'] = df[col].pct_change(period)
        return result
    
    def _create_rolling(self, df: pd.DataFrame, windows: list) -> pd.DataFrame:
        """Create rolling statistic features."""
        result = pd.DataFrame()
        for col in df.columns:
            for window in windows:
                result[f'roll_mean_{window}'] = df[col].rolling(window).mean()
                result[f'roll_std_{window}'] = df[col].rolling(window).std()
                result[f'roll_min_{window}'] = df[col].rolling(window).min()
                result[f'roll_max_{window}'] = df[col].rolling(window).max()
        return result
    
    def _create_technical(self, df: pd.DataFrame, indicators: list) -> pd.DataFrame:
        """Create technical indicator features."""
        result = pd.DataFrame()
        close = df['close'] if 'close' in df.columns else df.iloc[:, 0]
        
        if 'rsi' in indicators:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            result['rsi_14'] = 100 - (100 / (1 + rs))
        
        if 'macd' in indicators:
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            result['macd'] = ema12 - ema26
            result['macd_signal'] = result['macd'].ewm(span=9).mean()
        
        if 'bb' in indicators:
            sma = close.rolling(20).mean()
            std = close.rolling(20).std()
            result['bb_upper'] = sma + 2 * std
            result['bb_lower'] = sma - 2 * std
        
        return result
```

---

## 5. Concrete Implementations

### 5.1 LightGBM Implementation

```python
class LGBMFunction(MLFunction):
    """
    LightGBM-based prediction function.
    
    Example config:
        price_ml:
            type=function
            function=ml.lightgbm
            inputs:[price_table]
            model_name: price_predictor
            version: latest
            framework: lightgbm
    """
    
    DEFAULT_PARAMS = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
    }
    
    def __init__(self, columns: str | list[str], model_name: str, **kwargs):
        super().__init__(
            columns=columns,
            model_name=model_name,
            framework='lightgbm',
            **kwargs
        )
        self.registry = kwargs.pop('registry', ModelRegistry())
    
    def _load_model(self):
        """Load LightGBM model from registry."""
        self._model = self.registry.load(self.model_name, self.version)
        self._config = self.registry.get_config(self.model_name, self.version)
    
    def _get_features(self, conn, input_table: str) -> pd.DataFrame:
        """Extract features from ClickHouse."""
        feature_cols = self._config.get('feature_columns', self.columns)
        
        query = f"""
            SELECT {', '.join(feature_cols)}
            FROM {input_table}
            ORDER BY timestamp
        """
        return conn.query(query)
    
    def apply(self, conn: "ClickHouseConnection", input_table: str) -> str:
        """Run LightGBM inference."""
        if self._model is None:
            self._load_model()
        
        # Get features
        features = self._get_features(conn, input_table)
        
        # Handle missing values
        features = features.fillna(method='ffill').fillna(0)
        
        # Make predictions
        predictions = self._model.predict(features)
        
        # Create output table
        output_table = self._generate_output_table_name("lgbm")
        
        columns = "timestamp, symbol, price Float64, prediction Float64, confidence Float64"
        select = f"""
            SELECT 
                timestamp,
                symbol,
                close as price,
                {predictions[0]} as prediction,
                0.95 as confidence
            FROM {input_table}
        """
        
        return self._create_and_insert(conn, output_table, columns, select)
```

### 5.2 ONNX Implementation (Framework Agnostic)

```python
class ONNXFunction(MLFunction):
    """
    ONNX-based prediction function.
    
    Advantage: Framework-agnostic, can run with ONNX Runtime
    """
    
    def __init__(self, columns: str | list[str], model_name: str, **kwargs):
        super().__init__(
            columns=columns,
            model_name=model_name,
            framework='onnx',
            **kwargs
        )
    
    def _load_model(self):
        """Load ONNX model with ONNX Runtime."""
        import onnxruntime as ort
        
        entry = self.registry.index[self.model_name][self.version]
        sess = ort.InferenceSession(
            entry['artifact_path'],
            providers=['CPUExecutionProvider']
        )
        
        # Get input/output names from model
        self._input_name = sess.get_inputs()[0].name
        self._output_name = sess.get_outputs()[0].name
        self._session = sess
    
    def apply(self, conn: "ClickHouseConnection", input_table: str) -> str:
        """Run ONNX inference."""
        if self._session is None:
            self._load_model()
        
        features = self._get_features(conn, input_table)
        features = features.fillna(0).values.astype(np.float32)
        
        predictions = self._session.run(
            [self._output_name],
            {self._input_name: features}
        )[0]
        
        # Create output table
        output_table = self._generate_output_table_name("onnx")
        # ... rest of implementation
```

---

## 6. Training Pipeline

### 6.1 Training Function

```python
def train_ml_model(
    model_name: str,
    training_data: pd.DataFrame,
    target_column: str,
    feature_config: dict,
    framework: str = "sklearn",
    model_params: dict = None,
    registry: ModelRegistry = None
) -> str:
    """
    Train an ML model and register it.
    
    Args:
        model_name: Name for the model
        training_data: DataFrame with features and target
        target_column: Name of target column
        feature_config: Feature engineering config
        framework: One of sklearn/lightgbm/xgboost
        model_params: Model hyperparameters
        registry: ModelRegistry instance
        
    Returns:
        Version string (e.g., "v1_20240115")
    """
    # Build features
    builder = FeatureBuilder(feature_config)
    X = builder.build(training_data[feature_config['feature_columns']])
    y = training_data[target_column]
    
    # Train based on framework
    if framework == 'sklearn':
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(
            **(model_params or {'max_iter': 100, 'learning_rate': 0.1})
        )
        model.fit(X, y)
    
    elif framework == 'lightgbm':
        import lightgbm as lgb
        train_data = lgb.Dataset(X, label=y)
        params = model_params or {}
        model = lgb.train(params, train_data, num_boost_round=100)
    
    elif framework == 'xgboost':
        import xgboost as xgb
        model = xgb.XGBRegressor(
            **(model_params or {'n_estimators': 100, 'learning_rate': 0.1})
        )
        model.fit(X, y)
    
    # Build config
    config = {
        'model_name': model_name,
        'target_column': target_column,
        'feature_columns': feature_config['feature_columns'],
        'model_params': model_params or {},
    }
    
    # Register model
    version = registry.register(model_name, model, config, framework)
    
    return version
```

### 6.2 Training Example

```python
# Example training workflow
registry = ModelRegistry("./models")

# Load training data
training_data = conn.query("SELECT * FROM price_features WHERE date >= '2023-01-01'")

# Feature configuration
feature_config = {
    'feature_columns': [
        'lag_1', 'lag_2', 'lag_7',
        'returns_1d', 'returns_7d',
        'roll_mean_7', 'roll_std_7',
    ],
    'target_column': 'price'
}

# Train LightGBM model
version = train_ml_model(
    model_name='price_predictor',
    training_data=training_data,
    target_column='price',
    feature_config=feature_config,
    framework='lightgbm',
    model_params={
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
    },
    registry=registry
)

print(f"Trained model version: {version}")
```

---

## 7. Integration with DAG

### 7.1 Defining ML in DSL

```yaml
# tradsl config
price:
  type: timeseries
  adapter: parquet
  path: /data/prices.parquet

features:
  type: function
  function: custom.feature_engineering
  inputs: [price]

model:
  type: function
  function: ml.lightgbm
  inputs: [features]
  model_name: price_predictor
  version: latest
```

### 7.2 Multi-Model Composition

```yaml
# Ensemble of multiple models
ensemble:
  type: function
  function: ml.ensemble
  inputs: [model_lgbm, model_xgb, model_sklearn]
  weights: [0.4, 0.4, 0.2]
  method: weighted_average
```

---

## 8. Summary and Recommendations

### 8.1 Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| Separate architecture + weights | Security (no pickle), small size, versioning |
| Registry pattern | Simple, versioned, framework-agnostic |
| Framework-specific classes | Clean API, each framework has unique loading |
| ONNX for inference | Hardware optimization, framework swap |
| Feature config in JSON | Reproducibility, no hardcoded feature logic |

### 8.2 Priority Implementation

1. **Phase 1**: Registry + basic sklearn/LightGBM support
2. **Phase 2**: ONNX export/import for framework swap
3. **Phase 3**: Ensemble support
4. **Phase 4**: Advanced feature engineering

### 8.3 Files to Create

```
tradsl/
├── __init__.py                  # Add MLFunction, LGBMFunction, ONNXFunction exports
├── functions.py                 # Add MLFunction class at end
├── ml/
│   ├── __init__.py
│   ├── registry.py            # ModelRegistry class
│   ├── features.py          # FeatureBuilder class
│   └── models/
│       ├── __init__.py
│       ├── lightgbm.py     # LGBMFunction
│       ├── onnx.py        # ONNXFunction
│       └── ensemble.py     # EnsembleFunction
```

---

## 9. Appendix: Security Considerations

### What We AVOID
- ❌ `pickle.load()` from untrusted sources
- ❌ Running arbitrary code from model files
- ❌ Storing full model with code in database

### What We USE
- ✅ `joblib` for internal sklearn models (trusted environment)
- ✅ `safetensors` for PyTorch (secure tensor storage)
- ✅ `ONNX` for cross-framework distribution
- ✅ File hash verification before loading

### Model Validation

```python
def load_verified(registry: ModelRegistry, model_name: str, version: str = "latest"):
    """Load model with integrity verification."""
    entry = registry.index[model_name][version]
    
    # Verify hash
    with open(entry['artifact_path'], 'rb') as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    
    if actual_hash != entry['artifact_hash']:
        raise ValueError(f"Model integrity check failed for {model_name}/{version}")
    
    return registry.load(model_name, version)
```

---

*Document Version: 1.0*
*Last Updated: 2024-01-15*
