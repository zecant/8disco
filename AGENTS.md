# AGENTS.md - TradSL Development Guide

This file provides guidelines for agentic coding agents working on the TradSL codebase.

## Build, Lint, and Test Commands

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run a single test file
python -m pytest tests/test_models_impl.py

# Run a single test
python -m pytest tests/test_models_impl.py::TestRandomForestModel::test_fit_and_predict_classifier

# Run tests matching a pattern
python -m pytest -k "test_init"

# Run with verbose output
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=tradsl --cov-report=term-missing

# Run/skip integration tests
python -m pytest -m integration
python -m pytest -m "not integration"

# Skip slow tests
python -m pytest -m "not slow"
```

### Installation

```bash
pip install -e .          # Install in development mode
pip install -e ".[dev]"   # Install with all dependencies
```

## Code Style Guidelines

### Imports (PEP 8)

```python
# Standard library
import os
import sys
from typing import Optional, Dict, Any, Type, List
from dataclasses import dataclass

# Third-party packages
import numpy as np
import pandas as pd
import pytest
import joblib

# Local application imports
from tradsl.models import BaseTrainableModel, TradingAction
from tradsl.exceptions import TradSLError, ConfigError
```

### Type Hints

Use type hints for all function signatures:

```python
def process_data(
    features: np.ndarray,
    labels: Optional[np.ndarray] = None,
    params: Dict[str, Any] = None
) -> float:
    """Process features and return a score.

    Args:
        features: Input feature array of shape (n_samples, n_features)
        labels: Optional target labels
        params: Configuration parameters

    Returns:
        Processed score as a float

    Raises:
        ValueError: If features are invalid
    """
    ...
```

### Naming Conventions

- **Classes**: `CamelCase` (e.g., `RandomForestModel`, `ParameterAgent`)
- **Functions/methods**: `snake_case` (e.g., `fit_model`, `load_historical`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_ITERATIONS`, `DEFAULT_TIMEOUT`)
- **Private methods**: Prefix with underscore (e.g., `_internal_method`)
- **Variables**: `snake_case` (e.g., `feature_array`, `model_config`)

### Dataclasses

Use dataclasses for simple data containers:

```python
@dataclass
class TrainResult:
    """Result of training pass."""
    checkpoint_path: str
    config: Dict[str, Any]
    n_blocks: int
    final_metrics: Optional[PerformanceMetrics] = None
```

### Error Handling

- Use custom exceptions from `tradsl.exceptions`
- Raise specific exceptions with clear messages
- Chain exceptions with `from e`

```python
from tradsl.exceptions import AdapterError, ModelError

def load_data(self, symbol: str) -> pd.DataFrame:
    if not symbol:
        raise AdapterError("Symbol cannot be empty")
    
    try:
        data = self._fetch_data(symbol)
    except Exception as e:
        raise AdapterError(f"Failed to load {symbol}: {e}") from e
```

### Docstrings

Use Google-style docstrings for all public APIs.

## Testing Conventions

### Test Structure

Use class-based test organization:

```python
import pytest
import numpy as np
from tradsl.models_impl import RandomForestModel

class TestRandomForestModel:
    def test_init_default(self):
        model = RandomForestModel()
        assert model.model_type == "classifier"
    
    def test_fit_and_predict(self):
        model = RandomForestModel(n_estimators=10, random_state=42)
        features = np.random.randn(100, 5)
        labels = np.random.choice([0, 1], size=100)
        
        model.fit(features, labels)
        
        assert model.is_fitted is True
        result = model.predict(features[0])
        assert isinstance(result, float)
```

### Test Fixtures

The test suite uses an autouse fixture that cleans the model registry:

```python
# tests/conftest.py
@pytest.fixture(autouse=True)
def clean_registry():
    """Clean registry before each test."""
    from tradsl import clear_registry
    clear_registry()
    yield
    clear_registry()
```

### Test Markers

- `@pytest.mark.integration` - Tests requiring external dependencies
- `@pytest.mark.slow` - Tests that take significant time to run

## Project Structure

```
tradsl/
├── __init__.py          # Public API exports
├── models.py            # Abstract base classes
├── models_impl.py       # Model implementations
├── agent_framework.py   # RL agent framework
├── adapters.py          # Data adapters
├── yf_adapter.py        # Yahoo Finance adapter
├── training.py          # Training utilities
├── backtest.py          # Backtesting engine
├── exceptions.py        # Custom exceptions
├── schema.py            # Validation schema
├── parser.py            # DSL parser
└── ...
```

## Development Workflow

1. **Write tests first** - Follow TDD when adding new features
2. **Run tests frequently** - Use single test for fast iteration
3. **Verify all tests pass** before committing
4. **Add integration tests** for external dependencies (mark with `@pytest.mark.integration`)
