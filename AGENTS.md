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

# Run with coverage (if installed)
python -m pytest tests/ --cov=tradsl --cov-report=term-missing

# Run integration tests only
python -m pytest -m integration

# Skip integration tests
python -m pytest -m "not integration"
```

### Installation

```bash
# Install in development mode
pip install -e .

# Install with all dependencies
pip install -e ".[dev]"
```

### Type Checking (Optional)

```bash
# If mypy is configured
python -m mypy tradsl/
```

## Code Style Guidelines

### General Principles

- Write clean, readable code with clear intent
- Keep functions focused on a single responsibility
- Use descriptive names for variables, functions, and classes
- Add docstrings to public APIs

### Imports

Order imports as follows (per PEP 8):

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
from dataclasses import dataclass
from typing import Optional

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
- Handle exceptions at appropriate boundaries

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

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """Short description of what the function does.

    Longer description if needed, explaining the behavior,
    side effects, or important implementation details.

    Args:
        param1: Description of first parameter
        param2: Description of second parameter

    Returns:
        Description of return value

    Raises:
        ValueError: When param2 is negative

    Example:
        >>> result = function_name("test", 5)
        >>> print(result)
        True
    """
```

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

### Test Naming

- Test method names should describe the behavior being tested
- Format: `test_<what_is_being_tested>`

### Fixtures

Use conftest.py fixtures for shared setup:

```python
# tests/conftest.py
import pytest
from tradsl import clear_registry

@pytest.fixture(autouse=True)
def clean_registry():
    """Clean registry before each test."""
    clear_registry()
    yield
    clear_registry()
```

## Project Structure

```
tradsl/
├── __init__.py          # Public API exports
├── models.py            # Abstract base classes
├── models_impl.py       # Model implementations
├── agent_framework.py   # RL agent framework
├── adapters.py         # Data adapters
├── yf_adapter.py       # Yahoo Finance adapter
├── training.py         # Training utilities
├── backtest.py         # Backtesting engine
├── exceptions.py       # Custom exceptions
├── schema.py           # Validation schema
├── parser.py           # DSL parser
└── ...
```

## Development Workflow

1. **Write tests first** - Follow TDD when adding new features
2. **Run tests frequently** - Use single test for fast iteration
3. **Verify all tests pass** before committing
4. **Add integration tests** for external dependencies (mark with `@pytest.mark.integration`)
