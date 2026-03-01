# tradsl

A powerful domain-specific configuration language for trading systems. Define your trading strategy, models, data adapters, and backtest configuration in a single declarative DSL with built-in feature computation and backtesting.

## Features

- **Declarative DSL** - Define timeseries, models, agents, adapters, and parameters in a simple text format
- **Parameter Blocks** - Reusable parameter sets (`mlparams:`, `windowparams:`) that can be referenced across your config
- **Data Adapters** - Plug in any data source adapter via class path resolution
- **Submodels** - Use models as inputs to other models for ensemble/hierarchical strategies
- **DAG Execution** - Automatic topological ordering with cycle detection
- **Type Safety** - Schema validation for all config blocks
- **Feature Engine** - Compute features via DAG with support for timeseries functions and model predictions
- **Training Scheduler** - Rolling/expanding window training with configurable retrain schedules
- **Backtest Interpreter** - Run full backtests with event-driven feature computation
- **Violent Failure** - Explicit errors for invalid sizer output, model failures, and data issues

## Installation

```bash
pip install tradsl
```

Requires: `pandas`, `numpy`

## Quick Start

```python
import tradsl
from tradsl.utils import TradslInterpreter, load_timeseries, compute_features
from datetime import datetime

# Define your strategy
config_str = """
mlparams:
  lr=0.001
  epochs=100

:yfinance
type=adapter
class=adapters.YFAdapter

:nvda
type=timeseries
adapter=yfinance
parameters=["nvda"]

:vix
type=timeseries
adapter=yfinance
parameters=["^VIX"]

:nvda_ma30
type=timeseries
function=rolling_mean
inputs=[nvda]
params=mlparams

:signal_model
type=model
class=RandomForest
inputs=[nvda, vix, nvda_ma30]
params=mlparams
dotraining=true
retrain_schedule=weekly
training_window=rolling
training_window_size=500

:agent
type=agent
inputs=[signal_model, vix]
tradable=[nvda]
sizer=kelly_sizer

:backtest
type=backtest
start=2020-01-01
end=2024-01-01
capital=100000
"""

# Define your Python functions/classes
def rolling_mean(data, **kwargs):
    window = kwargs.get('window', 30)
    return data.rolling(window=window).mean()

class RandomForest:
    def __init__(self, lr=0.001, epochs=100, **kwargs):
        self.lr = lr
        self.epochs = epochs
    
    def train(self, X, y, **kwargs):
        # Training logic here
        pass
    
    def predict(self, X, **kwargs):
        # Return dict of output_name -> values
        return {'allocation': [0.5] * len(X)}

def kelly_sizer(signals, tradable):
    """Must return dict mapping each tradable symbol to allocation weight"""
    return {sym: 1.0 / len(tradable) for sym in tradable}

# Parse and resolve
config = tradsl.parse(config_str, context={
    'rolling_mean': rolling_mean,
    'RandomForest': RandomForest,
    'kelly_sizer': kelly_sizer,
    'adapters.YFAdapter': YourAdapterClass,
})

# Run backtest
interpreter = TradslInterpreter(config)
results = interpreter.run_backtest(
    start=datetime(2020, 1, 1),
    end=datetime(2024, 1, 1),
    frequency='1min'
)
```

## DSL Syntax

### Parameter Blocks

```
mlparams:
  lr=0.001
  epochs=100

windowparams:
  window=30
  min_periods=10
```

### Adapters

```
:yfinance
type=adapter
class=adapters.YFAdapter
```

### Timeseries

```
:nvda
type=timeseries
adapter=yfinance
parameters=["nvda"]
```

Or derived timeseries:

```
:nvda_ma30
type=timeseries
function=rolling_mean
inputs=[nvda]
params=windowparams
```

### Models

```
:signal_model
type=model
class=RandomForest
inputs=[nvda, vix]
params=mlparams
dotraining=true
retrain_schedule=weekly
training_window=rolling
training_window_size=500
load_from=./models/signal_model.pkl
```

### Submodels (Model Ensemble)

```
:ensemble
type=model
class=EnsembleModel
inputs=[signal_model, another_model]
```

### Agent

```
:agent
type=agent
inputs=[signal_model, vix]
tradable=[nvda]
sizer=kelly_sizer
```

### Backtest

```
:backtest
type=backtest
start=2020-01-01
end=2024-01-01
capital=100000
```

## Output Structure

The `parse()` function returns a dict with:

- **Config blocks** - Your timeseries, models, agent with resolved callables
- **`_params`** - Parameter blocks dict
- **`_adapters`** - Instantiated adapter objects
- **`_backtest`** - Backtest configuration
- **`_execution_order`** - Topological sort of nodes
- **`_graph`** - Dependency graph (deps, reverse_deps)

## Utilities

### load_timeseries(config, start, end, frequency)

Load historical data from configured adapters:

```python
from tradsl.utils import load_timeseries
from datetime import datetime

df = load_timeseries(config, datetime(2020, 1, 1), datetime(2024, 1, 1))
# Returns DataFrame with columns like nvda_close, vix_volume, etc.
```

### compute_features(config, data)

Compute all features via DAG:

```python
from tradsl.utils import compute_features

features = compute_features(config, raw_data)
# Adds computed columns for timeseries functions and model predictions
```

### TradslInterpreter

Full backtest runner:

```python
from tradsl.utils import TradslInterpreter

interpreter = TradslInterpreter(config)
interpreter.load_data(start, end)
interpreter.compute_initial_features()
interpreter.train_models()

results = interpreter.run_backtest(start, end, frequency='1min')
```

## Context Resolution

When calling `tradsl.parse()`, pass a `context` dict with:

- **Functions**: `rolling_mean`, `kelly_sizer`, etc.
- **Model classes**: `RandomForest`, `LSTM`, etc.
- **Adapter classes**: `adapters.YFAdapter` (full class path as key)

```python
config = tradsl.parse(source, context={
    'my_function': my_function,
    'adapters.YFAdapter': YFAdapterClass,
})
```

## Violent Failure

The system uses **violent failure** - explicit errors are raised for:

- **Sizer output**: Wrong keys, None, NaN/Inf, negative values, non-dict returns
- **Model prediction**: Failures or wrong return types
- **Training**: Instantiation failures, training errors
- **Data loading**: Adapter failures or empty data

This prevents silent failures and makes debugging explicit.

## Testing

```bash
# Run all tests
python -m pytest tradsl/ -v

# Run specific test file
python -m pytest tradsl/test_parser.py -v
```

## License

MIT
