# tradsl

A powerful domain-specific configuration language for trading systems. Define your trading strategy, models, data adapters, and backtest configuration in a single declarative DSL.

## Features

- **Declarative DSL** - Define timeseries, models, agents, and parameters in a simple text format
- **Parameter Blocks** - Reusable parameter sets (`mlparams:`, `windowparams:`) that can be referenced across your config
- **Data Adapters** - Plug in any data source adapter via class path resolution
- **Submodels** - Use models as inputs to other models for ensemble/hierarchical strategies
- **DAG Execution** - Automatic topological ordering with cycle detection
- **Type Safety** - Schema validation for all config blocks

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import tradsl

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

:signal_model
type=model
class=RandomForest
inputs=[nvda, vix, nvda_ma30]
params=mlparams

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
def rolling_mean(data, window=30):
    pass

class RandomForest:
    pass

def kelly_sizer(signals, tradable):
    pass

# Parse and resolve
config = tradsl.parse(config_str, context={
    'rolling_mean': rolling_mean,
    'RandomForest': RandomForest,
    'kelly_sizer': kelly_sizer,
})

# Access resolved config
agent_config = config['agent']
model_config = config['signal_model']

# Execution order (topological sort)
print(config['_execution_order'])
# ['vix', 'nvda', 'nvda_ma30', 'signal_model', 'agent']
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

## Development

```bash
# Run tests
python -m pytest tradsl/ -v

# Run specific test file
python -m pytest tradsl/test_parser.py -v
```

## License

MIT
