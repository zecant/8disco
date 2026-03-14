# TradSL

A domain-specific language and execution engine for trading systems.

## What is TradSL?

TradSL lets you define trading strategies using a declarative DSL rather than imperative Python. This means:

- **LLM-friendly**: Write strategies in a constrained, validated format
- **Backtest-ready**: Built-in training, walk-forward testing, and bootstrap validation
- **Reproducible**: Deterministic execution with seed control
- **Extensible**: Register custom adapters, functions, models, and reward functions

## Quick Start

```python
import tradsl
from tradsl import train, test

dsl = """
# Data source
yfinance:
type=adapter
class=tradsl.adapters.YFinanceAdapter
interval=1d

# Price data
spy:
type=timeseries
adapter=yfinance
parameters=[SPY]
tradable=true

# Features
returns:
type=timeseries
function=log_returns
inputs=[spy]

sma_20:
type=timeseries
function=sma
inputs=[spy]
params=sma_cfg

sma_cfg:
window=20

# Backtest config
_backtest:
type=backtest
start=2020-01-01
end=2024-12-31
test_start=2024-01-01
capital=100000
training_mode=random_blocks
block_size_min=30
block_size_max=120
n_training_blocks=40
seed=42
"""

# Train your strategy
result = train(dsl, checkpoint_dir="./checkpoints")

# Backtest on held-out data
test_result = test(dsl, result.checkpoint_path)
```

## Key Features

### Feature Functions
Built-in: `sma`, `ema`, `rsi`, `log_returns`, `volatility`, `roc`, `zscore`
Cross-asset: `spread`, `ratio`, `rolling_correlation`, `beta`

### Position Sizers
- `FixedSizer` - Fixed unit count
- `FractionalSizer` - Fixed percentage of portfolio
- `KellySizer` - Kelly criterion
- `VolatilityTargetingSizer` - Constant volatility targeting
- `MaxDrawdownSizer` - Drawdown-based sizing

### Reward Functions
- `SimplePnLReward` - Raw P&L
- `AsymmetricHighWaterMarkReward` - Prospect theory inspired
- `DifferentialSharpeReward` - Online Sharpe ratio

### Training
- Randomized block training
- Walk-forward testing
- Bootstrap confidence intervals

## Installation

```bash
pip install -e .
```

## Extension Points

### Custom Adapter

```python
from tradsl import BaseAdapter, register_adapter

class MyAdapter(BaseAdapter):
    def load_historical(self, symbol, start, end, frequency):
        # Fetch data from your source
        return dataframe
    
    def supports_frequency(self, frequency):
        return frequency in ['1d', '1h']
    
    def max_lookback(self, frequency):
        return None  # Unlimited

register_adapter("my_adapter", MyAdapter)
```

### Custom Function

```python
from tradsl import register_function
from tradsl.functions import FeatureCategory

def my_signal(arr, period=20, **params):
    # Your computation
    return float(value)

register_function(
    name="my_signal",
    func=my_signal,
    category=FeatureCategory.SAFE,
    description="My custom signal",
    min_lookback=period
)
```

## Architecture

- **Parser**: DSL → Python dict
- **Validator**: Schema validation with cross-node checks
- **Resolver**: String names → Python objects
- **DAG**: Topological execution order
- **Engine**: Incremental computation via circular buffers

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_functions.py -v

# Skip slow tests
pytest -m "not slow"
```

## License

MIT
