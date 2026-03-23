# TradSL Abstractions Documentation

**TradSL** (Trading Strategy Domain-Specific Language) is a Python library that provides a declarative configuration language for building and backtesting trading strategies. It compiles to an executable Directed Acyclic Graph (DAG) that processes time-series data through a pipeline of transformations, ML models, and portfolio management functions.

- **Version**: 0.0.3
- **Python**: >=3.8
- **Repository**: https://github.com/zecant/8disco

## Project Structure

```
tradsl/
├── __init__.py           # Package entry point with public API
├── dag.py                # DAG construction, validation, execution
├── parser.py             # DSL parser
├── functions.py          # Function abstract base class
├── adapters.py           # Adapter base class and YFinanceAdapter
├── pricetransforms.py    # EMA, PairwiseCorrelation functions
├── mlfunctions.py        # MLFunction, Regressor, Classifier, Agent
├── circular_buffer.py     # Fixed-size circular buffer
├── portfolio_state.py     # Portfolio state data class
├── portfolio_adapter.py   # Portfolio data adapter
├── portfolio_function.py  # Portfolio execution function
├── sizing.py             # Position sizing functions
├── execution.py          # Execution price models
├── backtest.py           # Backtesting engine
├── exceptions.py         # Custom exceptions
├── ml/                   # ML subpackage
│   ├── __init__.py
│   ├── agents.py         # DummyAgent, TabularQAgent
│   ├── classifiers.py     # RandomForest, GradientBoosting, SVM, KNN
│   └── regressors.py      # RandomForest, GradientBoosting, Linear, SVR, KNN
└── portfolio/             # Portfolio subpackage
    └── __init__.py
```

---

## Core Abstractions

### 1. Node

Represents a single node in the DAG. Two types exist:

| Type | Description |
|------|-------------|
| `timeseries` | Data source node (pulls from external adapter) |
| `function` | Transformation node (processes inputs) |

**Attributes**:
- `name`: Unique identifier for the node
- `type`: Either `"timeseries"` or `"function"`
- `attrs`: All parsed key-value pairs from DSL config
- `inputs`: List of node names this depends on (empty for timeseries)
- `function`: Function name string (for type=function)
- `adapter`: Adapter name string (for type=timeseries)
- `window`: Lookback window size (default 1)
- `buffer_size`: Computed buffer size after build
- `ready`: Boolean tracking if node has enough data

**Source**: `tradsl/dag.py`

---

### 2. DAG

The core execution engine managing the entire pipeline.

**Key Methods**:

| Method | Description |
|--------|-------------|
| `from_config(config)` | Create DAG from parsed DSL config |
| `validate()` | Validate configuration constraints |
| `detect_cycles()` | Detect circular dependencies using DFS |
| `topological_sort()` | Compute execution order using Kahn's algorithm |
| `compute_buffer_sizes()` | Calculate buffer requirements via reverse propagation |
| `resolve(registry)` | Resolve function/adapter names to Python objects |
| `step()` | Execute one tick (advance all nodes by one time step) |
| `build()` | Full pipeline: validate → detect cycles → sort → compute buffers |
| `values()` | Get current values from all nodes |

**Builder Pattern**: DAG uses method chaining:
```python
dag = DAG.from_config(config).build()
```

**Source**: `tradsl/dag.py`

---

### 3. Function

Abstract base class for all transformation functions.

```python
class Function(ABC):
    @abstractmethod
    def apply(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Apply function to DataFrame"""
        pass
```

**Contract**:
- Input: DataFrame with columns from upstream input nodes
- Output: DataFrame with computed results, or `None` if insufficient data

**Source**: `tradsl/functions.py`

---

### 4. Adapter

Abstract base class for data sources.

```python
class Adapter(ABC):
    @abstractmethod
    def set_start(self, start_time: datetime) -> None:
        """Initialize adapter with start time"""
    
    @abstractmethod
    def tick(self) -> Optional[pd.DataFrame]:
        """Return current data point, or None if exhausted"""
```

**Concrete Implementations**:
- `YFinanceAdapter`: Fetches OHLCV data from Yahoo Finance

**Source**: `tradsl/adapters.py`

---

### 5. MLFunction

Abstract base for ML-powered functions with warmup support. Extends `Function` with:

```python
class MLFunction(Function):
    def __init__(self, warmup: int = ...):
        self.warmup = warmup
    
    def _train(self, data: pd.DataFrame) -> None:
        """Train ML model on historical data"""
    
    def _predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Make prediction using trained model"""
    
    def apply(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        self._train(data)  # Retrain each step
        return self._predict(data) if self._is_warmed_up() else None
```

**Subclasses**:

| Class | Purpose |
|-------|---------|
| `Regressor` | Regression tasks (predicts continuous value) |
| `Classifier` | Classification tasks (predicts class/label) |
| `Agent` | RL/agent tasks (outputs action, confidence, asset) |

**ML Regressors** (`tradsl/ml/regressors.py`):
- `RandomForestRegressor`
- `GradientBoostingRegressor`
- `LinearRegressor`
- `SVRRegressor`
- `KNNRegressor`

**ML Classifiers** (`tradsl/ml/classifiers.py`):
- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `SVMClassifier`
- `KNNClassifier`

**Agents** (`tradsl/ml/agents.py`):
- `DummyAgent`: Baseline agent
- `TabularQAgent`: Q-learning agent

**Source**: `tradsl/mlfunctions.py`

---

### 6. CircularBuffer

Fixed-size circular buffer for time-series values with O(1) push.

```python
class CircularBuffer:
    def __init__(self, size: int):
        self._data = np.empty(size, dtype=object)
        self._head = 0  # Next write position
    
    @property
    def size(self) -> int: ...
    @property
    def count(self) -> int: ...
    @property
    def is_ready(self) -> bool: ...  # True when full
    
    def push(self, value: Any) -> None: ...
    def latest(self) -> Any: ...      # O(1)
    def contents(self) -> list: ...   # O(window)
    def __getitem__(self, index: int) -> Any: ...  # Random access
```

**Performance**:
| Operation | Complexity |
|-----------|------------|
| Push | O(1) amortized |
| Latest | O(1) |
| Contents | O(window) |

**Source**: `tradsl/circular_buffer.py`

---

### 7. PortfolioState

Immutable-ish state container for portfolio holdings and cash.

```python
@dataclass
class PortfolioState:
    cash: float
    currency: str = "USD"
    holdings: dict[str, int] = field(default_factory=dict)
```

**Source**: `tradsl/portfolio_state.py`

---

### 8. SizingFunction

Abstract base for position sizing strategies.

```python
class SizingFunction(ABC):
    @abstractmethod
    def compute(
        self,
        agent_output: pd.DataFrame,
        portfolio_state: PortfolioState,
        price: float,
    ) -> int:
        """Return number of shares (positive=buy, negative=sell, 0=hold)"""
```

**Concrete Implementations**:
- `FractionalSizing`: Size as fraction of portfolio NAV

**Source**: `tradsl/sizing.py`

---

### 9. ExecutionModel

Abstract base for execution price models.

```python
class ExecutionModel(ABC):
    @abstractmethod
    def calculate(
        self,
        sizing_output: pd.DataFrame,
        price_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return sizing output with execution_price, execution_cost columns"""
```

**Concrete Implementations**:
- `OhlcAvgExecution`: Midpoint of open/close prices

**Source**: `tradsl/execution.py`

---

### 10. Backtester

Backtesting engine that runs the DAG over a date range.

```python
class Backtester:
    def __init__(
        dag: DAG,
        start_date: datetime,
        end_date: datetime,
        tick_speed: TickSpeed = TickSpeed.ONE_DAY,
    )
    def run() -> pd.DataFrame: ...
```

**TickSpeed Options**:
- `ONE_MINUTE`
- `FIVE_MINUTES`
- `FIFTEEN_MINUTES`
- `ONE_HOUR`
- `ONE_DAY`

**Source**: `tradsl/backtest.py`

---

## Design Patterns

### Registry Pattern

Functions and adapters are resolved from a registry at execution time:

```python
default_registry = {
    "pricetransforms": pricetransforms,
    "ml": ml,
    "portfolio": portfolio,
    "yfinance": YFinanceAdapter,
    "portfolioadapter": PortfolioAdapter,
}
```

**Resolution Logic**:
- Simple names: `"sma"` → `registry["sma"]`
- Dotted names: `"pricetransforms.ema"` → traverses `registry["pricetransforms"]` then gets `.ema`

---

### Template Method Pattern

Abstract base classes define the interface:

```
Function.apply()        → Subclasses implement transformation
Adapter.tick()          → Subclasses implement data fetching
SizingFunction.compute() → Subclasses implement sizing logic
ExecutionModel.calculate() → Subclasses implement execution pricing
```

---

### Strategy Pattern

Interchangeable algorithms:
- Sizing functions (e.g., `FractionalSizing`)
- Execution models (e.g., `OhlcAvgExecution`)

---

## Data Flow

```
DSL String
    ↓
parse() → Config Dict
    ↓
DAG.from_config() → Node objects
    ↓
build() → Validate → Detect Cycles → Topological Sort → Buffer Sizes
    ↓
resolve(registry) → Instantiate Functions/Adapters → Create Buffers
    ↓
step() → For each node in execution_order:
          - Timeseries: adapter.tick() → push to buffer
          - Function: glue_inputs() → fn.apply() → push to buffer
    ↓
values() → [(node_name, latest_value), ...]
```

### Input Gluing (`_glue_inputs`)

Joins multiple input buffers into a single DataFrame:

1. Get contents from each input buffer
2. Prefix column names with source node name
3. Merge on DatetimeIndex using `pd.merge_asof`
4. Return aligned DataFrame

---

## Buffer Size Computation

Algorithm for calculating required buffer sizes:

```
1. Initialize buffer_map: all nodes = 0
2. Mark sink node (last in execution order) = 1
3. For each node in reverse execution order:
   For each dependency:
     buffer_map[dep] = max(buffer_map[dep], node.window)
4. Set each node.buffer_size from buffer_map
```

This ensures nodes have enough historical data for their window requirements.

---

## DSL Syntax

```yaml
# Block definition (node)
block_name:
    type=timeseries|function
    adapter=adapter_name        # for timeseries
    function=function_name      # for function
    inputs=[input1, input2]     # for function
    window=20                   # optional, default 1
    
# Example
price:
    type=timeseries
    adapter=yfinance
    symbol=AAPL

sma:
    type=function
    function=pricetransforms.ema
    inputs=[price]
    window=20
```

### Parser Value Types

| Type | Examples |
|------|----------|
| Lists | `[1, 2, 3]`, `["a", "b"]` |
| Quoted strings | `"hello"`, `'world'` |
| Booleans | `true`, `false` |
| None | `none` |
| Integers | `42` |
| Floats | `3.14`, `1e10` |
| Unquoted strings | `identifier` |

---

## Validation Rules

### Timeseries Nodes
- Cannot have `function` attribute
- Cannot have `inputs` attribute
- Must have `adapter` attribute

### Function Nodes
- Must have `function` attribute
- Must have non-empty `inputs` attribute
- All input references must exist in DAG

### Agent Invariant (optional)
- Second-to-last node must be `ml.agents.*`
- Last node must be `portfolio.*`

---

## Exception Hierarchy

```
TradSLException (base)
├── ParseError: DSL syntax errors with line number
├── CycleError: Circular dependency with path
├── ConfigError: Invalid node configuration
├── ResolutionError: Cannot resolve function/adapter name
└── InvariantError: Required invariant violated
```

**Source**: `tradsl/exceptions.py`

---

## Usage Example

```python
from tradsl import DAG, parse, default_registry

dsl = '''
price:
    type=timeseries
    adapter=yfinance
    symbol=AAPL

sma_fast:
    type=function
    function=pricetransforms.ema
    inputs=[price]
    window=5

sma_slow:
    type=function
    function=pricetransforms.ema
    inputs=[price]
    window=20
'''

config = parse(dsl)
dag = DAG.from_config(config)
dag.build()
dag.resolve(default_registry)

from tradsl.backtest import Backtester, TickSpeed
from datetime import datetime

backtester = Backtester(
    dag=dag,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 3, 1),
    tick_speed=TickSpeed.ONE_DAY
)
results = backtester.run()
```

---

## Performance Characteristics

| Operation | Complexity |
|-----------|------------|
| Buffer push | O(1) amortized |
| Buffer latest | O(1) |
| Buffer contents | O(window) |
| Topological sort | O(V + E) |
| Cycle detection | O(V + E) |
| Buffer size computation | O(V + E) |

---

## Key Interfaces Summary

### Function Contract
```python
def apply(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
    # data: DataFrame with columns from input nodes
    # Returns: DataFrame with computed results, or None if not enough data
```

### Adapter Contract
```python
def set_start(self, start_time: datetime) -> None: ...
def tick(self) -> Optional[pd.DataFrame]:
    # Returns: DataFrame with current data point, or None if exhausted
```

### Sizing Function Contract
```python
def compute(
    self,
    agent_output: pd.DataFrame,
    portfolio_state: PortfolioState,
    price: float,
) -> int:
    # Returns: Number of shares (positive=buy, negative=sell, 0=hold)
```

### Execution Model Contract
```python
def calculate(
    self,
    sizing_output: pd.DataFrame,
    price_data: pd.DataFrame,
) -> pd.DataFrame:
    # Returns: Sizing output with execution_price, execution_cost columns
```
