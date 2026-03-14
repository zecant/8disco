"""
TradSL - Trading Strategy DSL

A domain-specific configuration language for trading systems.
Parses DSL, validates, resolves, builds DAG, and provides extension points.
"""
from typing import Any, Callable, Dict, Optional, Type
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger("tradsl")

def set_log_level(level: int) -> None:
    """Set the logging level for tradsl.
    
    Args:
        level: logging.DEBUG, logging.INFO, logging.WARNING, etc.
    """
    logger.setLevel(level)
    for mod in ['tradsl.adapters', 'tradsl.backtest', 'tradsl.agent_framework', 'tradsl.training']:
        logging.getLogger(mod).setLevel(level)

from tradsl.parser import parse
from tradsl.schema import validate
from tradsl.dag import build_dag, resolve, DAG, DAGMetadata
from tradsl.exceptions import (
    TradSLError, ParseError, ConfigError, ResolutionError, CycleError,
    AdapterError, FeatureError, ModelError, ExecutionError, ValidationError
)
from tradsl.adapters import BaseAdapter
from tradsl.models import (
    BaseTrainableModel, BaseAgentArchitecture, TradingAction
)
from tradsl.models_impl import (
    SklearnModel, RandomForestModel, LinearModel,
    DecisionTreeModel, GradientBoostingModel
)
from tradsl.agent_framework import ParameterizedAgent, TabularAgent, PPOUpdate, DQNUpdate
from tradsl.sizers import (
    BasePositionSizer, FractionalSizer, FixedSizer, KellySizer,
    VolatilityTargetingSizer, MaxDrawdownSizer, EnsembleSizer
)
from tradsl.rewards import (
    BaseRewardFunction, SimplePnLReward, AsymmetricHighWaterMarkReward,
    DifferentialSharpeReward, RewardContext
)
from tradsl.circular_buffer import CircularBuffer
from tradsl.training import BlockSampler, BlockTrainer, WalkForwardTester, ReplayBuffer, TrainingConfig
from tradsl.bootstrap import IIDBootstrap, BlockBootstrap, BootstrapEnsemble
from tradsl.backtest import BacktestEngine, TransactionCosts, ExecutionMode
from tradsl.statistics import PerformanceMetrics, compute_metrics
from tradsl.yf_adapter import YFinanceAdapter
from tradsl.testdata_adapter import TestDataAdapter, create_test_adapter
from tradsl.sparse import SparseTimeSeries, SparseBuffer
from tradsl.nt_integration import (
    TradSLStrategy, TradSLNTStrategy, Portfolio, Position
)


_registry: Dict[str, Dict[str, Any]] = {
    'adapters': {},
    'functions': {},
    'trainable_models': {},
    'label_functions': {},
    'agents': {},
    'sizers': {},
    'rewards': {},
}

# Auto-register built-in label functions
from tradsl.labels import LABEL_REGISTRY
for name, spec in LABEL_REGISTRY.items():
    _registry['label_functions'][name] = {
        'func': spec.func,
        'description': spec.description,
        'requires_future_bars': spec.requires_future_bars
    }

# Auto-register built-in adapters
from tradsl.adapters import BaseAdapter
from tradsl.yf_adapter import YFinanceAdapter
from tradsl.testdata_adapter import TestDataAdapter
_registry['adapters']['tradsl.adapters.YFinanceAdapter'] = YFinanceAdapter
_registry['adapters']['tradsl.testdata_adapter.TestDataAdapter'] = TestDataAdapter
_registry['adapters']['yf'] = YFinanceAdapter
_registry['adapters']['test_adapter'] = TestDataAdapter

# Auto-register built-in sizers
_registry['sizers']['fractional'] = FractionalSizer
_registry['sizers']['fixed'] = FixedSizer
_registry['sizers']['kelly'] = KellySizer
_registry['sizers']['volatility_targeting'] = VolatilityTargetingSizer
_registry['sizers']['max_drawdown'] = MaxDrawdownSizer
_registry['sizers']['ensemble'] = EnsembleSizer


@dataclass
class ValidationResult:
    """Result of validation pass."""
    config: Dict[str, Any]
    dag: Optional[DAG] = None
    metadata: Optional[DAGMetadata] = None


@dataclass
class TrainResult:
    """Result of training pass."""
    checkpoint_path: str
    config: Dict[str, Any]
    n_blocks: int
    total_bars: int
    final_metrics: Optional[PerformanceMetrics] = None


@dataclass
class TestResult:
    """Result of walk-forward test pass."""
    checkpoint_path: str
    equity_curve: np.ndarray
    trades: list
    metrics: PerformanceMetrics
    config: Dict[str, Any]


@dataclass
class RunResult:
    """Result of full pipeline (train + test + bootstrap)."""
    train_result: TrainResult
    test_result: TestResult
    bootstrap_result: Optional[Any] = None
    config: Optional[Dict[str, Any]] = None


def parse_dsl(dsl_string: str) -> Dict[str, Any]:
    """
    Parse DSL string into raw Python dict.

    Args:
        dsl_string: Complete DSL content

    Returns:
        Raw parsed dict

    Raises:
        ParseError: On syntax violation
    """
    return parse(dsl_string)


def validate_config(raw: Dict[str, Any], function_registry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Validate parsed DSL config against schema.

    Args:
        raw: Output from parse()
        function_registry: Optional function registry for validation

    Returns:
        Validated config with defaults filled

    Raises:
        ConfigError: With list of all validation errors
    """
    return validate(raw, function_registry)


def resolve_config(config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Resolve string names to Python objects.

    Args:
        config: Validated config
        context: Optional context with registered components

    Returns:
        Config with resolved Python objects
    """
    return resolve(config, context)


def build_execution_dag(config: Dict[str, Any]) -> DAG:
    """
    Build execution DAG from validated config.

    Args:
        config: Validated config

    Returns:
        DAG with computed metadata

    Raises:
        CycleError: If circular dependency detected
    """
    return build_dag(config)


def load(dsl_string: str, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
    """
    Full pipeline: parse → validate → resolve → build_dag

    Args:
        dsl_string: Complete DSL content
        context: Optional context with registered components

    Returns:
        ValidationResult with config, dag, and metadata

    Raises:
        ParseError: On syntax violation
        ConfigError: On validation failure
        CycleError: On circular dependency
    """
    raw = parse(dsl_string)
    validated = validate(raw)
    resolved = resolve(validated, context)
    dag = build_dag(resolved)

    return ValidationResult(
        config=resolved,
        dag=dag,
        metadata=dag.metadata
    )


def _load_data_from_adapters(
    resolved: Dict[str, Any],
    backtest_config: Dict[str, Any]
) -> Optional[tuple]:
    """
    Automatically load data from adapters defined in DSL.
    
    Args:
        resolved: Resolved configuration with adapter instances
        backtest_config: Backtest configuration with date range
    
    Returns:
        Tuple of (data array, dates array) or None if loading fails
    """
    from datetime import datetime
    
    start = backtest_config.get('start', '2024-01-01')
    end = backtest_config.get('end', '2024-12-31')
    interval = '1d'
    
    try:
        start_date = datetime.strptime(start, '%Y-%m-%d')
        end_date = datetime.strptime(end, '%Y-%m-%d')
    except (ValueError, TypeError):
        return None
    
    timeseries_blocks = {}
    for name, block in resolved.items():
        if isinstance(block, dict) and block.get('type') == 'timeseries':
            adapter_class = block.get('_adapter')
            if adapter_class is not None:
                params = block.get('parameters', [])
                if params:
                    timeseries_blocks[name] = {
                        'adapter_class': adapter_class,
                        'symbol': params[0] if params else None,
                        'tradable': block.get('tradable', False),
                        'interval': block.get('interval', '1d')
                    }
    
    if not timeseries_blocks:
        return None
    
    data_arrays = {}
    ohlcv_data = {}
    primary_df = None
    for name, info in timeseries_blocks.items():
        adapter_class = info['adapter_class']
        symbol = info['symbol']
        interval = info.get('interval', '1d')
        
        try:
            if isinstance(adapter_class, type):
                adapter = adapter_class(interval=interval)
            else:
                adapter = adapter_class
            
            df = adapter.load_historical(symbol, start_date, end_date, interval)
            if df is not None and len(df) > 0:
                ohlcv_data[name] = {
                    'open': df['open'].values if 'open' in df.columns else df['close'].values,
                    'high': df['high'].values if 'high' in df.columns else df['close'].values,
                    'low': df['low'].values if 'low' in df.columns else df['close'].values,
                    'close': df['close'].values,
                    'volume': df['volume'].values if 'volume' in df.columns else np.zeros(len(df)),
                }
                data_arrays[name] = df['close'].values
                if info['tradable'] or primary_df is None:
                    primary_df = df
        except Exception as e:
            logger.warning(f"Failed to load {symbol}: {e}")
            continue
    
    if not data_arrays:
        return None
    
    primary_name = None
    for name, info in timeseries_blocks.items():
        if info['tradable'] and name in data_arrays:
            primary_name = name
            break
    
    if primary_name is None:
        primary_name = list(data_arrays.keys())[0]
    
    min_len = min(len(arr) for arr in data_arrays.values())
    
    combined = []
    combined.append(data_arrays[primary_name][:min_len])
    
    for name, arr in data_arrays.items():
        if name != primary_name:
            combined.append(arr[:min_len])
    
    dates = None
    if primary_df is not None:
        dates = primary_df.index[:min_len].values
    
    return np.column_stack(combined), dates


def train(
    dsl_string: str,
    context: Optional[Dict[str, Any]] = None,
    data: Optional[np.ndarray] = None,
    checkpoint_dir: str = "/tmp/tradsl_checkpoints"
) -> TrainResult:
    """
    Full training pipeline: parse → validate → resolve → build_dag → train.

    Args:
        dsl_string: Complete DSL content
        context: Optional context with registered components
        data: Price/feature data array (required)
        checkpoint_dir: Directory to save model checkpoints

    Returns:
        TrainResult with checkpoint path and training metrics

    Raises:
        ParseError: On syntax violation
        ConfigError: On validation failure
        CycleError: On circular dependency
    """
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)

    raw = parse(dsl_string)
    validated = validate(raw)
    resolved = resolve(validated, context)
    dag = build_dag(resolved)

    backtest_config = resolved.get('_backtest', {})
    
    if data is None:
        result = _load_data_from_adapters(resolved, backtest_config)
        if result is None:
            raise ValueError("Could not automatically load data from adapters. Please provide data array.")
        data, dates = result

    training_config = TrainingConfig(
        training_window=backtest_config.get('training_window', 504),
        retrain_schedule=backtest_config.get('retrain_schedule', 'every_n_bars'),
        retrain_n=backtest_config.get('retrain_n', 252),
        n_training_blocks=backtest_config.get('n_training_blocks', 40),
        block_size_min=backtest_config.get('block_size_min', 30),
        block_size_max=backtest_config.get('block_size_max', 120),
        seed=backtest_config.get('seed', 42)
    )

    from tradsl.models_impl import RandomForestModel
    from tradsl.rewards import AsymmetricHighWaterMarkReward
    from tradsl.sizers import KellySizer

    model_config = resolved.get('model', {})
    agent_config = resolved.get('agent', {})
    backtest_config = resolved.get('_backtest', {})

    n_estimators = model_config.get('params', {}).get('n_estimators', 10) if model_config.get('params') else 10
    replay_buffer_size = model_config.get('replay_buffer_size', 4096)
    
    model_class_name = model_config.get('class')
    policy_model_class = None
    if model_class_name:
        policy_model_class = _registry.get('trainable_models', {}).get(model_class_name)
        if policy_model_class is None:
            policy_model_class = _registry.get('agents', {}).get(model_class_name)
    
    if policy_model_class is None:
        from tradsl.models_impl import RandomForestModel
        policy_model_class = RandomForestModel
    
    policy_params = model_config.get('params', {})
    if 'class' not in policy_params:
        policy_params['random_state'] = training_config.seed
    policy = policy_model_class(**policy_params)
    
    # Use retrain_n from model block if available, otherwise from training_config
    agent_update_n = model_config.get('retrain_n', training_config.retrain_n)
    # Also check agent config for update_n (takes priority)
    agent_update_n = agent_config.get('update_n', agent_update_n)
    
    # Use update_schedule from agent config, or fall back to model block, then training_config
    agent_update_schedule = agent_config.get('update_schedule', training_config.retrain_schedule)
    
    agent = ParameterizedAgent(
        policy_model=policy,
        update_schedule=agent_update_schedule,
        update_n=agent_update_n,
        replay_buffer_size=replay_buffer_size,
        seed=training_config.seed
    )

    reward_name = model_config.get('reward_function', 'asymmetric_high_water_mark')
    reward_fn_class = _registry.get('rewards', {}).get(reward_name)
    if reward_fn_class is None:
        reward_fn_class = AsymmetricHighWaterMarkReward
    reward_fn = reward_fn_class()

    sizer_name = agent_config.get('sizer', 'kelly')
    sizer_class = _registry.get('sizers', {}).get(sizer_name)
    if sizer_class is None:
        sizer_class = KellySizer
    
    # Get sizer params from agent config
    sizer_params_name = agent_config.get('sizer_params')
    sizer_params = {}
    if sizer_params_name and sizer_params_name in resolved:
        sizer_params = resolved[sizer_params_name]
    
    # Create sizer with params if provided
    if sizer_params:
        sizer = sizer_class(**sizer_params)
    else:
        sizer = sizer_class()

    trainable_models = {}
    for name, block in resolved.items():
        if isinstance(block, dict) and block.get('type') == 'trainable_model':
            model_class = _registry.get('trainable_models', {}).get(block.get('class'))
            if model_class:
                model_params = block.get('params', {})
                model = model_class(**model_params)
                trainable_models[name] = {
                    'model': model,
                    'label_function': block.get('label_function'),
                    'retrain_schedule': block.get('retrain_schedule', 'every_n_bars'),
                    'training_window': block.get('training_window', 504)
                }

    label_functions = _registry.get('label_functions', {})

    block_sampler = BlockSampler(training_config)
    
    symbol = list(agent_config.get('tradable', ['UNKNOWN']))[0] if agent_config.get('tradable') else 'UNKNOWN'
    
    trainer = BlockTrainer(
        training_config, agent, reward_fn, block_sampler,
        trainable_models=trainable_models,
        label_functions=label_functions,
        capital=backtest_config.get('capital', 100000.0),
        sizer=sizer,
        symbol=symbol
    )

    capital = backtest_config.get('capital', 100000.0)

    source_timeseries = {}
    for name, block in resolved.items():
        if isinstance(block, dict) and block.get('type') == 'timeseries':
            if not block.get('inputs'):
                source_timeseries[name] = block

    ts_to_col = {}
    col_idx = 0
    primary_name = None
    for name, block in source_timeseries.items():
        if block.get('tradable') or primary_name is None:
            ts_to_col[name] = 0
            primary_name = name
            col_idx = 1
        else:
            ts_to_col[name] = col_idx
            col_idx += 1

    for name, block in resolved.items():
        if isinstance(block, dict) and block.get('type') == 'timeseries':
            if name not in ts_to_col:
                ts_to_col[name] = col_idx
                col_idx += 1

    node_buffers: Dict[str, CircularBuffer] = {}
    buffer_sizes = dag.metadata.node_buffer_sizes if dag.metadata else {}

    for name, block in resolved.items():
        if isinstance(block, dict) and block.get('type') == 'timeseries':
            buf_size = buffer_sizes.get(name, 20)
            node_buffers[name] = CircularBuffer(buf_size)

    node_functions: Dict[str, Any] = {}
    for name, block in resolved.items():
        if isinstance(block, dict) and block.get('type') == 'timeseries':
            if block.get('function'):
                node_functions[name] = {
                    'func': block.get('_function'),
                    'params': block.get('params', {}),
                    'inputs': block.get('inputs', [])
                }

    def compute_node_output(node_name: str) -> Optional[float]:
        if node_name not in node_functions:
            return None

        fn_info = node_functions[node_name]
        func = fn_info['func']
        params = fn_info['params']
        inputs = fn_info['inputs']

        if not inputs:
            return None

        input_arrays = []
        for inp in inputs:
            if inp in node_buffers:
                arr = node_buffers[inp].to_array()
                if arr is not None:
                    input_arrays.append(arr)

        if not input_arrays:
            return None

        try:
            if len(input_arrays) == 1:
                return func(input_arrays[0], **params)
            else:
                return func(*input_arrays, **params)
        except Exception:
            return None

    def execute_dag_bar(bar_idx: int, bar_data: np.ndarray) -> Dict[str, float]:
        for ts_name, col_idx in ts_to_col.items():
            if col_idx < len(bar_data):
                node_buffers[ts_name].push(float(bar_data[col_idx]))

        execution_order = dag.metadata.execution_order if dag.metadata else resolved.keys()
        node_outputs: Dict[str, float] = {}

        for name in execution_order:
            if name in ts_to_col:
                node_outputs[name] = float(bar_data[ts_to_col[name]]) if ts_to_col[name] < len(bar_data) else 0.0
            elif name in node_functions:
                output = compute_node_output(name)
                if output is not None:
                    node_outputs[name] = output
                    buf_size = buffer_sizes.get(name, 20)
                    if name not in node_buffers:
                        node_buffers[name] = CircularBuffer(buf_size)
                    node_buffers[name].push(output)

        return node_outputs

    def execute_bar_fn(bar_idx, bar_data, portfolio=None):
        current_price = float(bar_data[-1]) if len(bar_data) > 0 else 100.0
        
        if portfolio is None:
            portfolio_state = {
                'portfolio_value': capital,
                'position': 0.0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'drawdown': 0.0,
                'high_water_mark': capital,
                'bars_in_trade': 0
            }
            current_position = 0.0
        else:
            portfolio_state = portfolio.get_state(current_price)
            current_position = portfolio.position
        
        observation_dict = execute_dag_bar(bar_idx, bar_data)
        
        feature_names = list(observation_dict.keys())
        obs_array = np.array([observation_dict.get(k, 0.0) for k in feature_names], dtype=np.float64)
        
        action, conviction = agent.observe(obs_array, portfolio_state, bar_idx)
        
        quantity = sizer.calculate_size(
            action=action,
            conviction=conviction,
            current_position=current_position,
            portfolio_value=portfolio_state.get('portfolio_value', capital),
            instrument_id=symbol,
            current_price=current_price
        )
        
        return {
            'action': action,
            'quantity': quantity,
            'price': current_price,
            'symbol': symbol,
            'portfolio_value': portfolio_state.get('portfolio_value', capital),
            'position': current_position
        }

    warmup_bars = dag.metadata.warmup_bars if dag.metadata else 20

    result = trainer.train(data, warmup_bars, execute_bar_fn)

    checkpoint_path = os.path.join(checkpoint_dir, f"model_{training_config.seed}.joblib")
    agent.save_checkpoint(checkpoint_path)

    return TrainResult(
        checkpoint_path=checkpoint_path,
        config=resolved,
        n_blocks=result.get('n_blocks', 0),
        total_bars=result.get('total_bars', 0)
    )


def test(
    dsl_string: str,
    checkpoint_path: str,
    data: Optional[np.ndarray] = None,
    context: Optional[Dict[str, Any]] = None,
    dates: Optional[np.ndarray] = None
) -> TestResult:
    """
    Run walk-forward test on saved checkpoint.

    Args:
        dsl_string: Complete DSL content
        checkpoint_path: Path to saved model checkpoint
        data: Price/feature data array for test period
        context: Optional context with registered components
        dates: Optional date array for test_start resolution

    Returns:
        TestResult with equity curve, trades, and metrics

    Raises:
        ParseError: On syntax violation
        ConfigError: On validation failure
    """
    raw = parse(dsl_string)
    validated = validate(raw)
    resolved = resolve(validated, context)
    dag = build_dag(resolved)

    backtest_config = resolved.get('_backtest', {})
    
    if data is None:
        result = _load_data_from_adapters(resolved, backtest_config)
        if result is None:
            raise ValueError("Could not automatically load data from adapters. Please provide data array.")
        data, dates_from_load = result
        if dates is None:
            dates = dates_from_load
    
    test_start_str = backtest_config.get('test_start')
    if test_start_str and dates is not None:
        try:
            import pandas as pd
            test_date = pd.Timestamp(test_start_str, tz='UTC')
            mask = dates >= test_date
            if mask.any():
                test_start_idx = int(np.argmax(mask))
            else:
                test_start_idx = int(len(data) * 0.7)
        except Exception:
            test_start_idx = int(len(data) * 0.7)
    else:
        test_start_idx = int(len(data) * 0.7)
    
    model_config = resolved.get('model', {})
    agent_config = resolved.get('agent', {})

    from tradsl.models_impl import RandomForestModel
    from tradsl.agent_framework import ParameterizedAgent

    n_estimators = model_config.get('params', {}).get('n_estimators', 10) if model_config.get('params') else 10
    
    policy = RandomForestModel(n_estimators=n_estimators, random_state=42)
    agent = ParameterizedAgent(policy_model=policy, seed=42)
    agent.load_checkpoint(checkpoint_path)

    reward_name = model_config.get('reward_function', 'asymmetric_high_water_mark')
    reward_fn_class = _registry.get('rewards', {}).get(reward_name)
    if reward_fn_class is None:
        reward_fn_class = AsymmetricHighWaterMarkReward
    reward_fn = reward_fn_class()

    sizer_name = agent_config.get('sizer', 'kelly')
    sizer_class = _registry.get('sizers', {}).get(sizer_name)
    if sizer_class is None:
        sizer_class = KellySizer
    sizer = sizer_class()

    capital = backtest_config.get('capital', 100000.0)
    commission = backtest_config.get('commission', 0.001)
    symbol = list(agent_config.get('tradable', ['UNKNOWN']))[0] if agent_config.get('tradable') else 'UNKNOWN'

    transaction_costs = TransactionCosts(commission_rate=commission)

    engine = BacktestEngine(
        config=backtest_config,
        agent=agent,
        sizer=sizer,
        reward_function=reward_fn,
        transaction_costs=transaction_costs
    )

    test_source_timeseries = {}
    for name, block in resolved.items():
        if isinstance(block, dict) and block.get('type') == 'timeseries':
            if not block.get('inputs'):
                test_source_timeseries[name] = block

    test_ts_to_col = {}
    test_col_idx = 0
    test_primary_name = None
    for name, block in test_source_timeseries.items():
        if block.get('tradable') or test_primary_name is None:
            test_ts_to_col[name] = 0
            test_primary_name = name
            test_col_idx = 1
        else:
            test_ts_to_col[name] = test_col_idx
            test_col_idx += 1

    for name, block in resolved.items():
        if isinstance(block, dict) and block.get('type') == 'timeseries':
            if name not in test_ts_to_col:
                test_ts_to_col[name] = test_col_idx
                test_col_idx += 1

    test_node_buffers: Dict[str, CircularBuffer] = {}
    test_buffer_sizes = dag.metadata.node_buffer_sizes if dag.metadata else {}

    for name, block in resolved.items():
        if isinstance(block, dict) and block.get('type') == 'timeseries':
            buf_size = test_buffer_sizes.get(name, 20)
            test_node_buffers[name] = CircularBuffer(buf_size)

    test_node_functions: Dict[str, Any] = {}
    for name, block in resolved.items():
        if isinstance(block, dict) and block.get('type') == 'timeseries':
            if block.get('function'):
                test_node_functions[name] = {
                    'func': block.get('_function'),
                    'params': block.get('params', {}),
                    'inputs': block.get('inputs', [])
                }

    def test_compute_node_output(node_name: str) -> Optional[float]:
        if node_name not in test_node_functions:
            return None

        fn_info = test_node_functions[node_name]
        func = fn_info['func']
        params = fn_info['params']
        inputs = fn_info['inputs']

        if not inputs:
            return None

        input_arrays = []
        for inp in inputs:
            if inp in test_node_buffers:
                arr = test_node_buffers[inp].to_array()
                if arr is not None:
                    input_arrays.append(arr)

        if not input_arrays:
            return None

        try:
            if len(input_arrays) == 1:
                return func(input_arrays[0], **params)
            else:
                return func(*input_arrays, **params)
        except Exception:
            return None

    def execute_test_dag_bar(bar_idx: int, bar_data: np.ndarray) -> Dict[str, float]:
        for ts_name, col_idx in test_ts_to_col.items():
            if col_idx < len(bar_data):
                test_node_buffers[ts_name].push(float(bar_data[col_idx]))

        execution_order = dag.metadata.execution_order if dag.metadata else resolved.keys()
        node_outputs: Dict[str, float] = {}

        for name in execution_order:
            if name in test_ts_to_col:
                node_outputs[name] = float(bar_data[test_ts_to_col[name]]) if test_ts_to_col[name] < len(bar_data) else 0.0
            elif name in test_node_functions:
                output = test_compute_node_output(name)
                if output is not None:
                    node_outputs[name] = output
                    buf_size = test_buffer_sizes.get(name, 20)
                    if name not in test_node_buffers:
                        test_node_buffers[name] = CircularBuffer(buf_size)
                    test_node_buffers[name].push(output)

        return node_outputs

    def execute_bar_fn(bar_idx, bar_data, portfolio=None):
        current_price = float(bar_data[-1]) if len(bar_data) > 0 else 100.0
        
        if portfolio is not None:
            portfolio_state = portfolio.get_state(current_price)
            current_position = portfolio.position
        else:
            portfolio_state = {
                'portfolio_value': engine.portfolio_value,
                'position': engine.position,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'drawdown': engine.drawdown,
                'high_water_mark': engine.high_water_mark,
                'bars_in_trade': 0
            }
            current_position = engine.position
        
        test_observation_dict = execute_test_dag_bar(bar_idx, bar_data)
        
        test_feature_names = list(test_observation_dict.keys())
        test_obs_array = np.array([test_observation_dict.get(k, 0.0) for k in test_feature_names], dtype=np.float64)
        
        action, conviction = agent.observe(test_obs_array, portfolio_state, bar_idx)
        
        quantity = sizer.calculate_size(
            action=action,
            conviction=conviction,
            current_position=current_position,
            portfolio_value=portfolio_state.get('portfolio_value', engine.portfolio_value),
            instrument_id=symbol,
            current_price=current_price
        )
        
        return {
            'action': action,
            'quantity': quantity,
            'price': current_price,
            'symbol': symbol,
            'portfolio_value': portfolio_state.get('portfolio_value', engine.portfolio_value),
            'position': current_position
        }

    result = engine.run(data[test_start_idx:], execute_bar_fn)

    return TestResult(
        checkpoint_path=checkpoint_path,
        equity_curve=result.equity_curve,
        trades=[t.__dict__ for t in result.trades],
        metrics=result.metrics,
        config=resolved
    )


def run(
    dsl_string: str,
    context: Optional[Dict[str, Any]] = None,
    data: Optional[np.ndarray] = None,
    checkpoint_dir: str = "/tmp/tradsl_checkpoints"
) -> RunResult:
    """
    Full pipeline: train + test + bootstrap.

    Args:
        dsl_string: Complete DSL content
        context: Optional context with registered components
        data: Price/feature data array
        checkpoint_dir: Directory for checkpoints

    Returns:
        RunResult with train, test, and bootstrap results
    """
    raw = parse(dsl_string)
    validated = validate(raw)
    resolved = resolve(validated, context)
    backtest_config = resolved.get('_backtest', {})
    
    dates = None
    if data is None:
        result = _load_data_from_adapters(resolved, backtest_config)
        if result is None:
            raise ValueError("Could not load data from adapters.")
        data, dates = result
    
    train_result = train(dsl_string, context, data, checkpoint_dir)
    test_result = test(dsl_string, train_result.checkpoint_path, data, context, dates)

    bootstrap_result = None
    backtest_config = test_result.config.get('_backtest', {})
    if backtest_config.get('bootstrap_enabled', False):
        if data is not None:
            returns = np.diff(test_result.equity_curve) / test_result.equity_curve[:-1]
            bootstrap = IIDBootstrap(seed=backtest_config.get('seed', 42))
            bootstrap_result = bootstrap.compute_all_metrics(returns, n_samples=100)

    return RunResult(
        train_result=train_result,
        test_result=test_result,
        bootstrap_result=bootstrap_result,
        config=test_result.config
    )


def register_adapter(name: str, adapter_class: Type) -> None:
    """
    Register a data adapter.

    Args:
        name: Adapter name for DSL reference
        adapter_class: Adapter class implementing BaseAdapter
    """
    _registry['adapters'][name] = adapter_class


def register_function(
    name: str,
    func: Callable,
    category: str = "safe",
    description: str = "",
    min_lookback: int = 1,
    output_type: Type = float
) -> None:
    """
    Register a feature function.

    Args:
        name: Function name for DSL reference
        func: Callable with signature (arr: np.ndarray, **params) -> float
        category: FeatureCategory - 'safe', 'cyclical_structural', or 'date_derived'
        description: Human-readable description
        min_lookback: Minimum lookback window required
        output_type: float or dict for multi-output
    """
    _registry['functions'][name] = {
        'func': func,
        'category': category,
        'description': description,
        'min_lookback': min_lookback,
        'output_type': output_type
    }


def register_trainable_model(name: str, model_class: Type) -> None:
    """
    Register a trainable model class.

    Args:
        name: Model name for DSL reference
        model_class: Class implementing BaseTrainableModel
    """
    _registry['trainable_models'][name] = model_class


def register_label_function(
    name: str,
    func: Callable,
    description: str = "",
    requires_future_bars: int = 0
) -> None:
    """
    Register a label function for trainable models.

    Args:
        name: Label function name for DSL reference
        func: Callable with signature (features, prices, bar_index, **params) -> np.ndarray
        description: Human-readable description
        requires_future_bars: Number of future bars needed for label computation
    """
    _registry['label_functions'][name] = {
        'func': func,
        'description': description,
        'requires_future_bars': requires_future_bars
    }


def register_agent(name: str, agent_class: Type) -> None:
    """
    Register an agent architecture class.

    Args:
        name: Agent name for DSL reference
        agent_class: Class implementing BaseAgentArchitecture
    """
    _registry['agents'][name] = agent_class


def register_sizer(name: str, sizer_class: Type) -> None:
    """
    Register a position sizer.

    Args:
        name: Sizer name for DSL reference
        sizer_class: Class implementing BasePositionSizer
    """
    _registry['sizers'][name] = sizer_class


def register_reward(name: str, reward_class: Type) -> None:
    """
    Register a reward function.

    Args:
        name: Reward function name for DSL reference
        reward_class: Class implementing BaseRewardFunction
    """
    _registry['rewards'][name] = reward_class


def get_registry() -> Dict[str, Dict[str, Any]]:
    """Get current registry state (for testing)."""
    return _registry.copy()


def get_function_registry() -> Dict[str, Any]:
    """Get function registry for validation."""
    return _registry.get('functions', {})


def clear_registry() -> None:
    """Clear all registered components (for testing)."""
    for key in _registry:
        _registry[key].clear()


__all__ = [
    'parse_dsl',
    'validate_config',
    'resolve_config',
    'build_execution_dag',
    'load',
    'register_adapter',
    'register_function',
    'register_trainable_model',
    'register_label_function',
    'register_agent',
    'register_sizer',
    'register_reward',
    'get_registry',
    'get_function_registry',
    'clear_registry',
    'ValidationResult',
    'DAG',
    'DAGMetadata',
    'TradSLError',
    'ParseError',
    'ConfigError',
    'ResolutionError',
    'CycleError',
    'AdapterError',
    'FeatureError',
    'ModelError',
    'ExecutionError',
    'ValidationError',
    'BaseAdapter',
    'BaseTrainableModel',
    'BaseAgentArchitecture',
    'BasePositionSizer',
    'BaseRewardFunction',
    'TradingAction',
    'FractionalSizer',
    'FixedSizer',
    'KellySizer',
    'SimplePnLReward',
    'AsymmetricHighWaterMarkReward',
    'DifferentialSharpeReward',
    'RewardContext',
    'CircularBuffer',
    'BlockSampler',
    'BlockTrainer',
    'WalkForwardTester',
    'TrainingConfig',
    'IIDBootstrap',
    'BlockBootstrap',
    'BootstrapEnsemble',
    'YFinanceAdapter',
    'ParameterizedAgent',
    'TabularAgent',
    'PPOUpdate',
    'DQNUpdate',
    'SklearnModel',
    'RandomForestModel',
    'LinearModel',
    'DecisionTreeModel',
    'GradientBoostingModel',
    'VolatilityTargetingSizer',
    'MaxDrawdownSizer',
    'EnsembleSizer',
    'SparseTimeSeries',
    'SparseBuffer',
    'BacktestEngine',
    'TransactionCosts',
    'ExecutionMode',
    'PerformanceMetrics',
    'compute_metrics',
    'train',
    'test',
    'run',
    'TrainResult',
    'TestResult',
    'RunResult',
    'TradSLStrategy',
    'TradSLNTStrategy',
    'Portfolio',
    'Position',
    'TestDataAdapter',
    'create_test_adapter',
]
