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
    training_config_dict = resolved.get('_training', {})

    training_config = TrainingConfig(
        training_window=training_config_dict.get('training_window', 504),
        retrain_schedule=training_config_dict.get('retrain_schedule', 'every_n_bars'),
        retrain_n=training_config_dict.get('retrain_n', 252),
        n_training_blocks=training_config_dict.get('n_training_blocks', 40),
        block_size_min=training_config_dict.get('block_size_min', 30),
        block_size_max=training_config_dict.get('block_size_max', 120),
        seed=training_config_dict.get('seed', 42)
    )

    if data is None:
        raise ValueError("data array is required for training")

    from tradsl.models_impl import RandomForestModel
    from tradsl.rewards import AsymmetricHighWaterMarkReward
    from tradsl.sizers import KellySizer

    policy = RandomForestModel(n_estimators=10, random_state=42)
    agent = ParameterizedAgent(
        policy_model=policy,
        update_schedule=training_config.retrain_schedule,
        update_n=training_config.retrain_n,
        replay_buffer_size=4096,
        seed=training_config.seed
    )

    reward_fn = AsymmetricHighWaterMarkReward()
    sizer = KellySizer()

    block_sampler = BlockSampler(training_config)
    trainer = BlockTrainer(training_config, agent, reward_fn, block_sampler)

    def execute_bar_fn(bar_idx, bar_data):
        return {'portfolio_value': 100000.0, 'position': 0.0}

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
    data: np.ndarray,
    context: Optional[Dict[str, Any]] = None
) -> TestResult:
    """
    Run walk-forward test on saved checkpoint.

    Args:
        dsl_string: Complete DSL content
        checkpoint_path: Path to saved model checkpoint
        data: Price/feature data array for test period
        context: Optional context with registered components

    Returns:
        TestResult with equity curve, trades, and metrics

    Raises:
        ParseError: On syntax violation
        ConfigError: On validation failure
    """
    raw = parse(dsl_string)
    validated = validate(raw)
    resolved = resolve(validated, context)

    backtest_config = resolved.get('_backtest', {})
    test_start = backtest_config.get('test_start', '2022-01-01')

    from tradsl.models_impl import RandomForestModel
    from tradsl.agent_framework import ParameterizedAgent

    policy = RandomForestModel(n_estimators=10, random_state=42)
    agent = ParameterizedAgent(policy_model=policy, seed=42)
    agent.load_checkpoint(checkpoint_path)

    capital = backtest_config.get('capital', 100000.0)
    commission = backtest_config.get('commission', 0.001)

    transaction_costs = TransactionCosts(commission_rate=commission)

    engine = BacktestEngine(
        config=backtest_config,
        agent=agent,
        sizer=KellySizer(),
        reward_function=AsymmetricHighWaterMarkReward(),
        transaction_costs=transaction_costs
    )

    def execute_bar_fn(bar_idx, bar_data):
        return {'portfolio_value': engine.portfolio_value, 'position': engine.position}

    test_start_idx = int(len(data) * 0.7)

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
    train_result = train(dsl_string, context, data, checkpoint_dir)

    test_result = test(dsl_string, train_result.checkpoint_path, data, context)

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
