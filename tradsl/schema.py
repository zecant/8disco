"""
Schema Validation for TradSL

Validates parsed DSL config against schema rules.
See Section 5 of the specification.
"""
from typing import Any, Dict, List, Optional, Set, Tuple, Type
from dataclasses import dataclass, field
from enum import Enum
from tradsl.exceptions import ConfigError, ResolutionError, CycleError


class FeatureCategory(Enum):
    """Feature classification for date-blind mode."""
    SAFE = "safe"
    CYCLICAL_STRUCTURAL = "cyclical_structural"
    DATE_DERIVED = "date_derived"


@dataclass
class FieldSpec:
    """Schema specification for a single field."""
    type: Type
    required: bool = False
    allowed_values: Optional[Set[str]] = None
    range: Optional[Tuple[Any, Any]] = None
    default: Any = None


BLOCK_SCHEMAS: Dict[str, Dict[str, FieldSpec]] = {
    'timeseries': {
        'type': FieldSpec(type=str, required=True, allowed_values={'timeseries'}),
        'adapter': FieldSpec(type=str, required=False),
        'function': FieldSpec(type=str, required=False),
        'inputs': FieldSpec(type=list, required=False),
        'params': FieldSpec(type=str, required=False),
        'tradable': FieldSpec(type=bool, required=False, default=False),
        'changed_tracking': FieldSpec(type=bool, required=False, default=False),
        'sparse': FieldSpec(type=bool, required=False, default=False),
    },
    'trainable_model': {
        'type': FieldSpec(type=str, required=True, allowed_values={'trainable_model'}),
        'class': FieldSpec(type=str, required=True),
        'inputs': FieldSpec(type=list, required=True),
        'label_function': FieldSpec(type=str, required=True),
        'label_params': FieldSpec(type=str, required=False),
        'retrain_schedule': FieldSpec(type=str, required=False),
        'retrain_n': FieldSpec(type=int, required=False),
        'training_window': FieldSpec(type=int, required=True),
        'params': FieldSpec(type=str, required=False),
        'dotraining': FieldSpec(type=bool, required=False, default=True),
        'load_from': FieldSpec(type=str, required=False),
    },
    'model': {
        'type': FieldSpec(type=str, required=True, allowed_values={'model'}),
        'class': FieldSpec(type=str, required=True),
        'inputs': FieldSpec(type=list, required=True),
        'portfolio_inputs': FieldSpec(type=list, required=False),
        'action_space': FieldSpec(type=str, required=True, allowed_values={'discrete', 'continuous'}),
        'reward_function': FieldSpec(type=str, required=True),
        'reward_params': FieldSpec(type=str, required=False),
        'replay_buffer_size': FieldSpec(type=int, required=False, default=4096),
        'replay_buffer_type': FieldSpec(type=str, required=False, allowed_values={'uniform', 'prioritized'}, default='uniform'),
        'update_schedule': FieldSpec(type=str, required=False),
        'update_n': FieldSpec(type=int, required=False),
        'update_threshold': FieldSpec(type=float, required=False),
        'training_window': FieldSpec(type=int, required=True),
        'entropy_coeff': FieldSpec(type=float, required=False, default=0.01),
        'dotraining': FieldSpec(type=bool, required=False, default=True),
        'load_from': FieldSpec(type=str, required=False),
    },
    'agent': {
        'type': FieldSpec(type=str, required=True, allowed_values={'agent'}),
        'inputs': FieldSpec(type=list, required=True),
        'tradable': FieldSpec(type=list, required=True),
        'sizer': FieldSpec(type=str, required=True),
        'sizer_params': FieldSpec(type=str, required=False),
        'update_schedule': FieldSpec(type=str, required=False, allowed_values={'every_n_bars', 'performance_degradation'}, default='every_n_bars'),
        'update_n': FieldSpec(type=int, required=False, default=10),
    },
    'adapter': {
        'type': FieldSpec(type=str, required=True, allowed_values={'adapter'}),
        'class': FieldSpec(type=str, required=True),
    },
    'backtest': {
        'type': FieldSpec(type=str, required=True, allowed_values={'backtest'}),
        'start': FieldSpec(type=str, required=True),
        'end': FieldSpec(type=str, required=True),
        'test_start': FieldSpec(type=str, required=True),
        'capital': FieldSpec(type=float, required=False, default=100000.0),
        'commission': FieldSpec(type=float, required=False, default=0.001),
        'slippage': FieldSpec(type=float, required=False, default=0.0),
        'market_impact_coeff': FieldSpec(type=float, required=False, default=0.0),
        'training_mode': FieldSpec(type=str, required=False, allowed_values={'random_blocks', 'contiguous'}, default='random_blocks'),
        'block_size_min': FieldSpec(type=int, required=False, default=30),
        'block_size_max': FieldSpec(type=int, required=False, default=120),
        'n_training_blocks': FieldSpec(type=int, required=False, default=40),
        'block_reset_portfolio': FieldSpec(type=bool, required=False, default=True),
        'walk_forward_test': FieldSpec(type=bool, required=False, default=True),
        'bootstrap_enabled': FieldSpec(type=bool, required=False, default=True),
        'bootstrap_samples': FieldSpec(type=int, required=False, default=1000),
        'bootstrap_block_length': FieldSpec(type=str, required=False, default='auto'),
        'date_blind': FieldSpec(type=bool, required=False, default=True),
        'regime_stratify': FieldSpec(type=bool, required=False, default=True),
        'seed': FieldSpec(type=int, required=False, default=42),
    },
}


VALID_PORTFOLIO_INPUTS: Set[str] = {
    'position', 'position_value', 'unrealized_pnl', 'unrealized_pnl_pct',
    'realized_pnl', 'portfolio_value', 'cash', 'drawdown', 'time_in_trade',
    'high_water_mark', 'position_weight', 'rolling_sharpe', 'rolling_volatility'
}


def validate(raw: Dict[str, Any], function_registry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Validate parsed DSL config.

    Args:
        raw: Output from parser.parse()
        function_registry: Optional dict of registered functions for date-blind validation

    Returns:
        Validated config with defaults filled

    Raises:
        ConfigError: With list of all validation errors
    """
    errors: List[str] = []

    adapters: Dict[str, Dict] = {}
    timeseries_nodes: Dict[str, Dict] = {}
    trainable_models: Dict[str, Dict] = {}
    models: Dict[str, Dict] = {}
    agents: Dict[str, Dict] = {}
    backtest: Optional[Dict] = None
    params: Dict[str, Dict] = {}

    for name, block in raw.items():
        if name.startswith('_'):
            continue

        block_type = block.get('type')
        if block_type is None:
            params[name] = block
        elif block_type == 'adapter':
            adapters[name] = block
        elif block_type == 'timeseries':
            timeseries_nodes[name] = block
        elif block_type == 'trainable_model':
            trainable_models[name] = block
        elif block_type == 'model':
            models[name] = block
        elif block_type == 'agent':
            agents[name] = block
        elif block_type == 'backtest':
            if backtest:
                errors.append(f"Multiple backtest blocks found: '{name}'")
            backtest = block
        else:
            errors.append(f"Unknown block type '{block_type}' for block '{name}'")

    for name, block in timeseries_nodes.items():
        field_errors = _validate_fields(name, block, 'timeseries')
        errors.extend(field_errors)

        adapter = block.get('adapter')
        function = block.get('function')

        if adapter and function:
            errors.append(f"Block '{name}': cannot have both 'adapter' and 'function'")
        if not adapter and not function:
            errors.append(f"Block '{name}': must have either 'adapter' or 'function'")
        if function and not block.get('inputs'):
            errors.append(f"Block '{name}': 'function' requires 'inputs'")

    for name, block in trainable_models.items():
        field_errors = _validate_fields(name, block, 'trainable_model')
        errors.extend(field_errors)

        if block.get('dotraining') and not block.get('update_schedule'):
            errors.append(f"Block '{name}': 'dotraining=true' requires 'update_schedule'")

        if block.get('training_window') and block.get('retrain_n'):
            if block['training_window'] <= block['retrain_n']:
                errors.append(f"Block '{name}': 'training_window' must be greater than 'retrain_n'")

    for name, block in models.items():
        field_errors = _validate_fields(name, block, 'model')
        errors.extend(field_errors)

        portfolio_inputs = block.get('portfolio_inputs', [])
        for inp in portfolio_inputs:
            if inp not in VALID_PORTFOLIO_INPUTS:
                errors.append(f"Block '{name}': invalid portfolio_input '{inp}'. Valid: {VALID_PORTFOLIO_INPUTS}")

    for name, block in agents.items():
        field_errors = _validate_fields(name, block, 'agent')
        errors.extend(field_errors)

    if backtest:
        field_errors = _validate_fields('_backtest', backtest, 'backtest')
        errors.extend(field_errors)

        start = backtest.get('start')
        end = backtest.get('end')
        test_start = backtest.get('test_start')

        if start and test_start and test_start <= start:
            errors.append("backtest: 'test_start' must be after 'start'")
        if end and test_start and test_start >= end:
            errors.append("backtest: 'test_start' must be before 'end'")

        block_size_min = backtest.get('block_size_min', 30)
        block_size_max = backtest.get('block_size_max', 120)
        if block_size_min >= block_size_max:
            errors.append("backtest: 'block_size_min' must be less than 'block_size_max'")

        capital = backtest.get('capital', 100000)
        if capital <= 0:
            errors.append("backtest: 'capital' must be positive")

    all_errors = _validate_cross_references(
        timeseries_nodes, trainable_models, models, agents, adapters, params, backtest
    )
    errors.extend(all_errors)

    if len(agents) == 0 and (timeseries_nodes or trainable_models or models):
        pass  # Agent required only if there are nodes to connect
    elif len(agents) != 1:
        errors.append(f"Expected exactly one agent block, found {len(agents)}")

    if errors:
        raise ConfigError("Validation failed", errors=errors)

    validated = raw.copy()

    backtest = validated.get('_backtest') or validated.get('backtest')
    if backtest and backtest.get('date_blind', True):
        date_derived_errors = _validate_date_blind(timeseries_nodes, function_registry)
        if date_derived_errors:
            raise ConfigError("Validation failed", errors=date_derived_errors)

    for name, block in list(timeseries_nodes.items()) + list(trainable_models.items()) + \
                       list(models.items()) + list(agents.items()):
        if name in raw:
            validated[name] = _fill_defaults(block, BLOCK_SCHEMAS.get(block.get('type', ''), {}))

    backtest_key = '_backtest' if '_backtest' in validated else 'backtest'
    if backtest and backtest_key in validated:
        validated[backtest_key] = _fill_defaults(backtest, BLOCK_SCHEMAS.get('backtest', {}))

    return validated


def _validate_fields(block_name: str, block: Dict, block_type: str) -> List[str]:
    """Validate fields in a single block."""
    errors = []
    schema = BLOCK_SCHEMAS.get(block_type, {})

    for field_name, spec in schema.items():
        value = block.get(field_name)

        if value is None:
            if spec.required:
                errors.append(f"Block '{block_name}': missing required field '{field_name}'")
            continue

        if not isinstance(value, spec.type):
            errors.append(
                f"Block '{block_name}': field '{field_name}' expected {spec.type.__name__}, "
                f"got {type(value).__name__} = {value!r}"
            )
            continue

        if spec.allowed_values and value not in spec.allowed_values:
            errors.append(
                f"Block '{block_name}': field '{field_name}' must be one of {spec.allowed_values}, "
                f"got {value!r}"
            )

        if spec.range and isinstance(value, (int, float)):
            min_val, max_val = spec.range
            if value < min_val or value > max_val:
                errors.append(
                    f"Block '{block_name}': field '{field_name}' must be in {spec.range}, "
                    f"got {value}"
                )

    return errors


def _validate_date_blind(
    timeseries_nodes: Dict[str, Dict],
    function_registry: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Validate that no DATE_DERIVED functions are used when date_blind=true."""
    errors = []
    
    if function_registry is None:
        from tradsl import get_function_registry
        function_registry = get_function_registry()
    
    for name, block in timeseries_nodes.items():
        func_name = block.get('function')
        if not func_name:
            continue
        
        if func_name in function_registry:
            func_info = function_registry[func_name]
            category = func_info.get('category')
            
            if category == FeatureCategory.DATE_DERIVED.value:
                errors.append(
                    f"Block '{name}': function '{func_name}' is classified as DATE_DERIVED "
                    f"and cannot be used when backtest.date_blind=true"
                )
    
    return errors


def _fill_defaults(block: Dict, schema: Dict[str, FieldSpec]) -> Dict:
    """Fill in default values for missing optional fields."""
    result = dict(block)
    for field_name, spec in schema.items():
        if field_name not in result and spec.default is not None:
            result[field_name] = spec.default
    return result


def _validate_cross_references(
    timeseries: Dict[str, Dict],
    trainable_models: Dict[str, Dict],
    models: Dict[str, Dict],
    agents: Dict[str, Dict],
    adapters: Dict[str, Dict],
    params: Dict[str, Dict],
    backtest: Optional[Dict]
) -> List[str]:
    """Validate references between blocks."""
    errors = []

    all_nodes: Dict[str, Dict] = {}
    all_nodes.update(timeseries)
    all_nodes.update(trainable_models)
    all_nodes.update(models)

    node_names = set(all_nodes.keys())

    for name, block in timeseries.items():
        adapter = block.get('adapter')
        if adapter and adapter not in adapters:
            errors.append(f"Block '{name}': adapter '{adapter}' not found")

        params_ref = block.get('params')
        if params_ref and params_ref not in params:
            errors.append(f"Block '{name}': params '{params_ref}' not found")

        inputs = block.get('inputs', [])
        for inp in inputs:
            if inp not in node_names:
                errors.append(f"Block '{name}': input '{inp}' not found")

    for name, block in trainable_models.items():
        label_func = block.get('label_function')
        if label_func:
            params_ref = block.get('label_params')
            if params_ref and params_ref not in params:
                errors.append(f"Block '{name}': label_params '{params_ref}' not found")

        inputs = block.get('inputs', [])
        for inp in inputs:
            if inp not in node_names:
                errors.append(f"Block '{name}': input '{inp}' not found")

        params_ref = block.get('params')
        if params_ref and params_ref not in params:
            errors.append(f"Block '{name}': params '{params_ref}' not found")

    for name, block in models.items():
        inputs = block.get('inputs', [])
        for inp in inputs:
            if inp not in node_names:
                errors.append(f"Block '{name}': input '{inp}' not found")

        reward_func = block.get('reward_function')
        if reward_func:
            params_ref = block.get('reward_params')
            if params_ref and params_ref not in params:
                errors.append(f"Block '{name}': reward_params '{params_ref}' not found")

        params_ref = block.get('params')
        if params_ref and params_ref not in params:
            errors.append(f"Block '{name}': params '{params_ref}' not found")

    for name, block in agents.items():
        inputs = block.get('inputs', [])
        for inp in inputs:
            if inp not in node_names:
                errors.append(f"Block '{name}': input '{inp}' not found")

        sizer = block.get('sizer')
        if sizer:
            params_ref = block.get('sizer_params')
            if params_ref and params_ref not in params:
                errors.append(f"Block '{name}': sizer_params '{params_ref}' not found")

        tradable = block.get('tradable', [])
        for sym in tradable:
            if sym not in timeseries:
                errors.append(f"Block '{name}': tradable symbol '{sym}' not found in timeseries nodes")
            elif not timeseries[sym].get('tradable', False):
                errors.append(f"Block '{name}': symbol '{sym}' must have tradable=true in timeseries block")

    return errors
