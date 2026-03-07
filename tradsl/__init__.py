from .parser import parse_config
from .schema import validate, ConfigError
from .resolver import resolve, ResolutionError
from .dag import build_dag, CycleError
from .functions import BUILTIN_FUNCTIONS


__all__ = [
    'parse_config',
    'validate', 
    'resolve',
    'build_dag',
    'parse',
    'ConfigError',
    'ResolutionError',
    'CycleError',
    'BUILTIN_FUNCTIONS'
]


def parse(source: str, context: dict = None) -> dict:
    """
    Parse DSL config string and return resolved dictionary.
    
    DSL Syntax:
        # Parameter blocks (reusable)
        mlparams:
          lr=0.001
          epochs=100
        
        # Adapter definitions
        :yfinance
        type=adapter
        class=adapters.YFAdapter
        
        # Timeseries with adapter
        :nvda
        type=timeseries
        adapter=yfinance
        parameters=["nvda"]
        
        # Derived timeseries with function (uses built-in functions automatically)
        :nvda_ma30
        type=timeseries
        function=sma
        inputs=[nvda]
        params=window=30
        
        # Models (can use other models as inputs - submodels)
        :signal_model
        type=model
        class=RandomForest
        inputs=[nvda, vix]
        params=mlparams
        
        # Agent (orchestrates trading)
        :agent
        type=agent
        inputs=[signal_model, vix]
        tradable=[nvda]
        sizer=kelly_sizer
        
        # Backtest configuration
        :backtest
        type=backtest
        start=2020-01-01
        end=2024-01-01
        capital=100000
    
    Args:
        source: DSL config string
        context: Python context for resolving function/class names and adapters.
                 Pass {name: callable} for functions, classes, sizers.
                 Built-in functions (sma, ema, z_score, rsi, etc.) are
                 automatically available.
                 For adapter class paths like 'adapters.YFAdapter', pass
                 {'adapters.YFAdapter': YourAdapterClass}.
        
    Returns:
        Dict with:
        - Config blocks (agent, timeseries, models, etc.) with resolved callables
        - _params: {name: dict} for parameter blocks
        - _adapters: {name: instantiated adapter} for adapter definitions  
        - _backtest: {dict} for backtest config
        - _execution_order: [list] topological sort of nodes
        - _graph: {deps, reverse_deps} dependency info
        
    Example:
        >>> from tradsl.models import DecisionTreeModel
        >>> 
        >>> config = parse('''
        ... :agent
        ... type=agent
        ... inputs=[spy]
        ... tradable=[spy]
        ... sizer=equal_weight
        ... 
        ... :spy
        ... type=timeseries
        ... inputs=[]
        ... ''', context={
        ...     'equal_weight': lambda signals, tradable: ...,
        ...     'DecisionTreeModel': DecisionTreeModel,
        ... })
    """
    raw = parse_config(source)
    validated = validate(raw)
    
    # Merge built-in functions with user-provided context
    # User context takes precedence over built-ins
    merged_context = dict(BUILTIN_FUNCTIONS)
    if context:
        merged_context.update(context)
    
    resolved = resolve(validated, merged_context)
    return build_dag(resolved)
