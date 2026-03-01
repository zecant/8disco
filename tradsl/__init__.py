from .parser import parse_config
from .schema import validate, ConfigError
from .resolver import resolve, ResolutionError
from .dag import build_dag, CycleError


__all__ = [
    'parse_config',
    'validate', 
    'resolve',
    'build_dag',
    'parse',
    'ConfigError',
    'ResolutionError',
    'CycleError'
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
        
        # Derived timeseries with function
        :nvda_ma30
        type=timeseries
        function=rolling_mean
        inputs=[nvda]
        params=mlparams
        
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
        >>> def rolling_mean(data, window=30): pass
        >>> class RandomForest: pass
        >>> def kelly_sizer(signals, tradable): pass
        >>> 
        >>> config = parse('''
        ... mlparams:
        ... lr=0.001
        ... 
        ... :agent
        ... type=agent
        ... inputs=[spy, vix]
        ... tradable=[spy]
        ... sizer=kelly_sizer
        ... 
        ... :spy
        ... type=timeseries
        ... inputs=[]
        ... ''', context={
        ...     'kelly_sizer': kelly_sizer,
        ...     'rolling_mean': rolling_mean,
        ...     'RandomForest': RandomForest,
        ... })
        >>> config['agent']['sizer'](...)
    """
    raw = parse_config(source)
    validated = validate(raw)
    resolved = resolve(validated, context)
    return build_dag(resolved)
