"""
Backtesting module for trading strategies.
"""

from .engine import BacktestEngine, BacktestResult
from .execution import ExecutionBackend, Fill, Order

NAUTILUS_AVAILABLE = False

__all__ = [
    'BacktestEngine',
    'BacktestResult',
    'ExecutionBackend', 
    'Fill',
    'Order',
    'NAUTILUS_AVAILABLE',
]

def __getattr__(name):
    """Lazy load NautilusTrader modules if available."""
    global NAUTILUS_AVAILABLE
    
    if name == 'NautilusAdapter' or name == 'NautilusResultParser':
        try:
            from .nautilus_adapter import NautilusAdapter, NautilusResultParser
            NAUTILUS_AVAILABLE = True
            return NautilusAdapter if name == 'NautilusAdapter' else NautilusResultParser
        except (ImportError, TypeError):
            return None
    
    if name in ('NautilusStrategy', 'NautilusStrategyConfig', 'create_nautilus_strategy'):
        try:
            from .nautilus_strategy import NautilusStrategy, NautilusStrategyConfig, create_nautilus_strategy
            NAUTILUS_AVAILABLE = True
            if name == 'NautilusStrategy':
                return NautilusStrategy
            elif name == 'NautilusStrategyConfig':
                return NautilusStrategyConfig
            else:
                return create_nautilus_strategy
        except (ImportError, TypeError):
            return None
    
    if name in ('NautilusBackend', 'run_nautilus_backtest'):
        try:
            from .nautilus_backend import NautilusBackend, run_nautilus_backtest
            NAUTILUS_AVAILABLE = True
            return NautilusBackend if name == 'NautilusBackend' else run_nautilus_backtest
        except (ImportError, TypeError):
            return None
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
