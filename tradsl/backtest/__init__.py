"""
Backtesting module for trading strategies.

Requires NautilusTrader for NT-backed backtesting.
This module fails fast - if NautilusTrader is not installed, imports will fail.
"""

from .engine import BacktestEngine, BacktestResult
from .execution import ExecutionBackend, Fill, Order

# Import NT modules - fails fast if not available
from .nautilus_adapter import NautilusAdapter, NautilusResultParser
from .nautilus_strategy import NautilusStrategy, NautilusStrategyConfig, create_nautilus_strategy
from .nautilus_backend import NautilusBackend, run_nautilus_backtest

__all__ = [
    # Internal backtest
    'BacktestEngine',
    'BacktestResult',
    'ExecutionBackend', 
    'Fill',
    'Order',
    # NautilusTrader backtest
    'NautilusAdapter',
    'NautilusResultParser',
    'NautilusStrategy',
    'NautilusStrategyConfig',
    'create_nautilus_strategy',
    'NautilusBackend',
    'run_nautilus_backtest',
]
