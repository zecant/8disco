"""
TradSL - Trading Strategy Domain-Specific Language.

A declarative configuration language for building and backtesting trading strategies
that compiles to an executable DAG (Directed Acyclic Graph).

Example usage:
    from tradsl import DAG, parse, default_registry

    config = parse('''
        price:
            type=timeseries
            adapter=yfinance
            symbol=AAPL

        sma:
            type=function
            function=pricetransforms.ema
            inputs=[price]
            window=20
    ''')

    dag = DAG.from_config(config)
    dag.resolve(default_registry)
    dag.build()

    dag.step()
    print(dag.values())
"""
from tradsl.dag import DAG, Node
from tradsl.exceptions import CycleError, ConfigError, InvariantError, ResolutionError
from tradsl.functions import Function
from tradsl.adapters import Adapter, YFinanceAdapter
from tradsl.portfolio_adapter import PortfolioAdapter
from tradsl.portfolio_state import PortfolioState
from tradsl.portfolio_function import PortfolioFunction

__version__ = "0.0.3"

__all__ = [
    "DAG",
    "Node",
    "Function",
    "Adapter",
    "YFinanceAdapter",
    "PortfolioAdapter",
    "PortfolioState",
    "PortfolioFunction",
    "CycleError",
    "ConfigError",
    "InvariantError",
    "ResolutionError",
    "default_registry",
]


import tradsl.pricetransforms as pricetransforms
import tradsl.ml as ml
import tradsl.portfolio as portfolio

default_registry: dict[str, object] = {
    "pricetransforms": pricetransforms,
    "ml": ml,
    "portfolio": portfolio,
    "yfinance": YFinanceAdapter,
    "portfolioadapter": PortfolioAdapter,
}
