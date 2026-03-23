"""
Portfolio components for TradSL.

Provides shared state and execution for portfolio management.

Example usage in DSL:
    portfolio:
        type=timeseries
        adapter=portfolioadapter
        symbols=[AAPL, TSLA]
        initial_cash=10000

    executor:
        type=function
        function=portfolio.execute
        inputs=[agent, price]
        sizing_fn=portfolio.sizing.fractional
        execution_model=portfolio.execution.ohlc_avg

Registry setup:
    from tradsl import default_registry
    dag.resolve(default_registry)  # portfolio.* is auto-registered
"""
from tradsl.portfolio_state import PortfolioState
from tradsl.portfolio_function import PortfolioFunction
from tradsl import sizing
from tradsl import execution

execute = PortfolioFunction

__all__ = ["PortfolioState", "PortfolioFunction", "execute", "sizing", "execution"]
