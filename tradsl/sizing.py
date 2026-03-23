"""
Sizing functions for TradSL portfolio execution.

Sizing functions determine position quantity based on agent output,
portfolio state, and current price.

Example usage in DSL:
    executor:
        type=function
        function=portfolio.execute
        inputs=[agent, price]
        symbol=AAPL
        sizing_fn=portfolio.sizing.fractional
        fraction=0.1

Registry setup:
    from tradsl import default_registry
    dag.resolve(default_registry)  # portfolio.sizing.* is auto-registered
"""
from abc import ABC, abstractmethod
import pandas as pd

from tradsl.portfolio_state import PortfolioState


class SizingFunction(ABC):
    """Abstract base class for position sizing functions."""
    
    @abstractmethod
    def compute(
        self,
        agent_output: pd.DataFrame,
        portfolio_state: PortfolioState,
        price: float,
    ) -> int:
        """Compute position size in shares.
        
        Args:
            agent_output: DataFrame with columns [action, confidence, asset]
            portfolio_state: Current portfolio state
            price: Current price per share
            
        Returns:
            Number of shares to trade (positive for buy, negative for sell, 0 for hold)
        """
        pass


class FractionalSizing(SizingFunction):
    """Size position as a fraction of portfolio NAV (Net Asset Value)."""
    
    def __init__(self, fraction: float = 0.1):
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be between 0 and 1")
        self.fraction = fraction
    
    def compute(
        self,
        agent_output: pd.DataFrame,
        portfolio_state: PortfolioState,
        price: float,
    ) -> int:
        if price <= 0:
            return 0
        
        action = int(agent_output["action"].iloc[-1])
        if action == 1:
            return 0
        
        nav = portfolio_state.cash
        for symbol, qty in portfolio_state.holdings.items():
            nav += qty * price * 0.5
        
        target_value = nav * self.fraction
        
        if action == 0:
            return int(target_value / price)
        elif action == 2:
            holding = portfolio_state.get_holding(
                agent_output["asset"].iloc[-1]
            )
            return -min(holding, int(target_value / price))
        
        return 0


fractional = FractionalSizing
