"""
Execution models for TradSL portfolio execution.

Execution models determine the actual execution price for trades based on
price data (OHLCV) and the sizing output (ticker, quantity).

Example usage in DSL:
    executor:
        type=function
        function=portfolio.execute
        inputs=[agent, price]
        sizing_fn=portfolio.sizing.fractional
        execution_model=portfolio.execution.ohlc_avg

Registry setup:
    from tradsl import default_registry
    dag.resolve(default_registry)  # portfolio.execution.* is auto-registered
"""
from abc import ABC, abstractmethod
import pandas as pd


class ExecutionModel(ABC):
    """Abstract base class for execution price models."""
    
    @abstractmethod
    def calculate(self, sizing_output: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate execution price and return updated sizing output.
        
        Args:
            sizing_output: DataFrame with columns [quantity, action, asset, confidence]
            price_data: DataFrame with OHLCV data (open, high, low, close, volume)
                       indexed by timestamp
            
        Returns:
            DataFrame with sizing output plus execution columns:
            [quantity, action, asset, confidence, execution_price, execution_cost]
        """
        pass


class OhlcAvgExecution(ExecutionModel):
    """Execute at midpoint of open and close prices."""
    
    def calculate(self, sizing_output: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        result = sizing_output.copy()
        
        open_col = [c for c in price_data.columns if c.endswith('.open')][0]
        close_col = [c for c in price_data.columns if c.endswith('.close')][0]
        
        open_price = price_data[open_col].iloc[-1]
        close_price = price_data[close_col].iloc[-1]
        
        execution_price = (open_price + close_price) / 2
        
        result['execution_price'] = execution_price
        result['execution_cost'] = abs(result['quantity']) * execution_price
        
        return result


ohlc_avg = OhlcAvgExecution
