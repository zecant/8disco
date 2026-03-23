from typing import Optional, Type
import pandas as pd

from tradsl.functions import Function
from tradsl.portfolio_state import PortfolioState
from tradsl.sizing import SizingFunction, FractionalSizing
from tradsl.execution import ExecutionModel, OhlcAvgExecution


class PortfolioFunction(Function):
    def __init__(
        self,
        state: PortfolioState,
        sizing_fn: Optional[Type[SizingFunction]] = None,
        sizing_params: Optional[dict] = None,
        execution_model: Optional[Type[ExecutionModel]] = None,
    ):
        self.state = state
        sizing_params = sizing_params or {}
        if sizing_fn is not None:
            self.sizing = sizing_fn(**sizing_params)
        else:
            self.sizing = FractionalSizing()
        
        if execution_model is not None:
            self.execution = execution_model()
        else:
            self.execution = OhlcAvgExecution()
    
    def apply(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        if len(data) < 2:
            return None
        
        agent_output = data.iloc[[-2]] if len(data) >= 2 else data
        price_data = data.iloc[[-1]]
        
        price = self._extract_price(price_data)
        
        sizing_output = pd.DataFrame({
            "quantity": [0],
            "action": [int(agent_output["action"].iloc[-1])],
            "asset": [str(agent_output["asset"].iloc[-1])],
            "confidence": [float(agent_output["confidence"].iloc[-1])],
        })
        
        sizing_output["quantity"] = self.sizing.compute(
            agent_output, self.state, price
        )
        
        sizing_output = self.execution.calculate(sizing_output, price_data)
        
        quantity = int(sizing_output["quantity"].iloc[-1])
        action = int(sizing_output["action"].iloc[-1])
        asset = str(sizing_output["asset"].iloc[-1])
        execution_cost = float(sizing_output["execution_cost"].iloc[-1])
        
        if quantity != 0:
            self.state.update_holding(asset, quantity)
            if action == 0:
                self.state.cash -= execution_cost
            elif action == 2:
                self.state.cash += execution_cost
        
        return sizing_output
    
    def _extract_price(self, price_data: pd.DataFrame) -> float:
        close_col = [c for c in price_data.columns if c.endswith('.close')][0]
        return float(price_data[close_col].iloc[-1])
