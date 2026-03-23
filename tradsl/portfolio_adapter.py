from datetime import datetime
from typing import Optional
import pandas as pd

from tradsl.adapters import Adapter
from tradsl.portfolio_state import PortfolioState


class PortfolioAdapter(Adapter):
    def __init__(
        self,
        symbols: list[str],
        initial_cash: float = 10000.0,
        currency: str = "USD",
        state: Optional[PortfolioState] = None,
        dag=None,
    ):
        super().__init__(dag=dag)
        self.symbols = symbols
        self.initial_cash = initial_cash
        self.currency = currency
        self.state = state if state is not None else PortfolioState(
            cash=initial_cash,
            currency=currency,
            holdings={s: 0 for s in symbols},
        )
        self._started = False

    def set_start(self, start_time: datetime) -> None:
        self.state.reset(self.initial_cash, self.symbols)
        self._started = True

    def tick(self) -> Optional[pd.DataFrame]:
        if not self._started:
            return None
        
        data = {"cash": [self.state.cash]}
        
        for symbol in self.symbols:
            col_name = f"{symbol}_holding"
            data[col_name] = [self.state.get_holding(symbol)]
        
        df = pd.DataFrame(data)
        if self.dag is not None and self.dag._current_timestamp is not None:
            df.index = pd.DatetimeIndex([self.dag._current_timestamp])
        return df

    def reset(self) -> None:
        self.state.reset(self.initial_cash, self.symbols)
        self._started = False
