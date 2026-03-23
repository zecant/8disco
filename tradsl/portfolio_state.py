from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PortfolioState:
    cash: float
    currency: str = "USD"
    holdings: dict[str, int] = field(default_factory=dict)

    def update_holding(self, symbol: str, quantity: int) -> None:
        current = self.holdings.get(symbol, 0)
        self.holdings[symbol] = current + quantity

    def get_holding(self, symbol: str) -> int:
        return self.holdings.get(symbol, 0)

    def reset(self, initial_cash: float, symbols: list[str]) -> None:
        self.cash = initial_cash
        self.holdings = {s: 0 for s in symbols}
