"""
Backtest Engine for TradSL

Section 15: Execution modes, transaction costs.
"""
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger("tradsl.backtest")

from tradsl.statistics import compute_metrics, PerformanceMetrics


class ExecutionMode(Enum):
    """Backtest execution mode."""
    TRAINING = "training"
    TEST = "test"
    LIVE = "live"


@dataclass
class TransactionCosts:
    """Transaction cost model parameters."""
    commission_rate: float = 0.001
    slippage: float = 0.0
    market_impact_coeff: float = 0.0
    
    def compute_commission(
        self,
        quantity: float,
        price: float
    ) -> float:
        """Calculate commission cost."""
        return abs(quantity * price) * self.commission_rate
    
    def compute_slippage(
        self,
        price: float,
        side: str
    ) -> float:
        """Calculate slippage cost."""
        if side == "buy":
            return price * self.slippage
        else:
            return -price * self.slippage
    
    def compute_market_impact(
        self,
        quantity: float,
        price: float,
        adv: float
    ) -> float:
        """Calculate market impact cost."""
        if adv <= 0 or self.market_impact_coeff == 0:
            return 0.0
        return self.market_impact_coeff * np.sqrt(abs(quantity) / adv) * price
    
    def total_cost(
        self,
        quantity: float,
        price: float,
        side: str,
        adv: float = 0.0
    ) -> float:
        """Calculate total transaction cost."""
        commission = self.compute_commission(quantity, price)
        slippage = self.compute_slippage(price, side)
        impact = self.compute_market_impact(quantity, price, adv)
        
        return commission + abs(slippage) + abs(impact)


@dataclass
class Trade:
    """Record of a single trade."""
    timestamp: int
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    slippage: float
    pnl: float = 0.0
    holding_period: int = 0


@dataclass
class BacktestResult:
    """Result of a backtest run."""
    equity_curve: np.ndarray
    trades: List[Trade]
    metrics: PerformanceMetrics
    mode: ExecutionMode
    
    def summary(self) -> str:
        """Generate summary."""
        m = self.metrics
        return f"""
Backtest Results ({self.mode.value})
{'=' * 40}
Total Return: {m.total_return:.2%}
CAGR: {m.cagr:.2%}
Sharpe: {m.sharpe_ratio:.3f}
Sortino: {m.sortino_ratio:.3f}
Calmar: {m.calmar_ratio:.3f}
Max Drawdown: {m.max_drawdown:.2%}
Win Rate: {m.win_rate:.2%}
Profit Factor: {m.profit_factor:.3f}
Trades: {m.n_trades}
"""


class BacktestEngine:
    """
    Backtest engine with support for training, test, and live modes.
    
    Section 15.1: Execution modes:
    - Training: Portfolio resets at block boundaries, shuffling enabled
    - Test: No portfolio reset, sequential execution
    - Live: No portfolio reset, sequential execution
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        agent,
        sizer,
        reward_function,
        transaction_costs: Optional[TransactionCosts] = None
    ):
        self.config = config
        self.agent = agent
        self.sizer = sizer
        self.reward_function = reward_function
        self.transaction_costs = transaction_costs or TransactionCosts(
            commission_rate=config.get('commission', 0.001),
            slippage=config.get('slippage', 0.0),
            market_impact_coeff=config.get('market_impact_coeff', 0.0)
        )
        
        self.mode = ExecutionMode.TRAINING if config.get('training_mode') == 'random_blocks' else ExecutionMode.TEST
        
        self._reset()
    
    def _reset(self) -> None:
        """Reset backtest state."""
        self.equity_curve = []
        self.trades = []
        self.portfolio_value = self.config.get('capital', 100000.0)
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_time = 0
        self.high_water_mark = self.portfolio_value
        self.drawdown = 0.0
    
    def run(
        self,
        data: np.ndarray,
        execute_bar_fn: Callable[[int, np.ndarray], Dict[str, Any]]
    ) -> BacktestResult:
        """
        Run backtest.
        
        Args:
            data: Price/feature data
            execute_bar_fn: Function to execute strategy logic
                            Must return portfolio_state dict with:
                            - action: TradingAction (BUY, SELL, HOLD, FLATTEN)
                            - quantity: float
                            - price: float (current price)
                            - symbol: str
                            - portfolio_value: float
        
        Returns:
            BacktestResult with equity curve, trades, metrics
        """
        from tradsl.models import TradingAction
        
        self._reset()
        
        for bar_idx in range(len(data)):
            portfolio_state = execute_bar_fn(bar_idx, data[bar_idx])
            
            action = portfolio_state.get('action', TradingAction.HOLD)
            current_price = portfolio_state.get('price', data[bar_idx][-1] if len(data[bar_idx]) > 0 else 100.0)
            symbol = portfolio_state.get('symbol', 'UNKNOWN')
            quantity = portfolio_state.get('quantity', 0.0)
            
            if action == TradingAction.BUY and quantity > 0:
                self.execute_trade(
                    symbol=symbol,
                    side="buy",
                    quantity=quantity,
                    price=current_price,
                    timestamp=bar_idx
                )
            elif action == TradingAction.SELL and quantity > 0:
                self.execute_trade(
                    symbol=symbol,
                    side="sell",
                    quantity=quantity,
                    price=current_price,
                    timestamp=bar_idx
                )
            elif action == TradingAction.FLATTEN:
                if self.position != 0:
                    self.execute_trade(
                        symbol=symbol,
                        side="sell" if self.position > 0 else "buy",
                        quantity=abs(self.position),
                        price=current_price,
                        timestamp=bar_idx
                    )
            
            self.equity_curve.append(self.portfolio_value)
            
            if self.portfolio_value > self.high_water_mark:
                self.high_water_mark = self.portfolio_value
            
            if self.high_water_mark > 0:
                current_dd = (self.high_water_mark - self.portfolio_value) / self.high_water_mark
                self.drawdown = max(self.drawdown, current_dd)
        
        metrics = compute_metrics(
            np.array(self.equity_curve),
            [t.__dict__ for t in self.trades],
            self.config.get('frequency', '1d')
        )
        
        return BacktestResult(
            equity_curve=np.array(self.equity_curve),
            trades=self.trades,
            metrics=metrics,
            mode=self.mode
        )
    
    def execute_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: int,
        adv: float = 1_000_000
    ) -> Trade:
        """Execute a trade with transaction costs."""
        cost = self.transaction_costs.total_cost(quantity, price, side, adv)
        
        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            commission=self.transaction_costs.compute_commission(quantity, price),
            slippage=self.transaction_costs.compute_slippage(price, side)
        )
        
        if side == "buy":
            self.portfolio_value -= (quantity * price + trade.commission + abs(trade.slippage))
            self.position += quantity
            if self.position == quantity:
                self.entry_price = price
                self.entry_time = timestamp
        else:
            self.portfolio_value += (quantity * price - trade.commission - abs(trade.slippage))
            if self.position != 0:
                pnl = (price - self.entry_price) * quantity
                trade.pnl = pnl - trade.commission - abs(trade.slippage)
            self.position -= quantity
            if abs(self.position) < 1e-9:
                self.position = 0.0
                self.entry_price = 0.0
        
        self.trades.append(trade)
        
        return trade
    
    def set_mode(self, mode: ExecutionMode) -> None:
        """Set execution mode."""
        self.mode = mode
        
        if mode == ExecutionMode.TRAINING:
            self._reset()
