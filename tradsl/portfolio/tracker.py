"""
Portfolio tracking for backtesting.

Manages positions, tracks equity, and generates portfolio features.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd


@dataclass
class Position:
    """Represents a single position in a symbol."""
    symbol: str
    quantity: float = 0.0          # Number of units (shares/contracts)
    entry_price: float = 0.0      # Average entry price
    current_price: float = 0.0    # Current market price
    
    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Total cost basis of position."""
        return self.quantity * self.entry_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        return (self.current_price - self.entry_price) * self.quantity
    
    @property
    def realized_pnl(self) -> float:
        """Realized profit/loss (tracked separately)."""
        return 0.0  # Would need to track this separately
    
    @property
    def total_pnl(self) -> float:
        """Total PnL (realized + unrealized)."""
        return self.unrealized_pnl  # + self.realized_pnl
    
    @property
    def weight(self) -> float:
        """Weight of position in portfolio (0 to 1)."""
        return 0.0  # Calculated relative to total equity
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        return self.quantity == 0
    
    def update_price(self, price: float):
        """Update current price."""
        self.current_price = price
    
    def __repr__(self) -> str:
        return f"Position({self.symbol}: {self.quantity} @ {self.current_price}, PnL: {self.unrealized_pnl:.2f})"


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio state at a point in time."""
    timestamp: Any
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    starting_cash: float = 0.0
    
    @property
    def equity(self) -> float:
        """Total equity (cash + position values)."""
        return self.cash + self.position_value
    
    @property
    def position_value(self) -> float:
        """Total value of all positions."""
        return sum(p.market_value for p in self.positions.values())
    
    @property
    def total_pnl(self) -> float:
        """Total unrealized PnL across all positions."""
        return sum(p.unrealized_pnl for p in self.positions.values())
    
    @property
    def total_weight(self) -> float:
        """Total weight of all positions."""
        if self.equity == 0:
            return 0.0
        return self.position_value / self.equity
    
    @property
    def returns(self) -> float:
        """Total returns since start."""
        if self.starting_cash == 0:
            return 0.0
        return (self.equity - self.starting_cash) / self.starting_cash
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """Check if has position in symbol."""
        return symbol in self.positions and not self.positions[symbol].is_flat
    
    def to_features(self) -> Dict[str, float]:
        """
        Convert portfolio state to feature dict for model input.
        
        Returns features like:
        - portfolio_cash
        - portfolio_equity
        - portfolio_position_value
        - portfolio_total_pnl
        - portfolio_returns
        - position_{symbol}: quantity
        - position_value_{symbol}: market value
        - position_pnl_{symbol}: unrealized PnL
        """
        features = {
            'portfolio_cash': self.cash,
            'portfolio_equity': self.equity,
            'portfolio_position_value': self.position_value,
            'portfolio_total_pnl': self.total_pnl,
            'portfolio_total_weight': self.total_weight,
            'portfolio_returns': self.returns,
        }
        
        for symbol, pos in self.positions.items():
            features[f'position_{symbol}'] = pos.quantity
            features[f'position_value_{symbol}'] = pos.market_value
            features[f'position_pnl_{symbol}'] = pos.unrealized_pnl
            features[f'position_weight_{symbol}'] = pos.weight
        
        return features
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'cash': self.cash,
            'equity': self.equity,
            'position_value': self.position_value,
            'total_pnl': self.total_pnl,
            'returns': self.returns,
            'positions': {
                sym: {
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                }
                for sym, pos in self.positions.items()
            }
        }


class PortfolioTracker:
    """
    Tracks portfolio state through backtesting.
    
    Manages positions, calculates equity, tracks PnL,
    and generates portfolio features for model input.
    """
    
    def __init__(
        self,
        starting_cash: float,
        commission_rate: float = 0.001,
        slippage: float = 0.0,
        symbols: Optional[List[str]] = None
    ):
        """
        Initialize portfolio tracker.
        
        Args:
            starting_cash: Initial cash amount
            commission_rate: Commission rate per trade (e.g., 0.001 = 0.1%)
            slippage: Slippage per unit (added to costs)
            symbols: List of tradable symbols
        """
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.symbols = symbols or []
        
        self.positions: Dict[str, Position] = {
            sym: Position(symbol=sym) for sym in self.symbols
        }
        
        self.history: List[PortfolioSnapshot] = []
        self._trades: List[Dict[str, Any]] = []
        self._equity_curve: List[float] = []
    
    @property
    def equity(self) -> float:
        """Total portfolio equity."""
        return self.cash + self.position_value
    
    @property
    def position_value(self) -> float:
        """Total position market value."""
        return sum(p.market_value for p in self.positions.values())
    
    @property
    def total_pnl(self) -> float:
        """Total unrealized PnL."""
        return sum(p.unrealized_pnl for p in self.positions.values())
    
    @property
    def returns(self) -> float:
        """Total returns."""
        if self.starting_cash == 0:
            return 0.0
        return (self.equity - self.starting_cash) / self.starting_cash
    
    def get_snapshot(self, timestamp: Any = None) -> PortfolioSnapshot:
        """Get current portfolio snapshot."""
        return PortfolioSnapshot(
            timestamp=timestamp,
            cash=self.cash,
            positions=dict(self.positions),
            starting_cash=self.starting_cash
        )
    
    def record_snapshot(self, timestamp: Any = None):
        """Record a snapshot to history."""
        snapshot = self.get_snapshot(timestamp)
        self.history.append(snapshot)
        self._equity_curve.append(snapshot.equity)
    
    def update_prices(self, prices: Dict[str, float]):
        """
        Update current prices for all positions.
        
        Args:
            prices: Dict of {symbol: price}
        """
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)
            else:
                pos = Position(symbol=symbol, current_price=price)
                self.positions[symbol] = pos
    
    def execute_trade(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: Any = None
    ) -> Dict[str, float]:
        """
        Execute a trade.
        
        Args:
            symbol: Trading symbol
            quantity: Number of units (+ for buy, - for sell)
            price: Execution price
            timestamp: Trade timestamp
        
        Returns:
            Dict with trade details (cost, commission, slippage)
        """
        original_quantity = quantity
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        
        pos = self.positions[symbol]
        
        trade_value = abs(quantity * price)
        commission = trade_value * self.commission_rate
        slippage_cost = abs(quantity) * self.slippage
        total_cost = trade_value + commission + slippage_cost
        
        if quantity > 0:
            # BUY
            if self.cash < total_cost:
                # Not enough cash - scale down
                max_quantity = (self.cash - commission - slippage_cost) / (price + self.slippage)
                quantity = max(0, max_quantity)
                trade_value = abs(quantity * price)
                total_cost = trade_value + commission + slippage_cost
            
            self.cash -= total_cost
            
            if pos.quantity == 0:
                pos.entry_price = price
                pos.quantity = quantity
            else:
                # Weighted average entry price
                total_cost_basis = pos.cost_basis + trade_value
                pos.quantity += quantity
                pos.entry_price = total_cost_basis / pos.quantity if pos.quantity > 0 else 0
            
            pos.current_price = price
        
        else:
            # SELL
            quantity = abs(quantity)
            
            if pos.quantity < quantity:
                quantity = pos.quantity
            
            trade_value = quantity * price
            commission = trade_value * self.commission_rate
            slippage_cost = quantity * self.slippage
            
            self.cash += trade_value - commission - slippage_cost
            
            pos.quantity -= quantity
            if pos.quantity == 0:
                pos.entry_price = 0.0
            
            pos.current_price = price
        
        is_buy = original_quantity > 0
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'quantity': abs(original_quantity),
            'side': 'buy' if is_buy else 'sell',
            'price': price,
            'commission': commission,
            'slippage': slippage_cost,
            'total_cost': total_cost if is_buy else -(trade_value - commission - slippage_cost),
        }
        
        self._trades.append(trade_record)
        
        return trade_record
    
    def set_target_position(
        self,
        symbol: str,
        target_quantity: float,
        current_price: float,
        timestamp: Any = None
    ) -> Optional[Dict[str, float]]:
        """
        Set target position (convenience method).
        
        Args:
            symbol: Symbol to set position for
            target_quantity: Target quantity
            current_price: Current market price
            timestamp: Timestamp
        
        Returns:
            Trade record if trade executed, None otherwise
        """
        current_qty = self.positions.get(symbol, Position(symbol=symbol)).quantity
        
        delta = target_quantity - current_qty
        
        if abs(delta) < 1e-8:  # Floating point tolerance
            return None
        
        return self.execute_trade(symbol, delta, current_price, timestamp)
    
    def flatten_all(self, prices: Dict[str, float], timestamp: Any = None):
        """Close all positions at current prices."""
        trades = []
        for symbol, pos in self.positions.items():
            if not pos.is_flat:
                price = prices.get(symbol, pos.current_price)
                if price > 0:
                    trade = self.execute_trade(symbol, -pos.quantity, price, timestamp)
                    trades.append(trade)
        return trades
    
    def get_current_quantities(self) -> Dict[str, float]:
        """Get current position quantities."""
        return {
            sym: pos.quantity 
            for sym, pos in self.positions.items()
        }
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current position weights."""
        if self.equity == 0:
            return {}
        
        return {
            sym: pos.market_value / self.equity
            for sym, pos in self.positions.items()
            if not pos.is_flat
        }
    
    def get_trades(self) -> List[Dict[str, Any]]:
        """Get all trades."""
        return list(self._trades)
    
    def get_equity_curve(self) -> pd.Series:
        """Get equity curve as pandas Series."""
        return pd.Series(self._equity_curve)
    
    def get_history(self) -> List[PortfolioSnapshot]:
        """Get full snapshot history."""
        return list(self.history)
    
    def summary(self) -> Dict[str, Any]:
        """Get portfolio summary."""
        return {
            'starting_cash': self.starting_cash,
            'ending_cash': self.cash,
            'ending_equity': self.equity,
            'total_return': self.returns,
            'total_pnl': self.total_pnl,
            'position_value': self.position_value,
            'num_trades': len(self._trades),
            'num_positions': sum(1 for p in self.positions.values() if not p.is_flat),
        }
    
    def reset(self):
        """Reset portfolio to initial state."""
        self.cash = self.starting_cash
        self.positions = {sym: Position(symbol=sym) for sym in self.symbols}
        self.history.clear()
        self._trades.clear()
        self._equity_curve.clear()
