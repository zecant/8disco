"""
Execution abstraction layer for backtesting.

Provides interfaces for order execution with different backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    timestamp: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED
    
    def is_pending(self) -> bool:
        return self.status == OrderStatus.PENDING
    
    def __repr__(self) -> str:
        return f"Order({self.order_id}: {self.side.value} {self.quantity} {self.symbol} @ {self.limit_price or 'MKT'} [{self.status.value}])"


@dataclass 
class Fill:
    """Represents a trade fill."""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float = 0.0
    slippage: float = 0.0
    timestamp: Optional[Any] = None
    
    @property
    def total_cost(self) -> float:
        """Total cost including commission and slippage."""
        cost = self.quantity * self.price
        if self.side == OrderSide.BUY:
            return cost + self.commission + self.slippage
        return cost - self.commission - self.slippage
    
    def __repr__(self) -> str:
        return f"Fill({self.fill_id}: {self.side.value} {self.quantity} {self.symbol} @ {self.price})"


class ExecutionBackend(ABC):
    """
    Abstract backend for order execution.
    
    Different implementations can simulate fills differently:
    - Immediate execution (market orders)
    - Queue-based (limit orders)
    - NautilusTrader integration
    """
    
    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        """
        Submit an order for execution.
        
        Args:
            order: Order to submit
        
        Returns:
            Updated order with fill information
        """
        pass
    
    @abstractmethod
    def get_fills(self) -> List[Fill]:
        """Get all fills since last check."""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset the execution backend."""
        pass


class ImmediateExecutionBackend(ExecutionBackend):
    """
    Simple immediate execution backend.
    
    Fills market orders immediately at current price.
    Supports basic slippage modeling.
    """
    
    def __init__(
        self,
        commission_rate: float = 0.0,
        slippage_bps: float = 0.0,  # Basis points
        slippage_model: str = 'fixed'  # 'fixed' or 'random'
    ):
        """
        Initialize execution backend.
        
        Args:
            commission_rate: Commission as fraction (e.g., 0.001 = 0.1%)
            slippage_bps: Slippage in basis points (e.g., 10 = 0.1%)
            slippage_model: 'fixed' or 'random'
        """
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        self.slippage_model = slippage_model
        
        self._pending_orders: Dict[str, Order] = {}
        self._fills: List[Fill] = []
        self._order_counter = 0
    
    def submit_order(self, order: Order) -> Order:
        """Execute order immediately at given price."""
        self._order_counter += 1
        
        if order.order_id is None or order.order_id == "":
            order.order_id = f"ORD_{self._order_counter}"
        
        if order.order_type == OrderType.MARKET:
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_price = order.metadata.get('current_price', 0.0)
            
            fill = self._create_fill(order)
            self._fills.append(fill)
        
        elif order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                order.status = OrderStatus.REJECTED
            else:
                current_price = order.metadata.get('current_price', 0.0)
                can_fill = (
                    (order.side == OrderSide.BUY and current_price <= order.limit_price) or
                    (order.side == OrderSide.SELL and current_price >= order.limit_price)
                )
                
                if can_fill:
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = order.quantity
                    order.filled_price = order.limit_price
                    
                    fill = self._create_fill(order)
                    self._fills.append(fill)
                else:
                    self._pending_orders[order.order_id] = order
        
        return order
    
    def _create_fill(self, order: Order) -> Fill:
        """Create a fill from an order."""
        price = order.filled_price
        
        slippage = 0.0
        if self.slippage_bps > 0:
            if self.slippage_model == 'fixed':
                slippage = price * (self.slippage_bps / 10000)
            else:
                import numpy as np
                slippage = price * (np.random.uniform(0, self.slippage_bps) / 10000)
        
        commission = price * order.filled_quantity * self.commission_rate
        
        return Fill(
            fill_id=f"FILL_{len(self._fills) + 1}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.filled_quantity,
            price=price,
            commission=commission,
            slippage=slippage,
            timestamp=order.timestamp
        )
    
    def get_fills(self) -> List[Fill]:
        """Get and clear fills."""
        fills = list(self._fills)
        self._fills.clear()
        return fills
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if order_id in self._pending_orders:
            order = self._pending_orders.pop(order_id)
            order.status = OrderStatus.CANCELLED
            return True
        return False
    
    def reset(self):
        """Reset the backend."""
        self._pending_orders.clear()
        self._fills.clear()
        self._order_counter = 0


class SimulationExecutionBackend(ExecutionBackend):
    """
    More sophisticated simulation backend.
    
    Supports:
    - Limit orders with queue
    - Market impact modeling
    - Fill probability modeling
    """
    
    def __init__(
        self,
        commission_rate: float = 0.0,
        slippage_bps: float = 0.0,
        fill_probability: float = 1.0,
        market_impact: float = 0.0
    ):
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        self.fill_probability = fill_probability
        self.market_impact = market_impact
        
        self._pending_orders: Dict[str, Order] = {}
        self._fills: List[Fill] = []
        self._order_counter = 0
        self._fill_counter = 0
    
    def submit_order(self, order: Order) -> Order:
        """Submit order (market or limit)."""
        self._order_counter += 1
        
        if order.order_id is None or order.order_id == "":
            order.order_id = f"ORD_{self._order_counter}"
        
        if order.order_type == OrderType.MARKET:
            return self._execute_market_order(order)
        else:
            return self._execute_limit_order(order)
    
    def _execute_market_order(self, order: Order) -> Order:
        """Execute market order with probability."""
        import numpy as np
        
        current_price = order.metadata.get('current_price', 0.0)
        
        if np.random.random() > self.fill_probability:
            order.status = OrderStatus.REJECTED
            return order
        
        executed_price = current_price * (1 + self.slippage_bps / 10000)
        
        if order.side == OrderSide.BUY:
            executed_price *= (1 + self.market_impact)
        else:
            executed_price *= (1 - self.market_impact)
        
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = executed_price
        
        fill = self._create_fill(order)
        self._fills.append(fill)
        
        return order
    
    def _execute_limit_order(self, order: Order) -> Order:
        """Execute or queue limit order."""
        import numpy as np
        
        current_price = order.metadata.get('current_price', 0.0)
        
        if order.limit_price is None:
            order.status = OrderStatus.REJECTED
            return order
        
        can_fill = (
            (order.side == OrderSide.BUY and current_price <= order.limit_price) or
            (order.side == OrderSide.SELL and current_price >= order.limit_price)
        )
        
        if can_fill and np.random.random() <= self.fill_probability:
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_price = order.limit_price
            
            fill = self._create_fill(order)
            self._fills.append(fill)
        else:
            self._pending_orders[order.order_id] = order
        
        return order
    
    def _create_fill(self, order: Order) -> Fill:
        """Create fill from order."""
        price = order.filled_price
        slippage = price * (self.slippage_bps / 10000)
        commission = price * order.filled_quantity * self.commission_rate
        
        self._fill_counter += 1
        
        return Fill(
            fill_id=f"FILL_{self._fill_counter}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.filled_quantity,
            price=price,
            commission=commission,
            slippage=slippage,
            timestamp=order.timestamp
        )
    
    def get_fills(self) -> List[Fill]:
        """Get and clear fills."""
        fills = list(self._fills)
        self._fills.clear()
        return fills
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        if order_id in self._pending_orders:
            order = self._pending_orders.pop(order_id)
            order.status = OrderStatus.CANCELLED
            return True
        return False
    
    def check_pending_orders(self, current_prices: Dict[str, float]):
        """Check if pending orders can be filled at current prices."""
        filled_orders = []
        
        for order_id, order in list(self._pending_orders.items()):
            current_price = current_prices.get(order.symbol, 0.0)
            
            can_fill = (
                (order.side == OrderSide.BUY and current_price <= order.limit_price) or
                (order.side == OrderSide.SELL and current_price >= order.limit_price)
            )
            
            if can_fill:
                order.filled_price = order.limit_price
                order.status = OrderStatus.FILLED
                
                fill = self._create_fill(order)
                self._fills.append(fill)
                filled_orders.append(order_id)
        
        for order_id in filled_orders:
            del self._pending_orders[order_id]
    
    def reset(self):
        """Reset backend state."""
        self._pending_orders.clear()
        self._fills.clear()
        self._order_counter = 0
        self._fill_counter = 0


def create_execution_backend(
    backend_type: str = 'immediate',
    **params
) -> ExecutionBackend:
    """
    Factory to create execution backend.
    
    Args:
        backend_type: 'immediate' or 'simulation'
        **params: Backend parameters
    
    Returns:
        ExecutionBackend instance
    """
    if backend_type == 'immediate':
        return ImmediateExecutionBackend(**params)
    elif backend_type == 'simulation':
        return SimulationExecutionBackend(**params)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
