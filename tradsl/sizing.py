"""
Position sizing abstractions for trading strategies.

Provides standardized interfaces for calculating position sizes
from trading signals and portfolio state.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np

from tradsl.signals import TradingSignal, SignalBatch, TradingAction


@dataclass
class Allocation:
    """Represents a position allocation for a single symbol."""
    symbol: str
    weight: float           # Weight of portfolio (0.0 to 1.0)
    target_value: float     # Dollar value to allocate
    target_quantity: float  # Number of units (shares/contracts)
    action: TradingAction   # BUY, SELL, or HOLD
    
    def __post_init__(self):
        self.weight = max(0.0, min(1.0, self.weight))


@dataclass
class AllocationPlan:
    """Collection of allocations for all symbols."""
    allocations: Dict[str, Allocation] = field(default_factory=dict)
    total_weight: float = 0.0
    timestamp: Optional[Any] = None
    
    def add(self, allocation: Allocation):
        """Add an allocation."""
        self.allocations[allocation.symbol] = allocation
        self._recalculate_total()
    
    def get(self, symbol: str) -> Optional[Allocation]:
        """Get allocation for a symbol."""
        return self.allocations.get(symbol)
    
    def get_actionable(self) -> List[Allocation]:
        """Get allocations that require action (buy/sell, not hold)."""
        return [
            a for a in self.allocations.values() 
            if a.action in (TradingAction.BUY, TradingAction.SELL)
        ]
    
    def get_weights(self) -> Dict[str, float]:
        """Get weights dict for all symbols."""
        return {sym: alloc.weight for sym, alloc in self.allocations.items()}
    
    def get_target_values(self) -> Dict[str, float]:
        """Get target values dict for all symbols."""
        return {sym: alloc.target_value for sym, alloc in self.allocations.items()}
    
    def get_target_quantities(self) -> Dict[str, float]:
        """Get target quantities dict for all symbols."""
        return {sym: alloc.target_quantity for sym, alloc in self.allocations.items()}
    
    def _recalculate_total(self):
        """Recalculate total weight."""
        self.total_weight = sum(a.weight for a in self.allocations.values())
    
    def __len__(self) -> int:
        return len(self.allocations)
    
    def __iter__(self):
        return iter(self.allocations.values())


class PositionSizer(ABC):
    """
    Abstract base class for position sizers.
    
    A position sizer determines how much capital to allocate to each position
    based on trading signals and current portfolio state.
    """
    
    @abstractmethod
    def calculate(
        self, 
        signals: SignalBatch, 
        portfolio_value: float,
        current_positions: Optional[Dict[str, float]] = None,
        prices: Optional[Dict[str, float]] = None
    ) -> AllocationPlan:
        """
        Calculate position allocations from signals.
        
        Args:
            signals: SignalBatch with TradingSignals for each symbol
            portfolio_value: Total portfolio value (cash + positions)
            current_positions: Current position sizes in units {symbol: quantity}
            prices: Current prices {symbol: price}
        
        Returns:
            AllocationPlan with target allocations
        """
        pass
    
    def validate_allocation(self, allocation: AllocationPlan, portfolio_value: float) -> bool:
        """
        Validate an allocation plan.
        
        Args:
            allocation: The allocation plan to validate
            portfolio_value: Total portfolio value
        
        Returns:
            True if valid, False otherwise
        """
        if abs(allocation.total_weight - 1.0) > 0.01:
            return False
        
        for alloc in allocation.allocations.values():
            if alloc.weight < 0 or alloc.weight > 1:
                return False
            if allocation.total_weight > 1.01:
                return False
        
        return True


class EqualWeightSizer(PositionSizer):
    """
    Equal weight allocation across all signals.
    
    Allocates equal weight to each symbol with an actionable signal.
    """
    
    def __init__(self, max_positions: int = 10, **kwargs):
        self.max_positions = max_positions
    
    def calculate(
        self, 
        signals: SignalBatch, 
        portfolio_value: float,
        current_positions: Optional[Dict[str, float]] = None,
        prices: Optional[Dict[str, float]] = None
    ) -> AllocationPlan:
        actionable = signals.get_actionable()
        
        if not actionable:
            return AllocationPlan()
        
        n_positions = min(len(actionable), self.max_positions)
        weight_per_position = 1.0 / n_positions
        
        plan = AllocationPlan()
        
        for symbol, signal in actionable.items():
            action = signal.action
            
            if action == TradingAction.HOLD:
                # Keep current position
                weight = current_positions.get(symbol, 0.0) / portfolio_value if portfolio_value > 0 else 0
            else:
                weight = weight_per_position
            
            target_value = weight * portfolio_value
            target_quantity = 0.0
            if prices and symbol in prices and prices[symbol] > 0:
                target_quantity = target_value / prices[symbol]
            
            allocation = Allocation(
                symbol=symbol,
                weight=weight,
                target_value=target_value,
                target_quantity=target_quantity,
                action=action
            )
            plan.add(allocation)
        
        return plan


class FixedFractionSizer(PositionSizer):
    """
    Fixed fraction position sizer.
    
    Allocates a fixed fraction of portfolio to each position.
    """
    
    def __init__(self, fraction: float = 0.1, **kwargs):
        if fraction <= 0 or fraction > 1:
            raise ValueError("fraction must be between 0 and 1")
        self.fraction = fraction
    
    def calculate(
        self, 
        signals: SignalBatch, 
        portfolio_value: float,
        current_positions: Optional[Dict[str, float]] = None,
        prices: Optional[Dict[str, float]] = None
    ) -> AllocationPlan:
        actionable = signals.get_actionable()
        
        plan = AllocationPlan()
        
        if current_positions is None:
            current_positions = {}
        
        for symbol, signal in actionable.items():
            action = signal.action
            
            if action == TradingAction.HOLD:
                current_qty = current_positions.get(symbol, 0)
                current_value = current_qty * prices.get(symbol, 0) if prices and symbol in prices else 0
                weight = current_value / portfolio_value if portfolio_value > 0 else 0
            else:
                weight = self.fraction
            
            target_value = weight * portfolio_value
            target_quantity = 0.0
            if prices and symbol in prices and prices[symbol] > 0:
                target_quantity = target_value / prices[symbol]
            
            allocation = Allocation(
                symbol=symbol,
                weight=weight,
                target_value=target_value,
                target_quantity=target_quantity,
                action=action
            )
            plan.add(allocation)
        
        return plan


class KellySizer(PositionSizer):
    """
    Kelly Criterion position sizer.
    
    Sizes positions based on Kelly formula: f* = (bp - q) / b
    where b = odds received, p = probability of win, q = probability of loss
    
    Simplified version using confidence as probability estimate.
    """
    
    def __init__(
        self, 
        kelly_fraction: float = 1.0,
        max_kelly: float = 0.25,
        **kwargs
    ):
        self.kelly_fraction = kelly_fraction  # Kelly fraction (0.5 = half Kelly)
        self.max_kelly = max_kelly           # Cap on Kelly allocation
    
    def calculate(
        self, 
        signals: SignalBatch, 
        portfolio_value: float,
        current_positions: Optional[Dict[str, float]] = None,
        prices: Optional[Dict[str, float]] = None
    ) -> AllocationPlan:
        actionable = signals.get_actionable()
        
        if not actionable:
            return AllocationPlan()
        
        plan = AllocationPlan()
        
        kelly_weights = {}
        
        for symbol, signal in actionable.items():
            if signal.action == TradingAction.HOLD:
                continue
            
            confidence = signal.confidence
            
            # Kelly formula simplified: f = 2*p - 1 (for symmetric payoff)
            # Using confidence as probability estimate
            kelly = (2 * confidence - 1) * self.kelly_fraction
            
            # Apply cap
            kelly = min(kelly, self.max_kelly)
            kelly = max(kelly, 0)  # No negative positions
            
            kelly_weights[symbol] = kelly
        
        # Normalize weights
        total_kelly = sum(kelly_weights.values())
        if total_kelly > 1.0:
            kelly_weights = {k: v / total_kelly for k, v in kelly_weights.items()}
        
        if current_positions is None:
            current_positions = {}
        
        for symbol, weight in kelly_weights.items():
            signal = actionable[symbol]
            action = signal.action
            
            target_value = weight * portfolio_value
            target_quantity = 0.0
            if prices and symbol in prices and prices[symbol] > 0:
                target_quantity = target_value / prices[symbol]
            
            allocation = Allocation(
                symbol=symbol,
                weight=weight,
                target_value=target_value,
                target_quantity=target_quantity,
                action=action
            )
            plan.add(allocation)
        
        return plan


class ConfidenceWeightedSizer(PositionSizer):
    """
    Position sizer that weights by signal confidence.
    
    Allocates proportionally to signal confidence.
    """
    
    def __init__(self, scale_factor: float = 1.0, **kwargs):
        self.scale_factor = scale_factor
    
    def calculate(
        self, 
        signals: SignalBatch, 
        portfolio_value: float,
        current_positions: Optional[Dict[str, float]] = None,
        prices: Optional[Dict[str, float]] = None
    ) -> AllocationPlan:
        actionable = signals.get_actionable()
        
        if not actionable:
            return AllocationPlan()
        
        # Weight by confidence
        weights = {}
        for symbol, signal in actionable.items():
            if signal.action == TradingAction.HOLD:
                continue
            weights[symbol] = signal.confidence * self.scale_factor
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        plan = AllocationPlan()
        
        if current_positions is None:
            current_positions = {}
        
        for symbol, weight in weights.items():
            signal = actionable[symbol]
            action = signal.action
            
            target_value = weight * portfolio_value
            target_quantity = 0.0
            if prices and symbol in prices and prices[symbol] > 0:
                target_quantity = target_value / prices[symbol]
            
            allocation = Allocation(
                symbol=symbol,
                weight=weight,
                target_value=target_value,
                target_quantity=target_quantity,
                action=action
            )
            plan.add(allocation)
        
        return plan


# Registry of available sizers
SIZER_REGISTRY = {
    'equal': EqualWeightSizer,
    'fixed_fraction': FixedFractionSizer,
    'fixed': FixedFractionSizer,
    'kelly': KellySizer,
    'confidence': ConfidenceWeightedSizer,
}


def create_sizer(sizer_type: str, **params) -> PositionSizer:
    """
    Factory function to create a position sizer.
    
    Args:
        sizer_type: Type name ('equal', 'fixed_fraction', 'kelly', 'confidence')
        **params: Parameters for the sizer
    
    Returns:
        PositionSizer instance
    
    Raises:
        ValueError: If sizer_type is unknown
    """
    if sizer_type.lower() not in SIZER_REGISTRY:
        raise ValueError(
            f"Unknown sizer type: {sizer_type}. "
            f"Available: {list(SIZER_REGISTRY.keys())}"
        )
    
    return SIZER_REGISTRY[sizer_type.lower()](**params)
