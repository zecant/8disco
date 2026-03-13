"""
Position Sizers for TradSL

Section 12: BasePositionSizer interface and implementations.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np
from dataclasses import dataclass


# Valid portfolio state keys (Section 12.3)
VALID_PORTFOLIO_KEYS = {
    'position', 'position_value', 'unrealized_pnl', 'unrealized_pnl_pct',
    'realized_pnl', 'portfolio_value', 'cash', 'drawdown', 'time_in_trade',
    'high_water_mark', 'position_weight', 'rolling_sharpe', 'rolling_volatility'
}


@dataclass
class SizerSpec:
    """Specification for a registered sizer."""
    cls: type
    description: str


class BasePositionSizer(ABC):
    """
    Section 12.1: Position sizer interface.
    
    Calculates target position size from action, conviction,
    and portfolio state.
    """
    
    @abstractmethod
    def calculate_size(
        self,
        action,
        conviction: float,
        current_position: float,
        portfolio_value: float,
        instrument_id: str,
        current_price: float
    ) -> float:
        """
        Calculate target position size in units.
        
        Args:
            action: TradingAction enum or continuous float
            conviction: Confidence in [0, 1]
            current_position: Current position in units
            portfolio_value: Total portfolio value in base currency
            instrument_id: Instrument identifier
            current_price: Current market price
        
        Returns:
            Target position in units (+ long, - short, 0 flat)
        """
        pass
    
    def reset(self) -> None:
        """Reset sizer state at block boundaries. Override if needed."""
        pass


class FixedSizer(BasePositionSizer):
    """Fixed number of units per trade."""
    
    def __init__(self, quantity: int = 1):
        self.quantity = quantity
    
    def calculate_size(
        self,
        action,
        conviction: float,
        current_position: float,
        portfolio_value: float,
        instrument_id: str,
        current_price: float
    ) -> float:
        from tradsl.models import TradingAction
        
        if action == TradingAction.FLATTEN:
            return 0.0
        if action == TradingAction.HOLD:
            return current_position
        
        direction = 1.0
        if action == TradingAction.SELL:
            direction = -1.0
        
        return float(self.quantity) * direction


class FractionalSizer(BasePositionSizer):
    """
    Fractional position sizer.
    
    Scales position by a fixed fraction of portfolio value.
    
    Args:
        fraction: Fraction of portfolio per trade (e.g., 0.1 = 10%)
        max_position: Maximum position as fraction of portfolio
    """
    
    def __init__(
        self,
        fraction: float = 0.1,
        max_position: float = 1.0,
        min_position: float = 0.0
    ):
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if not 0 < max_position <= 1:
            raise ValueError(f"max_position must be in (0, 1], got {max_position}")
        
        self.fraction = fraction
        self.max_position = max_position
        self.min_position = min_position
    
    def calculate_size(
        self,
        action,
        conviction: float,
        current_position: float,
        portfolio_value: float,
        instrument_id: str,
        current_price: float
    ) -> float:
        from tradsl.models import TradingAction
        
        if portfolio_value <= 0:
            return 0.0
        
        if action == TradingAction.FLATTEN:
            return 0.0
        if action == TradingAction.HOLD:
            return current_position
        
        direction = 1.0
        if action == TradingAction.SELL:
            direction = -1.0
        
        target_value = portfolio_value * self.fraction * conviction
        target_value = min(target_value, portfolio_value * self.max_position)
        target_value = max(target_value, portfolio_value * self.min_position)
        
        target_units = target_value / current_price
        
        return direction * target_units


class KellySizer(BasePositionSizer):
    """
    Kelly criterion position sizer.
    
    f* = (p * b - q) / b
    where p = win probability, q = 1-p, b = win/loss ratio
    
    Args:
        max_kelly: Maximum Kelly fraction to use (default 0.25)
        estimated_win_rate: Prior win probability estimate
    """
    
    def __init__(
        self,
        max_kelly: float = 0.25,
        estimated_win_rate: float = 0.5,
        win_loss_ratio: float = 2.0
    ):
        if not 0 < max_kelly <= 1:
            raise ValueError(f"max_kelly must be in (0, 1], got {max_kelly}")
        
        self.max_kelly = max_kelly
        self.estimated_win_rate = estimated_win_rate
        self.win_loss_ratio = win_loss_ratio
    
    def calculate_size(
        self,
        action,
        conviction: float,
        current_position: float,
        portfolio_value: float,
        instrument_id: str,
        current_price: float
    ) -> float:
        from tradsl.models import TradingAction
        
        if portfolio_value <= 0:
            return 0.0
        
        if action == TradingAction.FLATTEN:
            return 0.0
        if action == TradingAction.HOLD:
            return current_position
        if action == TradingAction.HOLD:
            return current_position
        
        p = conviction * self.estimated_win_rate
        q = 1 - p
        b = self.win_loss_ratio
        
        kelly = (p * b - q) / b
        kelly = max(0, min(kelly, self.max_kelly))
        
        direction = 1.0
        if action == TradingAction.SELL:
            direction = -1.0
        
        target_value = portfolio_value * kelly
        target_units = target_value / current_price
        
        return direction * target_units


class VolatilityTargetingSizer(BasePositionSizer):
    """
    Volatility-targeting sizer.
    
    Scales position inversely to realized volatility
    to maintain constant portfolio volatility.
    
    Args:
        target_vol: Target annualized volatility (e.g., 0.15 = 15%)
    """
    
    def __init__(self, target_vol: float = 0.15):
        if target_vol <= 0:
            raise ValueError(f"target_vol must be positive, got {target_vol}")
        
        self.target_vol = target_vol
    
    def calculate_size(
        self,
        action,
        conviction: float,
        current_position: float,
        portfolio_value: float,
        instrument_id: str,
        current_price: float,
        realized_vol: float = None
    ) -> float:
        from tradsl.models import TradingAction
        
        if portfolio_value <= 0:
            return 0.0
        
        if action == TradingAction.FLATTEN:
            return 0.0
        if action == TradingAction.HOLD:
            return current_position
        
        direction = 1.0
        if action == TradingAction.SELL:
            direction = -1.0
        
        vol = realized_vol if realized_vol else self.target_vol
        
        if vol <= 0:
            vol = self.target_vol
        
        vol_scalar = self.target_vol / vol
        vol_scalar = min(vol_scalar, 2.0)
        
        target_value = portfolio_value * vol_scalar * conviction
        target_units = target_value / current_price
        
        return direction * target_units


class MaxDrawdownSizer(BasePositionSizer):
    """
    Max drawdown-reducing sizer.
    
    Reduces position size linearly as portfolio drawdown grows from 0 to max_drawdown.
    At max_drawdown, position is reduced to zero.
    
    Args:
        max_drawdown: Maximum drawdown threshold (e.g., 0.2 = 20%)
    """
    
    def __init__(self, max_drawdown: float = 0.2):
        if not 0 < max_drawdown <= 1:
            raise ValueError(f"max_drawdown must be in (0, 1], got {max_drawdown}")
        
        self.max_drawdown = max_drawdown
    
    def calculate_size(
        self,
        action,
        conviction: float,
        current_position: float,
        portfolio_value: float,
        instrument_id: str,
        current_price: float,
        current_drawdown: float = 0.0
    ) -> float:
        from tradsl.models import TradingAction
        
        if portfolio_value <= 0:
            return 0.0
        
        if action == TradingAction.FLATTEN:
            return 0.0
        if action == TradingAction.HOLD:
            return current_position
        
        direction = 1.0
        if action == TradingAction.SELL:
            direction = -1.0
        
        dd_fraction = min(current_drawdown / self.max_drawdown, 1.0)
        size_reduction = 1.0 - dd_fraction
        
        base_size = portfolio_value * 0.1 * conviction * size_reduction
        target_units = base_size / current_price
        
        return direction * target_units


class EnsembleSizer(BasePositionSizer):
    """
    Ensemble position sizer.
    
    Combines multiple sizers with weighted average.
    Weights must sum to 1.0.
    
    Args:
        sizers: List of (sizer_instance, weight) tuples
    """
    
    def __init__(self, sizers: list):
        if not sizers:
            raise ValueError("At least one sizer required")
        
        total_weight = sum(w for _, w in sizers)
        if abs(total_weight - 1.0) > 1e-9:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        self.sizers = sizers
    
    def calculate_size(
        self,
        action,
        conviction: float,
        current_position: float,
        portfolio_value: float,
        instrument_id: str,
        current_price: float,
        **kwargs
    ) -> float:
        total_size = 0.0
        
        for sizer, weight in self.sizers:
            size = sizer.calculate_size(
                action, conviction, current_position, portfolio_value,
                instrument_id, current_price, **kwargs
            )
            total_size += weight * size
        
        return total_size


# Sizer registry
SIZER_REGISTRY: Dict[str, SizerSpec] = {
    'fixed': SizerSpec(
        cls=FixedSizer,
        description="Fixed number of units per trade"
    ),
    'fractional': SizerSpec(
        cls=FractionalSizer,
        description="Fixed fraction of portfolio per trade"
    ),
    'kelly': SizerSpec(
        cls=KellySizer,
        description="Kelly criterion sizing"
    ),
    'volatility_targeting': SizerSpec(
        cls=VolatilityTargetingSizer,
        description="Volatility-targeted sizing"
    ),
    'max_drawdown': SizerSpec(
        cls=MaxDrawdownSizer,
        description="Reduces position as drawdown grows"
    ),
    'ensemble': SizerSpec(
        cls=EnsembleSizer,
        description="Weighted combination of other sizers"
    ),
}


def get_sizer(name: str) -> Optional[type]:
    """Get sizer class by name."""
    spec = SIZER_REGISTRY.get(name)
    return spec.cls if spec else None
