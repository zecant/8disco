"""
Signal abstractions for trading strategies.

Provides standardized signal types and structures for model outputs.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
import numpy as np


class SignalType(Enum):
    """Types of signals a model can produce."""
    ACTION = "action"       # Discrete buy/sell/hold
    THRESHOLD = "threshold" # Entry/exit levels
    CONTINUOUS = "continuous" # Raw continuous value


class TradingAction(Enum):
    """Discrete trading actions."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    FLATTEN = "flatten"


@dataclass
class TradingSignal:
    """
    Standardized trading signal from model prediction.
    
    This is the canonical signal format that all models should output
    and all sizers should accept.
    """
    signal_type: SignalType
    symbol: str
    
    # For ACTION type signals
    action: Optional[TradingAction] = None
    
    # For THRESHOLD type signals  
    entry_threshold: Optional[float] = None
    exit_threshold: Optional[float] = None
    
    # For CONTINUOUS type signals
    value: Optional[float] = None
    
    # Confidence/risk factor [0, 1] - used by position sizers
    confidence: float = 0.5
    
    # Metadata for additional context
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_actionable(self) -> bool:
        """Check if signal is strong enough to act on."""
        return self.action in (TradingAction.BUY, TradingAction.SELL)
    
    @property
    def is_buy(self) -> bool:
        return self.action == TradingAction.BUY
    
    @property
    def is_sell(self) -> bool:
        return self.action == TradingAction.SELL
    
    @property
    def is_hold(self) -> bool:
        return self.action == TradingAction.HOLD
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'signal_type': self.signal_type.value,
            'symbol': self.symbol,
            'action': self.action.value if self.action else None,
            'entry_threshold': self.entry_threshold,
            'exit_threshold': self.exit_threshold,
            'value': self.value,
            'confidence': self.confidence,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingSignal':
        """Create from dictionary."""
        action = None
        if data.get('action'):
            action = TradingAction(data['action'])
        
        return cls(
            signal_type=SignalType(data.get('signal_type', 'action')),
            symbol=data['symbol'],
            action=action,
            entry_threshold=data.get('entry_threshold'),
            exit_threshold=data.get('exit_threshold'),
            value=data.get('value'),
            confidence=data.get('confidence', 0.5),
            metadata=data.get('metadata', {})
        )
    
    @classmethod
    def hold(cls, symbol: str, confidence: float = 0.5) -> 'TradingSignal':
        """Create a hold signal."""
        return cls(
            signal_type=SignalType.ACTION,
            symbol=symbol,
            action=TradingAction.HOLD,
            confidence=confidence
        )
    
    @classmethod
    def buy(cls, symbol: str, confidence: float = 0.5, **metadata) -> 'TradingSignal':
        """Create a buy signal."""
        return cls(
            signal_type=SignalType.ACTION,
            symbol=symbol,
            action=TradingAction.BUY,
            confidence=confidence,
            metadata=metadata
        )
    
    @classmethod
    def sell(cls, symbol: str, confidence: float = 0.5, **metadata) -> 'TradingSignal':
        """Create a sell signal."""
        return cls(
            signal_type=SignalType.ACTION,
            symbol=symbol,
            action=TradingAction.SELL,
            confidence=confidence,
            metadata=metadata
        )
    
    @classmethod
    def from_model_output(cls, symbol: str, model_output: Dict[str, Any]) -> 'TradingSignal':
        """
        Create TradingSignal from model output dict.
        
        Handles common model output formats:
        - {action: 'buy'/'sell'/'hold', confidence: 0.5, ...}
        - {action: 'buy'/'sell'/'hold', proba_buy: 0.7, proba_sell: 0.1, ...}
        """
        action_str = model_output.get('action', 'hold')
        confidence = model_output.get('confidence', 0.5)
        
        action_map = {
            'buy': TradingAction.BUY,
            'sell': TradingAction.SELL,
            'hold': TradingAction.HOLD,
            'flatten': TradingAction.FLATTEN
        }
        
        action = action_map.get(action_str.lower(), TradingAction.HOLD)
        
        metadata = {
            k: v for k, v in model_output.items() 
            if k not in ('action', 'confidence')
        }
        
        return cls(
            signal_type=SignalType.ACTION,
            symbol=symbol,
            action=action,
            confidence=float(confidence),
            metadata=metadata
        )


@dataclass
class SignalBatch:
    """Collection of signals from multiple symbols/models."""
    signals: Dict[str, TradingSignal] = field(default_factory=dict)
    timestamp: Optional[Any] = None
    
    def add(self, signal: TradingSignal):
        """Add a signal to the batch."""
        self.signals[signal.symbol] = signal
    
    def get(self, symbol: str) -> Optional[TradingSignal]:
        """Get signal for a symbol."""
        return self.signals.get(symbol)
    
    def get_actionable(self) -> Dict[str, TradingSignal]:
        """Get only actionable signals (buy/sell, not hold)."""
        return {
            sym: sig for sym, sig in self.signals.items()
            if sig.is_actionable
        }
    
    def get_actions(self, action: TradingAction) -> Dict[str, TradingSignal]:
        """Get signals for a specific action."""
        return {
            sym: sig for sym, sig in self.signals.items()
            if sig.action == action
        }
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert all signals to dict."""
        return {
            sym: sig.to_dict() for sym, sig in self.signals.items()
        }
    
    def __len__(self) -> int:
        return len(self.signals)
    
    def __iter__(self):
        return iter(self.signals.values())
    
    def __getitem__(self, symbol: str) -> TradingSignal:
        return self.signals[symbol]
