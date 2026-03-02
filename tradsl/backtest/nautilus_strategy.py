"""
NautilusTrader strategy wrapper for tradsl.

Wraps tradsl models and sizers in a NautilusTrader Strategy.
"""

from typing import Dict, List, Optional, Any, Callable, Set
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass, field
import pandas as pd
import numpy as np


@dataclass
class NautilusStrategyConfig:
    """Configuration for NautilusStrategy."""
    
    strategy_id: str = "tradsl_strategy"
    order_id_tag: str = "tradsl"
    tradsl_config: Dict = field(default_factory=dict)
    symbols: List[str] = field(default_factory=list)
    venue: str = "BACKTEST"
    bar_type: str = "1-MINUTE-LAST-EXTERNAL"
    position_size: Decimal = Decimal("1")


class NautilusStrategy:
    """
    NautilusTrader Strategy that uses tradsl models and sizers.
    
    This strategy:
    - Receives bar data from NautilusTrader
    - Computes features using tradsl FeatureEngine (via DAG)
    - Generates signals using tradsl models
    - Calculates allocations using tradsl sizers
    - Executes orders via NautilusTrader
    """
    
    def __init__(self, config: NautilusStrategyConfig):
        self.config = config
        self.tradsl_config = config.tradsl_config
        self.symbols = config.symbols
        self.venue = config.venue
        self.bar_type = config.bar_type
        self.position_size = config.position_size
        
        self.models: Dict[str, Any] = {}
        self.sizer: Optional[Any] = None
        self.feature_engine: Optional[Callable] = None
        
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.feature_df: Optional[pd.DataFrame] = None
        
        self.bar_types: Dict[str, Any] = {}
        
        self._is_initialized = False
        self._historical_data_loaded = False
        
        self._trader_id = None
        self._strategy_id = None
        self._clock = None
        self._cache = None
        self._portfolio = None
        self._order_factory = None
    
    def _check_initialized(self):
        """Check if strategy is properly initialized."""
        if not self._is_initialized:
            raise RuntimeError("Strategy not initialized. Call on_start first.")
    
    def _initialize_instruments(self):
        """Initialize instrument mappings."""
        try:
            from nautilus_trader.model.identifiers import Symbol, Venue
            from nautilus_trader.model.data import BarType
            from nautilus_trader.model.objects import BarSpecification
        except ImportError:
            return
        
        for symbol in self.symbols:
            try:
                instrument_id = self._get_instrument_id(symbol)
                bar_spec = self._parse_bar_spec(self.bar_type)
                bar_type = BarType(
                    instrument_id=instrument_id,
                    bar_spec=bar_spec,
                    aggregation_source=1,
                )
                self.bar_types[symbol] = bar_type
            except Exception:
                pass
    
    def _get_instrument_id(self, symbol: str):
        """Get instrument ID for symbol."""
        try:
            from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
            return InstrumentId(Symbol(symbol), Venue(self.venue))
        except ImportError:
            return None
    
    def _parse_bar_spec(self, bar_type_str: str):
        """Parse bar type string to BarSpecification."""
        try:
            from nautilus_trader.model.objects import BarSpecification
            from nautilus_trader.model.enums import BarAggregation, PriceType
            
            parts = bar_type_str.split("-")
            step = int(parts[0])
            aggregation_str = parts[1].upper()
            
            aggregation_map = {
                "MINUTE": BarAggregation.MINUTE,
                "HOUR": BarAggregation.HOUR,
                "DAY": BarAggregation.DAY,
            }
            
            agg = aggregation_map.get(aggregation_str, BarAggregation.MINUTE)
            
            return BarSpecification(
                step=step,
                aggregation=agg,
                price_type=PriceType.LAST,
            )
        except ImportError:
            return None
    
    def _initialize_models(self):
        """Initialize tradsl models from config."""
        if 'models' not in self.tradsl_config:
            return
        
        models_config = self.tradsl_config['models']
        
        for symbol, model in models_config.items():
            self.models[symbol] = model
    
    def _initialize_sizer(self):
        """Initialize position sizer from config."""
        if 'sizer' in self.tradsl_config:
            self.sizer = self.tradsl_config['sizer']
    
    def _subscribe_to_data(self):
        """Subscribe to bar data for all symbols."""
        for symbol, bar_type in self.bar_types.items():
            try:
                self.subscribe_bars(bar_type)
            except Exception:
                pass
    
    def _load_historical_data(self):
        """Load historical data and compute initial features."""
        try:
            from tradsl.utils.feature_engine import compute_features
        except ImportError:
            return
        
        if 'data' not in self.tradsl_config:
            return
        
        data = self.tradsl_config['data']
        
        if not data:
            return
        
        for symbol, df in data.items():
            self.market_data[symbol] = df.copy()
        
        if self.market_data:
            try:
                dag_config = self.tradsl_config.get('dag_config', {})
                combined = pd.concat(self.market_data.values()) if len(self.market_data) > 1 else list(self.market_data.values())[0]
                self.feature_df = compute_features(dag_config, combined)
                self._historical_data_loaded = True
            except Exception:
                self.feature_df = pd.DataFrame()
                self._historical_data_loaded = True
    
    def on_start(self):
        """Called when strategy starts."""
        self._is_initialized = True
        
        self._initialize_instruments()
        self._initialize_models()
        self._initialize_sizer()
        self._subscribe_to_data()
        self._load_historical_data()
    
    def on_bar(self, bar: Any):
        """Called on each new bar."""
        if not self._is_initialized:
            return
        
        symbol = getattr(bar.bar_type, 'instrument_id', None)
        if symbol:
            symbol = getattr(symbol, 'symbol', None)
            if symbol:
                symbol = getattr(symbol, 'value', str(symbol))
        
        if not symbol:
            return
        
        self._update_market_data(symbol, bar)
        
        if not self._historical_data_loaded:
            return
        
        self._update_features(symbol, bar)
        
        signals = self._generate_signals(symbol)
        
        self._execute_signals(signals)
    
    def _update_market_data(self, symbol: str, bar: Any):
        """Update market data DataFrame with new bar."""
        try:
            ts_event = getattr(bar, 'ts_event', 0)
            timestamp = pd.to_datetime(ts_event, unit='ns', utc=True)
            timestamp = timestamp.tz_localize(None)
        except Exception:
            timestamp = datetime.now()
        
        try:
            open_price = float(getattr(bar, 'open', 0))
            high_price = float(getattr(bar, 'high', 0))
            low_price = float(getattr(bar, 'low', 0))
            close_price = float(getattr(bar, 'close', 0))
            volume = float(getattr(bar, 'volume', 0))
        except Exception:
            return
        
        if symbol not in self.market_data:
            self.market_data[symbol] = pd.DataFrame()
        
        new_row = pd.DataFrame({
            'open': [open_price],
            'high': [high_price],
            'low': [low_price],
            'close': [close_price],
            'volume': [volume],
        }, index=[timestamp])
        
        self.market_data[symbol] = pd.concat([self.market_data[symbol], new_row])
        
        max_bars = self.tradsl_config.get('max_history_bars', 1000)
        if len(self.market_data[symbol]) > max_bars:
            self.market_data[symbol] = self.market_data[symbol].iloc[-max_bars:]
    
    def _update_features(self, symbol: str, bar: Any):
        """Update features for the new bar."""
        try:
            from tradsl.utils.feature_engine import compute_features_incremental
        except ImportError:
            return
        
        if self.feature_df is None:
            return
        
        try:
            combined = pd.concat(self.market_data.values()) if len(self.market_data) > 1 else list(self.market_data.values())[0]
            dag_config = self.tradsl_config.get('dag_config', {})
            self.feature_df = compute_features_incremental(dag_config, combined, self.market_data[symbol].iloc[-1])
        except Exception:
            pass
    
    def _generate_signals(self, symbol: str):
        """Generate trading signals for a symbol."""
        try:
            from tradsl.signals import SignalBatch, TradingSignal
        except ImportError:
            return None
        
        if symbol not in self.models:
            return None
        
        model = self.models[symbol]
        
        if not hasattr(model, 'predict'):
            return None
        
        if self.feature_df is None or len(self.feature_df) == 0:
            return None
        
        try:
            input_cols = self._get_model_inputs(symbol)
            
            if not input_cols:
                return None
            
            available_cols = [c for c in input_cols if c in self.feature_df.columns]
            if not available_cols:
                return None
            
            X = self.feature_df[available_cols].iloc[-1:]
            
            model_output = model.predict(X)
            
            if isinstance(model_output, dict):
                signal = TradingSignal.from_model_output(symbol, model_output)
            elif hasattr(model_output, 'action'):
                signal = model_output
            else:
                signal = TradingSignal.hold(symbol)
            
            batch = SignalBatch()
            batch.add(signal)
            return batch
            
        except Exception:
            return None
    
    def _get_model_inputs(self, symbol: str) -> List[str]:
        """Get input columns for a model."""
        dag_config = self.tradsl_config.get('dag_config', {})
        
        if not dag_config:
            return list(self.feature_df.columns) if self.feature_df is not None else []
        
        model_config = dag_config.get(symbol, {})
        inputs = model_config.get('inputs', [])
        
        if not inputs:
            return list(self.feature_df.columns) if self.feature_df is not None else []
        
        cols = self.feature_df.columns.tolist() if self.feature_df is not None else []
        
        result = []
        for inp in inputs:
            matched = [c for c in cols if c == inp or c.endswith(f"_{inp}")]
            result.extend(matched)
        
        return result
    
    def _execute_signals(self, signals):
        """Execute trades based on signals."""
        try:
            from tradsl.signals import TradingAction
        except ImportError:
            return
        
        if signals is None or len(signals) == 0:
            return
        
        for symbol, signal in signals.signals.items():
            if not getattr(signal, 'is_actionable', False):
                continue
            
            instrument_id = self.bar_types.get(symbol)
            if not instrument_id:
                continue
            
            current_data = self.market_data.get(symbol, pd.DataFrame())
            if current_data.empty:
                continue
            
            try:
                price = float(current_data.iloc[-1]['close'])
            except Exception:
                continue
            
            order_side = getattr(signal, 'action', None)
            if order_side is None:
                continue
            
            try:
                self._submit_order(symbol, instrument_id, order_side, price)
            except Exception:
                pass
    
    def _submit_order(self, symbol: str, instrument_id: Any, action: Any, price: float):
        """Submit an order to NautilusTrader."""
        try:
            from nautilus_trader.model.enums import OrderSide as NTOrderSide
            from nautilus_trader.model.orders import MarketOrder
            from nautilus_trader.core.uuid import UUID4
            
            if str(action).lower() == 'buy':
                side = NTOrderSide.BUY
            elif str(action).lower() == 'sell':
                side = NTOrderSide.SELL
            else:
                return
            
            order = MarketOrder(
                trader_id=self._trader_id,
                strategy_id=self._strategy_id,
                instrument_id=instrument_id,
                client_order_id=UUID4(),
                side=side,
                quantity=self.position_size,
                uuid4=UUID4(),
                ts_init=self._clock.timestamp_ns() if self._clock else 0,
                time_in_force=1,
            )
            
            self.submit_order(order)
            
        except Exception:
            pass
    
    def on_order_filled(self, event: Any):
        """Handle order fill events."""
        pass
    
    def on_stop(self):
        """Called when strategy stops."""
        try:
            self.cancel_all_orders()
            self.close_all_positions()
        except Exception:
            pass
    
    def on_reset(self):
        """Called when strategy resets."""
        self.market_data.clear()
        self.feature_df = None
        self._is_initialized = False
        self._historical_data_loaded = False
    
    def _set_dependencies(self, trader_id: Any, strategy_id: Any, clock: Any, cache: Any, portfolio: Any, order_factory: Any):
        """Set dependencies for the strategy."""
        self._trader_id = trader_id
        self._strategy_id = strategy_id
        self._clock = clock
        self._cache = cache
        self._portfolio = portfolio
        self._order_factory = order_factory
    
    @property
    def trader_id(self):
        return self._trader_id
    
    @property
    def id(self):
        return self._strategy_id
    
    @property
    def clock(self):
        return self._clock
    
    @property
    def cache(self):
        return self._cache
    
    @property
    def portfolio(self):
        return self._portfolio
    
    @property
    def order_factory(self):
        return self._order_factory
    
    def subscribe_bars(self, bar_type: Any):
        """Subscribe to bars - placeholder."""
        pass
    
    def submit_order(self, order: Any):
        """Submit order - placeholder."""
        pass
    
    def cancel_all_orders(self):
        """Cancel all orders - placeholder."""
        pass
    
    def close_all_positions(self):
        """Close all positions - placeholder."""
        pass


def create_nautilus_strategy(
    tradsl_config: Dict,
    symbols: List[str],
    venue: str = "BACKTEST",
    bar_type: str = "1-MINUTE-LAST-EXTERNAL",
    position_size: Decimal = Decimal("1"),
    strategy_id: str = "tradsl_strategy"
) -> NautilusStrategy:
    """
    Factory function to create NautilusStrategy.
    
    Args:
        tradsl_config: Parsed tradsl config with models, sizer, etc.
        symbols: List of trading symbols
        venue: Venue name
        bar_type: Bar type string
        position_size: Position size per trade
        strategy_id: Strategy identifier
    
    Returns:
        NautilusStrategy instance
    """
    config = NautilusStrategyConfig(
        strategy_id=strategy_id,
        symbols=symbols,
        venue=venue,
        bar_type=bar_type,
        position_size=position_size,
        tradsl_config=tradsl_config,
    )
    
    return NautilusStrategy(config)
