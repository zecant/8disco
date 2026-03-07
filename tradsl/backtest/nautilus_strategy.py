"""
NautilusTrader strategy wrapper for tradsl.

Wraps tradsl models and sizers in a NautilusTrader Strategy.
Requires NautilusTrader to be installed - fails fast if not available.
"""

from typing import Dict, List, Optional, Any, Callable
from decimal import Decimal
import pandas as pd
import numpy as np

# Import NautilusTrader - fail fast if not available
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.config import StrategyConfig as NTConfig
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.enums import OrderSide, BarAggregation, PriceType
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.core.uuid import UUID4


class NautilusStrategyConfig(NTConfig, frozen=True):
    """
    Configuration for NautilusStrategy.
    
    Inherits from NT StrategyConfig to be compatible with NautilusTrader.
    """
    
    strategy_id: str = "tradsl_strategy"
    order_id_tag: str = "tradsl"
    tradsl_config: Dict = None
    symbols: List[str] = None
    venue: str = "BACKTEST"
    bar_type: str = "1-MINUTE-LAST-EXTERNAL"
    position_size: Decimal = Decimal("1")


class NautilusStrategy(Strategy):
    """
    NautilusTrader Strategy that uses tradsl models and sizers.
    
    This strategy:
    - Receives bar data from NautilusTrader
    - Computes features using tradsl FeatureEngine (via DAG)
    - Generates signals using tradsl models
    - Calculates allocations using tradsl sizers
    - Executes orders via NautilusTrader
    
    Must inherit from NT Strategy to work with NT engine.
    """
    
    def __init__(self, config: NautilusStrategyConfig):
        super().__init__(config)
        
        self._tradsl_config = config.tradsl_config or {}
        self._symbols = config.symbols or []
        self._venue = config.venue
        self._bar_type = config.bar_type
        self._position_size = config.position_size
        
        self.models: Dict[str, Any] = {}
        self.sizer: Optional[Any] = None
        self.feature_engine: Optional[Callable] = None
        
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.feature_df: Optional[pd.DataFrame] = None
        self.bar_types: Dict[str, InstrumentId] = {}
        
        self._historical_data_loaded = False
        self._instruments: Dict[str, Any] = {}
        
        self.instrument_ids: Dict[str, InstrumentId] = {}
        
        self._initialize_tradsl()
    
    @property
    def tradsl_config(self) -> Dict:
        return self._tradsl_config
    
    @property
    def symbols(self) -> List[str]:
        return self._symbols
    
    @property
    def venue(self) -> str:
        return self._venue
    
    @property
    def bar_type_str(self) -> str:
        return self._bar_type
    
    @property
    def position_size(self):
        return self._position_size
    
    def on_start(self):
        """Called when strategy starts."""
        self._log.info("NautilusStrategy starting...")
        
        self._setup_instruments()
        
        self._subscribe_to_data()
        
        self._load_historical_data()
        
        self._log.info(f"NautilusStrategy started with {len(self.symbols)} symbols")
    
    def _setup_instruments(self):
        """Set up instrument IDs and bar types for each symbol."""
        for symbol in self._symbols:
            instrument_id = InstrumentId(Symbol(symbol), Venue(self._venue))
            self.instrument_ids[symbol] = instrument_id
            
            bar_spec = self._parse_bar_spec(self._bar_type)
            
            bar_type_nt = BarType(
                instrument_id=instrument_id,
                bar_spec=bar_spec,
                aggregation_source=1,
            )
            self.bar_types[symbol] = bar_type_nt
    
    def _parse_bar_spec(self, bar_type_str: str) -> BarSpecification:
        """Parse bar type string to BarSpecification."""
        parts = bar_type_str.split("-")
        step = int(parts[0])
        aggregation_str = parts[1].upper()
        
        aggregation_map = {
            "MINUTE": BarAggregation.MINUTE,
            "HOUR": BarAggregation.HOUR,
            "DAY": BarAggregation.DAY,
            "WEEK": BarAggregation.WEEK,
            "MONTH": BarAggregation.MONTH,
        }
        
        agg = aggregation_map.get(aggregation_str, BarAggregation.MINUTE)
        
        return BarSpecification(
            step=step,
            aggregation=agg,
            price_type=PriceType.LAST,
        )
    
    def _initialize_tradsl(self):
        """Initialize tradsl models and sizers from config."""
        from tradsl.models import DecisionTreeModel
        from tradsl.sizing import EqualWeightSizer, create_sizer
        
        if 'models' in self._tradsl_config:
            models_config = self._tradsl_config['models']
            for symbol, model in models_config.items():
                self.models[symbol] = model
                self._log.info(f"Loaded model for {symbol}")
        
        if 'sizer' in self._tradsl_config:
            sizer_config = self._tradsl_config['sizer']
            if isinstance(sizer_config, dict):
                sizer_type = sizer_config.get('type', 'equal')
                sizer_params = sizer_config.get('params', {})
                self.sizer = create_sizer(sizer_type, **sizer_params)
            else:
                self.sizer = sizer_config
            self._log.info(f"Loaded sizer: {type(self.sizer).__name__}")
    
    def _subscribe_to_data(self):
        """Subscribe to bar data for all symbols."""
        for symbol, bar_type in self.bar_types.items():
            self.subscribe_bars(bar_type)
            self._log.info(f"Subscribed to {bar_type}")
    
    def _load_historical_data(self):
        """Load historical data from config if available."""
        from tradsl.utils.feature_engine import compute_features
        
        if 'data' not in self._tradsl_config:
            return
        
        data = self._tradsl_config['data']
        
        if not data:
            return
        
        for symbol, df in data.items():
            self.market_data[symbol] = df.copy()
        
        if self.market_data:
            try:
                dag_config = self._tradsl_config.get('dag_config', {})
                combined = pd.concat(self.market_data.values()) if len(self.market_data) > 1 else list(self.market_data.values())[0]
                self.feature_df = compute_features(dag_config, combined)
                self._historical_data_loaded = True
                self._log.info(f"Loaded historical data: {len(self.feature_df)} rows")
            except Exception as e:
                self._log.error(f"Failed to compute features: {e}")
                raise
    
    def on_bar(self, bar: Bar):
        """Called on each new bar."""
        if not self._historical_data_loaded:
            return
        
        symbol = bar.bar_type.instrument_id.symbol.value
        
        self._update_market_data(symbol, bar)
        
        self._update_features(symbol)
        
        signals = self._generate_signals(symbol)
        
        self._execute_signals(signals)
    
    def _update_market_data(self, symbol: str, bar: Bar):
        """Update market data DataFrame with new bar."""
        timestamp = pd.to_datetime(bar.ts_event, unit='ns', utc=True).tz_localize(None)
        
        if symbol not in self.market_data:
            self.market_data[symbol] = pd.DataFrame()
        
        new_row = pd.DataFrame({
            'open': [float(bar.open)],
            'high': [float(bar.high)],
            'low': [float(bar.low)],
            'close': [float(bar.close)],
            'volume': [float(bar.volume)],
        }, index=[timestamp])
        
        self.market_data[symbol] = pd.concat([self.market_data[symbol], new_row])
        
        max_bars = self._tradsl_config.get('max_history_bars', 1000)
        if len(self.market_data[symbol]) > max_bars:
            self.market_data[symbol] = self.market_data[symbol].iloc[-max_bars:]
    
    def _update_features(self, symbol: str):
        """Update features for the new bar."""
        from tradsl.utils.feature_engine import compute_features_incremental
        
        if self.feature_df is None:
            return
        
        try:
            combined = pd.concat(self.market_data.values()) if len(self.market_data) > 1 else list(self.market_data.values())[0]
            dag_config = self._tradsl_config.get('dag_config', {})
            self.feature_df = compute_features_incremental(dag_config, combined, self.market_data[symbol].iloc[-1])
        except Exception as e:
            self._log.warning(f"Feature update failed: {e}")
    
    def _generate_signals(self, symbol: str):
        """Generate trading signals for a symbol."""
        from tradsl.signals import SignalBatch, TradingSignal
        
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
            
        except Exception as e:
            self._log.warning(f"Signal generation failed: {e}")
            return None
    
    def _get_model_inputs(self, symbol: str) -> List[str]:
        """Get input columns for a model."""
        dag_config = self._tradsl_config.get('dag_config', {})
        
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
        from tradsl.signals import TradingAction
        
        if signals is None or len(signals) == 0:
            return
        
        for symbol, signal in signals.signals.items():
            if not signal.is_actionable:
                continue
            
            instrument_id = self.instrument_ids.get(symbol)
            if not instrument_id:
                continue
            
            current_data = self.market_data.get(symbol)
            if current_data is None or current_data.empty:
                continue
            
            try:
                price = float(current_data.iloc[-1]['close'])
            except Exception:
                continue
            
            self._submit_order(symbol, instrument_id, signal, price)
    
    def _submit_order(self, symbol: str, instrument_id: InstrumentId, signal, price: float):
        """Submit an order to NautilusTrader."""
        action = signal.action
        
        if str(action).lower() == 'buy':
            side = OrderSide.BUY
        elif str(action).lower() == 'sell':
            side = OrderSide.SELL
        else:
            return
        
        order = MarketOrder(
            trader_id=self.trader_id,
            strategy_id=self.id,
            instrument_id=instrument_id,
            client_order_id=UUID4(),
            side=side,
            quantity=self._position_size,
            uuid4=UUID4(),
            ts_init=self.clock.timestamp_ns(),
            time_in_force=1,
        )
        
        self.submit_order(order)
        
        self._log.info(f"Submitted {side.value} order for {symbol}")
    
    def on_order_filled(self, event):
        """Handle order filled events."""
        self._log.info(f"Order filled: {event}")
    
    def on_stop(self):
        """Called when strategy stops."""
        self._log.info("NautilusStrategy stopping...")
        
        self.cancel_all_orders()
        
        self.close_all_positions()
        
        self._log.info("NautilusStrategy stopped")
    
    def on_reset(self):
        """Called when strategy resets."""
        self.market_data.clear()
        self.feature_df = None
        self._historical_data_loaded = False


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
    
    Raises:
        ImportError: If NautilusTrader is not installed
    """
    config = NautilusStrategyConfig(
        strategy_id=strategy_id,
        order_id_tag=strategy_id.lower().replace("_", "-"),
        symbols=symbols,
        venue=venue,
        bar_type=bar_type,
        position_size=position_size,
        tradsl_config=tradsl_config,
    )
    
    return NautilusStrategy(config)
