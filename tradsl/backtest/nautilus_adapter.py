"""
NautilusTrader adapter for tradsl.

Converts tradsl data formats to NautilusTrader objects.
"""

from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
from datetime import datetime
import pandas as pd
import numpy as np

from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.enums import BarAggregation, PriceType
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig


class NautilusAdapter:
    """
    Adapter for converting tradsl data to NautilusTrader format.
    
    Handles:
    - Creating instruments from symbol/venue pairs
    - Converting DataFrames to NT Bars
    - Loading data into NT BacktestEngine
    """
    
    @staticmethod
    def create_instrument(
        symbol: str,
        venue: str = "BACKTEST",
        currency: str = "USD",
        price_precision: int = 2,
        size_precision: int = 0
    ):
        """
        Create a NautilusTrader Equity instrument.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            venue: Venue name (e.g., "BINANCE", "BACKTEST")
            currency: Quote currency
            price_precision: Decimal precision for prices
            size_precision: Decimal precision for quantities
        
        Returns:
            Equity instrument
        """
        from nautilus_trader.model.instruments import Equity
        
        instrument_id = InstrumentId(Symbol(symbol), Venue(venue))
        
        return Equity(
            instrument_id=instrument_id,
            raw_symbol=Symbol(symbol),
            currency=currency,
            price_precision=price_precision,
            price_increment=Price.from_str("0.01"),
            lot_size=Quantity.from_int(1),
            ts_event=0,
            ts_init=0,
        )
    
    @staticmethod
    def parse_bar_type(bar_type_str: str) -> BarType:
        """
        Parse bar type string to NT BarType.
        
        Examples:
            "1-MINUTE-LAST-EXTERNAL"
            "1-HOUR-LAST-INTERNAL"
            "1-DAY-LAST-EXTERNAL"
        
        Args:
            bar_type_str: Bar type string
        
        Returns:
            BarType object
        """
        parts = bar_type_str.split("-")
        
        if len(parts) != 4:
            raise ValueError(
                f"Invalid bar type format: {bar_type_str}. "
                "Expected: '{{step}}-{{aggregation}}-{{price_type}}-{{source}}'"
                "Example: '1-MINUTE-LAST-EXTERNAL'"
            )
        
        step = int(parts[0])
        aggregation_str = parts[1].upper()
        price_type_str = parts[2].upper()
        source_str = parts[3].upper()
        
        aggregation_map = {
            "MINUTE": BarAggregation.MINUTE,
            "HOUR": BarAggregation.HOUR,
            "DAY": BarAggregation.DAY,
            "WEEK": BarAggregation.WEEK,
            "MONTH": BarAggregation.MONTH,
            "TICK": BarAggregation.TICK,
            "TICK_IMBALANCE": BarAggregation.TICK_IMBALANCE,
            "TICK_VOLUME": BarAggregation.TICK_VOLUME,
            "VOLUME": BarAggregation.VOLUME,
            "VOLUME_IMBALANCE": BarAggregation.VOLUME_IMBALANCE,
        }
        
        price_type_map = {
            "BID": PriceType.BID,
            "ASK": PriceType.ASK,
            "LAST": PriceType.LAST,
            "MID": PriceType.MID,
            "INTERNAL": PriceType.INTERNAL,
            "EXTERNAL": PriceType.EXTERNAL,
        }
        
        aggregation = aggregation_map.get(aggregation_str)
        if aggregation is None:
            raise ValueError(f"Unknown bar aggregation: {aggregation_str}")
        
        price_type = price_type_map.get(price_type_str)
        if price_type is None:
            raise ValueError(f"Unknown price type: {price_type_str}")
        
        from nautilus_trader.model.enums import AggregationSource
        source = AggregationSource.EXTERNAL if source_str == "EXTERNAL" else AggregationSource.INTERNAL
        
        return BarType(
            instrument_id=None,  # Will be set when used
            bar_spec=None,  # Will be created
            aggregation_source=source,
        ).__class__(
            bar_spec=None,
            aggregation=aggregation,
            price_type=price_type,
        )
    
    @staticmethod
    def create_bar_spec(step: int, aggregation: str) -> 'BarSpecification':
        """Create BarSpecification from step and aggregation."""
        from nautilus_trader.model.objects import BarSpecification
        
        aggregation_map = {
            "MINUTE": BarAggregation.MINUTE,
            "HOUR": BarAggregation.HOUR,
            "DAY": BarAggregation.DAY,
            "WEEK": BarAggregation.WEEK,
            "MONTH": BarAggregation.MONTH,
        }
        
        agg = aggregation_map.get(aggregation.upper())
        if agg is None:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        return BarSpecification(
            step=step,
            aggregation=agg,
            price_type=PriceType.LAST,
        )
    
    @staticmethod
    def dataframe_to_bars(
        df: pd.DataFrame,
        symbol: str,
        venue: str = "BACKTEST",
        bar_type: str = "1-MINUTE-LAST-EXTERNAL"
    ) -> List[Bar]:
        """
        Convert OHLCV DataFrame to NautilusTrader Bars.
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume
            symbol: Trading symbol
            venue: Venue name
            bar_type: Bar type string
        
        Returns:
            List of NT Bar objects
        """
        df = df.copy()
        
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        df.index = df.index.tz_localize(None)
        
        bar_spec = NautilusAdapter._parse_bar_spec(bar_type)
        
        parts = bar_type.split("-")
        if len(parts) >= 4:
            price_type_str = parts[2].upper()
        else:
            price_type_str = "LAST"
        
        price_type_map = {
            "BID": PriceType.BID,
            "ASK": PriceType.ASK,
            "LAST": PriceType.LAST,
            "MID": PriceType.MID,
            "INTERNAL": PriceType.INTERNAL,
            "EXTERNAL": PriceType.EXTERNAL,
        }
        price_type = price_type_map.get(price_type_str, PriceType.LAST)
        
        instrument_id = InstrumentId(Symbol(symbol), Venue(venue))
        
        bars = []
        
        for idx, row in df.iterrows():
            ts = int(idx.timestamp() * 1e9)
            
            bar = Bar(
                bar_type=BarType(
                    instrument_id=instrument_id,
                    bar_spec=bar_spec,
                    aggregation_source=(
                        1  # EXTERNAL
                    ),
                ),
                open=NautilusAdapter._to_price(row.get('open', row.get('close', 0))),
                high=NautilusAdapter._to_price(row.get('high', row.get('close', 0))),
                low=NautilusAdapter._to_price(row.get('low', row.get('close', 0))),
                close=NautilusAdapter._to_price(row.get('close', 0)),
                volume=NautilusAdapter._to_quantity(row.get('volume', 0)),
                ts_event=ts,
                ts_init=ts,
            )
            bars.append(bar)
        
        return bars
    
    @staticmethod
    def _parse_bar_spec(bar_type: str) -> 'BarSpecification':
        """Parse bar type string to BarSpecification."""
        from nautilus_trader.model.objects import BarSpecification
        
        parts = bar_type.split("-")
        if len(parts) < 2:
            raise ValueError(f"Invalid bar type: {bar_type}")
        
        step = int(parts[0])
        aggregation = parts[1].upper()
        
        aggregation_map = {
            "MINUTE": BarAggregation.MINUTE,
            "HOUR": BarAggregation.HOUR,
            "DAY": BarAggregation.DAY,
            "WEEK": BarAggregation.WEEK,
            "MONTH": BarAggregation.MONTH,
            "TICK": BarAggregation.TICK,
            "VOLUME": BarAggregation.VOLUME,
        }
        
        agg = aggregation_map.get(aggregation)
        if agg is None:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        return BarSpecification(
            step=step,
            aggregation=agg,
            price_type=PriceType.LAST,
        )
    
    @staticmethod
    def _to_price(value: float) -> Price:
        """Convert float to NT Price."""
        if pd.isna(value) or value is None:
            value = 0.0
        return Price.from_str(f"{float(value):.2f}")
    
    @staticmethod
    def _to_quantity(value: float) -> Quantity:
        """Convert float to NT Quantity."""
        if pd.isna(value) or value is None:
            value = 0.0
        return Quantity.from_int(int(float(value)))
    
    @staticmethod
    def load_data_to_engine(
        engine: BacktestEngine,
        data: Dict[str, pd.DataFrame],
        venue: str = "BACKTEST",
        bar_type: str = "1-MINUTE-LAST-EXTERNAL"
    ):
        """
        Load multiple instruments' data into NT BacktestEngine.
        
        Args:
            engine: NT BacktestEngine
            data: Dict mapping symbol -> OHLCV DataFrame
            venue: Venue name
            bar_type: Bar type string
        """
        bar_spec = NautilusAdapter._parse_bar_spec(bar_type)
        
        parts = bar_type.split("-")
        if len(parts) >= 4:
            price_type_str = parts[2].upper()
            source_str = parts[3].upper()
        else:
            price_type_str = "LAST"
            source_str = "EXTERNAL"
        
        price_type_map = {
            "BID": PriceType.BID,
            "ASK": PriceType.ASK,
            "LAST": PriceType.LAST,
            "MID": PriceType.MID,
        }
        price_type = price_type_map.get(price_type_str, PriceType.LAST)
        
        from nautilus_trader.model.enums import AggregationSource
        source = AggregationSource.EXTERNAL if source_str == "EXTERNAL" else AggregationSource.INTERNAL
        
        for symbol, df in data.items():
            instrument = NautilusAdapter.create_instrument(symbol, venue)
            engine.add_instrument(instrument)
            
            bars = NautilusAdapter.dataframe_to_bars(df, symbol, venue, bar_type)
            
            for bar in bars:
                engine.add_data(bar)
    
    @staticmethod
    def create_engine(
        venue: str = "BACKTEST",
        starting_balances: Optional[Dict[str, Decimal]] = None
    ) -> BacktestEngine:
        """
        Create and configure a NT BacktestEngine.
        
        Args:
            venue: Venue name
            starting_balances: Dict of currency -> balance
        
        Returns:
            Configured BacktestEngine
        """
        from nautilus_trader.model.enums import Venue, OmsType, AccountType
        from nautilus_trader.model.currencies import USD
        from nautilus_trader.model.objects import Money
        
        engine = BacktestEngine(
            config=BacktestEngineConfig(
                logging=LoggingConfig(log_level="ERROR"),
            )
        )
        
        if starting_balances is None:
            starting_balances = {"USD": Decimal("100000")}
        
        venue_obj = Venue(venue)
        
        engine.add_venue(
            venue=venue_obj,
            oms_type=OmsType.NETTING,
            account_type=AccountType.CASH,
            base_currency=USD,
            starting_balances=[
                Money(amount, currency)
                for currency, amount in starting_balances.items()
            ],
        )
        
        return engine


class NautilusResultParser:
    """Parse NT backtest results to tradsl format."""
    
    @staticmethod
    def extract_equity_curve(analyzer) -> pd.Series:
        """Extract equity curve from NT analyzer."""
        if analyzer is None:
            return pd.Series()
        
        try:
            returns = analyzer.returns()
            if returns is None or returns.empty:
                return pd.Series()
            
            equity = (1 + returns).cumprod()
            return equity
        except Exception:
            return pd.Series()
    
    @staticmethod
    def extract_trades(analyzer) -> List[Dict[str, Any]]:
        """Extract trades from NT analyzer."""
        if analyzer is None:
            return []
        
        try:
            trades = analyzer.get_orders_fill_csv()
            return trades
        except Exception:
            return []
    
    @staticmethod
    def extract_summary(analyzer) -> Dict[str, Any]:
        """Extract performance summary from NT analyzer."""
        if analyzer is None:
            return {}
        
        summary = {}
        
        try:
            returns = analyzer.returns()
            if returns is not None and not returns.empty:
                summary['total_return'] = float(returns.iloc[-1]) if len(returns) > 0 else 0.0
                summary['sharpe_ratio'] = float(analyzer.sharpe_ratio()) or 0.0
                summary['max_drawdown'] = float(analyzer.max_drawdown()) or 0.0
                summary['volatility'] = float(returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
        except Exception:
            pass
        
        return summary
