"""
NautilusTrader execution backend for tradsl.

Provides NautilusTrader-based backtesting execution.
"""

from typing import Dict, List, Optional, Any, Union, Callable
from decimal import Decimal
from datetime import datetime
import pandas as pd
import numpy as np

from tradsl.backtest.execution import ExecutionBackend, Order, Fill
from tradsl.backtest.engine import BacktestResult


class NautilusBackend:
    """
    NautilusTrader-based backtesting backend.
    
    This backend:
    - Uses tradsl for feature computation via DAG
    - Uses tradsl models for signal generation
    - Uses tradsl sizers for position sizing
    - Uses NautilusTrader for order execution and simulation
    
    Data Flow:
    1. Load data into NT BacktestEngine
    2. Create NautilusStrategy with tradsl models/sizers
    3. Run backtest - strategy receives bars, computes features via tradsl
    4. Extract results from NT analyzer
    """
    
    def __init__(
        self,
        tradsl_config: Dict,
        venue: str = "BACKTEST",
        bar_type: str = "1-MINUTE-LAST-EXTERNAL",
        starting_balance: Decimal = Decimal("100000"),
        commission: Decimal = Decimal("0.001"),
        position_size: Decimal = Decimal("1"),
        log_level: str = "ERROR",
    ):
        """
        Initialize NautilusTrader backend.
        
        Args:
            tradsl_config: Parsed tradsl config with models, sizers, DAG
            venue: Venue name
            bar_type: Bar type string
            starting_balance: Starting capital
            commission: Commission rate
            position_size: Default position size per trade
            log_level: NT logging level
        """
        self.tradsl_config = tradsl_config
        self.venue = venue
        self.bar_type = bar_type
        self.starting_balance = starting_balance
        self.commission = commission
        self.position_size = position_size
        self.log_level = log_level
        
        self.engine = None
        self.strategy = None
        self.analyzer = None
    
    def run(
        self,
        data: Dict[str, pd.DataFrame],
        models: Optional[Dict[str, Any]] = None,
        sizer: Optional[Any] = None,
        train_models: bool = False,
    ) -> BacktestResult:
        """
        Run backtest using NautilusTrader.
        
        Args:
            data: Dict mapping symbol -> OHLCV DataFrame
            models: Dict mapping symbol -> trained model
            sizer: Position sizer instance
            train_models: Whether to train models before backtest
        
        Returns:
            BacktestResult with equity curve, trades, summary
        """
        from nautilus_trader.backtest.engine import BacktestEngine
        from nautilus_trader.backtest.config import BacktestEngineConfig
        from nautilus_trader.config import LoggingConfig
        from nautilus_trader.model.enums import Venue, OmsType, AccountType
        from nautilus_trader.model.currencies import USD
        from nautilus_trader.model.objects import Money
        
        config_with_models = dict(self.tradsl_config)
        config_with_models['models'] = models or {}
        if sizer:
            config_with_models['sizer'] = sizer
        config_with_models['data'] = data
        
        self.engine = BacktestEngine(
            config=BacktestEngineConfig(
                logging=LoggingConfig(log_level=self.log_level),
            )
        )
        
        self.engine.add_venue(
            venue=Venue(self.venue),
            oms_type=OmsType.NETTING,
            account_type=AccountType.CASH,
            base_currency=USD,
            starting_balances=[
                Money(self.starting_balance, USD)
            ],
        )
        
        self._add_instruments(data.keys())
        
        self._add_data(data)
        
        from tradsl.backtest.nautilus_strategy import NautilusStrategy, NautilusStrategyConfig
        
        strategy_config = NautilusStrategyConfig(
            strategy_id="tradsl_strategy",
            symbols=list(data.keys()),
            venue=self.venue,
            bar_type=self.bar_type,
            position_size=self.position_size,
            tradsl_config=config_with_models,
        )
        
        self.strategy = NautilusStrategy(strategy_config)
        
        self.engine.add_strategy(self.strategy)
        
        if train_models:
            self._train_models(data, models or {})
        
        self.engine.run()
        
        self.analyzer = self.engine.analyzer
        
        return self._extract_results()
    
    def _add_instruments(self, symbols):
        """Add instruments to the engine."""
        from nautilus_trader.model.identifiers import Symbol
        from nautilus_trader.model.instruments import Equity
        from nautilus_trader.model.objects import Price, Quantity
        
        for symbol in symbols:
            instrument_id = self._get_instrument_id(symbol)
            
            instrument = Equity(
                instrument_id=instrument_id,
                raw_symbol=Symbol(symbol),
                currency="USD",
                price_precision=2,
                price_increment=Price.from_str("0.01"),
                lot_size=Quantity.from_int(1),
                ts_event=0,
                ts_init=0,
            )
            
            self.engine.add_instrument(instrument)
    
    def _add_data(self, data: Dict[str, pd.DataFrame]):
        """Add bar data to the engine."""
        for symbol, df in data.items():
            bars = self._dataframe_to_bars(df, symbol)
            
            for bar in bars:
                self.engine.add_data(bar)
    
    def _dataframe_to_bars(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> List:
        """Convert DataFrame to NT Bars."""
        from nautilus_trader.model.data import Bar, BarType
        from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
        from nautilus_trader.model.objects import Price, Quantity
        from nautilus_trader.model.objects import BarSpecification
        from nautilus_trader.model.enums import BarAggregation, PriceType
        
        df = df.copy()
        
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        df.index = df.index.tz_localize(None)
        
        parts = self.bar_type.split("-")
        step = int(parts[0])
        aggregation_str = parts[1].upper()
        
        aggregation_map = {
            "MINUTE": BarAggregation.MINUTE,
            "HOUR": BarAggregation.HOUR,
            "DAY": BarAggregation.DAY,
        }
        
        agg = aggregation_map.get(aggregation_str, BarAggregation.MINUTE)
        
        bar_spec = BarSpecification(
            step=step,
            aggregation=agg,
            price_type=PriceType.LAST,
        )
        
        instrument_id = self._get_instrument_id(symbol)
        
        bars = []
        
        for idx, row in df.iterrows():
            ts = int(idx.timestamp() * 1e9)
            
            def p(val):
                if pd.isna(val) or val is None:
                    return Price.from_str("0.00")
                return Price.from_str(f"{float(val):.2f}")
            
            def q(val):
                if pd.isna(val) or val is None:
                    return Quantity.from_int(0)
                return Quantity.from_int(int(float(val)))
            
            bar = Bar(
                bar_type=BarType(
                    instrument_id=instrument_id,
                    bar_spec=bar_spec,
                    aggregation_source=1,
                ),
                open=p(row.get('open', row.get('close', 0))),
                high=p(row.get('high', row.get('close', 0))),
                low=p(row.get('low', row.get('close', 0))),
                close=p(row.get('close', 0)),
                volume=q(row.get('volume', 0)),
                ts_event=ts,
                ts_init=ts,
            )
            bars.append(bar)
        
        return bars
    
    def _get_instrument_id(self, symbol: str):
        """Get instrument ID for symbol."""
        from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
        return InstrumentId(Symbol(symbol), Venue(self.venue))
    
    def _train_models(self, data: Dict[str, pd.DataFrame], models: Dict[str, Any]):
        """Train models on historical data."""
        from tradsl.utils.feature_engine import compute_features
        
        if not models:
            return
        
        combined_data = pd.concat(data.values()) if len(data) > 1 else list(data.values())[0]
        
        dag_config = self.tradsl_config.get('_execution_order', {})
        
        try:
            features = compute_features(dag_config, combined_data)
        except Exception:
            features = combined_data
        
        for symbol, model in models.items():
            if not hasattr(model, 'train'):
                continue
            
            if symbol not in data:
                continue
            
            try:
                X = features.iloc[:-1] if len(features) > 1 else features
                y = None
                
                if f"{symbol}_target" in features.columns:
                    y = features[f"{symbol}_target"].iloc[:-1]
                
                if y is not None and len(X) > 0 and len(y) > 0:
                    model.train(X, y)
            except Exception as e:
                pass
    
    def _extract_results(self) -> BacktestResult:
        """Extract results from NT analyzer."""
        equity_curve = pd.Series()
        trades = []
        portfolio_history = []
        signals_history = []
        summary = {}
        
        if self.analyzer is not None:
            try:
                returns = self.analyzer.returns()
                if returns is not None and not returns.empty:
                    equity_curve = (1 + returns).cumprod()
            except Exception:
                pass
            
            try:
                trades = self._extract_trades()
            except Exception:
                pass
            
            summary = self._extract_summary()
        
        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            portfolio_history=portfolio_history,
            signals_history=signals_history,
            summary=summary,
        )
    
    def _extract_trades(self) -> List[Dict[str, Any]]:
        """Extract trades from analyzer."""
        trades = []
        
        try:
            positions = self.analyzer.positions()
            if positions:
                for pos in positions:
                    trades.append({
                        'symbol': str(pos.instrument_id),
                        'quantity': float(pos.quantity),
                        'pnl': float(pos.pnl) if hasattr(pos, 'pnl') else 0.0,
                    })
        except Exception:
            pass
        
        return trades
    
    def _extract_summary(self) -> Dict[str, Any]:
        """Extract performance summary."""
        summary = {
            'starting_balance': float(self.starting_balance),
        }
        
        if self.analyzer is None:
            return summary
        
        try:
            returns = self.analyzer.returns()
            if returns is not None and not returns.empty:
                summary['total_return'] = float(returns.iloc[-1]) if len(returns) > 0 else 0.0
        except Exception:
            pass
        
        try:
            sharpe = self.analyzer.sharpe_ratio()
            if sharpe:
                summary['sharpe_ratio'] = float(sharpe)
        except Exception:
            pass
        
        try:
            max_dd = self.analyzer.max_drawdown()
            if max_dd:
                summary['max_drawdown'] = float(max_dd)
        except Exception:
            pass
        
        return summary
    
    def reset(self):
        """Reset the backend for re-run."""
        if self.engine:
            self.engine.reset()
        self.engine = None
        self.strategy = None
        self.analyzer = None


def run_nautilus_backtest(
    tradsl_config: Dict,
    data: Dict[str, pd.DataFrame],
    models: Optional[Dict[str, Any]] = None,
    sizer: Optional[Any] = None,
    venue: str = "BACKTEST",
    bar_type: str = "1-MINUTE-LAST-EXTERNAL",
    starting_balance: Decimal = Decimal("100000"),
    commission: Decimal = Decimal("0.001"),
    position_size: Decimal = Decimal("1"),
    train_models: bool = False,
) -> BacktestResult:
    """
    Convenience function to run a NautilusTrader backtest.
    
    Args:
        tradsl_config: Parsed tradsl config
        data: Dict mapping symbol -> OHLCV DataFrame
        models: Dict mapping symbol -> model
        sizer: Position sizer
        venue: Venue name
        bar_type: Bar type string
        starting_balance: Starting capital
        commission: Commission rate
        position_size: Position size per trade
        train_models: Whether to train models
    
    Returns:
        BacktestResult
    """
    backend = NautilusBackend(
        tradsl_config=tradsl_config,
        venue=venue,
        bar_type=bar_type,
        starting_balance=starting_balance,
        commission=commission,
        position_size=position_size,
    )
    
    return backend.run(
        data=data,
        models=models,
        sizer=sizer,
        train_models=train_models,
    )
