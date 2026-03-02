"""
Main backtest engine orchestrator.

Coordinates data loading, feature computation, signal generation,
position sizing, and execution in a modular pipeline.
"""

from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np

from tradsl.signals import TradingSignal, SignalBatch, SignalType, TradingAction
from tradsl.sizing import PositionSizer, AllocationPlan, Allocation, create_sizer
from tradsl.portfolio.tracker import PortfolioTracker, PortfolioSnapshot
from .execution import (
    ExecutionBackend, 
    create_execution_backend,
    Order, 
    OrderSide, 
    OrderType,
    Fill
)


@dataclass
class BacktestConfig:
    """Configuration for backtest run."""
    start: datetime
    end: datetime
    starting_cash: float = 100000.0
    commission_rate: float = 0.001
    slippage: float = 0.0
    
    frequency: str = '1D'
    symbols: List[str] = field(default_factory=list)
    
    execution_backend: str = 'immediate'
    execution_params: Dict[str, Any] = field(default_factory=dict)
    
    sizer_type: str = 'equal'
    sizer_params: Dict[str, Any] = field(default_factory=dict)
    
    record_portfolio_features: bool = True
    record_trades: bool = True
    
    min_samples_for_training: int = 100
    
    on_bar_callback: Optional[Callable] = None
    on_trade_callback: Optional[Callable] = None


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    equity_curve: pd.Series
    trades: List[Dict[str, Any]]
    portfolio_history: List[Dict[str, Any]]
    signals_history: List[Dict[str, Any]]
    summary: Dict[str, Any]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert portfolio history to DataFrame."""
        return pd.DataFrame(self.portfolio_history)


class BacktestEngine:
    """
    Main backtest orchestration engine.
    
    Modular design separating:
    - Data layer (adapters)
    - Feature layer (feature computation)
    - Signal layer (models)
    - Sizing layer (position sizers)
    - Execution layer (order execution)
    - Portfolio layer (position tracking)
    """
    
    def __init__(
        self,
        config: BacktestConfig,
        data: pd.DataFrame,
        models: Optional[Dict[str, Any]] = None,
        features_func: Optional[Callable] = None,
    ):
        """
        Initialize backtest engine.
        
        Args:
            config: Backtest configuration
            data: Market data DataFrame with MultiIndex (symbol, timestamp) or single symbol
            models: Dict of {symbol: model} for signal generation
            features_func: Optional function to compute features from data
        """
        self.config = config
        self.data = data
        self.models = models or {}
        self.features_func = features_func
        
        self._initialize_components()
        
        self._results: Optional[BacktestResult] = None
    
    def _initialize_components(self):
        """Initialize all components."""
        self.execution = create_execution_backend(
            self.config.execution_backend,
            commission_rate=self.config.commission_rate,
            slippage_bps=self.config.slippage * 10000,
            **self.config.execution_params
        )
        
        self.portfolio = PortfolioTracker(
            starting_cash=self.config.starting_cash,
            commission_rate=self.config.commission_rate,
            slippage=self.config.slippage,
            symbols=self.config.symbols
        )
        
        self.sizer = create_sizer(
            self.config.sizer_type,
            **self.config.sizer_params
        )
        
        self._trades: List[Dict[str, Any]] = []
        self._signals_history: List[Dict[str, Any]] = []
        self._portfolio_history: List[Dict[str, Any]] = []
    
    def run(self) -> BacktestResult:
        """
        Run the backtest.
        
        Returns:
            BacktestResult with equity curve, trades, and history
        """
        if self.data.empty:
            return self._empty_result()
        
        bar_iterator = self._get_bar_iterator()
        
        for bar_data in bar_iterator:
            self._process_bar(bar_data)
        
        return self._build_results()
    
    def _get_bar_iterator(self):
        """Get iterator over bars."""
        if isinstance(self.data.index, pd.MultiIndex):
            return self.data.groupby(level=0)
        else:
            return [(self.data['symbol'].iloc[0] if 'symbol' in self.data.columns else 'default', self.data)]
    
    def _process_bar(self, bar_data):
        """Process a single bar."""
        if isinstance(bar_data, tuple):
            symbol, df = bar_data
            if df.empty:
                return
            row = df.iloc[-1] if len(df) > 0 else None
            timestamp = df.index[-1] if hasattr(df.index, '__getitem__') else None
        else:
            symbol = bar_data.get('symbol', 'default')
            row = bar_data
            timestamp = bar_data.get('timestamp')
        
        if row is None:
            return
        
        current_prices = self._extract_prices(row, symbol)
        
        self.portfolio.update_prices(current_prices)
        
        features = self._compute_features(row, symbol)
        
        signals = self._generate_signals(features, symbol)
        
        allocations = self._calculate_allocations(
            signals, 
            self.portfolio.equity,
            self.portfolio.get_current_quantities(),
            current_prices
        )
        
        fills = self._execute_allocations(allocations, current_prices, timestamp)
        
        self._apply_fills_to_portfolio(fills, timestamp)
        
        if self.config.record_trades and fills:
            for fill in fills:
                self._trades.append({
                    'timestamp': timestamp,
                    'symbol': fill.symbol,
                    'side': fill.side.value,
                    'quantity': fill.quantity,
                    'price': fill.price,
                    'commission': fill.commission,
                    'slippage': fill.slippage,
                })
        
        portfolio_features = self.portfolio.get_snapshot(timestamp).to_features()
        
        self._signals_history.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'signals': {sig.symbol: sig.to_dict() for sig in signals.signals.values()},
            'allocations': allocations.get_weights(),
            'portfolio_features': portfolio_features,
        })
        
        if self.config.record_portfolio_features:
            self._portfolio_history.append({
                'timestamp': timestamp,
                'equity': self.portfolio.equity,
                'cash': self.portfolio.cash,
                'position_value': self.portfolio.position_value,
                'total_pnl': self.portfolio.total_pnl,
                'positions': self.portfolio.get_current_quantities(),
                **portfolio_features
            })
        
        if self.config.on_bar_callback:
            self.config.on_bar_callback(timestamp, signals, allocations, self.portfolio.get_snapshot(timestamp))
    
    def _extract_prices(self, row, symbol: str) -> Dict[str, float]:
        """Extract current prices from bar data."""
        if isinstance(row, pd.Series):
            price = float(row.get('close', row.get('price', 0)))
        else:
            price = float(row.get('close', row.get('price', 0)))
        
        return {symbol: price}
    
    def _compute_features(self, row, symbol: str) -> Dict[str, float]:
        """Compute features for the bar."""
        if self.features_func:
            return self.features_func(row, symbol)
        
        if isinstance(row, pd.Series):
            return row.to_dict()
        return dict(row)
    
    def _generate_signals(self, features: Dict[str, float], symbol: str) -> SignalBatch:
        """Generate trading signals from models."""
        batch = SignalBatch()
        
        if not self.models:
            return batch
        
        model = self.models.get(symbol)
        if model is None:
            return batch
        
        if not hasattr(model, 'predict'):
            return batch
        
        try:
            X = pd.DataFrame([features])
            model_output = model.predict(X)
            
            if isinstance(model_output, dict):
                signal = TradingSignal.from_model_output(symbol, model_output)
            elif isinstance(model_output, TradingSignal):
                signal = model_output
            else:
                signal = TradingSignal.hold(symbol)
            
            batch.add(signal)
        
        except Exception:
            pass
        
        return batch
    
    def _calculate_allocations(
        self,
        signals: SignalBatch,
        portfolio_value: float,
        current_positions: Dict[str, float],
        prices: Dict[str, float]
    ) -> AllocationPlan:
        """Calculate position allocations from signals."""
        if signals.signals:
            return self.sizer.calculate(
                signals,
                portfolio_value,
                current_positions,
                prices
            )
        return AllocationPlan()
    
    def _execute_allocations(
        self,
        allocations: AllocationPlan,
        prices: Dict[str, float],
        timestamp: Any
    ) -> List[Fill]:
        """Execute allocations through execution backend."""
        fills = []
        
        for allocation in allocations.get_actionable():
            symbol = allocation.symbol
            price = prices.get(symbol, 0)
            
            if price <= 0:
                continue
            
            side = OrderSide.BUY if allocation.action == TradingAction.BUY else OrderSide.SELL
            
            order = Order(
                order_id=f"ORD_{len(self._trades)}",
                symbol=symbol,
                side=side,
                quantity=abs(allocation.target_quantity),
                order_type=OrderType.MARKET,
                metadata={'current_price': price},
                timestamp=timestamp
            )
            
            executed_order = self.execution.submit_order(order)
            
            if executed_order.is_filled():
                new_fills = self.execution.get_fills()
                fills.extend(new_fills)
        
        return fills
    
    def _apply_fills_to_portfolio(self, fills: List[Fill], timestamp: Any):
        """Apply fills to portfolio."""
        for fill in fills:
            quantity = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
            self.portfolio.execute_trade(
                fill.symbol,
                quantity,
                fill.price,
                timestamp
            )
    
    def _build_results(self) -> BacktestResult:
        """Build final results."""
        equity_curve = pd.Series(
            [s['equity'] for s in self._portfolio_history],
            index=[s['timestamp'] for s in self._portfolio_history]
        )
        
        returns = equity_curve.pct_change().dropna()
        
        summary = {
            'starting_cash': self.config.starting_cash,
            'ending_cash': self.portfolio.cash,
            'ending_equity': self.portfolio.equity,
            'total_return': (self.portfolio.equity - self.config.starting_cash) / self.config.starting_cash,
            'total_pnl': self.portfolio.total_pnl,
            'num_trades': len(self._trades),
            'num_bars': len(self._portfolio_history),
        }
        
        if len(returns) > 0:
            summary['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            summary['max_drawdown'] = self._calculate_max_drawdown(equity_curve)
            summary['volatility'] = returns.std() * np.sqrt(252)
        else:
            summary['sharpe_ratio'] = 0
            summary['max_drawdown'] = 0
            summary['volatility'] = 0
        
        return BacktestResult(
            equity_curve=equity_curve,
            trades=self._trades,
            portfolio_history=self._portfolio_history,
            signals_history=self._signals_history,
            summary=summary
        )
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        return drawdown.min()
    
    def _empty_result(self) -> BacktestResult:
        """Return empty result."""
        return BacktestResult(
            equity_curve=pd.Series(),
            trades=[],
            portfolio_history=[],
            signals_history=[],
            summary={}
        )
    
    def get_results(self) -> Optional[BacktestResult]:
        """Get backtest results."""
        return self._results
    
    def reset(self):
        """Reset engine for re-run."""
        self._initialize_components()
        self._results = None


def run_backtest(
    data: pd.DataFrame,
    models: Dict[str, Any],
    start: Union[datetime, str],
    end: Union[datetime, str],
    symbols: List[str],
    starting_cash: float = 100000,
    commission_rate: float = 0.001,
    slippage: float = 0.0,
    sizer_type: str = 'equal',
    sizer_params: Optional[Dict[str, Any]] = None,
    features_func: Optional[Callable] = None,
    **kwargs
) -> BacktestResult:
    """
    Convenience function to run a backtest.
    
    Args:
        data: Market data DataFrame
        models: Dict of {symbol: model}
        start: Start datetime
        end: End datetime
        symbols: List of trading symbols
        starting_cash: Starting capital
        commission_rate: Commission rate
        slippage: Slippage
        sizer_type: Position sizer type
        sizer_params: Position sizer params
        features_func: Feature computation function
    
    Returns:
        BacktestResult
    """
    if isinstance(start, str):
        start = pd.to_datetime(start)
    if isinstance(end, str):
        end = pd.to_datetime(end)
    
    config = BacktestConfig(
        start=start,
        end=end,
        starting_cash=starting_cash,
        commission_rate=commission_rate,
        slippage=slippage,
        symbols=symbols,
        sizer_type=sizer_type,
        sizer_params=sizer_params or {},
    )
    
    engine = BacktestEngine(
        config=config,
        data=data,
        models=models,
        features_func=features_func,
    )
    
    return engine.run()
