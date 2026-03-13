"""
NautilusTrader Integration for TradSL

Section 13: Real NT Strategy implementation with solid position tracking.
"""
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger("tradsl.nt_integration")

from tradsl.circular_buffer import CircularBuffer
from tradsl.models import TradingAction


@dataclass
class Position:
    """
    Solid position tracking for a single instrument.
    
    Handles:
    - Multiple entry prices (average down/up)
    - Partial fills
    - Realized/unrealized P&L
    - Position tracking through all states
    """
    instrument_id: str
    quantity: float = 0.0
    entry_price: float = 0.0
    entry_value: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    _entry_prices: List[float] = field(default_factory=list)
    _entry_quantities: List[float] = field(default_factory=list)
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) < 1e-9
    
    @property
    def avg_entry_price(self) -> float:
        if self.quantity == 0:
            return 0.0
        total_qty = sum(abs(q) for q in self._entry_quantities)
        if total_qty == 0:
            return 0.0
        total_value = sum(p * abs(q) for p, q in zip(self._entry_prices, self._entry_quantities))
        return total_value / total_qty
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.entry_price if self.entry_price else 0.0
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L at current price."""
        if self.is_flat or current_price == 0:
            return 0.0
        avg_price = self.avg_entry_price
        if avg_price == 0:
            return 0.0
        return (current_price - avg_price) * self.quantity
    
    def add_fill(
        self,
        price: float,
        quantity: float,
        commission: float
    ) -> Dict[str, float]:
        """
        Add a fill to the position.
        
        Args:
            price: Fill price
            quantity: Fill quantity (+ for buy, - for sell)
            commission: Commission paid for this fill
            
        Returns:
            Dict with 'realized_pnl', 'new_quantity', 'avg_price'
        """
        self.total_commission += commission
        
        if quantity > 0:
            self._entry_prices.append(price)
            self._entry_quantities.append(quantity)
        else:
            self._reduce_position(price, abs(quantity), commission)
        
        total_qty = sum(abs(q) for q in self._entry_quantities)
        self.quantity = sum(self._entry_quantities)
        
        new_avg = self.avg_entry_price
        self.entry_price = new_avg if new_avg else price
        
        return {
            'realized_pnl': self.realized_pnl,
            'new_quantity': self.quantity,
            'avg_price': self.entry_price
        }
    
    def _reduce_position(
        self,
        price: float,
        reduce_qty: float,
        commission: float
    ) -> None:
        """Reduce position by closing some or all entries."""
        remaining = reduce_qty
        
        while remaining > 1e-9 and self._entry_prices:
            entry_qty = abs(self._entry_quantities[0])
            
            if entry_qty <= remaining:
                pnl = (price - self._entry_prices[0]) * entry_qty
                self.realized_pnl += pnl
                remaining -= entry_qty
                self._entry_prices.pop(0)
                self._entry_quantities.pop(0)
            else:
                pnl = (price - self._entry_prices[0]) * remaining
                self.realized_pnl += pnl
                self._entry_quantities[0] += remaining
                remaining = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Export position state."""
        return {
            'instrument_id': self.instrument_id,
            'quantity': self.quantity,
            'avg_entry_price': self.avg_entry_price,
            'market_value': self.market_value,
            'realized_pnl': self.realized_pnl,
            'total_commission': self.total_commission,
            'is_long': self.is_long,
            'is_short': self.is_short,
            'is_flat': self.is_flat,
        }


@dataclass
class Portfolio:
    """
    Full portfolio tracking across multiple instruments.
    
    Tracks:
    - Cash balance
    - All positions
    - High water mark
    - Drawdown
    - Rolling metrics
    """
    starting_capital: float
    cash: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    high_water_mark: float = 0.0
    equity_history: List[float] = field(default_factory=list)
    returns_history: List[float] = field(default_factory=list)
    _rolling_window: int = 252
    
    def __post_init__(self):
        self.cash = self.starting_capital
        self.high_water_mark = self.starting_capital
    
    @property
    def total_equity(self) -> float:
        """Total equity = cash + sum(market values of all positions)."""
        total_mv = sum(p.market_value for p in self.positions.values())
        return self.cash + total_mv
    
    @property
    def total_realized_pnl(self) -> float:
        """Sum of realized P&L across all positions."""
        return sum(p.realized_pnl for p in self.positions.values())
    
    def total_unrealized_pnl(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """Sum of unrealized P&L across all positions."""
        prices = current_prices or {}
        total = 0.0
        for instr_id, pos in self.positions.items():
            price = prices.get(instr_id, pos.entry_price) if pos.entry_price else 0.0
            total += pos.unrealized_pnl(price)
        return total
    
    @property
    def current_drawdown(self) -> float:
        """Current drawdown from high water mark."""
        if self.high_water_mark == 0:
            return 0.0
        return (self.high_water_mark - self.total_equity) / self.high_water_mark
    
    @property
    def rolling_sharpe(self) -> float:
        """Rolling Sharpe ratio."""
        if len(self.returns_history) < 2:
            return 0.0
        
        recent = self.returns_history[-self._rolling_window:]
        if not recent:
            return 0.0
        
        mean_ret = np.mean(recent)
        std_ret = np.std(recent)
        
        if std_ret < 1e-9:
            return 0.0
        
        return mean_ret / std_ret * np.sqrt(252)
    
    @property
    def rolling_volatility(self) -> float:
        """Rolling annualized volatility."""
        recent = self.returns_history[-self._rolling_window:]
        if len(recent) < 2:
            return 0.0
        return np.std(recent) * np.sqrt(252)
    
    def get_or_create_position(self, instrument_id: str) -> Position:
        """Get existing position or create new one."""
        if instrument_id not in self.positions:
            self.positions[instrument_id] = Position(instrument_id=instrument_id)
        return self.positions[instrument_id]
    
    def update_equity(self) -> None:
        """Update equity history and high water mark."""
        equity = self.total_equity
        self.equity_history.append(equity)
        
        if equity > self.high_water_mark:
            self.high_water_mark = equity
        
        if len(self.equity_history) >= 2:
            prev_equity = self.equity_history[-2]
            if prev_equity > 0:
                ret = (equity - prev_equity) / prev_equity
                self.returns_history.append(ret)
    
    def to_dict(self, current_prices: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Export full portfolio state."""
        prices = current_prices or {}
        
        return {
            'cash': self.cash,
            'total_equity': self.total_equity,
            'total_realized_pnl': self.total_realized_pnl,
            'total_unrealized_pnl': self.total_unrealized_pnl(prices),
            'high_water_mark': self.high_water_mark,
            'current_drawdown': self.current_drawdown,
            'rolling_sharpe': self.rolling_sharpe,
            'rolling_volatility': self.rolling_volatility,
            'positions': {k: v.to_dict() for k, v in self.positions.items()},
        }


@dataclass
class ExecutionState:
    """Runtime state for strategy execution."""
    bar_index: int = 0
    node_buffers: Dict[str, CircularBuffer] = field(default_factory=dict)
    node_outputs: Dict[str, float] = field(default_factory=dict)
    current_observation: Optional[np.ndarray] = None
    pre_fill_equity: float = 0.0
    bars_in_position: int = 0
    current_instrument: Optional[str] = None
    pending_order_id: Optional[str] = None


class TradSLNTStrategy:
    """
    NautilusTrader Strategy implementation for TradSL.
    
    This class is designed to be subclassed by a real NT Strategy.
    The user should inherit from this and nautilus_trader.trading.Strategy.
    
    Key features:
    - Solid position tracking
    - Proper order handling with LimitOrder
    - Experience recording for RL
    - Portfolio state from live tracking
    """
    
    def __init__(
        self,
        config: Any,
        agent: Any,
        sizer: Any,
        reward_function: Any,
        portfolio: Optional[Portfolio] = None,
        min_order_size: float = 1.0,
        date_blind: bool = True,
    ):
        """
        Initialize TradSL NT Strategy.
        
        Args:
            config: TradSLConfig with DAG and execution details
            agent: RL agent (ParameterizedAgent or similar)
            sizer: Position sizer
            reward_function: Reward function for RL
            portfolio: Portfolio instance (created if None)
            min_order_size: Minimum order quantity
            date_blind: Whether to strip DATE_DERIVED features
        """
        self._config = config
        self._agent = agent
        self._sizer = sizer
        self._reward_function = reward_function
        self._date_blind = date_blind
        self._min_order_size = min_order_size
        
        starting_capital = config.backtest_config.get('capital', 100000.0)
        self._portfolio = portfolio or Portfolio(starting_capital=starting_capital)
        
        self._state = ExecutionState()
        self._pending_fills: List[Dict] = []
        
        self._init_buffers()
    
    def _init_buffers(self) -> None:
        """Initialize circular buffers for all DAG nodes."""
        for node_name, size in self._config.node_buffer_sizes.items():
            self._state.node_buffers[node_name] = CircularBuffer(size)
    
    @property
    def portfolio(self) -> Portfolio:
        """Access portfolio for external use."""
        return self._portfolio
    
    @property
    def config(self) -> Any:
        """Access config for external use."""
        return self._config
    
    def on_bar(self, bar) -> None:
        """
        Process new bar from NT.
        
        Section 13.2 execution order:
        1. Push bar to source buffers
        2. Check warmup
        3. Propagate dirty flags
        4. Recompute dirty nodes
        5. Check agent update schedule
        6. Assemble observation
        7. Strip DATE_DERIVED if date_blind
        8. Get portfolio state
        9. Call agent.observe()
        10. Call sizer.calculate_size()
        11. Submit order if delta >= min_order_size
        12. Increment bar_index
        """
        instrument_id = str(bar.bar_type.instrument_id)
        self._state.current_instrument = instrument_id
        
        if self._state.bar_index < self._config.warmup_bars:
            self._push_bar_to_buffers(bar)
            self._state.bar_index += 1
            return
        
        self._state.pre_fill_equity = self._portfolio.total_equity
        
        self._push_bar_to_buffers(bar)
        
        self._recompute_dirty_nodes()
        
        if hasattr(self._agent, 'should_update'):
            if self._agent.should_update(self._state.bar_index):
                self._agent.update_policy(self._state.bar_index)
        
        observation = self._assemble_observation()
        
        if observation is None:
            self._state.bar_index += 1
            return
        
        portfolio_state = self._get_portfolio_state(bar)
        
        action, conviction = self._agent.observe(
            observation,
            portfolio_state,
            self._state.bar_index
        )
        
        if action == TradingAction.HOLD:
            self._state.bar_index += 1
            return
        
        position = self._portfolio.get_or_create_position(instrument_id)
        
        target_size = self._sizer.calculate_size(
            action=action,
            conviction=conviction,
            current_position=position.quantity,
            portfolio_value=self._portfolio.total_equity,
            instrument_id=instrument_id,
            current_price=float(bar.close),
            current_drawdown=self._portfolio.current_drawdown
        )
        
        delta = target_size - position.quantity
        
        if abs(delta) >= self._min_order_size:
            self._submit_limit_order(instrument_id, action, delta, float(bar.close))
        
        self._portfolio.update_equity()
        self._state.bar_index += 1
    
    def on_order_filled(self, event) -> None:
        """
        Process order fill from NT.
        
        Section 13.2:
        1. Update position with fill
        2. Compute reward from portfolio state change
        3. Record experience in replay buffer
        4. Log fill details
        """
        from nautilus_trader.model import enums as nt_enums
        
        instrument_id = str(event.order_id.instrument_id)
        
        fill_price = float(event.fill_price)
        fill_qty = float(event.fill_qty)
        commission = float(event.commission) if hasattr(event, 'commission') else 0.0
        
        position = self._portfolio.get_or_create_position(instrument_id)
        position.add_fill(fill_price, fill_qty, commission)
        
        trade_value = abs(fill_price * fill_qty)
        if fill_qty > 0:
            self._portfolio.cash -= trade_value + commission
        else:
            self._portfolio.cash += trade_value - commission
        
        post_fill_equity = self._portfolio.total_equity
        
        from tradsl.rewards import RewardContext
        
        ctx = RewardContext(
            fill_price=fill_price,
            fill_quantity=fill_qty,
            fill_side="buy" if fill_qty > 0 else "sell",
            commission_paid=commission,
            pre_fill_portfolio_value=self._state.pre_fill_equity,
            post_fill_portfolio_value=post_fill_equity,
            unrealized_pnl=position.unrealized_pnl(fill_price),
            realized_pnl=position.realized_pnl,
            bars_in_trade=self._state.bars_in_position,
            high_water_mark=self._portfolio.high_water_mark,
            baseline_capital=self._config.backtest_config.get('capital', 100000.0),
            current_drawdown=self._portfolio.current_drawdown
        )
        
        reward = self._reward_function.compute(ctx)
        
        if self._state.current_observation is not None:
            next_obs = self._get_next_observation()
            next_portfolio_state = self._get_portfolio_state_from_portfolio()
            
            self._agent.record_experience(
                observation=self._state.current_observation,
                portfolio_state=next_portfolio_state,
                action=TradingAction.BUY if fill_qty > 0 else TradingAction.SELL,
                conviction=1.0,
                reward=reward,
                next_observation=next_obs,
                next_portfolio_state=next_portfolio_state,
                done=False
            )
        
        if position.quantity != 0:
            self._state.bars_in_position += 1
        else:
            self._state.bars_in_position = 0
    
    def on_order_canceled(self, event) -> None:
        """Handle order cancellation."""
        self._state.pending_order_id = None
    
    def on_order_rejected(self, event) -> None:
        """Handle order rejection."""
        self._state.pending_order_id = None
    
    def _push_bar_to_buffers(self, bar) -> None:
        """Push bar OHLCV data to source node buffers."""
        close = float(bar.close)
        
        for node_name in self._config.source_nodes:
            if node_name in self._state.node_buffers:
                self._state.node_buffers[node_name].push(close)
    
    def _recompute_dirty_nodes(self) -> None:
        """Recompute dirty nodes in topological order."""
        for node_name in self._config.execution_order:
            if node_name in self._config.source_nodes:
                continue
            
            node_config = self._config.dag_config.get(node_name, {})
            inputs = node_config.get('inputs', [])
            
            input_values = []
            for inp in inputs:
                if inp in self._state.node_outputs:
                    input_values.append(self._state.node_outputs[inp])
            
            if not input_values or None in input_values:
                continue
            
            if len(input_values) == 1:
                output = self._compute_node(node_name, input_values[0])
            else:
                output = self._compute_node_multi(node_name, input_values)
            
            if output is not None:
                self._state.node_outputs[node_name] = output
                if node_name in self._state.node_buffers:
                    self._state.node_buffers[node_name].push(output)
    
    def _compute_node(self, node_name: str, input_value: float) -> Optional[float]:
        """Compute a single-input function node."""
        node_config = self._config.dag_config.get(node_name, {})
        func_name = node_config.get('function')
        params = node_config.get('params', {})
        
        if not func_name:
            return None
        
        buffer = self._state.node_buffers.get(node_name)
        if buffer is None:
            return None
        
        arr = buffer.to_array()
        if arr is None:
            return None
        
        try:
            from tradsl.functions import compute_function
            return compute_function(func_name, arr, **params)
        except Exception:
            return None
    
    def _compute_node_multi(self, node_name: str, input_values: List[float]) -> Optional[float]:
        """Compute a multi-input function node."""
        return input_values[0] if input_values else None
    
    def _assemble_observation(self) -> Optional[np.ndarray]:
        """Assemble observation vector from node outputs."""
        features = []
        
        model_config = None
        for name in self._config.model_nodes:
            model_config = self._config.dag_config.get(name, {})
            break
        
        if model_config is None:
            return None
        
        input_names = model_config.get('inputs', [])
        
        for inp in input_names:
            value = self._state.node_outputs.get(inp)
            if value is None:
                return None
            features.append(value)
        
        if not features:
            return None
        
        obs = np.array(features, dtype=np.float64)
        self._state.current_observation = obs
        return obs
    
    def _get_next_observation(self) -> Optional[np.ndarray]:
        """Get observation for next step (after fill)."""
        return self._assemble_observation()
    
    def _get_portfolio_state(self, bar) -> Dict[str, float]:
        """Get portfolio state dict for agent."""
        instrument_id = self._state.current_instrument or ""
        position = self._portfolio.positions.get(instrument_id)
        
        current_price = float(bar.close) if bar else 0.0
        
        pos_qty = position.quantity if position else 0.0
        pos_mv = position.market_value if position else 0.0
        pos_unr = position.unrealized_pnl(current_price) if position else 0.0
        pos_rlzd = position.realized_pnl if position else 0.0
        pos_avg = position.avg_entry_price if position else 0.0
        
        total_eq = self._portfolio.total_equity
        
        return {
            'position': pos_qty,
            'position_value': pos_mv,
            'unrealized_pnl': pos_unr,
            'unrealized_pnl_pct': pos_unr / total_eq if total_eq > 0 else 0.0,
            'realized_pnl': pos_rlzd,
            'portfolio_value': total_eq,
            'cash': self._portfolio.cash,
            'drawdown': self._portfolio.current_drawdown,
            'time_in_trade': self._state.bars_in_position,
            'high_water_mark': self._portfolio.high_water_mark,
            'position_weight': pos_mv / total_eq if total_eq > 0 else 0.0,
            'rolling_sharpe': self._portfolio.rolling_sharpe,
            'rolling_volatility': self._portfolio.rolling_volatility,
        }
    
    def _get_portfolio_state_from_portfolio(self) -> Dict[str, float]:
        """Get portfolio state without bar (for experience recording)."""
        instr = self._state.current_instrument or ""
        position = self._portfolio.positions.get(instr)
        
        total_eq = self._portfolio.total_equity
        
        return {
            'position': position.quantity if position else 0.0,
            'position_value': position.market_value if position else 0.0,
            'unrealized_pnl': position.unrealized_pnl(position.entry_price) if position and position.entry_price else 0.0,
            'realized_pnl': position.realized_pnl if position else 0.0,
            'portfolio_value': total_eq,
            'cash': self._portfolio.cash,
            'drawdown': self._portfolio.current_drawdown,
            'time_in_trade': self._state.bars_in_position,
            'high_water_mark': self._portfolio.high_water_mark,
            'rolling_sharpe': self._portfolio.rolling_sharpe,
            'rolling_volatility': self._portfolio.rolling_volatility,
        }
    
    def _submit_limit_order(
        self,
        instrument_id: str,
        action: TradingAction,
        quantity: float,
        reference_price: float
    ) -> None:
        """
        Submit a limit order via NT.
        
        This creates and submits a real LimitOrder through NT's order system.
        """
        from nautilus_trader.model.enums import OrderSide, TimeInForce
        from nautilus_trader.model import Quantity, Price
        from nautilus_trader.model.orders import LimitOrder
        from nautilus_trader.core import UUID4
        
        side = OrderSide.BUY if quantity > 0 else OrderSide.SELL
        
        order = LimitOrder(
            instrument_id=instrument_id,
            client_order_id=UUID4(),
            side=side,
            quantity=Quantity(abs(quantity)),
            price=Price(reference_price),
            time_in_force=TimeInForce.GTC,
            emulation_trigger=None,
            trigger_type=None,
            ts_init=0,
        )
        
        self._state.pending_order_id = str(order.client_order_id)
        
        self.submit_order(order)
    
    def on_start(self) -> None:
        """Called by NT on strategy start."""
        self._state = ExecutionState()
        self._init_buffers()
    
    def on_stop(self) -> None:
        """Called by NT on strategy stop."""
        self.cancel_all_orders()
    
    def on_save(self) -> Dict[str, Any]:
        """Save strategy state for checkpointing."""
        return {
            'bar_index': self._state.bar_index,
            'portfolio': self._portfolio.to_dict({}),
            'node_outputs': self._state.node_outputs,
        }
    
    def on_load(self, state: Dict[str, Any]) -> None:
        """Load strategy state from checkpoint."""
        self._state.bar_index = state.get('bar_index', 0)
        self._state.node_outputs = state.get('node_outputs', {})
    
    def get_results(self) -> Dict[str, Any]:
        """Get final results for analysis."""
        return {
            'equity_curve': np.array(self._portfolio.equity_history),
            'returns': np.array(self._portfolio.returns_history),
            'final_equity': self._portfolio.total_equity,
            'total_realized_pnl': self._portfolio.total_realized_pnl,
            'total_bars': self._state.bar_index,
            'portfolio_state': self._portfolio.to_dict({}),
        }


class TradSLStrategy(TradSLNTStrategy):
    """
    Standalone TradSL Strategy for backtesting without NT.
    
    This class can be used directly for backtesting without
    requiring a full NautilusTrader instance.
    """
    
    def __init__(
        self,
        config: Any,
        agent: Any,
        sizer: Any,
        reward_function: Any,
        min_order_size: float = 1.0,
        date_blind: bool = True,
    ):
        super().__init__(
            config=config,
            agent=agent,
            sizer=sizer,
            reward_function=reward_function,
            portfolio=None,
            min_order_size=min_order_size,
            date_blind=date_blind,
        )
        
        self._bars_processed = 0
    
    def process_bar(
        self,
        close: float,
        high: Optional[float] = None,
        low: Optional[float] = None,
        open_price: Optional[float] = None,
        volume: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Process a single bar (for backtesting without NT).
        
        Args:
            close: Closing price
            high: High price (defaults to close)
            low: Low price (defaults to close)
            open_price: Open price (defaults to close)
            volume: Volume (defaults to 0)
            timestamp: Bar timestamp
            
        Returns:
            Dict with execution results
        """
        instrument_id = self._state.current_instrument or "TEST"
        
        class MockBar:
            def __init__(self, c, h, l, o, v, ts, instr):
                self.close = c
                self.high = h if h else c
                self.low = l if l else c
                self.open = o if o else c
                self.volume = v if v else 0.0
                self.ts_init = ts if ts else 0
                
                class MockBarType:
                    def __init__(self, iid):
                        self.instrument_id = iid
                
                self.bar_type = MockBarType(instr)
        
        bar = MockBar(
            close,
            high,
            low,
            open_price,
            volume,
            timestamp.timestamp() * 1e9 if timestamp else self._bars_processed * 1e9,
            instrument_id
        )
        
        self.on_bar(bar)
        self._bars_processed += 1
        
        return {
            'bar_index': self._state.bar_index,
            'equity': self._portfolio.total_equity,
            'position': self._portfolio.positions.get(
                instrument_id,
                Position(instrument_id=instrument_id)
            ).to_dict(),
        }
    
    def run_backtest(
        self,
        prices: List[float],
        warmup_bars: int = 20
    ) -> Dict[str, Any]:
        """
        Run backtest on price series.
        
        Args:
            prices: List of closing prices
            warmup_bars: Number of warmup bars
            
        Returns:
            Dict with backtest results
        """
        self._config.warmup_bars = warmup_bars
        self._state.bar_index = 0
        
        results = []
        for i, price in enumerate(prices):
            self._state.current_instrument = "TEST"
            result = self.process_bar(close=price)
            results.append(result)
        
        return self.get_results()
