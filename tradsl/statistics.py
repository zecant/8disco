"""
Performance Statistics for TradSL

Section 17: Metrics computation and reporting.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


# Annualization factors per frequency (Section 17.3)
ANNUALIZATION_FACTORS = {
    '1min': 252 * 390,
    '5min': 252 * 78,
    '15min': 252 * 26,
    '30min': 252 * 13,
    '1h': 252 * 6.5,
    '4h': 252 * 1.5,
    '1d': 252,
    '1wk': 52,
    '1mo': 12
}


@dataclass
class PerformanceMetrics:
    """Computed performance metrics."""
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int
    avg_drawdown_duration: float
    win_rate: float
    profit_factor: float
    avg_win_loss: float
    n_trades: int
    avg_holding_period: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'total_return': self.total_return,
            'cagr': self.cagr,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'avg_drawdown': self.avg_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'avg_drawdown_duration': self.avg_drawdown_duration,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_win_loss': self.avg_win_loss,
            'n_trades': self.n_trades,
            'avg_holding_period': self.avg_holding_period
        }


def compute_metrics(
    equity_curve: np.ndarray,
    trades: List[Dict[str, Any]],
    frequency: str = '1d'
) -> PerformanceMetrics:
    """
    Compute all performance metrics.
    
    Args:
        equity_curve: Portfolio value at each bar
        trades: List of trade dicts with 'pnl', 'duration'
        frequency: Data frequency for annualization
    
    Returns:
        PerformanceMetrics object
    """
    if len(equity_curve) < 2:
        return _empty_metrics()
    
    returns = np.diff(equity_curve) / equity_curve[:-1]
    returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
    
    if len(returns) == 0:
        return _empty_metrics()
    
    periods_per_year = ANNUALIZATION_FACTORS.get(frequency, 252)
    
    total_ret = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    
    n_years = len(equity_curve) / periods_per_year
    cagr = (equity_curve[-1] / equity_curve[0]) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    sharpe = _sharpe_ratio(returns, periods_per_year)
    sortino = _sortino_ratio(returns, periods_per_year)
    
    max_dd, avg_dd = _drawdown_metrics(equity_curve)
    
    dd_duration, avg_dd_duration = _drawdown_duration(equity_curve)
    
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    win_rate, profit_factor, avg_win_loss = _trade_metrics(trades)
    
    avg_holding = np.mean([t.get('duration', 0) for t in trades]) if trades else 0.0
    
    return PerformanceMetrics(
        total_return=total_ret,
        cagr=cagr,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        avg_drawdown=avg_dd,
        max_drawdown_duration=dd_duration,
        avg_drawdown_duration=avg_dd_duration,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win_loss=avg_win_loss,
        n_trades=len(trades),
        avg_holding_period=avg_holding
    )


def _sharpe_ratio(returns: np.ndarray, periods_per_year: int) -> float:
    """Compute Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=0)
    
    if std_ret == 0:
        return 0.0
    
    return mean_ret / std_ret * np.sqrt(periods_per_year)


def _sortino_ratio(returns: np.ndarray, periods_per_year: int) -> float:
    """Compute Sortino ratio."""
    if len(returns) < 2:
        return 0.0
    
    mean_ret = np.mean(returns)
    neg_returns = returns[returns < 0]
    
    if len(neg_returns) == 0:
        return 0.0
    
    downside_std = np.std(neg_returns, ddof=0)
    
    if downside_std == 0:
        return 0.0
    
    return mean_ret / downside_std * np.sqrt(periods_per_year)


def _drawdown_metrics(equity: np.ndarray) -> tuple:
    """Compute max and average drawdown."""
    if len(equity) < 2:
        return 0.0, 0.0
    
    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / running_max
    
    max_dd = float(np.max(drawdowns))
    avg_dd = float(np.mean(drawdowns[drawdowns > 0])) if np.any(drawdowns > 0) else 0.0
    
    return max_dd, avg_dd


def _drawdown_duration(equity: np.ndarray) -> tuple:
    """Compute max and average drawdown duration in bars."""
    if len(equity) < 2:
        return 0, 0.0
    
    running_max = np.maximum.accumulate(equity)
    in_drawdown = equity < running_max
    
    if not np.any(in_drawdown):
        return 0, 0.0
    
    durations = []
    current_duration = 0
    
    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
        else:
            if current_duration > 0:
                durations.append(current_duration)
            current_duration = 0
    
    if current_duration > 0:
        durations.append(current_duration)
    
    if not durations:
        return 0, 0.0
    
    return max(durations), np.mean(durations)


def _trade_metrics(trades: List[Dict[str, Any]]) -> tuple:
    """Compute win rate, profit factor, avg win/loss."""
    if not trades:
        return 0.0, 0.0, 0.0
    
    pnls = [t.get('pnl', 0) for t in trades]
    
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    
    win_rate = len(wins) / len(pnls) if pnls else 0.0
    
    total_wins = sum(wins) if wins else 0.0
    total_losses = abs(sum(losses)) if losses else 0.0
    
    profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
    
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = abs(np.mean(losses)) if losses else 0.0
    
    avg_win_loss = avg_win / avg_loss if avg_loss > 0 else 0.0
    
    return win_rate, profit_factor, avg_win_loss


def _empty_metrics() -> PerformanceMetrics:
    """Return empty metrics."""
    return PerformanceMetrics(
        total_return=0.0,
        cagr=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        calmar_ratio=0.0,
        max_drawdown=0.0,
        avg_drawdown=0.0,
        max_drawdown_duration=0,
        avg_drawdown_duration=0.0,
        win_rate=0.0,
        profit_factor=0.0,
        avg_win_loss=0.0,
        n_trades=0,
        avg_holding_period=0.0
    )
