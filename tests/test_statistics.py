"""
Tests for statistics module.
"""
import pytest
import numpy as np
from tradsl.statistics import (
    compute_metrics,
    PerformanceMetrics,
    ANNUALIZATION_FACTORS
)


class TestAnnualizationFactors:
    """Tests for annualization factors."""
    
    def test_all_frequencies_defined(self):
        """Test all expected frequencies are defined."""
        expected = ['1min', '5min', '15min', '30min', '1h', '4h', '1d', '1wk', '1mo']
        for freq in expected:
            assert freq in ANNUALIZATION_FACTORS
    
    def test_daily_factor_is_252(self):
        """Test daily uses 252, not 365."""
        assert ANNUALIZATION_FACTORS['1d'] == 252
    
    def test_minute_factors(self):
        """Test minute factors are reasonable."""
        assert ANNUALIZATION_FACTORS['1min'] > ANNUALIZATION_FACTORS['1h']
        assert ANNUALIZATION_FACTORS['1h'] > ANNUALIZATION_FACTORS['1d']


class TestComputeMetrics:
    """Tests for compute_metrics function."""
    
    def test_empty_equity_returns_zeros(self):
        """Test empty equity curve returns zero metrics."""
        result = compute_metrics(np.array([]), [], '1d')
        assert result.total_return == 0.0
        assert result.sharpe_ratio == 0.0
    
    def test_single_value_returns_zeros(self):
        """Test single value returns zero metrics."""
        result = compute_metrics(np.array([100000.0]), [], '1d')
        assert result.total_return == 0.0
    
    def test_positive_return(self):
        """Test positive return calculation."""
        equity = np.array([100000, 110000, 121000])
        trades = [
            {'pnl': 10000, 'duration': 5},
            {'pnl': 10000, 'duration': 10}
        ]
        result = compute_metrics(equity, trades, '1d')
        
        assert result.total_return > 0
        assert result.n_trades == 2
    
    def test_negative_return(self):
        """Test negative return calculation."""
        equity = np.array([100000, 90000, 81000])
        trades = [{'pnl': -10000, 'duration': 5}]
        result = compute_metrics(equity, trades, '1d')
        
        assert result.total_return < 0
    
    def test_sharpe_calculation(self):
        """Test Sharpe ratio is calculated correctly."""
        # Constant returns should give infinite Sharpe
        equity = np.array([100000, 105000, 110000, 115000, 120000])
        trades = []
        result = compute_metrics(equity, trades, '1d')
        
        # With constant positive returns, Sharpe should be positive
        assert result.sharpe_ratio >= 0
    
    def test_drawdown_calculation(self):
        """Test drawdown is calculated."""
        equity = np.array([100000, 120000, 90000, 110000])
        trades = []
        result = compute_metrics(equity, trades, '1d')
        
        assert result.max_drawdown > 0
        # Max drawdown should be (120000 - 90000) / 120000 = 0.25
        assert result.max_drawdown > 0.2
        assert result.max_drawdown < 0.3
    
    def test_win_rate(self):
        """Test win rate calculation."""
        equity = np.array([100000, 110000, 105000, 115000])
        trades = [
            {'pnl': 10000, 'duration': 5},
            {'pnl': -5000, 'duration': 3},
            {'pnl': 10000, 'duration': 4}
        ]
        result = compute_metrics(equity, trades, '1d')
        
        assert result.win_rate == pytest.approx(2/3, rel=0.01)
    
    def test_profit_factor(self):
        """Test profit factor calculation."""
        equity = np.array([100000, 110000, 105000, 115000])
        trades = [
            {'pnl': 10000, 'duration': 5},
            {'pnl': -5000, 'duration': 3},
        ]
        result = compute_metrics(equity, trades, '1d')
        
        # Profit factor = wins / losses = 10000 / 5000 = 2.0
        assert result.profit_factor == pytest.approx(2.0, rel=0.01)
    
    def test_empty_trades(self):
        """Test with no trades."""
        equity = np.array([100000, 105000, 110000])
        result = compute_metrics(equity, [], '1d')
        
        assert result.n_trades == 0
        assert result.win_rate == 0.0
        assert result.profit_factor == 0.0
    
    def test_to_dict(self):
        """Test conversion to dict."""
        equity = np.array([100000, 110000])
        trades = [{'pnl': 10000, 'duration': 5}]
        result = compute_metrics(equity, trades, '1d')
        
        d = result.to_dict()
        assert isinstance(d, dict)
        assert 'total_return' in d
        assert 'sharpe_ratio' in d
        assert 'max_drawdown' in d


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""
    
    def test_dataclass_creation(self):
        """Test creating PerformanceMetrics."""
        m = PerformanceMetrics(
            total_return=0.1,
            cagr=0.08,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=0.5,
            max_drawdown=0.15,
            avg_drawdown=0.08,
            max_drawdown_duration=30,
            avg_drawdown_duration=15,
            win_rate=0.6,
            profit_factor=1.5,
            avg_win_loss=1.2,
            n_trades=100,
            avg_holding_period=5.0
        )
        
        assert m.total_return == 0.1
        assert m.sharpe_ratio == 1.5
        assert m.n_trades == 100
    
    def test_dataclass_immutable(self):
        """Test that dataclass is properly defined."""
        m = PerformanceMetrics(
            total_return=0.1,
            cagr=0.08,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=0.5,
            max_drawdown=0.15,
            avg_drawdown=0.08,
            max_drawdown_duration=30,
            avg_drawdown_duration=15,
            win_rate=0.6,
            profit_factor=1.5,
            avg_win_loss=1.2,
            n_trades=100,
            avg_holding_period=5.0
        )
        
        # Should have all expected fields
        assert hasattr(m, 'total_return')
        assert hasattr(m, 'cagr')
        assert hasattr(m, 'sharpe_ratio')
        assert hasattr(m, 'max_drawdown')
        assert hasattr(m, 'win_rate')
