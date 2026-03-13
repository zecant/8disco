"""Tests for position sizers."""
import pytest
from tradsl.sizers import (
    FixedSizer, FractionalSizer, KellySizer, VolatilityTargetingSizer,
    SIZER_REGISTRY, get_sizer, VALID_PORTFOLIO_KEYS
)
from tradsl.models import TradingAction


class TestFixedSizer:
    def test_buy_action(self):
        sizer = FixedSizer(quantity=10)
        result = sizer.calculate_size(
            action=TradingAction.BUY,
            conviction=0.5,
            current_position=0.0,
            portfolio_value=100000.0,
            instrument_id="SPY",
            current_price=100.0
        )
        assert result == 10.0
    
    def test_sell_action(self):
        sizer = FixedSizer(quantity=10)
        result = sizer.calculate_size(
            action=TradingAction.SELL,
            conviction=0.5,
            current_position=0.0,
            portfolio_value=100000.0,
            instrument_id="SPY",
            current_price=100.0
        )
        assert result == -10.0
    
    def test_hold_returns_current(self):
        sizer = FixedSizer(quantity=10)
        result = sizer.calculate_size(
            action=TradingAction.HOLD,
            conviction=0.5,
            current_position=50.0,
            portfolio_value=100000.0,
            instrument_id="SPY",
            current_price=100.0
        )
        assert result == 50.0
    
    def test_flatten_returns_zero(self):
        sizer = FixedSizer(quantity=10)
        result = sizer.calculate_size(
            action=TradingAction.FLATTEN,
            conviction=0.5,
            current_position=50.0,
            portfolio_value=100000.0,
            instrument_id="SPY",
            current_price=100.0
        )
        assert result == 0.0


class TestFractionalSizer:
    def test_fractional_calculation(self):
        sizer = FractionalSizer(fraction=0.1)
        result = sizer.calculate_size(
            action=TradingAction.BUY,
            conviction=1.0,
            current_position=0.0,
            portfolio_value=100000.0,
            instrument_id="SPY",
            current_price=100.0
        )
        
        expected = (100000.0 * 0.1 * 1.0) / 100.0
        assert abs(result - expected) < 1e-6
    
    def test_conviction_scales_size(self):
        sizer = FractionalSizer(fraction=0.1)
        
        result_full = sizer.calculate_size(
            action=TradingAction.BUY,
            conviction=1.0,
            current_position=0.0,
            portfolio_value=100000.0,
            instrument_id="SPY",
            current_price=100.0
        )
        
        result_half = sizer.calculate_size(
            action=TradingAction.BUY,
            conviction=0.5,
            current_position=0.0,
            portfolio_value=100000.0,
            instrument_id="SPY",
            current_price=100.0
        )
        
        assert result_half == result_full * 0.5
    
    def test_max_position_limit(self):
        sizer = FractionalSizer(fraction=0.5, max_position=0.2)
        result = sizer.calculate_size(
            action=TradingAction.BUY,
            conviction=1.0,
            current_position=0.0,
            portfolio_value=100000.0,
            instrument_id="SPY",
            current_price=100.0
        )
        
        max_allowed = (100000.0 * 0.2) / 100.0
        assert result <= max_allowed
    
    def test_zero_portfolio(self):
        sizer = FractionalSizer(fraction=0.1)
        result = sizer.calculate_size(
            action=TradingAction.BUY,
            conviction=1.0,
            current_position=0.0,
            portfolio_value=0.0,
            instrument_id="SPY",
            current_price=100.0
        )
        assert result == 0.0
    
    def test_invalid_fraction_raises(self):
        with pytest.raises(ValueError):
            FractionalSizer(fraction=0.0)
        with pytest.raises(ValueError):
            FractionalSizer(fraction=1.5)


class TestKellySizer:
    def test_kelly_calculation(self):
        sizer = KellySizer(max_kelly=0.25, estimated_win_rate=0.5, win_loss_ratio=2.0)
        
        p = 0.5 * 1.0
        q = 1 - p
        b = 2.0
        kelly = (p * b - q) / b
        
        result = sizer.calculate_size(
            action=TradingAction.BUY,
            conviction=1.0,
            current_position=0.0,
            portfolio_value=100000.0,
            instrument_id="SPY",
            current_price=100.0
        )
        
        expected = (100000.0 * kelly * 1.0) / 100.0
        assert abs(result - expected) < 1e-6
    
    def test_max_kelly_enforced(self):
        sizer = KellySizer(max_kelly=0.1, estimated_win_rate=0.9, win_loss_ratio=2.0)
        
        result = sizer.calculate_size(
            action=TradingAction.BUY,
            conviction=1.0,
            current_position=0.0,
            portfolio_value=100000.0,
            instrument_id="SPY",
            current_price=100.0
        )
        
        max_allowed = (100000.0 * 0.1) / 100.0
        assert result <= max_allowed


class TestSizerRegistry:
    def test_get_sizer(self):
        cls = get_sizer('fractional')
        assert cls == FractionalSizer
    
    def test_registry_has_expected_sizers(self):
        expected = ['fixed', 'fractional', 'kelly', 'volatility_targeting']
        for name in expected:
            assert name in SIZER_REGISTRY
    
    def test_valid_portfolio_keys(self):
        assert 'position' in VALID_PORTFOLIO_KEYS
        assert 'portfolio_value' in VALID_PORTFOLIO_KEYS
        assert 'unrealized_pnl' in VALID_PORTFOLIO_KEYS
