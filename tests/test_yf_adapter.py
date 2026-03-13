"""Tests for YFinance adapter."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from tradsl.yf_adapter import YFinanceAdapter, create_yf_adapter
from tradsl.adapters import AdapterError, SymbolNotFound, DateRangeTruncated


class TestYFinanceAdapterInit:
    def test_init_default(self):
        adapter = YFinanceAdapter()
        
        assert adapter.interval == '1d'
        assert adapter.max_workers == 5
        assert adapter.timeout == 30
        assert adapter.retry_count == 3
    
    def test_init_custom(self):
        adapter = YFinanceAdapter(
            interval='1h',
            max_workers=10,
            timeout=60,
            retry_count=5
        )
        
        assert adapter.interval == '1h'
        assert adapter.max_workers == 10
        assert adapter.timeout == 60
        assert adapter.retry_count == 5


class TestYFinanceAdapterFrequency:
    def test_frequency_map(self):
        adapter = YFinanceAdapter()
        
        assert adapter.FREQUENCY_MAP['1min'] == '1m'
        assert adapter.FREQUENCY_MAP['5min'] == '5m'
        assert adapter.FREQUENCY_MAP['1h'] == '1h'
        assert adapter.FREQUENCY_MAP['1d'] == '1d'
    
    def test_valid_frequencies(self):
        adapter = YFinanceAdapter()
        
        assert adapter.supports_frequency('1min') is True
        assert adapter.supports_frequency('5min') is True
        assert adapter.supports_frequency('15min') is True
        assert adapter.supports_frequency('30min') is True
        assert adapter.supports_frequency('1h') is True
        assert adapter.supports_frequency('4h') is True
        assert adapter.supports_frequency('1d') is True
        assert adapter.supports_frequency('1wk') is True
        assert adapter.supports_frequency('1mo') is True
    
    def test_invalid_frequency(self):
        adapter = YFinanceAdapter()
        
        assert adapter.supports_frequency('2min') is False
        assert adapter.supports_frequency('unknown') is False
    
    def test_max_lookback(self):
        adapter = YFinanceAdapter()
        
        assert adapter.max_lookback('1min') is not None
        assert adapter.max_lookback('1d') is None
        assert adapter.max_lookback('1wk') is None


class TestYFinanceAdapterValidation:
    def test_validate_symbol_mock(self):
        adapter = YFinanceAdapter()
        
        result = adapter.validate_symbol('SPY')
        
        assert isinstance(result, bool)


class TestYFinanceDataProcessing:
    def test_process_dataframe_columns(self):
        adapter = YFinanceAdapter()
        
        df = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [99.0, 100.0, 101.0],
            'Close': [103.0, 104.0, 105.0],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2024-01-01', periods=3, tz='UTC'))
        
        result = adapter._process_dataframe(df, 'TEST')
        
        assert 'open' in result.columns
        assert 'high' in result.columns
        assert 'low' in result.columns
        assert 'close' in result.columns
        assert 'volume' in result.columns
        assert result.index.tz is not None
    
    def test_process_dataframe_lowercase_columns(self):
        adapter = YFinanceAdapter()
        
        df = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [105.0, 106.0],
            'low': [99.0, 100.0],
            'close': [103.0, 104.0],
            'volume': [1000000, 1100000]
        }, index=pd.date_range('2024-01-01', periods=2, tz='UTC'))
        
        result = adapter._process_dataframe(df, 'TEST')
        
        assert 'open' in result.columns
    
    def test_process_dataframe_missing_column_raises(self):
        adapter = YFinanceAdapter()
        
        df = pd.DataFrame({
            'Open': [100.0],
            'High': [105.0],
            'Close': [103.0],
            'Volume': [1000000]
        }, index=pd.date_range('2024-01-01', periods=1, tz='UTC'))
        
        with pytest.raises(AdapterError):
            adapter._process_dataframe(df, 'TEST')


class TestYFinanceDateValidation:
    def test_validate_date_range_empty(self):
        adapter = YFinanceAdapter()
        
        df = pd.DataFrame({
            'open': [100.0],
            'high': [105.0],
            'low': [99.0],
            'close': [103.0],
            'volume': [1000000]
        }, index=[pd.Timestamp('2024-01-01', tz='UTC')])
        
        with pytest.raises(DateRangeTruncated):
            adapter._validate_date_range(
                df,
                pd.Timestamp('2024-01-01', tz='UTC'),
                pd.Timestamp('2024-12-31', tz='UTC'),
                'TEST'
            )
    
    def test_validate_date_range_partial(self):
        adapter = YFinanceAdapter()
        
        df = pd.DataFrame({
            'open': [100.0],
            'high': [105.0],
            'low': [99.0],
            'close': [103.0],
            'volume': [1000000]
        }, index=[pd.Timestamp('2024-06-01', tz='UTC')])
        
        with pytest.raises(DateRangeTruncated):
            adapter._validate_date_range(
                df,
                pd.Timestamp('2024-01-01', tz='UTC'),
                pd.Timestamp('2024-12-31', tz='UTC'),
                'TEST'
            )


class TestYFinanceFactory:
    def test_create_yf_adapter(self):
        adapter = create_yf_adapter(interval='1h', timeout=60)
        
        assert isinstance(adapter, YFinanceAdapter)
        assert adapter.interval == '1h'
        assert adapter.timeout == 60


class TestYFinanceLoadHistorical:
    def test_load_historical_invalid_frequency(self):
        adapter = YFinanceAdapter()
        
        with pytest.raises(AdapterError):
            adapter.load_historical(
                'SPY',
                datetime(2024, 1, 1),
                datetime(2024, 12, 31),
                'invalid_freq'
            )
    
    def test_load_historical_symbol_not_found(self):
        adapter = YFinanceAdapter()
        
        with pytest.raises((SymbolNotFound, AdapterError)):
            adapter.load_historical(
                'THIS_IS_NOT_A_REAL_SYMBOL_XYZ123',
                datetime(2024, 1, 1),
                datetime(2024, 12, 31),
                '1d'
            )


class TestYFinanceGetSymbolInfo:
    def test_get_symbol_info(self):
        adapter = YFinanceAdapter()
        
        info = adapter.get_symbol_info('SPY')
        
        assert 'symbol' in info
        assert info['symbol'] == 'SPY'


class TestYFinanceGetMultiple:
    def test_get_multiple(self):
        adapter = YFinanceAdapter()
        
        results = adapter.get_multiple(
            ['SPY', 'AAPL'],
            datetime(2024, 1, 1),
            datetime(2024, 1, 31),
            '1d'
        )
        
        assert 'SPY' in results
        assert 'AAPL' in results


class TestYFinanceIntegration:
    @pytest.mark.integration
    def test_load_spy_data(self):
        adapter = YFinanceAdapter()
        
        try:
            df = adapter.load_historical(
                'SPY',
                datetime(2024, 1, 1),
                datetime(2024, 12, 31),
                '1d'
            )
            
            assert isinstance(df, pd.DataFrame)
            assert 'close' in df.columns
            assert 'open' in df.columns
            assert len(df) > 0
            assert df.index.is_monotonic_increasing
        except (SymbolNotFound, AdapterError):
            pytest.skip("yfinance not available or symbol not found")
    
    @pytest.mark.integration
    def test_load_vix_data(self):
        adapter = YFinanceAdapter()
        
        try:
            df = adapter.load_historical(
                '^VIX',
                datetime(2024, 1, 1),
                datetime(2024, 12, 31),
                '1d'
            )
            
            assert isinstance(df, pd.DataFrame)
            assert 'close' in df.columns
        except (SymbolNotFound, AdapterError):
            pytest.skip("yfinance not available or symbol not found")
    
    @pytest.mark.integration
    def test_load_forex_pair(self):
        adapter = YFinanceAdapter()
        
        try:
            df = adapter.load_historical(
                'EURUSD=X',
                datetime(2024, 1, 1),
                datetime(2024, 12, 31),
                '1d'
            )
            
            assert isinstance(df, pd.DataFrame)
            assert 'close' in df.columns
        except (SymbolNotFound, AdapterError):
            pytest.skip("yfinance not available or forex pair not found")


class TestYFinanceDataframeValidation:
    def test_validate_dataframe_missing_columns(self):
        adapter = YFinanceAdapter()
        
        df = pd.DataFrame({
            'open': [100.0],
            'close': [103.0]
        }, index=[pd.Timestamp('2024-01-01', tz='UTC')])
        
        with pytest.raises(AdapterError):
            adapter.validate_dataframe(df)
    
    def test_validate_dataframe_nan_close(self):
        adapter = YFinanceAdapter()
        
        df = pd.DataFrame({
            'open': [100.0],
            'high': [105.0],
            'low': [99.0],
            'close': [np.nan],
            'volume': [1000000]
        }, index=[pd.Timestamp('2024-01-01', tz='UTC')])
        
        with pytest.raises(AdapterError):
            adapter.validate_dataframe(df)
    
    def test_validate_dataframe_zero_close(self):
        adapter = YFinanceAdapter()
        
        df = pd.DataFrame({
            'open': [100.0],
            'high': [105.0],
            'low': [99.0],
            'close': [0.0],
            'volume': [1000000]
        }, index=[pd.Timestamp('2024-01-01', tz='UTC')])
        
        with pytest.raises(AdapterError):
            adapter.validate_dataframe(df)
    
    def test_validate_dataframe_negative_close(self):
        adapter = YFinanceAdapter()
        
        df = pd.DataFrame({
            'open': [100.0],
            'high': [105.0],
            'low': [99.0],
            'close': [-1.0],
            'volume': [1000000]
        }, index=[pd.Timestamp('2024-01-01', tz='UTC')])
        
        with pytest.raises(AdapterError):
            adapter.validate_dataframe(df)
    
    def test_validate_dataframe_duplicate_timestamps(self):
        adapter = YFinanceAdapter()
        
        df = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [105.0, 106.0],
            'low': [99.0, 100.0],
            'close': [103.0, 104.0],
            'volume': [1000000, 1100000]
        }, index=pd.DatetimeIndex(['2024-01-01', '2024-01-01'], tz='UTC'))
        
        with pytest.raises(AdapterError):
            adapter.validate_dataframe(df)
    
    def test_validate_dataframe_not_sorted(self):
        adapter = YFinanceAdapter()
        
        df = pd.DataFrame({
            'open': [101.0, 100.0],
            'high': [106.0, 105.0],
            'low': [100.0, 99.0],
            'close': [104.0, 103.0],
            'volume': [1100000, 1000000]
        }, index=pd.DatetimeIndex(['2024-01-02', '2024-01-01'], tz='UTC'))
        
        with pytest.raises(AdapterError):
            adapter.validate_dataframe(df)
