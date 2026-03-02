import os
import tempfile
import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import numpy as np

from tradsl.adapters import BaseAdapter, YFinanceAdapter, CSVAdapter, FREDAdapter
from tradsl.adapters.yfinance import YFAdapter
from tradsl.adapters.csv_adapter import CSVDataAdapter
from tradsl.adapters.fred import FREDDataAdapter
import unittest.mock


class TestBaseAdapter:
    """Tests for BaseAdapter abstract class."""
    
    def test_base_is_abstract(self):
        """BaseAdapter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAdapter()


class TestYFinanceAdapter:
    """Tests for YFinanceAdapter."""
    
    @pytest.fixture
    def adapter(self):
        return YFinanceAdapter(interval='1d')
    
    def test_init_default(self):
        adapter = YFinanceAdapter()
        assert adapter.interval == '1d'
        assert adapter.auto_adjust is True
    
    def test_init_custom(self):
        adapter = YFinanceAdapter(interval='1h', auto_adjust=False)
        assert adapter.interval == '1h'
        assert adapter.auto_adjust is False
    
    def test_load_historical_valid_symbol(self, adapter):
        """Test loading historical data for a valid symbol."""
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=30)
        
        df = adapter.load_historical('AAPL', start, end, '1d')
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
        assert df.index.tz is not None
    
    def test_load_historical_invalid_symbol(self, adapter):
        """Test that invalid symbol raises ValueError."""
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=30)
        
        with pytest.raises(ValueError, match="No data available"):
            adapter.load_historical('INVALID_SYMBOL_XYZ123', start, end, '1d')
    
    def test_load_historical_columns_lowercase(self, adapter):
        """Test that column names are normalized to lowercase."""
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=30)
        
        df = adapter.load_historical('MSFT', start, end, '1d')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert col in df.columns
    
    def test_load_historical_date_range(self, adapter):
        """Test that data is filtered by date range."""
        end = datetime(2024, 6, 1, tzinfo=timezone.utc)
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        
        df = adapter.load_historical('GOOGL', start, end, '1d')
        
        assert df.index.min() >= start
        assert df.index.max() <= end
    
    def test_load_historical_different_intervals(self, adapter):
        """Test loading with different time intervals."""
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=60)
        
        for interval in ['1d', '1wk']:
            df = adapter.load_historical('SPY', start, end, interval)
            assert not df.empty
    
    def test_get_fundamentals_income_statement(self, adapter):
        """Test fetching income statement."""
        result = adapter.get_fundamentals('AAPL', 'income_statement', 'annual')
        
        assert isinstance(result, dict)
    
    def test_get_fundamentals_balance_sheet(self, adapter):
        """Test fetching balance sheet."""
        result = adapter.get_fundamentals('AAPL', 'balance_sheet', 'quarterly')
        
        assert isinstance(result, dict)
    
    def test_get_fundamentals_cashflow(self, adapter):
        """Test fetching cashflow statement."""
        result = adapter.get_fundamentals('AAPL', 'cashflow', 'annual')
        
        assert isinstance(result, dict)
    
    def test_get_fundamentals_invalid_type(self, adapter):
        """Test that invalid statement_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid statement_type"):
            adapter.get_fundamentals('AAPL', 'invalid_type', 'annual')
    
    def test_get_price_targets(self, adapter):
        """Test fetching price targets and recommendations."""
        result = adapter.get_price_targets('AAPL')
        
        assert isinstance(result, dict)
        assert 'target_mean_price' in result
        assert 'recommendation' in result
    
    def test_get_key_metrics_timeseries(self, adapter):
        """Test fetching key metrics over time."""
        result = adapter.get_key_metrics_timeseries('AAPL', 'quarterly')
        
        assert isinstance(result, dict)
        expected_keys = ['revenue', 'net_income', 'operating_income', 'gross_profit',
                        'total_assets', 'total_debt', 'free_cash_flow']
        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], list)
    
    def test_validate_symbol_valid(self, adapter):
        """Test symbol validation for valid symbol."""
        assert adapter.validate_symbol('AAPL') is True
    
    def test_validate_symbol_invalid(self, adapter):
        """Test symbol validation for invalid symbol."""
        assert adapter.validate_symbol('INVALID_SYMBOL_XYZ123') is False
    
    def test_supports_historical(self, adapter):
        """Test that YFinance supports historical data."""
        assert adapter.supports_historical() is True
    
    def test_yf_adapter_alias(self):
        """Test that YFAdapter is an alias for YFinanceAdapter."""
        adapter = YFAdapter(interval='1d')
        assert isinstance(adapter, YFinanceAdapter)


class TestCSVAdapter:
    """Tests for CSVAdapter."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory with sample CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def sample_csv_data(self, temp_data_dir):
        """Create sample CSV files for testing."""
        csv_content = """timestamp,open,high,low,close,volume
2024-01-01,100.0,105.0,99.0,103.0,1000000
2024-01-02,103.0,108.0,102.0,106.0,1100000
2024-01-03,106.0,110.0,105.0,108.0,1200000
2024-01-04,108.0,112.0,107.0,110.0,1300000
2024-01-05,110.0,115.0,109.0,113.0,1400000
"""
        csv_path = Path(temp_data_dir) / 'TEST.csv'
        csv_path.write_text(csv_content)
        return temp_data_dir
    
    @pytest.fixture
    def sample_csv_mixed_case(self, temp_data_dir):
        """Create CSV with mixed case column names."""
        csv_content = """Date,Open,High,Low,Close,Volume
2024-01-01,100.0,105.0,99.0,103.0,1000000
2024-01-02,103.0,108.0,102.0,106.0,1100000
"""
        csv_path = Path(temp_data_dir) / 'MIXED.csv'
        csv_path.write_text(csv_content)
        return temp_data_dir
    
    @pytest.fixture
    def sample_csv_custom_columns(self, temp_data_dir):
        """Create CSV with custom column mapping."""
        csv_content = """date,Open_Price,High_Price,Low_Price,Close_Price,Volume_Traded
2024-01-01,100.0,105.0,99.0,103.0,1000000
2024-01-02,103.0,108.0,102.0,106.0,1100000
"""
        csv_path = Path(temp_data_dir) / 'CUSTOM.csv'
        csv_path.write_text(csv_content)
        return temp_data_dir
    
    def test_init_default(self, temp_data_dir):
        adapter = CSVAdapter(data_dir=temp_data_dir)
        assert adapter.data_dir == Path(temp_data_dir)
        assert adapter.timestamp_col == 'timestamp'
    
    def test_init_custom(self, temp_data_dir):
        adapter = CSVAdapter(
            data_dir=temp_data_dir,
            timestamp_col='date',
            date_format='%Y-%m-%d'
        )
        assert adapter.timestamp_col == 'date'
        assert adapter.date_format == '%Y-%m-%d'
    
    def test_load_historical(self, sample_csv_data):
        adapter = CSVAdapter(data_dir=sample_csv_data)
        
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 5, tzinfo=timezone.utc)
        
        df = adapter.load_historical('TEST', start, end)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
    
    def test_load_historical_mixed_case_columns(self, sample_csv_mixed_case):
        """Test loading CSV with mixed case column names."""
        adapter = CSVAdapter(data_dir=sample_csv_mixed_case, timestamp_col='Date')
        
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 3, tzinfo=timezone.utc)
        
        df = adapter.load_historical('MIXED', start, end)
        
        assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
    
    def test_load_historical_custom_column_mapping(self, sample_csv_custom_columns):
        """Test loading CSV with custom column mapping."""
        mapping = {
            'Date': 'timestamp',
            'Open_Price': 'open',
            'High_Price': 'high',
            'Low_Price': 'low',
            'Close_Price': 'close',
            'Volume_Traded': 'volume'
        }
        adapter = CSVAdapter(
            data_dir=sample_csv_custom_columns,
            column_mapping=mapping,
            timestamp_col='Date'
        )
        
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 3, tzinfo=timezone.utc)
        
        df = adapter.load_historical('CUSTOM', start, end)
        
        assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
    
    def test_load_historical_date_filtering(self, sample_csv_data):
        """Test that data is properly filtered by date range."""
        adapter = CSVAdapter(data_dir=sample_csv_data)
        
        start = datetime(2024, 1, 2, tzinfo=timezone.utc)
        end = datetime(2024, 1, 4, tzinfo=timezone.utc)
        
        df = adapter.load_historical('TEST', start, end)
        
        assert len(df) == 3
        assert df.index.min() >= start
        assert df.index.max() <= end
    
    def test_load_historical_file_not_found(self, temp_data_dir):
        """Test that missing file raises FileNotFoundError."""
        adapter = CSVAdapter(data_dir=temp_data_dir)
        
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 5, tzinfo=timezone.utc)
        
        with pytest.raises(FileNotFoundError):
            adapter.load_historical('NONEXISTENT', start, end)
    
    def test_load_historical_timezone_aware(self, sample_csv_data):
        """Test that returned index is timezone-aware."""
        adapter = CSVAdapter(data_dir=sample_csv_data)
        
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 5, tzinfo=timezone.utc)
        
        df = adapter.load_historical('TEST', start, end)
        
        assert df.index.tz is not None
    
    def test_validate_symbol_exists(self, sample_csv_data):
        """Test symbol validation for existing file."""
        adapter = CSVAdapter(data_dir=sample_csv_data)
        assert adapter.validate_symbol('TEST') is True
    
    def test_validate_symbol_not_exists(self, sample_csv_data):
        """Test symbol validation for non-existing file."""
        adapter = CSVAdapter(data_dir=sample_csv_data)
        assert adapter.validate_symbol('NONEXISTENT') is False
    
    def test_list_available_symbols(self, sample_csv_data):
        """Test listing available symbols."""
        adapter = CSVAdapter(data_dir=sample_csv_data)
        symbols = adapter.list_available_symbols()
        assert 'TEST' in symbols
    
    def test_supports_historical(self, sample_csv_data):
        """Test that CSV adapter supports historical data."""
        adapter = CSVAdapter(data_dir=sample_csv_data)
        assert adapter.supports_historical() is True
    
    def test_csv_adapter_alias(self, sample_csv_data):
        """Test that CSVDataAdapter is an alias for CSVAdapter."""
        adapter = CSVDataAdapter(data_dir=sample_csv_data)
        assert isinstance(adapter, CSVAdapter)
    
    def test_load_historical_with_format_string(self, temp_data_dir):
        """Test loading CSV with explicit date format."""
        csv_content = """date,open,high,low,close,volume
01/02/2024,100.0,105.0,99.0,103.0,1000000
01/03/2024,103.0,108.0,102.0,106.0,1100000
"""
        csv_path = Path(temp_data_dir) / 'FORMATTED.csv'
        csv_path.write_text(csv_content)
        
        adapter = CSVAdapter(
            data_dir=temp_data_dir,
            timestamp_col='date',
            date_format='%m/%d/%Y'
        )
        
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 5, tzinfo=timezone.utc)
        
        df = adapter.load_historical('FORMATTED', start, end)
        
        assert len(df) == 2


class TestAdaptersIntegration:
    """Integration tests for adapters with tradsl."""
    
    def test_yfinance_adapter_integration(self):
        """Test YFinanceAdapter works with tradsl data_loader."""
        from tradsl.utils.data_loader import load_timeseries
        
        config = {
            '_adapters': {
                'yf': YFinanceAdapter(interval='1d')
            },
            'aapl': {
                'type': 'timeseries',
                'adapter': 'yf',
                'parameters': ['AAPL']
            }
        }
        
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=30)
        
        df = load_timeseries(config, start, end, '1d')
        
        assert not df.empty
        assert 'AAPL_close' in df.columns
    
    def test_csv_adapter_integration(self):
        """Test CSVAdapter works with tradsl data_loader."""
        from tradsl.utils.data_loader import load_timeseries
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_content = """timestamp,open,high,low,close,volume
2024-01-01,100.0,105.0,99.0,103.0,1000000
2024-01-02,103.0,108.0,102.0,106.0,1100000
"""
            csv_path = Path(tmpdir) / 'SPY.csv'
            csv_path.write_text(csv_content)
            
            config = {
                '_adapters': {
                    'csv': CSVAdapter(data_dir=tmpdir)
                },
                'spy': {
                    'type': 'timeseries',
                    'adapter': 'csv',
                    'parameters': ['SPY']
                }
            }
            
            start = datetime(2024, 1, 1, tzinfo=timezone.utc)
            end = datetime(2024, 1, 3, tzinfo=timezone.utc)
            
            df = load_timeseries(config, start, end, '1d')
            
            assert not df.empty
            assert 'SPY_close' in df.columns


class TestFREDAdapter:
    """Tests for FREDAdapter."""
    
    @pytest.fixture
    def mock_fred(self, monkeypatch):
        """Mock the Fred class to avoid real API calls."""
        mock_fred_instance = unittest.mock.MagicMock()
        
        mock_series_data = pd.Series(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            index=pd.DatetimeIndex([
                '2024-01-01', '2024-02-01', '2024-03-01', 
                '2024-04-01', '2024-05-01'
            ])
        )
        mock_fred_instance.get_series.return_value = mock_series_data
        mock_fred_instance.get_series_first_release.return_value = mock_series_data
        mock_fred_instance.search.return_value = pd.DataFrame({
            'id': ['GDP'], 'title': ['Gross Domestic Product']
        })
        mock_fred_instance.get_series_info.return_value = {
            'id': 'GDP', 
            'title': 'Gross Domestic Product',
            'frequency': 'Quarterly'
        }
        
        def mock_fred_init(api_key):
            return mock_fred_instance
        
        monkeypatch.setattr('tradsl.adapters.fred.Fred', mock_fred_init)
        return mock_fred_instance
    
    @pytest.fixture
    def adapter(self, mock_fred):
        return FREDAdapter(api_key='abcd1234abcd1234abcd1234abcd1234')
    
    def test_init(self):
        adapter = FREDAdapter(api_key='abcd1234abcd1234abcd1234abcd1234')
        assert adapter.api_key == 'abcd1234abcd1234abcd1234abcd1234'
    
    def test_init_missing_fredapi(self, monkeypatch):
        monkeypatch.setattr('tradsl.adapters.fred.FRED_AVAILABLE', False)
        with pytest.raises(ImportError, match="fredapi"):
            FREDAdapter(api_key='test')
    
    def test_load_historical_valid_series(self, adapter):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 6, 1, tzinfo=timezone.utc)
        
        df = adapter.load_historical('GDP', start, end)
        
        assert isinstance(df, pd.DataFrame)
        assert 'value' in df.columns
        assert df.index.tz is not None
    
    def test_load_historical_invalid_series(self, mock_fred):
        mock_fred.get_series.return_value = pd.Series(dtype=float)
        
        adapter = FREDAdapter(api_key='abcd1234abcd1234abcd1234abcd1234')
        
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 6, 1, tzinfo=timezone.utc)
        
        with pytest.raises(ValueError, match="No data available"):
            adapter.load_historical('INVALID_SERIES_XYZ', start, end)
    
    def test_load_historical_returns_single_column(self, adapter):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 6, 1, tzinfo=timezone.utc)
        
        df = adapter.load_historical('GDP', start, end)
        
        assert list(df.columns) == ['value']
    
    def test_load_historical_timezone_aware(self, adapter):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 6, 1, tzinfo=timezone.utc)
        
        df = adapter.load_historical('GDP', start, end)
        
        assert df.index.tz is not None
    
    def test_load_historical_date_filtering(self, adapter):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)
        
        df = adapter.load_historical('GDP', start, end)
        
        assert df.index.min() >= start
        assert df.index.max() <= end
    
    def test_forward_fill_only_no_backfill(self, adapter):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 6, 1, tzinfo=timezone.utc)
        
        df = adapter.load_historical('GDP', start, end)
        
        assert df['value'].notna().any()
    
    def test_search_series(self, adapter, mock_fred):
        mock_fred.search.return_value = pd.DataFrame({
            'id': ['GDP'], 'title': ['Gross Domestic Product']
        })
        
        results = adapter.search_series('GDP')
        
        assert isinstance(results, list)
    
    def test_get_first_release(self, adapter):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 6, 1, tzinfo=timezone.utc)
        
        df = adapter.get_first_release('GDP', start, end)
        
        assert isinstance(df, pd.DataFrame)
        assert 'value' in df.columns
    
    def test_validate_symbol_valid(self, adapter):
        assert adapter.validate_symbol('GDP') is True
    
    def test_validate_symbol_invalid(self, mock_fred):
        mock_fred.get_series.side_effect = Exception("Invalid")
        
        adapter = FREDAdapter(api_key='abcd1234abcd1234abcd1234abcd1234')
        
        assert adapter.validate_symbol('INVALID_SERIES_XYZ') is False
    
    def test_supports_historical(self, adapter):
        assert adapter.supports_historical() is True
    
    def test_fred_adapter_alias(self, mock_fred):
        adapter = FREDDataAdapter(api_key='abcd1234abcd1234abcd1234abcd1234')
        assert isinstance(adapter, FREDAdapter)
    
    def test_get_series_info(self, adapter):
        info = adapter.get_series_info('GDP')
        
        assert isinstance(info, dict)


class TestFREDAdapterIntegration:
    """Integration tests for FRED adapter with tradsl."""
    
    def test_fred_adapter_integration(self, monkeypatch):
        from tradsl.utils.data_loader import load_timeseries
        
        mock_fred_instance = unittest.mock.MagicMock()
        mock_series_data = pd.Series(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            index=pd.DatetimeIndex([
                '2024-01-01', '2024-02-01', '2024-03-01', 
                '2024-04-01', '2024-05-01'
            ])
        )
        mock_fred_instance.get_series.return_value = mock_series_data
        
        def mock_fred_init(api_key):
            return mock_fred_instance
        
        monkeypatch.setattr('tradsl.adapters.fred.Fred', mock_fred_init)
        
        config = {
            '_adapters': {
                'fred': FREDAdapter(api_key='abcd1234abcd1234abcd1234abcd1234')
            },
            'gdp': {
                'type': 'timeseries',
                'adapter': 'fred',
                'parameters': ['GDP']
            }
        }
        
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 6, 1, tzinfo=timezone.utc)
        
        df = load_timeseries(config, start, end, '1d')
        
        assert 'GDP_value' in df.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
