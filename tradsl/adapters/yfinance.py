import os
os.environ['YFINANCE_QUIET'] = '1'

from datetime import datetime, timedelta, timezone
from typing import Optional
import pandas as pd
import yfinance as yf

from .base import BaseAdapter


class YFinanceAdapter(BaseAdapter):
    """
    Adapter for Yahoo Finance data via yfinance library.
    
    Supports:
    - Historical OHLCV data (load_historical)
    - Fundamental data (income statement, balance sheet, cash flow)
    - Option chains
    - Analyst price targets and recommendations
    - Key financial metrics over time
    """
    
    MAX_DAYS_FOR_INTERVAL = {
        '1m': 7,
        '5m': 60,
        '15m': 60,
        '30m': 60,
        '1h': 730,
        '1d': 10000,
        '1wk': 10000,
        '1mo': 10000,
    }
    
    def __init__(
        self, 
        interval: str = '1d',
        auto_adjust: bool = True,
        max_workers: int = 3
    ):
        """
        Initialize Yahoo Finance adapter.
        
        Args:
            interval: Default data interval for historical data
            auto_adjust: Whether to auto-adjust prices for splits/dividends
            max_workers: Max concurrent requests
        """
        self.interval = interval
        self.auto_adjust = auto_adjust
        self.max_workers = max_workers
    
    def load_historical(
        self, 
        symbol: str, 
        start: datetime, 
        end: datetime,
        frequency: str = None
    ) -> pd.DataFrame:
        """
        Load historical OHLCV data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'NVDA')
            start: Start datetime
            end: End datetime
            frequency: Data interval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)
                       Uses adapter default if not specified
        
        Returns:
            DataFrame with columns: [open, high, low, close, volume]
            Index: DatetimeIndex (timezone-aware UTC)
            
        Raises:
            ValueError: If symbol is invalid or no data available
        """
        interval = frequency if frequency else self.interval
        
        max_days = self.MAX_DAYS_FOR_INTERVAL.get(interval, 365)
        actual_days = min((end - start).days, max_days)
        
        adjusted_start = end - timedelta(days=actual_days)
        
        if actual_days == 1 and interval in ('1m', '5m', '15m', '30m', '1h'):
            adjusted_start = end - timedelta(days=5)
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=adjusted_start,
            end=end,
            interval=interval,
            auto_adjust=self.auto_adjust
        )
        
        if df.empty:
            raise ValueError(f"No data available for symbol '{symbol}'")
        
        df.columns = df.columns.str.lower()
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns {missing} for {symbol}")
        
        if actual_days == 1 and interval in ('1m', '5m', '15m', '30m', '1h'):
            if not df.empty:
                last_trading_day = df.index[-1].date()
                df = df[df.index.date == last_trading_day]
        
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
        
        return df[required_cols].sort_index()
    
    def get_fundamentals(
        self, 
        ticker: str, 
        statement_type: str, 
        period: str = 'annual'
    ) -> dict:
        """
        Get fundamental data from Yahoo Finance.
        
        Args:
            ticker: Stock symbol
            statement_type: 'income_statement', 'balance_sheet', 'cashflow'
            period: 'annual' or 'quarterly'
            
        Returns:
            Dictionary with statement data
            
        Raises:
            ValueError: If statement_type is invalid
        """
        stock = yf.Ticker(ticker)
        
        if statement_type == 'income_statement':
            df = stock.income_stmt if period == 'annual' else stock.quarterly_income_stmt
        elif statement_type == 'balance_sheet':
            df = stock.balance_sheet if period == 'annual' else stock.quarterly_balance_sheet
        elif statement_type == 'cashflow':
            df = stock.cashflow if period == 'annual' else stock.quarterly_cashflow
        else:
            raise ValueError(f"Invalid statement_type: {statement_type}")
        
        if df is None or df.empty:
            return {}
        
        result = {}
        for idx in df.index:
            result[idx] = {
                str(col.date()): float(val) if pd.notna(val) else None
                for col, val in df.loc[idx].items()
            }
        
        return result
    
    def get_option_chain(self, ticker: str, expiry: str) -> dict:
        """
        Get option chain for a ticker and expiry date.
        
        Args:
            ticker: Stock symbol
            expiry: Expiry date string (YYYY-MM-DD)
            
        Returns:
            Dictionary with 'calls' and 'puts' keys, each containing list of options
        """
        stock = yf.Ticker(ticker)
        
        try:
            option_chain = stock.option_chain(expiry)
        except Exception as e:
            raise ValueError(f"Failed to get option chain for {ticker} {expiry}: {e}")
        
        return {
            'calls': option_chain.calls.to_dict(orient='records'),
            'puts': option_chain.puts.to_dict(orient='records')
        }
    
    def get_price_targets(self, ticker: str) -> dict:
        """
        Get analyst price targets and recommendations.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary with target price, recommendation, and history
        """
        stock = yf.Ticker(ticker)
        
        target_mean_price = stock.info.get('targetMeanPrice')
        recommendation = stock.info.get('recommendationKey')
        
        recommendations = stock.recommendations
        history = None
        if recommendations is not None and not recommendations.empty:
            history = recommendations.tail(10).to_dict(orient='records')
        
        return {
            'ticker': ticker,
            'target_mean_price': target_mean_price,
            'recommendation': recommendation,
            'recommendation_history': history
        }
    
    def get_key_metrics_timeseries(
        self, 
        ticker: str, 
        period: str = 'quarterly'
    ) -> dict:
        """
        Get common financial metrics over time.
        
        Args:
            ticker: Stock symbol
            period: 'quarterly' or 'annual'
            
        Returns:
            Dictionary with metric timeseries:
            {
                'revenue': [{'date': '2024-03-31', 'value': 90000}, ...],
                'net_income': [...],
                'operating_income': [...],
                'gross_profit': [...],
                'total_assets': [...],
                'total_debt': [...],
                'free_cash_flow': [...]
            }
        """
        stock = yf.Ticker(ticker)
        
        if period == 'quarterly':
            income = stock.quarterly_income_stmt
            balance = stock.quarterly_balance_sheet
            cashflow = stock.quarterly_cashflow
        else:
            income = stock.income_stmt
            balance = stock.balance_sheet
            cashflow = stock.cashflow
        
        def extract_metric(df, metric_name):
            if df is None:
                return {}
            
            variations = [
                metric_name,
                metric_name.title(),
                metric_name.replace('_', ' ').title(),
                ' '.join(word.capitalize() for word in metric_name.split('_'))
            ]
            
            for var in variations:
                if var in df.index:
                    return df.loc[var].to_dict()
            
            for idx in df.index:
                if metric_name.lower() in idx.lower():
                    return df.loc[idx].to_dict()
            
            return {}
        
        metrics = {
            'revenue': extract_metric(income, 'Total Revenue'),
            'net_income': extract_metric(income, 'Net Income'),
            'operating_income': extract_metric(income, 'Operating Income'),
            'gross_profit': extract_metric(income, 'Gross Profit'),
            'total_assets': extract_metric(balance, 'Total Assets'),
            'total_debt': extract_metric(balance, 'Total Debt'),
            'free_cash_flow': extract_metric(cashflow, 'Free Cash Flow'),
        }
        
        result = {}
        for metric_name, data in metrics.items():
            result[metric_name] = [
                {
                    'date': str(date),
                    'value': float(value) if pd.notna(value) else None
                }
                for date, value in data.items()
            ]
            result[metric_name].sort(key=lambda x: x['date'])
        
        return result
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol exists on Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return 'symbol' in info or 'shortName' in info
        except Exception:
            return False


class YFAdapter(YFinanceAdapter):
    """Alias for YFinanceAdapter for backwards compatibility."""
    pass
