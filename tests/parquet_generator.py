"""
Parquet Generator for Testing TradSL.

Generates realistic-looking financial timeseries data in parquet format.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


class ParquetGenerator:
    """Generates realistic OHLCV timeseries data."""
    
    def __init__(
        self,
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
        frequency: str = "1D",  # 1D, 1H, 15min, 5min, 1min
    ):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.frequency = frequency
    
    def generate(
        self,
        symbol: str = "AAPL",
        initial_price: float = 100.0,
        volatility: float = 0.02,
        drift: float = 0.0001,
        volume_base: int = 1_000_000,
    ) -> pd.DataFrame:
        """
        Generate realistic OHLCV data using geometric Brownian motion.
        
        Args:
            symbol: Stock symbol
            initial_price: Starting price
            volatility: Daily volatility (0.02 = 2%)
            drift: Daily drift (0.0001 = ~2.5% annual return)
            volume_base: Base daily volume
            
        Returns:
            DataFrame with OHLCV data
        """
        date_range = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=self.frequency,
        )
        
        n = len(date_range)
        
        returns = np.random.normal(drift, volatility, n)
        price_path = initial_price * np.exp(np.cumsum(returns))
        
        intraday_range = np.abs(np.random.normal(0.005, 0.01, n))
        
        open_prices = price_path * (1 + np.random.uniform(-0.002, 0.002, n))
        high_prices = price_path * (1 + intraday_range)
        low_prices = price_path * (1 - intraday_range)
        close_prices = price_path
        
        high_prices = np.maximum.reduce([open_prices, close_prices, high_prices])
        low_prices = np.minimum.reduce([open_prices, close_prices, low_prices])
        
        volume = np.random.lognormal(
            np.log(volume_base),
            0.5,
            n,
        ).astype(int)
        
        df = pd.DataFrame({
            "symbol": symbol,
            "timestamp": date_range,
            "open": np.round(open_prices, 2),
            "high": np.round(high_prices, 2),
            "low": np.round(low_prices, 2),
            "close": np.round(close_prices, 2),
            "volume": volume,
        })
        
        return df
    
    def generate_multiple(
        self,
        symbols: list[str],
        correlation: float = 0.3,
    ) -> pd.DataFrame:
        """
        Generate correlated timeseries for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            correlation: Correlation between symbols (0-1)
            
        Returns:
            Combined DataFrame with all symbols
        """
        base_prices = {
            "AAPL": 180.0,
            "GOOG": 140.0,
            "MSFT": 380.0,
            "AMZN": 180.0,
            "TSLA": 250.0,
            "META": 500.0,
            "NVDA": 880.0,
            "JPM": 195.0,
        }
        
        initial_prices = [base_prices.get(s, 100.0) for s in symbols]
        
        date_range = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=self.frequency,
        )
        n = len(date_range)
        
        common_factor = np.random.normal(0, 1, n)
        idiosyncratic = np.random.normal(0, 1, (n, len(symbols)))
        
        returns = np.zeros((n, len(symbols)))
        for i in range(len(symbols)):
            returns[:, i] = (
                correlation * common_factor +
                np.sqrt(1 - correlation**2) * idiosyncratic[:, i]
            ) * 0.02
        
        price_paths = np.zeros((n, len(symbols)))
        for i in range(len(symbols)):
            price_paths[:, i] = initial_prices[i] * np.exp(
                np.cumsum(returns[:, i])
            )
        
        dfs = []
        for i, symbol in enumerate(symbols):
            intraday_range = np.abs(np.random.normal(0.005, 0.01, n))
            
            open_prices = price_paths[:, i] * (1 + np.random.uniform(-0.002, 0.002, n))
            high_prices = price_paths[:, i] * (1 + intraday_range)
            low_prices = price_paths[:, i] * (1 - intraday_range)
            close_prices = price_paths[:, i]
            
            high_prices = np.maximum.reduce([open_prices, close_prices, high_prices])
            low_prices = np.minimum.reduce([open_prices, close_prices, low_prices])
            
            volume = np.random.lognormal(15, 0.5, n).astype(int)
            
            df = pd.DataFrame({
                "symbol": symbol,
                "timestamp": date_range,
                "open": np.round(open_prices, 2),
                "high": np.round(high_prices, 2),
                "low": np.round(low_prices, 2),
                "close": np.round(close_prices, 2),
                "volume": volume,
            })
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True)
    
    def save(
        self,
        df: pd.DataFrame,
        path: str | Path,
    ) -> None:
        """Save DataFrame to parquet file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        print(f"Saved {len(df)} rows to {path}")
    
    def generate_and_save(
        self,
        symbol: str = "AAPL",
        path: str | Path = "data/parquet/aapl.parquet",
        **kwargs,
    ) -> str:
        """Generate data and save to parquet, returns the path."""
        df = self.generate(symbol=symbol, **kwargs)
        self.save(df, path)
        return str(path)


def generate_test_dataset(
    output_dir: str = "data/parquet",
) -> dict[str, str]:
    """
    Generate a complete test dataset for testing TradSL.
    
    Returns:
        Dict mapping symbol to parquet file path
    """
    generator = ParquetGenerator(
        start_date="2024-01-01",
        end_date="2024-12-31",
        frequency="1D",
    )
    
    base_prices = {
        "AAPL": 180.0,
        "GOOG": 140.0,
        "MSFT": 380.0,
        "AMZN": 180.0,
        "TSLA": 250.0,
    }
    
    paths = {}
    
    for symbol, price in base_prices.items():
        path = f"{output_dir}/{symbol}.parquet"
        generator.generate_and_save(
            symbol=symbol,
            path=path,
            initial_price=price,
        )
        paths[symbol] = path
    
    multi_path = f"{output_dir}/multi.parquet"
    df_multi = generator.generate_multiple(list(base_prices.keys()))
    generator.save(df_multi, multi_path)
    paths["multi"] = multi_path
    
    return paths


def generate_high_frequency(
    output_dir: str = "data/parquet",
) -> dict[str, str]:
    """Generate intraday (5-minute) data."""
    generator = ParquetGenerator(
        start_date="2024-06-01",
        end_date="2024-06-30",
        frequency="5min",
    )
    
    paths = {}
    for symbol in ["AAPL", "GOOG"]:
        path = f"{output_dir}/{symbol}_intraday.parquet"
        generator.generate_and_save(
            symbol=symbol,
            path=path,
            initial_price=180.0 if symbol == "AAPL" else 140.0,
            volume_base=10000,
        )
        paths[symbol] = path
    
    return paths


if __name__ == "__main__":
    print("Generating test dataset...")
    paths = generate_test_dataset()
    print("\nGenerated files:")
    for symbol, path in paths.items():
        print(f"  {symbol}: {path}")
    
    print("\nGenerating intraday data...")
    intra_paths = generate_high_frequency()
    print("\nGenerated intraday files:")
    for symbol, path in intra_paths.items():
        print(f"  {symbol}: {path}")
