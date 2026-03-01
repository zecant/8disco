from typing import Any
import pandas as pd
from datetime import datetime


class DataLoaderError(Exception):
    pass


def load_timeseries(
    config: dict,
    start: datetime,
    end: datetime,
    frequency: str = '1min'
) -> pd.DataFrame:
    """
    Load historical timeseries data from configured adapters.
    
    Args:
        config: Parsed DSL config with _adapters and timeseries definitions
        start: Start datetime
        end: End datetime
        frequency: Bar frequency (e.g., '1min', '1h', '1d')
    
    Returns:
        DataFrame with flat columns: f"{symbol}_{field}"
        e.g., nvda_close, nvda_volume, vix_close, etc.
    
    Raises:
        DataLoaderError: If adapter not found or fails to load
    """
    adapters = config.get('_adapters', {})
    timeseries_nodes = {k: v for k, v in config.items() 
                        if not k.startswith('_') and v.get('type') == 'timeseries'}
    
    data_frames = []
    
    for name, fields in timeseries_nodes.items():
        adapter_name = fields.get('adapter')
        parameters = fields.get('parameters', [])
        
        if not adapter_name:
            continue
        
        if adapter_name not in adapters:
            raise DataLoaderError(
                f"Timeseries '{name}' references adapter '{adapter_name}' "
                f"which is not defined in config"
            )
        
        adapter = adapters[adapter_name]
        
        if not hasattr(adapter, 'load_historical'):
            raise DataLoaderError(
                f"Adapter '{adapter_name}' does not have 'load_historical' method"
            )
        
        symbol = parameters[0] if parameters else name
        
        try:
            df = adapter.load_historical(
                symbol=symbol,
                start=start,
                end=end,
                frequency=frequency
            )
        except Exception as e:
            raise DataLoaderError(
                f"Failed to load data for '{name}' from adapter '{adapter_name}': {e}"
            )
        
        if df is None or df.empty:
            continue
        
        df = _flatten_columns(df, symbol)
        data_frames.append(df)
    
    if not data_frames:
        return pd.DataFrame()
    
    result = data_frames[0]
    for df in data_frames[1:]:
        result = result.join(df, how='outer')
    
    result = result.sort_index()
    result = result.ffill()
    result = result.dropna(how='all')
    
    return result


def _flatten_columns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Flatten multi-level columns to flat format.
    
    Input:  df with columns like ['close', 'high', 'low', ...]
            OR MultiIndex like [('nvda', 'close'), ('nvda', 'high'), ...]
    Output: df with columns like ['nvda_close', 'nvda_high', ...]
    """
    if df.columns.nlevels > 1:
        new_columns = [f"{col[0]}_{col[1]}" for col in df.columns]
        df.columns = new_columns
    else:
        new_columns = [f"{symbol}_{col}" for col in df.columns]
        df.columns = new_columns
    
    return df
