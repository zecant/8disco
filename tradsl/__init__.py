"""
TradSL - Trading Strategy Domain-Specific Language.

A declarative configuration language for building and backtesting trading strategies
that compiles to an executable DAG (Directed Acyclic Graph).

Example usage:
    from tradsl import DAG, parse

    config = parse('''
        price:
            type=timeseries
            adapter=parquet
            path=/data/prices/aapl.parquet
            symbol=AAPL
    ''')

    dag = DAG.from_config(config)
    dag.build()
    dag.resolve(default_registry)

    from tradsl.storage import ClickHouseConnection
    conn = ClickHouseConnection()
    results = dag.execute(conn)
"""
from tradsl.dag import DAG, Node
from tradsl.exceptions import CycleError, ConfigError, ResolutionError
from tradsl.functions import (
    TimeSeriesFunction,
    Function,
    SignalType,
    Lag,
    EMA,
    SMA,
    Returns,
    ExternalFunction,
)
from tradsl.adapters import Adapter, ParquetAdapter, CSVAdapter
from tradsl.storage import ClickHouseConnection

__version__ = "0.0.4"

__all__ = [
    "DAG",
    "Node",
    "TimeSeriesFunction",
    "Function",
    "SignalType",
    "Lag",
    "EMA",
    "SMA",
    "Returns",
    "ExternalFunction",
    "Adapter",
    "ParquetAdapter",
    "CSVAdapter",
    "ClickHouseConnection",
    "CycleError",
    "ConfigError",
    "ResolutionError",
    "default_registry",
]


default_registry: dict[str, object] = {
    "parquet": ParquetAdapter,
    "csv": CSVAdapter,
    "functions.lag": Lag,
    "functions.ema": EMA,
    "functions.sma": SMA,
    "functions.returns": Returns,
}
