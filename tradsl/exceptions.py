"""
TradSL Exception Hierarchy

All exceptions inherit from TradSLError. Each exception includes
contextual information for debugging.
"""
from typing import Optional, List, Any


class TradSLError(Exception):
    """Base exception for all TradSL errors."""

    def __init__(self, message: str, **context: Any):
        self.message = message
        self.context = context
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if self.context:
            ctx_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} ({ctx_str})"
        return self.message


class ParseError(TradSLError):
    """DSL syntax violation."""

    def __init__(self, message: str, line: Optional[int] = None, line_content: Optional[str] = None, **context: Any):
        self.line = line
        self.line_content = line_content
        super().__init__(message, line=line, line_content=line_content, **context)


class ConfigError(TradSLError):
    """Semantic or structural configuration violation."""

    def __init__(self, message: str, errors: Optional[List[str]] = None, **context: Any):
        self.errors = errors or []
        super().__init__(message, errors=errors, **context)

    @classmethod
    def collect(cls, *errors: "TradSLError") -> "ConfigError":
        """Collect multiple errors into a single ConfigError."""
        all_errors = []
        for e in errors:
            if isinstance(e, ConfigError):
                all_errors.extend(e.errors)
            else:
                all_errors.append(str(e))
        return cls("Multiple configuration errors", errors=all_errors)


class ResolutionError(TradSLError):
    """Name to Python object resolution failure."""

    def __init__(self, message: str, name: Optional[str] = None, **context: Any):
        super().__init__(message, name=name, **context)


class CycleError(TradSLError):
    """Circular dependency detected in DAG."""

    def __init__(self, cycle_path: List[str]):
        self.cycle_path = cycle_path
        cycle_str = " → ".join(cycle_path)
        super().__init__(
            f"Circular dependency detected: {cycle_str}",
            cycle_path=cycle_path
        )


class AdapterError(TradSLError):
    """Data source failure."""
    pass


class SymbolNotFound(AdapterError):
    """Requested symbol not found in data source."""

    def __init__(self, symbol: str, **context: Any):
        super().__init__(f"Symbol not found: {symbol}", symbol=symbol, **context)


class DateRangeTruncated(AdapterError):
    """Requested date range partially unavailable."""

    def __init__(self, requested_start, requested_end, actual_start, actual_end, **context: Any):
        super().__init__(
            f"Date range truncated: requested [{requested_start}, {requested_end}], "
            f"got [{actual_start}, {actual_end}]",
            requested_start=requested_start,
            requested_end=requested_end,
            actual_start=actual_start,
            actual_end=actual_end,
            **context
        )


class APIFailure(AdapterError):
    """External API call failed."""

    def __init__(self, message: str, api_name: Optional[str] = None, **context: Any):
        super().__init__(message, api_name=api_name, **context)


class FeatureError(TradSLError):
    """Feature computation failure."""
    pass


class ModelError(TradSLError):
    """Model inference or training failure."""
    pass


class ExecutionError(TradSLError):
    """Order execution failure."""
    pass


class ValidationError(TradSLError):
    """Backtest framework constraint violation."""
    pass
