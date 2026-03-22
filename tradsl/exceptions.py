"""
Custom exceptions for TradSL parser and DAG.
"""


class TradSLException(Exception):
    """Base exception for TradSL errors."""
    pass


class ParseError(TradSLException):
    """
    Raised when DSL parsing fails.

    Attributes:
        line: Line number where error occurred (1-indexed)
        line_content: Original line content
        block: Block name if applicable
        key: Key name if applicable
        expected: Expected format hint
    """

    def __init__(
        self,
        message: str,
        line: int | None = None,
        line_content: str | None = None,
        **kwargs
    ):
        super().__init__(message)
        self.line = line
        self.line_content = line_content
        self.block = kwargs.get('block')
        self.key = kwargs.get('key')
        self.expected = kwargs.get('expected')

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.line is not None:
            parts.append(f" at line {self.line}")
        if self.block:
            parts.append(f" in block '{self.block}'")
        if self.expected:
            parts.append(f" (expected: {self.expected})")
        return ''.join(parts)


class CycleError(TradSLException):
    """
    Raised when a circular dependency is detected in the DAG.

    Attributes:
        cycle: List of node names forming the cycle path.
    """

    def __init__(self, cycle: list[str] | str):
        if isinstance(cycle, str):
            cycle = [cycle]
        self.cycle = cycle
        message = "Circular dependency detected:\n  " + " → ".join(cycle) + " → " + cycle[0]
        message += "\n\nCheck the 'inputs' declarations for these nodes."
        super().__init__(message)

    def __str__(self) -> str:
        return super().__str__()


class ConfigError(TradSLException):
    """
    Raised when DAG configuration is invalid.

    Attributes:
        node: Node name where the error occurred.
        key: Configuration key that caused the error.
    """

    def __init__(self, message: str, node: str | None = None, key: str | None = None):
        super().__init__(message)
        self.node = node
        self.key = key

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.node:
            parts.append(f" in node '{self.node}'")
        if self.key:
            parts.append(f" at key '{self.key}'")
        return ''.join(parts)


class ResolutionError(TradSLException):
    """
    Raised when a name cannot be resolved.

    Attributes:
        name: The unresolved name.
        node: Node name where the unresolved reference was found.
        key: Key that contained the unresolved reference.
    """

    def __init__(
        self,
        name: str,
        node: str | None = None,
        key: str | None = None
    ):
        self.name = name
        self.node = node
        self.key = key
        message = f"Cannot resolve name: '{name}'"
        if node:
            message += f" (in node '{node}')"
        if key:
            message += f" (at key '{key}')"
        super().__init__(message)


class InvariantError(TradSLException):
    """
    Raised when a required invariant is violated.

    Attributes:
        invariant: Description of the invariant that was violated.
        node: Node name where the violation occurred.
    """

    def __init__(
        self,
        invariant: str,
        node: str | None = None
    ):
        self.invariant = invariant
        self.node = node
        message = f"Invariant violated: {invariant}"
        if node:
            message += f" (in node '{node}')"
        super().__init__(message)
