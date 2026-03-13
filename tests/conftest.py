"""Test configuration and fixtures for TradSL tests."""
import pytest
from tradsl import clear_registry


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean registry before each test."""
    clear_registry()
    yield
    clear_registry()
