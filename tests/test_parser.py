"""
Tests for the TradSL parser.
"""
import pytest
from tradsl.parser import parse
from tradsl.exceptions import ParseError


class TestParserBasics:
    """Basic parser functionality tests."""

    def test_empty_string(self):
        assert parse("") == {}

    def test_whitespace_only(self):
        assert parse("   \n   \n   ") == {}

    def test_single_block_no_keys(self):
        dsl = "my_block:"
        result = parse(dsl)
        assert "my_block" in result
        assert result["my_block"] == {}

    def test_single_block_with_single_key(self):
        dsl = """block:
key=value
"""
        result = parse(dsl)
        assert result == {"block": {"key": "value"}}

    def test_multiple_blocks(self):
        dsl = """block1:
key1=val1

block2:
key2=val2
"""
        result = parse(dsl)
        assert result == {
            "block1": {"key1": "val1"},
            "block2": {"key2": "val2"},
        }


class TestValueTypes:
    """Tests for different value type parsing."""

    def test_integer(self):
        assert parse("block:\nnum=42")["block"]["num"] == 42

    def test_negative_integer(self):
        assert parse("block:\nnum=-17")["block"]["num"] == -17

    def test_float(self):
        assert parse("block:\nnum=3.14")["block"]["num"] == 3.14

    def test_negative_float(self):
        assert parse("block:\nnum=-2.5")["block"]["num"] == -2.5

    def test_scientific_notation(self):
        assert parse("block:\nnum=1e10")["block"]["num"] == 1e10

    def test_boolean_true(self):
        assert parse("block:\nflag=true")["block"]["flag"] is True

    def test_boolean_true_uppercase(self):
        assert parse("block:\nflag=TRUE")["block"]["flag"] is True

    def test_boolean_false(self):
        assert parse("block:\nflag=false")["block"]["flag"] is False

    def test_none_value(self):
        assert parse("block:\nval=none")["block"]["val"] is None

    def test_none_value_uppercase(self):
        assert parse("block:\nval=None")["block"]["val"] is None

    def test_unquoted_string(self):
        assert parse("block:\nval=hello")["block"]["val"] == "hello"

    def test_quoted_double_string(self):
        assert parse('block:\nval="hello world"')["block"]["val"] == "hello world"

    def test_quoted_single_string(self):
        assert parse("block:\nval='hello world'")["block"]["val"] == "hello world"

    def test_string_with_equals(self):
        assert parse('block:\nval="a=b"')["block"]["val"] == "a=b"

    def test_empty_list(self):
        assert parse("block:\nitems=[]")["block"]["items"] == []

    def test_list_integers(self):
        assert parse("block:\nitems=[1, 2, 3]")["block"]["items"] == [1, 2, 3]

    def test_list_strings(self):
        assert parse('block:\nitems=["a", "b", "c"]')["block"]["items"] == ["a", "b", "c"]

    def test_list_mixed_types(self):
        assert parse('block:\nitems=[1, "two", 3.0]')["block"]["items"] == [1, "two", 3.0]

    def test_list_with_spaces(self):
        assert parse("block:\nitems=[ 1 , 2 , 3 ]")["block"]["items"] == [1, 2, 3]


class TestCommentsAndWhitespace:
    """Tests for comments and whitespace handling."""

    def test_comment_line(self):
        dsl = """# This is a comment
block:
key=value
"""
        assert parse(dsl) == {"block": {"key": "value"}}

    def test_comment_after_content(self):
        dsl = """block:
key=value # inline comment
"""
        assert parse(dsl) == {"block": {"key": "value # inline comment"}}

    def test_empty_lines_between_blocks(self):
        dsl = """block1:
key=val1

block2:
key=val2
"""
        assert parse(dsl) == {
            "block1": {"key": "val1"},
            "block2": {"key": "val2"},
        }


class TestDuplicateHandling:
    """Tests for duplicate key/block detection."""

    def test_duplicate_key_raises(self):
        dsl = """block:
key=val1
key=val2
"""
        with pytest.raises(ParseError) as exc:
            parse(dsl)
        assert "Duplicate key" in str(exc.value)
        assert exc.value.key == "key"
        assert exc.value.block == "block"
        assert exc.value.line == 3

    def test_duplicate_block_uses_last(self):
        dsl = """block1:
key=val1

block1:
key=val2
"""
        result = parse(dsl)
        assert "block1" in result
        assert result["block1"]["key"] == "val2"


class TestErrorCases:
    """Tests for various error conditions."""

    def test_key_value_before_block_raises(self):
        dsl = """key=value
block:
"""
        with pytest.raises(ParseError) as exc:
            parse(dsl)
        assert "before any block" in str(exc.value)
        assert exc.value.line == 1

    def test_empty_value_raises(self):
        dsl = """block:
key=
"""
        with pytest.raises(ParseError) as exc:
            parse(dsl)
        assert "Empty value" in str(exc.value)
        assert exc.value.line == 2

    def test_block_name_can_start_with_number(self):
        dsl = """123invalid:
key=value
"""
        result = parse(dsl)
        assert "123invalid" in result
        assert result["123invalid"]["key"] == "value"

    def test_invalid_line_format_raises(self):
        dsl = """block:
just_a_word
"""
        with pytest.raises(ParseError) as exc:
            parse(dsl)
        assert "Invalid line format" in str(exc.value)
        assert exc.value.line == 2


class TestComplexScenarios:
    """Tests for complex/realistic DSL content."""

    def test_realistic_config(self):
        dsl = """
# Trading strategy configuration
strategy:
name=Mean Reversion
enabled=true
max_position=1000
stop_loss=0.02
instruments=["AAPL", "GOOGL", "MSFT"]

# Data source config
data:
source=yfinance
symbols=["SPY", "QQQ"]
interval=1m
"""
        result = parse(dsl)
        assert result["strategy"]["name"] == "Mean Reversion"
        assert result["strategy"]["enabled"] is True
        assert result["strategy"]["max_position"] == 1000
        assert result["strategy"]["stop_loss"] == 0.02
        assert result["strategy"]["instruments"] == ["AAPL", "GOOGL", "MSFT"]
        assert result["data"]["source"] == "yfinance"
        assert result["data"]["symbols"] == ["SPY", "QQQ"]
        assert result["data"]["interval"] == "1m"

    def test_multiple_keys_same_block(self):
        dsl = """config:
host=localhost
port=8080
debug=true
timeout=30.5
"""
        result = parse(dsl)["config"]
        assert result["host"] == "localhost"
        assert result["port"] == 8080
        assert result["debug"] is True
        assert result["timeout"] == 30.5

    def test_key_with_underscores_and_numbers(self):
        dsl = """block:
my_key_1=value1
key_2=value2
"""
        result = parse(dsl)["block"]
        assert result["my_key_1"] == "value1"
        assert result["key_2"] == "value2"
