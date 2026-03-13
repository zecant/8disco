"""Tests for DSL Parser."""
import pytest
from tradsl import parse_dsl, ParseError


class TestParser:
    def test_simple_block(self):
        dsl = """my_block:
key=value
"""
        result = parse_dsl(dsl)
        assert 'my_block' in result
        assert result['my_block']['key'] == 'value'

    def test_param_block(self):
        dsl = """my_params:
key1=value1
key2=value2
"""
        result = parse_dsl(dsl)
        assert 'my_params' in result
        assert result['my_params']['key1'] == 'value1'
        assert result['my_params']['key2'] == 'value2'

    def test_multiple_blocks(self):
        dsl = """block_a:
type=timeseries
key=val_a

block_b:
type=model
key=val_b
"""
        result = parse_dsl(dsl)
        assert 'block_a' in result
        assert 'block_b' in result

    def test_comments_ignored(self):
        dsl = """# This is a comment
my_block:
# Another comment
key=value
"""
        result = parse_dsl(dsl)
        assert result['my_block']['key'] == 'value'

    def test_empty_lines_ignored(self):
        dsl = """

my_block:

key=value

"""
        result = parse_dsl(dsl)
        assert result['my_block']['key'] == 'value'

    def test_integer_value(self):
        dsl = """block:
count=42
"""
        result = parse_dsl(dsl)
        assert result['block']['count'] == 42

    def test_float_value(self):
        dsl = """block:
rate=1.5
"""
        result = parse_dsl(dsl)
        assert result['block']['rate'] == 1.5

    def test_float_scientific_notation(self):
        dsl = """block:
val=1.5e-3
"""
        result = parse_dsl(dsl)
        assert result['block']['val'] == 0.0015

    def test_bool_true(self):
        dsl = """block:
enabled=true
"""
        result = parse_dsl(dsl)
        assert result['block']['enabled'] is True

    def test_bool_false(self):
        dsl = """block:
enabled=False
"""
        result = parse_dsl(dsl)
        assert result['block']['enabled'] is False

    def test_bool_case_insensitive(self):
        dsl = """block:
val=TRUE
val2=FALSE
val3=True
"""
        result = parse_dsl(dsl)
        assert result['block']['val'] is True
        assert result['block']['val2'] is False
        assert result['block']['val3'] is True

    def test_none_value(self):
        dsl = """block:
val=none
"""
        result = parse_dsl(dsl)
        assert result['block']['val'] is None

    def test_quoted_string_double(self):
        dsl = """block:
val="hello world"
"""
        result = parse_dsl(dsl)
        assert result['block']['val'] == 'hello world'

    def test_quoted_string_single(self):
        dsl = """block:
val='hello world'
"""
        result = parse_dsl(dsl)
        assert result['block']['val'] == 'hello world'

    def test_list_value(self):
        dsl = """block:
vals=[a, b, c]
"""
        result = parse_dsl(dsl)
        assert result['block']['vals'] == ['a', 'b', 'c']

    def test_list_integers(self):
        dsl = """block:
vals=[1, 2, 3]
"""
        result = parse_dsl(dsl)
        assert result['block']['vals'] == [1, 2, 3]

    def test_list_mixed(self):
        dsl = """block:
vals=[1, "two", 3.0]
"""
        result = parse_dsl(dsl)
        assert result['block']['vals'] == [1, 'two', 3.0]

    def test_unquoted_string(self):
        dsl = """block:
val=some_text
"""
        result = parse_dsl(dsl)
        assert result['block']['val'] == 'some_text'

    def test_duplicate_key_raises(self):
        dsl = """block:
key=val1
key=val2
"""
        with pytest.raises(ParseError) as exc:
            parse_dsl(dsl)
        assert 'Duplicate key' in str(exc.value)

    def test_key_value_before_block_raises(self):
        dsl = """key=value

block:
"""
        with pytest.raises(ParseError) as exc:
            parse_dsl(dsl)
        assert 'before any block header' in str(exc.value)

    def test_invalid_block_format_raises(self):
        dsl = """not_a_block
key=value
"""
        with pytest.raises(ParseError) as exc:
            parse_dsl(dsl)
        assert 'Invalid line format' in str(exc.value)
