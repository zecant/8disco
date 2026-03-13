"""
DSL Parser - Parses TradSL domain-specific language

Parses DSL string into raw Python dict without validation.
See Section 4 of the specification.
"""
import re
from typing import Any, Dict, List, Optional, Union
from tradsl.exceptions import ParseError


def parse(dsl_string: str) -> Dict[str, Any]:
    """
    Parse DSL string into raw Python dict.

    Args:
        dsl_string: Complete DSL content

    Returns:
        Dict with structure:
        {
            'block_name': {'key': value, ...},
            ...
        }

    Raises:
        ParseError: On syntax violation with line number and suggestion
    """
    lines = dsl_string.split('\n')
    result: Dict[str, Any] = {}
    current_block: Optional[str] = None
    current_block_type: Optional[str] = None
    current_data: Dict[str, Any] = {}

    for line_num, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()

        if not line or line.startswith('#'):
            continue

        if line.endswith(':') and not '=' in line:
            if current_block:
                if current_block in result:
                    raise ParseError(
                        f"Duplicate block name: '{current_block}'",
                        line=line_num,
                        line_content=raw_line,
                        block=current_block
                    )
                result[current_block] = current_data
            match = re.match(r'^(\w+):$', line)
            if not match:
                raise ParseError(
                    f"Invalid block format: '{line}'",
                    line=line_num,
                    line_content=raw_line,
                    expected="'block_name:'"
                )
            current_block = match.group(1)
            current_data = {}
            continue

        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()

            if current_block is None:
                raise ParseError(
                    f"Key-value pair before any block header: '{line}'",
                    line=line_num,
                    line_content=raw_line,
                    expected="Block header like 'block_name:'"
                )

            parsed_value = _parse_value(value, line_num, raw_line)

            if key in current_data:
                raise ParseError(
                    f"Duplicate key '{key}' in block '{current_block}'",
                    line=line_num,
                    line_content=raw_line,
                    key=key,
                    block=current_block
                )

            current_data[key] = parsed_value
            continue

        raise ParseError(
            f"Invalid line format: '{line}'",
            line=line_num,
            line_content=raw_line,
            expected="block header, param header, or key=value"
        )

    if current_block:
        result[current_block] = current_data

    return result


def _parse_value(value: str, line_num: int, line_content: str) -> Any:
    """
    Parse a value string into typed Python value.

    Attempt resolution in order:
    1. list [...]
    2. quoted string "..." or '...'
    3. bool (true/false)
    4. none
    5. int
    6. float
    7. unquoted string
    """
    value = value.strip()

    if not value:
        raise ParseError(
            "Empty value",
            line=line_num,
            line_content=line_content
        )

    if value.startswith('[') and value.endswith(']'):
        return _parse_list(value, line_num, line_content)

    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        return value[1:-1]

    lower = value.lower()
    if lower == 'true':
        return True
    if lower == 'false':
        return False
    if lower == 'none':
        return None

    if '.' in value or 'e' in value.lower():
        try:
            return float(value)
        except ValueError:
            pass

    try:
        return int(value)
    except ValueError:
        pass

    return value


def _parse_list(value: str, line_num: int, line_content: str) -> List[Any]:
    """Parse a list value [...]"""
    inner = value[1:-1].strip()
    if not inner:
        return []

    result = []
    parts = []
    current = []
    in_string = False
    string_char = None

    for char in inner:
        if char in ('"', "'") and (not in_string or char == string_char):
            if in_string:
                in_string = False
                string_char = None
            else:
                in_string = True
                string_char = char
            current.append(char)
        elif char == ',' and not in_string:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(char)

    if current:
        parts.append(''.join(current).strip())

    for part in parts:
        if part:
            result.append(_parse_value(part, line_num, line_content))

    return result
