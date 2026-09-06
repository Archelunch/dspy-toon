# Copyright (c) 2025 dspy-toon
# SPDX-License-Identifier: MIT
"""TOON 4.1 encoding and decoding of the JSON data model.

Python integers retain arbitrary precision; decimal/exponent tokens use finite
Python floats. Decimal inputs use the float approximation. Dates use ISO 8601,
Pydantic models use model_dump(), tuples and sets become arrays, and unsupported
values become null. See the upgrade notes for migration from the old codec.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any, TypedDict

from pydantic import BaseModel

SPEC_VERSION = "4.1"
JsonValue = Any


class EncodeOptions(TypedDict, total=False):
    """Encoding options; indent is retained as an alias for indentSize."""

    indent: int
    indentSize: int
    delimiter: str


@dataclass
class DecodeOptions:
    """Decoding options; indentSize takes precedence over the legacy indent."""

    indent: int = 2
    strict: bool = True
    indentSize: int | None = None


class ToonDecodeError(ValueError):
    """Malformed TOON input."""


_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?", re.I)
_NUMERIC_LIKE = re.compile(r"[+-]?[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?", re.I)
_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
_ESCAPES = {"\\": "\\", '"': '"', "n": "\n", "r": "\r", "t": "\t"}


def _indent_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("indentSize must be a positive integer")
    return value


def normalize_value(value: Any) -> JsonValue:
    """Normalize Python values recursively, including models inside containers."""
    if isinstance(value, BaseModel):
        return normalize_value(value.model_dump())
    if isinstance(value, str):
        value.encode("utf-8")  # Reject unpaired surrogates, including in keys.
        return value
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, (float, Decimal)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {normalize_value(str(k)): normalize_value(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        try:
            value = sorted(value)
        except TypeError:
            value = sorted(value, key=repr)
    if isinstance(value, (list, tuple)):
        return [normalize_value(item) for item in value]
    return None


def _quote(value: str) -> str:
    escapes = {v: "\\" + k for k, v in _ESCAPES.items()}
    return '"' + "".join(escapes.get(c, f"\\u{ord(c):04x}" if ord(c) < 32 else c) for c in value) + '"'


def encode_key(key: str) -> str:
    """Encode a key using TOON's ASCII unquoted-key grammar."""
    return key if _KEY.fullmatch(key) else _quote(key)


def _primitive(value: Any, delimiter: str) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == 0:
            return "0"
        result = format(Decimal(str(value)), "f")
        return result.rstrip("0").rstrip(".") if "." in result else result
    must_quote = (
        not value
        or value != value.strip(" \t")
        or value in ("true", "false", "null")
        or _NUMERIC_LIKE.fullmatch(value)
        or value.startswith(("-", "#", "\ufeff"))
        or any(c in ':"\\[]{}' or ord(c) < 32 or c == delimiter for c in value)
    )
    return _quote(value) if must_quote else value


# A field list is shared by header rendering, column detection and row decoding.
@dataclass
class _Field:
    name: str
    children: list[_Field] | None = None


def _columns(rows: list[Any]) -> list[_Field] | None:
    if not rows or any(not isinstance(row, dict) or not row for row in rows):
        return None
    keys = list(rows[0])
    if any(row.keys() != rows[0].keys() for row in rows):
        return None
    fields: list[_Field] = []
    for key in keys:
        values = [row[key] for row in rows]
        if all(not isinstance(value, (dict, list)) for value in values):
            fields.append(_Field(key))
        elif children := _columns(values):
            fields.append(_Field(key, children))
        else:
            return None
    return fields


def _field_list(fields: list[_Field], delimiter: str) -> str:
    return (
        "{"
        + delimiter.join(
            encode_key(f.name) + (_field_list(f.children, delimiter) if f.children else "") for f in fields
        )
        + "}"
    )


def _leaves(row: dict[str, Any], fields: list[_Field]) -> Iterator[Any]:
    for field in fields:
        if field.children:
            yield from _leaves(row[field.name], field.children)
        else:
            yield row[field.name]


class _Encoder:
    def __init__(self, indent: int, delimiter: str):
        self.indent = " " * indent
        self.delimiter = delimiter
        self.lines: list[str] = []

    def emit(self, depth: int, content: str) -> None:
        self.lines.append(self.indent * depth + content)

    def value(self, value: Any, depth: int = 0, key: str | None = None, item: bool = False) -> None:
        prefix = encode_key(key) if key is not None else ""
        if isinstance(value, (list, dict)):
            rows = list(value.values()) if isinstance(value, dict) else value
            fields = None if item or (isinstance(value, dict) and len(value) < 2) else _columns(rows)
            if fields:
                marker = ":" if isinstance(value, dict) else ""
                delim = self.delimiter if self.delimiter != "," else ""
                self.emit(depth, f"{prefix}[{len(value)}{marker}{delim}]{_field_list(fields, self.delimiter)}:")
                entries = value.items() if isinstance(value, dict) else ((None, row) for row in value)
                for entry_key, row in entries:
                    cells = self.delimiter.join(_primitive(v, self.delimiter) for v in _leaves(row, fields))
                    row_prefix = encode_key(entry_key) + ": " if entry_key is not None else ""
                    self.emit(depth + 1, row_prefix + cells)
                return
        if isinstance(value, dict):
            if key is not None:
                self.emit(depth, prefix + ":")
                depth += 1
            for name, child in value.items():
                self.value(child, depth, name)
        elif isinstance(value, list):
            if not value and not item:
                self.emit(depth, (prefix + ": " if key is not None else "") + "[]")
                return
            delim = self.delimiter if self.delimiter != "," else ""
            header = f"{prefix}[{len(value)}{delim}]:"
            if all(not isinstance(v, (dict, list)) for v in value):
                cells = self.delimiter.join(_primitive(v, self.delimiter) for v in value)
                self.emit(depth, header + (" " + cells if cells else ""))
            else:
                self.emit(depth, header)
                for child in value:
                    self.list_item(child, depth + 1)
        else:
            self.emit(depth, (prefix + ": " if key is not None else "") + _primitive(value, self.delimiter))

    def list_item(self, value: Any, depth: int) -> None:
        if isinstance(value, dict) and not value:
            self.emit(depth, "-")
            return
        start = len(self.lines)
        # An object's first field has logical depth d+1; only its first line
        # moves onto the hyphen. Its nested scope remains at d+2 (§10).
        self.value(value, depth + 1 if isinstance(value, dict) else depth, item=True)
        self.lines[start] = self.indent * depth + "- " + self.lines[start].lstrip(" ")


def encode(value: Any, options: EncodeOptions | None = None) -> str:
    """Encode a Python value as TOON 4.1, with optional indentSize and delimiter."""
    options = options or {}
    if "lengthMarker" in options:
        raise ValueError("lengthMarker was removed in TOON 2.0")
    delimiter = options.get("delimiter", ",")
    if delimiter not in (",", "\t", "|"):
        raise ValueError("delimiter must be comma, tab or pipe")
    encoder = _Encoder(_indent_size(options.get("indentSize", options.get("indent", 2))), delimiter)
    encoder.value(normalize_value(value))
    return "\n".join(encoder.lines)


def _unquoted(text: str, chars: str) -> Iterator[tuple[int, str]]:
    """Locate syntax characters outside quoted tokens."""
    quoted = False
    escaped = False
    for i, char in enumerate(text):
        if quoted and escaped:
            escaped = False
        elif quoted and char == "\\":
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif not quoted and char in chars:
            yield i, char


def _find(text: str, char: str) -> int:
    return next((i for i, _ in _unquoted(text, char)), -1)


def _string(token: str) -> str:
    token = token.strip(" ")
    if not token.startswith('"'):
        return token
    chars: list[str] = []
    i = 1
    while i < len(token):
        c = token[i]
        if c == '"':
            if i != len(token) - 1:
                raise ToonDecodeError("Unexpected characters after closing quote")
            return "".join(chars)
        if c == "\\":
            i += 1
            if i >= len(token):
                break
            escape = token[i]
            if escape == "u":
                digits = token[i + 1 : i + 5]
                if not re.fullmatch(r"[0-9a-fA-F]{4}", digits):
                    raise ToonDecodeError("Invalid Unicode escape")
                c = chr(int(digits, 16))
                if 0xD800 <= ord(c) <= 0xDFFF:
                    raise ToonDecodeError("Surrogate Unicode escapes are not allowed")
                i += 4
            elif escape in _ESCAPES:
                c = _ESCAPES[escape]
            else:
                raise ToonDecodeError(f"Invalid escape: \\{escape}")
        elif ord(c) < 32 and c != "\t":
            raise ToonDecodeError("Unescaped control character")
        chars.append(c)
        i += 1
    raise ToonDecodeError("Unterminated quoted string")


def _parse_primitive(token: str) -> Any:
    token = token.strip(" ")
    if token.startswith('"'):
        return _string(token)
    if token in ("true", "false", "null"):
        return {"true": True, "false": False, "null": None}[token]
    if _NUMBER.fullmatch(token):
        if not any(c in token.lower() for c in ".e"):
            return int(token)
        number = float(token)
        if not math.isfinite(number):
            raise ToonDecodeError("Number outside the finite float range")
        return number
    return token


def _cells(text: str, delimiter: str) -> list[Any]:
    if not text.strip(" "):
        return []
    ends = [i for i, _ in _unquoted(text, delimiter)] + [len(text)]
    values = []
    start = 0
    for end in ends:
        values.append(_parse_primitive(text[start:end]))
        start = end + 1
    return values


@dataclass
class _Header:
    key: str | None
    count: int
    delimiter: str
    fields: list[_Field] | None
    keyed: bool
    inline: str


class _HeaderSyntax(ToonDecodeError):
    """Header errors that may fall through to key-value parsing in lenient mode."""


def _parse_fields(text: str, delimiter: str, strict: bool) -> list[_Field]:
    if not text.startswith("{") or not text.endswith("}"):
        raise _HeaderSyntax("Malformed field list")
    fields: list[_Field] = []
    depth = 0
    start = 1
    group_start = -1
    group_end = -1
    for i, c in _unquoted(text, "{},\t|"):
        if i == 0:
            continue
        if c == "{":
            if depth == 0:
                group_start = i
            depth += 1
        elif c == "}" and depth:
            depth -= 1
            if depth == 0:
                group_end = i
        elif depth == 0:
            if c != delimiter and i != len(text) - 1:
                raise _HeaderSyntax("Field delimiter mismatch")
            name_end = group_start if group_start >= 0 else i
            raw = text[start:name_end].strip(" ")
            if not raw or (group_start >= 0 and group_end != i - 1):
                raise _HeaderSyntax("Empty or malformed field")
            name = _string(raw)
            if strict and any(f.name == name for f in fields):
                raise _HeaderSyntax(f"Duplicate field: {name}")
            children = _parse_fields(text[group_start : group_end + 1], delimiter, strict) if group_start >= 0 else None
            fields.append(_Field(name, children))
            start = i + 1
            group_start = group_end = -1
    if depth or start != len(text) or not fields:
        raise _HeaderSyntax("Unbalanced or empty field list")
    return fields


def _header(text: str, strict: bool) -> _Header | None:
    bracket = _find(text, "[")
    colon = _find(text, ":")
    if bracket < 0 or (0 <= colon < bracket) or text == "[]":
        return None
    try:
        raw_key = text[:bracket]
        if raw_key and raw_key[-1].isspace():
            raise _HeaderSyntax("Whitespace before bracket segment")
        key = _string(raw_key) if raw_key else None
        match = re.match(r"\[(0|[1-9][0-9]*)(:)?([\t|,]?)\]", text[bracket:])
        if not match or (match[2] and match[3] == ","):
            raise _HeaderSyntax("Invalid bracket segment")
        # Accept legacy explicit comma headers emitted by earlier releases.
        count, keyed, delimiter = int(match[1]), bool(match[2]), match[3] or ","
        tail = text[bracket + match.end() :]
        colon = _find(tail, ":")
        if colon < 0:
            raise ToonDecodeError("Missing header colon")
        fields_text, inline = tail[:colon], tail[colon + 1 :].strip(" ")
        fields = _parse_fields(fields_text, delimiter, strict) if fields_text else None
        if keyed and not fields:
            raise _HeaderSyntax("Keyed header requires fields")
        if fields and inline:
            raise _HeaderSyntax("Tabular header cannot carry inline values")
        return _Header(key, count, delimiter, fields, keyed, inline)
    except _HeaderSyntax:
        if strict:
            raise
        return None


@dataclass
class _Line:
    depth: int
    text: str
    number: int
    gap: bool = False


class _Decoder:
    def __init__(self, text: str, indent: int, strict: bool):
        self.strict = strict
        self.lines = []
        self.pos = 0
        blank = False
        for number, raw in enumerate(text.removeprefix("\ufeff").split("\n"), 1):
            raw = raw.removesuffix("\r").rstrip(" ")
            content = raw.lstrip(" ")
            if content.startswith("#"):
                continue
            if not content.strip(" \t"):
                blank = True
                continue
            spaces = len(raw) - len(content)
            if content.startswith("\t"):
                if strict:
                    raise ToonDecodeError(f"Line {number}: tabs are not indentation")
                prefix = raw[: len(raw) - len(raw.lstrip(" \t"))]
                spaces = len(prefix.expandtabs(indent))
                content = raw.lstrip(" \t")
            if strict and spaces % indent:
                raise ToonDecodeError(f"Line {number}: indentation must be a multiple of {indent}")
            self.lines.append(_Line(spaces // indent, content, number, blank))
            blank = False

    def put(self, result: dict[str, Any], key: str, value: Any) -> None:
        if self.strict and key in result:
            raise ToonDecodeError(f"Duplicate key: {key}")
        result[key] = value

    def object(self, depth: int, result: dict[str, Any] | None = None) -> dict[str, Any]:
        result = {} if result is None else result
        while self.pos < len(self.lines):
            line = self.lines[self.pos]
            if line.depth < depth:
                break
            if line.depth > depth:
                if self.strict:
                    raise ToonDecodeError(f"Line {line.number}: unexpected indentation")
                if _find(line.text, ":") < 0 and not line.text.startswith("- "):
                    raise ToonDecodeError("Scalar line outside root primitive position")
                self.pos += 1
                continue
            self.pos += 1
            key, value = self.field(line.text, depth)
            self.put(result, key, value)
        return result

    def field(self, text: str, depth: int) -> tuple[str, Any]:
        header = _header(text, self.strict)
        if header and header.key is not None:
            return header.key, self.array(header, depth)
        if header and self.strict:
            raise ToonDecodeError("Keyless header in object field")
        colon = _find(text, ":")
        if colon < 0:
            raise ToonDecodeError("Missing colon in object field")
        key = _string(text[:colon])
        token = text[colon + 1 :].strip(" ")
        value = self.object(depth + 1) if not token else ([] if token == "[]" else _parse_primitive(token))
        return key, value

    def row(self, cells: Iterator[Any], fields: list[_Field]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for field in fields:
            if field.children:
                value = self.row(cells, field.children)
            else:
                try:
                    value = next(cells)
                except StopIteration:
                    continue
            result[field.name] = value
        return result

    def array(self, header: _Header, depth: int) -> Any:
        if header.inline:
            values = _cells(header.inline, header.delimiter)
            self.check_count(len(values), header.count)
            return values
        values = []
        entries: dict[str, Any] = {}
        start = self.pos
        while self.pos < len(self.lines):
            line = self.lines[self.pos]
            if line.depth <= depth:
                break
            if line.depth != depth + 1:
                if self.strict:
                    raise ToonDecodeError(f"Line {line.number}: unexpected row/item indentation")
                self.pos += 1
                continue
            if header.fields:
                colon = _find(line.text, ":")
                delim = _find(line.text, header.delimiter)
                if not header.keyed and colon >= 0 and (delim < 0 or colon < delim):
                    break
                if header.keyed and colon < 0:
                    if self.strict:
                        raise ToonDecodeError("Missing colon in keyed entry")
                    self.pos += 1
                    continue
                self.pos += 1
                text = line.text[colon + 1 :] if header.keyed else line.text
                cells = _cells(text, header.delimiter)
                width = sum(1 for _ in self.field_paths(header.fields))
                self.check_count(len(cells), width)
                value = self.row(iter(cells), header.fields)
                if header.keyed:
                    self.put(entries, _string(line.text[:colon]), value)
                values.append(value)
            elif line.text == "-" or line.text.startswith("- "):
                self.pos += 1
                values.append(self.item(line.text[1:].lstrip(" "), line.depth))
            else:
                break
        if self.strict and any(line.gap for line in self.lines[start + 1 : self.pos]):
            raise ToonDecodeError("Blank line inside header span")
        self.check_count(len(values), header.count)
        return entries if header.keyed else values

    def field_paths(self, fields: list[_Field]) -> Iterator[str]:
        for field in fields:
            if field.children:
                yield from self.field_paths(field.children)
            else:
                yield field.name

    def check_count(self, actual: int, expected: int) -> None:
        if self.strict and actual != expected:
            raise ToonDecodeError(f"Expected {expected} values, got {actual}")

    def item(self, text: str, depth: int) -> Any:
        if not text:
            return self.object(depth + 1)
        header = _header(text, self.strict)
        if header and header.key is None:
            if header.fields:
                raise ToonDecodeError("Keyless tabular header is only valid at root")
            return self.array(header, depth)
        if header or _find(text, ":") >= 0:
            key, value = self.field(text, depth + 1)
            return self.object(depth + 1, {key: value})
        return [] if text == "[]" else _parse_primitive(text)

    def decode(self) -> Any:
        if not self.lines:
            return {}
        first = self.lines[0]
        if self.strict and first.depth != 0:
            raise ToonDecodeError("Root must start at depth zero")
        header = _header(first.text, self.strict)
        if first.text == "[]":
            self.pos = 1
            result: Any = []
        elif header and header.key is None:
            self.pos = 1
            result = self.array(header, 0)
        elif len(self.lines) == 1 and not header and _find(first.text, ":") < 0:
            self.pos = 1
            result = _parse_primitive(first.text)
        else:
            result = self.object(0)
        if self.strict and self.pos < len(self.lines):
            raise ToonDecodeError("Trailing content after root value")
        return result


def decode(input_str: str, options: DecodeOptions | Mapping[str, Any] | None = None) -> JsonValue:
    """Decode TOON with strict validation by default; accept legacy empty arrays.

    Non-strict mode ignores count/width mismatches, uses last-write-wins for
    duplicate keys and floors indentation depth. Leading tabs expand to
    indentSize tab stops in non-strict mode. Decimal/exponent numbers outside
    finite float range error.
    """
    if isinstance(options, Mapping):
        indent = options.get("indentSize", options.get("indent", 2))
        strict = options.get("strict", True)
    else:
        options = options or DecodeOptions()
        indent = options.indentSize if options.indentSize is not None else options.indent
        strict = options.strict
    return _Decoder(input_str, _indent_size(indent), strict).decode()
