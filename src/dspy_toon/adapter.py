# Copyright (c) 2025 dspy-toon
# SPDX-License-Identifier: MIT
"""TOON Adapter for DSPy.

A DSPy adapter that uses TOON (Token-Oriented Object Notation) format for
structured outputs, achieving significant token reduction compared to JSON
while maintaining readability.
"""

import inspect
import json
import types
from typing import Any, Literal, Union, get_args, get_origin

from dspy.adapters.base import Adapter
from dspy.signatures.signature import Signature
from pydantic import BaseModel

from .toon import decode, encode

# Comment symbol for schema descriptions
COMMENT_SYMBOL = "#"


def _render_type_str(
    annotation: Any,
    depth: int = 0,
    indent: int = 0,
    seen_models: set[type] | None = None,
) -> str:
    """Recursively renders a type annotation into TOON-like schema string."""
    # Primitive types
    if annotation is str:
        return "string"
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    if annotation is bool:
        return "boolean"

    # Pydantic models
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return _build_toon_schema(annotation, indent, seen_models)

    try:
        origin = get_origin(annotation)
        args = get_args(annotation)
    except Exception:
        return str(annotation)

    # Optional[T] or T | None
    if origin in (types.UnionType, Union):
        non_none_args = [arg for arg in args if arg is not type(None)]
        type_render = " or ".join(
            [_render_type_str(arg, depth + 1, indent, seen_models) for arg in non_none_args]
        )
        if len(non_none_args) < len(args):
            return f"{type_render} or null"
        return type_render

    # Literal[T1, T2, ...]
    if origin is Literal:
        return " or ".join(f'"{arg}"' for arg in args)

    # list[T]
    if origin is list:
        inner_type = args[0] if args else Any
        if inspect.isclass(inner_type) and issubclass(inner_type, BaseModel):
            fields = list(inner_type.model_fields.keys())
            fields_str = ",".join(fields)
            return f"[COUNT]{{{fields_str}}}:\n  value1,value2,...\n  (one row per item, COUNT = number of items)"
        else:
            inner_str = _render_type_str(inner_type, depth + 1, indent, seen_models)
            return f"[COUNT]: {inner_str},... (COUNT = number of items)"

    # dict[K, V]
    if origin is dict:
        key_type = _render_type_str(args[0], depth + 1, indent, seen_models) if args else "string"
        val_type = (
            _render_type_str(args[1], depth + 1, indent, seen_models) if len(args) > 1 else "any"
        )
        return f"dict[{key_type}, {val_type}]"

    # Fallback
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation)


def _build_toon_schema(
    pydantic_model: type[BaseModel],
    indent: int = 0,
    seen_models: set[type] | None = None,
) -> str:
    """Builds a TOON-style schema from a Pydantic model."""
    seen_models = seen_models or set()

    if pydantic_model in seen_models:
        return f"<{pydantic_model.__name__}>"

    seen_models.add(pydantic_model)

    lines = []
    current_indent = "  " * indent

    for name, field in pydantic_model.model_fields.items():
        if field.description:
            lines.append(f"{current_indent}{COMMENT_SYMBOL} {field.description}")

        rendered_type = _render_type_str(
            field.annotation, indent=indent + 1, seen_models=seen_models
        )

        if "\n" in rendered_type:
            lines.append(f"{current_indent}{name}:")
            for line in rendered_type.split("\n"):
                lines.append(f"{current_indent}  {line}")
        else:
            lines.append(f"{current_indent}{name}: {rendered_type}")

    return "\n".join(lines)


def _get_output_schema(field_name: str, field_type: Any) -> str:
    """Generate TOON output schema for a field."""
    origin = get_origin(field_type)
    args = get_args(field_type)

    # List of Pydantic models -> tabular format
    if origin is list and args:
        inner_type = args[0]
        if inspect.isclass(inner_type) and issubclass(inner_type, BaseModel):
            fields = list(inner_type.model_fields.keys())
            fields_str = ",".join(fields)
            # Show example with actual count
            return f"""{field_name}:
[2]{{{fields_str}}}:
  Alice,35,engineer
  Bob,28,designer
(Replace 2 with actual count, add one row per item)"""

    # Pydantic model -> object format
    if inspect.isclass(field_type) and issubclass(field_type, BaseModel):
        schema = _build_toon_schema(field_type, indent=1)
        return f"{field_name}:\n{schema}"

    # List of primitives
    if origin is list:
        inner_str = _render_type_str(args[0] if args else Any)
        return f"{field_name}: [3]: val1,val2,val3 (replace 3 with actual count)"

    # Simple types
    return f"{field_name}: {_render_type_str(field_type)}"


class ToonAdapter(Adapter):
    """DSPy adapter using TOON (Token-Oriented Object Notation) format.

    TOON achieves significant token reduction compared to JSON while maintaining
    readability. This adapter generates TOON-formatted schemas and parses
    TOON responses from LLMs.

    Key features:
    - Compact `key: value` syntax (no braces/brackets for objects)
    - Tabular format `[N,]{fields}:` for uniform object arrays
    - Falls back to JSON parsing when TOON fails

    Example Usage:
        ```python
        import dspy
        from pydantic import BaseModel, Field
        from dspy_toon import ToonAdapter

        class Person(BaseModel):
            name: str = Field(description="Full name")
            age: int

        class ExtractPerson(dspy.Signature):
            '''Extract person from text.'''
            text: str = dspy.InputField()
            person: Person = dspy.OutputField()

        llm = dspy.LM("openai/gpt-4o-mini")
        dspy.configure(lm=llm, adapter=ToonAdapter())

        extractor = dspy.Predict(ExtractPerson)
        result = extractor(text="Alice is 30 years old.")
        print(result.person)
        ```

    TOON Format Examples:
        Simple object:
        ```
        name: Alice
        age: 30
        ```

        Tabular array:
        ```
        [2,]{id,name}:
          1,Alice
          2,Bob
        ```
    """

    def format_field_description(self, signature: type[Signature]) -> str:
        """Format input/output field descriptions."""
        sections = []

        if signature.input_fields:
            sections.append("Input fields:")
            for name, field in signature.input_fields.items():
                desc = f" - {field.description}" if field.description else ""
                sections.append(f"  {name}: {_render_type_str(field.annotation)}{desc}")

        if signature.output_fields:
            sections.append("\nOutput fields:")
            for name, field in signature.output_fields.items():
                desc = f" - {field.description}" if field.description else ""
                sections.append(f"  {name}: {_render_type_str(field.annotation)}{desc}")

        return "\n".join(sections)

    def format_field_structure(self, signature: type[Signature]) -> str:
        """Format the output structure instructions in TOON format."""
        sections = []

        sections.append("""
TOON Format (NOT JSON):
- Simple values: key: value
- Arrays of objects use tabular format with header:
  [COUNT]{field1,field2}:
    value1,value2
    value3,value4
  where COUNT is the actual number of items
- No JSON braces {} or brackets []
- No quotes around simple strings
""")

        sections.append("Output structure:")
        for name, field in signature.output_fields.items():
            sections.append(_get_output_schema(name, field.annotation))

        return "\n".join(sections)

    def format_task_description(self, signature: type[Signature]) -> str:
        """Format the task description from signature docstring."""
        return signature.__doc__ or "Complete the task based on the inputs."

    def format_demos(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Format few-shot examples."""
        messages = []

        for demo in demos:
            # Format input
            input_parts = []
            for name in signature.input_fields.keys():
                if name in demo:
                    value = demo[name]
                    if isinstance(value, BaseModel):
                        input_parts.append(f"{name}:\n{encode(value.model_dump())}")
                    else:
                        input_parts.append(f"{name}: {value}")

            # Format output
            output_parts = []
            for name in signature.output_fields.keys():
                if name in demo:
                    value = demo[name]
                    if isinstance(value, BaseModel):
                        output_parts.append(f"{name}:\n{encode(value.model_dump())}")
                    elif isinstance(value, list):
                        output_parts.append(f"{name}:\n{encode(value)}")
                    else:
                        output_parts.append(f"{name}: {value}")

            if input_parts:
                messages.append({"role": "user", "content": "\n".join(input_parts)})
            if output_parts:
                messages.append({"role": "assistant", "content": "\n".join(output_parts)})

        return messages

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        """Format the user message with inputs."""
        parts = []
        if prefix:
            parts.append(prefix)

        for name, field in signature.input_fields.items():
            if name in inputs:
                value = inputs[name]
                if isinstance(value, BaseModel):
                    parts.append(f"{name}:\n{encode(value.model_dump())}")
                elif isinstance(value, (list, dict)):
                    parts.append(f"{name}:\n{encode(value)}")
                else:
                    parts.append(f"{name}: {value}")

        if main_request:
            parts.append("\nProvide output in TOON format as shown above.")

        if suffix:
            parts.append(suffix)

        return "\n\n".join(parts)

    def format_conversation_history(
        self,
        signature: type[Signature],
        history_field_name: str,
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Format conversation history."""
        history = inputs.pop(history_field_name, [])
        messages = []

        for entry in history:
            if isinstance(entry, dict):
                if "user" in entry:
                    messages.append({"role": "user", "content": str(entry["user"])})
                if "assistant" in entry:
                    messages.append({"role": "assistant", "content": str(entry["assistant"])})

        return messages

    def _get_history_field_name(self, signature: type[Signature]) -> str | None:
        """Check if signature has a history field."""
        for name, field in signature.input_fields.items():
            if "history" in name.lower():
                return name
        return None

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        """Parse TOON-formatted LLM output into field values.

        Attempts to parse as TOON first, falls back to JSON if that fails.
        """
        result = {}
        completion = completion.strip()

        # Try parsing each output field
        for field_name, field in signature.output_fields.items():
            value = self._extract_field_value(completion, field_name, field.annotation)
            if value is not None:
                result[field_name] = value

        # If we got results, return them
        if result:
            return result

        # Try full TOON parsing
        try:
            parsed = decode(completion)
            if isinstance(parsed, dict):
                for field_name, field in signature.output_fields.items():
                    if field_name in parsed:
                        result[field_name] = self._convert_field(
                            parsed[field_name], field.annotation
                        )
                if result:
                    return result
        except Exception:
            pass

        # Try JSON parsing as fallback
        try:
            json_str = completion
            if "```json" in completion:
                json_str = completion.split("```json")[1].split("```")[0]
            elif "```" in completion:
                json_str = completion.split("```")[1].split("```")[0]

            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                for field_name, field in signature.output_fields.items():
                    if field_name in parsed:
                        result[field_name] = self._convert_field(
                            parsed[field_name], field.annotation
                        )
        except Exception:
            pass

        return result

    def _extract_field_value(self, completion: str, field_name: str, field_type: Any) -> Any | None:
        """Extract a specific field value from TOON output."""
        import re

        # Look for field_name: followed by tabular array
        # Pattern: field_name:\n[COUNT]{fields}:\n  rows...
        pattern = rf"{field_name}:\s*\n(\[\d+\]\{{[^}}]+\}}:[\s\S]*?)(?=\n\w+:|$)"
        match = re.search(pattern, completion)

        if match:
            toon_array = match.group(1).strip()
            try:
                parsed = decode(toon_array)
                return self._convert_field(parsed, field_type)
            except Exception:
                pass

        # Look for simple field_name: value
        pattern = rf"^{field_name}:\s*(.+)$"
        match = re.search(pattern, completion, re.MULTILINE)

        if match:
            value_str = match.group(1).strip()
            # Check if it's start of a nested structure
            if not value_str or value_str.startswith("["):
                return None

            try:
                parsed = decode(value_str)
                return self._convert_field(parsed, field_type)
            except Exception:
                return value_str

        return None

    def _convert_field(self, value: Any, field_type: Any) -> Any:
        """Convert parsed value to the expected field type."""
        origin = get_origin(field_type)
        args = get_args(field_type)

        # Handle Optional types
        if origin in (types.UnionType, Union):
            non_none_args = [arg for arg in args if arg is not type(None)]
            if non_none_args:
                return self._convert_field(value, non_none_args[0])

        # List of Pydantic models
        if origin is list and args:
            inner_type = args[0]
            if inspect.isclass(inner_type) and issubclass(inner_type, BaseModel):
                if isinstance(value, list):
                    return [
                        inner_type.model_validate(item) if isinstance(item, dict) else item
                        for item in value
                    ]

        # Pydantic model
        if inspect.isclass(field_type) and issubclass(field_type, BaseModel):
            if isinstance(value, dict):
                return field_type.model_validate(value)

        return value
