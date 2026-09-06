# Copyright (c) 2025 dspy-toon
# SPDX-License-Identifier: MIT
"""DSPy adapter for TOON 4.1 structured inputs and outputs."""

import inspect
import json
import logging
import types
from typing import Annotated, Any, Literal, TypeGuard, Union, get_args, get_origin

from dspy.adapters.base import Adapter  # type: ignore[import-untyped]
from dspy.adapters.types import History  # type: ignore[import-untyped]
from dspy.adapters.utils import apply_output_field_defaults  # type: ignore[import-untyped]
from dspy.signatures.signature import Signature  # type: ignore[import-untyped]
from dspy.utils.exceptions import AdapterParseError  # type: ignore[import-untyped]
from pydantic import BaseModel, TypeAdapter

from .toon import decode, encode, encode_key

logger = logging.getLogger(__name__)


def _is_model(annotation: Any) -> TypeGuard[type[BaseModel]]:
    return inspect.isclass(annotation) and issubclass(annotation, BaseModel)


def _schema_value(annotation: Any, seen: frozenset[type] = frozenset()) -> Any:
    """Create type placeholders with the same container shape as the annotation."""
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        return _schema_value(args[0], seen)
    if origin in (Union, types.UnionType):
        return _schema_value(next((a for a in args if a is not type(None)), type(None)), seen)
    if origin is Literal:
        return args[0]
    if _is_model(annotation):
        if annotation in seen:
            return f"<{annotation.__name__}>"
        return {n: _schema_value(f.annotation, seen | {annotation}) for n, f in annotation.model_fields.items()}
    if origin is list:
        inner = args[0] if args else Any
        return [_schema_value(inner, seen), _schema_value(inner, seen)]
    if origin is dict:
        inner = args[1] if len(args) > 1 else Any
        return {"key1": _schema_value(inner, seen), "key2": _schema_value(inner, seen)}
    return {str: "string", int: "int", float: "float", bool: "boolean", type(None): None}.get(annotation, "any")


def _nested_descriptions(annotation: Any, path: str, seen: frozenset[type]) -> list[str]:
    """Retain nested field guidance when an array schema is rendered as rows."""
    if _is_model(annotation):
        if annotation in seen:
            return []
        lines: list[str] = []
        for name, field in annotation.model_fields.items():
            field_path = f"{path}.{name}"
            if field.description:
                lines.extend(f"# {field_path}: {line}" for line in field.description.splitlines())
            lines.extend(_nested_descriptions(field.annotation, field_path, seen | {annotation}))
        return lines
    suffix = "[]" if get_origin(annotation) is list else ""
    return [line for arg in get_args(annotation) for line in _nested_descriptions(arg, path + suffix, seen)]


def _render_type_str(
    annotation: Any,
    depth: int = 0,
    indent: int = 0,
    seen_models: set[type] | None = None,
    field_name: str | None = None,
) -> str:
    """Render a type as a TOON-shaped schema; placeholders are not literal output."""
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        return _render_type_str(args[0], depth, indent, seen_models, field_name)
    if origin in (Union, types.UnionType):
        return " or ".join(
            "null" if a is type(None) else _render_type_str(a, depth + 1, indent, seen_models, field_name) for a in args
        )
    if origin is Literal:
        return " or ".join(json.dumps(a, ensure_ascii=False) for a in args)
    if _is_model(annotation):
        return _build_toon_schema(annotation, indent, seen_models)
    if origin is list:
        sample = _schema_value(annotation, frozenset(seen_models or ()))
        shape = encode({field_name: sample} if field_name else sample).replace("[2]", "[COUNT]")
        descriptions = _nested_descriptions(annotation, field_name or "items", frozenset(seen_models or ()))
        return "\n".join([*descriptions, shape])
    if origin is dict:
        key = _render_type_str(args[0]) if args else "string"
        value = _render_type_str(args[1], seen_models=seen_models) if len(args) > 1 else "any"
        return f"dict[{key}, {value}]"
    return {str: "string", int: "int", float: "float", bool: "boolean", type(None): "null"}.get(
        annotation, getattr(annotation, "__name__", str(annotation))
    )


def _build_toon_schema(
    pydantic_model: type[BaseModel],
    indent: int = 0,
    seen_models: set[type] | None = None,
) -> str:
    """Build a schema with branch-local recursion tracking and field descriptions."""
    seen = set(seen_models or ())
    if pydantic_model in seen:
        return f"<{pydantic_model.__name__}>"
    seen.add(pydantic_model)
    lines: list[str] = []
    for name, field in pydantic_model.model_fields.items():
        if field.description:
            lines.extend(f"# {line}" for line in field.description.splitlines())
        annotation = field.annotation
        args = get_args(annotation)
        non_null = [a for a in args if a is not type(None)]
        inner = non_null[0] if get_origin(annotation) in (Union, types.UnionType) and len(non_null) == 1 else annotation
        if get_origin(inner) is list:
            lines.append(_render_type_str(annotation, seen_models=seen, field_name=name))
        elif _is_model(inner):
            lines.append(f"{encode_key(name)}:")
            lines.append(_build_toon_schema(inner, 1, seen))
            if inner is not annotation:
                lines.append(f"# {name} may instead be null")
        else:
            lines.append(f"{encode_key(name)}: {_render_type_str(annotation, seen_models=seen)}")
    return "\n".join("  " * indent + line for part in lines for line in part.splitlines())


def _get_output_schema(field_name: str, field_type: Any) -> str:
    """Generate a shape-correct example from the type, using the real encoder."""
    return encode({field_name: _schema_value(field_type)})


class ToonAdapter(Adapter):
    """Use TOON 4.1 for structured prompts and validated DSPy outputs.

    Inherits DSPy's synchronous/asynchronous calls, callbacks, native response
    handling and demo formatting. Complete JSON responses are accepted as a
    fallback; malformed TOON and invalid field values raise AdapterParseError.
    """

    def format_field_description(self, signature: type[Signature]) -> str:
        """Describe signature fields and their types."""
        sections = []
        for label, fields in (("Input", signature.input_fields), ("Output", signature.output_fields)):
            if fields:
                sections.append(f"{label} fields:")
                for name, field in fields.items():
                    desc = f" - {field.description}" if field.description else ""
                    sections.append(f"  {name}: {_render_type_str(field.rebuild_annotation())}{desc}")
        return "\n".join(sections)

    def format_field_structure(self, signature: type[Signature]) -> str:
        """Explain the wire format separately from the type placeholders."""
        rules = """Respond with a TOON 4.1 object containing every output field. No code fences.
- Simple values use key: value; booleans are true/false and absent nullable values are null.
- Empty arrays are field: []; null and [] are different values.
- Primitive arrays: field[COUNT]: item1,item2. Replace COUNT with the actual length.
- Uniform object arrays: field[COUNT]{name,age}: followed by one row per item, indented two spaces.
- Nested uniform columns: field[COUNT]{name,address{city,country}}: with flat rows in header order.
- Dictionaries of uniform objects: field[COUNT:]{age,city}: followed by indented key: age,city rows.
- Other arrays use an indented - item per element. For - field: on a list-item line,
  that field's children are four spaces deeper than the hyphen; sibling fields are two spaces deeper.
- Quote strings containing delimiters, colons, quotes, brackets, backslashes or control characters,
  strings starting with # or -, numeric-looking strings, and literal strings true/false/null.
- Escape newlines inside strings as \\n, tabs as \\t and quotes as \\".

Output shape examples (type words are placeholders; use actual values and counts):"""
        return "\n".join([rules, *(_get_output_schema(n, f.annotation) for n, f in signature.output_fields.items())])

    def format_task_description(self, signature: type[Signature]) -> str:
        """Use the signature's task instructions."""
        return signature.instructions or "Complete the task based on the inputs."

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        """Encode the complete input object so nested values retain their depth."""
        content = encode({n: inputs[n] for n in signature.input_fields if n in inputs})
        request = "Provide output in TOON format as shown above." if main_request else ""
        return "\n\n".join(part for part in (prefix, content, request, suffix) if part)

    def format_assistant_message_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message: str | None = None,
    ) -> str:
        """Encode demonstration fields together, preserving explicit null values."""
        values = {
            n: outputs[n] if n in outputs else missing_field_message
            for n in signature.output_fields
            if n in outputs or missing_field_message is not None
        }
        return encode(values)

    def format_conversation_history(
        self,
        signature: type[Signature],
        history_field_name: str,
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Delegate DSPy History to DSPy; retain legacy user/assistant pairs."""
        history = inputs.get(history_field_name)
        if history is None:
            inputs.pop(history_field_name, None)
            return []
        if isinstance(history, History):
            return super().format_conversation_history(signature, history_field_name, inputs)
        if not isinstance(history, list):
            raise TypeError("History must be a dspy.History or a list of messages")
        messages: list[dict[str, Any]] = []
        for message in history:
            if "user" in message or "assistant" in message:
                messages.extend(
                    {"role": role, "content": str(message[role])} for role in ("user", "assistant") if role in message
                )
            else:
                messages.extend(
                    [
                        {"role": "user", "content": self.format_user_message_content(signature, message)},
                        {"role": "assistant", "content": self.format_assistant_message_content(signature, message)},
                    ]
                )
        inputs.pop(history_field_name, None)
        return messages

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        """Parse one complete document, then validate every output annotation."""
        text = completion.strip()
        if text.startswith("```") and text.endswith("```"):
            _, separator, body = text.partition("\n")
            if separator:
                text = body[:-3].strip()
        parsed: Any = None
        try:
            if text.startswith("{"):
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    parsed = decode(text)
            else:
                parsed = decode(text)
            if not isinstance(parsed, dict) or parsed.keys() - signature.output_fields.keys():
                raise ValueError("Response must be an object containing only signature output fields")
            parsed = apply_output_field_defaults(signature, parsed)
            if parsed.keys() != signature.output_fields.keys():
                raise ValueError("Response is missing required output fields")
            return {
                name: TypeAdapter(field.rebuild_annotation()).validate_python(parsed[name])
                for name, field in signature.output_fields.items()
            }
        except (ValueError, TypeError) as error:
            logger.debug("TOON response parsing failed: %s", error)
            raise AdapterParseError(
                adapter_name="ToonAdapter",
                signature=signature,
                lm_response=completion,
                parsed_result=parsed if isinstance(parsed, dict) else {},
                message=str(error),
            ) from error
