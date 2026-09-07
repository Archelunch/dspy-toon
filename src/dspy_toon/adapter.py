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


def _field_constraints(annotation: Any, path: str) -> list[str]:
    """Describe validation constraints, including those inside array models."""
    schema = TypeAdapter(annotation).json_schema()
    definitions = schema.get("$defs", {})
    lines: list[str] = []

    def visit(node: dict[str, Any], name: str, seen: frozenset[str]) -> None:
        ref = node.get("$ref")
        if ref:
            if ref in seen:
                lines.append(f"{name}: same structure as {ref.rsplit('/', 1)[-1]}")
                return
            node = {**definitions.get(ref.rsplit("/", 1)[-1], {}), **node}
            seen = seen | {ref}
        rules = []
        if "enum" in node:
            rules.append("one of " + ", ".join(json.dumps(v, ensure_ascii=False) for v in node["enum"]))
        if "const" in node:
            rules.append("exactly " + json.dumps(node["const"], ensure_ascii=False))
        for key, label in (
            ("minimum", "minimum"),
            ("maximum", "maximum"),
            ("exclusiveMinimum", "greater than"),
            ("exclusiveMaximum", "less than"),
            ("multipleOf", "multiple of"),
            ("minLength", "minimum characters"),
            ("maxLength", "maximum characters"),
            ("minItems", "minimum items"),
            ("maxItems", "maximum items"),
            ("pattern", "pattern"),
            ("format", "format"),
            ("minProperties", "minimum entries"),
            ("maxProperties", "maximum entries"),
        ):
            if key in node:
                rules.append(f"{label}: {node[key]}")
        if rules:
            lines.append(f"{name}: {'; '.join(rules)}")
        for operator in ("anyOf", "oneOf", "allOf"):
            branches = node.get(operator, [])
            if operator != "allOf" and any(branch.get("type") == "null" for branch in branches):
                lines.append(f"{name}: null is allowed")
            for index, branch in enumerate(branches):
                label = name if operator == "allOf" else f"{name} (alternative {index + 1})"
                visit(branch, label, seen)
        properties = node.get("properties", {})
        if properties:
            required = node.get("required", [])
            if required and len(required) != len(properties):
                lines.append(f"{name}: required fields {', '.join(required)}")
            for key, child in properties.items():
                visit(child, f"{name}.{key}", seen)
        if isinstance(node.get("items"), dict):
            visit(node["items"], name + "[]", seen)
        if isinstance(node.get("additionalProperties"), dict):
            visit(node["additionalProperties"], name + ".*", seen)

    visit(schema, path, frozenset())
    return lines


def _container_features(annotation: Any, seen: frozenset[type] = frozenset()) -> set[str]:
    """Select syntax rules from both input and output container types."""
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation is Any:
        return {"array", "object", "keyed", "table"}
    if _is_model(annotation):
        if annotation in seen:
            return {"object"}
        return {"object"}.union(
            *(_container_features(field.annotation, seen | {annotation}) for field in annotation.model_fields.values())
        )
    features: set[str] = set()
    if origin is list:
        features.add("array")
        inner = args[0] if args else Any
        if "object" in _container_features(inner, seen):
            features.add("table")
    elif origin is dict:
        features.add("object")
        inner = args[1] if len(args) > 1 else Any
        if "object" in _container_features(inner, seen):
            features.add("keyed")
    for arg in args:
        if (arg is not Ellipsis and isinstance(arg, type)) or get_origin(arg) is not None:
            features.update(_container_features(arg, seen))

    return features


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
                    if label == "Output":
                        sections.extend(_field_constraints(field.rebuild_annotation(), name))
        return "\n".join(sections)

    def format_field_structure(self, signature: type[Signature]) -> str:
        """Explain the wire format separately from the type placeholders."""
        features = set().union(*(_container_features(field.annotation) for field in signature.fields.values()))
        rules = [
            "Respond with only the declared output fields. Include the shown object fields. No prose or code fences.",
            "Simple values use key: value. Booleans are true/false; use null only for nullable fields.",
            'String values must retain their string type: write "749", "false" or "null" when these are strings.',
            "Quote strings containing commas, colons, brackets, quotes, backslashes or control characters, "
            "or starting with # or -.",
            r"Inside quoted strings, escape backslash as \\, newline as \n, tab as \t and quote as \". "
            r'For example, a literal backslash before x is written "\\x". Do not use escapes such as \( or \m.',
        ]
        if "object" in features:
            rules.append("Nested fields go on following lines, indented two more spaces than their parent.")
        if "array" in features:
            rules.append(
                "Empty arrays use field: []. Primitive arrays use field[COUNT]: item1,item2. "
                "COUNT is the actual number of items. Double-quote each string item or cell, including simple strings. "
                'For example: cities[2]: "Paris, France","Berlin" has two items.'
            )
        if "table" in features:
            rules.extend(
                [
                    "Uniform object arrays use field[COUNT]{name,age}: followed by indented rows in column order. "
                    "Each row must contain one cell per column. Quote individual cells, never a whole row.",
                    "Nested uniform columns use field[COUNT]{name,address{city,country}}: "
                    "with flat rows in header order.",
                ]
            )
        if "array" in features and "object" in features:
            rules.append(
                "Other arrays use an indented - item per element. For - field: on a list-item line, "
                "children are four spaces deeper than the hyphen; sibling fields are two spaces deeper."
            )
        if "keyed" in features:
            rules.append(
                "Dictionaries of uniform objects use field[COUNT:]{age,city}: followed by indented key: age,city rows. "
                "COUNT is the number of keyed entries."
            )
        rules.append("Output shape examples (type words are placeholders; use actual values and counts):")
        return "\n".join([*rules, *(_get_output_schema(n, f.annotation) for n, f in signature.output_fields.items())])

    def user_message_output_requirements(self, signature: type[Signature]) -> str:
        """Repeat the output fields and type-sensitive rules after the input."""
        fields = ", ".join(encode_key(name) for name in signature.output_fields)
        lines = [f"Return only these output fields in order: {fields}."]
        for name, field in signature.output_fields.items():
            if field.annotation is str:
                lines.append(
                    f"{encode_key(name)} is a string. Quote numeric-looking answers, "
                    f'for example {encode_key(name)}: "749".'
                )
            elif "array" in _container_features(field.annotation):
                lines.append(
                    f"For {encode_key(name)}, use exact item counts and one cell per header column. "
                    "Double-quote each string item or cell so commas inside a string cannot create extra cells."
                )
        return "\n".join(lines)

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
        request = self.user_message_output_requirements(signature) if main_request else ""
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
