"""Turning frozen dataclasses into JSON and back.

Records cross a subprocess boundary and land in run artifacts, so both directions
are needed. :func:`decode` follows a class's own type hints rather than a
hand-written field list, which cannot drift when a field is added.
"""

from __future__ import annotations

import dataclasses
import json
import types as _types
import typing
from typing import Any, TypeVar

T = TypeVar("T")


def to_json(obj: Any, *, indent: int | None = 2) -> str:
    """Serialise a dataclass, or a dict/list containing them (tuples become
    JSON arrays)."""
    return json.dumps(_plain(obj), indent=indent, default=str)


def _plain(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def _convert(annotation: Any, value: Any) -> Any:
    if value is None:
        return None
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, _types.UnionType):
        inner = [a for a in typing.get_args(annotation) if a is not type(None)]
        return _convert(inner[0], value) if len(inner) == 1 else value
    if origin is tuple:
        item = typing.get_args(annotation)[0]
        return tuple(_convert(item, v) for v in value)
    if dataclasses.is_dataclass(annotation):
        return decode(annotation, value)
    return value


def decode(cls: type[T], data: dict | None) -> T | None:
    """Rebuild a dataclass from parsed JSON, following its own type hints.

    Measurements cross a subprocess boundary as JSON. A hand-written decoder per
    class invites the bug where a field is added to the dataclass and silently
    dropped on the way back; this cannot drift because it reads the annotations.
    """
    if data is None:
        return None
    hints = typing.get_type_hints(cls)
    return cls(**{
        f.name: _convert(hints[f.name], data.get(f.name))
        for f in dataclasses.fields(cls)
        if f.name in data
    })
