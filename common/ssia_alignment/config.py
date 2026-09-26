"""Small YAML configuration loader for the independent retraining branch."""
from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any

import yaml


class ConfigDict(dict):
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error

    def to_dict(self) -> dict:
        return _unwrap(self)


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return ConfigDict({key: _wrap(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


def _unwrap(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _unwrap(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_unwrap(item) for item in value]
    return value


def _merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    bases = payload.pop("_base_", None)
    if bases is None:
        return payload
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for base in bases:
        base_path = Path(base) if Path(base).is_absolute() else path.parent / base
        merged = _merge(merged, _load_yaml(base_path.resolve()))
    return _merge(merged, payload)


def _parse_scalar(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return raw


def load_config(path: str | Path, overrides: list[str] | tuple[str, ...] = ()) -> ConfigDict:
    payload = _load_yaml(Path(path).resolve())
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"configuration override must be key=value, got {override!r}")
        dotted, raw = override.split("=", 1)
        node = payload
        parts = dotted.split(".")
        for key in parts[:-1]:
            node = node.setdefault(key, {})
        node[parts[-1]] = _parse_scalar(raw)
    return _wrap(payload)
