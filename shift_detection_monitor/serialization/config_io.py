"""Configuration serialization, parsing, and pretty-printing.

Supports YAML and JSON formats with round-trip guarantees:
    parse(serialize(config)) == config
    parse(pretty_print(config)) == config
"""

from __future__ import annotations

import json

import yaml

from shift_detection_monitor.config import MonitorConfig
from shift_detection_monitor.types import ConfigValidationError


def serialize_config(config: MonitorConfig) -> str:
    """Serialize a MonitorConfig to a YAML string.

    Parameters
    ----------
    config : MonitorConfig
        The configuration object to serialize.

    Returns
    -------
    str
        A YAML-formatted string representation of the config.
    """
    data = config.model_dump()
    return yaml.dump(data, default_flow_style=False, sort_keys=False)


def parse_config(yaml_str: str) -> MonitorConfig:
    """Parse a YAML string into a MonitorConfig.

    Parameters
    ----------
    yaml_str : str
        A YAML-formatted string.

    Returns
    -------
    MonitorConfig
        The parsed configuration object.

    Raises
    ------
    ConfigValidationError
        If the YAML is invalid or contains invalid/missing fields.
    """
    try:
        data = yaml.safe_load(yaml_str)
    except yaml.YAMLError as exc:
        raise ConfigValidationError(f"Invalid YAML: {exc}") from exc

    if data is None:
        raise ConfigValidationError("Empty YAML document")

    if not isinstance(data, dict):
        raise ConfigValidationError(
            f"Expected a YAML mapping at the top level, got {type(data).__name__}"
        )

    try:
        return MonitorConfig.model_validate(data)
    except Exception as exc:
        raise ConfigValidationError(f"Config validation failed: {exc}") from exc


def serialize_config_json(config: MonitorConfig) -> str:
    """Serialize a MonitorConfig to a JSON string.

    Parameters
    ----------
    config : MonitorConfig
        The configuration object to serialize.

    Returns
    -------
    str
        A JSON-formatted string representation of the config.
    """
    return config.model_dump_json()


def parse_config_json(json_str: str) -> MonitorConfig:
    """Parse a JSON string into a MonitorConfig.

    Parameters
    ----------
    json_str : str
        A JSON-formatted string.

    Returns
    -------
    MonitorConfig
        The parsed configuration object.

    Raises
    ------
    ConfigValidationError
        If the JSON is invalid or contains invalid/missing fields.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ConfigValidationError(f"Invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigValidationError(
            f"Expected a JSON object at the top level, got {type(data).__name__}"
        )

    try:
        return MonitorConfig.model_validate(data)
    except Exception as exc:
        raise ConfigValidationError(f"Config validation failed: {exc}") from exc


def pretty_print_config(config: MonitorConfig) -> str:
    """Produce a human-readable YAML representation of a MonitorConfig.

    The output is a valid YAML config file with section comments
    describing each top-level block.

    Parameters
    ----------
    config : MonitorConfig
        The configuration object to pretty-print.

    Returns
    -------
    str
        A human-readable YAML string that can be parsed back into
        an equivalent MonitorConfig.
    """
    data = config.model_dump()

    section_comments: dict[str, str] = {
        "stream": "# --- Stream Simulator ---",
        "detector": "# --- Shift Detector ---",
        "mmd": "# --- MMD Detector ---",
        "conformal": "# --- Conformal Abstention ---",
        "reference_window": "# --- Reference Window ---",
        "factorial": "# --- Factorial Evaluation Design ---",
        "controls": "# --- Negative / Positive Controls ---",
        "variance": "# --- Variance Decomposition ---",
        "output_dir": "# --- Output ---",
    }

    lines: list[str] = ["# Shift Detection Monitor Configuration", ""]

    for key, value in data.items():
        comment = section_comments.get(key)
        if comment:
            lines.append(comment)

        section_yaml = yaml.dump(
            {key: value}, default_flow_style=False, sort_keys=False
        )
        lines.append(section_yaml)

    return "\n".join(lines)
