"""Property tests for configuration serialization round-trip (Property P9).

**Validates: Requirements 11.4**

P9: For any valid MonitorConfig, parse(pretty_print(parse(yaml))) == parse(yaml);
α round-trip within 1e-9 tolerance.
"""

from __future__ import annotations

from hypothesis import given, settings

from shift_detection_monitor.serialization.config_io import (
    parse_config,
    parse_config_json,
    pretty_print_config,
    serialize_config,
    serialize_config_json,
)
from tests.strategies import st_monitor_config


# ---------------------------------------------------------------------------
# Property P9: YAML round-trip
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(config=st_monitor_config())
def test_yaml_roundtrip(config):
    """parse(serialize(config)) == config for YAML."""
    yaml_str = serialize_config(config)
    restored = parse_config(yaml_str)
    assert restored == config


@settings(max_examples=100)
@given(config=st_monitor_config())
def test_yaml_pretty_print_roundtrip(config):
    """parse(pretty_print(config)) == config."""
    pretty = pretty_print_config(config)
    restored = parse_config(pretty)
    assert restored == config


@settings(max_examples=100)
@given(config=st_monitor_config())
def test_yaml_double_roundtrip(config):
    """parse(pretty_print(parse(serialize(config)))) == parse(serialize(config)).

    This is the full P9 property: serialise → parse → pretty-print → parse
    must be idempotent.
    """
    yaml_str = serialize_config(config)
    first_parse = parse_config(yaml_str)
    pretty = pretty_print_config(first_parse)
    second_parse = parse_config(pretty)
    assert second_parse == first_parse


# ---------------------------------------------------------------------------
# Property P9 (JSON variant): JSON round-trip
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(config=st_monitor_config())
def test_json_roundtrip(config):
    """parse_json(serialize_json(config)) == config."""
    json_str = serialize_config_json(config)
    restored = parse_config_json(json_str)
    assert restored == config


# ---------------------------------------------------------------------------
# α round-trip within 1e-9 tolerance
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(config=st_monitor_config())
def test_alpha_roundtrip_tolerance(config):
    """α survives YAML round-trip within 1e-9."""
    yaml_str = serialize_config(config)
    restored = parse_config(yaml_str)
    assert abs(restored.detector.alpha - config.detector.alpha) < 1e-9
    assert (
        abs(restored.conformal.target_error_rate - config.conformal.target_error_rate)
        < 1e-9
    )


# ---------------------------------------------------------------------------
# Example tests: ConfigValidationError on invalid / missing fields
# ---------------------------------------------------------------------------

import pytest

from shift_detection_monitor.types import ConfigValidationError


class TestConfigValidationErrors:
    """Example-based tests for invalid configuration handling."""

    def test_invalid_alpha_too_high(self):
        """Alpha >= 1.0 must raise ConfigValidationError."""
        yaml_str = "detector:\n  alpha: 1.5\n"
        with pytest.raises(ConfigValidationError, match="Config validation failed"):
            parse_config(yaml_str)

    def test_invalid_alpha_zero(self):
        """Alpha == 0 must raise ConfigValidationError."""
        yaml_str = "detector:\n  alpha: 0.0\n"
        with pytest.raises(ConfigValidationError, match="Config validation failed"):
            parse_config(yaml_str)

    def test_invalid_alpha_negative(self):
        """Negative alpha must raise ConfigValidationError."""
        yaml_str = "detector:\n  alpha: -0.1\n"
        with pytest.raises(ConfigValidationError, match="Config validation failed"):
            parse_config(yaml_str)

    def test_invalid_window_size_too_small(self):
        """Window size < 10 must raise ConfigValidationError."""
        yaml_str = "detector:\n  window_size: 3\n"
        with pytest.raises(ConfigValidationError, match="Config validation failed"):
            parse_config(yaml_str)

    def test_invalid_mixing_proportion(self):
        """Mixing proportion > 1.0 must raise ConfigValidationError."""
        yaml_str = "stream:\n  mixing_proportion: 1.5\n"
        with pytest.raises(ConfigValidationError, match="Config validation failed"):
            parse_config(yaml_str)

    def test_invalid_window_mode(self):
        """Invalid window_mode must raise ConfigValidationError."""
        yaml_str = "detector:\n  window_mode: invalid_mode\n"
        with pytest.raises(ConfigValidationError, match="Config validation failed"):
            parse_config(yaml_str)

    def test_invalid_yaml_syntax(self):
        """Malformed YAML must raise ConfigValidationError."""
        yaml_str = "detector:\n  alpha: [unclosed"
        with pytest.raises(ConfigValidationError, match="Invalid YAML"):
            parse_config(yaml_str)

    def test_empty_yaml(self):
        """Empty YAML document must raise ConfigValidationError."""
        with pytest.raises(ConfigValidationError, match="Empty YAML"):
            parse_config("")

    def test_non_mapping_yaml(self):
        """YAML that is not a mapping must raise ConfigValidationError."""
        with pytest.raises(ConfigValidationError, match="Expected a YAML mapping"):
            parse_config("- item1\n- item2\n")

    def test_invalid_json_syntax(self):
        """Malformed JSON must raise ConfigValidationError."""
        with pytest.raises(ConfigValidationError, match="Invalid JSON"):
            parse_config_json("{bad json")

    def test_non_object_json(self):
        """JSON that is not an object must raise ConfigValidationError."""
        with pytest.raises(ConfigValidationError, match="Expected a JSON object"):
            parse_config_json("[1, 2, 3]")

    def test_invalid_reference_window_min_size(self):
        """Reference window min_size < 50 must raise ConfigValidationError."""
        yaml_str = "reference_window:\n  size: 100\n  min_size: 10\n"
        with pytest.raises(ConfigValidationError, match="Config validation failed"):
            parse_config(yaml_str)

    def test_invalid_n_bootstrap_too_low(self):
        """n_bootstrap < 100 must raise ConfigValidationError."""
        yaml_str = "mmd:\n  n_bootstrap: 50\n"
        with pytest.raises(ConfigValidationError, match="Config validation failed"):
            parse_config(yaml_str)
