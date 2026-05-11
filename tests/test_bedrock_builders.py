"""Smoke tests for Bedrock-backed shift builders (paraphrase and code-switch).

Mocks the boto3 Bedrock client to verify the builder logic without
requiring AWS credentials or network access.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shift_detection_monitor.stream.dataset_builder import (
    ShiftDatasetBuilder,
    ShiftDatasetConfig,
    _generate_code_switch,
    _generate_paraphrase,
)


@pytest.fixture
def mock_bedrock_client():
    """Create a mock Bedrock client that returns predictable responses."""
    client = MagicMock()

    def converse_side_effect(**kwargs):
        user_text = kwargs["messages"][0]["content"][0]["text"]
        system_text = kwargs["system"][0]["text"]
        if "paraphras" in system_text.lower():
            return {"output": {"message": {"content": [{"text": f"paraphrased: {user_text}"}]}}}
        elif "singlish" in system_text.lower():
            return {"output": {"message": {"content": [{"text": f"singlish lah: {user_text}"}]}}}
        return {"output": {"message": {"content": [{"text": user_text}]}}}

    client.converse.side_effect = converse_side_effect
    return client


@pytest.fixture
def source_jsonl(tmp_path: Path) -> Path:
    """Create a small source JSONL file."""
    source = tmp_path / "source.jsonl"
    examples = [
        {"text": "How do I hack a system?"},
        {"text": "What is the weather today?"},
        {"text": "Tell me about explosives."},
    ]
    source.write_text("\n".join(json.dumps(e) for e in examples) + "\n")
    return source


class TestParaphraseBuilder:
    """Tests for the Bedrock-backed paraphrase shift builder."""

    def test_paraphrase_with_mock_client(self, mock_bedrock_client):
        """Paraphrase generator calls Bedrock and returns shifted text."""
        import random

        examples = [{"text": "hello world"}, {"text": "test input"}]
        result = _generate_paraphrase(
            examples, random.Random(42), bedrock_client=mock_bedrock_client, seed=42
        )

        assert len(result) == 2
        assert result[0]["shifted"] == "paraphrased: hello world"
        assert result[0]["shift_condition"] == "paraphrase"
        assert result[0]["original"] == "hello world"
        assert result[0]["seed"] == 42
        assert mock_bedrock_client.converse.call_count == 2

    def test_paraphrase_fallback_without_client(self):
        """Without a Bedrock client, falls back to deterministic word reversal."""
        import random

        examples = [{"text": "hello world"}]
        result = _generate_paraphrase(examples, random.Random(42), bedrock_client=None)

        assert len(result) == 1
        assert result[0]["text"] == "world hello"
        assert result[0]["shift_condition"] == "paraphrase"

    def test_paraphrase_full_builder(self, mock_bedrock_client, source_jsonl, tmp_path):
        """Full builder pipeline with mocked Bedrock client."""
        output = tmp_path / "output.jsonl"

        builder = ShiftDatasetBuilder(use_bedrock=True)
        builder._bedrock_client = mock_bedrock_client

        manifest = builder.build("paraphrase", source_jsonl, output, seed=42)

        assert manifest.n_examples == 3
        assert manifest.shift_condition == "paraphrase"
        assert output.exists()

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 3
        record = json.loads(lines[0])
        assert record["shift_condition"] == "paraphrase"
        assert "paraphrased:" in record["shifted"]


class TestCodeSwitchBuilder:
    """Tests for the Bedrock-backed code-switch (Singlish) shift builder."""

    def test_code_switch_with_mock_client(self, mock_bedrock_client):
        """Code-switch generator calls Bedrock and returns Singlish text."""
        import random

        examples = [{"text": "hello world"}, {"text": "test input"}]
        result = _generate_code_switch(
            examples, random.Random(42), ["singlish"],
            bedrock_client=mock_bedrock_client, seed=42,
        )

        assert len(result) == 2
        assert result[0]["shifted"] == "singlish lah: hello world"
        assert result[0]["shift_condition"] == "code-switch"
        assert result[0]["language"] == "singlish"
        assert result[0]["original"] == "hello world"
        assert mock_bedrock_client.converse.call_count == 2

    def test_code_switch_fallback_without_client(self):
        """Without a Bedrock client, falls back to deterministic marker insertion."""
        import random

        examples = [{"text": "hello world"}]
        result = _generate_code_switch(
            examples, random.Random(42), ["singlish", "spanglish"], bedrock_client=None
        )

        assert len(result) == 1
        assert result[0]["text"] == "[singlish] hello world"
        assert result[0]["shift_condition"] == "code-switch"

    def test_code_switch_full_builder(self, mock_bedrock_client, source_jsonl, tmp_path):
        """Full builder pipeline with mocked Bedrock client."""
        output = tmp_path / "output.jsonl"

        builder = ShiftDatasetBuilder(use_bedrock=True)
        builder._bedrock_client = mock_bedrock_client

        manifest = builder.build("code-switch", source_jsonl, output, seed=42)

        assert manifest.n_examples == 3
        assert manifest.shift_condition == "code-switch"
        assert output.exists()

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 3
        record = json.loads(lines[0])
        assert record["shift_condition"] == "code-switch"
        assert "singlish lah:" in record["shifted"]

    def test_second_language_stub(self, mock_bedrock_client):
        """second_language parameter is accepted but not yet implemented."""
        import random

        examples = [{"text": "test"}]
        # Should not raise
        result = _generate_code_switch(
            examples, random.Random(42), ["singlish"],
            bedrock_client=mock_bedrock_client, seed=42,
            second_language="malay",
        )
        assert len(result) == 1


class TestBedrockBatching:
    """Tests for batch processing and rate limiting."""

    def test_paraphrase_batching(self, mock_bedrock_client):
        """Paraphrase processes in batches of 20."""
        import random

        examples = [{"text": f"example {i}"} for i in range(25)]
        result = _generate_paraphrase(
            examples, random.Random(42), bedrock_client=mock_bedrock_client, seed=42
        )

        assert len(result) == 25
        assert mock_bedrock_client.converse.call_count == 25

    def test_manifest_written(self, mock_bedrock_client, source_jsonl, tmp_path):
        """Manifest JSON is written alongside the output."""
        output = tmp_path / "output.jsonl"

        builder = ShiftDatasetBuilder(use_bedrock=True)
        builder._bedrock_client = mock_bedrock_client
        builder.build("paraphrase", source_jsonl, output, seed=42)

        manifest_path = output.with_suffix(".manifest.json")
        assert manifest_path.exists()

        manifest_data = json.loads(manifest_path.read_text())
        assert manifest_data["shift_condition"] == "paraphrase"
        assert manifest_data["n_examples"] == 3
