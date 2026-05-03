"""
Offline shift dataset builder for generating, validating, and version-controlling
shifted corpora for each shift condition.

The builder dispatches to per-condition generators that produce deterministic
shifted outputs from reference inputs. Each generator is a skeleton that applies
a deterministic transformation — real LLM API calls and GCG optimization are
placeholders.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ShiftDatasetConfig(BaseModel):
    """Configuration for the ShiftDatasetBuilder."""

    generator_version: str = "0.1.0"
    model_used: str = "deterministic-skeleton"
    default_n_examples: int | None = None
    paraphrase_model: str = "gpt-4"
    code_switch_languages: list[str] = Field(
        default_factory=lambda: ["singlish", "spanglish"]
    )
    adversarial_suffix_length: int = 20
    compositional_context_length: int = 512
    compositional_position: int = 256


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetManifest:
    """Manifest for a generated shift dataset."""

    generator_version: str
    model_used: str
    generation_params: dict[str, Any]
    seed: int
    n_examples: int
    shift_condition: str
    created_at: str  # ISO 8601
    human_validation_flags: dict[str, bool]


@dataclass(frozen=True)
class ValidationReport:
    """Report from validating a shifted corpus against its manifest."""

    shift_condition: str
    n_validated: int
    n_passed: int
    pass_rate: float
    issues: list[str]


# ---------------------------------------------------------------------------
# Per-condition generators
# ---------------------------------------------------------------------------


def _generate_paraphrase(
    examples: list[dict[str, Any]], rng: random.Random
) -> list[dict[str, Any]]:
    """Paraphrase shift: reverse word order as a deterministic placeholder.

    In production this would call a GPT-4-class API at temperature 0.
    """
    result = []
    for ex in examples:
        text = ex.get("text", "")
        words = text.split()
        words.reverse()
        shifted = dict(ex)
        shifted["text"] = " ".join(words)
        shifted["shift_condition"] = "paraphrase"
        result.append(shifted)
    return result


def _generate_code_switch(
    examples: list[dict[str, Any]], rng: random.Random, languages: list[str]
) -> list[dict[str, Any]]:
    """Code-switch shift: insert language markers as a deterministic placeholder.

    In production this would use prompt-engineered translation or LINCE data.
    """
    result = []
    for i, ex in enumerate(examples):
        lang = languages[i % len(languages)]
        text = ex.get("text", "")
        shifted = dict(ex)
        shifted["text"] = f"[{lang}] {text}"
        shifted["shift_condition"] = "code-switch"
        shifted["language"] = lang
        result.append(shifted)
    return result


def _generate_adversarial_suffix(
    examples: list[dict[str, Any]], rng: random.Random, suffix_length: int
) -> list[dict[str, Any]]:
    """Adversarial suffix shift: append a deterministic suffix string.

    In production this would use GCG/nanoGCG optimization.
    """
    # Deterministic suffix from the RNG
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    suffix = "".join(rng.choice(chars) for _ in range(suffix_length))

    result = []
    for ex in examples:
        text = ex.get("text", "")
        shifted = dict(ex)
        shifted["text"] = f"{text} {suffix}"
        shifted["shift_condition"] = "adversarial-suffix"
        result.append(shifted)
    return result


def _generate_compositional(
    examples: list[dict[str, Any]],
    rng: random.Random,
    context_length: int,
    position: int,
) -> list[dict[str, Any]]:
    """Compositional/long-context shift: embed text at configurable position in padding.

    In production this would use configurable padding contexts.
    """
    padding_char = "."
    result = []
    for ex in examples:
        text = ex.get("text", "")
        # Create padding context
        pre_pad = padding_char * min(position, context_length)
        post_pad = padding_char * max(0, context_length - position - len(text))
        shifted = dict(ex)
        shifted["text"] = f"{pre_pad}{text}{post_pad}"
        shifted["shift_condition"] = "compositional-long-context"
        result.append(shifted)
    return result


def _generate_temporal(
    examples: list[dict[str, Any]], rng: random.Random
) -> list[dict[str, Any]]:
    """Temporal shift: filter/copy examples with timestamp metadata.

    In production this would scrape post-cutoff jailbreaks from WildJailbreak etc.
    """
    result = []
    for ex in examples:
        shifted = dict(ex)
        shifted["shift_condition"] = "temporal"
        # Add a synthetic future timestamp if not present
        if "timestamp" not in shifted:
            shifted["timestamp"] = "2025-01-01T00:00:00Z"
        result.append(shifted)
    return result


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

_VALID_CONDITIONS = {
    "paraphrase",
    "code-switch",
    "adversarial-suffix",
    "compositional-long-context",
    "temporal",
}


class ShiftDatasetBuilder:
    """Offline tool for generating, validating, and version-controlling shifted corpora."""

    def __init__(self, config: ShiftDatasetConfig | None = None) -> None:
        self._config = config or ShiftDatasetConfig()

    def build(
        self,
        shift_condition: str,
        source_path: Path,
        output_path: Path,
        seed: int,
    ) -> DatasetManifest:
        """Generate a shifted corpus from source inputs.

        Parameters
        ----------
        shift_condition : str
            One of the five supported shift conditions.
        source_path : Path
            Path to a JSONL file of source examples.
        output_path : Path
            Path where the shifted JSONL will be written.
        seed : int
            Random seed for deterministic generation.

        Returns
        -------
        DatasetManifest
            Manifest describing the generated dataset.
        """
        if shift_condition not in _VALID_CONDITIONS:
            raise ValueError(
                f"Unknown shift condition: {shift_condition!r}. "
                f"Valid conditions: {sorted(_VALID_CONDITIONS)}"
            )

        # Read source examples
        examples = _read_jsonl(source_path)

        # Deterministic RNG
        rng = random.Random(seed)

        # Dispatch to per-condition generator
        if shift_condition == "paraphrase":
            shifted = _generate_paraphrase(examples, rng)
            gen_params: dict[str, Any] = {"model": self._config.paraphrase_model}
        elif shift_condition == "code-switch":
            shifted = _generate_code_switch(
                examples, rng, self._config.code_switch_languages
            )
            gen_params = {"languages": self._config.code_switch_languages}
        elif shift_condition == "adversarial-suffix":
            shifted = _generate_adversarial_suffix(
                examples, rng, self._config.adversarial_suffix_length
            )
            gen_params = {"suffix_length": self._config.adversarial_suffix_length}
        elif shift_condition == "compositional-long-context":
            shifted = _generate_compositional(
                examples,
                rng,
                self._config.compositional_context_length,
                self._config.compositional_position,
            )
            gen_params = {
                "context_length": self._config.compositional_context_length,
                "position": self._config.compositional_position,
            }
        elif shift_condition == "temporal":
            shifted = _generate_temporal(examples, rng)
            gen_params = {}
        else:
            raise ValueError(f"Unhandled shift condition: {shift_condition!r}")

        # Write output JSONL
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_path, shifted)

        # Create manifest
        manifest = DatasetManifest(
            generator_version=self._config.generator_version,
            model_used=self._config.model_used,
            generation_params=gen_params,
            seed=seed,
            n_examples=len(shifted),
            shift_condition=shift_condition,
            created_at=datetime.now(timezone.utc).isoformat(),
            human_validation_flags={},
        )

        # Write manifest alongside the output
        manifest_path = output_path.with_suffix(".manifest.json")
        _write_manifest(manifest_path, manifest)

        return manifest

    def validate(
        self, corpus_path: Path, manifest: DatasetManifest
    ) -> ValidationReport:
        """Validate a shifted corpus against its manifest.

        Checks:
        - Number of examples matches manifest
        - Each example has the expected shift_condition field
        """
        examples = _read_jsonl(corpus_path)
        issues: list[str] = []
        n_passed = 0

        if len(examples) != manifest.n_examples:
            issues.append(
                f"Expected {manifest.n_examples} examples, found {len(examples)}"
            )

        for i, ex in enumerate(examples):
            condition = ex.get("shift_condition")
            if condition == manifest.shift_condition:
                n_passed += 1
            else:
                issues.append(
                    f"Example {i}: expected shift_condition={manifest.shift_condition!r}, "
                    f"got {condition!r}"
                )

        pass_rate = n_passed / len(examples) if examples else 0.0

        return ValidationReport(
            shift_condition=manifest.shift_condition,
            n_validated=len(examples),
            n_passed=n_passed,
            pass_rate=pass_rate,
            issues=issues,
        )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    examples: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write a list of dicts to a JSONL file."""
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def _write_manifest(path: Path, manifest: DatasetManifest) -> None:
    """Write a DatasetManifest to a JSON file."""
    data = {
        "generator_version": manifest.generator_version,
        "model_used": manifest.model_used,
        "generation_params": manifest.generation_params,
        "seed": manifest.seed,
        "n_examples": manifest.n_examples,
        "shift_condition": manifest.shift_condition,
        "created_at": manifest.created_at,
        "human_validation_flags": manifest.human_validation_flags,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
