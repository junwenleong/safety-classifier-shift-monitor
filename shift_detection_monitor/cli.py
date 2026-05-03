"""CLI entry point for the Shift Detection Monitor.

Subcommands:
    run             Run full evaluation from config file
    build-dataset   Build shift dataset
    validate-config Validate a config file
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from shift_detection_monitor.serialization.config_io import parse_config
from shift_detection_monitor.types import ConfigValidationError

logger = logging.getLogger(__name__)


def _cmd_validate_config(args: argparse.Namespace) -> int:
    """Validate a configuration file."""
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        return 1

    try:
        yaml_str = config_path.read_text(encoding="utf-8")
        config = parse_config(yaml_str)
        print(f"Config is valid: {config_path}")
        print(f"  Classifiers: {config.factorial.classifiers}")
        print(f"  Shift conditions: {config.factorial.shift_conditions}")
        print(f"  Regimes: {config.factorial.ground_truth_regimes}")
        print(f"  Window sizes: {config.factorial.window_sizes}")
        print(f"  Seeds: {len(config.factorial.seeds)}")
        print(f"  Alpha: {config.detector.alpha}")
        return 0
    except ConfigValidationError as exc:
        print(f"Config validation error: {exc}", file=sys.stderr)
        return 1


def _cmd_run(args: argparse.Namespace) -> int:
    """Run full evaluation from config file."""
    config_path = Path(args.config)
    output_path = Path(args.output)

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        return 1

    try:
        yaml_str = config_path.read_text(encoding="utf-8")
        config = parse_config(yaml_str)
    except ConfigValidationError as exc:
        print(f"Config validation error: {exc}", file=sys.stderr)
        return 1

    # Import harness here to avoid heavy imports on validate-config
    from shift_detection_monitor.evaluation.harness import EvaluationHarness

    # Build classifier instances
    classifiers = _build_classifiers(config.factorial.classifiers)
    if not classifiers:
        print("Error: No classifiers could be instantiated.", file=sys.stderr)
        return 1

    harness = EvaluationHarness(config=config, classifiers=classifiers)
    results = harness.run(output_path)

    print(f"Evaluation complete. {len(results)} results written to {output_path}")
    return 0


def _cmd_build_dataset(args: argparse.Namespace) -> int:
    """Build shift dataset."""
    from shift_detection_monitor.stream.dataset_builder import (
        ShiftDatasetBuilder,
        ShiftDatasetConfig,
    )

    source_path = Path(args.source)
    output_path = Path(args.output)

    if not source_path.exists():
        print(f"Error: Source file not found: {source_path}", file=sys.stderr)
        return 1

    builder = ShiftDatasetBuilder(ShiftDatasetConfig())
    try:
        manifest = builder.build(
            shift_condition=args.shift_condition,
            source_path=source_path,
            output_path=output_path,
            seed=args.seed,
        )
        print(f"Dataset built: {manifest.n_examples} examples")
        print(f"  Shift condition: {manifest.shift_condition}")
        print(f"  Output: {output_path}")
        return 0
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _build_classifiers(
    classifier_names: list[str],
) -> dict:
    """Attempt to build classifier instances. Returns available classifiers."""
    classifiers = {}

    for name in classifier_names:
        try:
            if name == "deberta-v3-large":
                from shift_detection_monitor.classifiers.deberta import DeBERTaAdapter

                classifiers[name] = DeBERTaAdapter()
            elif name == "llama-guard-3-8b":
                from shift_detection_monitor.classifiers.llama_guard import (
                    LlamaGuard3Adapter,
                )

                classifiers[name] = LlamaGuard3Adapter()
            elif name == "shieldgemma-9b":
                from shift_detection_monitor.classifiers.shieldgemma import (
                    ShieldGemmaAdapter,
                )

                classifiers[name] = ShieldGemmaAdapter()
            elif name == "gpt-oss-safeguard":
                from shift_detection_monitor.classifiers.gpt_oss_safeguard import (
                    GptOssSafeguardAdapter,
                )

                classifiers[name] = GptOssSafeguardAdapter()
            else:
                logger.warning("Unknown classifier: %s", name)
        except Exception:
            logger.warning("Could not instantiate classifier: %s", name, exc_info=True)

    return classifiers


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="shift-detection-monitor",
        description="Shift Detection Monitor for safety classifiers",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run
    run_parser = subparsers.add_parser("run", help="Run full evaluation from config")
    run_parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to YAML config file",
    )
    run_parser.add_argument(
        "--output",
        "-o",
        default="results/results.jsonl",
        help="Path for JSONL output (default: results/results.jsonl)",
    )

    # build-dataset
    build_parser = subparsers.add_parser("build-dataset", help="Build shift dataset")
    build_parser.add_argument(
        "--shift-condition",
        required=True,
        choices=[
            "paraphrase",
            "code-switch",
            "adversarial-suffix",
            "compositional-long-context",
            "temporal",
        ],
        help="Shift condition to generate",
    )
    build_parser.add_argument(
        "--source",
        required=True,
        help="Path to source JSONL file",
    )
    build_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path for output JSONL file",
    )
    build_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    # validate-config
    validate_parser = subparsers.add_parser(
        "validate-config", help="Validate a config file"
    )
    validate_parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to YAML config file",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "run":
        return _cmd_run(args)
    elif args.command == "build-dataset":
        return _cmd_build_dataset(args)
    elif args.command == "validate-config":
        return _cmd_validate_config(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
