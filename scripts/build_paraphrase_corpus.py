"""Build paraphrase shift corpus from a source JSONL file using Bedrock.

Usage:
    # First, generate source.jsonl on the Mac Studio (which has WildGuardMix cached):
    #   python scripts/export_wildguardmix_source.py
    #
    # Then run this on the MacBook (which has AWS SSO):
    #   python scripts/build_paraphrase_corpus.py

Requires: aws sso login --profile icpo-assistant
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shift_detection_monitor.stream.dataset_builder import ShiftDatasetBuilder


def main():
    parser = argparse.ArgumentParser(
        description="Build paraphrase shift corpus via Bedrock Claude.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=Path, default=Path("data/reference/source.jsonl"),
                        help="Source JSONL file with 'text' field")
    parser.add_argument("--output", type=Path, default=Path("data/shifted/paraphrase/output.jsonl"),
                        help="Output path for shifted corpus")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=None, help="Limit to first N examples (default: all)")
    args = parser.parse_args()

    if not args.source.exists():
        print(f"ERROR: Source file not found: {args.source}")
        print("Generate it first:")
        print("  python scripts/export_wildguardmix_source.py")
        sys.exit(1)

    # If --n is set, create a truncated source
    if args.n:
        lines = args.source.read_text().strip().split("\n")[:args.n]
        tmp_source = args.source.with_suffix(".tmp.jsonl")
        tmp_source.write_text("\n".join(lines) + "\n")
        source = tmp_source
    else:
        source = args.source

    print(f"Building paraphrase corpus from {source} ({sum(1 for _ in open(source))} examples)")
    print(f"Output: {args.output}")

    builder = ShiftDatasetBuilder(use_bedrock=True, bedrock_profile="icpo-assistant")
    manifest = builder.build("paraphrase", source, args.output, seed=args.seed)

    print(f"\nDone! {manifest.n_examples} examples written.")
    print(f"Manifest: {args.output.with_suffix('.manifest.json')}")

    # Clean up temp file
    if args.n:
        tmp_source.unlink()


if __name__ == "__main__":
    main()
