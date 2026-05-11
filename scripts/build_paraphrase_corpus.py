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
    parser.add_argument("--resume", action="store_true", help="Skip already-processed examples")
    args = parser.parse_args()

    if not args.source.exists():
        print(f"ERROR: Source file not found: {args.source}")
        print("Generate it first:")
        print("  python scripts/export_wildguardmix_source.py")
        sys.exit(1)

    # Determine how many to skip if resuming
    skip = 0
    if args.resume and args.output.exists():
        skip = sum(1 for line in open(args.output) if line.strip())
        print(f"Resuming: {skip} examples already done")

    # If --n is set, create a truncated source (accounting for skip)
    all_lines = args.source.read_text().strip().split("\n")
    target_n = args.n or len(all_lines)
    remaining_lines = all_lines[skip:target_n]

    if not remaining_lines:
        print(f"Nothing to do — {skip} examples already processed.")
        sys.exit(0)

    tmp_source = args.source.with_suffix(".tmp.jsonl")
    tmp_source.write_text("\n".join(remaining_lines) + "\n")

    print(f"Building paraphrase corpus: {len(remaining_lines)} examples (skipped {skip})")
    print(f"Output: {args.output}")

    builder = ShiftDatasetBuilder(use_bedrock=True, bedrock_profile="icpo-assistant")
    tmp_output = args.output.with_suffix(".new.jsonl")
    manifest = builder.build("paraphrase", tmp_source, tmp_output, seed=args.seed + skip)

    # Append to existing output
    with open(args.output, "a") as out:
        out.write(tmp_output.read_text())
    tmp_output.unlink()
    tmp_output.with_suffix(".manifest.json").unlink(missing_ok=True)
    tmp_source.unlink()

    total = sum(1 for line in open(args.output) if line.strip())
    print(f"\nDone! {manifest.n_examples} new + {skip} existing = {total} total.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
