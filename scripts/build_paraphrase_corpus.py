"""Build paraphrase/code-switch shift corpus incrementally via Bedrock.

Writes one line at a time, flushes after each. Supports --resume to pick up
where it left off after a timeout.

Usage:
    python scripts/build_paraphrase_corpus.py --n 500
    python scripts/build_paraphrase_corpus.py --n 500 --resume
    python scripts/build_paraphrase_corpus.py --shift code-switch --n 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shift_detection_monitor.stream.dataset_builder import (
    _CODE_SWITCH_SINGLISH_SYSTEM_PROMPT,
    _PARAPHRASE_SYSTEM_PROMPT,
    _bedrock_converse,
    _create_bedrock_client,
)


def main():
    parser = argparse.ArgumentParser(
        description="Build shift corpus via Bedrock (incremental, resumable).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=Path, default=Path("data/reference/source.jsonl"))
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path (default: data/shifted/<shift>/output.jsonl)")
    parser.add_argument("--shift", choices=["paraphrase", "code-switch"], default="paraphrase")
    parser.add_argument("--n", type=int, default=None, help="Limit to first N examples")
    parser.add_argument("--start-from", type=int, default=0, help="Skip first N examples explicitly")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output is None:
        args.output = Path(f"data/shifted/{args.shift}/output.jsonl")

    if not args.source.exists():
        print(f"ERROR: {args.source} not found. Run: python scripts/export_wildguardmix_source.py")
        sys.exit(1)

    # Read source
    with open(args.source) as f:
        all_examples = [json.loads(line) for line in f if line.strip()]

    if args.n:
        all_examples = all_examples[:args.n]

    # Determine skip count
    skip = args.start_from
    if args.resume and args.output.exists():
        skip = sum(1 for line in open(args.output) if line.strip())
        print(f"Resuming: {skip} already done")

    examples = all_examples[skip:]
    if not examples:
        print(f"Nothing to do — {skip}/{len(all_examples)} already processed.")
        sys.exit(0)

    # Select system prompt
    system_prompt = _PARAPHRASE_SYSTEM_PROMPT if args.shift == "paraphrase" else _CODE_SWITCH_SINGLISH_SYSTEM_PROMPT

    print(f"Building {args.shift} corpus: {len(examples)} remaining (skip={skip}, total={len(all_examples)})")
    print(f"Output: {args.output}")

    # Connect to Bedrock
    client = _create_bedrock_client("icpo-assistant")

    # Write incrementally
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (args.resume or args.start_from) else "w"
    done = 0

    with open(args.output, mode) as out:
        for i, ex in enumerate(examples):
            text = ex.get("text", "")
            try:
                shifted_text = _bedrock_converse(client, system_prompt, text)
            except Exception as e:
                print(f"\n  ERROR at example {skip + i}: {e}")
                print(f"  Processed {done} this run, {skip + done} total. Use --resume to continue.")
                sys.exit(1)

            record = {
                "original": text,
                "shifted": shifted_text,
                "text": shifted_text,
                "shift_condition": args.shift,
                "seed": args.seed,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            done += 1

            if done % 20 == 0:
                print(f"  {skip + done}/{len(all_examples)} done")
                time.sleep(1.0)  # Rate limit between batches

    print(f"\nDone! {done} new examples. Total: {skip + done}/{len(all_examples)}")


if __name__ == "__main__":
    main()
