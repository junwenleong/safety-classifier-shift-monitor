"""Build compositional long-context shift corpus via Bedrock.

Embeds harmful prompts within long innocuous context at varying positions.
Uses Claude to generate benign padding text.

Usage:
    python scripts/build_compositional_corpus.py --n 300
    python scripts/build_compositional_corpus.py --n 300 --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shift_detection_monitor.stream.dataset_builder import (
    _bedrock_converse,
    _create_bedrock_client,
)

_PADDING_SYSTEM_PROMPT = (
    "Generate a paragraph of benign, innocuous text about everyday topics "
    "(cooking, weather, travel, hobbies, office work). Write exactly {n_words} words. "
    "Do not include any harmful, unsafe, or controversial content. "
    "Return only the text, nothing else."
)

POSITIONS = ["beginning", "middle", "end"]


def main():
    parser = argparse.ArgumentParser(
        description="Build compositional long-context shift corpus via Bedrock.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=Path, default=Path("data/reference/source.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/shifted/compositional/output.jsonl"))
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.source.exists():
        print(f"ERROR: {args.source} not found.")
        sys.exit(1)

    with open(args.source) as f:
        all_examples = [json.loads(line) for line in f if line.strip()][:args.n]

    skip = 0
    if args.resume and args.output.exists():
        skip = sum(1 for line in open(args.output) if line.strip())
        print(f"Resuming: {skip} already done")

    examples = all_examples[skip:]
    if not examples:
        print(f"Nothing to do — {skip}/{len(all_examples)} already processed.")
        sys.exit(0)

    print(f"Building compositional corpus: {len(examples)} remaining (skip={skip}, total={len(all_examples)})")
    client = _create_bedrock_client("icpo-assistant")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    done = 0
    with open(args.output, "a" if skip else "w") as out:
        for i, ex in enumerate(examples):
            text = ex.get("text", "")
            position = POSITIONS[(skip + i) % 3]
            n_words = 350  # ~300-500 tokens

            try:
                # Generate padding
                prompt = _PADDING_SYSTEM_PROMPT.format(n_words=n_words)
                padding = _bedrock_converse(client, prompt, f"Topic seed: {(skip + i) % 7}")

                # Embed at position
                if position == "beginning":
                    shifted = f"{text}\n\n{padding}"
                elif position == "end":
                    shifted = f"{padding}\n\n{text}"
                else:  # middle
                    half = len(padding) // 2
                    shifted = f"{padding[:half]}\n\n{text}\n\n{padding[half:]}"

            except Exception as e:
                print(f"\n  ERROR at example {skip + i}: {e}")
                print(f"  Processed {done} this run, {skip + done} total. Use --resume to continue.")
                sys.exit(1)

            record = {
                "original": text,
                "shifted": shifted,
                "text": shifted,
                "position": position,
                "shift_condition": "compositional-long-context",
                "seed": args.seed,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            done += 1

            if done % 20 == 0:
                print(f"  {skip + done}/{len(all_examples)} done")
                time.sleep(1.0)

    print(f"\nDone! {done} new examples. Total: {skip + done}/{len(all_examples)}")


if __name__ == "__main__":
    main()
