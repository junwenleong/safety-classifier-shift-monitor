"""Export WildGuardMix prompts to source.jsonl for the paraphrase builder.

Run this on the Mac Studio (which has WildGuardMix cached from fine-tuning).
Then copy data/reference/source.jsonl to the MacBook for Bedrock processing.
"""

import json
from pathlib import Path

from datasets import load_dataset

def main():
    out = Path("data/reference/source.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    ds = ds.shuffle(seed=42)

    with open(out, "w") as f:
        for i in range(500):
            f.write(json.dumps({"text": ds[i]["prompt"]}) + "\n")

    print(f"Wrote 500 examples to {out}")

if __name__ == "__main__":
    main()
