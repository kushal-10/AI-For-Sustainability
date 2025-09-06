#!/usr/bin/env python3
from pathlib import Path
from openai import OpenAI

OUT_DIR = Path("data/batch_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def save_outputs(batch_ids):
    client = OpenAI()
    for bid in batch_ids:
        b = client.batches.retrieve(bid)
        out_id = getattr(b, "output_file_id", None)
        if not out_id:
            print(f"{bid}: no output_file_id")
            continue
        # download
        content = client.files.content(out_id).read().decode("utf-8", errors="replace")
        out_path = OUT_DIR / f"{bid}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"{bid}: saved {out_path} ({len(content.splitlines())} lines)")

if __name__ == "__main__":
    batch_ids = [
        "batch_68bb1133f91c8190a4ffce61a7fa8cb8",
        "batch_68bb113b58e48190bf0f3a29f3317c68",
        "batch_68bb11416f648190b5575dc5a1c0adfe",
        "batch_68bb11471cd88190b29eebf5c1ace8ba",
        "batch_68bb114dac3c8190a2750e636ffb50f6",
        "batch_68bb11551a14819098bced1c704f5fab",
        "batch_68bb115809b881908ded21919ce8b479",
    ]
    save_outputs(batch_ids)
