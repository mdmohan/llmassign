"""Randomly split the hand-authored instruction dataset into 80%/20% JSONL files."""

import argparse
import json
import random
from pathlib import Path


def split_instruction_dataset(input_file, output_dir, seed=7):
    records = json.loads(Path(input_file).read_text(encoding="utf-8"))
    random.Random(seed).shuffle(records)

    split_at = int(len(records) * 0.80)
    train_records = records[:split_at]
    evaluation_records = records[split_at:]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, rows in (("train.jsonl", train_records), ("evaluation.jsonl", evaluation_records)):
        with (output_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Seed: {seed}")
    print(f"Total examples: {len(records)}")
    print(f"Training examples: {len(train_records)}")
    print(f"Evaluation examples: {len(evaluation_records)}")


if __name__ == "__main__":
    submission_dir = Path(__file__).resolve().parent
    default_data_dir = submission_dir / "lora-instructions"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=Path, default=default_data_dir / "instruction_dataset.json")
    parser.add_argument("--output-dir", type=Path, default=default_data_dir)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    split_instruction_dataset(args.input_file, args.output_dir, args.seed)
