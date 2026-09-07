#!/usr/bin/env python3
"""Tokenize and pack the accepted Cisco corpus for continual pre-training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cisco_cpt.tokenization import build_packed_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repository_root = config_path.parents[3]

    def resolve(relative_path: str) -> Path:
        return (repository_root / relative_path).resolve()

    report = build_packed_dataset(
        manifest_path=resolve(config["manifest_path"]),
        document_root=resolve(config["document_root"]),
        packed_root=resolve(config["packed_root"]),
        reports_root=resolve(config["reports_root"]),
        model_id=str(config["model_id"]),
        revision=str(config["revision"]),
        cache_dir=resolve(config["cache_dir"]),
        sequence_length=int(config["sequence_length"]),
        parquet_batch_size=int(config["parquet_batch_size"]),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()