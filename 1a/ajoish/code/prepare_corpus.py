#!/usr/bin/env python3
"""Build the independent cleaned Cisco corpus from immutable PDF inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cisco_cpt.corpus import build_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repository_root = config_path.parents[3]
    corpus_roots = {
        name: (repository_root / relative_path).resolve()
        for name, relative_path in config["corpus_roots"].items()
    }
    output_root = (repository_root / config["output_root"]).resolve()
    report = build_corpus(
        corpus_roots,
        output_root,
        seed=int(config["seed"]),
        eval_fraction=float(config["eval_fraction"]),
        expected_document_count=int(config["expected_document_count"]),
        page_timeout_seconds=int(config["page_timeout_seconds"]),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
