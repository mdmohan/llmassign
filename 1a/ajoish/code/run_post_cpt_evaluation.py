#!/usr/bin/env python3
"""Evaluate the final CPT model against the frozen baseline contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cisco_cpt.corpus import sha256_file
from cisco_cpt.evaluation import run_post_cpt_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repository_root = config_path.parents[3]
    result = run_post_cpt_evaluation(
        config=config,
        repository_root=repository_root,
        config_sha256=sha256_file(config_path),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
