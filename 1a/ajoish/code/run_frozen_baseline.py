#!/usr/bin/env python3
"""Run the frozen SmolLM2 architecture audit, prompts, and base perplexity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cisco_cpt.baseline import run_frozen_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repository_root = config_path.parents[3]
    result = run_frozen_baseline(config, repository_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()