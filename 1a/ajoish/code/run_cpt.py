#!/usr/bin/env python3
"""Run the gated SmolLM2 continual pre-training workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cisco_cpt.corpus import sha256_file
from cisco_cpt.training import run_cpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Use 'auto' for the latest checkpoint, or provide a repository-relative path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repository_root = config_path.parents[3]
    result = run_cpt(
        config=config,
        repository_root=repository_root,
        config_sha256=sha256_file(config_path),
        mode=args.mode,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()