#!/usr/bin/env python3
"""Create the frozen general and corpus-grounded Cisco evaluation prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cisco_cpt.prompts import build_frozen_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_frozen_prompts(
        args.manifest.resolve(),
        args.extracted_root.resolve(),
        args.output.resolve(),
        args.report.resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()