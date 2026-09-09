"""Create Murali-compatible and cross-run CPT evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cisco_cpt.comparable import prepare_comparable_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    repository_root = args.repository_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else repository_root / "1a" / "ajoish" / "evaluation" / "comparable"
    )
    result = prepare_comparable_evaluation(repository_root, output_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()