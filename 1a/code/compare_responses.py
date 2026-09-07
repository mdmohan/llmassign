"""Score response JSON files and optionally compare two evaluation folders.

The scoring is intentionally deterministic and dependency-free. Concept recall
uses normalized phrase matching, while source-continuation prompts also report
exact-prefix, longest-common-prefix, and expected-token coverage metrics.
These lexical metrics complement, but do not replace, semantic human review.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
DEFAULT_QUERY_DIR = CODE_DIR.parent / "evaluation" / "queries"


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").casefold())


def _normalized(text: str) -> str:
    return " ".join(_words(text))


def _concept_matches(concept: str, response: str) -> bool:
    """Match a concept phrase, allowing explicit 'x or y' alternatives."""
    normalized_response = f" {_normalized(response)} "
    alternatives = re.split(r"\s+or\s+", concept, flags=re.IGNORECASE)
    normalized_alternatives = [_normalized(value) for value in alternatives]
    for alternative in normalized_alternatives:
        if not alternative:
            continue
        if f" {alternative} " in normalized_response:
            return True
        # Tokenizers often decode compact notation as spaced symbols (H 2 O).
        compact = alternative.replace(" ", "")
        if len(compact) <= 5 and compact in normalized_response.replace(" ", ""):
            return True
    return False


def _longest_common_prefix_ratio(response: str, expected: str) -> float:
    response_words = _words(response)
    expected_words = _words(expected)
    if not expected_words:
        return 0.0
    matched = 0
    for actual, reference in zip(response_words, expected_words):
        if actual != reference:
            break
        matched += 1
    return matched / len(expected_words)


def _expected_token_coverage(response: str, expected: str) -> float:
    expected_counts = Counter(_words(expected))
    if not expected_counts:
        return 0.0
    response_counts = Counter(_words(response))
    overlap = sum(
        min(count, response_counts[token])
        for token, count in expected_counts.items()
    )
    return overlap / sum(expected_counts.values())


def _repetition_ratio(response: str, ngram_size: int = 3) -> float:
    words = _words(response)
    ngrams = [tuple(words[i : i + ngram_size]) for i in range(len(words) - ngram_size + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def _query_metadata(query_dir: Path) -> dict[str, dict[str, dict]]:
    metadata: dict[str, dict[str, dict]] = {}
    if not query_dir.is_dir():
        raise ValueError(f"Query directory does not exist: {query_dir}")

    for path in sorted(query_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        queries = data.get("queries")
        if not isinstance(queries, list):
            continue
        metadata[path.name] = {
            str(query.get("id")): query
            for query in queries
            if query.get("id") is not None
        }
    return metadata


def _query_filename(response_data: dict, response_path: Path, metadata: dict) -> str:
    supplied_source = response_data.get("query_source")
    if supplied_source:
        source_name = Path(str(supplied_source)).name
        if source_name in metadata:
            return source_name

    matching_names = [name for name in metadata if Path(name).stem in response_path.stem]
    if len(matching_names) == 1:
        return matching_names[0]
    if not matching_names:
        raise ValueError(f"Cannot identify query source for {response_path}")
    raise ValueError(f"Ambiguous query source for {response_path}: {matching_names}")


def _set_label(query_filename: str) -> str:
    labels = {
        "domain_baseline_queries.json": "domain",
        "domain_exact_generation_queries.json": "exact",
        "domain_generalization_queries.json": "generalize",
        "generic_forgetting_queries.json": "general",
    }
    return labels.get(
        query_filename,
        Path(query_filename).stem.replace("_queries", ""),
    )


def score_record(record: dict) -> dict:
    response = str(record.get("response") or "")
    concepts = record.get("expected_concepts") or []
    concept_hits = [
        concept for concept in concepts if _concept_matches(str(concept), response)
    ]
    expected = str(record.get("expected_continuation") or "")
    exact_prefix = bool(expected) and _normalized(response).startswith(_normalized(expected))

    return {
        **record,
        "concept_hits": concept_hits,
        "concept_hit_count": len(concept_hits),
        "concept_count": len(concepts),
        "concept_recall": len(concept_hits) / len(concepts) if concepts else None,
        "exact_continuation_prefix": exact_prefix if expected else None,
        "continuation_lcp": (
            _longest_common_prefix_ratio(response, expected) if expected else None
        ),
        "continuation_coverage": (
            _expected_token_coverage(response, expected) if expected else None
        ),
        "repetition_ratio": _repetition_ratio(response),
    }


def load_response_folder(folder: Path, query_dir: Path) -> tuple[dict, dict]:
    folder = folder.expanduser().resolve()
    query_dir = query_dir.expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"Response directory does not exist: {folder}")

    response_paths = sorted(folder.glob("*.json"))
    if not response_paths:
        raise ValueError(f"Response directory contains no JSON files: {folder}")

    metadata = _query_metadata(query_dir)
    scored: dict[tuple[str, str], dict] = {}
    provenance = {}
    for response_path in response_paths:
        data = json.loads(response_path.read_text(encoding="utf-8"))
        query_filename = _query_filename(data, response_path, metadata)
        query_records = metadata.get(query_filename, {})
        results = data.get("results")
        if not isinstance(results, list):
            raise ValueError(f"'results' must be a list in {response_path}")

        provenance[response_path.name] = {
            "model_name": data.get("model_name"),
            "evaluation_stage": data.get("evaluation_stage"),
            "query_file": query_filename,
            "responses": len(results),
        }
        for result in results:
            query_id = str(result.get("id"))
            enriched = {**query_records.get(query_id, {}), **result}
            enriched["query_set"] = _set_label(query_filename)
            enriched["query_file"] = query_filename
            enriched["response_file"] = response_path.name
            key = (query_filename, query_id)
            if key in scored:
                raise ValueError(f"Duplicate response for {query_filename}:{query_id}")
            scored[key] = score_record(enriched)
    return scored, provenance


def _percent(value) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def _clip(value, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "…"


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(str(value)))

    def format_row(row) -> str:
        return " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row))

    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))


def _summary_rows(scored: dict) -> list[list[str]]:
    groups = defaultdict(list)
    for record in scored.values():
        groups[record["query_set"]].append(record)

    rows = []
    for name in sorted(groups):
        records = groups[name]
        concept_values = [
            record["concept_recall"]
            for record in records
            if record["concept_recall"] is not None
        ]
        exact_records = [
            record
            for record in records
            if record["exact_continuation_prefix"] is not None
        ]
        rows.append(
            [
                name,
                str(len(records)),
                _percent(sum(concept_values) / len(concept_values)) if concept_values else "-",
                (
                    f"{sum(record['exact_continuation_prefix'] for record in exact_records)}"
                    f"/{len(exact_records)}"
                    if exact_records
                    else "-"
                ),
                (
                    _percent(sum(record["continuation_lcp"] for record in exact_records) / len(exact_records))
                    if exact_records
                    else "-"
                ),
                (
                    _percent(sum(record["continuation_coverage"] for record in exact_records) / len(exact_records))
                    if exact_records
                    else "-"
                ),
                _percent(sum(record["repetition_ratio"] for record in records) / len(records)),
            ]
        )
    return rows


def print_single_folder(scored: dict, provenance: dict, preview_chars: int) -> None:
    models = sorted({str(value.get("model_name")) for value in provenance.values()})
    stages = sorted({str(value.get("evaluation_stage")) for value in provenance.values()})
    print(f"Model(s): {', '.join(models)}")
    print(f"Stage(s): {', '.join(stages)}")
    print(f"Response files: {len(provenance)} | Queries: {len(scored)}\n")

    rows = []
    for record in scored.values():
        exact = record["exact_continuation_prefix"]
        rows.append(
            [
                record["query_set"],
                str(record.get("id", "")),
                str(record.get("category", "")),
                f"{record['concept_hit_count']}/{record['concept_count']}",
                "yes" if exact else ("no" if exact is not None else "-"),
                _percent(record["continuation_lcp"]),
                _percent(record["continuation_coverage"]),
                _percent(record["repetition_ratio"]),
                _clip(record.get("response"), preview_chars),
            ]
        )
    _print_table(
        ["Set", "ID", "Category", "Concepts", "Exact", "LCP", "Coverage", "Repeat", "Response preview"],
        rows,
    )

    print("\nSummary")
    _print_table(
        ["Set", "N", "Avg concepts", "Exact", "Avg LCP", "Avg coverage", "Avg repeat"],
        _summary_rows(scored),
    )


def _comparison_values(before: dict, after: dict) -> dict:
    before_concepts = before["concept_recall"] or 0.0
    after_concepts = after["concept_recall"] or 0.0
    before_lcp = before["continuation_lcp"]
    after_lcp = after["continuation_lcp"]
    return {
        "concept_delta": after_concepts - before_concepts,
        "hit_delta": after["concept_hit_count"] - before["concept_hit_count"],
        "lcp_delta": (
            after_lcp - before_lcp
            if before_lcp is not None and after_lcp is not None
            else 0.0
        ),
        "repeat_delta": after["repetition_ratio"] - before["repetition_ratio"],
    }


def _verdict(changes: dict) -> str:
    if changes["concept_delta"] > 0:
        return "Knowledge improved"
    if changes["concept_delta"] < 0:
        return "Possible forgetting"
    if changes["lcp_delta"] > 0:
        return "Continuation improved"
    if changes["lcp_delta"] < 0:
        return "Continuation regressed"
    if changes["repeat_delta"] <= -0.10:
        return "Less repetitive"
    if changes["repeat_delta"] >= 0.10:
        return "More repetitive"
    return "No clear change"


def _repeat_change(before: dict, after: dict) -> str:
    delta = after["repetition_ratio"] - before["repetition_ratio"]
    if abs(delta) < 0.005:
        return f"{_percent(before['repetition_ratio'])} → {_percent(after['repetition_ratio'])}"
    direction = "better" if delta < 0 else "worse"
    return (
        f"{_percent(before['repetition_ratio'])} → "
        f"{_percent(after['repetition_ratio'])} ({direction})"
    )


def _change_note(changes: dict) -> str:
    notes = []
    if changes["hit_delta"] > 0:
        suffix = "concept" if changes["hit_delta"] == 1 else "concepts"
        notes.append(f"+{changes['hit_delta']} expected {suffix}")
    elif changes["hit_delta"] < 0:
        suffix = "concept" if changes["hit_delta"] == -1 else "concepts"
        notes.append(f"{changes['hit_delta']} expected {suffix}")
    if changes["lcp_delta"] > 0.0001:
        notes.append(f"exact prefix +{100 * changes['lcp_delta']:.1f} pp")
    elif changes["lcp_delta"] < -0.0001:
        notes.append(f"exact prefix {100 * changes['lcp_delta']:.1f} pp")
    if changes["repeat_delta"] <= -0.10:
        notes.append("less repetitive")
    elif changes["repeat_delta"] >= 0.10:
        notes.append("more repetitive")
    return "; ".join(notes) if notes else "no measured change"


def _print_comparison_summary(baseline: dict, candidate: dict) -> None:
    groups = defaultdict(list)
    for key, before in baseline.items():
        groups[before["query_set"]].append((before, candidate[key]))

    rows = []
    for name in sorted(groups):
        pairs = groups[name]
        base_concepts = sum((before["concept_recall"] or 0.0) for before, _ in pairs) / len(pairs)
        new_concepts = sum((after["concept_recall"] or 0.0) for _, after in pairs) / len(pairs)
        gains = 0
        losses = 0
        for before, after in pairs:
            changes = _comparison_values(before, after)
            if changes["concept_delta"] > 0 or changes["lcp_delta"] > 0:
                gains += 1
            if changes["concept_delta"] < 0 or changes["lcp_delta"] < 0:
                losses += 1

        exact_pairs = [
            (before, after)
            for before, after in pairs
            if before["continuation_lcp"] is not None
        ]
        if exact_pairs:
            base_lcp = sum(before["continuation_lcp"] for before, _ in exact_pairs) / len(exact_pairs)
            new_lcp = sum(after["continuation_lcp"] for _, after in exact_pairs) / len(exact_pairs)
            lcp_display = f"{_percent(base_lcp)} → {_percent(new_lcp)}"
        else:
            lcp_display = "-"

        base_repeat = sum(before["repetition_ratio"] for before, _ in pairs) / len(pairs)
        new_repeat = sum(after["repetition_ratio"] for _, after in pairs) / len(pairs)
        repeat_direction = "better" if new_repeat < base_repeat else "worse"
        if abs(new_repeat - base_repeat) < 0.005:
            repeat_direction = "same"
        rows.append(
            [
                name,
                str(len(pairs)),
                f"{_percent(base_concepts)} → {_percent(new_concepts)}",
                str(gains),
                str(losses),
                lcp_display,
                f"{_percent(base_repeat)} → {_percent(new_repeat)} ({repeat_direction})",
            ]
        )

    print("Executive summary")
    _print_table(
        ["Set", "Queries", "Expected concepts", "Improved", "Regressed", "Exact prefix", "Repetition (lower is better)"],
        rows,
    )


def print_comparison(
    baseline: dict,
    candidate: dict,
    preview_chars: int,
    show_all: bool = False,
    show_responses: bool = False,
) -> None:
    missing = sorted(set(baseline) - set(candidate))
    extra = sorted(set(candidate) - set(baseline))
    if missing or extra:
        raise ValueError(
            f"Response sets do not align; missing={len(missing)}, extra={len(extra)}"
        )

    print(
        "Expected concepts measures literal reference phrases found in each response.\n"
        "Exact prefix applies only to source-continuation prompts.\n"
        "For repetition, lower is better.\n"
    )
    _print_comparison_summary(baseline, candidate)

    changed_pairs = []
    all_rows = []
    side_by_side_rows = []
    for key, before in baseline.items():
        after = candidate[key]
        changes = _comparison_values(before, after)
        concept_display = (
            f"{before['concept_hit_count']}/{before['concept_count']} → "
            f"{after['concept_hit_count']}/{after['concept_count']}"
        )
        lcp_display = (
            f"{_percent(before['continuation_lcp'])} → {_percent(after['continuation_lcp'])}"
            if before["continuation_lcp"] is not None
            else "-"
        )
        row = [
            before["query_set"],
            str(before.get("id", "")),
            _verdict(changes),
            concept_display,
            lcp_display,
            _repeat_change(before, after),
        ]
        all_rows.append(row)
        if changes["hit_delta"] != 0 or abs(changes["lcp_delta"]) > 0.0001:
            changed_pairs.append((before, after))

        expected = "; ".join(
            str(value) for value in before.get("expected_concepts", [])
        )
        side_by_side_rows.append(
            [
                before["query_set"],
                str(before.get("id", "")),
                _clip(expected, 42),
                (
                    f"[{before['concept_hit_count']}/{before['concept_count']}] "
                    f"{_clip(before.get('response'), preview_chars)}"
                ),
                (
                    f"[{after['concept_hit_count']}/{after['concept_count']}] "
                    f"{_clip(after.get('response'), preview_chars)}"
                ),
                _change_note(changes),
            ]
        )

    print("\nSide-by-side responses")
    print("The [x/y] prefix means x of y expected concepts were found.")
    _print_table(
        ["Set", "ID", "Expected concepts", "Baseline response", "Post-CPT response", "What changed"],
        side_by_side_rows,
    )

    repetition_changes = sorted(
        (
            (
                abs(after["repetition_ratio"] - before["repetition_ratio"]),
                before,
                after,
            )
            for key, before in baseline.items()
            for after in [candidate[key]]
        ),
        key=lambda item: item[0],
        reverse=True,
    )[:6]
    print("\nLargest repetition changes")
    _print_table(
        ["Set", "ID", "Interpretation", "Repetition"],
        [
            [
                before["query_set"],
                str(before.get("id", "")),
                (
                    "No repetition change"
                    if abs(after["repetition_ratio"] - before["repetition_ratio"]) < 0.005
                    else (
                        "Less repetitive"
                        if after["repetition_ratio"] < before["repetition_ratio"]
                        else "More repetitive"
                    )
                ),
                _repeat_change(before, after),
            ]
            for _, before, after in repetition_changes
        ],
    )

    if show_all:
        print("\nAll queries")
        _print_table(
            ["Set", "ID", "Interpretation", "Concept hits", "Exact prefix", "Repetition"],
            all_rows,
        )

    if show_responses and changed_pairs:
        print("\nResponses for knowledge or continuation changes")
        for before, after in changed_pairs:
            concepts = ", ".join(str(value) for value in before.get("expected_concepts", []))
            print(f"\n[{before['query_set']} / {before.get('id')}] {_verdict(_comparison_values(before, after))}")
            print(f"Expected: {concepts}")
            print(f"Baseline:  {_clip(before.get('response'), preview_chars)}")
            print(f"New model: {_clip(after.get('response'), preview_chars)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score one response folder or compare baseline and candidate folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("baseline_dir", type=Path, help="Baseline response JSON directory.")
    parser.add_argument(
        "--compare-dir",
        type=Path,
        default=None,
        help="Optional post-CPT response directory for side-by-side comparison.",
    )
    parser.add_argument(
        "--query-dir",
        type=Path,
        default=DEFAULT_QUERY_DIR,
        help="Original containing source query JSON files used to enrich old results.",
    )
    parser.add_argument(
        "--response-preview-chars",
        type=int,
        default=72,
        help="Maximum response-preview characters when responses are displayed.",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Include a compact row for every query in comparison mode.",
    )
    parser.add_argument(
        "--show-responses",
        action="store_true",
        help="Show before/after response previews for measured knowledge changes.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.response_preview_chars < 10:
        parser.error("--response-preview-chars must be at least 10")
    try:
        baseline, provenance = load_response_folder(args.baseline_dir, args.query_dir)
        if args.compare_dir is None:
            print_single_folder(baseline, provenance, args.response_preview_chars)
        else:
            candidate, _ = load_response_folder(args.compare_dir, args.query_dir)
            print_comparison(
                baseline,
                candidate,
                preview_chars=args.response_preview_chars,
                show_all=args.show_all,
                show_responses=args.show_responses,
            )
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
