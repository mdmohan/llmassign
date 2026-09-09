"""Create a grounded instruction-tuning dataset from cleaned networking text.

The script is intentionally extractive: every response is copied from a source
passage after whitespace normalization.  It does not ask an external model to
invent answers.  This makes the resulting JSONL suitable for a defensible QLoRA
experiment and lets the companion report verify source grounding.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from cli_parsers import build_instruction_dataset_parser


GENERATION_PROMPT = (
    "Read the text and generate instruction-response pairs in JSON format based "
    "ONLY on this text. Each entry must have instruction and response keys."
)

DOMAIN_TERMS = re.compile(
    r"\b(?:cisco|nexus|nx-os|network|switch|router|routing|ospf|bgp|vlan|vxlan|"
    r"evpn|interface|fabric|data center|hyperfabric|dashboard|aci|ethernet|ip|"
    r"route|packet|protocol|neighbor|adjacency|lsa|vrf|vpc|telemetry|gpu|roce)\b",
    re.IGNORECASE,
)

HEADING_PREFIXES = (
    "About ",
    "Configuring ",
    "Configure ",
    "Verifying ",
    "Verify ",
    "Guidelines and Limitations",
    "Guidelines for ",
    "Prerequisites for ",
    "Prerequisites",
    "Default Settings for ",
    "Default Settings",
    "Troubleshooting ",
    "Understanding ",
    "Using ",
    "Enabling ",
    "Disabling ",
    "Creating ",
    "Managing ",
    "Monitoring ",
    "Overview of ",
    "Benefits of ",
    "Features of ",
    "Deployment scenarios",
    "Product overview",
    "Protocol Security",
)

GENERIC_HEADINGS = {
    "about",
    "abstract",
    "before you begin",
    "chapter",
    "configure",
    "configuration example",
    "default settings",
    "document conventions",
    "example",
    "features and benefits",
    "guidelines and limitations",
    "introduction",
    "is-is",
    "local area",
    "name",
    "overview",
    "ospf",
    "ospfv2",
    "ospfv3",
    "prerequisites",
    "procedure",
    "product overview",
    "related information",
    "summary",
    "verify",
    "wide area",
    "wireless",
}

BOILERPLATE = re.compile(
    r"(?:all rights reserved|copyright notice|full copyright statement|"
    r"table of contents|document history|printed in usa|cisco capital|"
    r"trademark|disclaims all warranties|feedback on this page|"
    r"this chapter contains the following sections|where to go next|"
    r"obtaining documentation|technical support)",
    re.IGNORECASE,
)

NON_CONTENT_HEADING = re.compile(
    r"(?:references|iana considerations|acknowledg(?:e)?ments|authors?' addresses|"
    r"contributors|status of this memo|requirements language|requirements notation|"
    r"conventions used in this document)",
    re.IGNORECASE,
)

HEADING_FRAGMENT = re.compile(
    r"(?:\bperform the following\b|\bthe following (?:example|topology)\b|"
    r"\bis considered invalid\b|\bmay cause unexpected\b|"
    r"\bto display .+ enter (?:any|one)\b|\binstallation guide, release\b)",
    re.IGNORECASE,
)

OUT_OF_SCOPE_HEADING = re.compile(
    r"\b(?:wi-fi|wireless protected setup)\b",
    re.IGNORECASE,
)

TOPIC_STOPWORDS = {
    "about", "advanced", "and", "basic", "chapter", "configure", "configuring",
    "create", "creating", "default", "disable", "disabling", "enable", "enabling",
    "example", "examples", "for", "from", "guidelines", "how", "limitations",
    "manage", "managing", "monitor", "monitoring", "of", "on", "overview", "settings",
    "the", "to", "troubleshoot", "troubleshooting", "understanding", "using", "verify",
    "verifying", "with", "work",
}

# High-value product and use-case passages are named explicitly so the dataset
# covers the chatbot's product-recommendation goal in addition to configuration.
# The extracted response still comes directly from the named source document.
CURATED_PASSAGES = (
    (
        "06_n9300_platform_data_sheet.txt",
        "What data-center deployments are Cisco N9300 platform switches designed for?",
        "Product overview Organizations everywhere recognize",
        "Models Table 1 summarizes",
    ),
    (
        "06_n9300_platform_data_sheet.txt",
        "What capacity and port configuration does the Cisco N9332PQ provide?",
        "The Cisco N9332PQ Switch is a 1-rack-unit",
        "Cisco N9332PQ Switch",
    ),
    (
        "06_n9300_platform_data_sheet.txt",
        "What capacity and port configurations do the Cisco N9372PX and N9372PX-E provide?",
        "The Cisco N9372PX and N9372PX-E Switches are",
        "Cisco N9372PX-E Switch",
    ),
    (
        "06_n9300_platform_data_sheet.txt",
        "What capacity and uplink configuration does the Cisco N9396PX provide?",
        "The Cisco N9396PX Switch is a 2RU switch",
        "Cisco N9396PX Switch",
    ),
    (
        "07_n9500_series_data_sheet.txt",
        "Which operating modes and data-center technologies do Cisco N9500 Series switches support?",
        "Product overview Application architectures and deployment modes",
        "Cisco N9000 Series Switch Chassis",
    ),
    (
        "07_n9500_series_data_sheet.txt",
        "What Ethernet interface scale can Cisco N9500 Series modular switches provide?",
        "The Cisco N9500 Series modular switches support a comprehensive selection",
        "The supervisor, system controller",
    ),
    (
        "07_n9500_series_data_sheet.txt",
        "Which deployment roles are supported by Cisco N9500 Series switches?",
        "The supervisor, system controller, power supplies",
        "| Feature | Benefit |",
    ),
    (
        "07_n9500_series_data_sheet.txt",
        "Why are Cisco N9500 Series switches suitable as spine switches?",
        "Spine-leaf fabric architecture The high port-density",
        "Spine-Leaf Architecture using Cisco N9300 and N9500 Switches",
    ),
    (
        "07_n9500_series_data_sheet.txt",
        "Why can Cisco N9500 Series switches serve as core, aggregation, or gateway switches?",
        "Core, Aggregation, and gateway roles The Cisco N9500 Series Switches support",
        "End-of-row access layer switch",
    ),
    (
        "14_data_center_networking_ai_ml_solution_overview.txt",
        "What does Cisco provide for high-bandwidth, lossless, low-latency AI/ML networks?",
        "AI/ML Networking with Cisco Nexus 9000",
        "Trends and challenges",
    ),
    (
        "14_data_center_networking_ai_ml_solution_overview.txt",
        "What network design does Cisco recommend for large GPU clusters?",
        "For large GPU clusters, a spine-leaf network",
        "• Handling network congestion:",
    ),
    (
        "14_data_center_networking_ai_ml_solution_overview.txt",
        "How can a Cisco Nexus switch design connect a smaller 64-GPU cluster?",
        "• Designing a network for AI: Smaller GPU",
        "Cisco Nexus Dashboard simplifies configuring",
    ),
    (
        "14_data_center_networking_ai_ml_solution_overview.txt",
        "How does Cisco Nexus Dashboard help detect and troubleshoot AI-fabric congestion?",
        "Cisco Nexus Dashboard simplifies configuring",
        "Use cases",
    ),
    (
        "10_nexus_dashboard_data_sheet_2026.txt",
        "What role does Cisco Nexus Dashboard provide for on-premises data centers?",
        "For organizations managing on-premises data centers, the Cisco Nexus Dashboard serves",
        "Cisco Nexus Dashboard helps to do the following:",
    ),
    (
        "10_nexus_dashboard_data_sheet_2026.txt",
        "How does Cisco Nexus Dashboard help configure data-center networks?",
        "● Configure: Leverage a centralized interface",
        "● Manage: See a unified view",
    ),
    (
        "10_nexus_dashboard_data_sheet_2026.txt",
        "How does Cisco Nexus Dashboard help manage Cisco ACI and NX-OS environments?",
        "● Manage: See a unified view",
        "● Analyze: minimize downtime",
    ),
    (
        "10_nexus_dashboard_data_sheet_2026.txt",
        "How does Cisco Nexus Dashboard use telemetry and analytics to analyze network problems?",
        "● Analyze: minimize downtime",
        "Cisco Nexus Dashboard: powering automation",
    ),
    (
        "17_nexus_hyperfabric_data_sheet.txt",
        "What is Cisco Nexus Hyperfabric and what operational problem does it solve?",
        "Product overview As the cloud-managed operating model for Cisco Nexus One",
        "Cisco Nexus Hyperfabric lifecycle",
    ),
    (
        "17_nexus_hyperfabric_data_sheet.txt",
        "How does the Nexus Hyperfabric design-to-deploy lifecycle work?",
        "Customers log in to Nexus Hyperfabric to begin building a validated fabric design",
        "Cisco Nexus Hyperfabric lifecycle",
    ),
    (
        "17_nexus_hyperfabric_data_sheet.txt",
        "What does the Nexus Hyperfabric cloud controller manage across the fabric lifecycle?",
        "Features and benefits Table 1. Feature and benefits Prominent feature Cloud-managed solution",
        "Intuitive for IT generalists",
    ),
    (
        "17_nexus_hyperfabric_data_sheet.txt",
        "Why is Cisco Nexus Hyperfabric suitable for IT generalists and application or DevOps teams?",
        "Intuitive for IT generalists and application and DevOps teams to operate",
        "High-performance fabrics",
    ),
    (
        "17_nexus_hyperfabric_data_sheet.txt",
        "What scale and port speeds can Cisco Nexus Hyperfabric support?",
        "High-performance fabrics Fabrics need high-performance bandwidth",
        "Vertical stack solution",
    ),
    (
        "17_nexus_hyperfabric_data_sheet.txt",
        "How does Nexus Hyperfabric monitor unmanaged Cisco NX-OS switches?",
        "Monitor unmanaged Cisco NX-OS switches Cisco Nexus Hyperfabric now includes",
        "Cisco Cloud Control",
    ),
    (
        "17_nexus_hyperfabric_data_sheet.txt",
        "What two approaches does Cisco Nexus Hyperfabric offer for AI infrastructure?",
        "AI solutions with Cisco Nexus Hyperfabric Cisco Nexus Hyperfabric delivers",
        "AI solutions with Cisco Nexus Hyperfabric Platform support",
    ),
    (
        "17_nexus_hyperfabric_data_sheet.txt",
        "When are Nexus Hyperfabric Essentials and Premier subscriptions used?",
        "Licensing A subscription entitlement is needed for every Nexus Hyperfabric switch",
        "Table 3. Entitlement feature tiers",
    ),
)

PREFERRED_EVALUATION_SOURCES = (
    "06_n9300_platform_data_sheet.txt",
    "07_n9500_series_data_sheet.txt",
    "14_data_center_networking_ai_ml_solution_overview.txt",
    "13_nexus_dashboard_telemetry_white_paper.txt",
)


@dataclass(frozen=True)
class Pair:
    instruction: str
    response: str
    source: str
    section: str
    method: str
    score: int


def compact_text(text: str) -> str:
    """Normalize whitespace without changing the words in the source passage."""
    return re.sub(r"\s+", " ", text).strip(" -\t\r\n")


def normalized_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_rank(seed: int, source: str) -> str:
    return sha256_text(f"{seed}:{source}")


def split_blocks(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]


def clean_heading(block: str) -> str:
    heading = compact_text(block)
    heading = re.sub(r"^(?:chapter\s+)?\d+(?:\.\d+)*\.?\s+", "", heading, flags=re.I)
    heading = re.sub(r"\s+Chapter\s+\d+\s*$", "", heading, flags=re.I)
    heading = re.sub(r"\s+Examples?\s*$", "", heading, flags=re.I)
    return heading.strip(" .:?!-")


def is_heading(block: str) -> bool:
    if "\n" in block.strip():
        return False
    raw = compact_text(block)
    if not 3 <= len(raw) <= 120 or len(raw.split()) > 17:
        return False
    if re.search(r"[|{}=]", raw) or BOILERPLATE.search(raw):
        return False
    heading = clean_heading(raw)
    if (
        not heading
        or heading.casefold() in GENERIC_HEADINGS
        or NON_CONTENT_HEADING.search(heading)
        or HEADING_FRAGMENT.search(heading)
        or OUT_OF_SCOPE_HEADING.search(heading)
    ):
        return False
    if heading.count("(") != heading.count(")") or heading.count('"') % 2:
        return False
    if re.search(r"\s\d+-\d+$", heading):
        return False
    if re.search(r"\b(?:and|for|from|in|of|on|the|to|with)\s*$", heading, re.I):
        return False
    if raw.startswith(HEADING_PREFIXES):
        return True
    # RFC sections commonly use numbered headings such as "2.1. Packet format".
    return bool(re.match(r"^\d+(?:\.\d+)+\.?\s+[A-Z]", raw))


def usable_response(block: str) -> bool:
    answer = compact_text(block)
    if not 180 <= len(answer) <= 2_400:
        return False
    if BOILERPLATE.search(answer) or answer.count("|") > 8 or "(cid:" in answer:
        return False
    if len(re.findall(r"https?://|www\.", answer, flags=re.I)) > 1:
        return False
    letters = sum(character.isalpha() for character in answer)
    if letters / max(len(answer), 1) < 0.55:
        return False
    if not DOMAIN_TERMS.search(answer):
        return False
    first_alpha = next((character for character in answer if character.isalpha()), "")
    if first_alpha and first_alpha.islower():
        return False
    if re.search(r"\b(?:a|an|and|for|in|of|or|the|to|with)\s*$", answer, re.I):
        return False
    if answer.endswith(":"):
        return False
    return True


def heading_supported_by_response(heading: str, response: str) -> bool:
    """Reject likely PDF section-boundary mistakes using lexical evidence."""
    topic_words = {
        word.casefold()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", clean_heading(heading))
        if word.casefold() not in TOPIC_STOPWORDS
    }
    if not topic_words:
        return True
    answer_words = {
        word.casefold() for word in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", response)
    }
    overlap = topic_words & answer_words
    required_overlap = 2 if len(topic_words) >= 2 else 1
    return len(overlap) >= required_overlap


def make_instruction(heading: str, source: str) -> str:
    title = clean_heading(heading)
    lower = title.casefold()
    replacements = (
        ("about ", "What is {topic}?"),
        ("configuring ", "How do you configure {topic}?"),
        ("configure ", "How do you configure {topic}?"),
        ("verifying ", "How do you verify {topic}?"),
        ("verify ", "How do you verify {topic}?"),
        ("troubleshooting ", "How do you troubleshoot {topic}?"),
        ("understanding ", "How does {topic} work?"),
        ("using ", "How is {topic} used?"),
        ("enabling ", "How do you enable {topic}?"),
        ("disabling ", "How do you disable {topic}?"),
        ("creating ", "How do you create {topic}?"),
        ("managing ", "How do you manage {topic}?"),
        ("monitoring ", "How do you monitor {topic}?"),
        ("overview of ", "What is {topic}?"),
        ("benefits of ", "What are the benefits of {topic}?"),
        ("features of ", "What are the features of {topic}?"),
    )
    for prefix, template in replacements:
        if lower.startswith(prefix):
            topic = title[len(prefix) :].strip(" .:-")
            if template == "What is {topic}?" and re.search(r"(?:files|interfaces|routes|settings)$", topic, re.I):
                return f"What are {topic}?"
            return template.format(topic=topic)

    if lower.startswith("guidelines and limitations"):
        topic = re.sub(r"^guidelines and limitations(?:\s+for)?\s*", "", title, flags=re.I)
        return f"What guidelines and limitations apply to {topic}?"
    if lower.startswith("guidelines for "):
        return f"What guidelines apply to {title[15:].strip()}?"
    if lower.startswith("prerequisites for "):
        return f"What are the prerequisites for {title[18:].strip()}?"
    if lower.startswith("default settings for "):
        return f"What are the default settings for {title[21:].strip()}?"
    if lower == "protocol security":
        rfc = Path(source).stem.upper()
        return f"What does {rfc} say about protocol security?"
    if Path(source).stem.casefold().startswith("rfc"):
        return f"According to {Path(source).stem.upper()}, what is {title}?"
    return f"Explain {title}."


def candidate_score(heading: str, answer: str) -> int:
    score = 0
    length = len(answer)
    if 280 <= length <= 1_400:
        score += 3
    elif length <= 1_800:
        score += 1
    if re.search(r"\b(?:must|should|use|configure|command|supports|provides|enables)\b", answer, re.I):
        score += 2
    if re.search(r"\b(?:Cisco|Nexus|NX-OS|OSPF|BGP|VXLAN|EVPN|Hyperfabric)\b", answer, re.I):
        score += 2
    if heading.casefold().startswith(("about ", "understanding ", "configuring ", "troubleshooting ")):
        score += 2
    if re.search(r"\b(?:page|figure|table)\s+\d+", answer, re.I):
        score -= 2
    return score


def extract_heading_pairs(text: str, source: str) -> list[Pair]:
    blocks = split_blocks(text)
    pairs: list[Pair] = []
    for index, block in enumerate(blocks[:-1]):
        if not is_heading(block):
            continue
        heading = clean_heading(block)
        for answer_block in blocks[index + 1 : index + 4]:
            if is_heading(answer_block):
                # Only skip a duplicated heading. Crossing into a different
                # section can create a factually grounded but mismatched pair.
                if normalized_key(clean_heading(answer_block)) == normalized_key(heading):
                    continue
                break
            if not usable_response(answer_block):
                continue
            response = compact_text(answer_block)
            if not heading_supported_by_response(heading, response):
                continue
            instruction = make_instruction(block, source)
            if normalized_key(instruction) in {
                "what are the prerequisites for",
                "what are the default settings for",
                "what guidelines and limitations apply to",
            }:
                break
            pairs.append(
                Pair(
                    instruction=instruction,
                    response=response,
                    source=source,
                    section=heading,
                    method="heading_extraction",
                    score=candidate_score(heading, response),
                )
            )
            break
    return pairs


def extract_curated_pairs(source_texts: dict[str, str]) -> list[Pair]:
    pairs: list[Pair] = []
    for filename, instruction, start_marker, end_marker in CURATED_PASSAGES:
        text = source_texts.get(filename)
        if text is None:
            continue
        normalized = compact_text(text)
        start = normalized.find(start_marker)
        end = normalized.find(end_marker, start + len(start_marker)) if start >= 0 else -1
        if start < 0 or end <= start:
            continue
        response = normalized[start:end].strip(" -")
        if not 180 <= len(response) <= 2_800:
            continue
        pairs.append(
            Pair(
                instruction=instruction,
                response=response,
                source=filename,
                section=start_marker,
                method="curated_source_passage",
                score=100,
            )
        )
    return pairs


def deduplicate_candidates(pairs: Iterable[Pair]) -> list[Pair]:
    selected: list[Pair] = []
    instructions: set[str] = set()
    responses: set[str] = set()
    ordered = sorted(
        pairs,
        key=lambda pair: (-pair.score, pair.source.casefold(), pair.instruction.casefold()),
    )
    for pair in ordered:
        instruction_key = normalized_key(pair.instruction)
        response_key = normalized_key(pair.response)
        if instruction_key in instructions or response_key in responses:
            continue
        instructions.add(instruction_key)
        responses.add(response_key)
        selected.append(pair)
    return selected


def cap_by_source(pairs: Iterable[Pair], limit: int) -> dict[str, list[Pair]]:
    grouped: dict[str, list[Pair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.source].append(pair)
    for source, source_pairs in grouped.items():
        grouped[source] = sorted(
            source_pairs,
            key=lambda pair: (-pair.score, pair.instruction.casefold()),
        )[:limit]
    return dict(grouped)


def choose_evaluation_sources(
    grouped: dict[str, list[Pair]],
    evaluation_target: int,
    train_target: int,
    seed: int,
    preferred_sources: Iterable[str] = (),
) -> set[str]:
    preferred = [source for source in preferred_sources if source in grouped]
    preferred_set = set(preferred)
    ordered_sources = preferred + sorted(
        (source for source in grouped if source not in preferred_set),
        key=lambda source: source_rank(seed, source),
    )
    total = sum(len(grouped[source]) for source in ordered_sources)
    if total < evaluation_target + train_target:
        raise ValueError(
            f"Only {total} usable pairs remain, but {evaluation_target + train_target} were requested."
        )

    evaluation_sources: set[str] = set()
    evaluation_capacity = 0
    remaining_capacity = total
    minimum_sources = min(8, len(ordered_sources) - 1)
    for source in ordered_sources:
        source_count = len(grouped[source])
        if remaining_capacity - source_count < train_target:
            continue
        evaluation_sources.add(source)
        evaluation_capacity += source_count
        remaining_capacity -= source_count
        if evaluation_capacity >= evaluation_target and len(evaluation_sources) >= minimum_sources:
            break

    if evaluation_capacity < evaluation_target:
        raise ValueError("Could not make a document-separated evaluation split of the requested size.")
    return evaluation_sources


def round_robin_select(grouped: dict[str, list[Pair]], target: int, seed: int) -> list[Pair]:
    sources = sorted(grouped, key=lambda source: source_rank(seed, source))
    selected: list[Pair] = []
    rank = 0
    while len(selected) < target:
        added = False
        for source in sources:
            if rank < len(grouped[source]):
                selected.append(grouped[source][rank])
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
        rank += 1
    if len(selected) != target:
        raise ValueError(f"Could select only {len(selected)} of the requested {target} pairs.")
    return selected


def select_stratum(
    grouped: dict[str, list[Pair]],
    total_target: int,
    train_ratio: float,
    seed: int,
    preferred_evaluation_sources: Iterable[str] = (),
) -> tuple[list[Pair], list[Pair]]:
    """Select one source-disjoint train/evaluation split for a corpus stratum."""
    train_target = int(total_target * train_ratio)
    evaluation_target = total_target - train_target
    evaluation_sources = choose_evaluation_sources(
        grouped,
        evaluation_target,
        train_target,
        seed,
        preferred_evaluation_sources,
    )
    train_grouped = {
        source: pairs for source, pairs in grouped.items() if source not in evaluation_sources
    }
    evaluation_grouped = {
        source: pairs for source, pairs in grouped.items() if source in evaluation_sources
    }
    return (
        round_robin_select(train_grouped, train_target, seed),
        round_robin_select(evaluation_grouped, evaluation_target, seed + 1),
    )


def write_jsonl(path: Path, pairs: Iterable[Pair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            # Keep the QLoRA input schema deliberately minimal.
            record = {"instruction": pair.instruction, "response": pair.response}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def provenance_rows(pairs: Iterable[Pair]) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "instruction_sha256": sha256_text(pair.instruction),
            "response_sha256": sha256_text(pair.response),
            "source": pair.source,
            "section": pair.section,
            "generation_method": pair.method,
        }
        for index, pair in enumerate(pairs, start=1)
    ]


def split_statistics(pairs: list[Pair]) -> dict[str, object]:
    lengths = [len(pair.response) for pair in pairs]
    rfc_count = sum(Path(pair.source).stem.casefold().startswith("rfc") for pair in pairs)
    return {
        "minimum_response_characters": min(lengths),
        "average_response_characters": round(sum(lengths) / len(lengths), 2),
        "maximum_response_characters": max(lengths),
        "content_category_counts": {
            "cisco_product_and_networking": len(pairs) - rfc_count,
            "rfc": rfc_count,
        },
        "generation_method_counts": dict(sorted(Counter(pair.method for pair in pairs).items())),
    }


def validate_jsonl(path: Path, expected_count: int) -> None:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(records) != expected_count:
        raise AssertionError(f"{path} contains {len(records)} records; expected {expected_count}")
    for index, record in enumerate(records, start=1):
        if set(record) != {"instruction", "response"}:
            raise AssertionError(f"{path} record {index} does not have the required two-key schema")
        if not all(isinstance(record[key], str) and record[key].strip() for key in record):
            raise AssertionError(f"{path} record {index} contains an empty/non-text value")


def create_instruction_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    total_pairs: int = 200,
    train_ratio: float = 0.8,
    max_pairs_per_document: int = 10,
    seed: int = 42,
) -> dict[str, object]:
    """Create train/evaluation JSONL files and return their audit report."""
    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be greater than 0 and less than 1")

    text_files = sorted(path for path in input_path.rglob("*.txt") if path.is_file())
    if not text_files:
        raise ValueError(f"No .txt files found below: {input_path}")

    source_texts: dict[str, str] = {}
    source_paths: dict[str, Path] = {}
    all_candidates: list[Pair] = []
    for path in text_files:
        source = path.relative_to(input_path).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        source_texts[source] = text
        source_paths[source] = path
        all_candidates.extend(extract_heading_pairs(text, source))
    all_candidates.extend(extract_curated_pairs(source_texts))

    unique_candidates = deduplicate_candidates(all_candidates)
    grouped = cap_by_source(unique_candidates, max_pairs_per_document)

    # Keep Cisco/product/configuration material prominent instead of allowing
    # the much larger RFC collection to dominate the instruction dataset.
    rfc_grouped = {
        source: pairs for source, pairs in grouped.items() if Path(source).stem.casefold().startswith("rfc")
    }
    domain_grouped = {
        source: pairs for source, pairs in grouped.items() if source not in rfc_grouped
    }
    desired_domain_total = total_pairs // 2
    domain_capacity = sum(len(pairs) for pairs in domain_grouped.values())
    rfc_capacity = sum(len(pairs) for pairs in rfc_grouped.values())
    domain_total = min(desired_domain_total, domain_capacity)
    rfc_total = total_pairs - domain_total
    if rfc_total > rfc_capacity:
        domain_total += rfc_total - rfc_capacity
        rfc_total = rfc_capacity
    if domain_total > domain_capacity or domain_total + rfc_total < total_pairs:
        raise ValueError("Not enough balanced domain/RFC candidates for the requested dataset size.")

    domain_train, domain_evaluation = select_stratum(
        domain_grouped,
        domain_total,
        train_ratio,
        seed,
        PREFERRED_EVALUATION_SOURCES,
    )
    rfc_train, rfc_evaluation = select_stratum(
        rfc_grouped, rfc_total, train_ratio, seed + 10_000
    )
    train_pairs = domain_train + rfc_train
    evaluation_pairs = domain_evaluation + rfc_evaluation
    train_pairs.sort(key=lambda pair: source_rank(seed + 20_000, pair.instruction))
    evaluation_pairs.sort(key=lambda pair: source_rank(seed + 30_000, pair.instruction))

    # Verify every answer is recoverable from its source after the documented
    # whitespace normalization. This is the central grounding guarantee.
    for pair in train_pairs + evaluation_pairs:
        if pair.response not in compact_text(source_texts[pair.source]):
            raise AssertionError(f"Ungrounded response generated from {pair.source}")

    train_file = output_path / "train.jsonl"
    evaluation_file = output_path / "evaluation.jsonl"
    report_file = output_path / "instruction_dataset_report.json"
    write_jsonl(train_file, train_pairs)
    write_jsonl(evaluation_file, evaluation_pairs)
    validate_jsonl(train_file, len(train_pairs))
    validate_jsonl(evaluation_file, len(evaluation_pairs))

    train_sources = {pair.source for pair in train_pairs}
    evaluation_sources_used = {pair.source for pair in evaluation_pairs}
    report: dict[str, object] = {
        "schema_version": 1,
        "task": "QLoRA instruction-tuning dataset preparation",
        "generation_prompt": GENERATION_PROMPT,
        "generation_method": (
            "Deterministic extractive generation from cleaned source text; no external "
            "language model was used. Responses are whitespace-normalized source passages."
        ),
        "input_directory": str(input_path),
        "input_text_file_count": len(text_files),
        "candidate_pair_count_before_deduplication": len(all_candidates),
        "candidate_pair_count_after_deduplication": len(unique_candidates),
        "selected_pair_count": len(train_pairs) + len(evaluation_pairs),
        "requested_train_ratio": train_ratio,
        "actual_train_ratio": len(train_pairs) / (len(train_pairs) + len(evaluation_pairs)),
        "selected_content_balance": {
            "cisco_product_and_networking_pairs": domain_total,
            "rfc_pairs": rfc_total,
        },
        "split_unit": "source_document",
        "split_seed": seed,
        "source_overlap_count": len(train_sources & evaluation_sources_used),
        "grounding_check": "passed",
        "output_schema": ["instruction", "response"],
        "train": {
            "file": str(train_file),
            "count": len(train_pairs),
            "source_document_count": len(train_sources),
            "statistics": split_statistics(train_pairs),
            "sha256": sha256_text(train_file.read_text(encoding="utf-8")),
            "provenance": provenance_rows(train_pairs),
        },
        "evaluation": {
            "file": str(evaluation_file),
            "count": len(evaluation_pairs),
            "source_document_count": len(evaluation_sources_used),
            "statistics": split_statistics(evaluation_pairs),
            "sha256": sha256_text(evaluation_file.read_text(encoding="utf-8")),
            "provenance": provenance_rows(evaluation_pairs),
        },
        "source_files": {
            source: {
                "sha256": sha256_text(source_texts[source]),
                "selected_split": (
                    "train" if source in train_sources else "evaluation"
                    if source in evaluation_sources_used
                    else "not_selected"
                ),
                "selected_pair_count": sum(
                    pair.source == source for pair in train_pairs + evaluation_pairs
                ),
            }
            for source in sorted(source_paths)
        },
    }
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    args = build_instruction_dataset_parser().parse_args()
    try:
        report = create_instruction_dataset(
            args.input_dir,
            args.output_dir,
            total_pairs=args.total_pairs,
            train_ratio=args.train_ratio,
            max_pairs_per_document=args.max_pairs_per_document,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError, AssertionError) as error:
        print(f"error: {error}")
        return 1

    train = report["train"]
    evaluation = report["evaluation"]
    print("=== Instruction Dataset Summary ===")
    print(f"Input text files:       {report['input_text_file_count']:,}")
    print(f"Usable unique pairs:    {report['candidate_pair_count_after_deduplication']:,}")
    print(f"Training pairs:         {train['count']:,}")
    print(f"Evaluation pairs:       {evaluation['count']:,}")
    print(f"Source overlap:         {report['source_overlap_count']}")
    print(f"Grounding check:        {report['grounding_check']}")
    print(f"Output directory:       {Path(args.output_dir).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
