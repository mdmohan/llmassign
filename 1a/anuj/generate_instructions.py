"""
Assignment 1A - Part B, Step B1
Instruction dataset creation from the cleaned Cisco corpus (CPU only, no model).

Heuristic generation: Cisco config guides use consistent section headings
("Configuring X", "Restrictions for X", "Information About X", "How to ...",
"Prerequisites for X"). We pair each heading with the explanatory paragraph
that follows it to form an (instruction, response) pair, then normalise the
heading into a natural-language question.

Produces >= 100 pairs in JSONL and an 80/20 train/eval split.
"""
from __future__ import annotations
import argparse, json, random, re
from pathlib import Path

HEADING_RE = re.compile(
    r"^(Configuring|Restrictions for|Information About|How to Configure|"
    r"Prerequisites for|Overview of|About|Understanding|Guidelines for|"
    r"Verifying|Troubleshooting)\b.*",
    re.IGNORECASE,
)

def heading_to_question(h: str) -> str:
    h = h.strip().rstrip(":").strip()
    low = h.lower()
    if low.startswith("configuring"):
        topic = h[len("Configuring"):].strip()
        return f"How do I configure {topic} on a Cisco device?"
    if low.startswith("how to configure"):
        topic = h[len("How to Configure"):].strip()
        return f"How do I configure {topic} on a Cisco device?"
    if low.startswith("restrictions for"):
        topic = h[len("Restrictions for"):].strip()
        return f"What are the restrictions for {topic}?"
    if low.startswith("prerequisites for"):
        topic = h[len("Prerequisites for"):].strip()
        return f"What are the prerequisites for {topic}?"
    if low.startswith("information about") or low.startswith("about"):
        topic = re.sub(r"^(information about|about)\s*", "", h, flags=re.I).strip()
        return f"What is {topic} and how does it work?"
    if low.startswith("understanding"):
        topic = h[len("Understanding"):].strip()
        return f"Explain {topic} in Cisco networking."
    if low.startswith(("verifying", "monitoring")):
        return f"How do I {h[0].lower() + h[1:]}?"
    if low.startswith("troubleshooting"):
        topic = h[len("Troubleshooting"):].strip()
        return f"How do I troubleshoot {topic}?"
    if low.startswith("guidelines for"):
        topic = h[len("Guidelines for"):].strip()
        return f"What are the configuration guidelines for {topic}?"
    return f"Explain the following Cisco topic: {h}"

def clean_para(p: str) -> str:
    p = re.sub(r"\s+", " ", p).strip()
    return p


def looks_like_toc(p: str) -> bool:
    """Table-of-contents / index / header-footer fragments are not useful."""
    low = p.lower()
    if "on page" in low or "onpage" in low:
        return True
    if "configuration guide" in low and "cisco ios" in low:
        return True
    if re.search(r"catalyst\s*9\d{3}\s*switches", low):
        return True
    if p.count("\u2022") >= 2 or p.count("...") >= 2:
        return True
    return False


def space_ratio(p: str) -> float:
    """Broken PDF extraction mashes words together -> very few spaces."""
    if not p:
        return 0.0
    return p.count(" ") / len(p)

def build_pairs(corpus_dir: Path, max_resp_chars: int = 900) -> list[dict]:
    pairs = []
    for f in sorted(corpus_dir.glob("*.txt")):
        product = f.stem
        lines = f.read_text(encoding="utf-8").split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if HEADING_RE.match(line) and 8 <= len(line) <= 90:
                # gather the following non-empty paragraph(s)
                body, j = [], i + 1
                while j < len(lines) and len(" ".join(body)) < max_resp_chars:
                    nxt = lines[j].strip()
                    if HEADING_RE.match(nxt):
                        break
                    if nxt:
                        body.append(nxt)
                    elif body:
                        break
                    j += 1
                resp = clean_para(" ".join(body))
                # quality gates: reasonable length, real prose, not a ToC
                if (len(resp) >= 120 and not looks_like_toc(resp)
                        and space_ratio(resp) >= 0.10):
                    pairs.append({
                        "instruction": heading_to_question(line),
                        "response": resp[:max_resp_chars],
                        "source": product,
                    })
                i = j
            else:
                i += 1
    return pairs

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="domain_corpus", type=Path)
    ap.add_argument("--out", default="instruction_dataset.jsonl", type=Path)
    ap.add_argument("--min-pairs", default=100, type=int)
    ap.add_argument("--max-pairs", default=600, type=int)
    args = ap.parse_args()

    random.seed(0)
    pairs = build_pairs(args.corpus)
    # de-duplicate identical instructions, keep variety across products
    seen, uniq = set(), []
    for p in pairs:
        key = p["instruction"].lower()
        if key not in seen:
            seen.add(key); uniq.append(p)
    random.shuffle(uniq)
    uniq = uniq[: args.max_pairs]
    assert len(uniq) >= args.min_pairs, (
        f"Only {len(uniq)} pairs generated (< {args.min_pairs}). "
        "Relax the heading regex or min length.")

    n_train = int(len(uniq) * 0.8)
    train, eval_ = uniq[:n_train], uniq[n_train:]
    for split, rows in (("train", train), ("eval", eval_)):
        for r in rows:
            r["split"] = split

    with open(args.out, "w", encoding="utf-8") as fh:
        for r in train + eval_:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Generated {len(uniq)} instruction-response pairs")
    print(f"  train: {len(train)}   eval: {len(eval_)}")
    print(f"  written to {args.out}")
    print("\nSample pair:")
    print(json.dumps(train[0], indent=2, ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()
