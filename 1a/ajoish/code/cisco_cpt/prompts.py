"""Create the frozen general and corpus-grounded Cisco evaluation prompts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .corpus import sha256_file, write_json, write_jsonl


GENERAL_PROMPTS: tuple[dict[str, Any], ...] = (
    {"prompt_id": "general-factual-01", "category": "factual_knowledge", "prompt": "The capital of France is", "expected_concepts": ["Paris"], "assignment_required": True},
    {"prompt_id": "general-factual-02", "category": "factual_knowledge", "prompt": "Who wrote the novel 1984?", "expected_concepts": ["George Orwell"]},
    {"prompt_id": "general-factual-03", "category": "factual_knowledge", "prompt": "What is the largest planet in the Solar System?", "expected_concepts": ["Jupiter"]},
    {"prompt_id": "general-factual-04", "category": "factual_knowledge", "prompt": "In what year did World War II end?", "expected_concepts": ["1945"]},
    {"prompt_id": "general-factual-05", "category": "factual_knowledge", "prompt": "What is the chemical symbol for gold?", "expected_concepts": ["Au"]},
    {"prompt_id": "general-science-01", "category": "science_and_arithmetic", "prompt": "Water boils at what temperature in degrees Celsius at standard atmospheric pressure?", "expected_concepts": ["100 degrees Celsius"], "assignment_required": True},
    {"prompt_id": "general-science-02", "category": "science_and_arithmetic", "prompt": "The speed of light in vacuum is approximately", "expected_concepts": ["300,000 kilometers per second", "3 x 10^8 meters per second"], "assignment_required": True},
    {"prompt_id": "general-science-03", "category": "science_and_arithmetic", "prompt": "Calculate 17 multiplied by 24.", "expected_concepts": ["408"]},
    {"prompt_id": "general-science-04", "category": "science_and_arithmetic", "prompt": "What process do green plants use to convert light energy into chemical energy?", "expected_concepts": ["photosynthesis"]},
    {"prompt_id": "general-science-05", "category": "science_and_arithmetic", "prompt": "Solve for x: 3x + 7 = 22.", "expected_concepts": ["x = 5"]},
    {"prompt_id": "general-reasoning-01", "category": "reasoning", "prompt": "All squares are rectangles. Some rectangles are blue. Does it necessarily follow that some squares are blue? Explain briefly.", "expected_concepts": ["No", "does not necessarily follow"]},
    {"prompt_id": "general-reasoning-02", "category": "reasoning", "prompt": "What number comes next in the sequence 2, 6, 12, 20, 30?", "expected_concepts": ["42"]},
    {"prompt_id": "general-reasoning-03", "category": "reasoning", "prompt": "A fair coin is tossed twice. What is the probability of getting exactly one head?", "expected_concepts": ["1/2", "50%"]},
    {"prompt_id": "general-reasoning-04", "category": "reasoning", "prompt": "If today is Wednesday, what day of the week will it be 10 days from today?", "expected_concepts": ["Saturday"]},
    {"prompt_id": "general-reasoning-05", "category": "reasoning", "prompt": "A book and a pen cost $13 total. The book costs $9 more than the pen. How much does the pen cost?", "expected_concepts": ["$2", "2 dollars"]},
    {"prompt_id": "general-language-01", "category": "language", "prompt": "Give one synonym for the word concise.", "expected_concepts": ["brief", "succinct"]},
    {"prompt_id": "general-language-02", "category": "language", "prompt": "Correct this sentence: She don't have any apples.", "expected_concepts": ["She doesn't have any apples."]},
    {"prompt_id": "general-language-03", "category": "language", "prompt": "What is the plural form of analysis?", "expected_concepts": ["analyses"]},
    {"prompt_id": "general-language-04", "category": "language", "prompt": "Rewrite in active voice: The report was reviewed by Maya.", "expected_concepts": ["Maya reviewed the report."]},
    {"prompt_id": "general-language-05", "category": "language", "prompt": "What does the idiom break the ice mean?", "expected_concepts": ["initiate conversation", "make people feel more comfortable"]},
    {"prompt_id": "general-common-01", "category": "common_knowledge_and_instruction", "prompt": "List the four seasons in the Northern Hemisphere in calendar order, starting with spring.", "expected_concepts": ["spring", "summer", "autumn", "winter"]},
    {"prompt_id": "general-common-02", "category": "common_knowledge_and_instruction", "prompt": "What should you do first if you smell gas inside a building?", "expected_concepts": ["leave", "avoid flames or electrical switches", "contact emergency services"]},
    {"prompt_id": "general-common-03", "category": "common_knowledge_and_instruction", "prompt": "Name the primary colors of light.", "expected_concepts": ["red", "green", "blue"]},
    {"prompt_id": "general-common-04", "category": "common_knowledge_and_instruction", "prompt": "Arrange these units from smallest to largest: kilometer, centimeter, meter, millimeter.", "expected_concepts": ["millimeter", "centimeter", "meter", "kilometer"]},
    {"prompt_id": "general-common-05", "category": "common_knowledge_and_instruction", "prompt": "State one reason why strong, unique passwords should be used for different online accounts.", "expected_concepts": ["limits compromise", "credential reuse"]},
)


DOMAIN_PROMPTS: tuple[dict[str, Any], ...] = (
    {"prompt_id": "cisco-switching-01", "category": "switching_vlan_stp", "prompt": "What problem does VTP solve when VLAN configuration changes must be propagated across switches?", "expected_concepts": ["central configuration", "VTP domain", "propagates VLAN changes"], "document_fragment": "catalyst9300-vlan", "minimum_page": 13, "evidence_pattern": r"configuration changes centrally"},
    {"prompt_id": "cisco-switching-02", "category": "switching_vlan_stp", "prompt": "Why does a switch need a different bridge ID for each VLAN when using PVST+ or Rapid PVST+?", "expected_concepts": ["each VLAN is a logical bridge", "unique bridge ID"], "document_fragment": "catalyst9300-layer2", "minimum_page": 21, "evidence_pattern": r"each VLAN is considered", "assignment_required": True},
    {"prompt_id": "cisco-switching-03", "category": "switching_vlan_stp", "prompt": "How does vPC role priority influence the primary-device election, and which numeric value is preferred?", "expected_concepts": ["lower value", "better chance to become primary", "default 32667"], "document_fragment": "22-vpc-election", "minimum_page": 2, "evidence_pattern": r"range of values is from 1 to 65636"},
    {"prompt_id": "cisco-switching-04", "category": "switching_vlan_stp", "prompt": "On Cisco Nexus NX-OS, what does switchport mode trunk do and which VLANs can a trunk carry by default?", "expected_concepts": ["Layer 2 trunk", "multiple VLANs", "all VLANs by default"], "document_fragment": "interfaces-configuration-guide-release-106", "minimum_page": 100, "evidence_pattern": r"switchport mode trunk"},
    {"prompt_id": "cisco-switching-05", "category": "switching_vlan_stp", "prompt": "What are the default Layer 2 and Layer 3 port-channel load-balancing methods described for Cisco Nexus NX-OS?", "expected_concepts": ["src-dst-mac", "src-dst ip-l4port"], "document_fragment": "interfaces-configuration-guide-release-106", "minimum_page": 200, "evidence_pattern": r"default method for Layer 2 packets"},
    {"prompt_id": "cisco-routing-01", "category": "routing_ospf_bgp", "prompt": "What type of routing protocol is OSPF, and what information does each OSPF router advertise?", "expected_concepts": ["link-state", "active links", "link type", "metric", "neighbor router"], "document_fragment": "04-nexus9000-nxos-unicast", "minimum_page": 40, "evidence_pattern": r"link-state routing protocol", "assignment_required": True},
    {"prompt_id": "cisco-routing-02", "category": "routing_ospf_bgp", "prompt": "In the shown OSPF NSSA Type-7 LSA example, what forwarding address and metric are advertised?", "expected_concepts": ["192.168.1.1", "metric 20"], "document_fragment": "20-ospf-nssa", "minimum_page": 2, "evidence_pattern": r"Forward Address: 192\.168\.1\.1"},
    {"prompt_id": "cisco-routing-03", "category": "routing_ospf_bgp", "prompt": "How are route-target export and import values used to leak routes from VRF RED into VRF BLUE?", "expected_concepts": ["RED exports 1:1", "BLUE imports 1:1"], "document_fragment": "213908-configure-vrf", "minimum_page": 8, "evidence_pattern": r"route-target export 1:1"},
    {"prompt_id": "cisco-routing-04", "category": "routing_ospf_bgp", "prompt": "What is the purpose of an ICMP Redirect message in IP forwarding?", "expected_concepts": ["shorter path", "host sends traffic directly to the next gateway"], "document_fragment": "19-icmp-redirect", "minimum_page": 2, "evidence_pattern": r"advises the host to send its traffic"},
    {"prompt_id": "cisco-routing-05", "category": "routing_ospf_bgp", "prompt": "Which NX-OS interface command disables ICMP Redirects, and how can its status be verified?", "expected_concepts": ["no ip redirects", "show ip interface", "redirects disabled"], "document_fragment": "19-icmp-redirect", "minimum_page": 13, "evidence_pattern": r"no ip redirects"},
    {"prompt_id": "cisco-datacenter-01", "category": "datacenter_vpc_vxlan_nexus", "prompt": "What are the packet-processing roles of ingress and egress VTEPs in a VXLAN network?", "expected_concepts": ["ingress VTEP encapsulates", "egress VTEP decapsulates"], "document_fragment": "05-nexus9000-nxos-vxlan", "minimum_page": 529, "evidence_pattern": r"Ingress VTEP: Encapsulates", "assignment_required": True},
    {"prompt_id": "cisco-datacenter-02", "category": "datacenter_vpc_vxlan_nexus", "prompt": "What should be checked in the Cisco Nexus ISSU Support Matrix before upgrading a vPC pair?", "expected_concepts": ["current release", "target release", "supported upgrade path"], "document_fragment": "221034-install-upgrade", "minimum_page": 3, "evidence_pattern": r"Select Current release"},
    {"prompt_id": "cisco-datacenter-03", "category": "datacenter_vpc_vxlan_nexus", "prompt": "What is the recommended basic test for VXLAN connectivity between two VTEPs, and what alternative uses loopback interfaces?", "expected_concepts": ["ping between end hosts", "loopback interfaces on VTEPs"], "document_fragment": "11-test-vxlan-vtep", "minimum_page": 1, "evidence_pattern": r"ping test between two end hosts"},
    {"prompt_id": "cisco-datacenter-04", "category": "datacenter_vpc_vxlan_nexus", "prompt": "How does the Cisco N9500 chassis preserve switching service when a fabric module fails?", "expected_concepts": ["up to six fabric modules", "redundancy", "graceful degradation"], "document_fragment": "07-n9500", "minimum_page": 4, "evidence_pattern": r"support up to 6 fabric modules"},
    {"prompt_id": "cisco-datacenter-05", "category": "datacenter_vpc_vxlan_nexus", "prompt": "Give examples of breakout modes supported on Cisco Nexus switch ports.", "expected_concepts": ["4x10G", "4x25G", "2x50G"], "document_fragment": "interfaces-configuration-guide-release-106", "minimum_page": 30, "evidence_pattern": r"supported breakout modes"},
    {"prompt_id": "cisco-security-01", "category": "security_aaa_ise", "prompt": "In Cisco ISE, where should an administrator find and change an authorization rule that uses an unlicensed feature?", "expected_concepts": ["Policy", "Policy Sets", "authorization rule"], "document_fragment": "ise-admin", "minimum_page": 190, "evidence_pattern": r"Policy > Policy Sets"},
    {"prompt_id": "cisco-security-02", "category": "security_aaa_ise", "prompt": "How does IP Source Guard use DHCP snooping information to block source-address spoofing?", "expected_concepts": ["binding table", "source IP lookup", "untrusted interface", "blocks unmatched traffic"], "document_fragment": "catalyst9300-security", "minimum_page": 480, "evidence_pattern": r"source IP lookup table"},
    {"prompt_id": "cisco-security-03", "category": "security_aaa_ise", "prompt": "Which settings can be changed from the Cisco FDM identity policy page?", "expected_concepts": ["identity policy", "default action", "rules", "order"], "document_fragment": "secure-firewall-fdm", "minimum_page": 490, "evidence_pattern": r"To change the Default Action"},
    {"prompt_id": "cisco-security-04", "category": "security_aaa_ise", "prompt": "What happens when both Trusted Network Policy and Untrusted Network Policy are set to Do Nothing for Cisco Secure Client?", "expected_concepts": ["disables Always-On VPN", "disables Trusted Network Detection"], "document_fragment": "secure-client-admin", "minimum_page": 80, "evidence_pattern": r"disables Always-On VPN"},
    {"prompt_id": "cisco-security-05", "category": "security_aaa_ise", "prompt": "In NX-OS SSH configuration, what do the kexalgos and macs options control?", "expected_concepts": ["key exchange", "per-connection keys", "message authentication codes", "detect modification"], "document_fragment": "222090-configure-ciphers", "minimum_page": 15, "evidence_pattern": r"Key exchange methods"},
    {"prompt_id": "cisco-operations-01", "category": "wireless_sdwan_operations", "prompt": "What device property is one component of every Cisco SD-WAN TLOC, and why is it independent of interface addresses?", "expected_concepts": ["system IP address", "router ID", "independent of interfaces"], "document_fragment": "sdwan-systems", "minimum_page": 20, "evidence_pattern": r"four components of the Transport Location"},
    {"prompt_id": "cisco-operations-02", "category": "wireless_sdwan_operations", "prompt": "Which Cisco SD-WAN component orchestrates admission of a new device into the overlay network?", "expected_concepts": ["Cisco SD-WAN Validator", "orchestrates overlay admission"], "document_fragment": "sdwan-systems", "minimum_page": 20, "evidence_pattern": r"automatically orchestrates"},
    {"prompt_id": "cisco-operations-03", "category": "wireless_sdwan_operations", "prompt": "How is a device health score determined in Cisco Catalyst Assurance?", "expected_concepts": ["minimum of all KPI health scores", "five-minute window"], "document_fragment": "catalyst-assurance", "minimum_page": 110, "evidence_pattern": r"minimum of all KPI health scores"},
    {"prompt_id": "cisco-operations-04", "category": "wireless_sdwan_operations", "prompt": "What is the purpose of the command list in Cisco's TAC output quick reference for Nexus switches?", "expected_concepts": ["collect diagnostics", "TAC case", "paste into switch CLI", "commands may fail for unavailable features"], "document_fragment": "09-tac-requested", "minimum_page": 1, "evidence_pattern": r"generic set of commands to collect"},
    {"prompt_id": "cisco-operations-05", "category": "wireless_sdwan_operations", "prompt": "What warranty duration and replacement turnaround are stated for the Cisco N9500 platform?", "expected_concepts": ["one year", "10-day turnaround", "RMA"], "document_fragment": "07-n9500", "minimum_page": 13, "evidence_pattern": r"1-year limited hardware warranty"},
)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _find_evidence(
    extracted_path: Path, minimum_page: int, pattern: str
) -> tuple[int, str]:
    raw_text = extracted_path.read_text(encoding="utf-8")
    for page_match in re.finditer(
        r"<<<PAGE (\d+)>>>\n(.*?)(?=\n\n<<<PAGE \d+>>>\n|\Z)",
        raw_text,
        flags=re.DOTALL,
    ):
        page_number = int(page_match.group(1))
        if page_number < minimum_page:
            continue
        page_text = page_match.group(2)
        normalized_page = " ".join(page_text.split())
        evidence_match = re.search(pattern, normalized_page, flags=re.IGNORECASE)
        if evidence_match is None:
            continue
        start = max(0, evidence_match.start() - 220)
        end = min(len(normalized_page), evidence_match.end() + 520)
        excerpt = normalized_page[start:end]
        if not excerpt:
            raise ValueError(f"Evidence excerpt is empty on page {page_number}")
        return page_number, excerpt
    raise ValueError(
        f"No evidence matching {pattern!r} at page {minimum_page} or later in {extracted_path}"
    )


def build_frozen_prompts(
    manifest_path: Path,
    extracted_root: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    accepted = [record for record in manifest if record.get("accepted") is True]
    prompts: list[dict[str, Any]] = []

    for specification in GENERAL_PROMPTS:
        prompt = dict(specification)
        prompt.update(
            {
                "group": "general",
                "answer_type": "short_answer",
                "rubric": "Answer correctly, directly, and coherently without Cisco-specific content.",
                "assignment_required": bool(prompt.get("assignment_required", False)),
            }
        )
        prompts.append(prompt)

    for specification in DOMAIN_PROMPTS:
        fragment = str(specification["document_fragment"])
        matches = [record for record in accepted if fragment in str(record["document_id"])]
        if len(matches) != 1:
            raise ValueError(f"Expected one accepted document matching {fragment!r}, found {len(matches)}")
        source = matches[0]
        page_number, excerpt = _find_evidence(
            extracted_root / f"{source['document_id']}.txt",
            int(specification["minimum_page"]),
            str(specification["evidence_pattern"]),
        )
        prompt = {
            key: value
            for key, value in specification.items()
            if key not in {"document_fragment", "minimum_page", "evidence_pattern"}
        }
        prompt.update(
            {
                "group": "domain",
                "answer_type": "grounded_short_answer",
                "source_document_id": source["document_id"],
                "source_path": source["source_path"],
                "source_page": page_number,
                "source_excerpt": excerpt,
                "evidence_split": source["split"],
                "rubric": "Use the expected concepts and source evidence; answer factually, directly, and coherently.",
                "assignment_required": bool(prompt.get("assignment_required", False)),
            }
        )
        prompts.append(prompt)

    identifiers = [str(prompt["prompt_id"]) for prompt in prompts]
    if len(prompts) != 50 or len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Frozen prompt set must contain 50 unique prompt IDs")
    group_counts = {
        group: sum(prompt["group"] == group for prompt in prompts)
        for group in ("general", "domain")
    }
    if group_counts != {"general": 25, "domain": 25}:
        raise RuntimeError("Frozen prompt set must contain 25 general and 25 domain prompts")
    domain_split_counts = {
        split: sum(
            prompt["group"] == "domain" and prompt["evidence_split"] == split
            for prompt in prompts
        )
        for split in ("train", "eval")
    }
    if domain_split_counts != {"train": 15, "eval": 10}:
        raise RuntimeError("Domain prompts must contain 15 train-source and 10 eval-source records")
    required_counts = {
        group: sum(
            prompt["group"] == group and prompt["assignment_required"]
            for prompt in prompts
        )
        for group in ("general", "domain")
    }
    if required_counts != {"general": 3, "domain": 3}:
        raise RuntimeError("Exactly three prompts per group must be designated for assignment tables")

    write_jsonl(output_path, prompts)
    report = {
        "prompt_count": len(prompts),
        "group_counts": group_counts,
        "domain_evidence_split_counts": domain_split_counts,
        "assignment_required_counts": required_counts,
        "source_manifest_sha256": sha256_file(manifest_path),
        "frozen_prompts_sha256": sha256_file(output_path),
    }
    write_json(report_path, report)
    return report