"""Canonical Response Formatter and Parser for DSU Enquiry Chatbot.

Enforces Section 7 Response Schema:
- status
- intent
- answer
- sections: List[{ heading, items }]
- sources: List[{ title, url }]
- missing_information: List[str]
- requires_enquiry: bool
"""

import re
from typing import Any, Dict, List, Optional, Tuple


def parse_canonical_response(
    response_text: str,
    source: Optional[str] = None,
    show_enquiry_form: bool = False,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, str]], List[str], bool]:
    """Parses formatted response text into canonical schema components.

    Returns:
        (answer, sections, sources, missing_information, requires_enquiry)
    """
    if not response_text:
        return ("", [], [], [], show_enquiry_form)

    text = response_text.strip()
    sections: List[Dict[str, Any]] = []
    sources: List[Dict[str, str]] = []
    missing_information: List[str] = []

    # 1. Extract Answer (lead sentence or text following **Answer:**)
    answer_match = re.search(r"\*\*(?:Answer|Overview):\*\*\s*(.+?)(?=\n\n\*\*|\Z)", text, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        # Fallback: take the first paragraph before any section header or bullet list
        first_para = re.split(r"\n\n|\n(?=[-*•])", text)[0]
        answer = first_para.strip()

    # 2. Extract Sections (e.g. **Key details:**, **Next step:**, etc.)
    section_blocks = re.findall(
        r"\*\*([^*:]+):\*\*\s*\n(.*?)(?=\n\s*\*\*[^*:]+:\*\*|\Z)",
        text,
        re.DOTALL,
    )

    for heading, content in section_blocks:
        heading_clean = heading.strip()
        if heading_clean.lower() in ("answer", "overview"):
            continue

        if heading_clean.lower() in ("official sources", "sources", "references"):
            # Handled separately in sources extraction
            continue

        # Extract items from bullets or numbered list
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        items: List[str] = []
        for line in lines:
            bullet_match = re.match(r"^[-*•\d+.]\s*(.*)", line)
            if bullet_match:
                items.append(bullet_match.group(1).strip())
            else:
                items.append(line)

        if items:
            sections.append({
                "heading": heading_clean,
                "items": items,
            })

    # 3. Extract Sources ([Title](URL) markdown links)
    source_matches = re.findall(r"\[([^\]]+)\]\((https?://[^\)]+)\)", text)
    seen_urls = set()
    for title, url in source_matches:
        if url not in seen_urls:
            sources.append({"title": title.strip(), "url": url.strip()})
            seen_urls.add(url)

    # If no markdown sources were found in text, use the primary source if it's a URL
    if not sources and source and source.startswith("http"):
        sources.append({"title": "Official DSU Portal", "url": source})

    # 4. Extract Missing / Unverified Information
    missing_patterns = [
        re.compile(r"\b(?:not|un)\s*confirmed\b", re.IGNORECASE),
        re.compile(r"\bunverified\b", re.IGNORECASE),
        re.compile(r"\brequires?\s+(?:official\s+)?confirmation\b", re.IGNORECASE),
        re.compile(r"\bnot\s+detailed\s+in\s+the\s+public\b", re.IGNORECASE),
        re.compile(r"\bneeds?\s+confirmation\b", re.IGNORECASE),
        re.compile(r"\bplease\s+contact\s+(?:the\s+)?(?:dsu\s+)?admissions\b", re.IGNORECASE),
    ]

    # Check all section items and full text
    for sec in sections:
        for item in sec.get("items", []):
            if any(p.search(item) for p in missing_patterns):
                if item not in missing_information:
                    missing_information.append(item)

    if not missing_information and show_enquiry_form:
        for p in missing_patterns:
            m = p.search(text)
            if m:
                # Extract sentence containing pattern
                sentences = re.split(r"(?<=[.!?])\s+", text)
                for s in sentences:
                    if p.search(s) and s.strip() not in missing_information:
                        missing_information.append(s.strip())
                break

    requires_enquiry = bool(show_enquiry_form)

    return (answer, sections, sources, missing_information, requires_enquiry)
