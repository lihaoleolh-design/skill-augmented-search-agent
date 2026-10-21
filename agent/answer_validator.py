import re
from typing import Any


SUPPORT_THRESHOLD = 0.6

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for",
    "from", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "was", "were", "with", "answer", "established",
    "founded", "formed", "created",
}

FACT_PATTERNS = {
    "person": [r"\bwho\b", r"\bwhich person\b", r"\bwhose\b"],
    "organization": [
        r"\bwhich university\b", r"\bwhat university\b",
        r"\bwhich organization\b", r"\bwhat organization\b",
        r"\bwhich company\b", r"\bwhat company\b",
        r"\bwhich institution\b", r"\bwhat institution\b",
    ],
    "year": [
        r"\bwhich year\b", r"\bwhat year\b",
        r"\byear was\b",
        r"\bwhen was\b.*\b(?:founded|established|formed|created)\b",
    ],
    "date": [
        r"\bwhich dates?\b", r"\bwhat dates?\b",
        r"\bon what dates?\b", r"\bon which dates?\b",
        r"\bwhen did\b",
    ],
    "location": [
        r"\bwhere\b", r"\bwhich country\b", r"\bwhat country\b",
        r"\bwhich city\b", r"\bwhat city\b",
    ],
    "number": [r"\bhow many\b", r"\bwhat number\b"],
}

FACT_DESCRIPTIONS = {
    "person": "requested person",
    "organization": "requested organization",
    "year": "requested year",
    "date": "requested date",
    "location": "requested location",
    "number": "requested number",
    "generic": "requested answer",
}

MONTH_PATTERN = (
    r"(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december|jan|feb|mar|apr|jun|"
    r"jul|aug|sep|sept|oct|nov|dec)"
)

ORGANIZATION_PATTERN = re.compile(
    r"\b(?:[A-Z][\w'’.-]*\s+){1,6}"
    r"(?:University|College|Company|Corporation|Foundation|"
    r"Organization|Organisation|Institution)\b"
)

PERSON_WITH_TITLE_PATTERN = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|Prof)\.?\s+"
    r"(?:[A-Z][\w'’.-]*\s+){1,4}[A-Z][\w'’.-]*\b"
)


def _normalized(value: Any) -> str:
    text = value if isinstance(value, str) else str(value or "")
    return " ".join(re.findall(r"[^\W_]+", text.lower(), re.UNICODE))


def _significant_tokens(value: Any) -> set[str]:
    return {
        token
        for token in _normalized(value).split()
        if token not in STOPWORDS
    }


def extract_required_facts(question: str) -> list[dict[str, Any]]:
    """Extract answer slots explicitly requested by the question."""
    text = question.lower()
    matches = []

    for fact_type, patterns in FACT_PATTERNS.items():
        positions = [
            match.start()
            for pattern in patterns
            if (match := re.search(pattern, text))
        ]
        if positions:
            matches.append((min(positions), fact_type))

    if not matches:
        matches = [(0, "generic")]

    facts = []
    for index, (_position, fact_type) in enumerate(sorted(matches), start=1):
        facts.append({
            "fact_id": f"fact-{index}",
            "type": fact_type,
            "description": FACT_DESCRIPTIONS[fact_type],
            "required": True,
            "status": "pending",
            "values": [],
            "evidence_ids": [],
            "missing_reason": None,
        })

    return facts


def _answer_segments(answer: str) -> list[str]:
    return [
        segment.strip()
        for segment in re.split(r"[,;\n|]", answer)
        if segment.strip()
    ]


def _extract_values(answer: str, fact_type: str) -> list[str]:
    if fact_type == "year":
        relationship_years = re.findall(
            r"\b(?:established|founded|formed|created)\b"
            r"[^0-9]{0,40}"
            r"\b((?:1[0-9]{3}|20[0-9]{2}|2100))\b",
            answer,
            re.IGNORECASE,
        )
        if relationship_years:
            return list(dict.fromkeys(relationship_years))

        return list(dict.fromkeys(re.findall(
            r"\b(?:1[0-9]{3}|20[0-9]{2}|2100)\b",
            answer,
        )))

    if fact_type == "date":
        month_dates = re.findall(
            rf"\b{MONTH_PATTERN}\b(?:\s+[0-9]{{1,2}}(?:\s*[-–]\s*[0-9]{{1,2}})?)?(?:,?\s+[0-9]{{4}})?",
            answer,
            re.IGNORECASE,
        )
        iso_dates = re.findall(
            r"\b[0-9]{4}[-/][0-9]{1,2}[-/][0-9]{1,2}\b",
            answer,
        )
        return list(dict.fromkeys([
            value.strip()
            for value in [*month_dates, *iso_dates]
        ]))

    if fact_type == "number":
        return list(dict.fromkeys(re.findall(r"\b[0-9]+(?:\.[0-9]+)?\b", answer)))

    segments = _answer_segments(answer)

    if fact_type == "organization":
        organizations = list(dict.fromkeys(
            match.group(0).strip()
            for match in ORGANIZATION_PATTERN.finditer(answer)
        ))
        if organizations:
            return organizations[:1]

    if fact_type == "person":
        titled_people = list(dict.fromkeys(
            match.group(0).strip()
            for match in PERSON_WITH_TITLE_PATTERN.finditer(answer)
        ))
        if titled_people:
            return titled_people[:1]

    scalar_patterns = [
        r"\b(?:1[0-9]{3}|20[0-9]{2}|2100)\b",
        rf"\b{MONTH_PATTERN}\b(?:\s+[0-9]{{1,2}}(?:\s*[-–]\s*[0-9]{{1,2}})?)?(?:,?\s+[0-9]{{4}})?",
    ]
    candidates = []
    for segment in segments:
        cleaned = segment
        for pattern in scalar_patterns:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"\b(?:established|founded|formed|created)\b",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -–:()")
        if _significant_tokens(cleaned):
            candidates.append(cleaned)

    if fact_type == "organization":
        organization_words = {
            "university", "company", "organization", "organisation",
            "institution", "college", "corporation", "foundation",
        }
        preferred = [
            value
            for value in candidates
            if _significant_tokens(value).intersection(organization_words)
        ]
        if preferred:
            return preferred[:1]

    if fact_type == "person":
        non_organizations = [
            value
            for value in candidates
            if not _significant_tokens(value).intersection({
                "university", "company", "organization", "organisation",
                "institution", "college", "corporation", "foundation",
            })
        ]
        if non_organizations:
            return non_organizations[-1:]

    return candidates[:1] or ([answer.strip()] if answer.strip() else [])


def canonical_answer(
    validation: dict[str, Any],
    fallback: str,
) -> str:
    """Render only validated values in the question's requested order."""
    values = []
    for fact in validation.get("required_facts", []):
        for value in fact.get("values", []):
            cleaned = str(value).strip()
            if cleaned and cleaned not in values:
                values.append(cleaned)

    return "\n".join(values) if values else fallback.strip()


def _value_supported(
    value: str,
    item_text: str,
    fact_type: str,
) -> bool:
    value_normalized = _normalized(value)
    text_normalized = _normalized(item_text)
    if value_normalized and value_normalized in text_normalized:
        return True

    value_tokens = _significant_tokens(value)
    if not value_tokens:
        return False
    text_tokens = _significant_tokens(item_text)

    if fact_type in {"year", "date", "number"}:
        return value_tokens.issubset(text_tokens)

    return len(value_tokens.intersection(text_tokens)) / len(value_tokens) >= 0.8


def _entity_link_supported(entity_values: list[str], item_text: str) -> bool:
    if not entity_values:
        return True

    text_tokens = _significant_tokens(item_text)
    for entity in entity_values:
        entity_tokens = _significant_tokens(entity)
        if entity_tokens and len(entity_tokens.intersection(text_tokens)) / len(entity_tokens) >= 0.8:
            return True
    return False


def _validate_structured_facts(
    answer: str,
    evidence: list[dict[str, Any]],
    required_facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    facts = []
    entity_types = {"person", "organization", "location"}
    entity_values = []

    for requirement in required_facts:
        if requirement["type"] in entity_types:
            entity_values.extend(_extract_values(answer, requirement["type"]))

    for requirement in required_facts:
        fact = {**requirement}
        fact_type = fact["type"]
        values = _extract_values(answer, fact_type)
        supporting_ids = []

        if not values:
            fact.update({
                "status": "missing",
                "values": [],
                "evidence_ids": [],
                "missing_reason": "answer_value_missing",
            })
            facts.append(fact)
            continue

        all_values_supported = True
        for value in values:
            value_supported = False
            for item in evidence:
                item_text = item.get("text", "")
                if not _value_supported(value, item_text, fact_type):
                    continue

                if (
                    fact_type in {"year", "date", "number"}
                    and entity_values
                    and not _entity_link_supported(entity_values, item_text)
                ):
                    continue

                value_supported = True
                evidence_id = item.get("evidence_id")
                if evidence_id and evidence_id not in supporting_ids:
                    supporting_ids.append(evidence_id)

            if not value_supported:
                all_values_supported = False

        fact.update({
            "status": "supported" if all_values_supported else "missing",
            "values": values,
            "evidence_ids": supporting_ids,
            "missing_reason": (
                None if all_values_supported else "supporting_evidence_missing"
            ),
        })
        facts.append(fact)

    return facts


def validate_answer(
    answer: str,
    evidence: list[dict[str, Any]],
    required_facts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate every required answer fact against concrete evidence."""
    normalized_answer = _normalized(answer)

    if not normalized_answer:
        return {
            "valid": False,
            "evidence_complete": False,
            "reason": "empty_answer",
            "support_ratio": 0.0,
            "supporting_evidence_ids": [],
            "missing_terms": [],
            "required_facts": required_facts or [],
        }

    if normalized_answer == "unknown":
        return {
            "valid": False,
            "evidence_complete": False,
            "reason": "insufficient_evidence",
            "support_ratio": None,
            "supporting_evidence_ids": [],
            "missing_terms": [],
            "required_facts": required_facts or [],
        }

    usable = [
        item
        for item in evidence
        if isinstance(item, dict) and item.get("text")
    ]
    requirements = required_facts or [{
        "fact_id": "fact-1",
        "type": "generic",
        "description": "requested answer",
        "required": True,
        "status": "pending",
        "values": [],
        "evidence_ids": [],
        "missing_reason": None,
    }]

    if not usable:
        facts = [
            {
                **fact,
                "status": "missing",
                "values": _extract_values(answer, fact["type"]),
                "evidence_ids": [],
                "missing_reason": "supporting_evidence_missing",
            }
            for fact in requirements
        ]
    else:
        facts = _validate_structured_facts(answer, usable, requirements)

    missing = [fact for fact in facts if fact["status"] != "supported"]
    supporting_ids = list(dict.fromkeys(
        evidence_id
        for fact in facts
        for evidence_id in fact["evidence_ids"]
    ))
    supported_count = len(facts) - len(missing)
    support_ratio = supported_count / len(facts) if facts else 0.0
    valid = not missing

    return {
        "valid": valid,
        "evidence_complete": valid,
        "reason": "supported" if valid else "incomplete_evidence",
        "support_ratio": round(support_ratio, 4),
        "supporting_evidence_ids": supporting_ids,
        "missing_terms": [fact["description"] for fact in missing],
        "required_facts": facts,
    }
