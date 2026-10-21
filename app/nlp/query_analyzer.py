"""Query Analyzer for DSU College Enquiry Chatbot.

Analyzes user queries for:
1. Specific fee categories and quotas (management quota, CET, COMED-K, NRI, hostel mess, transport).
2. Multi-part requested fields (highest package, average package, placement percentage, recruiters, eligibility).
3. Evaluates whether a local FAQ match provides a complete, verified answer or is insufficient.
4. Enforces critical fee-category protection: NEVER substitutes a general tuition fee for an unverified quota.
"""

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class QueryAnalysis:
    """Detailed structural analysis of a student query."""
    raw_query: str
    specific_fee_category: Optional[str] = None
    requested_fields: List[str] = field(default_factory=list)
    is_multi_part: bool = False
    academic_year: Optional[str] = None
    is_comparison: bool = False
    programmes: List[str] = field(default_factory=list)


class QueryAnalyzer:
    """Identifies specific academic query parameters and evaluates local answer sufficiency."""

    FEE_CATEGORY_PATTERNS = [
        (
            "management_quota",
            re.compile(
                r"\b(?:management|mgmt|management['’]?s?)[-\s]+(?:quota|seat|category|fee|fees|admission|rate)\b"
                r"|\bmanagement[-\s]+quota\b|\bmgmt[-\s]+seat\b|\bmanagement[-\s]+fee\b",
                re.IGNORECASE
            )
        ),
        (
            "cet_quota",
            re.compile(
                r"\b(?:k?cet|government|govt|state)[-\s]+(?:quota|seat|category|fee|fees)\b"
                r"|\b(?:k?cet)[-\s]+fee\b|\bkcet[-\s]+cutoff\b",
                re.IGNORECASE
            )
        ),
        (
            "comedk_quota",
            re.compile(
                r"\bcomed-?k[-\s]+(?:quota|seat|category|fee|fees)\b"
                r"|\bcomed-?k[-\s]+fee\b|\bcomed-?k[-\s]+cutoff\b",
                re.IGNORECASE
            )
        ),
        (
            "nri_quota",
            re.compile(
                r"\b(?:nri|foreign|international)[-\s]+(?:quota|seat|category|fee|fees)\b",
                re.IGNORECASE
            )
        ),
        (
            "mess_fee",
            re.compile(
                r"\b(?:mess|food|dining|meal|meals|canteen)[-\s]+(?:fee|fees|charge|charges|cost|rate)\b"
                r"|\bmess[-\s]+charges?\b|\b(?:mess|meals?)\b",
                re.IGNORECASE
            )
        ),
        (
            "hostel_fee",
            re.compile(
                r"\b(?:hostel|residential|accommodation|room)[-\s]+(?:fee|fees|charge|charges|rent|cost|rate)\b"
                r"|\bhostel\b",
                re.IGNORECASE
            )
        ),
        (
            "transport_fee",
            re.compile(
                r"\b(?:transport|transportation|bus)[-\s]+(?:fee|fees|charge|charges|cost|pass|rate)\b",
                re.IGNORECASE
            )
        ),
        (
            "exam_fee",
            re.compile(
                r"\b(?:exam|examination|assessment)[-\s]+(?:fee|fees|charge|charges)\b",
                re.IGNORECASE
            )
        ),
        (
            "scholarship",
            re.compile(
                r"\b(?:scholarship|scholarships|fee[-\s]+concession|fee[-\s]+waiver|financial[-\s]+aid)\b",
                re.IGNORECASE
            )
        ),
    ]

    FIELD_PATTERNS = [
        ("highest_package", re.compile(r"\b(?:highest|top|maximum|max)\s+(?:package|ctc|salary|offer)\b", re.IGNORECASE)),
        ("average_package", re.compile(r"\b(?:average|avg|mean|median)\s+(?:package|ctc|salary|offer)\b", re.IGNORECASE)),
        ("placement_percentage", re.compile(r"\b(?:placement\s+(?:percentage|rate|ratio|record|stats?)|percentage\s+of\s+placements?)\b", re.IGNORECASE)),
        ("recruiters", re.compile(r"\b(?:recruiters|companies|hiring\s+partners|employers|top\s+firms)\b", re.IGNORECASE)),
        ("fee", re.compile(r"\b(?:fee|fees|cost|tuition|charge|charges)\b", re.IGNORECASE)),
        ("eligibility", re.compile(r"\b(?:eligib\w*|criteria|qualification|requirements|percentage\s+required|am\s+i\s+eligible)\b", re.IGNORECASE)),
        ("duration", re.compile(r"\b(?:duration|how\s+long|years|semesters)\b", re.IGNORECASE)),
        ("location", re.compile(r"\b(?:location|campus|where\s+is|address)\b", re.IGNORECASE)),
        ("hostel", re.compile(r"\b(?:hostel|accommodation|room|stay|living)\b", re.IGNORECASE)),
        ("transport", re.compile(r"\b(?:transport|transportation|bus|routes?)\b", re.IGNORECASE)),
        ("documents", re.compile(r"\b(?:documents?|certificates?|marks\s*cards?|what\s+to\s+bring|papers?|records?)\b", re.IGNORECASE)),
        ("scholarship", re.compile(r"\b(?:scholarship|scholarships|financial\s+aid|concession)\b", re.IGNORECASE)),
        ("career", re.compile(r"\b(?:career|careers|opportunities|prospects|job\s+prospects|roles)\b", re.IGNORECASE)),
        ("dean", re.compile(r"\b(?:dean|designation|head\s+of\s+department|who\s+heads)\b", re.IGNORECASE)),
    ]

    @classmethod
    def is_fee_types_comparison_query(cls, query: str) -> bool:
        """Detects whether user is comparing multiple general fee types (tuition vs hostel vs mess vs transport)."""
        q_lower = query.lower()
        has_diff = any(k in q_lower for k in ["difference", "compare", "distinguish", "breakdown", "types of fee", "different fee", "fee components"])
        fee_count = sum(1 for k in ["tuition", "hostel", "mess", "transport", "bus"] if k in q_lower)
        return (has_diff and fee_count >= 2) or fee_count >= 3

    @classmethod
    def analyze(cls, query: str, entities: Optional[Any] = None) -> QueryAnalysis:
        """Analyzes query text to detect specific categories, quotas, academic year, and requested fields."""
        q_lower = query.lower()
        specific_cat = None
        for cat_name, pattern in cls.FEE_CATEGORY_PATTERNS:
            if pattern.search(query):
                specific_cat = cat_name
                break

        if not specific_cat:
            if "management" in q_lower and any(k in q_lower for k in ["quota", "fee", "seat"]):
                specific_cat = "management_quota"
            elif any(k in q_lower for k in ["scholarship", "scholarships", "concession"]):
                specific_cat = "scholarship"

        requested: List[str] = []
        for field_name, pattern in cls.FIELD_PATTERNS:
            if pattern.search(query):
                requested.append(field_name)

        is_multi = len(requested) > 1

        # Extract academic year (e.g. 2026-27, 2026–27, 2026/27)
        year_match = re.search(r"\b(202\d)[-–/](?:202\d|(\d{2}))\b", query)
        academic_year = None
        if year_match:
            y1 = year_match.group(1)
            y2 = year_match.group(2) if year_match.group(2) else year_match.group(0)[-2:]
            academic_year = f"{y1}–{y2}"

        # Detect comparison intent
        is_comp = any(k in q_lower for k in [
            "compare", "comparison", "difference between", "versus", " vs ",
            "which is better", "how does bca compare", "how do they compare"
        ])

        # Detect extracted programmes
        progs = list(getattr(entities, "course_ids", [])) if entities else []
        if not progs:
            if re.search(r"\bbca\b", q_lower):
                progs.append("sca_bca")
            if re.search(r"\b(?:bsc|b\.sc)\s+data\s+science\b|\bdata\s+science\b", q_lower):
                progs.append("sca_bsc_ds")
            if re.search(r"\bmca\b", q_lower):
                progs.append("sca_mca")

        return QueryAnalysis(
            raw_query=query,
            specific_fee_category=specific_cat,
            requested_fields=requested,
            is_multi_part=is_multi,
            academic_year=academic_year,
            is_comparison=is_comp,
            programmes=progs,
        )

    @classmethod
    def is_local_faq_sufficient(cls, faq_match: Any, query: str, analysis: QueryAnalysis) -> bool:
        """Evaluates whether the retrieved local FAQ provides a complete, verified answer.

        Returns False if:
        1. Fee Comparison Query: Multiple distinct fee categories are compared.
        2. Programme Comparison Query: Comparing multiple programmes (e.g. BCA vs B.Sc. Data Science).
        3. Critical Fee-Category Protection: The query specifically requests a quota or fee
           category (e.g. management quota, CET, COMED-K, NRI, hostel fee, mess fee, transport fee)
           that is NOT verified in the FAQ answer. General tuition fees MUST NOT substitute.
        4. Placement queries with unverified placeholder packages where dynamic official retrieval
           holds verified statistics.
        5. Multi-part queries where the FAQ does not cover all requested fields.
        6. The FAQ status explicitly marks dynamic lookup required.
        """
        if not faq_match or not getattr(faq_match, "is_confident", False) or not getattr(faq_match, "answer", None):
            return False

        # Status check: only dynamic_lookup_required requires external retrieval
        status = getattr(faq_match, "status", "")
        if status == "dynamic_lookup_required":
            return False

        # Programme comparison queries cannot be satisfied by a single course FAQ
        if analysis.is_comparison and (len(analysis.programmes) >= 2 or ("bca" in query.lower() and "data science" in query.lower())):
            return False

        # Fee Comparison queries cannot be satisfied by a single course-specific FAQ
        if cls.is_fee_types_comparison_query(query):
            return False

        ans_lower = faq_match.answer.lower()
        q_lower = query.lower()

        # 1. Critical Fee-Category Protection:
        # Never substitute general tuition fees for an unverified quota or specific fee
        if "management" in q_lower or analysis.specific_fee_category == "management_quota":
            if not any(k in ans_lower for k in ["management quota", "management seat", "mgmt quota"]):
                return False

        if analysis.specific_fee_category:
            cat = analysis.specific_fee_category
            if cat == "cet_quota":
                if not any(k in ans_lower for k in ["kcet", "cet fee", "government quota"]):
                    return False
            elif cat == "comedk_quota":
                if not any(k in ans_lower for k in ["comedk", "comed-k"]):
                    return False
            elif cat == "nri_quota":
                if not any(k in ans_lower for k in ["nri", "foreign"]):
                    return False
            elif cat == "mess_fee":
                if not any(k in ans_lower for k in ["mess fee", "mess charge", "food charge"]):
                    return False
            elif cat == "transport_fee":
                if not any(k in ans_lower for k in ["transport fee", "bus fee"]):
                    return False
            elif cat == "exam_fee":
                if not any(k in ans_lower for k in ["exam fee", "examination fee"]):
                    return False
            elif cat == "scholarship":
                if not any(k in ans_lower for k in ["scholarship", "dsat", "concession", "merit"]):
                    return False

        # 2. Placement statistics & recruiter queries:
        is_placement_query = any(k in q_lower for k in ["placement", "package", "salary", "recruiter", "highest", "average", "ctc"])
        if is_placement_query:
            if any(k in ans_lower for k in ["couldn't verify package", "won't quote any", "details were not confirmed"]):
                return False

        # 3. Multi-part queries:
        if len(analysis.requested_fields) > 1:
            for fld in analysis.requested_fields:
                if fld == "highest_package" and not any(k in ans_lower for k in ["highest", "30+", "32 lpa", "30 lpa"]):
                    return False
                elif fld == "average_package" and not any(k in ans_lower for k in ["average", "avg"]):
                    return False
                elif fld == "placement_percentage" and not any(k in ans_lower for k in ["90%", "percentage", "rate"]):
                    return False
                elif fld == "fee" and not re.search(r"(?:inr|rs\.?|₹)\s*\d|\b\d{1,3},\d{2,3}(?:,\d{3})*|\bper\s+year\b|\byearly\s+total\b", ans_lower):
                    return False
                elif fld == "eligibility" and not any(k in ans_lower for k in ["10+2", "puc", "bachelor", "degree", "45%", "50%", "60%", "aggregate", "equivalent qualification"]):
                    return False
                elif fld == "duration" and not any(k in ans_lower for k in ["year", "semester", "duration"]):
                    return False
                elif fld == "location" and not any(k in ans_lower for k in ["campus", "kudlu", "harohalli", "located", "held at", "address"]):
                    return False
                elif fld == "hostel" and not any(k in ans_lower for k in ["hostel", "accommodation", "room", "stay", "mess"]):
                    return False
                elif fld == "transport" and not any(k in ans_lower for k in ["transport", "bus", "route"]):
                    return False
                elif fld == "documents" and not any(k in ans_lower for k in ["marks card", "certificate", "document", "tc", "photograph", "id proof", "aadhaar"]):
                    return False
                elif fld == "scholarship" and not any(k in ans_lower for k in ["scholarship", "dsat", "concession", "merit"]):
                    return False
                elif fld == "career" and not any(k in ans_lower for k in ["career", "role", "analyst", "developer", "placement", "recruit", "opportunit"]):
                    return False
                elif fld == "dean" and not any(k in ans_lower for k in ["senthil", "dean", "professor"]):
                    return False

        # Check if user asked for fee/cost/tuition, but FAQ answer has no fee information:
        if any(k in q_lower for k in ["fee", "fees", "cost", "tuition"]):
            if not re.search(r"(?:inr|rs\.?|₹)\s*\d|\b\d{1,3},\d{2,3}(?:,\d{3})*|\bper\s+year\b|\byearly\s+total\b", ans_lower):
                return False

        # Check if query requests documents/certificates but FAQ answer doesn't list them:
        if any(k in q_lower for k in ["document", "documents", "certificate", "certificates", "marks card", "marks cards", "checklist"]):
            if not any(k in ans_lower for k in ["marks card", "transfer certificate", "migration certificate", "aadhaar", "photograph"]):
                return False

        return True
