"""Entity Extractor for DSU College Enquiry Chatbot.

Extracts course references (course_id), school references (school_id),
degree levels, and specialisations from user queries using college_info.json.
Implements longest-match greedy entity resolution to resolve overlapping titles
(e.g., 'BCA in AI and Data Sciences' vs 'BCA').
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from app.nlp.preprocessor import TextPreprocessor


@dataclass
class ExtractedEntities:
    """Entities discovered in user query."""
    course_id: Optional[str] = None
    school_id: Optional[str] = None
    degree_level: Optional[str] = None
    specialisation_text: Optional[str] = None
    matched_aliases: List[str] = field(default_factory=list)
    course_ids: List[str] = field(default_factory=list)
    school_ids: List[str] = field(default_factory=list)

    @property
    def has_entities(self) -> bool:
        """Returns True if any entity was successfully recognized."""
        return bool(self.course_id or self.school_id or self.specialisation_text or self.course_ids)


class EntityExtractor:
    """Indexes and extracts academic entities using canonical IDs and aliases from college_info.json."""

    KNOWN_DEGREES = [
        "bca", "mca", "btech", "mtech", "bsc", "msc",
        "bba", "bcom", "bpharm", "mpharm", "pharmd",
        "bpt", "mpt", "bdes", "ba", "llb", "llm", "phd"
    ]

    def __init__(
        self,
        college_data: Optional[Dict[str, Any]] = None,
        preprocessor: Optional[TextPreprocessor] = None,
    ):
        self.preprocessor = preprocessor or TextPreprocessor()
        # Aliases sorted by length descending: (norm_alias, course_id, school_id, degree_level)
        self.course_aliases: List[Tuple[str, str, str, Optional[str]]] = []
        # School aliases sorted by length descending: (norm_alias, school_id)
        self.school_aliases: List[Tuple[str, str]] = []
        # Specialisation entries: (norm_spec, canonical_spec, course_id)
        self.specialisations: List[Tuple[str, str, Optional[str]]] = []

        if college_data:
            self.load_entities(college_data)

    def _add_course_alias(
        self,
        raw_alias: Optional[str],
        course_id: str,
        school_id: str,
        degree_level: Optional[str],
    ) -> None:
        """Adds normalized course alias and natural connector variations."""
        if not raw_alias:
            return

        norm = self.preprocessor.normalize(raw_alias)
        if not norm:
            return

        self.course_aliases.append((norm, course_id, school_id, degree_level))

        # Handle connector 'in' between degree and specialisation (e.g., 'bca ai and ds' <-> 'bca in ai and ds')
        for deg in self.KNOWN_DEGREES:
            if norm.startswith(deg + " ") and not norm.startswith(deg + " in "):
                rest = norm[len(deg) + 1:]
                self.course_aliases.append((f"{deg} in {rest}", course_id, school_id, degree_level))
            elif norm.startswith(deg + " in "):
                rest = norm[len(deg) + 4:]
                self.course_aliases.append((f"{deg} {rest}", course_id, school_id, degree_level))

    def _add_school_alias(self, raw_alias: Optional[str], school_id: str) -> None:
        """Adds normalized school alias and colloquial short-forms."""
        if not raw_alias:
            return

        norm = self.preprocessor.normalize(raw_alias)
        if not norm:
            return

        self.school_aliases.append((norm, school_id))

        # Colloquial variations (e.g. 'School of Engineering' -> 'Engineering')
        short = re.sub(r"^(?:school\s+of|college\s+of)\s+", "", norm).strip()
        if short and short != norm:
            self.school_aliases.append((short, school_id))
            # Also e.g. "engineering school", "law school"
            self.school_aliases.append((f"{short} school", school_id))
            self.school_aliases.append((f"{short} department", school_id))

    def load_entities(self, college_data: Dict[str, Any]) -> None:
        """Indexes all courses, schools, aliases, and specialisations from college_info.json."""
        self.course_aliases = []
        self.school_aliases = []
        self.specialisations = []

        # 1. School of Computer Applications (SCA)
        sca_id = "computer_applications"
        self._add_school_alias("School of Computer Applications", sca_id)
        self._add_school_alias("SCA", sca_id)
        self._add_school_alias("Computer Applications Department", sca_id)
        self._add_school_alias("Department of Computer Applications", sca_id)

        for course in college_data.get("sca", {}).get("courses", []):
            cid = course.get("id")
            if not cid:
                continue
            lvl = course.get("level")
            raw_aliases = [course.get("name"), course.get("full_name")] + course.get("aliases", [])
            for a in raw_aliases:
                self._add_course_alias(a, cid, sca_id, lvl)

        # 2. Other schools and constituent colleges
        for school in college_data.get("schools", []):
            sid = school.get("id")
            if not sid:
                continue

            s_aliases = [school.get("name"), school.get("abbreviation")] + school.get("aliases", [])
            for sa in s_aliases:
                self._add_school_alias(sa, sid)

            # Programmes directly under school
            for prog in school.get("programmes", []):
                pid = prog.get("id")
                if not pid:
                    continue
                lvl = prog.get("level")
                p_aliases = [prog.get("degree")] + prog.get("aliases", [])
                for pa in p_aliases:
                    self._add_course_alias(pa, pid, sid, lvl)

                for spec in prog.get("specialisations", []):
                    norm_spec = self.preprocessor.normalize(spec)
                    if norm_spec:
                        self.specialisations.append((norm_spec, spec, pid))

            # Constituent colleges under school (e.g. School of Health Sciences)
            for col in school.get("colleges", []):
                col_name = col.get("name")
                self._add_school_alias(col_name, sid)
                for prog in col.get("programmes", []):
                    pid = prog.get("id")
                    if not pid:
                        continue
                    lvl = prog.get("level")
                    p_aliases = [prog.get("degree")] + prog.get("aliases", [])
                    for pa in p_aliases:
                        self._add_course_alias(pa, pid, sid, lvl)

                    for spec in prog.get("specialisations", []):
                        norm_spec = self.preprocessor.normalize(spec)
                        if norm_spec:
                            self.specialisations.append((norm_spec, spec, pid))

        # Deduplicate and sort descending by length for greedy longest-match resolution
        self.course_aliases = sorted(list(set(self.course_aliases)), key=lambda x: len(x[0]), reverse=True)
        self.school_aliases = sorted(list(set(self.school_aliases)), key=lambda x: len(x[0]), reverse=True)
        self.specialisations = sorted(list(set(self.specialisations)), key=lambda x: len(x[0]), reverse=True)

    def extract(self, text: str) -> ExtractedEntities:
        """Extracts entities from text using non-overlapping greedy evaluation."""
        norm_text = self.preprocessor.normalize(text)
        if not norm_text:
            return ExtractedEntities()

        matched_course_spans: List[Tuple[int, int, str, str, Optional[str], str]] = []

        def spans_overlap(start: int, end: int, span_list: List[Tuple[int, int, Any, Any, Any, Any]]) -> bool:
            return any(not (end <= s or start >= e) for s, e, *_ in span_list)

        # 1. Match Courses (longest match first, non-overlapping)
        for alias, cid, sid, lvl in self.course_aliases:
            pattern = r"\b" + re.escape(alias) + r"\b"
            for m in re.finditer(pattern, norm_text):
                start, end = m.span()
                if not spans_overlap(start, end, matched_course_spans):
                    matched_course_spans.append((start, end, cid, sid, lvl, alias))

        # Sort matches by appearance in text
        matched_course_spans.sort(key=lambda x: x[0])

        matched_course_ids: List[str] = []
        matched_school_ids: List[str] = []
        matched_aliases: List[str] = []
        matched_course_id: Optional[str] = None
        matched_school_id: Optional[str] = None
        matched_degree_level: Optional[str] = None

        for _, _, cid, sid, lvl, alias in matched_course_spans:
            if cid not in matched_course_ids:
                matched_course_ids.append(cid)
            if sid not in matched_school_ids:
                matched_school_ids.append(sid)
            if alias not in matched_aliases:
                matched_aliases.append(alias)
            if matched_course_id is None:
                matched_course_id = cid
                matched_school_id = sid
                matched_degree_level = lvl

        # 2. Match Specialisations
        matched_specialisation: Optional[str] = None
        for norm_spec, canonical_spec, associated_cid in self.specialisations:
            pattern = r"\b" + re.escape(norm_spec) + r"\b"
            if re.search(pattern, norm_text):
                matched_specialisation = canonical_spec
                if norm_spec not in matched_aliases:
                    matched_aliases.append(norm_spec)
                # If course was not matched or generic, associate if applicable
                if not matched_course_id and associated_cid:
                    matched_course_id = associated_cid
                    if associated_cid not in matched_course_ids:
                        matched_course_ids.append(associated_cid)
                break

        # 3. Match School (if not already deduced from course, or if explicitly asked)
        matched_school_spans: List[Tuple[int, int, str, str]] = []
        for s_alias, sid in self.school_aliases:
            if s_alias == "management" and ("management quota" in norm_text or "management seat" in norm_text or "mgmt quota" in norm_text):
                continue
            pattern = r"\b" + re.escape(s_alias) + r"\b"
            for m in re.finditer(pattern, norm_text):
                start, end = m.span()
                if not any(not (end <= s or start >= e) for s, e, *_ in matched_school_spans):
                    matched_school_spans.append((start, end, sid, s_alias))

        matched_school_spans.sort(key=lambda x: x[0])
        for _, _, sid, s_alias in matched_school_spans:
            if sid not in matched_school_ids:
                matched_school_ids.append(sid)
            if s_alias not in matched_aliases:
                matched_aliases.append(s_alias)
            if matched_school_id is None:
                matched_school_id = sid

        return ExtractedEntities(
            course_id=matched_course_id,
            school_id=matched_school_id,
            degree_level=matched_degree_level,
            specialisation_text=matched_specialisation,
            matched_aliases=matched_aliases,
            course_ids=matched_course_ids,
            school_ids=matched_school_ids,
        )
