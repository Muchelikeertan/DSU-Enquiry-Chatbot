"""Automated Cross-File Integrity Validator for DSU Chatbot Knowledge Base.

Validates:
1. JSON syntax across intents.json, college_info.json, and faq_info.json.
2. All 10 integrity rules defined in intents.json.
3. Status values against college_info.meta.status_values.
4. Legacy aliases resolution and handler compatibility.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class RuleCheckResult:
    """Represents the outcome of a specific integrity rule check."""
    rule_number: int
    rule_name: str
    passed: bool
    description: str
    violations: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Overall validation summary across all files and rules."""
    is_valid: bool
    syntax_ok: bool
    total_intents: int = 0
    total_faqs: int = 0
    total_courses: int = 0
    total_schools: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rule_results: List[RuleCheckResult] = field(default_factory=list)


class JSONIntegrityValidator:
    """Validates the canonical JSON files against all business and structural integrity rules."""

    def __init__(
        self,
        intents_path: Optional[Path] = None,
        college_info_path: Optional[Path] = None,
        faq_info_path: Optional[Path] = None,
        intents_data: Optional[Dict[str, Any]] = None,
        college_data: Optional[Dict[str, Any]] = None,
        faq_data: Optional[Dict[str, Any]] = None,
    ):
        self.intents_path = intents_path
        self.college_info_path = college_info_path
        self.faq_info_path = faq_info_path

        self.intents_data = intents_data
        self.college_data = college_data
        self.faq_data = faq_data

    def load_files(self) -> Tuple[bool, List[str]]:
        """Loads and parses JSON files if raw dicts were not provided."""
        errors: List[str] = []

        if self.intents_data is None and self.intents_path:
            try:
                with open(self.intents_path, "r", encoding="utf-8") as f:
                    self.intents_data = json.load(f)
            except Exception as e:
                errors.append(f"Failed to load intents.json from {self.intents_path}: {e}")

        if self.college_data is None and self.college_info_path:
            try:
                with open(self.college_info_path, "r", encoding="utf-8") as f:
                    self.college_data = json.load(f)
            except Exception as e:
                errors.append(f"Failed to load college_info.json from {self.college_info_path}: {e}")

        if self.faq_data is None and self.faq_info_path:
            try:
                with open(self.faq_info_path, "r", encoding="utf-8") as f:
                    self.faq_data = json.load(f)
            except Exception as e:
                errors.append(f"Failed to load faq_info.json from {self.faq_info_path}: {e}")

        return len(errors) == 0, errors

    def _extract_all_course_ids(self) -> Set[str]:
        """Extracts all valid course IDs across SCA and other schools in college_info."""
        course_ids: Set[str] = set()
        if not self.college_data:
            return course_ids

        # 1. SCA courses
        sca_courses = self.college_data.get("sca", {}).get("courses", [])
        for c in sca_courses:
            if "id" in c:
                course_ids.add(c["id"])

        # 2. Other schools programmes & colleges programmes
        schools = self.college_data.get("schools", [])
        for s in schools:
            for p in s.get("programmes", []):
                if "id" in p:
                    course_ids.add(p["id"])
            for col in s.get("colleges", []):
                for p in col.get("programmes", []):
                    if "id" in p:
                        course_ids.add(p["id"])

        return course_ids

    def validate_all(self) -> ValidationResult:
        """Executes full validation suite including syntax and all 10 integrity rules."""
        syntax_ok, load_errors = self.load_files()
        if not syntax_ok or not self.intents_data or not self.college_data or not self.faq_data:
            return ValidationResult(
                is_valid=False,
                syntax_ok=syntax_ok,
                errors=load_errors,
            )

        intents_list = self.intents_data.get("intents", [])
        faqs_list = self.faq_data.get("faqs", [])
        all_course_ids = self._extract_all_course_ids()
        schools_list = self.college_data.get("schools", [])
        status_values = set(self.college_data.get("meta", {}).get("status_values", {}).keys())
        handler_compat = self.intents_data.get("handler_compatibility", {})
        legacy_aliases = self.intents_data.get("legacy_tag_aliases", {})

        rule_results: List[RuleCheckResult] = []
        all_errors: List[str] = []

        # ----------------------------------------------------------------------
        # Rule 1: Intent tags are unique. This file is the only registry.
        # ----------------------------------------------------------------------
        tags = [i.get("tag") for i in intents_list if "tag" in i]
        seen_tags: Set[str] = set()
        duplicate_tags: Set[str] = set()
        for t in tags:
            if t in seen_tags:
                duplicate_tags.add(t)
            seen_tags.add(t)

        r1_passed = len(duplicate_tags) == 0
        r1_viols = [f"Duplicate intent tag found: '{t}'" for t in duplicate_tags]
        rule_results.append(RuleCheckResult(
            rule_number=1,
            rule_name="Unique Intent Tags",
            passed=r1_passed,
            description="Intent tags must be strictly unique in intents.json.",
            violations=r1_viols,
        ))

        intent_map = {i["tag"]: i for i in intents_list if "tag" in i}

        # ----------------------------------------------------------------------
        # Rule 2: Every faq_info.faqs[*].intent must be a tag defined in intents.json.
        # ----------------------------------------------------------------------
        r2_viols: List[str] = []
        for faq in faqs_list:
            fid = faq.get("id", "UNKNOWN")
            intent = faq.get("intent")
            if not intent:
                r2_viols.append(f"FAQ '{fid}' has missing or empty intent.")
            elif intent not in intent_map:
                r2_viols.append(f"FAQ '{fid}' references unregistered intent '{intent}'.")

        rule_results.append(RuleCheckResult(
            rule_number=2,
            rule_name="FAQ Intent Registration",
            passed=len(r2_viols) == 0,
            description="Every FAQ intent must exist in intents.json registry.",
            violations=r2_viols,
        ))

        # ----------------------------------------------------------------------
        # Rule 3: Every faq course_id / course_ids must exist in college_info.
        # ----------------------------------------------------------------------
        r3_viols: List[str] = []
        for faq in faqs_list:
            fid = faq.get("id", "UNKNOWN")
            cid = faq.get("course_id")
            if cid and cid not in all_course_ids:
                r3_viols.append(f"FAQ '{fid}' has course_id '{cid}' which is not in college_info.")
            cids = faq.get("course_ids", [])
            for c in cids:
                if c not in all_course_ids:
                    r3_viols.append(f"FAQ '{fid}' has course_ids item '{c}' which is not in college_info.")

        rule_results.append(RuleCheckResult(
            rule_number=3,
            rule_name="Course ID Cross-Reference",
            passed=len(r3_viols) == 0,
            description="Every FAQ course_id or course_ids value must exist in college_info.",
            violations=r3_viols,
        ))

        # ----------------------------------------------------------------------
        # Rule 4: A static or form FAQ may only attach to an intent whose handler has faq_static: true.
        # ----------------------------------------------------------------------
        r4_viols: List[str] = []
        for faq in faqs_list:
            fid = faq.get("id", "UNKNOWN")
            intent = faq.get("intent")
            ans_mode = faq.get("answer_mode")
            if ans_mode in ("static", "form"):
                i_obj = intent_map.get(intent)
                if i_obj:
                    handler = i_obj.get("handler")
                    compat = handler_compat.get(handler, {})
                    if not compat.get("faq_static", False):
                        r4_viols.append(
                            f"FAQ '{fid}' with answer_mode '{ans_mode}' attached to intent '{intent}' "
                            f"(handler '{handler}' does not allow static FAQs)."
                        )

        rule_results.append(RuleCheckResult(
            rule_number=4,
            rule_name="Static FAQ Handler Compatibility",
            passed=len(r4_viols) == 0,
            description="Static and form FAQs may only attach to handlers with faq_static: true.",
            violations=r4_viols,
        ))

        # ----------------------------------------------------------------------
        # Rule 5: Dynamic FAQ's dynamic_handler must appear in faq_dynamic_handlers.
        # ----------------------------------------------------------------------
        r5_viols: List[str] = []
        for faq in faqs_list:
            fid = faq.get("id", "UNKNOWN")
            intent = faq.get("intent")
            ans_mode = faq.get("answer_mode")
            if ans_mode == "dynamic":
                dyn_h = faq.get("dynamic_handler")
                i_obj = intent_map.get(intent)
                if i_obj:
                    handler = i_obj.get("handler")
                    compat = handler_compat.get(handler, {})
                    allowed = compat.get("faq_dynamic_handlers", [])
                    if dyn_h not in allowed:
                        r5_viols.append(
                            f"FAQ '{fid}' has dynamic_handler '{dyn_h}' not in allowed list {allowed} "
                            f"for intent '{intent}' (handler '{handler}')."
                        )

        rule_results.append(RuleCheckResult(
            rule_number=5,
            rule_name="Dynamic FAQ Handler Compatibility",
            passed=len(r5_viols) == 0,
            description="Dynamic FAQ dynamic_handler must be in handler_compatibility.faq_dynamic_handlers.",
            violations=r5_viols,
        ))

        # ----------------------------------------------------------------------
        # Rule 6: show_enquiry_form is true only when form_policy is always; otherwise false.
        # ----------------------------------------------------------------------
        r6_viols: List[str] = []
        for faq in faqs_list:
            fid = faq.get("id", "UNKNOWN")
            show_form = faq.get("show_enquiry_form")
            form_pol = faq.get("form_policy")
            if form_pol == "always" and show_form is not True:
                r6_viols.append(f"FAQ '{fid}' has form_policy 'always' but show_enquiry_form is not True.")
            elif form_pol in ("never", "on_failure") and show_form is not False:
                r6_viols.append(f"FAQ '{fid}' has form_policy '{form_pol}' but show_enquiry_form is not False.")

        rule_results.append(RuleCheckResult(
            rule_number=6,
            rule_name="Enquiry Form Boolean Consistency",
            passed=len(r6_viols) == 0,
            description="show_enquiry_form must be true if and only if form_policy is 'always'.",
            violations=r6_viols,
        ))

        # ----------------------------------------------------------------------
        # Rule 7: Static FAQ form mention policy consistency.
        # ----------------------------------------------------------------------
        r7_viols: List[str] = []
        form_mention_pattern = re.compile(r"enquiry form", re.IGNORECASE)
        for faq in faqs_list:
            fid = faq.get("id", "UNKNOWN")
            ans_mode = faq.get("answer_mode")
            form_pol = faq.get("form_policy")
            ans_text = faq.get("answer") or ""

            if ans_mode == "dynamic":
                if form_pol != "on_failure":
                    r7_viols.append(f"Dynamic FAQ '{fid}' must have form_policy 'on_failure', got '{form_pol}'.")
            elif ans_mode == "form":
                if form_pol != "always":
                    r7_viols.append(f"Form FAQ '{fid}' must have form_policy 'always', got '{form_pol}'.")
            elif ans_mode == "static":
                mentions_form = bool(form_mention_pattern.search(ans_text))
                if mentions_form and form_pol != "always":
                    r7_viols.append(f"Static FAQ '{fid}' directs user to enquiry form but form_policy is '{form_pol}' (expected 'always').")
                elif not mentions_form and form_pol != "never":
                    r7_viols.append(f"Static FAQ '{fid}' does not mention enquiry form but form_policy is '{form_pol}' (expected 'never').")

        rule_results.append(RuleCheckResult(
            rule_number=7,
            rule_name="FAQ Form Policy Logic",
            passed=len(r7_viols) == 0,
            description="Form policy must match answer text instructions and answer mode.",
            violations=r7_viols,
        ))

        # ----------------------------------------------------------------------
        # Rule 8: Every tag used in context_rules must be defined in intents.json.
        # ----------------------------------------------------------------------
        r8_viols: List[str] = []
        ctx_rules = self.intents_data.get("context_rules", {})
        follow_ups = ctx_rules.get("follow_up_resolution", {})

        for fu_intent, mapping in follow_ups.items():
            for key, target in mapping.items():
                if isinstance(target, str):
                    if target not in intent_map:
                        r8_viols.append(f"context_rules.{fu_intent}.{key} targets unregistered intent '{target}'.")
                elif isinstance(target, dict):
                    if "bundle" in target:
                        for b_tag in target["bundle"]:
                            if b_tag not in intent_map:
                                r8_viols.append(f"context_rules.{fu_intent}.{key}.bundle targets unregistered intent '{b_tag}'.")
                    for subk, subv in target.items():
                        if isinstance(subv, str) and subk not in ("action", "detail"):
                            if subv not in intent_map:
                                r8_viols.append(f"context_rules.{fu_intent}.{key}.{subk} targets unregistered intent '{subv}'.")

        if "no_context_clarification" in ctx_rules:
            nc_intent = ctx_rules["no_context_clarification"].get("intent")
            if nc_intent and nc_intent not in intent_map:
                r8_viols.append(f"context_rules.no_context_clarification targets unregistered intent '{nc_intent}'.")

        if "ask_what_to_know" in ctx_rules:
            awk_intent = ctx_rules["ask_what_to_know"].get("intent")
            if awk_intent and awk_intent not in intent_map:
                r8_viols.append(f"context_rules.ask_what_to_know targets unregistered intent '{awk_intent}'.")

        rule_results.append(RuleCheckResult(
            rule_number=8,
            rule_name="Context Rules Intent References",
            passed=len(r8_viols) == 0,
            description="Every tag referenced in context_rules must exist in intents.json.",
            violations=r8_viols,
        ))

        # ----------------------------------------------------------------------
        # Rule 9: Every status value used in any file must be a key in college_info.meta.status_values.
        # ----------------------------------------------------------------------
        r9_viols: List[str] = []

        def check_status_deep(obj: Any, path: str):
            # Skip documentation schema definitions and the status enum registry itself
            if path.startswith("faq_info.schema") or path.startswith("college_info.meta.status_values"):
                return

            if isinstance(obj, dict):
                if "status" in obj and obj["status"] is not None:
                    st = obj["status"]
                    if st not in status_values:
                        r9_viols.append(f"Unrecognized status '{st}' found at {path}.status")
                for k, v in obj.items():
                    check_status_deep(v, f"{path}.{k}" if path else k)
            elif isinstance(obj, list):
                for idx, item in enumerate(obj):
                    check_status_deep(item, f"{path}[{idx}]")

        check_status_deep(self.college_data, "college_info")
        check_status_deep(self.faq_data, "faq_info")

        rule_results.append(RuleCheckResult(
            rule_number=9,
            rule_name="Status Values Reference",
            passed=len(r9_viols) == 0,
            description="Every status value in college_info and faq_info must match college_info.meta.status_values.",
            violations=r9_viols,
        ))

        # ----------------------------------------------------------------------
        # Rule 10: Legacy tag aliases must target valid registered intents.
        # ----------------------------------------------------------------------
        r10_viols: List[str] = []
        for alias, target in legacy_aliases.items():
            if target not in intent_map:
                r10_viols.append(f"Legacy alias '{alias}' points to unregistered intent '{target}'.")

        rule_results.append(RuleCheckResult(
            rule_number=10,
            rule_name="Legacy Tag Aliases",
            passed=len(r10_viols) == 0,
            description="All legacy_tag_aliases must resolve to registered intent tags.",
            violations=r10_viols,
        ))

        # Gather all violations into master error list
        for r in rule_results:
            if not r.passed:
                all_errors.extend(r.violations)

        is_all_valid = len(all_errors) == 0

        return ValidationResult(
            is_valid=is_all_valid,
            syntax_ok=True,
            total_intents=len(intents_list),
            total_faqs=len(faqs_list),
            total_courses=len(all_course_ids),
            total_schools=len(schools_list),
            errors=all_errors,
            warnings=[],
            rule_results=rule_results,
        )
