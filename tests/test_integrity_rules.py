"""Unit and regression tests for all 10 integrity rules in the DSU knowledge base."""

import copy
import json
import pytest
from app.config import settings
from app.validator.json_validator import JSONIntegrityValidator


@pytest.fixture(scope="module")
def loaded_data():
    """Fixture providing raw parsed data of the three canonical JSON files."""
    with open(settings.intents_json_path, "r", encoding="utf-8") as f:
        intents_data = json.load(f)
    with open(settings.college_info_json_path, "r", encoding="utf-8") as f:
        college_data = json.load(f)
    with open(settings.faq_info_json_path, "r", encoding="utf-8") as f:
        faq_data = json.load(f)
    return intents_data, college_data, faq_data


def test_full_knowledge_base_passes_all_rules(loaded_data):
    """Verify that the official knowledge files pass all 10 integrity rules with 0 violations."""
    intents_data, college_data, faq_data = loaded_data
    validator = JSONIntegrityValidator(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
    )
    result = validator.validate_all()

    assert result.syntax_ok is True
    assert result.is_valid is True
    assert len(result.errors) == 0, f"Found unexpected violations: {result.errors}"
    assert len(result.rule_results) == 10
    for r in result.rule_results:
        assert r.passed is True, f"Rule {r.rule_number} ({r.rule_name}) failed: {r.violations}"


def test_rule_1_unique_tags(loaded_data):
    """Rule 1: Intent tags are unique. This file is the only registry of intent tags."""
    intents_data, college_data, faq_data = loaded_data
    tags = [i["tag"] for i in intents_data["intents"]]
    assert len(tags) == len(set(tags)), "Found duplicate intent tags in intents.json"


def test_rule_1_detector_catches_duplicate():
    """Verify that the validator flags duplicate intent tags."""
    fake_intents = {
        "intents": [
            {"tag": "duplicate_tag", "handler": "canned_response"},
            {"tag": "duplicate_tag", "handler": "canned_response"},
        ],
        "handler_compatibility": {},
        "context_rules": {},
        "legacy_tag_aliases": {},
    }
    validator = JSONIntegrityValidator(
        intents_data=fake_intents,
        college_data={"meta": {"status_values": {}}, "sca": {"courses": []}, "schools": []},
        faq_data={"faqs": []},
    )
    result = validator.validate_all()
    r1 = next(r for r in result.rule_results if r.rule_number == 1)
    assert r1.passed is False
    assert any("duplicate_tag" in v for v in r1.violations)


def test_rule_2_every_faq_intent_in_registry(loaded_data):
    """Rule 2: Every faq_info.faqs[*].intent must be a tag defined in intents.json."""
    intents_data, college_data, faq_data = loaded_data
    registered_tags = {i["tag"] for i in intents_data["intents"]}
    for faq in faq_data["faqs"]:
        intent = faq["intent"]
        assert intent in registered_tags, f"FAQ {faq['id']} references unregistered intent '{intent}'"


def test_rule_2_detector_catches_unregistered_intent(loaded_data):
    """Verify validator flags FAQs that reference unknown intents."""
    intents_data, college_data, faq_data = loaded_data
    bad_faq_data = copy.deepcopy(faq_data)
    bad_faq_data["faqs"][0]["intent"] = "non_existent_fake_intent"

    validator = JSONIntegrityValidator(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=bad_faq_data,
    )
    result = validator.validate_all()
    r2 = next(r for r in result.rule_results if r.rule_number == 2)
    assert r2.passed is False
    assert any("non_existent_fake_intent" in v for v in r2.violations)


def test_rule_3_faq_course_ids_exist(loaded_data):
    """Rule 3: Every faq_info course_id / course_ids must exist in college_info."""
    intents_data, college_data, faq_data = loaded_data
    valid_courses = {c["id"] for c in college_data["sca"]["courses"]}
    for s in college_data["schools"]:
        for p in s.get("programmes", []):
            valid_courses.add(p["id"])
        for col in s.get("colleges", []):
            for p in col.get("programmes", []):
                valid_courses.add(p["id"])

    for faq in faq_data["faqs"]:
        if "course_id" in faq and faq["course_id"]:
            assert faq["course_id"] in valid_courses, f"FAQ {faq['id']} has invalid course_id {faq['course_id']}"
        if "course_ids" in faq and faq["course_ids"]:
            for cid in faq["course_ids"]:
                assert cid in valid_courses, f"FAQ {faq['id']} has invalid course_ids item {cid}"


def test_rule_4_static_faq_handler_compatibility(loaded_data):
    """Rule 4: A static or form FAQ may only attach to an intent whose handler has faq_static: true."""
    intents_data, college_data, faq_data = loaded_data
    intent_map = {i["tag"]: i for i in intents_data["intents"]}
    compat = intents_data["handler_compatibility"]

    for faq in faq_data["faqs"]:
        if faq["answer_mode"] in ("static", "form"):
            handler = intent_map[faq["intent"]]["handler"]
            assert compat[handler]["faq_static"] is True, (
                f"FAQ {faq['id']} with mode {faq['answer_mode']} attached to handler {handler} which has faq_static False"
            )


def test_rule_5_dynamic_faq_handler_compatibility(loaded_data):
    """Rule 5: A dynamic FAQ's dynamic_handler must appear in handler_compatibility.faq_dynamic_handlers."""
    intents_data, college_data, faq_data = loaded_data
    intent_map = {i["tag"]: i for i in intents_data["intents"]}
    compat = intents_data["handler_compatibility"]

    for faq in faq_data["faqs"]:
        if faq["answer_mode"] == "dynamic":
            handler = intent_map[faq["intent"]]["handler"]
            allowed = compat[handler]["faq_dynamic_handlers"]
            assert faq["dynamic_handler"] in allowed, (
                f"FAQ {faq['id']} dynamic_handler {faq['dynamic_handler']} not in allowed {allowed} for handler {handler}"
            )


def test_rule_6_show_enquiry_form_vs_form_policy(loaded_data):
    """Rule 6: show_enquiry_form is true only when form_policy is always; otherwise false."""
    _, _, faq_data = loaded_data
    for faq in faq_data["faqs"]:
        show_form = faq["show_enquiry_form"]
        policy = faq["form_policy"]
        if policy == "always":
            assert show_form is True, f"FAQ {faq['id']} policy is always but show_enquiry_form is not True"
        else:
            assert show_form is False, f"FAQ {faq['id']} policy is {policy} but show_enquiry_form is not False"


def test_rule_7_faq_form_policy_logic(loaded_data):
    """Rule 7: FAQ form_policy logic (always if mentions form/mode=form, never if no form mention, on_failure if dynamic)."""
    _, _, faq_data = loaded_data
    for faq in faq_data["faqs"]:
        ans_mode = faq["answer_mode"]
        policy = faq["form_policy"]
        ans = faq.get("answer") or ""

        if ans_mode == "dynamic":
            assert policy == "on_failure", f"FAQ {faq['id']} dynamic answer must have form_policy on_failure"
        elif ans_mode == "form":
            assert policy == "always", f"FAQ {faq['id']} form answer must have form_policy always"
        elif ans_mode == "static":
            if "enquiry form" in ans.lower():
                assert policy == "always", f"FAQ {faq['id']} mentions enquiry form but policy is {policy}"
            else:
                assert policy == "never", f"FAQ {faq['id']} does not mention enquiry form but policy is {policy}"


def test_rule_8_context_rules_references(loaded_data):
    """Rule 8: Every tag used in context_rules must be defined in intents.json."""
    intents_data, _, _ = loaded_data
    registered_tags = {i["tag"] for i in intents_data["intents"]}
    ctx = intents_data["context_rules"]
    for _, mapping in ctx.get("follow_up_resolution", {}).items():
        for _, target in mapping.items():
            if isinstance(target, str):
                assert target in registered_tags, f"context_rules targets unknown intent: {target}"
            elif isinstance(target, dict):
                if "bundle" in target:
                    for b_tag in target["bundle"]:
                        assert b_tag in registered_tags, f"context_rules bundle targets unknown intent: {b_tag}"


def test_rule_9_status_values_reference(loaded_data):
    """Rule 9: Every status value used must be a key in college_info.meta.status_values."""
    intents_data, college_data, faq_data = loaded_data
    allowed_statuses = set(college_data["meta"]["status_values"].keys())

    for faq in faq_data["faqs"]:
        assert faq["status"] in allowed_statuses, f"FAQ {faq['id']} has invalid status '{faq['status']}'"

    for course in college_data["sca"]["courses"]:
        if "campus_status" in course:
            assert course["campus_status"] in allowed_statuses
        if "fee" in course and isinstance(course["fee"], dict):
            assert course["fee"]["status"] in allowed_statuses
        if "eligibility" in course and isinstance(course["eligibility"], dict):
            assert course["eligibility"]["status"] in allowed_statuses


def test_rule_10_legacy_tag_aliases(loaded_data):
    """Rule 10: Legacy tag aliases must point to active registered intents."""
    intents_data, _, _ = loaded_data
    registered_tags = {i["tag"] for i in intents_data["intents"]}
    aliases = intents_data.get("legacy_tag_aliases", {})
    for alias, target in aliases.items():
        assert target in registered_tags, f"Legacy alias '{alias}' points to unknown intent '{target}'"
