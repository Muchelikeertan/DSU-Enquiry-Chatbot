"""Tests for JSON syntax and required root schema keys across all three knowledge files."""

import json
from pathlib import Path
import pytest
from app.config import settings


def test_files_exist():
    """Verify that all three canonical JSON files exist on disk."""
    assert settings.intents_json_path.exists(), f"Missing: {settings.intents_json_path}"
    assert settings.college_info_json_path.exists(), f"Missing: {settings.college_info_json_path}"
    assert settings.faq_info_json_path.exists(), f"Missing: {settings.faq_info_json_path}"


def test_intents_json_syntax_and_keys():
    """Verify intents.json is valid JSON and has mandatory root keys."""
    with open(settings.intents_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict), "intents.json must be a JSON object"
    expected_keys = [
        "schema_version",
        "last_updated",
        "meta",
        "legacy_tag_aliases",
        "handler_definitions",
        "handler_compatibility",
        "enquiry_form_rules",
        "context_rules",
        "intents",
    ]
    for key in expected_keys:
        assert key in data, f"intents.json missing required root key: '{key}'"

    assert len(data["intents"]) == 40, f"Expected 40 registered intents, got {len(data['intents'])}"


def test_college_info_json_syntax_and_keys():
    """Verify college_info.json is valid JSON and has mandatory root keys."""
    with open(settings.college_info_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict), "college_info.json must be a JSON object"
    expected_keys = [
        "meta",
        "university",
        "campuses",
        "contacts",
        "leadership",
        "schools",
        "sca",
        "facilities",
        "admissions",
        "enquiry",
        "data_conflicts",
    ]
    for key in expected_keys:
        assert key in data, f"college_info.json missing required root key: '{key}'"

    assert "courses" in data["sca"], "college_info.sca missing 'courses' array"
    assert len(data["sca"]["courses"]) == 7, f"Expected 7 SCA courses, got {len(data['sca']['courses'])}"
    assert len(data["schools"]) == 8, f"Expected 8 schools, got {len(data['schools'])}"


def test_faq_info_json_syntax_and_keys():
    """Verify faq_info.json is valid JSON and has mandatory root keys."""
    with open(settings.faq_info_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict), "faq_info.json must be a JSON object"
    expected_keys = [
        "academic_year",
        "last_verified",
        "source_policy",
        "status_values_ref",
        "intents_ref",
        "schema",
        "common_notes",
        "faqs",
    ]
    for key in expected_keys:
        assert key in data, f"faq_info.json missing required root key: '{key}'"

    assert len(data["faqs"]) == 80, f"Expected 80 FAQs, got {len(data['faqs'])}"
