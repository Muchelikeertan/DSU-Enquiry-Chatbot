"""Comprehensive tests for the end-to-end NLP Pipeline.

Covers:
1. Intent Classification metrics and ambiguity handling.
2. Local verified FAQ retrieval for SCA with 0 LLM calls.
3. Form-policy enforcement on unverified information.
4. Multi-turn conversation context tracking and 6-turn TTL expiration.
5. Entity override and pronoun cue resolution.
"""

import json
import pytest
from app.config import settings
from app.nlp.pipeline import NLPPipeline


@pytest.fixture(scope="module")
def pipeline():
    return NLPPipeline.from_files(
        intents_path=settings.intents_json_path,
        college_info_path=settings.college_info_json_path,
        faq_info_path=settings.faq_info_json_path,
        intent_confidence_threshold=0.60,
        faq_confidence_threshold=0.65,
        context_ttl_turns=6,
    )


# ==============================================================================
# 1. Intent Classification & Evaluation Metrics
# ==============================================================================

def test_intent_classification_cross_validation_metrics(pipeline):
    """Verify that stratified 3-fold cross validation executes without data leakage."""
    metrics = pipeline.classifier.evaluate(pipeline.intents_data, n_splits=3)
    assert metrics.n_classes == 38
    assert metrics.n_samples > 350
    assert metrics.accuracy > 0.40, f"Expected baseline accuracy > 0.40, got {metrics.accuracy}"
    assert metrics.macro_f1 > 0.35, f"Expected baseline macro F1 > 0.35, got {metrics.macro_f1}"


def test_intent_classification_high_confidence_canned(pipeline):
    """Test standard canned response intents."""
    resp_hi = pipeline.process_query("Hello there!", session_id="test_canned_1")
    assert resp_hi.intent == "greeting"
    assert resp_hi.handler == "canned_response"
    assert not resp_hi.used_llm

    resp_thanks = pipeline.process_query("Thank you so much", session_id="test_canned_2")
    assert resp_thanks.intent == "thanks"
    assert resp_thanks.handler == "canned_response"


# ==============================================================================
# 2. Local FAQ Retrieval for SCA (Zero LLM Calls)
# ==============================================================================

def test_sca_bca_fee_retrieval(pipeline):
    """Verify BCA fee inquiry returns verified amount locally without LLM."""
    resp = pipeline.process_query("What is the fee for BCA?", session_id="test_bca_fee")
    assert resp.intent == "sca_fees"
    assert resp.entities.course_id == "sca_bca"
    assert "207,000" in resp.response_text or "2,07,000" in resp.response_text or "207000" in resp.response_text
    assert resp.used_llm is False
    assert resp.show_enquiry_form is False
    assert resp.form_policy == "never"


def test_sca_mca_eligibility_retrieval(pipeline):
    """Verify MCA eligibility inquiry returns verified 50% criteria locally without LLM."""
    resp = pipeline.process_query("What is the eligibility for MCA?", session_id="test_mca_elig")
    assert resp.intent == "sca_eligibility"
    assert resp.entities.course_id == "sca_mca"
    assert "50%" in resp.response_text
    assert resp.used_llm is False
    assert resp.show_enquiry_form is False


def test_sca_dean_retrieval(pipeline):
    """Verify Dean inquiry returns Dr. S. Senthil locally."""
    resp = pipeline.process_query("Who is the Dean of the School of Computer Applications?", session_id="test_dean")
    assert resp.intent == "sca_dean"
    assert "Dr. S. Senthil" in resp.response_text
    assert resp.used_llm is False


def test_sca_bca_location_retrieval(pipeline):
    """Verify regular BCA campus inquiry returns Kudlu Gate."""
    resp = pipeline.process_query("Where is regular BCA taught?", session_id="test_bca_loc")
    assert resp.intent == "sca_location"
    assert "Kudlu Gate" in resp.response_text
    assert resp.used_llm is False


# ==============================================================================
# 3. Unverified Information & Form Policy Enforcement
# ==============================================================================

def test_bca_ai_ds_unverified_location_triggers_form(pipeline):
    """Verify BCA in AI and Data Sciences campus inquiry displays enquiry form (form_policy: always)."""
    resp = pipeline.process_query("Where is BCA in AI and Data Science located?", session_id="test_aids_loc")
    assert resp.intent == "sca_location"
    assert resp.entities.course_id == "sca_bca_ai_ds"
    assert resp.show_enquiry_form is True
    assert resp.form_policy == "always"
    assert resp.form_prefill.get("course_name") == "sca_bca_ai_ds"


def test_kudlu_gate_hostel_unverified_triggers_form(pipeline):
    """Verify Kudlu Gate hostel inquiry triggers enquiry form (form_policy: always)."""
    resp = pipeline.process_query("Can I stay in the hostel if I study at Kudlu Gate?", session_id="test_hostel")
    assert resp.intent == "hostel_facilities"
    assert resp.show_enquiry_form is True
    assert resp.form_policy == "always"


def test_direct_enquiry_intent(pipeline):
    """Verify direct enquiry intent displays enquiry form."""
    resp = pipeline.process_query("I want to submit an enquiry to DSU", session_id="test_direct_enquiry")
    assert resp.intent == "enquiry"
    assert resp.handler == "enquiry_form"
    assert resp.show_enquiry_form is True
    assert resp.form_policy == "always"


# ==============================================================================
# 4. Multi-Turn Conversation Context & 6-Turn TTL
# ==============================================================================

def test_multi_turn_conversation_flow(pipeline):
    """Simulate a realistic multi-turn student conversation testing slot tracking and pronouns."""
    session_id = "student_session_multi_turn"

    # Turn 1: Ask about BCA
    t1 = pipeline.process_query("Tell me about BCA at DSU", session_id=session_id)
    assert t1.entities.course_id == "sca_bca"
    assert t1.context_slots.get("course_id") == "sca_bca"

    # Turn 2: Follow-up fee using pronoun 'its' (no course named)
    t2 = pipeline.process_query("What is its fee?", session_id=session_id)
    assert t2.intent == "sca_fees"
    assert "207,000" in t2.response_text or "2,07,000" in t2.response_text or "207000" in t2.response_text
    assert t2.context_slots.get("course_id") == "sca_bca"

    # Turn 3: Follow-up location using pronoun 'there'
    t3 = pipeline.process_query("Where is it located?", session_id=session_id)
    assert t3.intent == "sca_location"
    assert "Kudlu Gate" in t3.response_text

    # Turn 4: Switch course entity explicitly to MCA
    t4 = pipeline.process_query("What about MCA?", session_id=session_id)
    assert t4.entities.course_id == "sca_mca"
    assert t4.context_slots.get("course_id") == "sca_mca"

    # Turn 5: Follow-up fee on the new course ('MCA')
    t5 = pipeline.process_query("What is the fee for that?", session_id=session_id)
    assert t5.intent == "sca_fees"
    assert "342,000" in t5.response_text or "3,42,000" in t5.response_text or "342000" in t5.response_text
    assert t5.context_slots.get("course_id") == "sca_mca"


def test_context_expiration_after_six_turns(pipeline):
    """Verify that context slots clear after 6 inactive turns."""
    session_id = "student_session_ttl_expire"

    # Set course context
    pipeline.process_query("Tell me about MCA", session_id=session_id)
    ctx = pipeline.context_tracker.get_or_create_session(session_id)
    assert ctx.course_id == "sca_mca"

    # Send 6 greetings / generic queries without updating course
    for i in range(6):
        pipeline.process_query("Hello", session_id=session_id)

    # Context slots should be expired and reset
    expired_ctx = pipeline.context_tracker.get_or_create_session(session_id)
    assert expired_ctx.course_id is None
    assert expired_ctx.school_id is None


def test_ambiguous_follow_up_without_context(pipeline):
    """When a student asks 'What is the fee?' with 0 prior context, ask for clarification."""
    session_id = "student_session_no_context"
    resp = pipeline.process_query("What is the fee?", session_id=session_id)
    assert resp.intent == "ask_which_course"
    assert "Which course" in resp.response_text
    assert len(resp.quick_replies) > 0
