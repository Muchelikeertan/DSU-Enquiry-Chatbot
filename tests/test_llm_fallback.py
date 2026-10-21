"""Automated Tests for Stage 3: Grounded LLM Fallback Integration (Groq API).

Validates:
1. NLP-first architecture: Verified School of Computer Applications (SCA) FAQs bypass Groq.
2. Permitted dynamic and non-SCA handlers invoke Groq appropriately with grounding context.
3. Grounding context assembly strictly adheres to college_info.json and negative guardrails.
4. Enquiry-form policies and fee fallback enforcement on dynamic and unverified queries.
5. Safe handling of API timeouts, authentication errors, rate limits, and network errors.
6. Credential protection: API keys are never leaked, logged, printed, or exposed to users.
7. Preservation of existing conversation context across multi-turn interactions.

All Groq API calls are mocked. No live API calls or real keys are used during testing.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import groq

from app.llm.handler import GroqLLMHandler, LLMResponse
from app.nlp.pipeline import NLPPipeline


@pytest.fixture
def canonical_paths():
    base = Path(__file__).resolve().parent.parent
    return {
        "intents": base / "intents.json",
        "college_info": base / "college_info.json",
        "faq_info": base / "faq_info.json",
    }


@pytest.fixture
def canonical_data(canonical_paths):
    with open(canonical_paths["intents"], "r", encoding="utf-8") as f:
        intents_data = json.load(f)
    with open(canonical_paths["college_info"], "r", encoding="utf-8") as f:
        college_data = json.load(f)
    with open(canonical_paths["faq_info"], "r", encoding="utf-8") as f:
        faq_data = json.load(f)
    return intents_data, college_data, faq_data


@pytest.fixture
def mock_groq_client():
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = "Grounded response from Dayananda Sagar University records."
    response = MagicMock()
    response.choices = [choice]
    client.chat.completions.create.return_value = response
    return client


@pytest.fixture
def pipeline_with_mock_llm(canonical_data, mock_groq_client):
    intents_data, college_data, faq_data = canonical_data
    llm_handler = GroqLLMHandler(
        api_key="mock_gsk_testkey12345",
        model="openai/gpt-oss-120b",
        timeout=15.0,
        max_retries=1,
        client=mock_groq_client,
    )
    return NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=llm_handler,
    ), mock_groq_client


# ==============================================================================
# 1. NLP-First Architecture: Verified SCA Queries Bypass Groq
# ==============================================================================

def test_verified_sca_bca_fee_bypasses_groq(pipeline_with_mock_llm):
    """Verified BCA fee query must be resolved locally by the FAQ retriever without calling Groq."""
    pipeline, mock_client = pipeline_with_mock_llm
    resp = pipeline.process_query("What is the fee for BCA?")

    assert not resp.used_llm
    assert "207,000" in resp.response_text or "2,07,000" in resp.response_text
    assert resp.status == "source_checked"
    mock_client.chat.completions.create.assert_not_called()


def test_verified_sca_mca_eligibility_bypasses_groq(pipeline_with_mock_llm):
    """Verified MCA eligibility FAQ must be returned locally without calling Groq."""
    pipeline, mock_client = pipeline_with_mock_llm
    resp = pipeline.process_query("What is the eligibility for MCA?")

    assert not resp.used_llm
    assert "50%" in resp.response_text
    assert resp.status == "source_checked"
    mock_client.chat.completions.create.assert_not_called()


def test_verified_sca_dean_bypasses_groq(pipeline_with_mock_llm):
    """Verified SCA dean query must return Dr. S. Senthil locally without calling Groq."""
    pipeline, mock_client = pipeline_with_mock_llm
    resp = pipeline.process_query("Who is the dean of computer applications?")

    assert not resp.used_llm
    assert "Dr. S. Senthil" in resp.response_text
    assert resp.status == "source_checked"
    mock_client.chat.completions.create.assert_not_called()


def test_canned_responses_never_invoke_groq(pipeline_with_mock_llm):
    """Greetings, thanks, and capability questions must use canned responses without calling Groq."""
    pipeline, mock_client = pipeline_with_mock_llm

    for phrase in ["Hello", "Thank you very much", "Goodbye", "What can you help me with?"]:
        resp = pipeline.process_query(phrase)
        assert not resp.used_llm
        mock_client.chat.completions.create.assert_not_called()


def test_enquiry_form_intent_never_invokes_groq(pipeline_with_mock_llm):
    """Explicit enquiry request must show form immediately without calling Groq."""
    pipeline, mock_client = pipeline_with_mock_llm
    resp = pipeline.process_query("I want to submit an enquiry form")

    assert not resp.used_llm
    assert resp.show_enquiry_form
    assert resp.form_policy == "always"
    mock_client.chat.completions.create.assert_not_called()


# ==============================================================================
# 2. Permitted Dynamic & Non-SCA Handlers Invoke Groq Appropriately
# ==============================================================================

def test_permitted_non_sca_courses_invokes_groq(pipeline_with_mock_llm):
    """Non-SCA course query (other_department_courses) must invoke Groq with grounded context."""
    pipeline, mock_client = pipeline_with_mock_llm

    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        "Dayananda Sagar University School of Engineering offers B.Tech, M.Tech, and Ph.D. programmes."
    )

    resp = pipeline.process_query("What courses are offered in School of Engineering?")

    assert resp.used_llm
    assert resp.intent == "other_department_courses"
    assert "School of Engineering" in resp.response_text
    mock_client.chat.completions.create.assert_called_once()

    # Inspect the system prompt & grounding context passed to Groq
    call_args = mock_client.chat.completions.create.call_args[1]
    messages = call_args["messages"]
    system_msg = messages[0]["content"]

    assert "GROUNDING CONTEXT:" in system_msg
    assert "School: School of Engineering" in system_msg or "soe" in system_msg


def test_permitted_non_sca_fees_enforces_enquiry_form_policy(pipeline_with_mock_llm):
    """Non-SCA fees (llm_fee_lookup) must trigger enquiry form when exact fee is unverified."""
    pipeline, mock_client = pipeline_with_mock_llm

    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        "Tuition fees for B.Tech programmes vary depending on the specialisation and admission quota "
        "(CET, COMEDK, or Management). The exact 2026-27 fee schedule cannot be confirmed from primary records. "
        "Please fill out the enquiry form below to receive the official schedule."
    )

    resp = pipeline.process_query("What is the fee for B.Tech at DSU?")

    assert resp.used_llm
    assert resp.show_enquiry_form
    assert resp.form_policy == "always"
    assert resp.form_prefill.get("course_name") == "soe_btech"
    mock_client.chat.completions.create.assert_called_once()


def test_unverified_faculty_in_other_departments_grounded_answer(pipeline_with_mock_llm):
    """Deans of non-SCA schools are not confirmed in primary records and must be stated honestly."""
    pipeline, mock_client = pipeline_with_mock_llm

    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        "Official primary records confirm Dr. S. Senthil as the Dean of School of Computer Applications. "
        "Individual deans for other schools are not listed in primary checked records. "
        "Please fill out the enquiry form for faculty queries."
    )

    resp = pipeline.process_query("Who is the dean of engineering?")

    assert resp.used_llm
    assert resp.show_enquiry_form
    mock_client.chat.completions.create.assert_called_once()


# ==============================================================================
# 3. Grounding Context Assembly & Critical Negative Guardrails
# ==============================================================================

def test_grounding_context_assembly_contains_critical_guardrails(canonical_data):
    """Verify that assemble_grounding_context strictly includes documented negative guardrails."""
    intents_data, college_data, _ = canonical_data
    handler = GroqLLMHandler(api_key=None, client=None)

    # 1. Negative guardrail: BCA AIDS campus unconfirmed
    ctx_bca_aids = handler.assemble_grounding_context(
        query="Where is BCA AI and Data Sciences held?",
        intent="course_location",
        course_id="sca_bca_aids",
        school_id="computer_applications",
        college_data=college_data,
    )
    assert "Teaching campus is unconfirmed on 2026 admissions list" in ctx_bca_aids
    assert "Do NOT assume Kudlu Gate" in ctx_bca_aids

    # 2. Negative guardrail: Kudlu Gate hostel unverified
    ctx_hostel = handler.assemble_grounding_context(
        query="Is there a hostel at Kudlu Gate campus?",
        intent="hostel_facilities",
        course_id=None,
        school_id=None,
        college_data=college_data,
    )
    assert "Hostel at Kudlu Gate: Unverified" in ctx_hostel
    assert "Harohalli Main Campus only" in ctx_hostel

    # 3. Negative guardrail: Third-party fee websites conflicting
    ctx_fee = handler.assemble_grounding_context(
        query="What is the fee for MBA?",
        intent="other_department_fees",
        course_id="scm_mba",
        school_id="commerce_management",
        college_data=college_data,
    )
    assert "Third-party fee websites: Aggregator sites show conflicting figures" in ctx_fee
    assert "Non-SCA Fees: Fee depends on quota" in ctx_fee


# ==============================================================================
# 4. Safe Handling of Groq API Failures, Timeouts & Retries
# ==============================================================================

def test_groq_api_timeout_safe_fallback(canonical_data):
    """When Groq API times out, the handler must catch it safely and return a friendly fallback."""
    intents_data, college_data, faq_data = canonical_data

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = groq.APITimeoutError(request=MagicMock())

    handler = GroqLLMHandler(api_key="mock_key", client=mock_client)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=handler,
    )

    resp = pipeline.process_query("What courses are offered in School of Engineering?")

    assert not resp.used_llm
    assert resp.needs_fallback
    assert resp.show_enquiry_form
    assert "unable to retrieve official dynamic details" in resp.response_text or "enquiry form" in resp.response_text
    assert "APITimeoutError" not in resp.response_text  # No raw error leak to user


def test_groq_api_authentication_error_protection(canonical_data):
    """Authentication errors must not leak API key to the user or crash the system."""
    intents_data, college_data, faq_data = canonical_data

    mock_client = MagicMock()
    secret_key = "gsk_supersecretkey999"
    auth_err = groq.AuthenticationError(
        message=f"Invalid API Key: {secret_key}",
        response=MagicMock(),
        body=None,
    )
    mock_client.chat.completions.create.side_effect = auth_err

    handler = GroqLLMHandler(api_key=secret_key, client=mock_client)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=handler,
    )

    resp = pipeline.process_query("What courses are offered in School of Engineering?")

    assert not resp.used_llm
    assert resp.needs_fallback
    assert resp.show_enquiry_form
    assert secret_key not in resp.response_text
    assert "AuthenticationError" not in resp.response_text


def test_groq_rate_limit_safe_fallback(canonical_data):
    """Rate limit errors from Groq must gracefully fall back to the enquiry form."""
    intents_data, college_data, faq_data = canonical_data

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = groq.RateLimitError(
        message="Rate limit reached for model openai/gpt-oss-120b",
        response=MagicMock(),
        body=None,
    )

    handler = GroqLLMHandler(api_key="mock_key", client=mock_client)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=handler,
    )

    resp = pipeline.process_query("What courses are offered in School of Engineering?")

    assert not resp.used_llm
    assert resp.needs_fallback
    assert resp.show_enquiry_form


def test_unconfigured_api_key_graceful_local_fallback(canonical_data):
    """When GROQ_API_KEY is not configured (None), the pipeline must gracefully fall back without errors."""
    intents_data, college_data, faq_data = canonical_data

    handler = GroqLLMHandler(api_key=None, client=None)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=handler,
    )

    # Fee query fallback
    fee_resp = pipeline.process_query("What is the fee for B.Tech?")
    assert not fee_resp.used_llm
    assert fee_resp.show_enquiry_form
    assert "couldn't find verified fee details" in fee_resp.response_text.lower()

    # Dynamic course query fallback
    course_resp = pipeline.process_query("What courses are offered in School of Engineering?")
    assert not course_resp.used_llm
    assert course_resp.show_enquiry_form
    assert "could not verify that detail" in course_resp.response_text.lower()


# ==============================================================================
# 5. Multi-Turn Context Preservation with Grounded LLM
# ==============================================================================

def test_multi_turn_context_preserved_with_llm(pipeline_with_mock_llm):
    """Slots tracked in previous turns must be passed into grounding context in subsequent turns."""
    pipeline, mock_client = pipeline_with_mock_llm
    session_id = "student_session_multi_turn_llm"

    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        "School of Engineering offers B.Tech with various specialisations."
    )

    # Turn 1: Ask about School of Engineering
    t1 = pipeline.process_query("What courses are offered in School of Engineering?", session_id=session_id)
    assert t1.context_slots.get("school_id") == "engineering"
    assert t1.used_llm

    # Turn 2: Follow up about B.Tech fee (re-uses school_id, sets course_id)
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        "B.Tech fees vary by specialisation and quota. Exact fees are unconfirmed. Please use the enquiry form."
    )
    t2 = pipeline.process_query("What is the fee for B.Tech?", session_id=session_id)
    assert t2.context_slots.get("school_id") == "engineering"
    assert t2.context_slots.get("course_id") == "soe_btech"
    assert t2.show_enquiry_form
    assert t2.used_llm


# ==============================================================================
# 6. Stage 3.1 Regression Tests: Enquiry-Form Trigger Detection & Policy
# ==============================================================================

def test_kudlu_gate_hostel_llm_equivalent_phrases_trigger_form(pipeline_with_mock_llm):
    """Live eval scenario: Kudlu Gate hostel unverified answer with 'admission form' and 'no verified information'."""
    pipeline, mock_client = pipeline_with_mock_llm

    # Exact phrasing from live evaluation that previously failed detection
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        "The university's official records confirm hostel facilities only at the Harohalli (Main) campus. "
        "There is no verified information confirming a hostel at the Kudlu Gate campus. "
        "For the most accurate and up-to-date details, please submit an enquiry through the university's admission form."
    )

    resp = pipeline.process_query("Is hostel accommodation available at the Kudlu Gate campus?")

    assert resp.used_llm
    assert resp.show_enquiry_form is True
    assert resp.form_policy == "always"
    assert resp.intent == "hostel_facilities"
    assert "harohalli" in resp.response_text.lower()
    assert "kudlu" in resp.response_text.lower()


def test_unverified_campus_information_triggers_form(pipeline_with_mock_llm):
    """Unverified campus inquiry (e.g. BCA AI & DS) directing to admission/enquiry form must trigger form."""
    pipeline, mock_client = pipeline_with_mock_llm

    # 1. Local retrieval path (SCA_LOC_BCA_AIDS FAQ: form_policy always)
    resp_local = pipeline.process_query("Which campus is BCA in AI and Data Science held at?")
    assert resp_local.show_enquiry_form is True
    assert resp_local.form_policy == "always"
    assert "isn't listed" in resp_local.response_text.lower() or "can't confirm" in resp_local.response_text.lower()

    # 2. Dynamic LLM path (handler with equivalent phrasing like 'unconfirmed' and 'admission form')
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        "Teaching campus location for this specialization is unconfirmed in official 2026 records. "
        "Please submit an admission form to confirm the campus location."
    )
    llm_resp = pipeline.llm_handler.generate_grounded_response(
        query="Where is the campus for this specialisation?",
        intent="course_location",
        course_id=None,
        school_id=None,
        college_data=pipeline.college_data,
    )
    assert llm_resp.show_enquiry_form is True
    assert llm_resp.form_policy == "always"
    assert "unconfirmed" in llm_resp.text.lower()


def test_non_sca_fees_enforces_enquiry_form_policy_without_exact_indicators(pipeline_with_mock_llm):
    """Non-SCA fees (other_department_fees) must trigger enquiry form even if wording varies."""
    pipeline, mock_client = pipeline_with_mock_llm

    # Response explaining quotas and admission desk without the exact phrase 'enquiry form'
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        "Tuition fees for B.Tech programmes vary depending on the specialisation and admission quota "
        "(CET, COMEDK, or Management). Official records do not list a fixed schedule. "
        "Please submit an enquiry through the university's admission form."
    )

    resp = pipeline.process_query("What is the tuition fee for B.Tech in Computer Science?")

    assert resp.used_llm
    assert resp.show_enquiry_form is True
    assert resp.form_policy == "always"
    assert resp.form_prefill.get("course_name") == "soe_btech"


def test_ordinary_answers_do_not_trigger_form(pipeline_with_mock_llm):
    """Ordinary verified answers and canned responses must NOT trigger the enquiry form."""
    pipeline, mock_client = pipeline_with_mock_llm

    # 1. Grounded LLM answer with verified facts (no unverified statements or form directions)
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        "The School of Engineering at Dayananda Sagar University offers B.Tech, M.Tech, and Ph.D. programmes."
    )
    eng_resp = pipeline.process_query("What programmes does the engineering school have?")
    assert eng_resp.used_llm
    assert eng_resp.show_enquiry_form is False
    assert eng_resp.form_policy == "on_failure"

    # 2. Grounded LLM answer for verified leadership / establishment detail
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        "Dayananda Sagar University was established under Karnataka Act No. 20 of 2014 and operates multiple campuses."
    )
    llm_ordinary = pipeline.llm_handler.generate_grounded_response(
        query="Tell me about the university establishment.",
        intent="university_info",
        course_id=None,
        school_id=None,
        college_data=pipeline.college_data,
        form_policy="on_failure",
    )
    assert llm_ordinary.show_enquiry_form is False
    assert llm_ordinary.form_policy == "on_failure"

    # 3. Grounded LLM answer with form_policy='never' does not trigger form even if words mention form
    llm_never = pipeline.llm_handler.generate_grounded_response(
        query="How do I submit my academic documents?",
        intent="admissions_process",
        course_id=None,
        school_id=None,
        college_data=pipeline.college_data,
        form_policy="never",
    )
    assert llm_never.show_enquiry_form is False
    assert llm_never.form_policy == "never"

    # 4. Verified Harohalli hostel FAQ (form_policy: never)
    # Notice: Even though UNI_HOSTEL text mentions 'couldn't verify Kudlu Gate', its policy is never,
    # so show_enquiry_form must remain False (preserving intended policy for the intent)
    hostel_resp = pipeline.process_query("Is hostel available at Harohalli campus?")
    assert hostel_resp.show_enquiry_form is False
    assert hostel_resp.form_policy == "never"

    # 5. Verified static FAQ answers (form_policy: never)
    bca_fee = pipeline.process_query("What is the annual fee for BCA?")
    assert not bca_fee.used_llm
    assert bca_fee.show_enquiry_form is False
    assert bca_fee.form_policy == "never"

    sca_dean = pipeline.process_query("Who is the Dean of the School of Computer Applications?")
    assert not sca_dean.used_llm
    assert sca_dean.show_enquiry_form is False
    assert sca_dean.form_policy == "never"

    # 6. Canned responses (form_policy: never, even when quick replies include 'Enquiry Form')
    greeting = pipeline.process_query("Hello")
    assert not greeting.used_llm
    assert greeting.show_enquiry_form is False
    assert greeting.form_policy == "never"


def test_detect_form_trigger_unit_cases():
    """Unit tests for GroqLLMHandler.detect_form_trigger covering equivalent target phrases."""
    handler = GroqLLMHandler(api_key=None, client=None)

    # Positive cases (equivalent phrases that MUST trigger)
    assert handler.detect_form_trigger("Please fill out the admission form.")
    assert handler.detect_form_trigger("For more details, submit an enquiry.")
    assert handler.detect_form_trigger("Please submit your query to admissions.")
    assert handler.detect_form_trigger("There is no verified information for this course.")
    assert handler.detect_form_trigger("Hostel at Kudlu Gate is unverified.")
    assert handler.detect_form_trigger("Teaching campus is unconfirmed on official records.")
    assert handler.detect_form_trigger("The exact fee cannot be confirmed.")
    assert handler.detect_form_trigger("Dean for other schools is not listed in checked records.")
    assert handler.detect_form_trigger("We could not verify the details.")
    assert handler.detect_form_trigger("This degree isn't listed in the official schedule.")
    assert handler.detect_form_trigger("Please contact the admissions office for details.")

    # Negative cases (ordinary verified answers that MUST NOT trigger)
    assert not handler.detect_form_trigger("DSU offers B.Tech in Computer Science and Engineering.")
    assert not handler.detect_form_trigger("Dr. S. Senthil is the Dean of School of Computer Applications.")
    assert not handler.detect_form_trigger("DSU provides hostel facilities at Harohalli Main Campus.")
    assert not handler.detect_form_trigger("MCA at DSU is a 2-year regular programme.")
    assert not handler.detect_form_trigger("Tuition fees are paid per semester.")
    assert not handler.detect_form_trigger("")
    assert not handler.detect_form_trigger(None)

