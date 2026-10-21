"""Comprehensive Regression Tests for Strict Information-Retrieval and Response Policy.

Verifies:
1. Existing verified FAQs are answered directly without unnecessary escalation or enquiry forms.
2. Official dynamic retrieval returns verified DSU information with valid clickable links.
3. Missing/unverified information triggers the appropriate Admissions Cell referral.
4. Critical fee-category protection: General tuition fee never substitutes for an unverified quota fee (management, CET, COMEDK, hostel/mess).
5. Multi-part questions receive complete answers for verified fields and escalate only unverified fields.
6. Follow-up questions retain programme and conversation context across turns.
7. Official citations are valid, clickable markdown links to official DSU domains.
8. Enquiry forms are NOT displayed unnecessarily when complete verified answers exist.
9. API failures and unavailable sources produce safe, graceful fallbacks without leaking secrets or crashing.
10. Prohibition of negative internal system limitation wording ('I could not find this information on the website', etc.).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import groq

from app.llm.handler import GroqLLMHandler, LLMResponse
from app.nlp.pipeline import NLPPipeline
from app.nlp.query_analyzer import QueryAnalyzer, QueryAnalysis
from app.retrieval.dsu_retriever import DSURetriever, RetrievedEvidence


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
    choice.message.content = "Grounded response from Dayananda Sagar University official records."
    response = MagicMock()
    response.choices = [choice]
    client.chat.completions.create.return_value = response
    return client


@pytest.fixture
def pipeline_with_mock_llm(canonical_data, mock_groq_client):
    intents_data, college_data, faq_data = canonical_data
    llm_handler = GroqLLMHandler(
        api_key="mock_gsk_testkey_retrieval_policy",
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
# 1. Verified FAQs Answered Directly Without Unnecessary Escalation
# ==============================================================================

def test_verified_bca_fee_answers_directly_without_form(pipeline_with_mock_llm):
    """BCA fee is verified locally and must be answered directly with no enquiry form."""
    pipeline, mock_client = pipeline_with_mock_llm
    resp = pipeline.process_query("What is the fee for BCA?")

    assert not resp.used_llm
    assert "207,000" in resp.response_text or "2,07,000" in resp.response_text
    assert resp.show_enquiry_form is False
    assert resp.form_policy == "never"
    assert resp.status == "source_checked"
    mock_client.chat.completions.create.assert_not_called()


def test_verified_mca_eligibility_answers_directly(pipeline_with_mock_llm):
    """MCA eligibility is verified locally and must be answered directly."""
    pipeline, mock_client = pipeline_with_mock_llm
    resp = pipeline.process_query("What is the eligibility for MCA?")

    assert not resp.used_llm
    assert "50%" in resp.response_text
    assert resp.show_enquiry_form is False
    assert resp.form_policy == "never"
    mock_client.chat.completions.create.assert_not_called()


# ==============================================================================
# 2. Critical Fee-Category Protection: No Substitution of General Fees for Quotas
# ==============================================================================

def test_management_quota_fee_never_substitutes_general_tuition(pipeline_with_mock_llm):
    """Asking for BCA management quota fee must NOT substitute the general tuition fee."""
    pipeline, mock_client = pipeline_with_mock_llm

    # Configure mock LLM response adhering to policy
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        "For the confirmed management quota fee for BCA, please contact the DSU Admissions Cell. "
        "You can submit an enquiry below to obtain the applicable amount."
    )

    resp = pipeline.process_query("What is the management quota fee for BCA?")

    # Must invoke grounded LLM/Admissions Cell handling rather than returning static INR 2,07,000
    assert resp.used_llm
    assert resp.show_enquiry_form is True
    assert resp.form_policy == "always"
    assert "admissions cell" in resp.response_text.lower() or "admissions" in resp.response_text.lower()
    # Ensure it did not simply return the general fee as a complete answer
    assert "enquiry below" in resp.response_text.lower() or "submit an enquiry" in resp.response_text.lower()
    assert resp.form_prefill.get("course_name") in ("sca_bca", "bca")


def test_management_quota_offline_fallback_directs_to_admissions_cell(canonical_data):
    """In offline mode without LLM, management quota query must direct to Admissions Cell."""
    intents_data, college_data, faq_data = canonical_data
    # Pipeline without LLM client
    offline_handler = GroqLLMHandler(api_key=None, client=None)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=offline_handler,
    )

    resp = pipeline.process_query("What is the management quota fee for BCA?")

    assert not resp.used_llm
    assert resp.show_enquiry_form is True
    assert resp.form_policy == "always"
    assert "admissions cell" in resp.response_text.lower()
    assert "management quota fee" in resp.response_text.lower()
    assert "207,000" not in resp.response_text  # General tuition fee must not substitute


def test_comedk_quota_fee_protection(pipeline_with_mock_llm):
    """COMED-K fee for B.Tech must not substitute a random generic fee."""
    pipeline, mock_client = pipeline_with_mock_llm
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        "For confirmed COMED-K quota fees for B.Tech, please contact the DSU Admissions Cell. "
        "You can submit an enquiry below to obtain the applicable government/consortium fee structure."
    )

    resp = pipeline.process_query("What is the COMEDK quota fee for B.Tech?")
    assert resp.used_llm
    assert resp.show_enquiry_form is True
    assert resp.form_policy == "always"
    assert "admissions cell" in resp.response_text.lower() or "enquiry below" in resp.response_text.lower()


# ==============================================================================
# 3. Dynamic Retrieval of Verified Official DSU Information
# ==============================================================================

def test_dynamic_retrieval_returns_grounded_placement_facts(pipeline_with_mock_llm):
    """Dynamic retrieval from official sources returns verified facts and clickable source links."""
    pipeline, mock_client = pipeline_with_mock_llm

    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        "**Answer:** Dayananda Sagar University has an active placement record with a highest package of **30+ LPA** "
        "and over **90%** of registered students placed.\n\n"
        "**Official sources:**\n- [DSU Placements & Career Support](https://www.dsu.edu.in/placements)"
    )

    resp = pipeline.process_query("What is the highest package at DSU?")
    assert resp.used_llm
    assert "30+ lpa" in resp.response_text.lower() or "30" in resp.response_text
    assert "https://www.dsu.edu.in" in resp.response_text
    # When fully answered, no unnecessary form escalation
    assert resp.show_enquiry_form is False


# ==============================================================================
# 4. Multi-Part Queries Tracking
# ==============================================================================

def test_multipart_placement_query_answers_verified_and_escalates_unverified(pipeline_with_mock_llm):
    """Multi-part question answers verified fields and escalates unresolved fields."""
    pipeline, mock_client = pipeline_with_mock_llm

    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        "**Answer:**\n"
        "- **Highest Package:** 30+ LPA (verified from official DSU placement records).\n"
        "- **Placement Percentage:** 90%+ record across eligible schools.\n"
        "- **Average Package:** Unconfirmed on official public pages; please contact the DSU Admissions Cell or Placement Office.\n\n"
        "**Official sources:**\n- [DSU Placement Portal](https://www.dsu.edu.in/placements)"
    )

    query = "What is the average package, highest package, and placement percentage at DSU?"
    qa = QueryAnalyzer.analyze(query)
    assert qa.is_multi_part
    assert "highest_package" in qa.requested_fields
    assert "average_package" in qa.requested_fields
    assert "placement_percentage" in qa.requested_fields

    resp = pipeline.process_query(query)
    assert resp.used_llm
    assert "30+" in resp.response_text or "highest" in resp.response_text.lower()
    assert "90%" in resp.response_text or "percentage" in resp.response_text.lower()
    assert "admissions cell" in resp.response_text.lower() or "placement office" in resp.response_text.lower()


# ==============================================================================
# 5. Multi-Turn Context & Programme Preservation
# ==============================================================================

def test_followup_preserves_programme_context_for_quota_query(pipeline_with_mock_llm):
    """Follow-up query preserves the programme slot for quota fee inquiry."""
    pipeline, mock_client = pipeline_with_mock_llm
    session_id = "test_policy_context_turn"

    # Turn 1: Ask about BCA
    t1 = pipeline.process_query("Tell me about BCA at DSU", session_id=session_id)
    assert t1.context_slots.get("course_id") == "sca_bca"

    # Turn 2: Follow-up about management quota fee
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        "For the confirmed management quota fee for BCA, please contact the DSU Admissions Cell. "
        "You can submit an enquiry below to obtain the applicable amount."
    )
    t2 = pipeline.process_query("What is the management quota fee?", session_id=session_id)
    assert t2.context_slots.get("course_id") == "sca_bca"
    assert t2.show_enquiry_form is True
    assert t2.form_prefill.get("course_name") == "sca_bca"


# ==============================================================================
# 6. Clickable Official Citations & Format
# ==============================================================================

def test_official_citations_are_clickable_markdown_links(pipeline_with_mock_llm):
    """Official sources must use clickable markdown syntax [title](url) to official DSU links."""
    pipeline, mock_client = pipeline_with_mock_llm

    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        "**Answer:** Scholarships are awarded to meritorious students through the Dayananda Sagar Scholarship scheme.\n\n"
        "**Official sources:**\n- [DSU Scholarship Scheme](https://www.dsu.edu.in/scholarships)"
    )

    resp = pipeline.process_query("Does DSU offer scholarships?")
    assert "[DSU Scholarship Scheme](https://www.dsu.edu.in/scholarships)" in resp.response_text


# ==============================================================================
# 7. Prohibition of Internal System Limitation Phrasing
# ==============================================================================

def test_no_forbidden_system_limitation_phrasing():
    """Verify that handler and prompt strictly forbid 'could not find on website' phrasing."""
    system_prompt = GroqLLMHandler.SYSTEM_PROMPT.lower()
    assert "i could not find this information on the website" in system_prompt
    assert "the available webpages do not contain the answer" in system_prompt
    assert "for confirmed" in system_prompt or "admissions cell" in system_prompt


# ==============================================================================
# 8. Safe Fallback on API Timeouts & Network Failures
# ==============================================================================

def test_safe_fallback_on_api_timeout(canonical_data):
    """Groq API timeout produces safe fallback without crash or leaking secrets."""
    intents_data, college_data, faq_data = canonical_data

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = groq.APITimeoutError(request=MagicMock())

    handler = GroqLLMHandler(api_key="secret_test_key_9999", client=mock_client)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=handler,
    )

    resp = pipeline.process_query("What is the management quota fee for BCA?")
    assert not resp.used_llm
    assert resp.needs_fallback
    assert resp.show_enquiry_form is True
    assert "secret_test_key_9999" not in resp.response_text
    assert "APITimeoutError" not in resp.response_text
    assert "admissions cell" in resp.response_text.lower() or "enquiry form" in resp.response_text.lower()


# ==============================================================================
# 9. Five User Acceptance Queries (Offline & Online Verification)
# ==============================================================================

def test_query_1_bsc_data_science_eligibility_and_annual_fee(canonical_data):
    """B.Sc. Data Science eligibility & fee: Fee INR 1,72,000 verified, eligibility needs confirmation."""
    intents_data, college_data, faq_data = canonical_data
    # Offline fallback test
    handler = GroqLLMHandler(api_key=None, client=None)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=handler,
    )
    resp = pipeline.process_query("What is the eligibility and annual fee for B.Sc. Data Science?")
    assert "172,000" in resp.response_text or "1,72,000" in resp.response_text
    assert "eligibility" in resp.response_text.lower()
    assert resp.show_enquiry_form is True
    assert "**Answer:**" in resp.response_text
    assert "**Key details:**" in resp.response_text
    assert "**Official sources:**" in resp.response_text
    assert "https://www.dsu.edu.in/eligibility" in resp.response_text
    # No callback promises
    assert "call you back" not in resp.response_text.lower()


def test_query_2_bca_eligibility_and_annual_fee(canonical_data):
    """BCA eligibility & fee: Both verified (INR 2,07,000, 45% aggregate), no form needed."""
    intents_data, college_data, faq_data = canonical_data
    handler = GroqLLMHandler(api_key=None, client=None)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=handler,
    )
    resp = pipeline.process_query("What is the eligibility and annual fee for BCA?")
    assert "207,000" in resp.response_text or "2,07,000" in resp.response_text
    assert "45%" in resp.response_text
    assert resp.show_enquiry_form is False
    assert "**Answer:**" in resp.response_text
    assert "**Key details:**" in resp.response_text
    assert "https://www.dsu.edu.in/eligibility" in resp.response_text


def test_query_3_bca_hostel_fee_boys_and_mess(canonical_data):
    """BCA hostel fee for boys & mess: Official 1575-hostel-fee source, mess included, no faculty pages."""
    intents_data, college_data, faq_data = canonical_data
    handler = GroqLLMHandler(api_key=None, client=None)
    retriever = DSURetriever()
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=handler,
        retriever=retriever,
    )
    # Check retriever doesn't return faculty pages for hostel query
    ev = retriever.search("BCA hostel fee for boys and does it include mess?")
    assert any("hostel-fee" in e.url for e in ev)
    assert not any("mohd-adil" in e.url for e in ev)
    assert not any("/faculty/" in e.url for e in ev)

    resp = pipeline.process_query("What is the BCA hostel fee for boys and does it include mess?")
    assert "mess" in resp.response_text.lower()
    assert "hostel" in resp.response_text.lower()
    assert "1575-hostel-fee" in resp.response_text or "hostel" in resp.source.lower()
    assert "call you back" not in resp.response_text.lower()


def test_query_4_bca_management_quota_fee(canonical_data):
    """BCA management quota fee: Quota protection, no general tuition fee substitution."""
    intents_data, college_data, faq_data = canonical_data
    handler = GroqLLMHandler(api_key=None, client=None)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=handler,
    )
    resp = pipeline.process_query("What is the BCA management quota fee for 2026–27?")
    assert "admissions cell" in resp.response_text.lower()
    assert "management quota" in resp.response_text.lower()
    assert resp.show_enquiry_form is True
    assert resp.form_policy == "always"
    assert "call you back" not in resp.response_text.lower()


def test_query_5_bsc_data_science_placement_packages(canonical_data):
    """B.Sc. Data Science placements: 30+ LPA highest, 90%+ record, average package escalated."""
    intents_data, college_data, faq_data = canonical_data
    handler = GroqLLMHandler(api_key=None, client=None)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=handler,
    )
    resp = pipeline.process_query("What is the B.Sc. Data Science average package, highest package, and placement percentage?")
    assert "30+ lpa" in resp.response_text.lower() or "30" in resp.response_text
    assert "90%" in resp.response_text
    assert "average package" in resp.response_text.lower()
    assert resp.show_enquiry_form is True
    assert "overview-placements" in resp.response_text or "placement" in resp.response_text.lower()
    assert "call you back" not in resp.response_text.lower()


def test_query_6_programme_comparison_bca_and_bsc_data_science(canonical_data):
    """Compare BCA and B.Sc. Data Science on duration, eligibility, and fees."""
    intents_data, college_data, faq_data = canonical_data
    handler = GroqLLMHandler(api_key=None, client=None)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=handler,
    )
    resp = pipeline.process_query("Compare BCA and B.Sc. Data Science on duration, eligibility and fees.")
    assert "207,000" in resp.response_text or "2,07,000" in resp.response_text
    assert "162,000" in resp.response_text or "1,62,000" in resp.response_text
    assert "45%" in resp.response_text
    assert "50%" in resp.response_text
    assert "3 years" in resp.response_text.lower() or "3-year" in resp.response_text.lower()
    assert "https://www.dsu.edu.in/computer-applications/bca" in resp.response_text
    assert "https://www.dsu.edu.in/computer-applications/bsc-data-science" in resp.response_text
    assert "**Answer:**" in resp.response_text
    assert "**Key details:**" in resp.response_text


def test_query_7_dean_and_leadership_sca(canonical_data):
    """Dean of School of Computer Applications: Dr. S. Senthil verified."""
    intents_data, college_data, faq_data = canonical_data
    handler = GroqLLMHandler(api_key=None, client=None)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=handler,
    )
    resp = pipeline.process_query("Who is the Dean of the School of Computer Applications at DSU, and what is their official designation?")
    assert "senthil" in resp.response_text.lower()
    assert "dean" in resp.response_text.lower()
    assert "dean-msg-sca" in resp.response_text or "computer-applications" in resp.response_text


def test_query_8_fee_components_breakdown(canonical_data):
    """Fee types comparison: tuition vs hostel vs mess vs transport distinct explanation."""
    intents_data, college_data, faq_data = canonical_data
    handler = GroqLLMHandler(api_key=None, client=None)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=handler,
    )
    resp = pipeline.process_query("Can you explain the difference between tuition fee, hostel fee, mess fee, and transport fee at DSU?")
    assert "which course are you interested in" not in resp.response_text.lower()
    assert "tuition" in resp.response_text.lower()
    assert "hostel" in resp.response_text.lower()
    assert "mess" in resp.response_text.lower()
    assert "transport" in resp.response_text.lower()


def test_query_9_scholarships_and_concessions(canonical_data):
    """Scholarships available at DSU: DSAT, Defence, Sports."""
    intents_data, college_data, faq_data = canonical_data
    handler = GroqLLMHandler(api_key=None, client=None)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=handler,
    )
    resp = pipeline.process_query("Are there scholarships available for B.Sc. Data Science students at DSU, and what are the eligibility criteria and benefits?")
    assert "bachelor of computer applications (bca) is a 3-year" not in resp.response_text.lower()
    assert any(k in resp.response_text.lower() for k in ["dsat", "merit", "defence", "sports", "concession"])
    assert "scholarship" in resp.response_text.lower()


def test_query_10_mec_stream_and_admissions_documents(canonical_data):
    """12th MEC with 60% eligible for BCA and documents list."""
    intents_data, college_data, faq_data = canonical_data
    handler = GroqLLMHandler(api_key=None, client=None)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=handler,
    )
    resp = pipeline.process_query("I completed 12th in MEC stream with 60% marks. Am I eligible for BCA at DSU, and what documents should I prepare?")
    assert any(k in resp.response_text.lower() for k in ["eligible", "yes", "comfortably", "fulfill", "meets"])
    assert any(k in resp.response_text.lower() for k in ["marks card", "transfer certificate", "tc", "migration", "aadhaar", "photograph"])


def test_query_11_four_turn_bca_conversation_sequence(canonical_data):
    """Four-turn conversational sequence retaining programme context across turns."""
    intents_data, college_data, faq_data = canonical_data
    handler = GroqLLMHandler(api_key=None, client=None)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=handler,
    )
    session_id = "test_turn_seq_session_101"

    # Turn 1: Overview
    t1 = pipeline.process_query("Tell me about the BCA programme at DSU.", session_id=session_id)
    assert "bca" in t1.response_text.lower() or "bachelor of computer applications" in t1.response_text.lower()
    assert t1.context_slots.get("course_id") == "sca_bca"

    # Turn 2: Follow-up fee inquiry without mentioning BCA
    t2 = pipeline.process_query("What is the annual fee?", session_id=session_id)
    assert "207,000" in t2.response_text or "2,07,000" in t2.response_text

    # Turn 3: Follow-up hostel inquiry
    t3 = pipeline.process_query("What about hostel facilities?", session_id=session_id)
    assert "hostel" in t3.response_text.lower()
    assert "harohalli" in t3.response_text.lower()

    # Turn 4: Application process
    t4 = pipeline.process_query("How do I apply for admission?", session_id=session_id)
    assert any(k in t4.response_text.lower() for k in ["admissions.dsu.edu.in", "online application", "apply"])


def test_query_12_unseen_paraphrases_and_typos(canonical_data):
    """Handles misspelled and heavily paraphrased academic queries."""
    intents_data, college_data, faq_data = canonical_data
    handler = GroqLLMHandler(api_key=None, client=None)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=handler,
    )
    # Typo query
    r1 = pipeline.process_query("wht is eligiblity and anuel fees for bsc data scince?")
    assert any(f in r1.response_text for f in ["162,000", "1,62,000", "87,000", "75,000", "172,000"])
    assert "eligibility" in r1.response_text.lower()

    # Paraphrased query
    r2 = pipeline.process_query("Give me the breakdown of residential room charges and meals for men.", session_id="session_typo_2")
    assert "hostel" in r2.response_text.lower()
    assert "mess" in r2.response_text.lower()


def test_query_13_irrelevant_and_unrelated_queries(canonical_data):
    """Out-of-domain queries direct to administration/enquiry form without fabricating policy."""
    intents_data, college_data, faq_data = canonical_data
    handler = GroqLLMHandler(api_key=None, client=None)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=handler,
    )
    r = pipeline.process_query("Can I bring my pet dog to the DSU library?")
    assert any(k in r.response_text.lower() for k in ["enquiry form", "admissions", "administration", "official records", "contact"])


