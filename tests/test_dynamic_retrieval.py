"""Automated Test Suite for Dynamic Knowledge Retrieval & General LLM Fallback.

Validates all 13 project-wide criteria:
1. Question answered sufficiently by JSON uses local response without LLM.
2. Question absent from JSON invokes dynamic retrieval and LLM fallback.
3. Incomplete local answer triggers retrieval and LLM generation.
4. Different unseen topics share the same universal fallback pipeline.
5. Relevant official evidence and DSU URLs reach the LLM grounding context.
6. Generated answers contain valid, verified clickable official source links.
7. Unsupported factual claims are never invented; limitations are stated honestly.
8. Ambiguous queries trigger clarification when necessary.
9. Conflicting sources are handled transparently.
10. Retrieval failures produce a safe, user-friendly fallback with enquiry form.
11. Groq errors and timeouts do not crash the application.
12. Follow-up questions leverage multi-turn conversation history.
13. Existing enquiry submission, DB integration, and API contracts remain intact.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import groq

from app.llm.handler import GroqLLMHandler, LLMResponse
from app.nlp.pipeline import NLPPipeline
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
def mock_retriever():
    """Mock retriever returning realistic verified official DSU evidence."""
    retriever = MagicMock(spec=DSURetriever)
    retriever.search.return_value = [
        RetrievedEvidence(
            title="DSU Placements & Career Cell",
            url="https://www.dsu.edu.in/placements",
            snippet="Dayananda Sagar University records top placement packages with 200+ recruiters.",
            content=(
                "Dayananda Sagar University Centre for Innovation & Placements reports top CTC of 32 LPA.\n"
                "Major recruiters include Amazon, Microsoft, Infosys, Capgemini, and Accenture.\n"
                "Dedicated training in technical aptitude and mock interviews is provided."
            ),
            relevance_score=0.92,
        )
    ]
    return retriever


@pytest.fixture
def mock_groq():
    """Mock Groq client returning structured grounded response."""
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = (
        "**Answer:** Dayananda Sagar University records excellent placement outcomes with top packages up to 32 LPA.\n\n"
        "**Key details:**\n"
        "- Over 200 top recruiters visit campus including Amazon, Microsoft, and Infosys.\n"
        "- Dedicated Centre for Innovation & Placements provides pre-placement training.\n\n"
        "**Official sources:**\n"
        "- [DSU Placements](https://www.dsu.edu.in/placements)"
    )
    resp = MagicMock()
    resp.choices = [choice]
    client.chat.completions.create.return_value = resp
    return client


# ==============================================================================
# Scenario 1: Sufficient Local JSON Query Uses Local Response
# ==============================================================================
def test_question_sufficient_by_json_uses_local_response(canonical_data, mock_groq, mock_retriever):
    """Questions fully answered by verified local FAQ must NOT call LLM or dynamic search."""
    intents_data, college_data, faq_data = canonical_data
    llm = GroqLLMHandler(api_key="mock_key", client=mock_groq)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=llm,
        retriever=mock_retriever,
    )

    resp = pipeline.process_query("What is the fee for BCA?")
    assert not resp.used_llm
    assert "207,000" in resp.response_text or "2,07,000" in resp.response_text
    assert resp.status == "source_checked"
    mock_groq.chat.completions.create.assert_not_called()
    mock_retriever.search.assert_not_called()


# ==============================================================================
# Scenario 2: Question Absent from JSON Invokes LLM Fallback
# ==============================================================================
def test_question_absent_from_json_invokes_llm_fallback(canonical_data, mock_groq, mock_retriever):
    """An unseen question absent from JSON files must invoke dynamic retrieval and LLM."""
    intents_data, college_data, faq_data = canonical_data
    llm = GroqLLMHandler(api_key="mock_key", client=mock_groq)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=llm,
        retriever=mock_retriever,
    )

    resp = pipeline.process_query("What is the highest salary package in DSU placements?")
    assert resp.used_llm
    assert mock_retriever.search.called
    assert mock_groq.chat.completions.create.called
    assert "32 LPA" in resp.response_text
    assert "https://www.dsu.edu.in/placements" in resp.source


# ==============================================================================
# Scenario 3: Incomplete Local Answer Triggers Dynamic Retrieval & LLM
# ==============================================================================
def test_incomplete_local_answer_triggers_retrieval_and_llm(canonical_data, mock_groq, mock_retriever):
    """Non-SCA or partially covered topics trigger dynamic retrieval to complete the answer."""
    intents_data, college_data, faq_data = canonical_data
    llm = GroqLLMHandler(api_key="mock_key", client=mock_groq)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=llm,
        retriever=mock_retriever,
    )

    resp = pipeline.process_query("What are the scholarship criteria and placements for BTech Aerospace Engineering?")
    assert resp.used_llm
    assert mock_retriever.search.called
    assert mock_groq.chat.completions.create.called


# ==============================================================================
# Scenario 4: Multiple Unseen Topics Share Reusable Universal Fallback
# ==============================================================================
@pytest.mark.parametrize("query,expected_keyword", [
    # These queries use topics that are NOT answered by local FAQ,
    # so the universal LLM fallback is triggered for all of them.
    ("Who is the vice chancellor of DSU?", "leadership"),
    ("What courses does the School of Law offer at DSU?", "law"),
    ("What are the research labs available at DSU?", "research"),
    ("Tell me about the NVIDIA AI factory partnership at DSU", "nvidia"),
])
def test_different_unseen_topics_share_reusable_fallback(canonical_data, mock_groq, mock_retriever, query, expected_keyword):
    """Verifies that diverse unseen topics use the same general LLM fallback without hardcoded rules."""
    intents_data, college_data, faq_data = canonical_data
    llm = GroqLLMHandler(api_key="mock_key", client=mock_groq)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=llm,
        retriever=mock_retriever,
    )

    resp = pipeline.process_query(query)
    assert resp.used_llm, (
        f"Expected LLM fallback for query '{query}' but got intent='{resp.intent}' "
        f"handler='{resp.handler}' used_llm={resp.used_llm}"
    )
    assert mock_groq.chat.completions.create.called


# ==============================================================================
# Scenario 5: Relevant Official Evidence and URLs Reach the LLM
# ==============================================================================
def test_relevant_official_evidence_and_urls_reach_llm(canonical_data, mock_groq, mock_retriever):
    """The grounding prompt sent to Groq must include the extracted official evidence and URLs."""
    intents_data, college_data, faq_data = canonical_data
    llm = GroqLLMHandler(api_key="mock_key", client=mock_groq)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=llm,
        retriever=mock_retriever,
    )

    # Use a query that is absent from local FAQ so it routes through the LLM fallback
    pipeline.process_query("Who is the vice chancellor of Dayananda Sagar University?")

    # Inspect the system prompt sent to Groq
    call_args = mock_groq.chat.completions.create.call_args
    assert call_args is not None, "Expected Groq to be called for a query absent from local FAQ"
    messages = call_args[1]["messages"]
    system_content = messages[0]["content"]

    # The mock_retriever returns placement evidence regardless of query —
    # verify that its content (URL, title, snippet) was injected into the Groq grounding context
    assert "OFFICIAL RETRIEVED DSU SOURCES & EVIDENCE" in system_content
    assert "https://www.dsu.edu.in/placements" in system_content
    assert "DSU Placements" in system_content or "32 LPA" in system_content


# ==============================================================================
# Scenario 6: Generated Answer Contains Valid Source Links
# ==============================================================================
def test_generated_answer_contains_valid_source_links(canonical_data, mock_groq, mock_retriever):
    """Response text must contain clickable markdown links pointing to official DSU domains."""
    intents_data, college_data, faq_data = canonical_data
    llm = GroqLLMHandler(api_key="mock_key", client=mock_groq)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=llm,
        retriever=mock_retriever,
    )

    resp = pipeline.process_query("Tell me about DSU recruiters")
    assert "[DSU Placements](https://www.dsu.edu.in/placements)" in resp.response_text
    assert resp.source.startswith("https://www.dsu.edu.in")


# ==============================================================================
# Scenario 7: Unsupported Information is Not Invented
# ==============================================================================
def test_unsupported_information_not_invented(canonical_data, mock_retriever):
    """When records do not confirm a claim, handler states limitation and triggers enquiry form."""
    intents_data, college_data, faq_data = canonical_data
    mock_client = MagicMock()
    choice = MagicMock()
    choice.message.content = (
        "**Answer:** Hostel accommodation is officially verified for the Harohalli Main Campus only. "
        "Hostel facilities at Kudlu Gate campus are not confirmed in official university records.\n\n"
        "Please submit an enquiry form for clarification from the hostel warden."
    )
    resp_obj = MagicMock()
    resp_obj.choices = [choice]
    mock_client.chat.completions.create.return_value = resp_obj

    llm = GroqLLMHandler(api_key="mock_key", client=mock_client)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=llm,
        retriever=mock_retriever,
    )

    resp = pipeline.process_query("Is there a boys hostel at Kudlu Gate campus?")
    assert resp.show_enquiry_form
    assert "not confirmed" in resp.response_text.lower() or "unverified" in resp.response_text.lower()


# ==============================================================================
# Scenario 8: Ambiguous Queries Trigger Clarification
# ==============================================================================
def test_ambiguous_queries_trigger_clarification(canonical_data, mock_groq, mock_retriever):
    """Course-dependent query without naming a course triggers ask_which_course clarification."""
    intents_data, college_data, faq_data = canonical_data
    llm = GroqLLMHandler(api_key="mock_key", client=mock_groq)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=llm,
        retriever=mock_retriever,
    )

    resp = pipeline.process_query("What is the fee?")
    assert resp.intent == "ask_which_course"
    assert "Which course" in resp.response_text
    assert len(resp.quick_replies) > 0
    mock_groq.chat.completions.create.assert_not_called()


# ==============================================================================
# Scenario 9: Conflicting Sources Handled Honestly
# ==============================================================================
def test_conflicting_sources_handled_honestly(canonical_data, mock_retriever):
    """When external aggregators conflict with official records, the model does not guess."""
    intents_data, college_data, faq_data = canonical_data
    mock_client = MagicMock()
    choice = MagicMock()
    choice.message.content = (
        "**Answer:** Official DSU tuition fees depend on admission quota (CET/COMEDK/Management). "
        "Third-party website figures conflict and are not verified. Please submit an enquiry form."
    )
    resp_obj = MagicMock()
    resp_obj.choices = [choice]
    mock_client.chat.completions.create.return_value = resp_obj

    llm = GroqLLMHandler(api_key="mock_key", client=mock_client)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=llm,
        retriever=mock_retriever,
    )

    resp = pipeline.process_query("Some site said MBA fee is 5 lakhs, is that true?")
    assert resp.show_enquiry_form
    assert "third-party" in resp.response_text.lower() or "conflict" in resp.response_text.lower() or "enquiry" in resp.response_text.lower()


# ==============================================================================
# Scenario 10: Retrieval Failure Produces Safe Fallback
# ==============================================================================
def test_retrieval_failure_produces_safe_fallback(canonical_data, mock_groq):
    """When the dynamic retriever throws an exception or network error, pipeline handles safely."""
    intents_data, college_data, faq_data = canonical_data
    failing_retriever = MagicMock(spec=DSURetriever)
    failing_retriever.search.side_effect = ConnectionError("DSU server unreachable")

    llm = GroqLLMHandler(api_key="mock_key", client=mock_groq)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=llm,
        retriever=failing_retriever,
    )

    # Must not crash, should continue with LLM grounding from local records
    resp = pipeline.process_query("What are the upcoming campus festivals at DSU?")
    assert resp.response_text
    assert mock_groq.chat.completions.create.called


# ==============================================================================
# Scenario 11: Groq Errors and Timeouts Do Not Crash the Application
# ==============================================================================
def test_groq_timeout_does_not_crash_application(canonical_data, mock_retriever):
    """When Groq times out, application catches error safely and returns polite fallback."""
    intents_data, college_data, faq_data = canonical_data
    failing_client = MagicMock()
    failing_client.chat.completions.create.side_effect = groq.APITimeoutError(request=MagicMock())

    llm = GroqLLMHandler(api_key="mock_key", client=failing_client)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=llm,
        retriever=mock_retriever,
    )

    resp = pipeline.process_query("Tell me about NVIDIA AI technology centre")
    assert not resp.used_llm
    assert resp.needs_fallback
    assert resp.show_enquiry_form
    assert "unable to retrieve official dynamic details" in resp.response_text
    assert "APITimeoutError" not in resp.response_text


# ==============================================================================
# Scenario 12: Follow-up Questions Use Relevant Conversation History
# ==============================================================================
def test_follow_up_questions_use_conversation_history(canonical_data, mock_groq, mock_retriever):
    """Multi-turn follow-ups pass conversation history to LLM grounding context."""
    intents_data, college_data, faq_data = canonical_data
    llm = GroqLLMHandler(api_key="mock_key", client=mock_groq)
    pipeline = NLPPipeline(
        intents_data=intents_data,
        college_data=college_data,
        faq_data=faq_data,
        llm_handler=llm,
        retriever=mock_retriever,
    )

    session_id = "test_hist_session_99"

    # Turn 1: Ask about BTech Mechanical
    pipeline.process_query("Tell me about BTech in Mechanical Engineering", session_id=session_id)

    # Turn 2: Follow-up "What are the job opportunities for it?"
    pipeline.process_query("What are the job opportunities for it?", session_id=session_id)

    # Inspect call args for Turn 2
    call_args = mock_groq.chat.completions.create.call_args
    messages = call_args[1]["messages"]
    system_content = messages[0]["content"]

    assert "RECENT CONVERSATION CONTEXT" in system_content
    assert "Mechanical" in system_content


# ==============================================================================
# Scenario 13: Existing Enquiry Submission and DB Functionality Intact
# ==============================================================================
def test_existing_enquiry_submission_and_api_intact():
    """Verify FastAPI /enquiries endpoint and validation contracts remain 100% functional."""
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from app.main import app
    from app.db import Base, get_db

    # Create an isolated in-memory SQLite DB that satisfies the real endpoint's commit/refresh
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_db():
        session = TestSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)
        payload = {
            "student_name": "Keerthi Kumar",
            "email": "keerthi@example.com",
            "phone_number": "9876543210",
            "course_name": "B.Tech Computer Science & Engineering",
            "school_or_department": "School of Engineering",
            "question": "What are the admission dates for 2026-27?",
        }
        response = client.post("/enquiries", json=payload)
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 201
    data = response.json()
    # EnquiryResponse schema returns fields directly (no nested 'data' wrapper)
    assert data["status"] == "processing"   # EnquiryStatus.PROCESSING
    assert data["email"] == "keerthi@example.com"
    assert data["question"] == "What are the admission dates for 2026-27?"
    assert "id" in data

