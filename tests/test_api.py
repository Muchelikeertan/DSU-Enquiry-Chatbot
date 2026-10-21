"""Integration and Unit Tests for FastAPI Application and Database Integration (Stage 4).

Covers:
1. GET  /health probe without secret leakage
2. POST /chat query handling with schema validation and telemetry logging
3. POST /chat validation on empty or whitespace messages
4. POST /enquiries validation and database persistence with status lifecycle
5. Database failure and unavailable service error handling
6. Prevention of unauthenticated admin endpoints
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.db import Base, Enquiry, ChatInteractionLog, get_db
from app.main import app, get_pipeline
from app.nlp.pipeline import ChatbotNLPResponse, ExtractedEntities


@pytest.fixture(scope="module")
def test_engine():
    """Isolated in-memory SQLite database for test persistence."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def db_session(test_engine):
    """Provides an isolated database session per test with clean rollback."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        # Clean data between tests
        session.query(Enquiry).delete()
        session.query(ChatInteractionLog).delete()
        session.commit()
        session.close()


@pytest.fixture
def mock_pipeline():
    """Mocked NLPPipeline for isolated API testing."""
    mock = MagicMock()
    mock.process_query.return_value = ChatbotNLPResponse(
        response_text="The annual fee for BCA is INR 2,07,000.",
        intent="sca_fees",
        confidence=0.95,
        handler="local",
        entities=ExtractedEntities(course_id="sca_bca", school_id="computer_applications"),
        context_slots={"course_id": "sca_bca"},
        show_enquiry_form=False,
        form_policy="never",
        form_prefill={},
        source="https://www.dsu.edu.in/eligibility",
        status="source_checked",
        used_llm=False,
    )
    return mock


@pytest.fixture
def client(test_engine, db_session, mock_pipeline):
    """FastAPI TestClient with dependency overrides for database and NLP pipeline."""
    import app.main as main_mod
    original_pipeline = main_mod._pipeline_instance
    main_mod._pipeline_instance = mock_pipeline

    def override_get_db():
        yield db_session

    def override_get_pipeline():
        return mock_pipeline

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_pipeline] = override_get_pipeline

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    main_mod._pipeline_instance = original_pipeline


# ==============================================================================
# 1. Health Probe Endpoint Tests
# ==============================================================================

def test_health_check_endpoint(client):
    """Health check must return 200 with status info and zero secret leakage."""
    resp = client.get("/health")
    assert resp.status_code == 200

    data = resp.json()
    assert data["status"] == "ok"
    assert data["app_name"] == settings.app_name
    assert data["environment"] == settings.app_env
    assert data["nlp_pipeline"] == "loaded"
    assert data["database"] == "connected"

    # Verify no credentials or API keys exposed
    raw_str = resp.text.lower()
    assert "password" not in raw_str
    assert "secret" not in raw_str
    assert "gsk_" not in raw_str


def test_health_check_database_not_connected():
    """Health check when DB session yields None and connection fails must report not_configured or disconnected."""
    def override_no_db():
        yield None

    app.dependency_overrides[get_db] = override_no_db
    with patch("app.main.check_db_connection", return_value=False):
        try:
            with TestClient(app) as test_client:
                resp = test_client.get("/health")
                assert resp.status_code == 200
                data = resp.json()
                assert data["database"] in ("not_configured", "disconnected")
        finally:
            app.dependency_overrides.clear()


# ==============================================================================
# 2. Chat Processing Endpoint Tests
# ==============================================================================

def test_chat_endpoint_valid_query(client, mock_pipeline, db_session):
    """Valid chat message must return 200 with structured ChatResponse and telemetry log."""
    payload = {
        "message": "What is the annual fee for BCA?",
        "session_id": "test_session_101",
    }
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert data["response_text"] == "The annual fee for BCA is INR 2,07,000."
    assert data["intent"] == "sca_fees"
    assert data["confidence"] == 0.95
    assert data["show_enquiry_form"] is False
    assert data["form_policy"] == "never"
    assert data["session_id"] == "test_session_101"

    mock_pipeline.process_query.assert_called_once_with(
        "What is the annual fee for BCA?",
        session_id="test_session_101",
    )

    # Verify telemetry interaction log was recorded without user message text
    logs = db_session.query(ChatInteractionLog).all()
    assert len(logs) == 1
    assert logs[0].session_id == "test_session_101"
    assert logs[0].query_intent == "sca_fees"
    assert logs[0].show_enquiry_form is False
    # Ensure personal message content was NOT stored in the database
    assert not hasattr(logs[0], "message_text")


def test_chat_endpoint_unverified_triggers_form(client, mock_pipeline):
    """Query returning unverified status must show enquiry form with policy always."""
    mock_pipeline.process_query.return_value = ChatbotNLPResponse(
        response_text="Kudlu Gate hostel accommodation is unverified. Please fill out the enquiry form.",
        intent="hostel_facilities",
        confidence=0.88,
        handler="local_then_llm",
        entities=ExtractedEntities(),
        context_slots={},
        show_enquiry_form=True,
        form_policy="always",
        form_prefill={"school_or_department": "computer_applications"},
        source="Groq LLM (openai/gpt-oss-120b)",
        status="source_checked_partial",
        used_llm=True,
    )

    payload = {"message": "Is hostel available at Kudlu Gate?"}
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert data["show_enquiry_form"] is True
    assert data["form_policy"] == "always"
    assert data["form_prefill"]["school_or_department"] == "computer_applications"
    assert data["used_llm"] is True


def test_chat_endpoint_validation_empty_and_whitespace(client):
    """Empty or whitespace-only messages must be rejected with 422 Unprocessable Entity."""
    # Empty string
    r1 = client.post("/chat", json={"message": ""})
    assert r1.status_code == 422

    # Whitespace only
    r2 = client.post("/chat", json={"message": "    \t\n  "})
    assert r2.status_code == 422

    # Missing message field
    r3 = client.post("/chat", json={"session_id": "sess_1"})
    assert r3.status_code == 422


# ==============================================================================
# 3. Enquiry Form Submission Endpoint Tests
# ==============================================================================

def test_enquiry_submission_success(client, db_session):
    """Valid enquiry form submission must be saved to database with status processing."""
    payload = {
        "student_name": "Keerthi Kumar",
        "email": "keerthi@example.com",
        "phone_number": "+91 9876543210",
        "school_or_department": "School of Computer Applications",
        "course_name": "BCA in AI and Data Science",
        "question": "Could you please confirm if hostel facilities are available at Kudlu Gate campus?",
        "session_id": "session_enq_001",
    }
    resp = client.post("/enquiries", json=payload)
    assert resp.status_code == 201

    data = resp.json()
    assert data["id"] is not None
    assert data["student_name"] == "Keerthi Kumar"
    assert data["email"] == "keerthi@example.com"
    assert data["phone_number"] == "+91 9876543210"
    assert data["school_or_department"] == "School of Computer Applications"
    assert data["course_name"] == "BCA in AI and Data Science"
    assert data["status"] == "processing"
    assert "created_at" in data

    # Verify direct database persistence
    saved = db_session.query(Enquiry).filter_by(id=data["id"]).first()
    assert saved is not None
    assert saved.student_name == "Keerthi Kumar"
    assert saved.email == "keerthi@example.com"
    assert saved.status == "processing"
    assert saved.session_id == "session_enq_001"


def test_enquiry_submission_validation_errors(client):
    """Invalid enquiry inputs must be rejected with 422 validation errors."""
    valid_base = {
        "student_name": "Ananya Rao",
        "email": "ananya@example.com",
        "question": "What is the fee structure?",
    }

    # Invalid email format
    bad_email = {**valid_base, "email": "not-an-email-address"}
    r1 = client.post("/enquiries", json=bad_email)
    assert r1.status_code == 422

    # Empty student_name
    empty_name = {**valid_base, "student_name": "   "}
    r2 = client.post("/enquiries", json=empty_name)
    assert r2.status_code == 422

    # Empty question
    empty_q = {**valid_base, "question": ""}
    r3 = client.post("/enquiries", json=empty_q)
    assert r3.status_code == 422

    # Missing question
    missing_q = {"student_name": "Ananya", "email": "ananya@example.com"}
    r4 = client.post("/enquiries", json=missing_q)
    assert r4.status_code == 422


def test_enquiry_submission_db_unavailable(client):
    """When database session is unavailable, return 503 Service Unavailable."""
    def override_no_db():
        yield None

    app.dependency_overrides[get_db] = override_no_db

    payload = {
        "student_name": "Rahul Verma",
        "email": "rahul@example.com",
        "question": "When does the MCA batch start?",
    }
    resp = client.post("/enquiries", json=payload)
    assert resp.status_code == 503
    assert "unavailable" in resp.json()["detail"].lower()


def test_enquiry_submission_database_error_handling(client, db_session):
    """Database exceptions during commit must return 500 without credential leakage."""
    payload = {
        "student_name": "Rahul Verma",
        "email": "rahul@example.com",
        "question": "When does the MCA batch start?",
    }
    with patch.object(db_session, "commit", side_effect=SQLAlchemyError("Simulated DB connection lost")):
        resp = client.post("/enquiries", json=payload)
        assert resp.status_code == 500
        assert "failed to save" in resp.json()["detail"].lower()
        # Confirm no database connection strings or internals in the response
        assert "sqlalchemy" not in resp.text.lower()
        assert "simulated" not in resp.text.lower()


def test_security_no_unauthenticated_admin_endpoints(client):
    """Ensure no unauthenticated admin deletion or modification endpoints exist."""
    del_resp = client.delete("/enquiries/1")
    assert del_resp.status_code in (404, 405)

    put_resp = client.put("/enquiries/1", json={"status": "completed"})
    assert put_resp.status_code in (404, 405)

    admin_resp = client.get("/admin/enquiries")
    assert admin_resp.status_code in (404, 405)
