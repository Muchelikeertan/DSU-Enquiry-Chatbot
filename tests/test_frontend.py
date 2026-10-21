"""Tests for Frontend Homepage and Chatbot Integration (Stage 5).

Covers:
1. Root GET / serves the authentic DSU homepage with HTTP 200 OK.
2. Static assets (/static/css/..., /static/js/...) are correctly served with appropriate MIME types.
3. Existing API endpoints (/health, /chat, /enquiries) remain completely functional.
4. Chat widget backend routes support query processing, quick replies, and enquiry form triggers.
5. Enquiry form submission validation and persistence with status 'processing'.
6. Recreated homepage verifies authentic DSU details (CET codes, helplines, campuses, schools).
7. Mobile responsiveness meta tags and widget controls (launcher, minimize, close).
"""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base, Enquiry, ChatInteractionLog, get_db
from app.main import app, get_pipeline
from app.nlp.pipeline import ChatbotNLPResponse, ExtractedEntities


@pytest.fixture(scope="module")
def test_engine():
    """In-memory SQLite database for isolated test execution."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def db_session(test_engine):
    """Provides isolated test session per test with clean rollback."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.query(Enquiry).delete()
        session.query(ChatInteractionLog).delete()
        session.commit()
        session.close()


@pytest.fixture
def mock_pipeline():
    """Mocked NLP Pipeline returning structured test answers."""
    mock = MagicMock()
    mock.process_query.return_value = ChatbotNLPResponse(
        response_text="The annual tuition fee for BCA at Dayananda Sagar University is INR 2,07,000.",
        intent="sca_fees",
        confidence=0.96,
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
    """FastAPI TestClient with dependency overrides."""
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
# 1. Homepage & Static Asset Tests
# ==============================================================================

def test_homepage_serves_http_200(client):
    """Verifies that GET / serves the DSU homepage with HTTP 200 and HTML content."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Dayananda Sagar University" in response.text
    assert 'id="dsu-chatbot-launcher"' in response.text
    assert 'id="dsu-chatbot-panel"' in response.text


def test_static_css_and_js_served(client):
    """Verifies that custom CSS and JS assets are successfully served."""
    css_dsu = client.get("/static/css/dsu-style.css")
    assert css_dsu.status_code == 200
    assert "text/css" in css_dsu.headers.get("content-type", "")

    css_chat = client.get("/static/css/chatbot.css")
    assert css_chat.status_code == 200
    assert "text/css" in css_chat.headers.get("content-type", "")

    js_home = client.get("/static/js/dsu-homepage.js")
    assert js_home.status_code == 200
    assert "javascript" in js_home.headers.get("content-type", "")

    js_chat = client.get("/static/js/chatbot.js")
    assert js_chat.status_code == 200
    assert "javascript" in js_chat.headers.get("content-type", "")


# ==============================================================================
# 2. Existing API Endpoints Functionality
# ==============================================================================

def test_api_health_endpoint_still_functional(client):
    """Verifies that GET /health remains fully functional alongside static routes."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "database" in data
    assert "nlp_pipeline" in data


def test_api_chat_endpoint_still_functional(client):
    """Verifies that POST /chat continues to process messages and return schema."""
    payload = {"message": "What is the fee for BCA?", "session_id": "test_frontend_session"}
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "INR 2,07,000" in data["response_text"]
    assert data["intent"] == "sca_fees"
    assert data["show_enquiry_form"] is False
    assert data["source"] == "https://www.dsu.edu.in/eligibility"


# ==============================================================================
# 3. Chatbot Widget Integration & Enquiry Trigger
# ==============================================================================

def test_chat_widget_receives_enquiry_form_trigger(client, mock_pipeline):
    """Verifies widget endpoint receives form trigger and prefill metadata."""
    mock_pipeline.process_query.return_value = ChatbotNLPResponse(
        response_text="I could not find verified information for this inquiry. Please submit an admission enquiry form.",
        intent="unverified_query",
        confidence=0.45,
        handler="groq_llm",
        entities=ExtractedEntities(course_id=None, school_id="computer_applications"),
        context_slots={"school_id": "computer_applications"},
        show_enquiry_form=True,
        form_policy="always",
        form_prefill={
            "course_name": "Bachelor of Computer Applications (BCA)",
            "school_or_department": "School of Computer Applications",
            "question": "What are the lateral entry requirements?",
        },
        source=None,
        status="unverified_fallback",
        used_llm=True,
    )

    payload = {"message": "What are the lateral entry requirements for BCA?"}
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["show_enquiry_form"] is True
    assert data["form_policy"] == "always"
    assert data["form_prefill"]["course_name"] == "Bachelor of Computer Applications (BCA)"
    assert data["form_prefill"]["school_or_department"] == "School of Computer Applications"


def test_enquiry_submission_from_widget(client, db_session):
    """Verifies that student enquiry submissions from the widget persist to DB."""
    enquiry_payload = {
        "student_name": "Siddharth Verma",
        "email": "siddharth.verma@example.com",
        "phone_number": "9876543210",
        "school_or_department": "School of Computer Applications",
        "course_name": "Bachelor of Computer Applications (BCA)",
        "question": "Interested in 2026 admissions and hostel availability.",
        "session_id": "test_widget_session_123",
    }

    response = client.post("/enquiries", json=enquiry_payload)
    assert response.status_code == 201
    data = response.json()
    assert data["id"] > 0
    assert data["student_name"] == "Siddharth Verma"
    assert data["status"] == "processing"

    # Verify persistence in database session
    saved_enquiry = db_session.query(Enquiry).filter_by(id=data["id"]).first()
    assert saved_enquiry is not None
    assert saved_enquiry.email == "siddharth.verma@example.com"
    assert saved_enquiry.course_name == "Bachelor of Computer Applications (BCA)"


def test_enquiry_submission_validation_error(client):
    """Verifies that invalid enquiry data returns HTTP 422 Unprocessable Entity."""
    invalid_payload = {
        "student_name": "",  # Empty name rejected
        "email": "invalid-email-format",
        "question": "Test query",
    }
    response = client.post("/enquiries", json=invalid_payload)
    assert response.status_code == 422


# ==============================================================================
# 4. Verified DSU Content & Layout Fidelity in HTML
# ==============================================================================

def test_homepage_contains_verified_dsu_content(client):
    """Verifies that HTML contains authentic, non-fabricated DSU content."""
    response = client.get("/")
    html = response.text

    # Statutory Codes
    assert "CET Code (B.Tech): <strong>E240</strong>" in html
    assert "Comed-K Code: <strong>E182</strong>" in html
    assert "PGCET Code (M.Tech): <strong>T970</strong>" in html

    # Verified Helplines
    assert "080 46461800" in html
    assert "+91 6366885507" in html
    assert "+91 9606022149" in html
    assert "admissions@dsu.edu.in" in html

    # Verified Campuses
    assert "Devarakaggalahalli, Harohalli, Kanakapura Road" in html
    assert "Kudlu Gate, Hosur Road" in html
    assert "Kumaraswamy Layout" in html

    # Key Schools
    assert "School of Engineering" in html
    assert "School of Computer Applications" in html
    assert "School of Law" in html
    assert "Commerce &amp; Management" in html

    # Leadership
    assert "Late Shri R. Dayananda Sagar" in html
    assert "Late Dr. Chandramma Sagar" in html
    assert "Dr. D. Hemachandra Sagar" in html
    assert "Dr. D. Premachandra Sagar" in html


def test_responsive_meta_and_widget_controls(client):
    """Verifies mobile viewport meta and accessible widget controls."""
    response = client.get("/")
    html = response.text

    # Viewport for responsive design
    assert '<meta name="viewport" content="width=device-width, initial-scale=1.0"' in html

    # Chatbot widget controls
    assert 'id="chat-close-btn"' in html
    assert 'id="chat-minimize-btn"' in html
    assert 'id="chat-input"' in html
    assert 'id="chat-send-btn"' in html
    assert 'id="mobile-nav-toggle"' in html
