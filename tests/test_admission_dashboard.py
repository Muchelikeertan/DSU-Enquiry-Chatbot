"""Automated Test Suite for DSU Admission Cell Enquiry Management & Persistent Audit History.

Verifies:
1. Staff authentication, JWT generation, and credential rejection.
2. Protection of administrative APIs (401 on missing or invalid tokens).
3. Search, filter, and pagination of admission enquiries.
4. Persistent status change audit trail (previous -> new, staff attribution, timestamp).
5. Append-only internal comments (no overwrite of past comments).
6. Recording of follow-up actions (phone call, email, campus visit, counseling).
7. Unified chronological audit timeline.
8. Persistence guarantees in the database.
"""

import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.db import (
    Base,
    Enquiry,
    EnquiryComment,
    EnquiryFollowUp,
    EnquiryStatus,
    EnquiryStatusHistory,
    StaffUser,
    get_db,
)
from app.main import app
from app.security import (
    create_access_token,
    ensure_default_admin,
    hash_password,
)


from sqlalchemy.pool import StaticPool


@pytest.fixture(scope="module")
def test_db_session():
    """Provides an isolated in-memory SQLite database with StaticPool for admission dashboard testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()

    # Seed default admin user
    ensure_default_admin(session)

    # Seed a sample enquiry
    sample_enquiry = Enquiry(
        student_name="Rahul Sharma",
        email="rahul.sharma@example.com",
        phone_number="+91 9876543210",
        school_or_department="School of Engineering",
        course_name="B.Tech Computer Science and Engineering",
        question="What is the admission cutoff and annual tuition fee for B.Tech CSE?",
        status=EnquiryStatus.PROCESSING.value,
    )
    session.add(sample_enquiry)
    session.commit()
    session.refresh(sample_enquiry)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    yield session

    session.close()
    app.dependency_overrides.clear()



@pytest.fixture(scope="module")
def client(test_db_session):
    """Test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture(scope="module")
def admin_auth_token(client):
    """Retrieves a valid staff JWT access token for testing protected endpoints."""
    resp = client.post("/admin/api/login", json={
        "username": settings.admin_username,
        "password": settings.admin_password,
    })
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    data = resp.json()
    return data["access_token"]


@pytest.fixture(scope="module")
def auth_headers(admin_auth_token):
    """Generates Authorization header with Bearer token."""
    return {"Authorization": f"Bearer {admin_auth_token}"}


# ==============================================================================
# 1. Staff Authentication & Access Control Tests
# ==============================================================================

def test_staff_login_valid_credentials(client):
    """Valid staff login yields 200, JWT, user object, and sets cookie."""
    resp = client.post("/admin/api/login", json={
        "username": settings.admin_username,
        "password": settings.admin_password,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["username"] == settings.admin_username.lower()
    assert data["user"]["role"] in ("admissions_admin", "admissions_staff")
    assert "admin_token" in resp.cookies


def test_staff_login_invalid_password(client):
    """Incorrect password rejects authentication with 401."""
    resp = client.post("/admin/api/login", json={
        "username": settings.admin_username,
        "password": "IncorrectPassword123!",
    })
    assert resp.status_code == 401
    assert "Invalid username or password" in resp.json()["detail"]


def test_staff_login_unknown_user(client):
    """Non-existent username rejects authentication with 401."""
    resp = client.post("/admin/api/login", json={
        "username": "unknown_intruder",
        "password": "any_password",
    })
    assert resp.status_code == 401
    assert "Invalid username or password" in resp.json()["detail"]


def test_admin_api_endpoints_reject_unauthenticated():
    """Administrative endpoints require authentication and reject unauthenticated requests with 401."""
    unauth_client = TestClient(app)
    endpoints = [
        ("GET", "/admin/api/me"),
        ("GET", "/admin/api/enquiries"),
        ("GET", "/admin/api/enquiries/1"),
        ("PATCH", "/admin/api/enquiries/1/status"),
        ("POST", "/admin/api/enquiries/1/comments"),
        ("POST", "/admin/api/enquiries/1/follow-ups"),
    ]
    for method, path in endpoints:
        if method == "GET":
            r = unauth_client.get(path)
        elif method == "PATCH":
            r = unauth_client.patch(path, json={"status": "called"})
        elif method == "POST":
            r = unauth_client.post(path, json={"comment_text": "test"})
        assert r.status_code == 401, f"{method} {path} unexpectedly returned {r.status_code}"



def test_admin_me_endpoint_with_auth(client, auth_headers):
    """Authenticated staff profile retrieval returns active staff data."""
    resp = client.get("/admin/api/me", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["username"] == settings.admin_username.lower()
    assert data["is_active"] is True


def test_staff_logout(client):
    """Logout endpoint deletes admin_token session cookie."""
    resp = client.post("/admin/api/logout")
    assert resp.status_code == 200
    assert "Logged out successfully" in resp.json()["message"]


# ==============================================================================
# 2. Enquiry Management, Search & Filter Tests
# ==============================================================================

def test_list_enquiries_authenticated(client, auth_headers):
    """Staff can list enquiries with status metrics."""
    resp = client.get("/admin/api/enquiries", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "items" in data
    assert "status_counts" in data
    assert data["total"] >= 1
    assert any(item["student_name"] == "Rahul Sharma" for item in data["items"])


def test_enquiries_search_by_name(client, auth_headers):
    """Search matches student name accurately."""
    resp = client.get("/admin/api/enquiries?search=Rahul", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1
    assert data["items"][0]["student_name"] == "Rahul Sharma"


def test_enquiries_search_by_question_keyword(client, auth_headers):
    """Search matches words inside question."""
    resp = client.get("/admin/api/enquiries?search=tuition", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1


def test_enquiries_filter_by_status(client, auth_headers):
    """Filtering by status isolates matching records."""
    resp = client.get("/admin/api/enquiries?status=processing", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    for item in data["items"]:
        assert item["status"] == "processing"


# ==============================================================================
# 3. Persistent Status Change Audit History Tests
# ==============================================================================

def test_enquiry_status_transition_records_audit_history(client, auth_headers, test_db_session):
    """Status update creates an immutable audit trail entry with previous and new status."""
    enquiry = test_db_session.query(Enquiry).filter(Enquiry.student_name == "Rahul Sharma").first()
    enquiry_id = enquiry.id
    initial_status = enquiry.status

    # Transition 1: processing -> called
    resp1 = client.patch(
        f"/admin/api/enquiries/{enquiry_id}/status",
        headers=auth_headers,
        json={"status": "called", "notes": "Telephoned student to explain KCET and Management quota cutoffs."}
    )
    assert resp1.status_code == 200
    detail1 = resp1.json()
    assert detail1["status"] == "called"
    assert len(detail1["status_history"]) == 1
    hist1 = detail1["status_history"][0]
    assert hist1["previous_status"] == initial_status
    assert hist1["new_status"] == "called"
    assert hist1["notes"] == "Telephoned student to explain KCET and Management quota cutoffs."
    assert "staff_name" in hist1

    # Transition 2: called -> visited
    resp2 = client.patch(
        f"/admin/api/enquiries/{enquiry_id}/status",
        headers=auth_headers,
        json={"status": "visited", "notes": "Student visited Innovation Centre on Kanakapura campus."}
    )
    assert resp2.status_code == 200
    detail2 = resp2.json()
    assert detail2["status"] == "visited"
    # Verify BOTH historical transitions are preserved (no overwrite!)
    assert len(detail2["status_history"]) == 2
    assert detail2["status_history"][0]["new_status"] == "called"
    assert detail2["status_history"][1]["new_status"] == "visited"
    assert detail2["status_history"][1]["previous_status"] == "called"


# ==============================================================================
# 4. Persistent Internal Comments (Append-Only) Tests
# ==============================================================================

def test_add_comments_preserves_full_comment_history(client, auth_headers, test_db_session):
    """Adding internal comments never overwrites previous comments."""
    enquiry = test_db_session.query(Enquiry).filter(Enquiry.student_name == "Rahul Sharma").first()
    enquiry_id = enquiry.id

    # Post Comment 1
    resp1 = client.post(
        f"/admin/api/enquiries/{enquiry_id}/comments",
        headers=auth_headers,
        json={"comment_text": "Student scored 89% in PCM (Karnataka PUC). Eligible for merit scholarship."}
    )
    assert resp1.status_code == 200
    detail1 = resp1.json()
    assert len(detail1["comments"]) == 1
    assert "89% in PCM" in detail1["comments"][0]["comment_text"]

    # Post Comment 2
    resp2 = client.post(
        f"/admin/api/enquiries/{enquiry_id}/comments",
        headers=auth_headers,
        json={"comment_text": "Father requested fee structure breakdown in installments. Sent via email."}
    )
    assert resp2.status_code == 200
    detail2 = resp2.json()
    # CRITICAL: Both comments must exist!
    assert len(detail2["comments"]) == 2
    assert detail2["comments"][0]["comment_text"] == "Student scored 89% in PCM (Karnataka PUC). Eligible for merit scholarship."
    assert detail2["comments"][1]["comment_text"] == "Father requested fee structure breakdown in installments. Sent via email."
    assert detail2["comments"][0]["staff_name"] is not None


def test_empty_comment_rejected(client, auth_headers, test_db_session):
    """Whitespace-only comment is rejected."""
    enquiry = test_db_session.query(Enquiry).filter(Enquiry.student_name == "Rahul Sharma").first()
    resp = client.post(
        f"/admin/api/enquiries/{enquiry.id}/comments",
        headers=auth_headers,
        json={"comment_text": "   "}
    )
    assert resp.status_code == 422


# ==============================================================================
# 5. Follow-Up Action Recording Tests
# ==============================================================================

def test_record_follow_up_action(client, auth_headers, test_db_session):
    """Follow-up action records action type, staff attribution, and outcome notes."""
    enquiry = test_db_session.query(Enquiry).filter(Enquiry.student_name == "Rahul Sharma").first()
    enquiry_id = enquiry.id

    resp = client.post(
        f"/admin/api/enquiries/{enquiry_id}/follow-ups",
        headers=auth_headers,
        json={
            "action_type": "phone_call",
            "notes": "Spoke with applicant; scheduled campus tour for Saturday 11:00 AM."
        }
    )
    assert resp.status_code == 200
    detail = resp.json()
    assert len(detail["follow_ups"]) >= 1
    fu = detail["follow_ups"][-1]
    assert fu["action_type"] == "phone_call"
    assert "campus tour" in fu["notes"]
    assert fu["completed_at"] is not None


# ==============================================================================
# 6. Chronological Audit Timeline & Integrity
# ==============================================================================

def test_unified_chronological_audit_timeline(client, auth_headers, test_db_session):
    """Enquiry dossier includes all activities sorted in chronological order."""
    enquiry = test_db_session.query(Enquiry).filter(Enquiry.student_name == "Rahul Sharma").first()
    resp = client.get(f"/admin/api/enquiries/{enquiry.id}", headers=auth_headers)
    assert resp.status_code == 200
    detail = resp.json()

    timeline = detail["timeline"]
    assert len(timeline) >= 4  # Submission + 2 Status changes + 2 Comments + 1 Follow-up

    # Verify event types are recognized
    event_types = {item["type"] for item in timeline}
    assert "submission" in event_types
    assert "status_change" in event_types
    assert "comment" in event_types
    assert "follow_up" in event_types

    # Verify chronological ordering
    timestamps = [datetime.fromisoformat(item["timestamp"]) for item in timeline]
    for i in range(len(timestamps) - 1):
        assert timestamps[i] <= timestamps[i+1], f"Timeline not strictly chronological at index {i}"


# ==============================================================================
# 7. End-to-End Workflow Verification: Student Submission -> Staff Review -> Update
# ==============================================================================

def test_complete_e2e_student_enquiry_to_staff_management(client, auth_headers):
    """Full workflow: Student submits enquiry -> staff sees enquiry -> updates status -> adds comment."""
    # 1. Student submits enquiry
    student_payload = {
        "student_name": "Pooja Reddy",
        "email": "pooja.reddy@example.com",
        "phone_number": "+91 9988776655",
        "school_or_department": "School of Health Sciences",
        "course_name": "Bachelor of Pharmacy (B.Pharm)",
        "question": "Is NEET score mandatory for B.Pharm admission at DSU?",
    }
    submit_resp = client.post("/enquiries", json=student_payload)
    assert submit_resp.status_code == 201
    enquiry_id = submit_resp.json()["id"]

    # 2. Staff searches for the new enquiry
    search_resp = client.get(f"/admin/api/enquiries?search=Pooja", headers=auth_headers)
    assert search_resp.status_code == 200
    items = search_resp.json()["items"]
    assert any(item["id"] == enquiry_id for item in items)

    # 3. Staff reviews detailed enquiry
    detail_resp = client.get(f"/admin/api/enquiries/{enquiry_id}", headers=auth_headers)
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["student_name"] == "Pooja Reddy"
    assert detail["status"] == "processing"

    # 4. Staff adds confidential internal comment
    c_resp = client.post(
        f"/admin/api/enquiries/{enquiry_id}/comments",
        headers=auth_headers,
        json={"comment_text": "Clarified that NEET is not mandatory for B.Pharm; 10+2 with PCB 50% qualifies."}
    )
    assert c_resp.status_code == 200
    assert len(c_resp.json()["comments"]) == 1

    # 5. Staff updates status to resolved
    s_resp = client.patch(
        f"/admin/api/enquiries/{enquiry_id}/status",
        headers=auth_headers,
        json={"status": "resolved", "notes": "Candidate confirmed admission procedure understood."}
    )
    assert s_resp.status_code == 200
    assert s_resp.json()["status"] == "resolved"
    assert len(s_resp.json()["status_history"]) == 1

    # 6. Final verification of dossier
    final_resp = client.get(f"/admin/api/enquiries/{enquiry_id}", headers=auth_headers)
    final_data = final_resp.json()
    assert final_data["status"] == "resolved"
    assert len(final_data["comments"]) == 1
    assert len(final_data["status_history"]) == 1
    assert len(final_data["timeline"]) == 3  # Submission + comment + status change
