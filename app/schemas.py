"""Pydantic Request and Response Schemas for DSU Enquiry Chatbot API.

Enforces strict input validation, type safety, and clean data contracts.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


class HealthResponse(BaseModel):
    """Application health and status summary without sensitive credentials."""
    status: str = "ok"
    app_name: str
    app_version: str
    environment: str
    nlp_pipeline: str
    llm_provider: str
    database: str


class ChatRequest(BaseModel):
    """Incoming user chat query."""
    message: str = Field(..., description="User message text", max_length=1000)
    session_id: Optional[str] = Field(None, description="Optional conversation session ID", max_length=100)

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Message cannot be empty or whitespace only.")
        return stripped


class ResponseSection(BaseModel):
    """Structured section within a chatbot response."""
    heading: str = Field(..., description="Section title or heading")
    items: List[str] = Field(default_factory=list, description="Bullet items or structured content")


class SourceCitation(BaseModel):
    """Authoritative source reference."""
    title: str = Field(..., description="Official title or description of the source")
    url: str = Field(..., description="Canonical source URL")


class ChatResponse(BaseModel):
    """Structured response from the chatbot NLP and grounding pipeline.

    Adheres strictly to the canonical response schema (Section 7) while preserving
    backward compatibility with existing frontend and telemetry fields.
    """
    # Canonical Section 7 Schema Fields
    status: str = Field(..., description="Response status (e.g. source_checked, source_checked_partial, dynamic_lookup_required)")
    intent: str = Field(..., description="Predicted or routed intent")
    answer: str = Field(..., description="Direct, grounded answer addressing the user's question")
    sections: List[ResponseSection] = Field(default_factory=list, description="Structured factual sections")
    sources: List[SourceCitation] = Field(default_factory=list, description="Verified official source citations")
    missing_information: List[str] = Field(default_factory=list, description="List of unverified or missing items requiring confirmation")
    requires_enquiry: bool = Field(False, description="Whether the enquiry form should be displayed")

    # Backward-compatible fields
    response_text: str = Field(..., description="Full formatted markdown response text")
    confidence: float = Field(..., description="Intent confidence score")
    handler: str = Field(..., description="Handler name")
    show_enquiry_form: bool = Field(False, description="Alias of requires_enquiry for backward compatibility")
    form_policy: str = Field("never", description="Form display policy")
    form_prefill: Dict[str, Any] = Field(default_factory=dict, description="Prefill metadata for enquiry form")
    source: Optional[str] = Field(None, description="Primary source URL or identifier")
    quick_replies: List[str] = Field(default_factory=list, description="Quick reply suggestions")
    used_llm: bool = Field(False, description="Whether an external LLM was utilized")
    session_id: str = Field(..., description="Conversation session ID")



class EnquiryCreate(BaseModel):
    """Student enquiry form submission payload matching college_info.enquiry.fields."""
    student_name: str = Field(..., min_length=1, max_length=150, description="Full name of student")
    email: str = Field(..., max_length=255, description="Valid email address")
    phone_number: Optional[str] = Field(None, max_length=30, description="Optional contact number")
    school_or_department: Optional[str] = Field(None, max_length=150, description="School or Department")
    course_name: Optional[str] = Field(None, max_length=150, description="Programme or course of interest")
    question: str = Field(..., min_length=1, max_length=2000, description="Student's enquiry or question")
    session_id: Optional[str] = Field(None, max_length=100, description="Associated chat session ID")

    @field_validator("email")
    @classmethod
    def validate_email_address(cls, v: str) -> str:
        stripped = v.strip()
        if not EMAIL_REGEX.match(stripped):
            raise ValueError("Invalid email address format.")
        return stripped.lower()

    @field_validator("student_name", "question")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Field cannot be empty or whitespace only.")
        return stripped

    @field_validator("phone_number", "school_or_department", "course_name")
    @classmethod
    def strip_optional_strings(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            stripped = v.strip()
            return stripped if stripped else None
        return None


class EnquiryResponse(BaseModel):
    """Serialized enquiry submission response."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    student_name: str
    email: str
    phone_number: Optional[str] = None
    school_or_department: Optional[str] = None
    course_name: Optional[str] = None
    question: str
    status: str
    created_at: datetime


# ============================================================================
# Admission Cell Dashboard & Audit History Schemas
# ============================================================================

class StaffLoginRequest(BaseModel):
    """Staff credential submission."""
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1, max_length=128)

    @field_validator("username")
    @classmethod
    def clean_username(cls, v: str) -> str:
        return v.strip().lower()


class StaffUserResponse(BaseModel):
    """Public staff profile representation."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None


class StaffLoginResponse(BaseModel):
    """Staff authentication response containing JWT."""
    access_token: str
    token_type: str = "bearer"
    user: StaffUserResponse


class EnquiryStatusUpdate(BaseModel):
    """Staff payload to transition enquiry status with audit note."""
    status: str = Field(..., description="Target status: new, processing, called, visited, on_hold, resolved, closed")
    notes: Optional[str] = Field(None, max_length=1000, description="Optional explanation for audit log")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        normalized = v.strip().lower()
        valid = {"new", "processing", "called", "visited", "on_hold", "resolved", "closed"}
        if normalized not in valid:
            raise ValueError(f"Invalid status '{normalized}'. Valid statuses are: {sorted(valid)}")
        return normalized


class EnquiryCommentCreate(BaseModel):
    """Staff payload to append an immutable internal enquiry comment."""
    comment_text: str = Field(..., min_length=1, max_length=3000, description="Internal comment content")

    @field_validator("comment_text")
    @classmethod
    def validate_non_empty_comment(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Comment text cannot be empty.")
        return stripped


class EnquiryCommentResponse(BaseModel):
    """Serialized enquiry comment representation."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    enquiry_id: int
    staff_id: Optional[int] = None
    staff_name: str
    comment_text: str
    created_at: datetime


class EnquiryStatusHistoryResponse(BaseModel):
    """Serialized status change audit record."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    enquiry_id: int
    previous_status: Optional[str] = None
    new_status: str
    staff_id: Optional[int] = None
    staff_name: str
    notes: Optional[str] = None
    created_at: datetime


class EnquiryFollowUpCreate(BaseModel):
    """Staff payload to record a follow-up action."""
    action_type: str = Field(..., description="Action type: phone_call, email, campus_visit, document_verification, counseling, other")
    notes: Optional[str] = Field(None, max_length=2000, description="Details of the interaction or outcome")
    scheduled_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v: str) -> str:
        normalized = v.strip().lower()
        valid = {"phone_call", "email", "campus_visit", "document_verification", "counseling", "other"}
        if normalized not in valid:
            raise ValueError(f"Invalid action type '{normalized}'. Valid types are: {sorted(valid)}")
        return normalized


class EnquiryFollowUpResponse(BaseModel):
    """Serialized follow-up record."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    enquiry_id: int
    staff_id: Optional[int] = None
    staff_name: str
    action_type: str
    scheduled_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    notes: Optional[str] = None
    created_at: datetime


class TimelineEventResponse(BaseModel):
    """Unified chronological audit timeline item."""
    type: str
    timestamp: datetime
    actor: str
    action: str
    details: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EnquiryDetailedResponse(BaseModel):
    """Detailed enquiry record with persistent comments, status history, and audit timeline."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    student_name: str
    email: str
    phone_number: Optional[str] = None
    school_or_department: Optional[str] = None
    course_name: Optional[str] = None
    question: str
    status: str
    session_id: Optional[str] = None
    assigned_staff_id: Optional[int] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    comments: List[EnquiryCommentResponse] = Field(default_factory=list)
    status_history: List[EnquiryStatusHistoryResponse] = Field(default_factory=list)
    follow_ups: List[EnquiryFollowUpResponse] = Field(default_factory=list)
    timeline: List[TimelineEventResponse] = Field(default_factory=list)

