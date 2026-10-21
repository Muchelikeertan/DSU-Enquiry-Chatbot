"""SQLAlchemy ORM Models for DSU Enquiry Chatbot & Admission Cell Management.

Defines:
1. StaffUser: Admission Cell staff accounts with secure salted password hashing.
2. Enquiry: Storage for student enquiry form submissions.
3. EnquiryComment: Persistent internal staff comments linked to enquiries.
4. EnquiryStatusHistory: Immutable audit log tracking all status transitions.
5. EnquiryFollowUp: Follow-up actions (phone call, email, campus visit, verification).
6. ChatInteractionLog: Telemetry log for chatbot queries without storing personal messages.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from app.db.database import Base


def utc_now():
    """Current UTC timestamp."""
    return datetime.now(timezone.utc)


class EnquiryStatus(str, Enum):
    """Lifecycle statuses for student enquiry submissions."""
    PROCESSING = "processing"
    ON_HOLD = "on_hold"
    CALLED = "called"
    COMPLETED = "completed"


class StaffRole(str, Enum):
    """Role-based access levels for Admission Cell personnel."""
    STAFF = "admissions_staff"
    ADMIN = "admissions_admin"


class StaffUser(Base):
    """Admission Cell staff member account."""

    __tablename__ = "staff_users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(150), nullable=False)
    role = Column(String(50), nullable=False, default=StaffRole.STAFF.value)
    hashed_password = Column(String(255), nullable=False)
    salt = Column(String(64), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    last_login = Column(DateTime, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes staff user summary without sensitive hash or salt."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }


class Enquiry(Base):
    """Student enquiry form submission record matching college_info.enquiry.fields."""

    __tablename__ = "enquiries"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    student_name = Column(String(150), nullable=False)
    email = Column(String(255), nullable=False, index=True)
    phone_number = Column(String(30), nullable=True)
    school_or_department = Column(String(150), nullable=True)
    course_name = Column(String(150), nullable=True)
    question = Column(Text, nullable=False)
    status = Column(String(50), nullable=False, default=EnquiryStatus.PROCESSING.value, index=True)
    admin_comments = Column(Text, nullable=True)
    assigned_staff_id = Column(Integer, ForeignKey("staff_users.id", ondelete="SET NULL"), nullable=True)
    session_id = Column(String(100), nullable=True, index=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    # Relationships
    assigned_staff = relationship("StaffUser", foreign_keys=[assigned_staff_id], lazy="joined")
    comments = relationship(
        "EnquiryComment",
        back_populates="enquiry",
        cascade="all, delete-orphan",
        order_by="EnquiryComment.created_at.asc()",
        lazy="selectin",
    )
    status_history = relationship(
        "EnquiryStatusHistory",
        back_populates="enquiry",
        cascade="all, delete-orphan",
        order_by="EnquiryStatusHistory.created_at.asc()",
        lazy="selectin",
    )
    follow_ups = relationship(
        "EnquiryFollowUp",
        back_populates="enquiry",
        cascade="all, delete-orphan",
        order_by="EnquiryFollowUp.created_at.asc()",
        lazy="selectin",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes core enquiry model to dictionary."""
        return {
            "id": self.id,
            "student_name": self.student_name,
            "email": self.email,
            "phone_number": self.phone_number,
            "school_or_department": self.school_or_department,
            "course_name": self.course_name,
            "question": self.question,
            "status": self.status,
            "admin_comments": self.admin_comments,
            "assigned_staff_id": self.assigned_staff_id,
            "assigned_staff_name": self.assigned_staff.full_name if self.assigned_staff else None,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def to_detailed_dict(self) -> Dict[str, Any]:
        """Serializes complete enquiry with historical audit trail, comments, and follow-ups."""
        data = self.to_dict()

        # Serialized collections
        comments_list = [c.to_dict() for c in (self.comments or [])]
        history_list = [h.to_dict() for h in (self.status_history or [])]
        follow_ups_list = [f.to_dict() for f in (self.follow_ups or [])]

        data["comments"] = comments_list
        data["status_history"] = history_list
        data["follow_ups"] = follow_ups_list

        # Unified chronological timeline
        timeline: List[Dict[str, Any]] = []

        # Submission entry
        timeline.append({
            "type": "submission",
            "timestamp": self.created_at.isoformat() if self.created_at else None,
            "actor": self.student_name,
            "title": "Enquiry Submitted",
            "details": f"Submitted enquiry regarding {self.course_name or self.school_or_department or 'DSU admissions'}",
        })

        for h in history_list:
            timeline.append({
                "type": "status_change",
                "timestamp": h["created_at"],
                "actor": h["staff_name"],
                "title": f"Status updated to '{h['new_status']}'",
                "details": f"Transitioned from '{h['previous_status'] or 'initial'}' to '{h['new_status']}'" + (f". Note: {h['notes']}" if h.get("notes") else ""),
            })

        for c in comments_list:
            timeline.append({
                "type": "comment",
                "timestamp": c["created_at"],
                "actor": c["staff_name"],
                "title": "Internal Staff Comment",
                "details": c["comment_text"],
            })

        for f in follow_ups_list:
            action_label = f["action_type"].replace("_", " ").title()
            timeline.append({
                "type": "follow_up",
                "timestamp": f["created_at"],
                "actor": f["staff_name"],
                "title": f"Follow-up: {action_label}",
                "details": f.get("notes") or f"Recorded {action_label} action.",
                "scheduled_at": f.get("scheduled_at"),
                "completed_at": f.get("completed_at"),
            })

        # Sort timeline chronologically
        timeline.sort(key=lambda x: x["timestamp"] or "")
        data["timeline"] = timeline

        return data


class EnquiryComment(Base):
    """Persistent internal Admission Cell comments linked to an enquiry."""

    __tablename__ = "enquiry_comments"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    enquiry_id = Column(Integer, ForeignKey("enquiries.id", ondelete="CASCADE"), nullable=False, index=True)
    staff_id = Column(Integer, ForeignKey("staff_users.id", ondelete="SET NULL"), nullable=True)
    staff_name = Column(String(150), nullable=False)
    comment_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    enquiry = relationship("Enquiry", back_populates="comments")
    staff = relationship("StaffUser")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "enquiry_id": self.enquiry_id,
            "staff_id": self.staff_id,
            "staff_name": self.staff_name,
            "comment_text": self.comment_text,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class EnquiryStatusHistory(Base):
    """Immutable audit trail for every status transition."""

    __tablename__ = "enquiry_status_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    enquiry_id = Column(Integer, ForeignKey("enquiries.id", ondelete="CASCADE"), nullable=False, index=True)
    previous_status = Column(String(50), nullable=True)
    new_status = Column(String(50), nullable=False)
    staff_id = Column(Integer, ForeignKey("staff_users.id", ondelete="SET NULL"), nullable=True)
    staff_name = Column(String(150), nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    enquiry = relationship("Enquiry", back_populates="status_history")
    staff = relationship("StaffUser")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "enquiry_id": self.enquiry_id,
            "previous_status": self.previous_status,
            "new_status": self.new_status,
            "staff_id": self.staff_id,
            "staff_name": self.staff_name,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class EnquiryFollowUp(Base):
    """Follow-up action records (phone calls, emails, campus visits)."""

    __tablename__ = "enquiry_follow_ups"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    enquiry_id = Column(Integer, ForeignKey("enquiries.id", ondelete="CASCADE"), nullable=False, index=True)
    staff_id = Column(Integer, ForeignKey("staff_users.id", ondelete="SET NULL"), nullable=True)
    staff_name = Column(String(150), nullable=False)
    action_type = Column(String(100), nullable=False)  # phone_call, email, campus_visit, etc.
    scheduled_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    enquiry = relationship("Enquiry", back_populates="follow_ups")
    staff = relationship("StaffUser")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "enquiry_id": self.enquiry_id,
            "staff_id": self.staff_id,
            "staff_name": self.staff_name,
            "action_type": self.action_type,
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ChatInteractionLog(Base):
    """Operational telemetry for chatbot requests without retaining private message text."""

    __tablename__ = "chat_interaction_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(100), nullable=True, index=True)
    query_intent = Column(String(100), nullable=True)
    confidence = Column(Float, nullable=True)
    handler = Column(String(50), nullable=True)
    used_llm = Column(Boolean, default=False, nullable=False)
    show_enquiry_form = Column(Boolean, default=False, nullable=False)
    status = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)
