"""Database module for DSU Enquiry Chatbot."""

from app.db.database import (
    Base,
    get_engine,
    get_session_factory,
    SessionLocal,
    get_db,
    init_db,
    check_db_connection,
)
from app.db.models import (
    Enquiry,
    EnquiryStatus,
    ChatInteractionLog,
    StaffUser,
    StaffRole,
    EnquiryComment,
    EnquiryStatusHistory,
    EnquiryFollowUp,
)

__all__ = [
    "Base",
    "get_engine",
    "get_session_factory",
    "SessionLocal",
    "get_db",
    "init_db",
    "check_db_connection",
    "Enquiry",
    "EnquiryStatus",
    "ChatInteractionLog",
    "StaffUser",
    "StaffRole",
    "EnquiryComment",
    "EnquiryStatusHistory",
    "EnquiryFollowUp",
]

