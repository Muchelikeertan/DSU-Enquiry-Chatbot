"""Admission Cell Administrative API Endpoints.

Provides secure management of student enquiries, staff authentication,
persistent internal comments, and complete audit history tracking.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.config import settings
from app.db import (
    Enquiry,
    EnquiryComment,
    EnquiryFollowUp,
    EnquiryStatus,
    EnquiryStatusHistory,
    StaffUser,
    get_db,
)
from app.schemas import (
    EnquiryCommentCreate,
    EnquiryDetailedResponse,
    EnquiryFollowUpCreate,
    EnquiryStatusUpdate,
    StaffLoginRequest,
    StaffLoginResponse,
    StaffUserResponse,
)
from app.security import (
    create_access_token,
    get_current_staff_user,
    verify_password,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/api", tags=["Admission Cell"])


# ==============================================================================
# Authentication Endpoints
# ==============================================================================

@router.post(
    "/login",
    response_model=StaffLoginResponse,
    summary="Staff login",
    description="Authenticates Admission Cell staff member and returns JWT session token.",
)
def staff_login(
    payload: StaffLoginRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    """Authenticates staff and issues secure JWT token & HTTP-only cookie."""
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable",
        )

    user = db.query(StaffUser).filter(StaffUser.username == payload.username.lower()).first()
    if not user or not verify_password(payload.password, user.hashed_password, user.salt):
        logger.warning("Failed login attempt for username: %s", payload.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    if not user.is_active:
        logger.warning("Inactive staff user attempted login: %s", payload.username)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated. Please contact administrator.",
        )

    # Record login timestamp
    user.last_login = datetime.now(timezone.utc)
    db.commit()

    token = create_access_token({
        "sub": user.username,
        "role": user.role,
        "user_id": user.id,
    })

    # Set HTTP-only cookie for browser navigation

    response.set_cookie(
        key="admin_token",
        value=token,
        httponly=True,
        samesite="lax",
        path="/",
        secure=False,
        max_age=settings.access_token_expire_minutes * 60,
    )

    return StaffLoginResponse(
        access_token=token,
        token_type="bearer",
        user=StaffUserResponse.model_validate(user),
    )


@router.post(
    "/logout",
    summary="Staff logout",
    description="Clears staff session cookie.",
)
def staff_logout(response: Response):
    """Logs out staff member by clearing session cookie."""
    response.delete_cookie(key="admin_token", path="/")
    return {"message": "Logged out successfully"}


@router.get(
    "/me",
    response_model=StaffUserResponse,
    summary="Get current staff profile",
    description="Returns authenticated staff user information.",
)
def get_current_staff(
    current_user: StaffUser = Depends(get_current_staff_user),
):
    """Returns currently authenticated staff user profile."""
    return StaffUserResponse.model_validate(current_user)


# ==============================================================================
# Enquiry Management Endpoints
# ==============================================================================

@router.get(
    "/enquiries",
    summary="List enquiries with search and filters",
    description="Retrieves enquiries with optional filtering by status, school, and text search.",
)
def list_enquiries(
    search: Optional[str] = Query(None, description="Search by student name, email, phone, or question"),
    status: Optional[str] = Query(None, description="Filter by status (new, processing, called, etc.)"),
    school: Optional[str] = Query(None, description="Filter by school or department"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: StaffUser = Depends(get_current_staff_user),
    db: Session = Depends(get_db),
):
    """Search and filter enquiries for Admission Cell dashboard."""
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable",
        )

    query = db.query(Enquiry)

    if status:
        query = query.filter(Enquiry.status == status.strip().lower())

    if school:
        query = query.filter(Enquiry.school_or_department.ilike(f"%{school.strip()}%"))

    if search:
        search_pattern = f"%{search.strip()}%"
        query = query.filter(
            or_(
                Enquiry.student_name.ilike(search_pattern),
                Enquiry.email.ilike(search_pattern),
                Enquiry.phone_number.ilike(search_pattern),
                Enquiry.question.ilike(search_pattern),
                Enquiry.course_name.ilike(search_pattern),
            )
        )

    total_count = query.count()
    items = query.order_by(Enquiry.created_at.desc()).offset(offset).limit(limit).all()

    # Aggregate status counts for dashboard metrics cards
    status_counts_raw = (
        db.query(Enquiry.status, func.count(Enquiry.id))
        .group_by(Enquiry.status)
        .all()
    )
    status_counts = {s: count for s, count in status_counts_raw}

    return {
        "total": total_count,
        "items": [item.to_dict() for item in items],
        "status_counts": status_counts,
        "limit": limit,
        "offset": offset,
    }


@router.get(
    "/enquiries/{enquiry_id}",
    summary="Get enquiry details with full audit history",
    description="Returns student enquiry, comments, status transitions, and follow-ups in chronological order.",
)
def get_enquiry_details(
    enquiry_id: int,
    current_user: StaffUser = Depends(get_current_staff_user),
    db: Session = Depends(get_db),
):
    """Retrieves full enquiry dossier with comments and audit trail."""
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable",
        )

    enquiry = db.query(Enquiry).filter(Enquiry.id == enquiry_id).first()
    if not enquiry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Enquiry with ID {enquiry_id} not found.",
        )

    return enquiry.to_detailed_dict()


@router.patch(
    "/enquiries/{enquiry_id}/status",
    summary="Update enquiry status with audit logging",
    description="Changes enquiry status and records audit record with staff attribution.",
)
def update_enquiry_status(
    enquiry_id: int,
    payload: EnquiryStatusUpdate,
    current_user: StaffUser = Depends(get_current_staff_user),
    db: Session = Depends(get_db),
):
    """Updates enquiry status and logs an immutable audit event."""
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable",
        )

    enquiry = db.query(Enquiry).filter(Enquiry.id == enquiry_id).first()
    if not enquiry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Enquiry with ID {enquiry_id} not found.",
        )

    old_status = enquiry.status
    new_status = payload.status

    if old_status != new_status:
        enquiry.status = new_status
        enquiry.updated_at = datetime.now(timezone.utc)

        # Create persistent status change audit record
        audit_record = EnquiryStatusHistory(
            enquiry_id=enquiry.id,
            previous_status=old_status,
            new_status=new_status,
            staff_id=current_user.id,
            staff_name=current_user.full_name,
            notes=payload.notes,
        )
        db.add(audit_record)
        db.commit()
        db.refresh(enquiry)
        logger.info(
            "Status changed for enquiry #%d by %s: %s -> %s",
            enquiry.id,
            current_user.username,
            old_status,
            new_status,
        )
    elif payload.notes:
        # If status didn't change but a note was supplied, still record audit note
        audit_record = EnquiryStatusHistory(
            enquiry_id=enquiry.id,
            previous_status=old_status,
            new_status=new_status,
            staff_id=current_user.id,
            staff_name=current_user.full_name,
            notes=payload.notes,
        )
        db.add(audit_record)
        db.commit()
        db.refresh(enquiry)

    return enquiry.to_detailed_dict()


@router.post(
    "/enquiries/{enquiry_id}/comments",
    summary="Add persistent internal staff comment",
    description="Saves internal comment linked to enquiry and staff. Never overwrites prior comments.",
)
def add_enquiry_comment(
    enquiry_id: int,
    payload: EnquiryCommentCreate,
    current_user: StaffUser = Depends(get_current_staff_user),
    db: Session = Depends(get_db),
):
    """Appends internal comment to the enquiry audit record."""
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable",
        )

    enquiry = db.query(Enquiry).filter(Enquiry.id == enquiry_id).first()
    if not enquiry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Enquiry with ID {enquiry_id} not found.",
        )

    new_comment = EnquiryComment(
        enquiry_id=enquiry.id,
        staff_id=current_user.id,
        staff_name=current_user.full_name,
        comment_text=payload.comment_text,
    )
    enquiry.updated_at = datetime.now(timezone.utc)

    db.add(new_comment)
    db.commit()
    db.refresh(enquiry)
    logger.info("Added comment to enquiry #%d by %s", enquiry.id, current_user.username)

    return enquiry.to_detailed_dict()


@router.post(
    "/enquiries/{enquiry_id}/follow-ups",
    summary="Record follow-up action",
    description="Stores follow-up event (call, email, visit) with timestamp and notes.",
)
def add_enquiry_follow_up(
    enquiry_id: int,
    payload: EnquiryFollowUpCreate,
    current_user: StaffUser = Depends(get_current_staff_user),
    db: Session = Depends(get_db),
):
    """Records follow-up interaction with the applicant."""
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable",
        )

    enquiry = db.query(Enquiry).filter(Enquiry.id == enquiry_id).first()
    if not enquiry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Enquiry with ID {enquiry_id} not found.",
        )

    completed_at = payload.completed_at
    if completed_at is None and payload.scheduled_at is None:
        completed_at = datetime.now(timezone.utc)

    follow_up = EnquiryFollowUp(
        enquiry_id=enquiry.id,
        staff_id=current_user.id,
        staff_name=current_user.full_name,
        action_type=payload.action_type,
        scheduled_at=payload.scheduled_at,
        completed_at=completed_at,
        notes=payload.notes,
    )
    enquiry.updated_at = datetime.now(timezone.utc)

    db.add(follow_up)
    db.commit()
    db.refresh(enquiry)
    logger.info("Recorded follow-up (%s) for enquiry #%d by %s", payload.action_type, enquiry.id, current_user.username)

    return enquiry.to_detailed_dict()
