"""FastAPI Backend Application for DSU College Enquiry Chatbot.

Provides endpoints:
- GET  /health: Application health & connection status
- POST /chat: Processes queries via NLPPipeline & grounded Groq LLM
- POST /enquiries: Validates and persists student enquiry form submissions
"""

import os
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import settings
from app.db import (
    Enquiry,
    EnquiryStatus,
    ChatInteractionLog,
    get_db,
    init_db,
    check_db_connection,
)
from app.llm.handler import GroqLLMHandler
from app.nlp.pipeline import NLPPipeline
from app.nlp.response_formatter import parse_canonical_response
from app.schemas import (
    ChatRequest,
    ChatResponse,
    EnquiryCreate,
    EnquiryResponse,
    HealthResponse,
    ResponseSection,
    SourceCitation,
)

logger = logging.getLogger(__name__)

# Global pipeline instance initialized on startup
_pipeline_instance: Optional[NLPPipeline] = None


def get_pipeline() -> NLPPipeline:
    """Dependency that yields the initialized NLP pipeline singleton."""
    global _pipeline_instance
    if _pipeline_instance is None:
        llm_handler = GroqLLMHandler(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            timeout=settings.groq_timeout_seconds,
            max_retries=settings.groq_max_retries,
        )
        _pipeline_instance = NLPPipeline.from_files(
            intents_path=settings.intents_json_path,
            college_info_path=settings.college_info_json_path,
            faq_info_path=settings.faq_info_json_path,
            llm_handler=llm_handler,
            intent_confidence_threshold=settings.intent_confidence_threshold,
            faq_confidence_threshold=settings.faq_confidence_threshold,
            context_ttl_turns=settings.context_ttl_turns,
        )
    return _pipeline_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management for database schema and pipeline loading."""
    logger.info("Initializing DSU Chatbot application services...")
    # Initialize DB tables if database is configured
    try:
        init_db()
        from app.db import SessionLocal
        from app.security import ensure_default_admin
        if SessionLocal is not None:
            with SessionLocal() as db_session:
                ensure_default_admin(db_session)
                logger.info("Admission Cell default admin account initialized.")
    except Exception as e:
        logger.warning("Database table creation or admin seeding deferred or failed: %s", type(e).__name__)

    # Preload the NLP pipeline singleton
    try:
        get_pipeline()
        logger.info("NLP pipeline and Groq integration loaded successfully.")
    except Exception as e:
        logger.error("Failed to preload NLP pipeline: %s", e)

    yield

    logger.info("Shutting down DSU Chatbot application services...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Official Dayananda Sagar University (DSU) Academic Enquiry Chatbot API.",
    lifespan=lifespan,
)

# CORS middleware configured for explicit local development origins with credentials support
ALLOWED_ORIGINS = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:5501",
    "http://localhost:5501",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files mounting and root homepage route
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def serve_homepage():
        """Serves the authentic DSU homepage with embedded floating chatbot widget."""
        index_file = os.path.join(STATIC_DIR, "index.html")
        if os.path.isfile(index_file):
            return FileResponse(index_file)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Homepage not found",
        )

    @app.get("/admin/login", include_in_schema=False)
    def serve_admin_login():
        """Serves the Admission Cell staff login page."""
        login_file = os.path.join(STATIC_DIR, "admin-login.html")
        if os.path.isfile(login_file):
            return FileResponse(login_file)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Admin login page not found",
        )

    @app.get("/admin", include_in_schema=False)
    @app.get("/admin/dashboard", include_in_schema=False)
    def serve_admin_dashboard():
        """Serves the Admission Cell enquiry management dashboard."""
        dash_file = os.path.join(STATIC_DIR, "admin-dashboard.html")
        if os.path.isfile(dash_file):
            return FileResponse(dash_file)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Admin dashboard page not found",
        )


# Mount Admission Cell REST APIs
from app.api.admin import router as admin_router
app.include_router(admin_router)



# ==============================================================================
# 1. Health Probe Endpoint
# ==============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check probe",
    description="Reports system health, NLP pipeline status, and database connectivity without exposing credentials.",
)
def health_check(db: Optional[Session] = Depends(get_db)):
    """Health status probe without sensitive credential leakage."""
    db_status = "not_configured"
    if db is not None:
        try:
            from sqlalchemy import text
            db.execute(text("SELECT 1"))
            db_status = "connected"
        except Exception:
            db_status = "disconnected"
    elif settings.database_url:
        is_connected = check_db_connection()
        db_status = "connected" if is_connected else "disconnected"

    pipeline_status = "loaded" if _pipeline_instance is not None else "ready"

    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        app_version=settings.app_version,
        environment=settings.app_env,
        nlp_pipeline=pipeline_status,
        llm_provider=settings.llm_provider,
        database=db_status,
    )


# ==============================================================================
# 2. Chat Processing Endpoint
# ==============================================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Process user query",
    description="Processes query through the NLP pipeline and returns grounded answer with form policy.",
)
def chat_endpoint(
    request: ChatRequest,
    pipeline: NLPPipeline = Depends(get_pipeline),
    db: Optional[Session] = Depends(get_db),
):
    """Processes chat messages through the NLP pipeline and returns structured response."""
    session_id = request.session_id or str(uuid.uuid4())

    try:
        nlp_resp = pipeline.process_query(request.message, session_id=session_id)
    except Exception as e:
        logger.error("Error processing query in NLP pipeline: %s", type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your message. Please try again.",
        )

    # Optional operational telemetry without storing personal user messages
    if db is not None:
        try:
            interaction_log = ChatInteractionLog(
                session_id=session_id,
                query_intent=str(nlp_resp.intent),
                confidence=float(nlp_resp.confidence) if nlp_resp.confidence is not None else None,
                handler=str(nlp_resp.handler),
                used_llm=bool(nlp_resp.used_llm),
                show_enquiry_form=bool(nlp_resp.show_enquiry_form),
                status=str(nlp_resp.status),
            )
            db.add(interaction_log)
            db.commit()
        except Exception as e:
            logger.warning("Failed to record telemetry interaction log: %s", type(e).__name__)
            db.rollback()

    answer, sections, sources, missing_info, req_enquiry = parse_canonical_response(
        response_text=nlp_resp.response_text,
        source=nlp_resp.source,
        show_enquiry_form=bool(nlp_resp.show_enquiry_form),
    )

    return ChatResponse(
        status=str(nlp_resp.status),
        intent=str(nlp_resp.intent),
        answer=answer,
        sections=[ResponseSection(**s) for s in sections],
        sources=[SourceCitation(**s) for s in sources],
        missing_information=missing_info,
        requires_enquiry=req_enquiry,
        response_text=nlp_resp.response_text,
        confidence=float(nlp_resp.confidence),
        handler=str(nlp_resp.handler),
        show_enquiry_form=bool(nlp_resp.show_enquiry_form),
        form_policy=str(nlp_resp.form_policy),
        form_prefill=nlp_resp.form_prefill or {},
        source=nlp_resp.source,
        quick_replies=nlp_resp.quick_replies or [],
        used_llm=bool(nlp_resp.used_llm),
        session_id=session_id,
    )


# ==============================================================================
# 3. Enquiry Form Submission Endpoint
# ==============================================================================

@app.post(
    "/enquiries",
    response_model=EnquiryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit student enquiry form",
    description="Validates and persists student enquiry form submission into PostgreSQL.",
)
def create_enquiry(
    enquiry_data: EnquiryCreate,
    db: Optional[Session] = Depends(get_db),
):
    """Validates and persists enquiry form submissions."""
    if db is None:
        logger.error("Enquiry submission attempted but database session is unavailable.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service is currently unavailable. Please try again later.",
        )
    # Prevent duplicate submissions caused by repeated clicks or rapid retries
    try:
        from datetime import datetime, timezone
        existing_duplicate = (
            db.query(Enquiry)
            .filter(
                Enquiry.email == str(enquiry_data.email),
                Enquiry.student_name == enquiry_data.student_name,
                Enquiry.question == enquiry_data.question,
            )
            .order_by(Enquiry.id.desc())
            .first()
        )
        if existing_duplicate and existing_duplicate.created_at:
            created_time = existing_duplicate.created_at
            now_time = datetime.now(timezone.utc) if created_time.tzinfo is not None else datetime.utcnow()
            if abs((now_time - created_time).total_seconds()) < 10:
                logger.info("Duplicate enquiry submission prevented; returning existing ID: %d", existing_duplicate.id)
                return existing_duplicate
    except Exception as e:
        logger.debug("Duplicate check skipped due to non-fatal error: %s", e)

    new_enquiry = Enquiry(
        student_name=enquiry_data.student_name,
        email=str(enquiry_data.email),
        phone_number=enquiry_data.phone_number,
        school_or_department=enquiry_data.school_or_department,
        course_name=enquiry_data.course_name,
        question=enquiry_data.question,
        status=EnquiryStatus.PROCESSING.value,
        session_id=enquiry_data.session_id,
    )

    try:
        db.add(new_enquiry)
        db.commit()
        db.refresh(new_enquiry)
        logger.info("Enquiry submitted successfully with ID: %d", new_enquiry.id)
    except SQLAlchemyError as e:
        db.rollback()
        logger.error("Database failure while saving enquiry submission: %s", type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save enquiry submission. Please try again later.",
        )

    return new_enquiry
