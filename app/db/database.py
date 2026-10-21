"""Database Session and Engine Management for DSU Enquiry Chatbot.

Provides:
- Thread-safe SQLAlchemy Engine & SessionLocal
- get_db FastAPI dependency with automatic resource cleanup
- init_db schema initialization
- check_db_connection health probe without credential leakage
"""

import logging
from typing import Generator, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from app.config import settings

logger = logging.getLogger(__name__)

Base = declarative_base()

_engine: Optional[Engine] = None
_SessionFactory: Optional[sessionmaker] = None


def normalize_database_url(url: Optional[str]) -> Optional[str]:
    """Ensures compatible driver specification for PostgreSQL."""
    if not url:
        return None
    # If standard postgres:// or postgresql:// without driver is provided, default to psycopg
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://") and "+psycopg" not in url and "+psycopg2" not in url:
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def get_engine(database_url: Optional[str] = None) -> Optional[Engine]:
    """Retrieves or creates a singleton SQLAlchemy Engine."""
    global _engine
    target_url = database_url or settings.database_url
    if not target_url:
        return None

    norm_url = normalize_database_url(target_url)

    if _engine is None or str(_engine.url) != norm_url:
        connect_args = {}
        if norm_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        elif "postgres" in norm_url:
            connect_args["connect_timeout"] = 2
        engine_kwargs = {
            "pool_pre_ping": True,
            "connect_args": connect_args,
        }
        if not norm_url.startswith("sqlite"):
            engine_kwargs["pool_size"] = 10
            engine_kwargs["max_overflow"] = 20
            engine_kwargs["pool_recycle"] = 300
        _engine = create_engine(norm_url, **engine_kwargs)
    return _engine


def get_session_factory(engine: Optional[Engine] = None) -> Optional[sessionmaker]:
    """Retrieves or creates sessionmaker for the active engine."""
    global _SessionFactory
    active_engine = engine or get_engine()
    if active_engine is None:
        return None
    if _SessionFactory is None or _SessionFactory.kw.get("bind") != active_engine:
        _SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=active_engine)
    return _SessionFactory


def SessionLocal() -> Optional[Session]:
    """Convenience factory to instantiate a new database session."""
    factory = get_session_factory()
    return factory() if factory else None



def init_db(engine: Optional[Engine] = None) -> bool:
    """Creates all registered database tables in the target database if reachable."""
    active_engine = engine or get_engine()
    if active_engine is None:
        return False
    if not check_db_connection(active_engine):
        logger.warning("Database unreachable; skipping table initialization.")
        return False
    try:
        from app.db import models  # Ensure all models are imported before create_all
        Base.metadata.create_all(bind=active_engine)
        # Ensure schema migrations for existing databases
        try:
            with active_engine.connect() as conn:
                conn.execute(text("ALTER TABLE enquiries ADD COLUMN IF NOT EXISTS assigned_staff_id INTEGER REFERENCES staff_users(id) ON DELETE SET NULL;"))
                conn.commit()
        except Exception:
            pass
        logger.info("Database tables initialized successfully.")
        return True
    except Exception as e:
        logger.warning("Database table creation failed: %s", type(e).__name__)
        return False


def check_db_connection(engine: Optional[Engine] = None) -> bool:
    """Probes the database with a safe query without exposing credentials."""
    active_engine = engine or get_engine()
    if active_engine is None:
        return False
    try:
        with active_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.warning("Database connectivity check failed: %s", type(e).__name__)
        return False


def get_db() -> Generator[Optional[Session], None, None]:
    """FastAPI dependency that provides an isolated database session per request."""
    factory = get_session_factory()
    if factory is None:
        yield None
        return

    session = factory()
    try:
        yield session
    finally:
        session.close()
