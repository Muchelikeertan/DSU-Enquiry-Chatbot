"""DSU College Enquiry Chatbot - Application Configuration.

Loads configuration from environment variables (.env) or uses sensible defaults.
Provides reliable path resolution for canonical JSON knowledge files.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root directory (parent of app/)
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings with environment variable override support."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # General Environment
    app_name: str = "DSU College Enquiry Chatbot"
    app_version: str = "1.0.0"
    app_env: str = "development"
    app_debug: bool = True
    host: str = "127.0.0.1"
    port: int = 8000

    # Canonical Data File Paths
    intents_json_path: Path = BASE_DIR / "intents.json"
    college_info_json_path: Path = BASE_DIR / "college_info.json"
    faq_info_json_path: Path = BASE_DIR / "faq_info.json"

    # NLP Pipeline Settings (Stage 2)
    intent_confidence_threshold: float = 0.60
    faq_confidence_threshold: float = 0.65
    context_ttl_turns: int = 6

    # LLM Settings (Stage 3 - Groq API)
    llm_provider: str = "groq"
    groq_api_key: Optional[str] = None
    groq_model: str = "openai/gpt-oss-120b"
    groq_timeout_seconds: float = 15.0
    groq_max_retries: int = 1

    # Database Settings (Stage 4)
    database_url: Optional[str] = None

    # Admission Cell Admin & Security Settings (Stage 5)
    secret_key: str = "***REMOVED***"
    admin_username: str = "admin"
    admin_password: str = "***REMOVED***"
    admin_email: str = "admissions@dsu.edu.in"
    access_token_expire_minutes: int = 480


settings = Settings()
