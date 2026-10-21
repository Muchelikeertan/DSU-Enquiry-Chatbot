"""Authentication and Security Module for DSU Admission Cell Management.

Provides:
- Cryptographic PBKDF2-HMAC-SHA256 password hashing and verification.
- Signed HMAC-SHA256 JWT tokens for stateless session management.
- FastAPI dependency for protecting staff endpoints.
- Safe bootstrap provisioning for the default Admission Cell administrator.
"""

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from typing import Any, Dict, Optional, Tuple

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.config import settings
from app.db.database import get_db
from app.db.models import StaffRole, StaffUser

logger = logging.getLogger(__name__)

# Bearer security scheme with auto_error=False to support cookie fallback
security_scheme = HTTPBearer(auto_error=False)


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Hashes a plaintext password using PBKDF2-HMAC-SHA256 with 100,000 iterations.

    Returns:
        (hash_hex, salt_hex)
    """
    if not salt:
        salt = secrets.token_hex(16)
    salt_bytes = salt.encode("utf-8")
    pwd_bytes = password.encode("utf-8")
    derived = hashlib.pbkdf2_hmac("sha256", pwd_bytes, salt_bytes, 100_000)
    return derived.hex(), salt


def verify_password(password: str, hashed_password: str, salt: str) -> bool:
    """Verifies plaintext password against stored PBKDF2 hash using constant-time comparison."""
    if not password or not hashed_password or not salt:
        return False
    computed_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(computed_hash, hashed_password)


def _b64url_encode(data: bytes) -> str:
    """URL-safe base64 encoding without trailing padding."""
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    """URL-safe base64 decoding with restored padding."""
    padding = 4 - (len(s) % 4)
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


def create_access_token(data: Dict[str, Any], expires_minutes: Optional[int] = None) -> str:
    """Creates a cryptographically signed HMAC-SHA256 JWT token."""
    exp_minutes = expires_minutes or settings.access_token_expire_minutes
    exp_timestamp = int(time.time() + (exp_minutes * 60))

    header = {"alg": "HS256", "typ": "JWT"}
    payload = {**data, "exp": exp_timestamp, "iat": int(time.time())}

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    unsigned_token = f"{_b64url_encode(header_bytes)}.{_b64url_encode(payload_bytes)}"

    signature = hmac.new(
        settings.secret_key.encode("utf-8"),
        unsigned_token.encode("utf-8"),
        hashlib.sha256,
    ).digest()

    return f"{unsigned_token}.{_b64url_encode(signature)}"


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decodes and validates token signature and expiry. Returns payload dict or None."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        unsigned_token = f"{parts[0]}.{parts[1]}"
        expected_sig = hmac.new(
            settings.secret_key.encode("utf-8"),
            unsigned_token.encode("utf-8"),
            hashlib.sha256,
        ).digest()

        actual_sig = _b64url_decode(parts[2])
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None

        payload_bytes = _b64url_decode(parts[1])
        payload = json.loads(payload_bytes.decode("utf-8"))

        # Verify expiration
        if "exp" in payload and payload["exp"] < int(time.time()):
            return None

        return payload
    except Exception as e:
        logger.debug("Failed to decode token: %s", type(e).__name__)
        return None


def get_current_staff_user(
    request: Request,
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    db: Optional[Session] = Depends(get_db),
) -> StaffUser:
    """Authenticates staff requests via Bearer token or admin_token cookie.

    Raises:
        HTTPException: 401 Unauthorized if missing, expired, or invalid.
    """
    token: Optional[str] = None

    if auth and auth.credentials:
        token = auth.credentials.strip()
    elif "admin_token" in request.cookies:
        token = request.cookies.get("admin_token")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required to access Admission Cell portal.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("user_id")
    username = payload.get("sub")

    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database session unavailable for authentication.",
        )

    staff_user = db.query(StaffUser).filter(
        (StaffUser.id == user_id) | (StaffUser.username == username)
    ).first()

    if not staff_user or not staff_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Staff user account not found or has been deactivated.",
        )

    return staff_user


def require_admin_role(
    current_user: StaffUser = Depends(get_current_staff_user),
) -> StaffUser:
    """Restricts endpoint access to administrators only."""
    if current_user.role != StaffRole.ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access required for this action.",
        )
    return current_user


def ensure_default_admin(db: Session) -> Optional[StaffUser]:
    """Safe bootstrap initializer creating the default Admission Cell Admin if no staff exists."""
    if db is None:
        return None

    try:
        existing = db.query(StaffUser).first()
        if existing is not None:
            return existing

        hashed_pwd, salt = hash_password(settings.admin_password)
        default_admin = StaffUser(
            username=settings.admin_username,
            email=settings.admin_email,
            full_name="DSU Admission Cell Administrator",
            role=StaffRole.ADMIN.value,
            hashed_password=hashed_pwd,
            salt=salt,
            is_active=True,
        )
        db.add(default_admin)
        db.commit()
        db.refresh(default_admin)
        logger.info("Default Admission Cell Administrator created successfully: %s", default_admin.username)
        return default_admin
    except Exception as e:
        logger.warning("Could not initialize default admin user: %s", type(e).__name__)
        db.rollback()
        return None
