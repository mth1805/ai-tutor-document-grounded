import uuid
import logging
from typing import Optional
import httpx
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from app.core.config import settings

logger = logging.getLogger(__name__)

security_scheme = HTTPBearer(auto_error=False)


class AuthenticatedUser(BaseModel):
    id: uuid.UUID
    email: Optional[str] = None


async def verify_supabase_token(token: str) -> AuthenticatedUser:
    """Verifies access token against Supabase Auth or local JWT secret."""

    # Local demo/test identities must never be accepted outside development.
    if settings.ENVIRONMENT == "development" and (
        token.startswith("test-token:") or token.startswith("test-user:")
    ):
        try:
            user_id_str = token.split(":", 1)[1]
            return AuthenticatedUser(
                id=uuid.UUID(user_id_str),
                email=f"user-{user_id_str[:8]}@test.example.com",
            )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid test token format",
            )

    # 2. Local JWT Secret Verification if configured
    if settings.SUPABASE_JWT_SECRET:
        try:
            payload = jwt.decode(
                token,
                settings.SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                audience="authenticated",
                options={"require": ["exp", "sub", "aud"]},
            )
            sub = payload.get("sub")
            if not sub:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token missing subject claim",
                )
            return AuthenticatedUser(
                id=uuid.UUID(sub),
                email=payload.get("email"),
            )
        except (jwt.PyJWTError, ValueError) as e:
            logger.warning("Local JWT verification failed: %s", e)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired authentication token",
            )

    # 3. Supabase Auth API verification
    if settings.SUPABASE_URL and settings.SUPABASE_ANON_KEY:
        auth_url = f"{settings.SUPABASE_URL.rstrip('/')}/auth/v1/user"
        headers = {
            "Authorization": f"Bearer {token}",
            "apikey": settings.SUPABASE_ANON_KEY,
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(auth_url, headers=headers)
                if res.status_code == 200:
                    user_data = res.json()
                    user_id = user_data.get("id")
                    if user_id:
                        return AuthenticatedUser(
                            id=uuid.UUID(user_id),
                            email=user_data.get("email"),
                        )
                elif res.status_code in (401, 403):
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid or expired Supabase authentication session",
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Authentication service temporarily unavailable",
                    )
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Supabase Auth API request error: %s", e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service temporarily unavailable",
            )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
) -> AuthenticatedUser:
    """Dependency enforcing that the request carries a verified user token."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await verify_supabase_token(credentials.credentials)
