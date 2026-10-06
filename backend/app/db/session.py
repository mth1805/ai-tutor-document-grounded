import logging
import json
from typing import AsyncGenerator
from fastapi import Depends, HTTPException, status
from sqlalchemy import event, text
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.core.config import settings
from app.core.security import AuthenticatedUser, get_current_user

logger = logging.getLogger(__name__)

engine = None
AsyncSessionLocal = None


@event.listens_for(Session, "after_begin")
def _apply_rls_context(session: Session, transaction, connection) -> None:
    """Apply the verified request identity to every PostgreSQL transaction.

    SET LOCAL is transaction-scoped, so it is reapplied after service commits and
    cannot leak a user's identity through a pooled connection to another request.
    """
    user_id = session.info.get("rls_user_id")
    claims = {"role": "authenticated", "aud": "authenticated"}
    if user_id is not None:
        claims["sub"] = str(user_id)
    connection.exec_driver_sql("SET LOCAL ROLE authenticated")
    connection.execute(
        text("SELECT set_config('request.jwt.claims', :claims, true)"),
        {"claims": json.dumps(claims)},
    )
    connection.execute(
        text("SELECT set_config('request.jwt.claim.sub', :user_id, true)"),
        {"user_id": str(user_id) if user_id is not None else ""},
    )


if settings.DATABASE_URL:
    try:
        engine = create_async_engine(
            settings.DATABASE_URL,
            echo=settings.ENVIRONMENT == "development",
            future=True,
            pool_pre_ping=True,
        )
        AsyncSessionLocal = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    except Exception as e:
        logger.warning("Failed to initialize database engine: %s", e)


async def get_db(
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> AsyncGenerator[AsyncSession | None, None]:
    """Yields a session with Supabase RLS bound to the verified request user."""
    if AsyncSessionLocal is None:
        if settings.ENVIRONMENT != "development":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database is not configured",
            )
        yield None
        return

    async with AsyncSessionLocal() as session:
        session.info["rls_user_id"] = current_user.id
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
