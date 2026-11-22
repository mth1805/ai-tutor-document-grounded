"""Transaction-local ingestion attempt identity for database fencing triggers."""
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def set_ingestion_attempt_context(
    db: AsyncSession | None,
    job_id: UUID | None,
    attempt: int | None,
) -> None:
    """Bind a queued worker's fencing token to its current PostgreSQL transaction."""
    if db is None or job_id is None or attempt is None:
        return
    await db.execute(
        text("""SELECT set_config('app.ingestion_job_id', :job_id, true),
                     set_config('app.ingestion_attempt', :attempt, true)"""),
        {"job_id": str(job_id), "attempt": str(attempt)},
    )
