import re
import uuid
import logging
from pathlib import Path
from typing import Optional, Dict
import httpx
from fastapi import HTTPException, status
from app.core.config import settings

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}

EXTENSION_TO_DEFAULT_MIME: Dict[str, str] = {
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

# In-memory storage dictionary for testing and local development without Supabase Storage
_IN_MEMORY_STORAGE: Dict[str, bytes] = {}


def sanitize_filename(filename: str) -> str:
    """Sanitizes an untrusted user-supplied filename to prevent path traversal and injection."""
    if not filename:
        return "document.bin"

    # 1. Strip any directory paths (both / and \)
    base = filename.replace("\\", "/").split("/")[-1].strip()

    # 2. Remove null bytes and control characters
    base = re.sub(r"[\x00-\x1f\x7f]", "", base)

    # 3. Disallow directory traversal identifiers
    if base in ("", ".", ".."):
        return "document.bin"

    # 4. Remove leading dots or hyphens
    base = base.lstrip(".-")

    # 5. Sanitize unsafe characters while preserving alphanumeric, unicode letters, dots, hyphens, and underscores
    base = re.sub(r'[^a-zA-Z0-9._\- ]', '_', base)
    if not base:
        base = "document.bin"

    # 6. Limit max length to 200 characters while preserving extension
    if len(base) > 200:
        parts = base.rsplit(".", 1)
        if len(parts) == 2:
            base = f"{parts[0][:190]}.{parts[1][:9]}"
        else:
            base = base[:200]

    return base


def validate_magic_bytes(extension: str, content: bytes) -> bool:
    """Verifies that the file content starts with the expected magic bytes for the declared extension."""
    if not content:
        return False

    ext = extension.lower()
    if ext == ".pdf":
        return content.startswith(b"%PDF-")
    elif ext == ".docx":
        return content.startswith(b"PK\x03\x04")
    elif ext == ".doc":
        return content.startswith(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1")
    elif ext == ".png":
        return content.startswith(b"\x89PNG\r\n\x1a\n")
    elif ext in (".jpg", ".jpeg"):
        return content.startswith(b"\xFF\xD8\xFF")
    elif ext == ".webp":
        return (
            len(content) >= 12
            and content.startswith(b"RIFF")
            and content[8:12] == b"WEBP"
        )
    elif ext == ".txt":
        # Check that it doesn't contain null bytes (not a binary executable disguised as text)
        if b"\x00" in content[:1024]:
            return False
        try:
            content[:1024].decode("utf-8")
            return True
        except UnicodeDecodeError:
            try:
                content[:1024].decode("latin-1")
                return True
            except UnicodeDecodeError:
                return False

    return False


def validate_upload_file(filename: str, content: bytes, mime_type: Optional[str] = None) -> tuple[str, str, int]:
    """Validates file size, non-emptiness, allowed extension, and content magic bytes.

    Returns:
        (sanitized_filename, validated_mime_type, file_size)
    """
    file_size = len(content)

    # 1. Non-empty check
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)",
        )

    # 2. Maximum file size check
    if file_size > settings.MAX_FILE_SIZE_BYTES:
        max_mb = settings.MAX_FILE_SIZE_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds maximum allowed size of {max_mb:.0f} MB",
        )

    # 3. Sanitized filename
    clean_name = sanitize_filename(filename)
    ext = Path(clean_name).suffix.lower()

    # 4. Extension check
    if ext not in ALLOWED_EXTENSIONS:
        allowed_list = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: {allowed_list}",
        )

    # 5. Magic bytes check
    if not validate_magic_bytes(ext, content):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File content does not match the declared extension '{ext}'",
        )

    # 6. Normalize MIME type
    final_mime = EXTENSION_TO_DEFAULT_MIME.get(ext, "application/octet-stream")
    if mime_type and "/" in mime_type and "octet-stream" not in mime_type:
        # If client provided a specific valid MIME type, keep it if reasonable
        if mime_type.startswith("image/") or mime_type.startswith("text/") or "pdf" in mime_type or "word" in mime_type:
            final_mime = mime_type

    return clean_name, final_mime, file_size


class StorageService:
    """Manages file persistence in private Supabase Storage or deterministic in-memory storage."""

    @staticmethod
    def generate_storage_path(
        workspace_id: uuid.UUID, document_id: uuid.UUID, filename: str
    ) -> str:
        """Constructs a deterministic, workspace-scoped storage path."""
        clean_name = sanitize_filename(filename)
        return f"{workspace_id}/{document_id}/{clean_name}"

    @classmethod
    async def upload_file(
        cls,
        storage_path: str,
        content: bytes,
        mime_type: str,
    ) -> str:
        """Uploads a file to private Supabase Storage or in-memory fallback.

        Returns:
            The confirmed storage path.
        """
        # If live Supabase Storage credentials exist, upload to Supabase Storage
        if settings.SUPABASE_URL and settings.SUPABASE_SERVICE_ROLE_KEY:
            url = f"{settings.SUPABASE_URL.rstrip('/')}/storage/v1/object/{settings.STORAGE_BUCKET_NAME}/{storage_path}"
            headers = {
                "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY}",
                "Content-Type": mime_type,
                "x-upsert": "true",
            }
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    res = await client.post(url, content=content, headers=headers)
                    if res.status_code not in (200, 201):
                        logger.error(
                            "Supabase storage upload failed status=%s",
                            res.status_code,
                        )
                        raise HTTPException(
                            status_code=status.HTTP_502_BAD_GATEWAY,
                            detail="Failed to persist file in storage service",
                        )
                    return storage_path
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Supabase Storage error category=%s", type(e).__name__)
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Failed to persist file in storage service",
                )

        # In-memory storage fallback for local development/testing
        _IN_MEMORY_STORAGE[storage_path] = content
        return storage_path

    @classmethod
    async def get_file(cls, storage_path: str) -> Optional[bytes]:
        """Retrieves raw file bytes from storage."""
        if settings.SUPABASE_URL and settings.SUPABASE_SERVICE_ROLE_KEY:
            url = f"{settings.SUPABASE_URL.rstrip('/')}/storage/v1/object/{settings.STORAGE_BUCKET_NAME}/{storage_path}"
            headers = {
                "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY}",
            }
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    res = await client.get(url, headers=headers)
                    if res.status_code == 200:
                        return res.content
                    elif res.status_code == 404:
                        return None
                    else:
                        logger.warning("Supabase storage get failed (%s)", res.status_code)
                        return None
            except Exception as e:
                logger.error("Supabase storage get error: %s", e)
                return None

        return _IN_MEMORY_STORAGE.get(storage_path)

    @classmethod
    async def delete_file(cls, storage_path: str) -> bool:
        """Deletes a file from storage."""
        if settings.SUPABASE_URL and settings.SUPABASE_SERVICE_ROLE_KEY:
            url = f"{settings.SUPABASE_URL.rstrip('/')}/storage/v1/object/{settings.STORAGE_BUCKET_NAME}/{storage_path}"
            headers = {
                "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY}",
            }
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    res = await client.delete(url, headers=headers)
                    return res.status_code in (200, 204, 404)
            except Exception as e:
                logger.error("Supabase storage delete error: %s", e)
                return False

        if storage_path in _IN_MEMORY_STORAGE:
            del _IN_MEMORY_STORAGE[storage_path]
            return True
        return False

    @classmethod
    async def create_signed_url(cls, storage_path: str, expires_in: int = 3600) -> Optional[str]:
        """Generates a time-limited signed URL for private bucket access without making it public."""
        if settings.SUPABASE_URL and settings.SUPABASE_SERVICE_ROLE_KEY:
            url = f"{settings.SUPABASE_URL.rstrip('/')}/storage/v1/object/sign/{settings.STORAGE_BUCKET_NAME}/{storage_path}"
            headers = {
                "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY}",
                "Content-Type": "application/json",
            }
            payload = {"expiresIn": expires_in}
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    res = await client.post(url, json=payload, headers=headers)
                    if res.status_code == 200:
                        data = res.json()
                        signed_url = data.get("signedURL")
                        if signed_url:
                            base = settings.SUPABASE_URL.rstrip('/')
                            return f"{base}/storage/v1{signed_url}"
            except Exception as e:
                logger.error("Supabase storage signed URL generation error: %s", e)

        # Fallback for local development/testing
        if storage_path in _IN_MEMORY_STORAGE:
            return f"/api/v1/documents/storage-mock/{storage_path}"
        return None
