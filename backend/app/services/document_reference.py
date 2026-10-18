"""Deterministic workspace document references; no model calls or new memory store."""
import re
import uuid
from dataclasses import dataclass
from typing import Sequence

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.message import Message
from app.services.document_service import DocumentService
from app.services.ingestion_service import _IN_MEMORY_CHUNKS
from app.tools.solver_router import fold


REFERENCE = re.compile(
    r"\b(?:(?:this|that|the uploaded|uploaded|the selected|selected)\s+(?:document|file|pdf)|"
    r"(?:file|tai lieu|pdf)\s+(?:nay|do|vua tai)|(?:trong|in the)\s+(?:tai lieu|document)|"
    r"(?:tom tat|summari[sz]e)\s+(?:the\s+)?(?:file|tai lieu|document|pdf))\b"
)


@dataclass
class DocumentResolution:
    document: Document | None = None
    message: str | None = None

    def retrieval_query(self, query: str) -> str:
        # Give retrieval the same filename context as a working explicit-name request.
        if self.document and fold(self.document.original_filename) not in fold(query):
            return f"{query[:700]} {self.document.original_filename}"[:1000]
        return query


class DocumentReferenceResolver:
    @staticmethod
    async def validate_id(db: AsyncSession | None, document_id: uuid.UUID,
                          workspace_id: uuid.UUID, user_id: uuid.UUID) -> Document:
        document = await DocumentService.get_document(db, document_id, user_id)
        if not document or document.workspace_id != workspace_id:
            raise HTTPException(404, "Document not found in active workspace or unauthorized.")
        return document

    @staticmethod
    async def usable_ids(db: AsyncSession | None, documents: Sequence[Document],
                         workspace_id: uuid.UUID, user_id: uuid.UUID) -> set[uuid.UUID]:
        eligible = [d for d in documents if d.workspace_id == workspace_id and d.user_id == user_id
                    and d.status == "processed" and d.embedding_status == "completed"]
        if not eligible:
            return set()
        if db is not None:
            result = await db.execute(select(DocumentChunk.document_id).where(
                DocumentChunk.document_id.in_([d.id for d in eligible]),
                DocumentChunk.workspace_id == workspace_id,
                DocumentChunk.user_id == user_id,
                DocumentChunk.embedding.is_not(None),
                func.length(func.trim(DocumentChunk.content)) > 0,
            ).distinct())
            return set(result.scalars().all())
        return {d.id for d in eligible if any(
            c.document_id == d.id and c.workspace_id == workspace_id
            and c.user_id == user_id and c.content.strip() and c.embedding is not None
            for c in _IN_MEMORY_CHUNKS.get(d.id, [])
        )}

    @classmethod
    async def resolve(cls, db: AsyncSession | None, workspace_id: uuid.UUID, user_id: uuid.UUID,
                      query: str, *, document_id: uuid.UUID | None = None,
                      active_document_id: uuid.UUID | None = None,
                      history: Sequence[Message] = ()) -> DocumentResolution:
        text = fold(query)
        referential = bool(REFERENCE.search(text) or re.search(r"\.(?:pdf|docx)\b", text))
        selected_id = document_id or active_document_id
        if not selected_id and not referential:
            return DocumentResolution()
        vietnamese = bool(re.search(r"\b(?:tom tat|tai lieu|nay|giai|file nay)\b", text))
        if selected_id:
            target = await cls.validate_id(db, selected_id, workspace_id, user_id)
        else:
            documents = await DocumentService.list_documents(db, workspace_id, user_id) or []
            matches = cls.filename_matches(query, documents)
            if len(matches) > 1:
                return cls.clarify(matches, vietnamese)
            target = matches[0] if matches else None
            # A named but missing file must not silently resolve to a different one.
            if target is None and re.search(r"\.(?:pdf|docx)\b", text):
                return DocumentResolution(message=(
                    "Không tìm thấy tên tài liệu này trong workspace hiện tại. Bạn muốn dùng tài liệu nào?"
                    if vietnamese else "I couldn't match that filename in this workspace. Which document do you mean?"))
            if target is None:
                # Only use bounded, persisted conversation context for a referential follow-up.
                for message in reversed(list(history)[-6:]):
                    if message.role == "user":
                        previous = cls.filename_matches(message.content, documents)
                    else:
                        ids = {str(c.get("document_id")) for c in (message.citations or [])}
                        previous = [d for d in documents if str(d.id) in ids]
                        if ids and not previous:
                            return DocumentResolution(message=(
                                "Tài liệu được nhắc đến trước đó không còn trong workspace này. Bạn muốn dùng tài liệu nào?"
                                if vietnamese else "The previously referenced document is no longer in this workspace. Which document do you mean?"))
                    if len(previous) > 1:
                        return cls.clarify(previous, vietnamese)
                    if previous:
                        target = previous[0]
                        break
            if target is None:
                usable_ids = await cls.usable_ids(db, documents, workspace_id, user_id)
                eligible = [d for d in documents if d.id in usable_ids]
                if len(eligible) > 1:
                    return cls.clarify(eligible, vietnamese)
                if eligible:
                    target = eligible[0]
                elif len(documents) == 1:
                    target = documents[0]
                elif documents:
                    return cls.clarify(documents, vietnamese)
                else:
                    return DocumentResolution(message=(
                        "Workspace hiện tại chưa có tài liệu được tải lên."
                        if vietnamese else "There are no uploaded documents in this workspace."))
        if target.id not in await cls.usable_ids(db, [target], workspace_id, user_id):
            name = target.original_filename
            if target.status != "processed" or target.embedding_status != "completed":
                message = (f"{name}: trạng thái xử lý {target.status}, embedding {target.embedding_status}. Tài liệu chưa sẵn sàng để tra cứu."
                           if vietnamese else f"{name}: processing status {target.status}, embedding status {target.embedding_status}. This document is not ready to search.")
            else:
                message = (f"{name} chưa có nội dung đã trích xuất và nhúng có thể sử dụng. Vui lòng xử lý lại tài liệu."
                           if vietnamese else f"{name} has no usable extracted and embedded content. Please reprocess the document.")
            return DocumentResolution(message=message)
        return DocumentResolution(document=target)

    @staticmethod
    def filename_matches(query: str, documents: Sequence[Document]) -> list[Document]:
        text = fold(query)
        return [d for d in documents if re.search(
            r"(?<!\w)" + re.escape(fold(d.original_filename)) + r"(?!\w)", text)]

    @staticmethod
    def clarify(documents: Sequence[Document], vietnamese: bool) -> DocumentResolution:
        names = ", ".join(d.original_filename for d in documents)
        return DocumentResolution(message=(
            f"Bạn muốn mình dùng tài liệu nào: {names}?" if vietnamese
            else f"Which document do you mean: {names}?"))
