"""Typed metadata filter contracts for consistent dense and sparse retrieval."""

from typing import Annotated

from pydantic import Field, StringConstraints

from .config import Name
from .contracts import Contract

FileType = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=32)]


class MetadataFilter(Contract):
    """Typed, SQL-injection-safe filter applied during candidate selection."""

    document_id: Name | None = None
    version: Annotated[int, Field(ge=1, le=2_147_483_647)] | None = None
    latest_only: bool = True
    file_type: FileType | None = None
    source: Name | None = None

    def normalized_extension(self) -> str | None:
        if self.file_type is None:
            return None
        ext = self.file_type.lower()
        return ext if ext.startswith(".") else f".{ext}"
