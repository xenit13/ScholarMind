from __future__ import annotations

from pydantic import BaseModel


class MemoryContext(BaseModel):
    text: str = ""
    hits: int = 0
