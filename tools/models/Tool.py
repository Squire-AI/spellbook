from typing import Any, Dict
from pydantic import BaseModel, Field


class Tool(BaseModel):
    name: str = Field(..., regex=r"^[A-Za-z0-9_]+$",
                      description="Name must only contain letters, numbers, and underscores, and must not contain spaces or other symbols")
    description: str = Field(...)
    parameters: Dict[str, Any]
    strict: bool = Field(True)
