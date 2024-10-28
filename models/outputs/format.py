from typing import List
from pydantic import BaseModel


class FormattedResponse(BaseModel):
    """
    base model for formatted response
    """
    content: str
    sources: List[str]
    tools_used: List[str]
