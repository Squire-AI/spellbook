from typing import List, Literal
from pydantic import BaseModel


class FormattedResponseSource(BaseModel):
    """
    base model for formatted response source
    """
    type: Literal["document"] | Literal["website"]
    source: str = ""
    url: str = ""


class FormattedResponse(BaseModel):
    """
    base model for formatted response
    """
    content: str
    sources: List[FormattedResponseSource]
    tools_used: List[str]
