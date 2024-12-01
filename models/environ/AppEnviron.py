from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class AppEnviron(BaseSettings):
    openai_api_key: str = Field(description="API Key for OpenAI")
    groq_api_key: Optional[str] = Field(None, description="API Key for Groq")
