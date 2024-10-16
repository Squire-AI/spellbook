from pydantic import Field
from pydantic_settings import BaseSettings


class AppEnviron(BaseSettings):
    openai_api_key: str = Field(description="API Key for OpenAI")
