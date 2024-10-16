"""
Parts of Agent:
- planning node
- tool node
- reasoning node
- output format node

features:
- multimodal
- token counting
- tool calling
- observability
"""
from openai import AsyncOpenAI
from models import AppEnviron

environ: AppEnviron = AppEnviron()
client = AsyncOpenAI(api_key=environ.openai_api_key)


if __name__ == "__main__":
    pass
