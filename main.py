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
from typing import Any, Dict, List
from openai import AsyncOpenAI
from models import AppEnviron

environ: AppEnviron = AppEnviron()
client = AsyncOpenAI(api_key=environ.openai_api_key)


async def generate_plan() -> List[Dict[str, Any]]:
    pass

if __name__ == "__main__":
    pass
