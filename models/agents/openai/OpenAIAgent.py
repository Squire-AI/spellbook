
from typing import List
from abc import ABC, abstractmethod
from openai import AsyncClient
from openai.types.chat import ChatCompletionMessageParam
from tools.models import Tool


class OpenAIAgent(ABC):
    def __init__(
            self,
            client: AsyncClient,
            model: str,
            system_prompt: str,
            temperature: float,
            max_iterations: int,
            tools: List[Tool],
            messages: List[ChatCompletionMessageParam]
    ) -> None:
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.messages = messages
        self.tools = tools

    @abstractmethod
    def run(self) -> any:
        """Run must be implemented in higher level use of open ai agent base class"""
        pass
