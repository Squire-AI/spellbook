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
import asyncio
import json
from typing import Any, Callable, Coroutine, Dict, List
from openai import AsyncOpenAI
from openai.types.chat import (ChatCompletionMessage, ChatCompletionMessageParam,
                               ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam, ChatCompletionFunctionMessageParam)
from models import AppEnviron, OpenAIAgent
from models.react.outputs import ReactChoiceOutput
from tools.defaults.react import REACT_PLANNING_TOOLS
from prompts.templates.react import REACT_PROMPT
from prompts.generator import generate_prompt
from tools.models import Tool
environ: AppEnviron = AppEnviron()
client = AsyncOpenAI(api_key=environ.openai_api_key)


class ReactAgent(OpenAIAgent):

    def __init__(self, tool_map: Dict[str, Callable[[Any], Coroutine[Any, Any, str]]], react_prompt: str = REACT_PROMPT, react_options: List[Tool] = REACT_PLANNING_TOOLS, **kwargs) -> None:
        super().__init__(**kwargs)
        self.react_options = react_options
        self.chain_of_thought_message_history = []
        self.react_prompt = react_prompt
        self.tool_completion_prompt: str = (
            "You use the most appropriate tool based on the prompt\n"
        )
        self.react_completion_prompt: str = self.__generate_react_prompt()
        self.react_loop_history: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": self.react_completion_prompt},
            * self.messages
        ]
        self.tool_map = tool_map

    async def run(self) -> ChatCompletionMessage:
        """runs chain of thought"""
        # runs loop
        # returns response in format or non-formatted
        await self.__loop()
        return

    async def __loop(self) -> None:
        """runs loop for steps in chain of thought """
        for _ in range(self.max_iterations):
            # execute react completion, get the action
            response = await self.__run_react_step()
            # if completed break loop and return completion message
            print(response)
            if response.choice == "ACTION":
                action_response = await self.__run_tool_completion(prompt=response.prompt)
            elif response.choice == "THOUGHT":
                action_response = {"role": "assistant",
                                   "content": response.prompt}
            elif response.choice == "OBSERVE":
                action_response = {"role": "assistant",
                                   "content": response.prompt}
            if response.choice == "COMPLETE":
                return
            self.react_loop_history.append(action_response)

    async def __run_react_step(self) -> ReactChoiceOutput:
        """runs step in the chain of thought"""
        tools = self.__format_tools(self.react_options)
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=self.react_loop_history,
            tools=tools,
            tool_choice={"type": "function",
                         "function": {"name": "StepChoice"}}
        )
        message = response.choices[0].message
        if len(message.tool_calls) == 0:
            # if no tools were called, an error should be
            # thrown since we want one step to be chosen
            raise Exception("Neither Thought, Action, Complete were called")
        func = message.tool_calls[0].function
        args = json.loads(func.arguments)
        print(func, args)
        return ReactChoiceOutput(
            choice=args["choice"],
            prompt=args["prompt"]
        )

    async def __run_tool_completion(self, prompt: str) -> ChatCompletionAssistantMessageParam:
        tools = self.__format_tools(self.tools)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.tool_completion_prompt
                },
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            tools=tools,
            tool_choice="required"
        )
        message = response.choices[0].message
        if len(message.tool_calls) == 0:
            # if no tools were called, an error should be
            # thrown since we want one step to be chosen
            raise Exception("No tool was called")
        func = message.tool_calls[0].function
        args = json.loads(func.arguments)
        # execute function
        response: str = await self.tool_map[func.name](**args)
        return {
            "role": "function",
            "name": func.name,
            "content": f"Question: {prompt} \nAnswer: {response}"
        }

    def __format_tools(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        return [{"function": {"name": tool.name, "description": tool.description, "parameters": tool.parameters}, "type": "function"} for tool in tools]

    def __format_tools_to_action_prompt(self, tools: List[Tool]) -> str:
        return "\n\n".join([f"Name:{tool.name}\n Description: {tool.description}" for tool in tools])

    def __generate_react_prompt(self) -> str:
        prompt = generate_prompt(self.react_prompt, {
            "SystemPrompt": self.system_prompt,
            "Actions": self.__format_tools_to_action_prompt(self.tools)
        })
        return prompt


async def search_tool(**kwargs) -> str:
    """"""
    print("used search")
    return """Elon musk is 55 yrs old, lionel messi is 33"""


async def calculator(**kwargs) -> str:
    print("used calculator")
    return "110"


if __name__ == "__main__":
    agent = ReactAgent(
        client=client,
        model="gpt-4o-mini",
        temperature=0,
        max_iterations=10,
        system_prompt="You are a helpful assistant",
        messages=[
            {"role": "user", "content": "what is elon musk's current age times 2 + lionel messi's age"}
        ],
        tools=[
            Tool(
                name="calculator",
                description="Calculator to calculate mathematical expressions",
                parameters={"type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "MathematicalExpression": {
                                    "type": "string",
                                    "description": "appropriate mathematical expressions"
                                }},
                            "required": [
                                "MathematicalExpression"
                            ]

                            },
            ),
            Tool(
                name="search_tool",
                description="Search tool for searching the web for queries",
                parameters={"type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "search query string"
                                }},
                            "required": [
                                "query"
                            ]
                            },

            )
        ],
        tool_map={
            "calculator": calculator,
            "search_tool": search_tool
        })

    async def run():
        response = await agent.run()
        print(response)
    asyncio.run(run())
