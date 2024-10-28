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
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List
from openai import AsyncOpenAI
from openai.types.chat import (ChatCompletionMessageParam,
                               ChatCompletionAssistantMessageParam)
from models import AppEnviron, OpenAIAgent
from models.outputs.format import FormattedResponse
from models.react.outputs import ReactChoiceOutput
from models.run.RunCallbackMessage import RunStepActionType, RunStepCallbackMessage, RunStepStatus
from tools.defaults.react import REACT_PLANNING_TOOLS, REACT_FORMATTED_OUTPUT_TOOL
from prompts.templates.react import REACT_PROMPT
from prompts.generator import generate_prompt
from tools.models import Tool
environ: AppEnviron = AppEnviron()
client = AsyncOpenAI(api_key=environ.openai_api_key)


class OpenAIReactAgent(OpenAIAgent):

    def __init__(self,
                 react_prompt: str = REACT_PROMPT,
                 react_options: List[Tool] = REACT_PLANNING_TOOLS,
                 react_formatted_response: Tool = REACT_FORMATTED_OUTPUT_TOOL,
                 ** kwargs) -> None:
        super().__init__(**kwargs)
        self.react_options = react_options
        self.chain_of_thought_message_history = []
        self.react_prompt = react_prompt
        self.tool_completion_prompt: str = (
            "You use the most appropriate tool based on the prompt\n"
        )
        self.tool_map = self.__init_tools_map()
        self.react_completion_prompt: str = self.__generate_react_prompt()
        self.react_loop_history: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": self.react_completion_prompt},
            * self.messages
        ]
        self.react_formatted_response = react_formatted_response
        self.run_history: Dict[str, RunStepCallbackMessage] = {}

    async def run(self) -> FormattedResponse:
        """runs chain of thought"""
        # runs loop
        # returns response in format or non-formatted
        await self.__loop()
        return await self.__generated_formatted_output()

    async def __loop(self) -> None:
        """runs loop for steps in chain of thought """
        for _ in range(self.max_iterations):
            # execute react completion, get the action
            response = await self.__run_react_step()
            # if completed break loop and return completion message
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
            step_message.status = RunStepStatus.FAILED
            raise Exception(
                "Neither Thought, Action, Observe, Complete were called")
        func = message.tool_calls[0].function

        args = json.loads(func.arguments)
        step_message = RunStepCallbackMessage(
            step_type=RunStepActionType[args["choice"]],
            status=RunStepStatus.PROCESSING,
            content=args["prompt"],
            completed_at=datetime.now().isoformat()
        )
        self.__set_status(id=step_message.id, step_message=step_message)

        return ReactChoiceOutput(
            choice=args["choice"],
            prompt=args["prompt"]
        )

    async def __run_tool_completion(self, prompt: str) -> ChatCompletionAssistantMessageParam:
        # returns OpenAI standardised format
        tools = self.__format_tools(self.tools)
        step_message = RunStepCallbackMessage(
            step_type=RunStepActionType.ACTION, status=RunStepStatus.PROCESSING)
        # returns tool message from tool agent
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
        # if no message throw error
        if len(response.choices) == 0:
            step_message.status = RunStepStatus.FAILED
            raise Exception("OpenAI had no response")

        message = response.choices[0].message
        if len(message.tool_calls) == 0:
            # if no tools were called, an error should be
            # thrown since we want one step to be chosen
            step_message.status = RunStepStatus.FAILED
            raise Exception("No tool was called")

        func = message.tool_calls[0].function
        args = json.loads(func.arguments)
        # execute function
        step_message.args = args
        step_message.tool_used = func.name
        self.__set_status(id=step_message.id, step_message=step_message)

        response: str = await self.tool_map[func.name](**args)

        step_message.status = RunStepStatus.COMPLETED
        step_message.completed_at = datetime.now().isoformat()

        self.__set_status(id=step_message.id, step_message=step_message)

        content: str = generate_prompt(
            template=(
                "Input Prompt:\n"
                "{prompt}\n\n"
                "Tool Response:\n"
                "{response}"
            ),
            variables={
                "prompt": prompt,
                "response": response
            }
        )
        return {
            "role": "function",
            "name": func.name,
            "content": content
        }

    def __format_tools(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        return [
            {"function":
             {"name": tool.name,
              "description": tool.description,
                 "parameters": tool.parameters
              },
             "type": "function"
             } for tool in tools]

    def __format_tools_to_action_prompt(self, tools: List[Tool]) -> str:
        return "\n\n".join([f"Name:{tool.name}\n Description: {tool.description}" for tool in tools])

    def __generate_react_prompt(self) -> str:
        prompt = generate_prompt(self.react_prompt, {
            "SystemPrompt": self.system_prompt,
            "Actions": self.__format_tools_to_action_prompt(self.tools)
        })
        return prompt

    def __init_tools_map(self) -> Dict[str, Callable[..., Coroutine[Any, Any, str]]]:
        """
        takes in tools and initialises tool map
        """
        return {tool.name: tool.function for tool in self.tools}

    async def __generated_formatted_output(self) -> FormattedResponse:
        """
        Takes in response and generates formatted output
        """
        format_tool = self.__format_tools([self.react_formatted_response])
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=self.react_loop_history,
            temperature=self.temperature,
            tools=format_tool,
            tool_choice={
                "type": "function",
                "function":
                {"name": self.react_formatted_response.name}
            }
        )
        # if no message throw error
        if len(response.choices) == 0:
            raise Exception("OpenAI had no response")
        message = response.choices[0].message
        if len(message.tool_calls) == 0:
            # if no tools were called, an error should be
            # thrown since we want one step to be chosen
            raise Exception("Message wasn't formatted")
        func = message.tool_calls[0].function
        args = json.loads(func.arguments)
        return FormattedResponse(**args)

    def __set_status(self, id: str, step_message: RunStepCallbackMessage) -> None:
        self.run_history[id] = step_message
        print("run history", [value for _,
              value in self.run_history.items()][-1])


async def search_tool(**kwargs) -> str:
    return """Elon musk is 55 yrs old, lionel messi is 33"""


async def calculator(**kwargs) -> str:
    return "110"


if __name__ == "__main__":
    agent = OpenAIReactAgent(
        client=client,
        model="gpt-4o-mini",
        temperature=0.7,
        max_iterations=10,
        system_prompt="You are a helpful assistant",
        messages=[
            {"role": "user", "content": "what is elon musk's current age times 2"}
        ],
        tools=[
            Tool(
                id="abc",
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
                function=calculator
            ),
            Tool(
                id="xyz",
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
                function=search_tool
            )
        ])

    async def run():
        response = await agent.run()
        print(response)
    asyncio.run(run())
