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
from pprint import pprint
from typing import Any, Callable, Coroutine, Dict, List, Optional
from openai import AsyncOpenAI
from openai.types.chat import (ChatCompletionMessageParam,
                               ChatCompletionAssistantMessageParam)
from models import AppEnviron, OpenAIAgent
from models.outputs.format import FormattedResponse
from models.react.outputs import ReactChoiceOutput
from models.run.RunCallbackMessage import RunStepActionType, RunStepCallbackMessage, RunStepStatus
from tools.defaults.react import (
    REACT_PLANNING_TOOLS,
    REACT_FORMATTED_OUTPUT_TOOL)
from prompts.templates.react import (REACT_PROMPT, REACT_ACTION_PROMPT,
                                     REACT_ACTION_FIELD_PROMPT)
from prompts.generator import generate_prompt
from tools.models import Tool
environ: AppEnviron = AppEnviron()
client = AsyncOpenAI(api_key=environ.openai_api_key)


class OpenAIReactAgent(OpenAIAgent):

    def __init__(self,
                 on_step_update: Optional[Callable[..., Coroutine[any, any, any]]],
                 react_prompt: str = REACT_PROMPT,
                 react_options: List[Tool] = REACT_PLANNING_TOOLS,
                 react_formatted_response: Tool = REACT_FORMATTED_OUTPUT_TOOL,
                 debug: bool = False,
                 ** kwargs) -> None:
        super().__init__(**kwargs)
        self.debug = debug
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
        self.on_step_update = on_step_update

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
            action_response = None
            response = await self.__run_react_step()
            # if completed break loop and return completion message
            if response.choice == "ACTION":
                action_response = await self.__run_tool_completion(prompt=response.prompt)
            elif response.choice == "THOUGHT":
                action_response = {"role": "assistant",
                                   "content": f"[THOUGHT]: {response.prompt}"}
            elif response.choice == "PAUSE":
                action_response = {"role": "assistant",
                                   "content": f"[PAUSE]: {response.prompt}"}
            if response.choice == "ANSWER":
                return
            if action_response:
                self.react_loop_history.append(action_response)
            if self.debug:
                pprint(self.react_loop_history[-1])

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

        func = message.tool_calls[0].function

        args = json.loads(func.arguments)
        step_message = RunStepCallbackMessage(
            step_type=RunStepActionType[args["choice"]],
            status=RunStepStatus.PROCESSING,
            content=args["prompt"],
            completed_at=datetime.now().isoformat()
        )
        if len(message.tool_calls) == 0:
            # if no tools were called, an error should be
            # thrown since we want one step to be chosen
            step_message.status = RunStepStatus.FAILED
            raise Exception(
                "Neither Thought, Action, Observe, Complete were called")
        await self.__set_status(id=step_message.id, step_message=step_message)

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
        await self.__set_status(id=step_message.id, step_message=step_message)

        response: str = await self.tool_map[func.name](**args)

        step_message.status = RunStepStatus.COMPLETED
        step_message.completed_at = datetime.now().isoformat()

        await self.__set_status(id=step_message.id, step_message=step_message)

        content: str = generate_prompt(
            template=(
                "[ACTION]:\n"
                "Input Prompt:\n"
                "{prompt}\n\n"
                "Action Response:\n"
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
        actions: List[str] = []
        for tool in tools:
            action_field_prompts: List[str] = []
            for action_field_name, action_details in tool.parameters["properties"].items():
                action_field_prompt: str = generate_prompt(REACT_ACTION_FIELD_PROMPT, {
                    "action_field_name": action_field_name,
                    "action_field_description": action_details["description"],
                    "action_field_type": action_details["type"]
                })
                action_field_prompts.append(action_field_prompt)
            action_prompt: str = generate_prompt(REACT_ACTION_PROMPT, {
                "action_name": tool.name,
                "action_description": tool.description,
                "action_fields": "\n".join(action_field_prompts)
            })
            actions.append(action_prompt)
        return "\n\n".join(actions)

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
        tools_used = list(
            set([step.tool_used for _, step in self.run_history.items() if step.tool_used]))
        return FormattedResponse(**{**args, "tools_used": tools_used})

    async def __set_status(self, id: str, step_message: RunStepCallbackMessage) -> None:
        self.run_history[id] = step_message
        await self.on_step_update(**self.run_history)
        print(self.run_history)
