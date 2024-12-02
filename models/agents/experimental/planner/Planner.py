"""
This is an experimental agent, it is design to take a more deterministic approach.
It organises apps used into agents, each with their own sets of tools (for example:
CRUD utils) It has a planning step, an LLM generates steps for where each step,
their instructions, the agent to use. Once the steps are generated, they are run
for each agent sequentially, the agents, generate a list of tool runs with their params.
These are executed.

NOTES:
    - this planning approach only applies for tools or tasks that are once step.
        Experience:
            - with ReAct agents, the former approach for updating a calendar is to use search calendar
              and then use update calendar tool. This is multi-steps for to achieve one task, which is to
              update a calendar.
            - with the planning agent, each task must not have tools that are dependent on the result
              of another tool.
            - There must be a way to merge these two steps into one, a "single-level" tool.
"""
import logging
import asyncio
from enum import StrEnum
import json
import os
from pydantic import BaseModel, ConfigDict
from typing import Any, Callable, Coroutine, Dict, List, Optional
from pprint import pprint
from openai import AsyncClient as OpenAIClient
import re
from openai.types.chat import ChatCompletionMessageParam


logging.basicConfig(level=logging.INFO)


def generate_prompt(template: str, variables: Dict[str, str]) -> str:
    """
    generates prompt based on prompt template and variables
    """
    validate_prompt_variables(template=template, variables=variables)
    return template.format(**variables)


def validate_prompt_variables(template: str, variables: Dict[str, str]) -> None:
    """
    validates if the variable is in the prompt
    """
    variables_in_prompt = re.findall(r'\{(.*?)\}', template)
    variables_in_prompt = [var for var in variables_in_prompt if isinstance(
        var, str) and re.match(r'^[A-Za-z0-9_]+$', var)]
    variable_names = set(variables.keys())
    intersect = variable_names.intersection(set(variables_in_prompt))

    if len(variable_names) != len(intersect):
        print(variable_names, intersect)
        remain = ",".join(variable_names-intersect)
        raise Exception(f"Prompt variable(s): {remain} is not within prompt")


client = OpenAIClient()
# groq_client = OpenAIClient(
#     base_url="https://api.groq.com/openai/v1",
#     api_key=os.environ["GROQ_API_KEY"]
# )


TOOLS_PLANNER_PROMPT_TEMPLATE = (
    "You are an excellent planner, you plan tools to use to fulfil a given task\n"
    "You are given the following tools and their necessary parameters, do your best to generate steps accordingly:\n\n"
    """
    {tool_prompts}
    """
    "Plan step by step, be as accurate as possible\n"
    "Steps:\n"
)

STEPS_PLANNER_PROMPT_TEMPLATE = (
    "System Prompt:\n"
    "{system_prompt}\n\n"
    "You are an excellent planner, you plan tools to use to fulfil a given task\n"
    "You are given the following steps and their necessary parameters, do your best to generate steps accordingly:\n\n"
    "Integrations:\n"
    "{integrations}"
    """
    {tool_prompts}
    """
    "Plan step by step, be as accurate as possible. If there are actions to be made in the\n"
    "same app, you can group them in the same step\n"
    "Steps:\n"
)

TOOLS_PROMPT = (
    "You are an excellent planner, you plan tools to use to fulfil a given task\n"
    "You are given the following tools and their necessary parameters, do your best to generate steps accordingly:\n\n"
    """
    Tool Name: AddEventToCalendar
    Description: Adds a single event to the calendar.
    Parameters:
        - name (description: Name of the event, type: string)
        - description (description: Description of the event, type: string)
        - start_datetime (description: Start datetime in ISO format, type: string)
        - end_datetime (description: End datetime in ISO format, type: string)
        - location (description: Location of the event, type: string)

    Tool Name: UpdateEventInCalendar
    Description: Updates an existing event in the calendar.
    Parameters:
        - event_id (description: Unique identifier of the event to update, type: string)
        - name (description: Updated name of the event, type: string)
        - description (description: Updated description of the event, type: string)
        - start_datetime (description: Updated start datetime in ISO format, type: string)
        - end_datetime (description: Updated end datetime in ISO format, type: string)
        - location (description: Updated location of the event, type: string)

    Tool Name: Calculator
    Description: Performs calculations based on mathematical expressions.
    Parameters:
        - expression (description: The mathematical expression to evaluate, type: string)

    Tool Name: Agent
    Description: Performs basic writing tasks like summarizing, reasoning, and writing.
    Parameters:
        - task (description: The specific writing task to perform, type: string)
        - content (description: The content to be processed or generated, type: string)

    Tool Name: EmailTool
    Description: Sends an email with the specified content and subject.
    Parameters:
        - recipient (description: The email address of the recipient, type: string)
        - subject (description: The subject of the email, type: string)
        - body (description: The body content of the email, type: string)
    """
    "Plan step by step, be as accurate as possible\n"
    "Steps:\n"
)


class ToolParameterType(StrEnum):
    """
    tool parameter type enum
    """
    STRING = "string"
    ARRAY = "array"
    INTEGER = "integer"
    FLOAT = "float"
    OBJECT = "object"
    BOOLEAN = "boolean"


class ToolParameter(BaseModel):
    """
    tool parameter base model
    """
    name: str
    description: str
    type: str

    def get_prompt(self) -> str:
        """returns string prompt of tool parameter"""
        return generate_prompt("- {name} (description: {description}, type: {type})", {
            "name": self.name,
            "description": self.description,
            "type": self.type
        })


class Tool(BaseModel):
    """
    base model for tool
    """
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)
    name: str
    description: str
    parameters: List[ToolParameter]
    function: Optional[Callable[..., Coroutine[Any, Any, str]]] = None

    def get_prompt(self) -> str:
        """returns string prompt of tool"""
        parameters = "\n".join([parameter.get_prompt()
                               for parameter in self.parameters])
        return generate_prompt(("Tool Name: {tool_name}\n"
                                "Description: {tool_description}\n"
                                "Parameters:\n"
                                "   {tool_parameters}"
                                ), {
                                    "tool_name": self.name,
                                    "tool_description": self.description,
                                    "tool_parameters": parameters
        })


class ToolStep(BaseModel):
    """
    base model tool step
    """
    name: str
    parameters: Dict[str, Any]


tool_parameters_json = {
    "tools": [
        Tool(
            name="AddEventToCalendar",
            description="Adds a single event to the calendar.",
            parameters=[
                ToolParameter(
                    name="name", description="Name of the event", type=ToolParameterType.STRING),
                ToolParameter(
                    name="description", description="Description of the event", type=ToolParameterType.STRING),
                ToolParameter(
                    name="start_datetime", description="Start datetime in ISO format", type=ToolParameterType.STRING),
                ToolParameter(
                    name="end_datetime", description="End datetime in ISO format", type=ToolParameterType.STRING),
                ToolParameter(
                    name="location", description="Location of the event", type=ToolParameterType.STRING)
            ]
        ),
        Tool(
            name="UpdateEventInCalendar",
            description="Updates an existing event in the calendar.",
            parameters=[
                ToolParameter(
                    name="event_id", description="Unique identifier of the event to update", type=ToolParameterType.STRING),
                ToolParameter(
                    name="name", description="Updated name of the event", type=ToolParameterType.STRING),
                ToolParameter(
                    name="description", description="Updated description of the event", type=ToolParameterType.STRING),
                ToolParameter(
                    name="start_datetime", description="Updated start datetime in ISO format", type=ToolParameterType.STRING),
                ToolParameter(
                    name="end_datetime", description="Updated end datetime in ISO format", type=ToolParameterType.STRING),
                ToolParameter(
                    name="location", description="Updated location of the event", type=ToolParameterType.STRING)
            ]
        ),
        Tool(
            name="Calculator",
            description="Performs calculations based on mathematical expressions.",
            parameters=[
                ToolParameter(
                    name="expression", description="The mathematical expression to evaluate", type=ToolParameterType.STRING)
            ]
        ),
        Tool(
            name="Agent",
            description="Performs basic writing tasks like summarizing, reasoning, and writing.",
            parameters=[
                ToolParameter(
                    name="task", description="The specific writing task to perform", type=ToolParameterType.STRING),
                ToolParameter(
                    name="content", description="The content to be processed or generated", type=ToolParameterType.STRING)
            ]
        ),
        Tool(
            name="EmailTool",
            description="Sends an email with the specified content and subject.",
            parameters=[
                ToolParameter(
                    name="recipient", description="list of email recipients", type=ToolParameterType.ARRAY),
                ToolParameter(
                    name="subject", description="The subject of the email", type=ToolParameterType.STRING),
                ToolParameter(
                    name="body", description="The body content of the email", type=ToolParameterType.STRING)
            ]
        )
    ]
}


class ToolUsePlanner:
    """
    class to plan tool use steps
    """

    def __init__(self,
                 client: OpenAIClient,
                 model: str,
                 messages: List[ChatCompletionMessageParam],
                 tools: List[Tool],
                 prompt_template: str = TOOLS_PLANNER_PROMPT_TEMPLATE) -> None:
        self.tools = tools
        self.model = model
        self.client = client
        self.messages = messages
        self.system_prompt = generate_prompt(prompt_template, {
                                             "tool_prompts": "\n\n".join([tool.get_prompt() for tool in self.tools])})
        self._tool_map = self._generate_tool_map()
        self._functions_map: Dict[str, Coroutine[Any,
                                                 Any, Any]] = self._generate_function_map()

    async def run(self) -> List[ToolStep]:
        """
        runs tool planner
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                *self.messages

            ],
            parallel_tool_calls=False,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "list_tool_use_steps",
                        "description": "Creates a list of tool use steps, with the required parameters",
                        "parameters": {
                            "type": "object",
                            "required": [
                                "steps"
                            ],
                            "properties": {
                                "steps": {
                                    "type": "array",
                                    "description": "An array of tool use parameters",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {
                                                "type": "string",
                                                "description": "Name of the tool"
                                            },
                                            "parameters": {
                                                "type": "string",
                                                "description": "stringified JSON of tool parameters"
                                            },

                                        },
                                        "additionalProperties": False,
                                        "required": [
                                            "name",
                                            "parameters"
                                        ]
                                    }
                                }
                            },
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
            ],
            tool_choice={
                "type": "function",
                "function":
                {"name": "list_tool_use_steps"}
            }
        )
        # validates llm generated steps
        self._validate_llm_response(response=response)
        steps = [ToolStep(name=tool_step["name"], parameters=json.loads(tool_step["parameters"])) for tool_step in json.loads(
            response.choices[0].message.tool_calls[0].function.arguments)["steps"]]

        self._validate_step_args(steps=steps)
        return steps

    def _validate_step_args(self, steps: List[ToolStep]) -> List[ToolStep]:
        """
        validates steps, returns list of malformed tool parameters
        """
        malformed_steps: List[ToolStep] = []
        for step in steps:
            step.name = step.name.replace("functions.", "")
            parameter_type_map = self._tool_map[step.name]
            for tool_parameter_name, tool_parameter_type in parameter_type_map.items():
                if not isinstance(step.parameters[tool_parameter_name], tool_parameter_type):
                    malformed_steps.append(malformed_steps)
        if len(malformed_steps) > 0:
            raise ValueError("Steps had malformed tool parameters")

    def _validate_llm_response(self, response):
        if len(response.choices) == 0:
            raise ValueError("No message was generated")
        if len(response.choices[0].message.tool_calls) == 0:
            raise ValueError("No steps were generated")
        response = json.loads(
            response.choices[0].message.tool_calls[0].function.arguments)
        if "steps" not in response:
            raise ValueError("There was no 'steps' generated")
        if not isinstance(response, dict):
            raise ValueError("LLM produced JSON is malformed")

    def _generate_tool_map(self) -> Dict[str, Any]:
        tool_map: Dict[str, Dict[str, type]] = {}
        type_map: Dict[ToolParameterType, type] = {
            ToolParameterType.STRING: str,
            ToolParameterType.INTEGER: int,
            ToolParameterType.FLOAT: float,
            ToolParameterType.BOOLEAN: bool,
            ToolParameterType.ARRAY: list,
            ToolParameterType.OBJECT: dict
        }
        for tool in self.tools:
            tool_map[tool.name] = {
                parameter.name: type_map[parameter.type]for parameter in tool.parameters}
        return tool_map

    def add_message(self, message: ChatCompletionMessageParam) -> None:
        """
        adds message
        """
        self.messages.append(message)

    def set_messages(self, messages: List[ChatCompletionMessageParam]) -> None:
        """
        sets messages
        """
        self.messages = messages

    def clear_messages(self) -> None:
        self.messages = []

    def _generate_function_map(self) -> Dict[str, Coroutine[Any, Any, str]]:
        return {tool.name: tool.function for tool in self.tools if tool.function}

    def get_function_map(self) -> Dict[str, Coroutine[Any, Any, str]]:
        """
        gets the function map of the tool
        """
        return self._functions_map

# generated_tool_steps = client.chat.completions.create(
#     model="gpt-4o-mini",
#     temperature=0,
#     messages=[
#         {
#             "role": "system",
#             "content": TOOLS_PROMPT
#         },
#         # {"role": "user",
#         #     "content": [{"type": "text", "text": "add all events to my calendar"},
#         #                 {
#         #         "type": "image_url",
#         #         "image_url": {
#         #             "url": "https://i.ibb.co/LSdX0RF/photo-6316392868339629238-y-1.jpg",
#         #         },
#         #     },]}
#         {
#             "role": "user",
#             "content": (
#                 "Hey there! Could you help me out by adding a few events to my calendar? "
#                 "First up, I've got a team meeting where we'll chat about project milestones and deliverables. "
#                 "It's happening on June 23, 2024, from 9 AM in Conference Room A. "
#                 "Then, there's a client presentation scheduled for the same day from 11 AM to noon, "
#                 "and we'll be doing that over Zoom. Finally, I've got a lunch date with a partner to discuss "
#                 "some collaboration opportunities at Downtown Bistro, from 1 PM to 2 PM. Thanks a bunch!"
#                 "I want you to summarise all of it then write an email about it"
#             )
#         }
#     ],
#     tools=[
#         {
#             "type": "function",
#             "function": {
#                 "name": "list_tool_use_steps",
#                 "description": "Creates a list of tool use steps, with the required parameters",
#                 "parameters": {
#                     "type": "object",
#                     "required": [
#                         "steps"
#                     ],
#                     "properties": {
#                         "steps": {
#                             "type": "array",
#                             "description": "An array of tool use parameters",
#                             "items": {
#                                 "type": "object",
#                                 "properties": {
#                                     "name": {
#                                         "type": "string",
#                                         "description": "Name of the tool"
#                                     },
#                                     "parameters": {
#                                         "type": "string",
#                                         "description": "stringified JSON of tool parameters"
#                                     },

#                                 },
#                                 "additionalProperties": False,
#                                 "required": [
#                                     "name",
#                                     "parameters"
#                                 ]
#                             }
#                         }
#                     },
#                     "additionalProperties": False
#                 },
#                 "strict": True
#             }
#         }
#     ],
#     tool_choice={
#         "type": "function",
#                 "function":
#                 {"name": "list_tool_use_steps"}
#     }
# ).choices[0].message.tool_calls[0].function.arguments

# pprint(generated_tool_steps)

# tool_steps = [{"tool_name": step["name"], "parameters": json.loads(
#     step["parameters"])} for step in json.loads(generated_tool_steps)["steps"]]
# pprint(tool_steps)


class PlannerIntegration(BaseModel):
    """
    base model planner integration
    """
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)
    name: str
    description: str
    planner: ToolUsePlanner

    def get_prompt(self) -> str:
        """
        returns prompt of planner integration
        """
        actions: str = generate_prompt("Actions: {actions}\n\n", {"actions": ", ".join([
                                       tool.name for tool in self.planner.tools])})
        return generate_prompt(("Integration Name: {integration_name}\n"
                                "Description: {integration_description}\n"
                                "{integration_actions}\n\n"), {
            "integration_name": self.name,
            "integration_description": self.description,
            "integration_actions": actions
        })


class PlannerAgent:
    """
    base class for planner agent
    """

    def __init__(self,
                 client: OpenAIClient,
                 model: str,
                 system_prompt: str,
                 messages: List[ChatCompletionMessageParam],
                 integrations: List[PlannerIntegration],
                 prompt_template: str = STEPS_PLANNER_PROMPT_TEMPLATE,
                 formatted: bool = False
                 ) -> None:
        self.formatted = formatted
        self.messages = messages
        self.system_prompt = system_prompt
        self.integrations = integrations
        self.prompt_template = prompt_template
        self.client = client
        self.model = model
        self.planner = ToolUsePlanner(
            client=self.client,
            model=self.model,
            prompt_template=self._generate_system_prompt(),
            messages=[
                *self.messages
            ],
            tools=[
                Tool(
                    name="AssistantTool",
                    description="Used for when you need to summarise, write or reflect on things. Use it in between tool use",
                    parameters=[
                        ToolParameter(
                            name="prompt", description="Prompt to give the assistant", type=ToolParameterType.STRING)
                    ]
                ),
                Tool(
                    name="IntegrationSelectorTool",
                    description="Chooses the right integration to accomplish the task",
                    parameters=[
                        ToolParameter(
                            name="integration_name", description="Name of Integration of choice", type=ToolParameterType.STRING),
                        ToolParameter(
                            name="prompt", description="Prompt to give the integration, it's a command of what you need to get done", type=ToolParameterType.STRING)
                    ]
                ),

            ]
        )
        self._integration_map: Dict[str,
                                    ToolUsePlanner] = self._generate_integration_map()

    async def run(self) -> ChatCompletionMessageParam:
        """
        run planner agent
        """
        steps = await self._generate_plan()
        current_prompt = None
        for _, step in enumerate(steps):
            # get integration prompt
            prompt = step.parameters["prompt"]
            if current_prompt is None:
                current_prompt = prompt
            response_prompt = ""
            if step.name == "IntegrationSelectorTool":
                response_prompt = await self._run_integration(
                    step.parameters["integration_name"], prompt=prompt)
            elif step.name == "AssistantTool":
                completion = await self._run_assistant_tool(prompt=prompt)
                response_prompt = completion.content
            # if not response_prompt:
            #     raise ValueError("No response was given")
            # if i == 0:
            #     current_prompt = self._generate_recursive_prompt(
            #         response_prompt, "")
            # elif i > 0 and current_prompt:
            #     current_prompt = self._generate_recursive_prompt(
            #         response_prompt, current_prompt)
            self.add_message(
                {
                    "role": "assistant",
                    "content": (
                        f"{step.name} has done the following:\n"
                        f"{response_prompt}"
                    )
                }
            )

        return await self._get_response()

    async def _get_response(self) -> ChatCompletionMessageParam:
        if self.formatted:
            raise NotImplementedError("Formatted output has not been created")
        else:
            return await self._run_assistant_tool(
                prompt="Summarise everything that has been accomplished")

    async def _run_integration(self, integration_name: str, prompt: str) -> str:
        if integration_name not in self._integration_map:
            raise ValueError("This integration does not exist")
        integration = self._integration_map[integration_name]
        integration.set_messages(messages=self.messages)
        integration.add_message(
            {
                "role": "user",
                "content": prompt
            }
        )
        steps: List[ToolStep] = await integration.run()
        integration.clear_messages()
        functions = integration.get_function_map()
        tasks = [functions[step.name](**step.parameters) for step in steps]
        responses: List[str] = await asyncio.gather(*tasks)
        response_prompt = generate_prompt(
            template=(
                f"{integration_name} has done the following:\n"
                "{responses}"
            ),
            variables={
                "responses": "\n\n".join(responses)
            }
        )
        return response_prompt

    async def _generate_plan(self) -> List[ToolStep]:
        return await self.planner.run()

    def _generate_integration_map(self) -> Dict[str, ToolUsePlanner]:
        return {integration.name: integration.planner for integration in self.integrations}

    def _generate_system_prompt(self) -> str:
        """
        generates planner system prompt
        """
        integration_prompt = "\n\n".join(
            [integration.get_prompt() for integration in self.integrations])

        return generate_prompt(
            self.prompt_template,
            {
                "integrations": integration_prompt,
                "system_prompt": self.system_prompt,
                "tool_prompts": "{tool_prompts}"
            }
        )

    def _generate_recursive_prompt(self, input_prompt: str, existing_prompt: str) -> str:
        return generate_prompt(
            (
                "{existing_prompt}\n"
                "{input_prompt}\n"
            ),
            {
                "input_prompt": "",
                "existing_prompt": generate_prompt(
                    (
                        "{existing_prompt}\n"
                        "{input_prompt}\n"
                    ),
                    {
                        "existing_prompt": existing_prompt,
                        "input_prompt": input_prompt
                    })})

    def add_message(self, message: ChatCompletionMessageParam) -> None:
        """
        add message to message history
        """
        self.messages.append(message)

    async def _run_assistant_tool(self, prompt: str) -> ChatCompletionMessageParam:
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                *self.messages,
                {
                    "role": "user",
                    "content": prompt
                }
            ])
        if len(response.choices) == 0:
            raise ValueError("Assistant did not produce anything")
        return response.choices[0].message


async def add_event_to_calendar(name: str, description: str, start_datetime: str, end_datetime: str, location: str) -> str:
    """
    Adds a single event to the calendar and returns the event name.
    """
    logging.info("Function: add_event_to_calendar - Adding event to calendar")
    # Logic to add event to calendar would go here
    return f"Added {name}"


async def update_event_in_calendar(event_id: str, name: str, description: str, start_datetime: str, end_datetime: str, location: str) -> str:
    """
    Updates an existing event in the calendar and returns the updated event name.
    """
    logging.info(
        "Function: update_event_in_calendar - Updating event in calendar")
    # Logic to update event in calendar would go here
    return f"Updated {name}"


async def delete_event_from_calendar(event_id: str) -> str:
    """
    Deletes an event from the calendar and returns the event ID.
    """
    logging.info(
        "Function: delete_event_from_calendar - Deleting event from calendar")
    # Logic to delete event from calendar would go here
    return f"Deleted event with ID {event_id}"


async def search_event_in_calendar(query: str, start_datetime: str, end_datetime: str) -> str:
    """
    Searches for events in the calendar and returns the search query.
    """
    logging.info(
        "Function: search_event_in_calendar - Searching for events in calendar")
    # Logic to search events in calendar would go here
    return f"Searched for events with query '{query}'"


async def send_email(recipient: str, subject: str, body: str) -> str:
    """
    Sends an email with the specified content and subject and returns a confirmation message.
    """
    logging.info("Function: send_email - Sending email")
    # Logic to send email would go here
    return f"Email sent to {recipient} with subject '{subject}'"


async def draft_email(subject: str, body: str) -> str:
    """
    Drafts an email with the specified content and subject and returns a draft confirmation.
    """
    logging.info("Function: draft_email - Drafting email")
    # Logic to draft email would go here
    return f"Drafted email with subject '{subject}'"


async def search_email(query: str, date_range: str) -> str:
    """
    Searches through emails based on a query and date range and returns the search query.
    """
    logging.info("Function: search_email - Searching emails")
    # Logic to search emails would go here
    return f"Searched emails with query '{query}' in date range '{date_range}'"


if __name__ == "__main__":

    calendar_planner = ToolUsePlanner(
        client=client,
        model="gpt-4o-mini",
        messages=[],
        tools=[
            Tool(
                name="AddEventToCalendar",
                description="Adds a single event to the calendar.",
                parameters=[
                    ToolParameter(
                        name="name", description="Name of the event", type=ToolParameterType.STRING),
                    ToolParameter(
                        name="description", description="Description of the event", type=ToolParameterType.STRING),
                    ToolParameter(
                        name="start_datetime", description="Start datetime in ISO format", type=ToolParameterType.STRING),
                    ToolParameter(
                        name="end_datetime", description="End datetime in ISO format", type=ToolParameterType.STRING),
                    ToolParameter(
                        name="location", description="Location of the event", type=ToolParameterType.STRING)
                ],
                function=add_event_to_calendar
            ),
            Tool(
                name="UpdateEventInCalendar",
                description="Updates an existing event in the calendar.",
                parameters=[
                    ToolParameter(
                        name="event_id", description="Unique identifier of the event to update", type=ToolParameterType.STRING),
                    ToolParameter(
                        name="name", description="Updated name of the event", type=ToolParameterType.STRING),
                    ToolParameter(
                        name="description", description="Updated description of the event", type=ToolParameterType.STRING),
                    ToolParameter(
                        name="start_datetime", description="Updated start datetime in ISO format", type=ToolParameterType.STRING),
                    ToolParameter(
                        name="end_datetime", description="Updated end datetime in ISO format", type=ToolParameterType.STRING),
                    ToolParameter(
                        name="location", description="Updated location of the event", type=ToolParameterType.STRING)
                ],
                function=update_event_in_calendar
            ),
            Tool(
                name="DeleteEventFromCalendar",
                description="Deletes an event from the calendar.",
                parameters=[
                    ToolParameter(
                        name="event_id", description="Unique identifier of the event to delete", type=ToolParameterType.STRING)
                ],
                function=delete_event_from_calendar
            ),
            Tool(
                name="SearchEventInCalendar",
                description="Searches for events in the calendar.",
                parameters=[
                    ToolParameter(
                        name="query", description="Search query to find events", type=ToolParameterType.STRING),
                    ToolParameter(
                        name="start_datetime", description="Start datetime to filter events", type=ToolParameterType.STRING),
                    ToolParameter(
                        name="end_datetime", description="End datetime to filter events", type=ToolParameterType.STRING)
                ],
                function=search_event_in_calendar
            ),

        ]
    )
    email_planner = ToolUsePlanner(
        client=client,
        model="gpt-4o-mini",
        messages=[],
        tools=[
            Tool(
                name="EmailTool",
                description="Sends an email with the specified content and subject.",
                parameters=[
                    ToolParameter(
                        name="recipient", description="The email address of the recipient", type=ToolParameterType.STRING),
                    ToolParameter(
                        name="subject", description="The subject of the email", type=ToolParameterType.STRING),
                    ToolParameter(
                        name="body", description="The body content of the email", type=ToolParameterType.STRING)
                ],
                function=send_email
            ),
            Tool(
                name="DraftEmail",
                description="Drafts an email with the specified content and subject.",
                parameters=[
                    ToolParameter(
                        name="subject", description="The subject of the email", type=ToolParameterType.STRING),
                    ToolParameter(
                        name="body", description="The body content of the email", type=ToolParameterType.STRING)
                ],
                function=draft_email
            ),
            Tool(
                name="SearchEmail",
                description="Searches through emails based on a query.",
                parameters=[
                    ToolParameter(
                        name="query", description="Search query to find emails", type=ToolParameterType.STRING),
                    ToolParameter(
                        name="date_range", description="Date range to filter emails", type=ToolParameterType.STRING)
                ],
                function=search_email
            )
        ]
    )
    planner_agent = PlannerAgent(
        client=client,
        model="gpt-4o-mini",
        system_prompt="You are a helpful assistant",
        integrations=[
            PlannerIntegration(
                name="CalendarIntegration",
                description="Handling Calendar Events",
                planner=calendar_planner),
            PlannerIntegration(
                name="EmailIntegration",
                description="Handling Emails",
                planner=email_planner)],
        messages=[
            {
                "role": "user",
                "content": (
                    "Hi! I need to add three new calendar events. First, a project kickoff meeting on March 15, 2024, "
                    "at 10:00 AM. Second, a team brainstorming session on March 20, 2024, at 2:00 PM. "
                    "Lastly, a client review meeting on March 25, 2024, at 1:00 PM. Could you also draft a summary "
                    "of these events and add it into my email?"
                )
            }
        ]
    )
    response = asyncio.run(
        planner_agent.run()
    )
    print(response)
    # step_planner = ToolUsePlanner(
    #     client=client,
    #     model="gpt-4o-mini",
    #     prompt_template=STEPS_PLANNER_PROMPT_TEMPLATE,
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": (
    #                 "Hi! I need to add three new calendar events. First, a project kickoff meeting on March 15, 2024, "
    #                 "at 10:00 AM. Second, a team brainstorming session on March 20, 2024, at 2:00 PM. "
    #                 "Lastly, a client review meeting on March 25, 2024, at 1:00 PM. Could you also draft a summary "
    #                 "of these events into my email?"
    #             )
    #         }
    #     ],
    #     tools=[
    #         Tool(
    #             name="AssistantTool",
    #             description="Used for when you need to summarise, write or reflect on things. Use it in between tool use",
    #             parameters=[
    #                 ToolParameter(
    #                     name="prompt", description="Prompt to give the assistant", type=ToolParameterType.STRING)
    #             ]
    #         ),
    #         Tool(
    #             name="IntegrationSelectorTool",
    #             description="Chooses the right integration to accomplish the task",
    #             parameters=[
    #                 ToolParameter(
    #                     name="IntegrationName", description="Name of Integration of choice", type=ToolParameterType.STRING),
    #                 ToolParameter(
    #                     name="prompt", description="Prompt to give the integration, it's a command of what you need to get done", type=ToolParameterType.STRING)
    #             ]
    #         ),

    #     ]
    # )
    # import time
    # start_time = time.time()
    # steps = asyncio.run(step_planner.run())

    # end_time = time.time()
    # print(f"Time taken to run: {end_time - start_time} seconds")
    # pprint(steps)
    # pprint([tool.model_dump(mode="json") for tool in planner.tools])
    # def generate_recursive_prompt(input_prompt: str, existing_prompt: str) -> str:
    #     return generate_prompt(
    #         (
    #             "{existing_prompt}\n\n"
    #             "{input_prompt}\n"
    #         ),
    #         {
    #             "input_prompt": "{input_prompt}",
    #             "existing_prompt": generate_prompt(
    #                 (
    #                     "{existing_prompt}\n\n"
    #                     "{input_prompt}\n"
    #                 ),
    #                 {
    #                     "existing_prompt": existing_prompt,
    #                     "input_prompt": input_prompt
    #                 })})
    # print(generate_recursive_prompt("input", "existing"))
    pass
