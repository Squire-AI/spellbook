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
import asyncio
from enum import StrEnum
import json
import os
from pydantic import BaseModel, ConfigDict
from typing import Any, Coroutine, Dict, List, Optional
from pprint import pprint
from openai import AsyncClient as OpenAIClient
import re
from openai.types.chat import ChatCompletionMessageParam


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
        remain = ",".join(variable_names-intersect)
        raise Exception(f"Prompt variable(s): {remain} is not within prompt")


client = OpenAIClient()
groq_client = OpenAIClient(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"]
)


TOOLS_PLANNER_PROMPT_TEMPLATE = (
    "You are an excellent planner, you plan tools to use to fulfil a given task\n"
    "You are given the following tools and their necessary parameters, do your best to generate steps accordingly:\n\n"
    """
    {tool_prompts}
    """
    "Plan step by step, be as accurate as possible\n"
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
    function: Optional[Coroutine[Any, Any, Any]] = None

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


if __name__ == "__main__":
    planner = ToolUsePlanner(
        client=groq_client,
        model="llama3-groq-8b-8192-tool-use-preview",
        messages=[
            {
                "role": "user",
                "content": (
                    "Hey there! Could you help me out by adding a few events to my calendar? "
                    "First up, I've got a team meeting where we'll chat about project milestones and deliverables. "
                    "It's happening on June 23, 2024, from 9 AM in Conference Room A. "
                    "Then, there's a client presentation scheduled for the same day from 11 AM to noon, "
                    "and we'll be doing that over Zoom. Finally, I've got a lunch date with a partner to discuss "
                    "some collaboration opportunities at Downtown Bistro, from 1 PM to 2 PM. Thanks a bunch!"
                    "I want you to summarise all of it then write an email about it"
                )
            }
        ],
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
    )
    steps = asyncio.run(planner.run())
    pprint(steps)
