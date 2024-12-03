"""
base models for tool related things
"""

from enum import StrEnum
from typing import Optional, Callable, Coroutine, Any, List, Dict
from prompts.generator import generate_prompt
from pydantic import BaseModel, ConfigDict


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
