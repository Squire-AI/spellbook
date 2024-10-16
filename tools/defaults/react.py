from typing import Any, Dict
from ..models import Tool

REACT_PLANNING_TOOLS: Dict[str, Tool] = {
    "THOUGHT": Tool(
        name="Thought",
        description="Use Thought to describe your thoughts about the question you have been asked",
        parameters={
            "type": "object",
            "properties": {
                    "Thought": {
                        "type": "string",
                        "description": "Thought of the question you have been asked"
                    },
            },
            "additionalProperties": False,
            "required": [
                "Thought"
            ]
        },
    ),
    "ACTION": Tool(
        name="Action",
        description="Use Action to run one of the actions available to you",
        parameters={
            "type": "object",
            "properties": {
                    "ActionName": {
                        "type": "string",
                        "description": "Name of the action you want to use"
                    },
                "ActionDescription": {
                        "type": "string",
                        "description": "Detailed description of how you want your action to run, for example for action = calculator: 'Use Calculator to find 3**2'"
                        }
            },
            "additionalProperties": False,
            "required": [
                "ActionName",
                "ActionDescription"
            ]
        },
    ),
    "COMPLETE": Tool(
        name="Complete",
        description="Use Complete when the task is completed",
        parameters={
            "type": "object",
            "properties": {
                    "Complete": {
                        "type": "boolean",
                        "description": ""
                    },
            },
            "additionalProperties": False,
            "required": [
                "Complete"
            ]
        },
    ),

}
