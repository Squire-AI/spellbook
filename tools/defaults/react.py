from typing import List
from ..models import Tool

REACT_PLANNING_TOOLS: List[Tool] = [
    Tool(
        name="StepChoice",
        description="You think step by step, this is a single step and you can choose between, THOUGHT, OBSERVE, ACTION, COMPLETE",
        parameters={
            "type": "object",
            "properties": {
                    "choice": {
                        "type": "string",
                        "enum": [
                            "THOUGHT",
                            "OBSERVE",
                            "ACTION",
                            "COMPLETE"
                        ],
                        "description": "The most appropriate step to take in the current step-by-step reasoning, if the task has been satisfied use COMPLETE"
                    },
                "prompt": {
                        "type": "string",
                        "description": "The most appropriate & descriptive prompt that describes what is needed in the current stage"
                        }
            },
            "additionalProperties": False,
            "required": [
                "choice", "prompt"
            ]
        },
    )

]

REACT_FORMATTED_OUTPUT_TOOL: Tool = Tool(
    name="ReactFormattedOutput",
    description="You take the message history & return a formatted response",
    parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Text response to the user's query"
            },
            "sources": {
                "type": "array",
                "description": "List of sources used for responses, if no sources used, leave as empty",
                "items": {
                    "type": "string",
                    "description": "name of the source used"
                }
            },
            "tools_used": {
                "type": "array",
                "description": "List of tools used for responses, if no tools used, leave as empty",
                "items": {
                    "type": "string",
                    "description": "name of the tool"
                }
            }
        },
        "additionalProperties": False,
        "required": [
            "content", "sources", "tools_used"
        ]
    },
)
