from typing import List
from ..models import Tool

REACT_PLANNING_TOOLS: List[Tool] = [
    Tool(
        name="StepChoice",
        description="You think step by step, this is a single step and you can choose between, THOUGHT, ACTION, PAUSE, ANSWER",
        parameters={
            "type": "object",
            "properties": {
                    "choice": {
                        "type": "string",
                        "enum": [
                            "THOUGHT",
                            "PAUSE",
                            "ACTION",
                            "ANSWER"
                        ],
                        "description": "The most appropriate step to take in the current step-by-step reasoning, if the task has been satisfied use COMPLETE"
                    },
                "prompt": {
                        "type": "string",
                        "description": "The most appropriate & descriptive prompt that describes what is needed in the current stage, for action, give as much detail as possible"
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
                "description": "The formatted response string"
            },
            "tools_used": {
                "type": "array",
                "description": "An array of tools used in the process",
                "items": {
                    "type": "string",
                    "description": "Name of the tool used"
                }
            },
            "sources": {
                "type": "array",
                "description": "Array of sources related to the information provided",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Name of the source"
                        },
                        "url": {
                            "type": "string",
                            "description": "URL of the source"
                        },
                        "type": {
                            "type": "string",
                            "enum": [
                                "document",
                                "website"
                            ],
                            "description": "Type of the source, either document or website"
                        }
                    },
                    "required": [
                        "source",
                        "url",
                        "type"
                    ]
                }
            }
        },
        "additionalProperties": False,
        "required": [
            "response", "sources", "tools_used"
        ]
    }
)
